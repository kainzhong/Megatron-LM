"""
A super janky implementation of the mhc in megatron. 
This is just to get a quick prototype working
"""

import functools
from typing import Any, Optional
import logging
logger = logging.getLogger(__name__)

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.typed_torch import apply_module, copy_signature

import transformer_engine.pytorch as te

from megatron.core.utils import (
    deprecate_inference_params,
    make_viewless_tensor,
    nvtx_range_pop,
    nvtx_range_push,
)

from megatron.core.inference.contexts import BaseInferenceContext


@torch.compile
def mHCPreNative(x, H_pre, n):
    """
    Reference operator for applying mHC's pre matrix H to a vector x.

    x: (B, T, nC)
    H_pre: (B, T, n)
    """
    H_pre = H_pre.contiguous()

    B, T, nC = x.shape
    C = nC // n
    x = x.view(B, T, n, C)  # (B, T, n, C)
    H_pre = H_pre.view(B, T, 1, n)  # (B, T, 1, n)

    out = (H_pre @ x).view(B, T, C) # (B, T, C)

    return out

@torch.compile
def mHCPostResNative(f, H_post, x, H_res, n):
    """
    Reference operator for applying mHC's post transformation and residual transformation

    f: (B, T, C)
    H_post: (B, T, n)
    x: (B, T, nC)
    H_res: (B, T, n, n)
    """

    B, T, nC = x.shape
    C = nC // n

    f = f.view(B, T, 1, C)
    H_post = H_post.view(B, T, n, 1)
    x = x.view(B, T, n, C)
    H_res = H_res.view(B, T, n, n)

    out = H_post @ f + H_res @ x # (B, T, n, C)
    out = out.view(B, T, nC)

    return out

@torch.compile
def mhcNativeCombinedOperators(x, phi, alpha, beta, n, iterations):
    eps = 1e-8

    B, T, nC = x.shape
    H = x @ phi  # (B, T, 2n + n^2)
    norm = torch.sum(x * x, dim=2)
    r = (norm / (nC ** 0.5)) + eps  # (B, T)

    H_pre = H[:, :, :n]  # (B, T, n)
    H_post = H[:, :, n:2*n]  # (B, T, n)
    H_res = H[:, :, 2*n:]  # (B, T, n^2)

    beta_pre = beta[0, :n]
    beta_post = beta[0, n:2*n]
    beta_res = beta[0, 2*n:2*n +  n*n]

    alpha_pre, alpha_post, alpha_res = alpha[0], alpha[1], alpha[2]

    H_pre = H_pre * alpha_pre
    H_post = H_post * alpha_post
    H_res = H_res * alpha_res

    H_pre = H_pre / r[:, :, None]
    H_post = H_post / r[:, :, None]
    H_res = H_res / r[:, :, None]

    H_pre = H_pre + beta_pre
    H_post = H_post + beta_post
    H_res = H_res + beta_res

    H_pre = F.sigmoid(H_pre)
    H_post = 2 * F.sigmoid(H_post)

    H_res = H_res.view(B, T, n, n)  # Reshape to (B, T, n, n)

    log_mu = torch.zeros(B, T, n, device=x.device, dtype=torch.float32)
    log_nu = torch.zeros(B, T, n, device=x.device, dtype=torch.float32)
    
    f = torch.zeros(B, T, n, device=x.device, dtype=torch.float32)
    g = torch.zeros(B, T, n, device=x.device, dtype=torch.float32)
    
    for _ in range(iterations):
        # Update f: logsumexp over the column dimension (3)
        f = log_mu - torch.logsumexp(H_res + g.unsqueeze(2), dim=3)
        # Update g: logsumexp over the row dimension (2)
        g = log_nu - torch.logsumexp(H_res + f.unsqueeze(3), dim=2)
        
    log_P = f.unsqueeze(3) + H_res + g.unsqueeze(2)
    H_res = torch.exp(log_P).view(B, T, n*n)  # Reshape back to (B, T, n^2)

    out = torch.cat([H_pre, H_post, H_res], dim=-1) # (B, T, 2n + n^2)

    H_pre = out[:, :, :n]  # (B, T, n)
    H_post = out[:, :, n:2*n]  # (B, T, n)
    H_res = out[:, :, 2*n:2*n + n*n]  # (B, T, n^2)

    return H_pre.to(x.dtype), H_post.to(x.dtype), H_res.to(x.dtype)

def mhcFusedCombinedOperators(x, phi, alpha, beta, n, iterations):
    """
    Triton implementation of the mHC combined operators for eq. 14-19 in the DeepSeek mHC paper.

    :param x: (B, T, nC) input features, where n is the number of Hyper-Connection streams and C is the hidden dimension of the input features.
    Note that x should be bfloat16
    :param phi: (2n + n^2, nC) projection matrix, which is the transposed tensor for the column-major layout.
    :param alpha: (3,) learnable scaling parameters for the pre, post and res components.
    :param beta: (1, 2n + n^2,) learnable bias parameters for the pre, post and res components.
    :param n: number of Hyper-Connection streams, which must be 4 for this implementation.
    :param iterations: number of Sinkhorn iterations to perform for the res component.

    :return: H_pre: (B, T, n), H_post: (B, T, n), H_res: (B, T, n^2), which are the transformation matrices
    """
    # phi_torch = torch.randn((nC, N), device='cuda', dtype=torch.float32, requires_grad=True) # For pytorch we just use row-major layout
    # phi_Triton_T = torch.randn((N, nC), device='cuda', dtype=torch.float32, requires_grad=True) # Triton implementation expects column-major layout
    H, r = te.mhc.mHCProjectionOp.apply(x, phi)
    out = te.mhc.mHCElementwiseOp.apply(H, alpha, beta, r, n)
    out = te.mhc.mHCSinkhornOp.apply(out, n, iterations)

    H_pre = out[:, :, :n]  # (B, T, n)
    H_post = out[:, :, n:2*n]  # (B, T, n)
    H_res = out[:, :, 2*n:2*n + n*n]  # (B, T, n^2)

    return H_pre, H_post, H_res


def mhc_bd(x_with_bias, residual_dtype, prob):
    x, bias = x_with_bias  # unpack

    # For fp32 residual connections: upcast x (and bias) to residual's dtype so that
    # the addition and output remain in fp32, preserving numerical precision in the
    # residual stream across layers. When fp32_residual_connection is enabled,
    # pipeline parallel communication dtype should be set to fp32 accordingly.
    if x.dtype != residual_dtype:
        x = x.to(residual_dtype)
        if bias is not None:
            bias = bias.to(residual_dtype)

    # The Dropout operation, Residual Addition and the tensor returning can be
    # done generically outside the if statement, but that stops fusing of Bias
    # Addition-Dropout-Residual Addition operation. So doing it together inside
    # the conditional branch to improve performance
    if bias is not None:
        x = x + bias
        out = torch.nn.functional.dropout(x, p=prob, training=True, inplace=False)
        return out
    else:
        out = torch.nn.functional.dropout(x, p=prob, training=True, inplace=False)
        return out

class MHCTransformerLayer(TransformerLayer):
    _tb_step: int = 0

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        self.mhc_streams = 4
        
        super().__init__(config, *args, **kwargs)

        dtype = config.params_dtype
        n = self.mhc_streams
        # These are all hardcoded for now. I will clean these up after I figure out how mcore is supposed to be used
        self.mhc_alpha_attn = torch.nn.Parameter(torch.ones(3, dtype=dtype, device="cuda"))
        self.mhc_beta_attn = torch.nn.Parameter(torch.zeros(1, 2*n + n*n, dtype=dtype, device="cuda"))
        self.mhc_phi_attn = torch.nn.Parameter(torch.zeros(24, n * config.hidden_size, dtype=dtype, device="cuda")) # Column-major layout for Triton
        torch.nn.init.xavier_normal_(self.mhc_phi_attn, gain=0.02)

        setattr(self.mhc_alpha_attn, 'sequence_parallel', True)
        setattr(self.mhc_beta_attn, 'sequence_parallel', True)
        setattr(self.mhc_phi_attn, 'sequence_parallel', True)

        self.mhc_alpha_mlp = torch.nn.Parameter(torch.ones(3, dtype=dtype, device="cuda"))
        self.mhc_beta_mlp = torch.nn.Parameter(torch.zeros(1, 2*n + n*n, dtype=dtype, device="cuda"))
        self.mhc_phi_mlp = torch.nn.Parameter(torch.zeros(24, n * config.hidden_size, dtype=dtype, device="cuda")) # Column-major layout for Triton
        torch.nn.init.xavier_normal_(self.mhc_phi_mlp, gain=0.02)

        setattr(self.mhc_alpha_mlp, 'sequence_parallel', True)
        setattr(self.mhc_beta_mlp, 'sequence_parallel', True)
        setattr(self.mhc_phi_mlp, 'sequence_parallel', True)

    def _forward_attention(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_context (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.
            sequence_len_offset (Tensor, optional): Offset along sequence dimension
                during inference.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                hidden_states (Tensor): Transformed hidden states before the MLP layernorm.
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )

        inference_context = deprecate_inference_params(inference_context, inference_params)

        residual = x

        x = x.transpose(0, 1) # (b, s, h) -- I will fix this to sbh later
        n = self.mhc_streams
        H_pre, H_post, H_res = mhcFusedCombinedOperators(x, self.mhc_phi_attn, self.mhc_alpha_attn, self.mhc_beta_attn, n, iterations=5)
        input_layernorm_output = te.mhc.mHCPreOp.apply(x, H_pre, n)
        input_layernorm_output = input_layernorm_output.transpose(0, 1) # (s, b, h) -- megatron prefers this


        # Optional Input Layer norm
        # if self.recompute_input_layernorm:
        #     self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
        #     with off_interface(self.offload_attn_norm, hidden_states, "attn_norm") as hidden_states:
        #         input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
        #             apply_module(self.input_layernorm), hidden_states
        #         )
        # else:
        #     with off_interface(self.offload_attn_norm, hidden_states, "attn_norm") as hidden_states:
        #         input_layernorm_output = apply_module(self.input_layernorm)(hidden_states)

        # Self attention.
        nvtx_range_push(suffix="self_attention")
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        nvtx_range_pop(suffix="self_attention")

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        nvtx_range_push(suffix="self_attn_bda")
        with self.bias_dropout_add_exec_handler():
            # No residual connection, just bias + dropout
            hidden_states = mhc_bd(
                attention_output_with_bias, residual.dtype, self.hidden_dropout
            )
        nvtx_range_pop(suffix="self_attn_bda")

        # Delay the offload of the attention norm until after the self_attn_bda has been computed
        # because the residual is needed in the self_attn_bda.
        if self.offload_attn_norm:
            hidden_states = off_interface.group_commit(
                hidden_states, name="attn_norm", forced_released_tensors=[residual]
            )

        hidden_states = hidden_states.transpose(0, 1) # (b, s, h) -- I will fix this to sbh later

        # Currently the hyper connection still uses bsh so we need to use the transposed x and hidden_states
        post_out = te.mhc.mHCPostResOp.apply(hidden_states, H_post, x, H_res.view(H_res.shape[0], H_res.shape[1], n, n), n)
        # and after the hyper connection, we transpose back to sbh for megatron
        post_out = post_out.transpose(0, 1) # (s, b, h) -- megatron prefers this

        return post_out, context

    @copy_signature(_forward_attention)
    def forward(self, hidden_states: Tensor, *args, **kwargs):
        if self.layer_number == 1:
            hidden_states = hidden_states.repeat(1, 1, self.mhc_streams) # (s, b, h) -> (s, b, h * mhc_streams)

        logger.info(f"[RANK={torch.distributed.get_rank()}] Layer {self.layer_number} before attention absmax: {hidden_states.abs().max().item()}")

        hidden_states, context = self._forward_attention(hidden_states, *args, **kwargs)
        logger.info(f"[RANK={torch.distributed.get_rank()}] Layer {self.layer_number} after attention absmax: {hidden_states.abs().max().item()}")
        
        output = self._forward_mlp(
            hidden_states,
            kwargs.get("inference_context", None),
            padding_mask=kwargs.get("padding_mask", None),
        )
        logger.info(f"[RANK={torch.distributed.get_rank()}] Layer {self.layer_number} after MLP absmax: {output.abs().max().item()}")

        if self.layer_number == self.config.num_layers:
            output = output.view(output.shape[0], output.shape[1], -1, self.config.hidden_size) # (s, b, h * mhc_streams) -> (s, b, h)
            output = output.sum(dim=2) # sum over the mhc_streams dimension to get back to (s, b, h)

            logger.info(f"[RANK={torch.distributed.get_rank()}] Layer {self.layer_number} after summing mhc streams absmax: {output.abs().max().item()}")

        logger.info(f"[RANK={torch.distributed.get_rank()}] Layer {self.layer_number} self.mhc_alpha_attn absmax: {self.mhc_alpha_attn.abs().max().item()}, self.mhc_beta_attn absmax: {self.mhc_beta_attn.abs().max().item()}, self.mhc_phi_attn absmax: {self.mhc_phi_attn.abs().max().item()}")
        logger.info(f"[RANK={torch.distributed.get_rank()}] Layer {self.layer_number} self.mhc_alpha_mlp absmax: {self.mhc_alpha_mlp.abs().max().item()}, self.mhc_beta_mlp absmax: {self.mhc_beta_mlp.abs().max().item()}, self.mhc_phi_mlp absmax: {self.mhc_phi_mlp.abs().max().item()}")
        logger.info(f"[RANK={torch.distributed.get_rank()}] Layer {self.layer_number} output dtype: {output.dtype}")
        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            # logger.info(f"type(self.self_attention.linear_qkv): {type(self.self_attention.linear_qkv)}, type(self.mlp.linear_fc1): {type(self.mlp.linear_fc1)}")

        return output, context

    def _forward_mlp(
        self,
        x: Tensor,
        inference_context: BaseInferenceContext | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor | list[Tensor | None]:
        """
        Perform a forward pass through the feed-forward layer.

        Args:
            hidden_states (Tensor): Transformed hidden states before the MLP layernorm.
                Shape [seq_length, batch_size, hidden_size].
            inference_context: Inference context for optimizations.
            padding_mask (Tensor, optional): Padding mask for MoE routing.
                Shape [bsz, seq_length]. True = padding (exclude), False = valid (include).
                Only used for MoE layers to exclude padding tokens from aux loss computations.
                The MoELayer will internally transform this to [seq_length, bsz] format.
        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        residual = x

        x = x.transpose(0, 1) # (b, s, h) -- I will fix this to sbh later
        n = self.mhc_streams
        H_pre, H_post, H_res = mhcFusedCombinedOperators(x, self.mhc_phi_mlp, self.mhc_alpha_mlp, self.mhc_beta_mlp, n, iterations=5)
        pre_mlp_layernorm_output = te.mhc.mHCPreOp.apply(x, H_pre, n)
        pre_mlp_layernorm_output = pre_mlp_layernorm_output.transpose(0, 1) # (s, b, h) -- megatron prefers this

        if self.config.fp32_residual_connection:
            residual = residual.float()

        nvtx_range_push(suffix="mlp")
        # Potentially chunk the MLP computation during prefill to minimize the peak activation size
        should_chunk_mlp_for_prefill = (
            self.config.mlp_chunks_for_prefill > 1
            and inference_context is not None
            and not inference_context.is_decode_only()
            and not isinstance(self.mlp, IdentityOp)
            and not self.config.transformer_impl == "inference_optimized"
        )

        using_fused_tp_inference_kernel = (not self.training) and (
            self.config.inference_fuse_tp_communication
        )

        if self.recompute_mlp:
            if self.config.fp8 or self.config.fp4:
                # import here to avoid circular import
                from megatron.core.extensions.transformer_engine import te_checkpoint

                mlp_output_with_bias = te_checkpoint(
                    self.mlp,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    pre_mlp_layernorm_output,
                    padding_mask=padding_mask,
                )
            else:
                mlp_output_with_bias = tensor_parallel.checkpoint(
                    functools.partial(self.mlp, padding_mask=padding_mask),
                    False,
                    pre_mlp_layernorm_output,
                )
        elif should_chunk_mlp_for_prefill:
            # Chunk input along sequence dimension
            num_chunks = min(self.config.mlp_chunks_for_prefill, pre_mlp_layernorm_output.shape[0])
            chunks = pre_mlp_layernorm_output.chunk(num_chunks, dim=0)

            # Compute outputs for each chunk
            outputs = [self.mlp(chunk) for chunk in chunks]

            # Aggregate chunk outputs
            mlp_output = torch.cat([out for out, _ in outputs], dim=0)
            bias_chunks = [bias for _, bias in outputs if bias is not None]
            bias_output = torch.stack(bias_chunks, dim=0).sum(dim=0) if bias_chunks else None
            mlp_output_with_bias = (mlp_output, bias_output)
        else:
            if using_fused_tp_inference_kernel:
                # Set the residual for fused reduce-scatter + add + layer-norm + all-gather
                # operation in MLP's fc2.
                self._set_fc2_residual(residual)
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output, padding_mask=padding_mask)

        nvtx_range_pop(suffix="mlp")

        if (
            self.is_moe_layer
            and self.config.cuda_graph_impl == "transformer_engine"
            and self.training
            and is_graph_capturing()
            and CudaGraphScope.moe_router in self.config.cuda_graph_scope
        ):
            if self.recompute_pre_mlp_layernorm:
                # Register the recompute hooks to all the cudagraph output tensors, because some
                # tensors are in parallel execution paths and they all need pre_mlp_layernorm to be
                # recomputed in backward pass. For example, the router path and the shared expert
                # path. So only register in one path is risky.
                for tensor in mlp_output_with_bias:
                    self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(tensor)
            return list(mlp_output_with_bias) + [residual]
        else:
            return self._forward_post_mlp(mlp_output_with_bias, residual, x, H_post, H_res)
        
    def _forward_post_mlp(
        self, mlp_output_with_bias: tuple[Tensor, Tensor | None], residual: Tensor, x: Tensor, H_post: Tensor, H_res: Tensor
    ) -> Tensor:
        """
        Perform operations after the MLP computation.

        Args:
            mlp_output_with_bias (Tensor): Output tensor of the MLP layer with bias.
            residual (Tensor): Residual tensor.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )
        
        n = self.mhc_streams

        using_fused_tp_inference_kernel = (not self.training) and (
            self.config.inference_fuse_tp_communication
        )

        if self.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of mlp_output_with_bias[0]
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        nvtx_range_push(suffix="mlp_bda")
        if using_fused_tp_inference_kernel:
            # In inference optimized transformer layer, there is no bias and dropout
            # The remaining residual add is already handled inside the
            # MLP module.
            hidden_states = mlp_output_with_bias[0]
        else:
            with self.bias_dropout_add_exec_handler():
                # No residual connection, just bias + dropout
                hidden_states = mhc_bd(
                    mlp_output_with_bias, residual.dtype, self.hidden_dropout
                )
        nvtx_range_pop(suffix="mlp_bda")
        # Delay the offload of the mlp norm until after the mlp_bda has been computed
        # because the residual is needed in the mlp_bda.
        if self.offload_mlp_norm:
            hidden_states = off_interface.group_commit(
                hidden_states, name="mlp_norm", forced_released_tensors=[residual]
            )

        hidden_states = hidden_states.transpose(0, 1) # (b, s, h) -- I will fix this to sbh later

        # Currently the hyper connection still uses bsh so we need to use the transposed x and hidden_states
        post_out = te.mhc.mHCPostResOp.apply(hidden_states, H_post, x, H_res.view(H_res.shape[0], H_res.shape[1], n, n), n)
        # and after the hyper connection, we transpose back to sbh for megatron
        post_out = post_out.transpose(0, 1) # (s, b, h) -- megatron prefers this

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=post_out, requires_grad=post_out.requires_grad, keep_graph=True
        )

        return output

class MHCModel(GPTModel):
    
    def __init__(self, config: TransformerConfig, transformer_layer_spec: ModuleSpec, *args, **kwargs):
        transformer_layer_spec.module = MHCTransformerLayer

        super().__init__(config, transformer_layer_spec, *args, **kwargs)
