#!/bin/bash

# Environment variables for performance tuning (matching your first script)
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16

CHECKPOINT_PATH=${1:-"checkpoints/gpt3_2b_bf16"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/gpt3_2b_bf16"}
TOKENIZER_ARG=${3:-"/lustre/fsw/coreai_dlfw_dev/gpt_1T_data/nemotron_2_256k.model"} # Path to tokenizer model, or "MOCK"
DATA_ARG=${4:-"wikitext_text_document"} # Data prefix, or "MOCK"

# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Distributed training setup
GPUS_PER_NODE=4
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters (GPT3 2B config)
TP_SIZE=1     
CP_SIZE=1     
PP_SIZE=1     
MICRO_BATCH_SIZE=96
GLOBAL_BATCH_SIZE=768
NUM_LAYERS=4
DTYPE="bf16"
SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=1024

# Data cache path
DATA_CACHE_PATH="${PWD}/benchmark_cache_gpt3_2b_bf16"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --attention-backend flash
    --num-layers $NUM_LAYERS
    --hidden-size 2048
    --ffn-hidden-size 5504
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 4
    --kv-channels 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000 
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --init-method-std 0.0198
    --untie-embeddings-and-output-weights
    --disable-bias-linear 
    --enable-hyper-connections
    --num-residual-streams 4
    --mhc-sinkhorn-iterations 20
    --mhc-init-gating-factor 0.01
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters 1000
    --lr-decay-iters 900
    --lr-warmup-iters 100
    --lr 1.2e-3
    --min-lr 1.2e-5
    --decoupled-lr 5.0e-4      # Specific to decoupled AdamW, ensure optimizer is compatible
    --decoupled-min-lr 4.5e-5  # Specific to decoupled AdamW
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --manual-gc 
    --rerun-mode disabled
    --exit-duration-in-mins 230 
)

# Distributed Data Parallel (DDP) arguments
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
)

# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"
        "--vocab-size 128256" 
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--tiktoken-pattern v2" 
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 8"
    )
else
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--data-path $DATA_ARG"
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split 99,1,0"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 8"
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --eval-interval 100
    --save-interval 50
    --log-params-norm
    --log-num-zeros-in-grad
    --log-memory-to-tensorboard
    --log-throughput
    --no-one-logger
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH" 
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

# Run the training command
TORCH_NCCL_TRACE_BUFFER_SIZE=1000 torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

set +x
