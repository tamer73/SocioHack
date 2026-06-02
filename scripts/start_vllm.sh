#!/bin/bash

# Set environment variables to avoid tokenizers parallelism warnings
export TOKENIZERS_PARALLELISM=false
# Avoid picking up broken user-level site-packages (e.g. pyarrow)
export PYTHONNOUSERSITE=1

# B200 GPU support: use a different conda env if sociohack doesn't support B200
# Set VLLM_CONDA_ENV to a B200-compatible env (e.g. vllm_b200), or pass as 2nd arg
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-${2}}"
if [ -n "$VLLM_CONDA_ENV" ]; then
    echo "Using conda env for vLLM (B200): $VLLM_CONDA_ENV"
    eval "$(conda shell.bash hook)"
    conda activate "$VLLM_CONDA_ENV"
fi

# Check if experiment name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Available experiments:"
    ls configs/*.yaml | grep -v base.yaml | sed 's,configs/,,' | sed 's/\.yaml//'
    exit 1
fi

# Get config value from config file
get_config() {
    local key="$1"
    local default="$2"
    local config_name="${3:-base}"
    
    python3 src/simple_config_loader.py --config-name "$config_name" --key "$key" --default "$default" 2>/dev/null
}

# Get config name from command line
CONFIG_NAME="${1:-base}"

# Get config values
MODEL_NAME=$(get_config "model.name_vllm" "YOUR_MODEL_NAME_VLLM" "$CONFIG_NAME")
PORT=$(get_config "vllm.port" "8422" "$CONFIG_NAME")
GPU_MEM_UTIL=$(get_config "vllm.gpu_memory_utilization" "0.85" "$CONFIG_NAME")
TENSOR_PARALLEL_SIZE=$(get_config "gpu.vllm.tensor_parallel_size" "1" "$CONFIG_NAME")
GPU_IDS=$(get_config "gpu.vllm.gpu_ids" "1" "$CONFIG_NAME")
# Limit KV cache window to avoid OOM (Qwen3.5 default context is 262144 tokens).
# 32768 covers our max_completion_length (2048) + prompt overhead with headroom.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU_IDS
# B200 workaround: FlashInfer JIT fails when system nvcc doesn't support compute_100a.
# Force FLASH_ATTN to bypass FlashInfer (override with VLLM_ATTENTION_BACKEND if needed).
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export MASTER_PORT=29506 
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

echo "Starting vLLM server for model: $MODEL_NAME"
echo "Using GPU: $GPU_IDS, Port: $PORT, GPU Memory: $GPU_MEM_UTIL"

# start vllm server
# --max_model_len: limits KV cache window, avoids OOM on models with large default context.
#   from writing compiled graphs to ~/.cache/vllm/torch_compile_cache. Slightly slower
#   first-token latency but saves several GB of disk writes per startup.
NCCL_DEBUG=INFO trl vllm-serve --model "$MODEL_NAME" --port "$PORT" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --max_model_len "$MAX_MODEL_LEN"