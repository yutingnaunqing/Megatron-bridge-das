#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Independent single-node reduced-layer SFT and torch profiler entry.
set -euo pipefail
CURRENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${CURRENT_DIR}/../.." && pwd)
MODE=${1:-dryrun}
case "${MODE}" in
    convert|prepare|preflight|dryrun|sft) ;;
    *) echo "Usage: $0 convert|prepare|preflight|dryrun|sft" >&2; exit 2 ;;
esac

# ===================================== Model and Data =====================================
export PYTHON_BIN=${PYTHON_BIN:-python3}
export EXPECTED_NUM_LAYERS=${EXPECTED_NUM_LAYERS:-16}
export MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-0}
export MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-0.1}
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/public/home/yugeng/work/models/Qwen3.8-27B-text-16layers}
export SEQ_LENGTH=${SEQ_LENGTH:-4096} THINKING_MODE=${THINKING_MODE:-disabled}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-16384} VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-256}
export RAW_DATA_PATH=${RAW_DATA_PATH:-/public/home/yugeng/work/datasets/HuggingfaceH4_ultrachat_200k}
if [[ ${SEQ_LENGTH} == 1024 && ${THINKING_MODE} == disabled && ${TRAIN_SAMPLES} == 16384 && ${VALIDATION_SAMPLES} == 256 ]]; then
    export DATA_PATH=${DATA_PATH:-/public/home/yugeng/work/datasets/ultrachat-qwen38-steady-train16384-val256-seq1024-disabled}
else
    export DATA_PATH=${DATA_PATH:-/public/home/yugeng/work/datasets/ultrachat-qwen38-train${TRAIN_SAMPLES}-val${VALIDATION_SAMPLES}-seq${SEQ_LENGTH}-${THINKING_MODE}}
fi

# ===================================== Training Config =====================================
export TRAIN_ITERS=${TRAIN_ITERS:-50} WARMUP_ITERS=${WARMUP_ITERS:-2}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32} MICRO_BATCH_SIZE=1
export LR=${LR:-1e-5} RECOMPUTE=${RECOMPUTE:-none}
export TP_COMM_OVERLAP=${TP_COMM_OVERLAP:-false}
export APPLY_ROPE_FUSION=${APPLY_ROPE_FUSION:-true}
export SAVE_INTERVAL=${SAVE_INTERVAL:-0} EVAL_INTERVAL=${EVAL_INTERVAL:-100} EVAL_ITERS=${EVAL_ITERS:-0}
export EXIT_INTERVAL=${EXIT_INTERVAL:-0} NUM_WORKERS=${NUM_WORKERS:-4}
export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0}
export MTP_AUDIT_INTERVAL=${MTP_AUDIT_INTERVAL:-100}
export LOG_THROUGHPUT=${LOG_THROUGHPUT:-true} RESUME_FROM=${RESUME_FROM:-}
export SAVE_DIR=${SAVE_DIR:-${ROOT}/outputs/qwen38-text-one-node-16layers-seq${SEQ_LENGTH}/$(date +%Y%m%d-%H%M%S)}

# ===================================== Torch Profiler =====================================
# Capture steps [start, end); traces: SAVE_DIR/torch_profile/rank-*.json.gz.
export PROFILE=${PROFILE:-true}
export PROFILE_STEP_START=${PROFILE_STEP_START:-5} PROFILE_STEP_END=${PROFILE_STEP_END:-6}
export PROFILE_RANKS=${PROFILE_RANKS:-'[0,2,4,6]'}
export PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-false} PROFILE_WITH_STACK=${PROFILE_WITH_STACK:-false}
for name in PROFILE PROFILE_RECORD_SHAPES PROFILE_WITH_STACK; do
    case "${!name,,}" in
        1|true) printf -v "${name}" '%s' true ;;
        0|false) printf -v "${name}" '%s' false ;;
        *) echo "${name} must be true/false or 1/0" >&2; exit 2 ;;
    esac
done
PROFILING_CONFIG=("profiling.use_pytorch_profiler=${PROFILE}" profiling.use_nsys_profiler=false)
if [[ ${PROFILE} == true ]]; then
    PROFILING_CONFIG+=(
        "profiling.profile_step_start=${PROFILE_STEP_START}"
        "profiling.profile_step_end=${PROFILE_STEP_END}"
        "profiling.profile_ranks=${PROFILE_RANKS}"
        "profiling.pytorch_profiler_collect_shapes=${PROFILE_RECORD_SHAPES}"
        "profiling.pytorch_profiler_collect_callstack=${PROFILE_WITH_STACK}"
    )
fi

# ===================================== Parallel Config =====================================
export NNODES=1 NODE_RANK=0
unset OMPI_COMM_WORLD_RANK
export GPUS=${GPUS:-8} TP=${TP:-2} PP=${PP:-4} CP=${CP:-1}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} MASTER_PORT=${MASTER_PORT:-30125}
export PYTHONPATH="${ROOT}:${ROOT}/hcu_megatron_bridge/src:${ROOT}/3rdparty/Megatron-Bridge/src:${ROOT}/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"

# convert/preflight are explicit tools; training never invokes the checker.
if [[ ${MODE} == convert ]]; then
    exec "${PYTHON_BIN}" "${ROOT}/tools/checkpoint/convert_qwen38_vl_to_text.py" \
        --source "${SOURCE_HF_PATH:-/public/home/yugeng/work/models/Qwen3.8-27B}" \
        --output "${LOCAL_HF_PATH}" --num-layers "${EXPECTED_NUM_LAYERS}" --mtp-num-layers "${MTP_NUM_LAYERS}"
elif [[ ${MODE} == preflight ]]; then
    exec "${PYTHON_BIN}" "${ROOT}/tools/checkpoint/check_qwen38_text_sft.py"
fi

# ===================================== Runtime and Cache =====================================
# The vendor environment references optional variables on a fresh SSH shell.
set +u
source "${DTK_ENV:-/opt/dtk/env.sh}"
set -u
source "${ROOT}/env/env.sh"
export HYLOG_DISABLE_ERROR_LOG=1
export NCCL_IB_DISABLE=0 NCCL_NET="NCCL RDMA Plugin v8"
export NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export NCCL_DEBUG_FILE=${SAVE_DIR}/transport/nccl_%h_%p.log
export TMPDIR=/public/home/yugeng/work/.q38tmp/$(hostname)
export QWEN38_CACHE=${QWEN38_CACHE:-${ROOT}/outputs/qwen38-sft-cache}
export TRITON_CACHE_DIR=${QWEN38_CACHE}/triton
export TRITON_HOME=${QWEN38_CACHE}/triton-home
export TORCHINDUCTOR_CACHE_DIR=${QWEN38_CACHE}/inductor
export MIOPEN_USER_DB_PATH=${QWEN38_CACHE}/miopen-db
export MIOPEN_CUSTOM_CACHE_DIR=${QWEN38_CACHE}/miopen-kernels
export TORCH_EXTENSIONS_DIR=${QWEN38_CACHE}/torch-extensions
export HF_HOME=${QWEN38_CACHE}/huggingface NEMO_HOME=${QWEN38_CACHE}/nemo
export XDG_CACHE_HOME=${QWEN38_CACHE}/xdg TORCH_HOME=${QWEN38_CACHE}/torch-home
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "${TMPDIR}" "${SAVE_DIR}/transport" "${SAVE_DIR}/train_logs" "${QWEN38_CACHE}"

if [[ ${MODE} == prepare ]]; then
    exec "${PYTHON_BIN}" "${ROOT}/data_preparation/prepare_ultrachat_sft.py" \
        --source "${RAW_DATA_PATH}" --output "${DATA_PATH}" \
        --max-train-samples "${TRAIN_SAMPLES}" --max-validation-samples "${VALIDATION_SAMPLES}" \
        --check-tokens --model "${LOCAL_HF_PATH}" --seq-length "${SEQ_LENGTH}" --thinking-mode "${THINKING_MODE}"
fi

# ===================================== Worker Launch =====================================
ARGS=("${CURRENT_DIR}/qwen38_text_sft_worker.py" --recipe qwen38_text_27b_sft_bw1000_bf16_config
      --mode sft --step_func gpt_step)
ARGS+=("${PROFILING_CONFIG[@]}")
if [[ ${MODE} == dryrun ]]; then
    "${PYTHON_BIN}" "${ARGS[@]}" --dry-run
else
    "${PYTHON_BIN}" -m torch.distributed.run --nnodes="${NNODES}" --nproc_per_node="${GPUS}" \
        --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
        "${ARGS[@]}" 2>&1 | tee "${SAVE_DIR}/train_logs/node-${NODE_RANK}.log"
fi
