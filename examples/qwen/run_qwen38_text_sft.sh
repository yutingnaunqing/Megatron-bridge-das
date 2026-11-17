#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Text SFT: one entry for preparation and native MPI/torchrun training.
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
export EXPECTED_NUM_LAYERS=${EXPECTED_NUM_LAYERS:-64}
export MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-0}
export MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-0.1}
if (( MTP_NUM_LAYERS == 1 )); then
    export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/public/home/yugeng/work/models/Qwen3.8-27B-text-64layers-mtp1}
else
    export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/public/home/yugeng/work/models/Qwen3.8-27B-text-64layers}
fi
export SEQ_LENGTH=${SEQ_LENGTH:-4096} THINKING_MODE=${THINKING_MODE:-disabled}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-16384} VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-256}
export RAW_DATA_PATH=${RAW_DATA_PATH:-/public/home/yugeng/work/datasets/HuggingfaceH4_ultrachat_200k}
if [[ ${SEQ_LENGTH} == 1024 && ${THINKING_MODE} == disabled && ${TRAIN_SAMPLES} == 16384 && ${VALIDATION_SAMPLES} == 256 ]]; then
    export DATA_PATH=${DATA_PATH:-/public/home/yugeng/work/datasets/ultrachat-qwen38-steady-train16384-val256-seq1024-disabled}
else
    export DATA_PATH=${DATA_PATH:-/public/home/yugeng/work/datasets/ultrachat-qwen38-train${TRAIN_SAMPLES}-val${VALIDATION_SAMPLES}-seq${SEQ_LENGTH}-${THINKING_MODE}}
fi

# ===================================== Training Config =====================================
export TRAIN_ITERS=${TRAIN_ITERS:-500} WARMUP_ITERS=${WARMUP_ITERS:-20}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64} MICRO_BATCH_SIZE=1
export LR=${LR:-1e-5} RECOMPUTE=${RECOMPUTE:-full}
export TP_COMM_OVERLAP=${TP_COMM_OVERLAP:-false}
export APPLY_ROPE_FUSION=${APPLY_ROPE_FUSION:-true}
export SAVE_INTERVAL=${SAVE_INTERVAL:-250} EVAL_INTERVAL=${EVAL_INTERVAL:-100} EVAL_ITERS=${EVAL_ITERS:-8}
export EXIT_INTERVAL=${EXIT_INTERVAL:-0} NUM_WORKERS=${NUM_WORKERS:-4}
export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0}
export MTP_AUDIT_INTERVAL=${MTP_AUDIT_INTERVAL:-100}
export LOG_THROUGHPUT=${LOG_THROUGHPUT:-true} RESUME_FROM=${RESUME_FROM:-}
export SAVE_DIR=${SAVE_DIR:-${ROOT}/outputs/qwen38-text-native-seq${SEQ_LENGTH}/$(date +%Y%m%d-%H%M%S)}

# ===================================== Torch Profiler =====================================
# Capture steps [start, end); traces: SAVE_DIR/torch_profile/rank-*.json.gz.
export PROFILE=${PROFILE:-false}
export PROFILE_STEP_START=${PROFILE_STEP_START:-5} PROFILE_STEP_END=${PROFILE_STEP_END:-6}
export PROFILE_RANKS=${PROFILE_RANKS:-'[0,4,8,12]'}
export PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-false} PROFILE_WITH_STACK=${PROFILE_WITH_STACK:-false}
for name in PROFILE PROFILE_RECORD_SHAPES PROFILE_WITH_STACK APPLY_ROPE_FUSION; do
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
export HOSTFILE=${HOSTFILE:-${CURRENT_DIR}/hostfile}
export NNODES=${NNODES:-$(awk 'NF && $1 !~ /^#/ {n++} END {print n}' "${HOSTFILE}")}
export GPUS=${GPUS:-8} TP=${TP:-2} PP=${PP:-4} CP=${CP:-1}
# Empty means even splitting. Remaining middle stages share the other layers equally.
# For 64 layers / PP4: last=13 gives [17,17,17,13]; first=20,last=12 gives [20,16,16,12].
export NUM_LAYERS_IN_FIRST_PIPELINE_STAGE=${NUM_LAYERS_IN_FIRST_PIPELINE_STAGE:-17}
export NUM_LAYERS_IN_LAST_PIPELINE_STAGE=${NUM_LAYERS_IN_LAST_PIPELINE_STAGE:-13}
export MPI_SSH_PORT=${MPI_SSH_PORT:-${SSH_PORT:-11463}}
export MASTER_PORT=${MASTER_PORT:-30121}
if (( NNODES > 1 )); then
    export MASTER_ADDR=${MASTER_ADDR:-$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "${HOSTFILE}")}
else
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi
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

# ===================================== MPI Launch =====================================
if [[ ${MODE} == sft ]] && (( NNODES > 1 )) && [[ -z ${OMPI_COMM_WORLD_RANK:-} && -z ${NODE_RANK:-} ]]; then
    prte_tmp=${PRTE_TMPDIR_BASE:-/public/home/yugeng/work/.q38tmp/prte}
    mkdir -p "${prte_tmp}"
    slots=${SAVE_DIR}/hostfile.slots
    awk -v count="${NNODES}" 'NF && $1 !~ /^#/ {if (++n <= count) print $1 " slots=1"}' "${HOSTFILE}" > "${slots}"
    args=(--hostfile "${slots}" -np "${NNODES}" --map-by ppr:1:node --bind-to none
          --prtemca prte_tmpdir_base "${prte_tmp}"
          --mca plm_rsh_no_tree_spawn 1
          --mca plm_rsh_args "-p ${MPI_SSH_PORT} -o BatchMode=yes -o ConnectTimeout=10")
    if (( EUID == 0 )); then args+=(--allow-run-as-root); fi
    for key in PATH LD_LIBRARY_PATH LIBRARY_PATH ROCM_PATH HYHAL_PATH PYTHON_BIN DTK_ENV \
        LOCAL_HF_PATH DATA_PATH HOSTFILE MPI_SSH_PORT NNODES GPUS TP PP CP SEQ_LENGTH MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE \
        TRAIN_ITERS LR WARMUP_ITERS EXIT_INTERVAL SAVE_INTERVAL EVAL_ITERS EVAL_INTERVAL NUM_WORKERS \
        THINKING_MODE RESUME_FROM SAVE_DIR MASTER_ADDR MASTER_PORT RECOMPUTE EXPECTED_NUM_LAYERS \
        OPTIMIZER_OFFLOAD_FRACTION MTP_NUM_LAYERS MTP_LOSS_SCALING_FACTOR MTP_AUDIT_INTERVAL \
        PROFILE PROFILE_STEP_START PROFILE_STEP_END PROFILE_RANKS PROFILE_RECORD_SHAPES PROFILE_WITH_STACK \
        LOG_THROUGHPUT TP_COMM_OVERLAP APPLY_ROPE_FUSION \
        NUM_LAYERS_IN_FIRST_PIPELINE_STAGE NUM_LAYERS_IN_LAST_PIPELINE_STAGE \
        HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES IB_TYPE NCCL_NET NCCL_IB_DISABLE NCCL_IB_HCA \
        NCCL_SOCKET_IFNAME NCCL_IB_GID_INDEX NCCL_TOPO_FILE NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE \
        NCCL_NET_PLUGIN RCCL_NET_PLANE SHCA_SHUT_UP_FWB RCCL_P2P_XHCL_CHANNEL_NUM \
        HYLOG_DISABLE_ERROR_LOG QWEN38_CACHE TMPDIR TRITON_CACHE_DIR TRITON_HOME TORCHINDUCTOR_CACHE_DIR \
        MIOPEN_USER_DB_PATH MIOPEN_CUSTOM_CACHE_DIR MIOPEN_FIND_MODE TORCH_EXTENSIONS_DIR \
        HF_HOME HF_HUB_CACHE NEMO_HOME XDG_CACHE_HOME TORCH_HOME PYTHONDONTWRITEBYTECODE \
        HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; do
        if [[ -v ${key} ]]; then args+=(-x "${key}"); fi
    done
    launch_status=0
    mpirun "${args[@]}" bash "${CURRENT_DIR}/run_qwen38_text_sft.sh" "${MODE}" 2>&1 \
        | tee "${SAVE_DIR}/train_logs/mpi_$(date +%F-%H%M).log" || launch_status=$?
    exit "${launch_status}"
fi

# ===================================== Worker Launch =====================================
NODE_RANK=${OMPI_COMM_WORLD_RANK:-${NODE_RANK:-0}}
ARGS=("${CURRENT_DIR}/qwen38_text_sft_worker.py" --recipe qwen38_text_27b_sft_bw1000_bf16_config
      --mode sft --step_func gpt_step)
ARGS+=("${PROFILING_CONFIG[@]}")
if [[ ${MODE} == dryrun ]]; then
    "${PYTHON_BIN}" "${ARGS[@]}" --dry-run
elif [[ -n ${OMPI_COMM_WORLD_RANK:-} ]]; then
    exec "${PYTHON_BIN}" -m torch.distributed.run --nnodes="${NNODES}" --nproc_per_node="${GPUS}" \
        --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" "${ARGS[@]}"
else
    "${PYTHON_BIN}" -m torch.distributed.run --nnodes="${NNODES}" --nproc_per_node="${GPUS}" \
        --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
        "${ARGS[@]}" 2>&1 | tee "${SAVE_DIR}/train_logs/node-${NODE_RANK}.log"
fi
