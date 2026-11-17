#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Full VL SFT: independent model, runtime and MPI/torchrun launch configuration.
set -euo pipefail
CURRENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${CURRENT_DIR}/../.." && pwd)
MODE=${1:-dryrun}
case "${MODE}" in dryrun|sft) ;; *) echo "Usage: $0 dryrun|sft" >&2; exit 2 ;; esac

# ===================================== Model and Training Config =====================================
export PYTHON_BIN=${PYTHON_BIN:-python3}
export SEQ_LENGTH=${SEQ_LENGTH:-4096}
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/public/home/yugeng/work/models/Qwen3.8-27B}
export DATA_PATH=${DATA_PATH:-/public/home/yugeng/work/datasets/mantis-birds-energon-v1}
export SAVE_DIR=${SAVE_DIR:-${ROOT}/outputs/qwen38-vl-full-sft-seq${SEQ_LENGTH}}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32} MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export TRAIN_ITERS=${TRAIN_ITERS:-500} WARMUP_ITERS=${WARMUP_ITERS:-20}
export LR=${LR:-1e-5} SAVE_INTERVAL=${SAVE_INTERVAL:-250}
export NUM_WORKERS=${NUM_WORKERS:-2} RECOMPUTE=${RECOMPUTE:-full}
export RESUME_FROM=${RESUME_FROM:-} LOG_THROUGHPUT=${LOG_THROUGHPUT:-true}
export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0}

# ===================================== Torch Profiler =====================================
export PROFILE=${PROFILE:-false}
export PROFILE_STEP_START=${PROFILE_STEP_START:-10} PROFILE_STEP_END=${PROFILE_STEP_END:-11}
export PROFILE_RANKS=${PROFILE_RANKS:-'[0]'}
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
export HOSTFILE=${HOSTFILE:-${CURRENT_DIR}/hostfile}
export NNODES=${NNODES:-$(awk 'NF && $1 !~ /^#/ {n++} END {print n}' "${HOSTFILE}")}
export GPUS=${GPUS:-8} TP=${TP:-4} PP=${PP:-2} CP=${CP:-1}
export MPI_SSH_PORT=${MPI_SSH_PORT:-${SSH_PORT:-11463}}
export MASTER_PORT=${MASTER_PORT:-30123}
if (( NNODES > 1 )); then
    export MASTER_ADDR=${MASTER_ADDR:-$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "${HOSTFILE}")}
else
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi
export PYTHONPATH="${ROOT}:${ROOT}/hcu_megatron_bridge/src:${ROOT}/3rdparty/Megatron-Bridge/src:${ROOT}/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM:${ROOT}/3rdparty/Megatron-Energon/src${PYTHONPATH:+:${PYTHONPATH}}"

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
        TRAIN_ITERS LR WARMUP_ITERS SAVE_INTERVAL NUM_WORKERS \
        RESUME_FROM SAVE_DIR MASTER_ADDR MASTER_PORT RECOMPUTE \
        OPTIMIZER_OFFLOAD_FRACTION MIN_PIXELS MAX_PIXELS \
        PROFILE PROFILE_STEP_START PROFILE_STEP_END PROFILE_RANKS PROFILE_RECORD_SHAPES PROFILE_WITH_STACK \
        LOG_THROUGHPUT \
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
    mpirun "${args[@]}" bash "${CURRENT_DIR}/run_qwen38_vl_sft_full.sh" "${MODE}" 2>&1 \
        | tee "${SAVE_DIR}/train_logs/mpi_$(date +%F-%H%M).log" || launch_status=$?
    exit "${launch_status}"
fi

# ===================================== Worker Launch =====================================
NODE_RANK=${OMPI_COMM_WORLD_RANK:-${NODE_RANK:-0}}
ARGS=("${CURRENT_DIR}/qwen38_vl_sft_worker.py"
      --recipe qwen38_vl_27b_sft_bw1000_bf16_config --mode sft --step_func qwen3_vl_step)
ARGS+=("${PROFILING_CONFIG[@]}")
if [[ ${MODE} == dryrun ]]; then
    "${PYTHON_BIN}" "${ARGS[@]}" --dry-run 2>&1 | tee "${SAVE_DIR}/train_logs/dryrun.log"
elif [[ -n ${OMPI_COMM_WORLD_RANK:-} ]]; then
    exec "${PYTHON_BIN}" -m torch.distributed.run --nnodes="${NNODES}" --nproc_per_node="${GPUS}" \
        --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" "${ARGS[@]}"
else
    "${PYTHON_BIN}" -m torch.distributed.run --nnodes="${NNODES}" --nproc_per_node="${GPUS}" \
        --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
        "${ARGS[@]}" 2>&1 | tee "${SAVE_DIR}/train_logs/node-${NODE_RANK}.log"
fi
