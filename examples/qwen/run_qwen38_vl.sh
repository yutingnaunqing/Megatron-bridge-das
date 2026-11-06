#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail
CURRENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
MEGATRON_PATH=$( dirname "$( dirname "${CURRENT_DIR}" )" )

# Run inside yg_qwen38 on m09r2n07; both nodes use the same mounted paths.
# Usage: bash examples/qwen/run_qwen38_vl.sh [train|dryrun] [KEY=VALUE ...]
MODE=${1:-train}
if [[ ${MODE} != train && ${MODE} != dryrun ]]; then
    echo "Usage: $0 [train|dryrun] [KEY=VALUE ...]" >&2
    exit 1
fi
if (( $# > 0 )); then shift; fi
for arg in "$@"; do
    [[ ${arg} == *=* ]] || { echo "Expected KEY=VALUE override: ${arg}" >&2; exit 1; }
done

# ===================================== Model and Data =====================================
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/public/home/yugeng/work/models/Qwen3.8-27B}
export DATA_PATH=${DATA_PATH:-/public/home/yugeng/work/datasets/star-source/star-energon}
export SAVE_DIR=${SAVE_DIR:-/public/home/yugeng/work/checkpoints/star-two-node-$(date +%Y%m%d_%H%M%S)}

# ===================================== Train Config =====================================
# Star has 3001 train samples. TRAIN_ITERS is a benchmark window, not an epoch.
# 4K still needs a memory check; changing datasets alone does not prevent OOM.
export SEQ_LENGTH=${SEQ_LENGTH:-4096}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export TRAIN_ITERS=${TRAIN_ITERS:-15}
export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0}

# ===================================== Parallel Config =====================================
export GPUS=${GPUS:-8} TP=${TP:-4} PP=${PP:-2} CP=${CP:-1}
TP_COMM_OVERLAP=${TP_COMM_OVERLAP:-true}
OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-false}
OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-false}
export MIOPEN_FIND_MODE=3
# Optional convolution tuning: MIOPEN_FIND_MODE=3 (hybrid), or 1 (full find).
# The launcher forwards it to both nodes when explicitly exported by the caller.
export SSH_PORT=${SSH_PORT:-11463}
export MASTER_PORT=${MASTER_PORT:-29743}
HOSTFILE=${HOSTFILE:-${CURRENT_DIR}/hostfile}

# ===================================== Torch Profiler =====================================
# Enable with: PROFILE=true bash examples/qwen/run_qwen38_vl.sh train
# Keep disabled for throughput measurements. END - START steps are captured;
# TRAIN_ITERS must reach END. These defaults assume a fresh run.
export PROFILE=${PROFILE:-true}
export PROFILE_STEP_START=${PROFILE_STEP_START:-10}
export PROFILE_STEP_END=${PROFILE_STEP_END:-11}
# Two pipeline stages at TP4, with less tracing overhead. Override for other layouts.
export PROFILE_RANKS=${PROFILE_RANKS:-'[0,4,8,12]'}
export PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-false}
export PROFILE_WITH_STACK=${PROFILE_WITH_STACK:-false}
# Output: ${SAVE_DIR}/torch_profile/rank-N.json.gz

# ===================================== Checkpoint and Entry =====================================
# Match the completed single-node run: fresh initialization, no checkpoint I/O.
unset CHECKPOINT_PATH
export CHECKPOINT_DIR=null LOAD_CHECKPOINT=null
export RECIPE_RUNNER=${MEGATRON_PATH}/run_recipe.py

# ===================================== Execution =====================================
exec bash "${CURRENT_DIR}/train_qwen38_vl_pretrain.sh" \
    qwen38_vl_27b_pretrain_bw1000_bf16_config \
    "${MODE}" "${HOSTFILE}" 2 \
    comm_overlap.tp_comm_overlap=true \
    comm_overlap.overlap_grad_reduce=false \
    comm_overlap.overlap_param_gather=false \
    model.bias_activation_fusion=true \
    model.recompute_granularity=full \
    model.recompute_method=uniform \
    model.recompute_num_layers=1 \
    "$@"
    
