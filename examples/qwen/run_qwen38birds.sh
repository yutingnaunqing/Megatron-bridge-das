#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail
CURRENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
MEGATRON_PATH=$( dirname "$( dirname "${CURRENT_DIR}" )" )

# Usage: bash examples/qwen/run_qwen38_vl_birds.sh [train|dryrun]
MODE=${1:-train}
if (( $# > 1 )) || [[ ${MODE} != train && ${MODE} != dryrun ]]; then
    echo "Usage: $0 [train|dryrun]" >&2
    exit 1
fi

# ===================================== Model and Data =====================================
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/public/home/yugeng/work/models/Qwen3.8-27B}
export DATA_PATH=${DATA_PATH:-/public/home/yugeng/work/datasets/mantis-birds-energon-v1}
export SAVE_DIR=${SAVE_DIR:-/public/home/yugeng/work/checkpoints/mantis-birds-two-node-$(date +%Y%m%d_%H%M%S)}

# ===================================== Train Config =====================================
export SEQ_LENGTH=${SEQ_LENGTH:-4096}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0}

# ===================================== Parallel Config =====================================
export GPUS=8 TP=4 PP=2 CP=1
export SSH_PORT=${SSH_PORT:-11463}
export MASTER_PORT=${MASTER_PORT:-29743}
HOSTFILE=${HOSTFILE:-${CURRENT_DIR}/hostfile}

# ===================================== Torch Profiler =====================================
export PROFILE=${PROFILE:-true}
export PROFILE_STEP_START=${PROFILE_STEP_START:-5}
export PROFILE_STEP_END=${PROFILE_STEP_END:-6}
export PROFILE_RANKS=${PROFILE_RANKS:-'[0,4,8,12]'}
export PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-false}
export PROFILE_WITH_STACK=${PROFILE_WITH_STACK:-false}

# ===================================== Checkpoint and Entry =====================================
unset CHECKPOINT_PATH
export CHECKPOINT_DIR=null LOAD_CHECKPOINT=null
export RECIPE_RUNNER=${MEGATRON_PATH}/run_recipe.py

# ===================================== Execution =====================================
exec bash "${CURRENT_DIR}/train_qwen38_vl_pretrain.sh" \
    qwen38_vl_27b_pretrain_bw1000_bf16_config \
    "${MODE}" "${HOSTFILE}" 2
