#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail
CURRENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
MEGATRON_PATH=$( dirname "$( dirname "${CURRENT_DIR}" )" )

# 16 language layers plus the full vision encoder on one 8-GPU node.
# Usage: bash examples/qwen/run_qwen38_vl_single_node.sh [train|dryrun] [KEY=VALUE ...]
MODE=${1:-train}
if [[ ${MODE} != train && ${MODE} != dryrun ]]; then
    echo "Usage: $0 [train|dryrun] [KEY=VALUE ...]" >&2
    exit 1
fi
if (( $# > 0 )); then shift; fi

# ===================================== Model and Data =====================================
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/public/home/yugeng/work/models/Qwen3.8-27B}
export DATA_PATH=${DATA_PATH:-/public/home/yugeng/work/datasets/star-source/star-energon}
export SAVE_DIR=${SAVE_DIR:-/public/home/yugeng/work/checkpoints/star-single-node-16layers-$(date +%Y%m%d_%H%M%S)}

# ===================================== Train Config =====================================
export SEQ_LENGTH=${SEQ_LENGTH:-4096}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export TRAIN_ITERS=${TRAIN_ITERS:-15}

# ===================================== Optimizer Config =====================================
export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0}
# CPU threads per rank; overrides env.sh's OMP_NUM_THREADS=1.
export TRAIN_OMP_NUM_THREADS=${TRAIN_OMP_NUM_THREADS:-4}
CPU_OPTIMIZER_TRANSFER_OVERLAP=${CPU_OPTIMIZER_TRANSFER_OVERLAP:-false}

# ===================================== Parallel Config =====================================
export GPUS=${GPUS:-8} TP=${TP:-4} PP=${PP:-2} CP=${CP:-1}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29745}
TP_COMM_OVERLAP=${TP_COMM_OVERLAP:-false}
OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-false}
OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-false}
export MIOPEN_FIND_MODE=3

# ===================================== Recompute Config =====================================
# Optimizer offload can be combined with recompute; TE activation offload stays off.
RECOMPUTE=${RECOMPUTE:-none}
case ${RECOMPUTE} in
    full) RECOMPUTE_ARGS=(model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1) ;;
    none) RECOMPUTE_ARGS=(model.recompute_granularity=null model.recompute_method=null model.recompute_num_layers=null) ;;
    *) echo "RECOMPUTE must be full or none" >&2; exit 1 ;;
esac


# ===================================== Torch Profiler =====================================
export PROFILE=${PROFILE:-false}
export PROFILE_STEP_START=${PROFILE_STEP_START:-10}
export PROFILE_STEP_END=${PROFILE_STEP_END:-11}
export PROFILE_RANKS=${PROFILE_RANKS:-'[0,4]'}
export PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-false}
export PROFILE_WITH_STACK=${PROFILE_WITH_STACK:-false}

# ===================================== Checkpoint and Entry =====================================
# Fresh initialization for performance testing, with no checkpoint I/O.
unset CHECKPOINT_PATH
export CHECKPOINT_DIR=null LOAD_CHECKPOINT=null
export RECIPE_RUNNER=${MEGATRON_PATH}/run_recipe.py

# ===================================== Execution =====================================
# Empty HOSTFILE plus NODES=1 selects the launcher's local torchrun branch.
exec bash "${CURRENT_DIR}/train_qwen38_vl_pretrain.sh" \
    qwen38_vl_27b_pretrain_8gpu_bw1000_bf16_config "${MODE}" "" 1 \
    "${RECOMPUTE_ARGS[@]}" \
    model.cpu_offloading=false \
    model.freeze_vision_model=false \
    model.freeze_vision_projection=false \
    model.bias_activation_fusion=true \
    comm_overlap.tp_comm_overlap=true \
    comm_overlap.overlap_grad_reduce=false \
    comm_overlap.overlap_param_gather=false \
    optimizer.overlap_cpu_optimizer_d2h_h2d=false \
    "$@"
