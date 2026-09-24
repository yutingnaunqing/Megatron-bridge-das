#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# 使用示例（在项目根目录执行）：
#   GPUS=1 SEQ_LENGTH=32 TRAIN_ITERS=1 bash examples/qwen38/train_qwen38_flash_next_pretrain.sh tiny train
#   GPUS=8 SEQ_LENGTH=128 TRAIN_ITERS=3 bash examples/qwen38/train_qwen38_flash_next_pretrain.sh smoke train
# 第一个位置参数为 tiny、smoke 或 official；第二个为 dryrun 或 train。
set -euo pipefail
CURRENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${CURRENT_DIR}/../.." && pwd)
PROFILE=${1:-smoke}
MODE=${2:-dryrun}
if (( $# >= 2 )); then shift 2; else set --; fi
case "$PROFILE" in tiny|smoke|official) ;; *) echo "Profile must be tiny, smoke or official" >&2; exit 2;; esac
case "$MODE" in dryrun|train) ;; *) echo "Mode must be dryrun or train" >&2; exit 2;; esac
GPUS=${GPUS:-8}
TRAIN_ITERS=${TRAIN_ITERS:-3}
SEQ_LENGTH=${SEQ_LENGTH:-128}
NUM_LAYERS=${NUM_LAYERS:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$GPUS}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/runs/qwen38/${PROFILE}_$(date +%Y%m%d_%H%M%S)}
PYTHON=${PYTHON:-python}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/hcu_megatron_bridge/src:${PROJECT_ROOT}/3rdparty/Megatron-Bridge/src:${PROJECT_ROOT}/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM:${PROJECT_ROOT}/3rdparty/Megatron-Energon/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTHONUNBUFFERED=1
export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-ERROR}
export GLOG_minloglevel=${GLOG_minloglevel:-3}
export HSA_FORCE_FINE_GRAIN_PCIE=${HSA_FORCE_FINE_GRAIN_PCIE:-1}
export HF_HUB_OFFLINE=1
# This launcher uses one node and DP only. Do not copy DeepSeek-specific IB,
# DSA, mHC, DeepEP or recipe-file replacement settings into this experiment.
mkdir -p -- "${OUTPUT_DIR}/train_logs"
ARGS=("${CURRENT_DIR}/run_qwen38.py" --profile "$PROFILE" --num-layers "$NUM_LAYERS"
      --seq-length "$SEQ_LENGTH" --train-iters "$TRAIN_ITERS"
      --global-batch-size "$GLOBAL_BATCH_SIZE" --output-dir "$OUTPUT_DIR" "$@")
cd -- "$PROJECT_ROOT"
if [[ "$MODE" == dryrun ]]; then
    "$PYTHON" "${ARGS[@]}" --dry-run 2>&1 | tee "${OUTPUT_DIR}/train_logs/dryrun.log"
else
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
        "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train_logs/train.log"
fi
