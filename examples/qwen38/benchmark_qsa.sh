#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# 使用示例（在项目根目录执行）：
#   HIP_VISIBLE_DEVICES=0 bash examples/qwen38/benchmark_qsa.sh smoke \
#     --sequence-lengths 128,2048 --warmup 2 --iterations 5 --profile-ops
# 第一个位置参数为 tiny、smoke 或 official；其余参数原样传给 benchmark_qsa.py。
set -euo pipefail
CURRENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${CURRENT_DIR}/../.." && pwd)
PROFILE=${1:-smoke}
if (( $# >= 1 )); then shift; else set --; fi
case "$PROFILE" in tiny|smoke|official) ;; *) echo "Profile must be tiny, smoke or official" >&2; exit 2;; esac
PYTHON=${PYTHON:-python}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/runs/qwen38/qsa_${PROFILE}_$(date +%Y%m%d_%H%M%S)}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/hcu_megatron_bridge/src:${PROJECT_ROOT}/3rdparty/Megatron-Bridge/src:${PROJECT_ROOT}/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM:${PROJECT_ROOT}/3rdparty/Megatron-Energon/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-ERROR}
export HSA_FORCE_FINE_GRAIN_PCIE=${HSA_FORCE_FINE_GRAIN_PCIE:-1}
cd -- "$PROJECT_ROOT"
"$PYTHON" "${CURRENT_DIR}/benchmark_qsa.py" --profile "$PROFILE" --output-dir "$OUTPUT_DIR" "$@"
