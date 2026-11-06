# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.8-VL recipe exports."""

from hcu_megatron.bridge.recipes.qwen38_vl.bw1000.qwen38_vl import (
    qwen38_vl_27b_pretrain_8gpu_bw1000_bf16_config,
    qwen38_vl_27b_pretrain_8gpu_smoke_bw1000_bf16_config,
    qwen38_vl_27b_pretrain_bw1000_bf16_config,
    qwen38_vl_27b_pretrain_bw1000_bf16_mock_config,
)

__all__ = [
    "qwen38_vl_27b_pretrain_8gpu_bw1000_bf16_config",
    "qwen38_vl_27b_pretrain_8gpu_smoke_bw1000_bf16_config",
    "qwen38_vl_27b_pretrain_bw1000_bf16_config",
    "qwen38_vl_27b_pretrain_bw1000_bf16_mock_config",
]
