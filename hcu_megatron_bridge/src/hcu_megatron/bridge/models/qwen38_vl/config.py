# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Typed HCU configuration without changing upstream dataclasses."""

from dataclasses import dataclass, field, fields

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig


@dataclass
class VisionOptimizationFields:
    """Options shared by the HCU provider and its vision configuration."""

    vision_hcu_enabled: bool = True
    vision_rope_backend: str = "reference"
    vision_metadata_backend: str = "reference"
    vision_metadata_cache_entries: int = 16
    vision_metadata_cache_bytes: int = 64 * 1024 * 1024
    vision_graph_mode: str = "none"
    vision_graph_patch_tokens: list[int] = field(default_factory=list)
    vision_graph_max_entries: int = 1
    vision_graph_max_bytes: int = 0
    vision_graph_capture_calls: int = 0
    vision_graph_warmup_calls: int = 3
    vision_graph_capture_error_mode: str = "thread_local"


@dataclass
class HCUVisionConfig(VisionOptimizationFields, Qwen3VLTransformerConfig):
    """Original vision config plus explicitly declared HCU options."""


def vision_config(config: Qwen3VLTransformerConfig, options: VisionOptimizationFields) -> HCUVisionConfig:
    """Shallow-copy configuration values without deepcopying process groups."""
    result = HCUVisionConfig(**{f.name: getattr(config, f.name) for f in fields(config) if f.init})
    for item in fields(VisionOptimizationFields):
        setattr(result, item.name, getattr(options, item.name))
    return result
