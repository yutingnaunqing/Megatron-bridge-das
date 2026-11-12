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

"""HCU provider subclasses, selected explicitly by HCU recipes."""

from dataclasses import dataclass, field, fields
from typing import Any

from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider, Qwen35VLMoEModelProvider

from .adapter import HCUVisionModel, attach_vision_optimizations
from .config import VisionOptimizationFields, vision_config


class _VisionProvider:
    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        if model.vision_model is not None:
            attach_vision_optimizations(model.vision_model, self)
        return model

    def build_vision_encoder_spec(self):
        spec = super().build_vision_encoder_spec()
        if self.vision_hcu_enabled:
            spec.params["transformer_config"] = vision_config(spec.params["transformer_config"], self)
            spec.module = HCUVisionModel
        elif (self.vision_rope_backend, self.vision_metadata_backend, self.vision_graph_mode) != (
            "reference",
            "reference",
            "none",
        ):
            raise ValueError("Disabled HCU vision requires reference backends")
        return spec


@dataclass
class HCUQwen35VLModelProvider(_VisionProvider, VisionOptimizationFields, Qwen35VLModelProvider):
    """Dense Qwen provider with local vision optimization fields."""

    # Upstream AutoBridge assigns these conversion values after construction.
    hf_text_config: Any = field(default=None)
    head_dim: int = 256
    audio_token_id: int = 248076
    hf_model_id: str | None = None
    hf_model_revision: str | None = None


@dataclass
class HCUQwen35VLMoEModelProvider(_VisionProvider, VisionOptimizationFields, Qwen35VLMoEModelProvider):
    """MoE Qwen provider; numerical validation currently covers the dense model."""

    # Upstream AutoBridge assigns these conversion values after construction.
    hf_text_config: Any = field(default=None)
    head_dim: int = 256
    audio_token_id: int = 248076
    hf_model_id: str | None = None
    hf_model_revision: str | None = None


def from_upstream(provider):
    """Construct an HCU provider using the converted upstream config values."""
    if type(provider) is Qwen35VLModelProvider:
        cls = HCUQwen35VLModelProvider
    elif type(provider) is Qwen35VLMoEModelProvider:
        cls = HCUQwen35VLMoEModelProvider
    else:
        raise TypeError(f"Unexpected Qwen provider: {type(provider).__qualname__}")
    target_fields = {item.name for item in fields(cls)}
    kwargs = {
        item.name: getattr(provider, item.name) for item in fields(cls) if item.init and hasattr(provider, item.name)
    }
    result = cls(**kwargs)
    # Preserve registered loading hooks and framework runtime state as well.
    # Unknown public conversion values must not silently disappear.
    for name, value in vars(provider).items():
        if name not in target_fields and not name.startswith("_"):
            raise ValueError(f"Undeclared upstream conversion value: {name}")
        if name not in kwargs:
            setattr(result, name, list(value) if isinstance(value, list) else value)
    return result
