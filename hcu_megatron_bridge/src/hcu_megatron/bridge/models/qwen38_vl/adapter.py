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

"""Instance-scoped adapters: no process-wide class patches or new parameters."""

import logging
import weakref
from types import MethodType

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel

from . import attention_methods, block_methods, vision_methods
from .compatibility import check_upstream_compatibility
from .config import HCUVisionConfig, VisionOptimizationFields, vision_config
from .vision_graph import FrozenPatchGraph
from .vision_metadata import VisionMetadataCache
from .vision_rope import validate_vision_rope_backend


logger = logging.getLogger(__name__)
_MODELS = weakref.WeakSet()


def _apply(self, fn, recurse=True):
    self.metadata_cache.clear()
    if self.patch_graph is not None:
        self.patch_graph.clear()
    return Qwen3VLVisionModel._apply(self, fn, recurse=recurse)


def attach_vision_optimizations(model: Qwen3VLVisionModel, options: VisionOptimizationFields) -> None:
    """Attach local methods once, preserving module names and parameter objects."""
    if getattr(model, "_hcu_vision_attached", False):
        raise RuntimeError("HCU vision adapter is already attached")
    if not options.vision_hcu_enabled:
        if (options.vision_rope_backend, options.vision_metadata_backend, options.vision_graph_mode) != (
            "reference",
            "reference",
            "none",
        ):
            raise ValueError("Disabled HCU vision requires reference RoPE/metadata and Graph none")
        return
    check_upstream_compatibility()
    old_config = model.config
    config = old_config if isinstance(old_config, HCUVisionConfig) else vision_config(old_config, options)
    validate_vision_rope_backend(config.vision_rope_backend)
    if config.vision_metadata_backend not in ("reference", "prepared"):
        raise ValueError("Unsupported vision metadata backend")
    if config.vision_graph_mode not in ("none", "patch_embed"):
        raise ValueError("Unsupported patch Graph mode")
    if config.cuda_graph_impl != "none":
        raise ValueError("HCU vision adapter currently requires upstream vision Graph none")
    if config.rotary_interleaved or not config.apply_rotary_pos_emb_in_fp32:
        raise ValueError("HCU vision adapter requires FP32 half-split rotary")
    bindings = []
    configs = []

    def bind(module, name, function):
        bindings.append((module, name, name in module.__dict__, module.__dict__.get(name)))
        setattr(module, name, MethodType(function, module))

    metadata_cache = VisionMetadataCache(
        max_entries=config.vision_metadata_cache_entries, max_bytes=config.vision_metadata_cache_bytes
    )
    patch_graph = None
    if config.vision_graph_mode == "patch_embed":
        patch_graph = FrozenPatchGraph(
            model.patch_embed,
            token_counts=tuple(config.vision_graph_patch_tokens),
            max_entries=config.vision_graph_max_entries,
            max_bytes=config.vision_graph_max_bytes,
            capture_calls=config.vision_graph_capture_calls,
            warmup_calls=config.vision_graph_warmup_calls,
            capture_error_mode=config.vision_graph_capture_error_mode,
        )
    for module in model.modules():
        if getattr(module, "config", None) is old_config:
            configs.append((module, old_config))
            module.config = config
    model.metadata_cache = metadata_cache
    model.patch_graph = patch_graph
    bind(model, "forward", vision_methods.forward)
    bind(model, "_apply", _apply)
    bind(model, "clear_vision_patch_graph", vision_methods.clear_vision_patch_graph)
    bind(model.decoder, "forward", block_methods.forward)
    bind(model.decoder, "_checkpointed_forward", block_methods._checkpointed_forward)
    for layer in model.decoder.layers:
        bind(layer.self_attention, "forward", attention_methods.forward)
    model._hcu_original_bindings = (bindings, configs)
    model._hcu_vision_attached = True
    _MODELS.add(model)
    logger.info(
        "HCU vision adapter attached: layers=%d rope=%s metadata=%s graph=%s",
        len(model.decoder.layers),
        config.vision_rope_backend,
        config.vision_metadata_backend,
        config.vision_graph_mode,
    )


def cleanup_vision_graphs() -> None:
    """Release runtime graph storage from the HCU runner's finally block."""
    for model in list(_MODELS):
        stats = model.clear_vision_patch_graph()
        if stats is not None:
            logger.info("HCU vision patch Graph final stats: %s", stats)


class HCUVisionModel(Qwen3VLVisionModel):
    """Spec-compatible vision model for standalone vision construction."""

    def __init__(self, transformer_config, *args, **kwargs):
        super().__init__(transformer_config, *args, **kwargs)
        attach_vision_optimizations(self, transformer_config)


def detach_vision_optimizations(model: Qwen3VLVisionModel) -> None:
    """Restore the exact original instance bindings for paired validation."""
    if not getattr(model, "_hcu_vision_attached", False):
        return
    if model.patch_graph is not None:
        model.patch_graph.clear()
    model.metadata_cache.clear()
    bindings, configs = model._hcu_original_bindings
    for module, name, present, value in reversed(bindings):
        if present:
            setattr(module, name, value)
        else:
            delattr(module, name)
    for module, config in configs:
        module.config = config
    for name in ("metadata_cache", "patch_graph", "_hcu_original_bindings", "_hcu_vision_attached"):
        delattr(model, name)
    _MODELS.discard(model)
