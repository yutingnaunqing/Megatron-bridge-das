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

"""Qwen3.8-27B vision-language pretraining recipes for BW1000."""

from __future__ import annotations

import os
from typing import Literal

import torch
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import (
    EnergonDatasetConfig,
    MockVLMSFTDatasetConfig,
    QwenVLEnergonTaskEncoderConfig,
)
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


_QWEN38_27B_BASE = "Qwen/Qwen3.8-27B"


def _qwen38_vl_27b_pretrain_config(
    *, dataset_backend: Literal["energon", "mock"],
) -> ConfigContainer:
    """Build the full multimodal model; only the dataset backend varies."""
    cfg = _pretrain_common()
    hf_path = os.environ.get("LOCAL_HF_PATH") or _QWEN38_27B_BASE
    micro_batch_size = int(os.environ.get("MICRO_BATCH_SIZE", "1"))
    seq_length = int(os.environ.get("SEQ_LENGTH", "1024"))

    # Keep the top-level ConditionalGeneration config, including vision_config.
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    if cfg.model.hf_text_config.model_type != "qwen3_5_text":
        raise ValueError("Qwen3.8-27B VL requires a dense qwen3_5_text configuration")
    cfg.model.seq_length = seq_length
    cfg.model.language_max_sequence_length = seq_length
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.model.pipeline_dtype = torch.bfloat16
    # MTP is independent of vision-language training; use the main LM objective.
    cfg.model.mtp_num_layers = None

    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # BW1000/ROCm policy, matching the existing Qwen VL recipes.
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.use_te_rng_tracker = False
    cfg.model.bias_activation_fusion = False
    cfg.model.bias_dropout_fusion = False
    cfg.model.apply_rope_fusion = False
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = False
    cfg.model.attention_backend = AttnBackend.auto
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.rng.te_rng_tracker = False
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False, overlap_grad_reduce=False, overlap_param_gather=False
    )

    cfg.train.train_iters = int(os.environ.get("TRAIN_ITERS", "50"))
    cfg.train.global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "8"))
    cfg.train.micro_batch_size = micro_batch_size
    cfg.train.manual_gc = False
    cfg.validation.eval_interval = 1000
    cfg.validation.eval_iters = 0

    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=0,
        # Resolve from the final train_iters after runner overrides are applied.
        lr_decay_iters=None,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        max_lr=3.0e-5,
        min_lr=3.0e-6,
        clip_grad=1.0,
        start_weight_decay=0.1,
        end_weight_decay=0.1,
    )
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False
    # CPU offload is opt-in for both full-model and reduced-layer recipes.
    offload_fraction = float(os.environ.get("OPTIMIZER_OFFLOAD_FRACTION", "0"))
    if not 0.0 <= offload_fraction <= 1.0:
        raise ValueError("OPTIMIZER_OFFLOAD_FRACTION must be between 0 and 1")
    cfg.optimizer.optimizer_cpu_offload = offload_fraction > 0
    cfg.optimizer.optimizer_offload_fraction = offload_fraction

    if dataset_backend == "energon":
        cfg.dataset = EnergonDatasetConfig(
            path=os.environ.get("DATA_PATH") or None,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            num_workers=2,
            task_encoder=QwenVLEnergonTaskEncoderConfig(hf_processor_path=hf_path),
            dataloader_type="external",
            do_validation=False,
            do_test=False,
            enable_in_batch_packing=False,
            pad_to_max_length=True,
        )
    elif dataset_backend == "mock":
        cfg.dataset = MockVLMSFTDatasetConfig(
            seq_length=seq_length,
            hf_processor_path=hf_path,
            prompt=os.environ.get("MOCK_PROMPT") or "Describe this image.",
            random_seed=int(os.environ.get("MOCK_RANDOM_SEED", "0")),
            image_size=(
                int(os.environ.get("MOCK_IMAGE_WIDTH", "256")),
                int(os.environ.get("MOCK_IMAGE_HEIGHT", "256")),
            ),
            num_images=int(os.environ.get("MOCK_NUM_IMAGES", "1")),
            num_base_examples=int(os.environ.get("MOCK_NUM_BASE_EXAMPLES", "1000")),
            num_workers=1,
            dataloader_type="single",
            data_sharding=True,
            pin_memory=True,
            persistent_workers=False,
            pad_to_max_length=True,
        )
    else:
        raise ValueError(f"Unsupported Qwen3.8-VL dataset backend: {dataset_backend!r}")

    cfg.tokenizer.tokenizer_model = hf_path
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.checkpoint.load = None
    cfg.checkpoint.save = None
    cfg.checkpoint.save_interval = 1000
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.mixed_precision = bf16_mixed()
    return cfg


def qwen38_vl_27b_pretrain_bw1000_bf16_config() -> ConfigContainer:
    """Return the full Qwen3.8-27B VL recipe with real Energon data."""
    return _qwen38_vl_27b_pretrain_config(dataset_backend="energon")


def qwen38_vl_27b_pretrain_bw1000_bf16_mock_config() -> ConfigContainer:
    """Return the same full model with synthetic images and conversations."""
    return _qwen38_vl_27b_pretrain_config(dataset_backend="mock")


def qwen38_vl_27b_pretrain_8gpu_smoke_bw1000_bf16_config() -> ConfigContainer:
    """Test the multimodal training path on 8 GPUs with four language layers."""
    cfg = _qwen38_vl_27b_pretrain_config(dataset_backend="mock")
    # Keep one complete hybrid block and the full vision encoder/projection.
    cfg.model.num_layers = 4
    cfg.model.hf_text_config.num_hidden_layers = 4
    cfg.model.hf_text_config.layer_types = cfg.model.hf_text_config.layer_types[:4]
    if isinstance(cfg.model.linear_attention_freq, list):
        cfg.model.linear_attention_freq = cfg.model.linear_attention_freq[:4]
    cfg.train.train_iters = int(os.environ.get("TRAIN_ITERS", "5"))
    return cfg


__all__ = [
    "qwen38_vl_27b_pretrain_bw1000_bf16_config",
    "qwen38_vl_27b_pretrain_bw1000_bf16_mock_config",
    "qwen38_vl_27b_pretrain_8gpu_smoke_bw1000_bf16_config",
]
