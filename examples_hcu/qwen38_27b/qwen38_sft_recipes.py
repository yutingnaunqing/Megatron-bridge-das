# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-27B full-parameter HCU SFT recipes using the dev Bridge builders."""

import json
import math
from pathlib import Path

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import EnergonDatasetConfig, GPTSFTDatasetConfig, QwenVLEnergonTaskEncoderConfig
from megatron.bridge.data.sft_processing import ChatSFTPreprocessingConfig
from megatron.bridge.recipes.common import _sft_common, _sft_common_vlm
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed
from megatron.bridge.training.utils.checkpoint_utils import checkpoint_exists

from prepare_qwen38_text_sft import chat_template_for_mode


def _configure_sft(cfg: ConfigContainer, *, hf_path: Path) -> None:
    model = cfg.model
    model.pipeline_dtype = torch.bfloat16
    model.virtual_pipeline_model_parallel_size = None
    model.transformer_impl = "transformer_engine"
    model.cuda_graph_impl = "none"
    model.use_te_rng_tracker = False
    model.bias_dropout_fusion = False
    model.cross_entropy_loss_fusion = False
    model.cpu_offloading = False
    model.recompute_modules = None

    cfg.rng.te_rng_tracker = False
    cfg.dist.distributed_timeout_minutes = 60
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False, overlap_grad_reduce=False, overlap_param_gather=False
    )
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.optimizer.min_lr = 0.0
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False
    cfg.scheduler.lr_decay_iters = None
    cfg.scheduler.use_checkpoint_opt_param_scheduler = True
    cfg.scheduler.override_opt_param_scheduler = False
    cfg.train.manual_gc = False

    cfg.checkpoint.finetune = False
    cfg.checkpoint.load_optim = True
    cfg.checkpoint.load_rng = True
    cfg.checkpoint.save_optim = True
    cfg.checkpoint.save_rng = True
    cfg.checkpoint.ckpt_format = "torch_dist"

    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = str(hf_path)
    cfg.logger.log_interval = 1
    cfg.peft = None
    cfg.mixed_precision = bf16_mixed()


def qwen38_text_27b_sft_config(
    *, hf_path: Path, data_path: Path, layers: int, mtp_layers: int, thinking_mode: str
) -> ConfigContainer:
    """Train all weights of the converted Qwen3.8 language-only HF model."""
    hf_config = json.loads((hf_path / "config.json").read_text())
    if (
        hf_config.get("architectures") != ["Qwen3_5ForCausalLM"]
        or hf_config.get("model_type") != "qwen3_5_text"
        or "vision_config" in hf_config
    ):
        raise ValueError("Text SFT requires a converted Qwen3.8 CausalLM checkpoint without vision")
    if (
        layers not in (4, 16, 64)
        or hf_config["num_hidden_layers"] != layers
        or hf_config["layer_types"] != ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        * (layers // 4)
    ):
        raise ValueError("Expected 4/16/64 layers in complete Qwen3.8 GDN/attention blocks")
    if mtp_layers not in (0, 1) or hf_config.get("mtp_num_hidden_layers", 0) != mtp_layers:
        raise ValueError("MTP_NUM_LAYERS must match the converted checkpoint (0 or 1)")
    if not (data_path / "training.jsonl").is_file():
        raise ValueError(f"Text SFT requires training.jsonl in {data_path}")

    cfg = _sft_common()
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).get_model_config()
    _configure_sft(cfg, hf_path=hf_path)
    cfg.model.mtp_num_layers = mtp_layers or None
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.tokenizer.chat_template = chat_template_for_mode((hf_path / "chat_template.jinja").read_text(), thinking_mode)
    cfg.dataset = GPTSFTDatasetConfig(
        dataset_root=str(data_path),
        seq_length=4096,
        preprocessing=ChatSFTPreprocessingConfig(loss_mode="assistant"),
        enable_offline_packing=False,
        do_validation=False,
        do_test=False,
        persistent_workers=False,
        dataset_kwargs={"pad_to_max_length": True},
    )
    return cfg


def qwen38_vl_27b_sft_config(*, hf_path: Path, data_path: Path) -> ConfigContainer:
    """Train the full Qwen3.8 vision tower, projector and language decoder."""
    hf_config = json.loads((hf_path / "config.json").read_text())
    text = hf_config.get("text_config", {})
    if (
        hf_config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
        or "vision_config" not in hf_config
        or text.get("model_type") != "qwen3_5_text"
        or text.get("num_hidden_layers") != 64
    ):
        raise ValueError("VL SFT requires the original 64-layer dense Qwen3.8 multimodal checkpoint")
    if not (data_path / ".nv-meta" / "split.yaml").is_file():
        raise ValueError(f"VL SFT requires an Energon dataset with .nv-meta/split.yaml in {data_path}")

    cfg = _sft_common_vlm()
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).get_model_config()
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=0,
        lr_decay_iters=None,
        adam_beta2=0.95,
        weight_decay=0.1,
        start_weight_decay=0.1,
        end_weight_decay=0.1,
    )
    _configure_sft(cfg, hf_path=hf_path)
    cfg.model.mtp_num_layers = None
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.bias_activation_fusion = False
    cfg.model.apply_rope_fusion = False
    cfg.dataset = EnergonDatasetConfig(
        path=str(data_path),
        seq_length=4096,
        micro_batch_size=1,
        task_encoder=QwenVLEnergonTaskEncoderConfig(hf_processor_path=str(hf_path)),
        dataloader_type="external",
        enable_in_batch_packing=False,
        pad_to_max_length=True,
        do_validation=False,
        do_test=False,
    )
    cfg.validation.eval_iters = 0
    return cfg


def finalize_sft_config(
    cfg: ConfigContainer, *, hf_path: Path, output: Path, resume: Path | None
) -> None:
    """Resolve dependent fields after applying the launcher's Bridge overrides."""
    cfg.model.sequence_parallel = cfg.model.tensor_model_parallel_size > 1
    offload = float(cfg.optimizer.optimizer_offload_fraction)
    if not 0 <= offload <= 1:
        raise ValueError("OPTIMIZER_OFFLOAD_FRACTION must be between 0 and 1")
    cfg.optimizer.optimizer_offload_fraction = offload
    cfg.optimizer.optimizer_cpu_offload = offload > 0
    cfg.train.exit_interval = cfg.train.exit_interval or None

    if isinstance(cfg.dataset, GPTSFTDatasetConfig):
        cfg.dataset.do_validation = cfg.validation.eval_iters > 0
        if cfg.dataset.do_validation and not (Path(cfg.dataset.dataset_root) / "validation.jsonl").is_file():
            raise ValueError(f"Text validation requires validation.jsonl in {cfg.dataset.dataset_root}")
        scale = cfg.model.mtp_loss_scaling_factor
        if cfg.model.mtp_num_layers and (not math.isfinite(scale) or scale <= 0):
            raise ValueError("MTP_LOSS_SCALING_FACTOR must be finite and positive")
    else:
        cfg.model.language_max_sequence_length = cfg.dataset.seq_length
        cfg.dataset.micro_batch_size = cfg.train.micro_batch_size

    if resume and not checkpoint_exists(str(resume)):
        raise ValueError(f"RESUME_FROM must point to a Megatron distributed checkpoint: {resume}")
    cfg.checkpoint.load = str(resume) if resume else None
    cfg.checkpoint.pretrained_checkpoint = None if resume else str(hf_path)
    cfg.checkpoint.save = str(output / "checkpoints") if cfg.checkpoint.save_interval else None
    cfg.logger.tensorboard_dir = str(output / "tb_logs")
    cfg.logger.save_config_filepath = str(output / "resolved_config.yaml")
