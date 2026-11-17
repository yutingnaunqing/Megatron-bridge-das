# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Dense Qwen3.8 language-only full SFT recipes for BW1000."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

from data_preparation.prepare_qwen38_text_sft import chat_template_for_mode
from megatron.bridge import AutoBridge
from megatron.bridge.data.builders.gpt_sft import GPTSFTDatasetConfig
from megatron.bridge.data.sft_processing import ChatSFTPreprocessingConfig
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


def qwen38_text_27b_sft_bw1000_bf16_config() -> ConfigContainer:
    """Build from a converted 4/16/64-layer HF directory, never from VL weights."""
    path = Path(os.environ["LOCAL_HF_PATH"]).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    if config.get("architectures") != ["Qwen3_5ForCausalLM"] or config.get("model_type") != "qwen3_5_text":
        raise ValueError("Convert the VL checkpoint to dense text CausalLM before SFT")
    layers = config["num_hidden_layers"]
    if os.environ.get("EXPECTED_NUM_LAYERS") and layers != int(os.environ["EXPECTED_NUM_LAYERS"]):
        raise ValueError("HF layer count does not match the requested full model")
    if layers not in (4, 16, 64) or config["layer_types"] != (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4):
        raise ValueError("Expected 4/16/64 layers in complete GDN/attention blocks")
    mtp_layers = int(os.environ.get("MTP_NUM_LAYERS", "0"))
    if mtp_layers not in (0, 1) or config.get("mtp_num_hidden_layers", 0) != mtp_layers:
        raise ValueError("MTP_NUM_LAYERS must match the converted checkpoint (0 or 1)")
    if "vision_config" in config:
        raise ValueError("SFT checkpoint must exclude vision")
    mtp_loss_factor = float(os.environ.get("MTP_LOSS_SCALING_FACTOR", "0.1"))
    if mtp_layers and (not math.isfinite(mtp_loss_factor) or mtp_loss_factor <= 0):
        raise ValueError("MTP_LOSS_SCALING_FACTOR must be finite and positive")
    cfg = _sft_common()
    cfg.model = AutoBridge.from_hf_pretrained(str(path)).to_megatron_provider(load_weights=False)
    seq = int(os.environ.get("SEQ_LENGTH", "1024"))
    cfg.model.seq_length = seq
    cfg.model.tensor_model_parallel_size = int(os.environ.get("TP", "2"))
    cfg.model.pipeline_model_parallel_size = int(os.environ.get("PP", "4"))
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.pipeline_model_parallel_layout = None
    first_layers = os.environ.get("NUM_LAYERS_IN_FIRST_PIPELINE_STAGE")
    last_layers = os.environ.get("NUM_LAYERS_IN_LAST_PIPELINE_STAGE")
    cfg.model.num_layers_in_first_pipeline_stage = int(first_layers) if first_layers else None
    cfg.model.num_layers_in_last_pipeline_stage = int(last_layers) if last_layers else None
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.sequence_parallel = cfg.model.tensor_model_parallel_size > 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.mtp_num_layers = mtp_layers or None
    cfg.model.mtp_loss_scaling_factor = mtp_loss_factor
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.use_te_rng_tracker = False
    cfg.model.bias_activation_fusion = True
    cfg.model.bias_dropout_fusion = False
    cfg.model.apply_rope_fusion = os.environ.get("APPLY_ROPE_FUSION", "false").lower() in ("true", "1")
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.cpu_offloading = False
    recompute = os.environ.get("RECOMPUTE", "full" if layers == 64 else "none")
    if recompute not in ("none", "full"):
        raise ValueError("RECOMPUTE must be none or full")
    cfg.model.recompute_granularity = "full" if recompute == "full" else None
    cfg.model.recompute_method = "uniform" if recompute == "full" else None
    cfg.model.recompute_num_layers = 1 if recompute == "full" else None
    cfg.rng.te_rng_tracker = False
    cfg.dist.distributed_timeout_minutes = 60
    tp_overlap = os.environ.get("TP_COMM_OVERLAP", "false").lower() in ("true", "1")
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=tp_overlap, overlap_grad_reduce=False,
                                         overlap_param_gather=False)
    cfg.train.train_iters = int(os.environ.get("TRAIN_ITERS", "50"))
    cfg.train.global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "8"))
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = False
    cfg.train.exit_interval = int(os.environ.get("EXIT_INTERVAL", "0")) or None
    cfg.optimizer.lr = float(os.environ.get("LR", "1e-5"))
    cfg.optimizer.min_lr = 0.0
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    offload_fraction = float(os.environ.get("OPTIMIZER_OFFLOAD_FRACTION", "0"))
    if not 0.0 <= offload_fraction <= 1.0:
        raise ValueError("OPTIMIZER_OFFLOAD_FRACTION must be between zero and one")
    cfg.optimizer.optimizer_cpu_offload = offload_fraction > 0
    cfg.optimizer.optimizer_offload_fraction = offload_fraction
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False
    cfg.scheduler.lr_warmup_iters = int(os.environ.get("WARMUP_ITERS", "1"))
    cfg.scheduler.lr_decay_iters = None
    cfg.scheduler.use_checkpoint_opt_param_scheduler = True
    cfg.scheduler.override_opt_param_scheduler = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    root = Path(os.environ["DATA_PATH"]).expanduser().resolve()
    validate = int(os.environ.get("EVAL_ITERS", "0")) > 0
    cfg.dataset = GPTSFTDatasetConfig(
        dataset_root=str(root), seq_length=seq,
        preprocessing=ChatSFTPreprocessingConfig(loss_mode="assistant"),
        enable_offline_packing=False, do_validation=validate, do_test=False,
        num_workers=int(os.environ.get("NUM_WORKERS", "2")), persistent_workers=False,
        dataset_kwargs={"pad_to_max_length": True},
    )
    cfg.validation.eval_iters = int(os.environ.get("EVAL_ITERS", "0"))
    cfg.validation.eval_interval = int(os.environ.get("EVAL_INTERVAL", "10"))
    cfg.tokenizer.tokenizer_model = str(path)
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.chat_template = chat_template_for_mode(
        (path / "chat_template.jinja").read_text(), os.environ.get("THINKING_MODE", "default"))
    save_dir = Path(os.environ["SAVE_DIR"]).expanduser().resolve()
    resume = os.environ.get("RESUME_FROM") or None
    cfg.checkpoint.pretrained_checkpoint = None if resume else str(path)
    cfg.checkpoint.load = resume
    cfg.checkpoint.finetune = False
    cfg.checkpoint.load_optim = True
    cfg.checkpoint.load_rng = True
    cfg.checkpoint.save = str(save_dir / "checkpoints")
    cfg.checkpoint.save_optim = True
    cfg.checkpoint.save_rng = True
    cfg.checkpoint.save_interval = int(os.environ.get("SAVE_INTERVAL", "10"))
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.logger.tensorboard_dir = str(save_dir / "tb_logs")
    cfg.logger.save_config_filepath = str(save_dir / "resolved_config.yaml")
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = os.environ.get("LOG_THROUGHPUT", "true").lower() == "true"
    cfg.peft = None
    cfg.mixed_precision = bf16_mixed()
    return cfg


__all__ = ["qwen38_text_27b_sft_bw1000_bf16_config"]
