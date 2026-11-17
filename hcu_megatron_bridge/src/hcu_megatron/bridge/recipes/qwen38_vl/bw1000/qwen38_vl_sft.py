# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Full pretrained Qwen3.8 VL SFT, independent of random-init performance recipes."""

import os
from pathlib import Path

from hcu_megatron.bridge.recipes.qwen38_vl.bw1000.qwen38_vl import _qwen38_vl_27b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


def qwen38_vl_27b_sft_bw1000_bf16_config() -> ConfigContainer:
    """Load all language/vision/projection weights and supervise assistant replies."""
    cfg = _qwen38_vl_27b_pretrain_config(dataset_backend="energon")
    if cfg.model.num_layers != 64 or cfg.model.hf_text_config.num_hidden_layers != 64:
        raise ValueError("Full VL SFT requires the original 64-layer multimodal checkpoint")
    cfg.model.tensor_model_parallel_size = int(os.environ.get("TP", "4"))
    cfg.model.pipeline_model_parallel_size = int(os.environ.get("PP", "2"))
    cfg.model.context_parallel_size = int(os.environ.get("CP", "1"))
    cfg.model.sequence_parallel = cfg.model.tensor_model_parallel_size > 1
    recompute = os.environ.get("RECOMPUTE", "none")
    cfg.model.recompute_granularity = "full" if recompute == "full" else None
    cfg.model.recompute_method = "uniform" if recompute == "full" else None
    cfg.model.recompute_num_layers = 1 if recompute == "full" else None
    cfg.model.cpu_offloading = False
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False, overlap_grad_reduce=False, overlap_param_gather=False
    )
    cfg.optimizer.lr = float(os.environ.get("LR", "1e-5"))
    cfg.optimizer.min_lr = 0.0
    cfg.scheduler.lr_warmup_iters = int(os.environ.get("WARMUP_ITERS", "20"))
    cfg.scheduler.lr_decay_iters = None
    cfg.scheduler.use_checkpoint_opt_param_scheduler = True
    cfg.scheduler.override_opt_param_scheduler = False
    cfg.dataset.num_workers = int(os.environ.get("NUM_WORKERS", "2"))
    cfg.dataset.task_encoder.min_pixels = int(os.environ.get("MIN_PIXELS", "200704"))
    cfg.dataset.task_encoder.max_pixels = int(os.environ.get("MAX_PIXELS", "1003520"))
    # Birds has only a train split. Do not report its training samples as held-out validation.
    cfg.validation.eval_iters = 0
    cfg.dataset.do_validation = False
    output = Path(os.environ["SAVE_DIR"]).resolve()
    resume = os.environ.get("RESUME_FROM") or None
    cfg.checkpoint.pretrained_checkpoint = None if resume else os.environ["LOCAL_HF_PATH"]
    cfg.checkpoint.load = resume
    cfg.checkpoint.finetune = False
    cfg.checkpoint.load_optim = True
    cfg.checkpoint.load_rng = True
    cfg.checkpoint.save = str(output / "checkpoints")
    cfg.checkpoint.save_interval = int(os.environ.get("SAVE_INTERVAL", "250"))
    cfg.checkpoint.save_optim = True
    cfg.checkpoint.save_rng = True
    cfg.logger.log_throughput = os.environ.get("LOG_THROUGHPUT", "true").lower() == "true"
    cfg.logger.tensorboard_dir = str(output / "tb_logs")
    cfg.logger.save_config_filepath = str(output / "resolved_config.yaml")
    cfg.peft = None
    return cfg


__all__ = ["qwen38_vl_27b_sft_bw1000_bf16_config"]
