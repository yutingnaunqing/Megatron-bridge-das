# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Reference text-pretraining recipe using the standard Megatron Bridge loop."""
import torch

from megatron.bridge.recipes.common import _pretrain_common

from .bridge import Qwen38ModelProvider


def qwen38_flash_next_pretrain_config(text_config, *, seq_length, train_iters, global_batch_size, recompute=False):
    c = text_config
    cfg = _pretrain_common()
    cfg.model = Qwen38ModelProvider(
        text_config=c,
        num_layers=c["num_hidden_layers"], hidden_size=c["hidden_size"],
        num_attention_heads=c["num_attention_heads"], num_query_groups=c["num_key_value_heads"],
        kv_channels=c["head_dim"], ffn_hidden_size=c["moe_intermediate_size"],
        num_moe_experts=c["num_experts"], moe_router_topk=c["num_experts_per_tok"],
        moe_ffn_hidden_size=c["moe_intermediate_size"],
        vocab_size=c["vocab_size"], seq_length=seq_length,
        params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
        bf16=True, normalization="RMSNorm", gated_linear_unit=True,
        add_bias_linear=False, share_embeddings_and_output_weights=False,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        context_parallel_size=1, expert_model_parallel_size=1, expert_tensor_parallel_size=1,
        sequence_parallel=False, attention_dropout=0.0, hidden_dropout=0.0,
        gradient_accumulation_fusion=False, cross_entropy_loss_fusion=False,
        reference_recompute=recompute,
    )
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = c["vocab_size"]
    cfg.tokenizer.use_tokenizer_vocab_size = False
    cfg.dataset.seq_length = seq_length
    cfg.dataset.num_workers = 0
    cfg.dataset.blend = None
    cfg.dataset.skip_getting_attention_mask_from_dataset = True
    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = 1
    cfg.train.exit_interval = 0
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = max(train_iters + 1, 1)
    cfg.optimizer.lr = 1e-4
    cfg.optimizer.min_lr = 1e-5
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_decay_iters = train_iters
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.checkpoint.load = None
    cfg.checkpoint.save = None
    cfg.checkpoint.async_save = False
    cfg.logger.log_interval = 1
    cfg.logger.logging_level = 20
    cfg.logger.log_throughput = False  # Stock FLOPs estimator does not describe this architecture.
    cfg.dist.enable_megatron_core_experimental = True
    return cfg
