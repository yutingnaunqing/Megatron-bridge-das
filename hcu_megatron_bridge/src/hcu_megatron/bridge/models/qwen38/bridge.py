# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Megatron-Bridge integration for the Qwen3.8 reference text backbone.

This provider intentionally implements DP only. New residual-stream semantics
require custom model construction rather than the stock GPT decoder. It does
not register a fake Qwen3.5 conversion mapping or claim HF checkpoint conversion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.gpt_step import _forward_step_common

from .model import TextBackbone


class Qwen38MegatronModel(MegatronModule):
    def __init__(self, config):
        super().__init__(config)
        self.pre_process = self.post_process = True
        self.share_embeddings_and_output_weights = False
        self.vp_stage = None
        self.pg_collection = config._pg_collection
        self.tp_group = config._pg_collection.tp
        self.backbone = TextBackbone(
            config.text_config, recompute=config.reference_recompute,
            max_sequence_length=config.reference_max_sequence_length,
        )
        # Preserve GDN time constants when the Bridge applies its BF16 wrapper.
        for layer in self.backbone.layers:
            if layer.layer_type == "linear_attention":
                layer.attention._keep_in_float32_parameter_names = ("A_log", "dt_bias")

    def set_input_tensor(self, input_tensor):
        if input_tensor is not None and any(t is not None for t in input_tensor):
            raise ValueError("Qwen3.8 reference backend does not implement pipeline parallelism")

    def forward(self, input_ids, position_ids=None, attention_mask=None, labels=None, packed_seq_params=None, **kwargs):
        if packed_seq_params is not None or attention_mask is not None:
            raise ValueError("Reference training requires unpacked causal batches with no custom mask")
        return self.backbone(input_ids, position_ids, labels)


@dataclass
class Qwen38ModelProvider(GPTModelProvider):
    text_config: dict = field(default_factory=dict)
    reference_recompute: bool = False
    reference_max_sequence_length: int = 4096
    qsa_loss_coefficient: float = 1.0

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        for name in (
            "tensor_model_parallel_size", "pipeline_model_parallel_size",
            "context_parallel_size", "expert_model_parallel_size", "expert_tensor_parallel_size",
        ):
            if getattr(self, name) != 1:
                raise ValueError(f"Qwen3.8 reference backend supports DP only: {name} must be 1")
        if self.sequence_parallel or self.mtp_num_layers or self.fp8 or self.fp4:
            raise ValueError("SP, MTP and quantized training are not implemented in the reference backend")
        if self.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("Virtual pipeline parallelism is not implemented")
        if self.recompute_granularity is not None or self.cuda_graph_impl != "none":
            raise ValueError("Use reference_recompute for this backend; CUDA Graphs are not implemented")
        expected = {
            "num_layers": self.text_config["num_hidden_layers"],
            "hidden_size": self.text_config["hidden_size"],
            "vocab_size": self.text_config["vocab_size"],
            "num_moe_experts": self.text_config["num_experts"],
            "num_attention_heads": self.text_config["num_attention_heads"],
        }
        for key, value in expected.items():
            if getattr(self, key) != value:
                raise ValueError(f"{key} disagrees with text_config; use the launcher's architecture options")
        return Qwen38MegatronModel(self)


def loss_func(loss_mask, qsa_coefficient, router_coefficient, output):
    mask = loss_mask.float()
    sums = (output.float() * mask).sum((-1, -2))
    total = sums[0] + qsa_coefficient * sums[1] + router_coefficient * sums[2]
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite Qwen3.8 training loss")
    count = mask.sum().detach().to(torch.int)
    metrics = {
        name: torch.stack((value.detach(), count))
        for name, value in zip(("lm loss", "qsa kl loss", "router aux loss"), sums)
    }
    return total, count, metrics


def forward_step(state, data_iterator, model, return_schedule_plan=False):
    output, loss_mask = _forward_step_common(state, data_iterator, model, return_schedule_plan)
    return output, partial(
        loss_func, loss_mask, state.cfg.model.qsa_loss_coefficient,
        state.cfg.model.text_config["router_aux_loss_coef"],
    )
