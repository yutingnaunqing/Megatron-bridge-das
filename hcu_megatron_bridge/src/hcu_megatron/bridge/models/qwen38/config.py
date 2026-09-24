# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Explicit architecture reductions; never silently reinterpret a checkpoint."""
from __future__ import annotations

import copy
import json
from pathlib import Path


def load_text_config(path, *, profile="smoke", num_layers=4, num_experts=None, ngram_vocab_size=None):
    raw = json.loads(Path(path).read_text())
    if raw.get("model_type") != "qwen4_exp":
        raise ValueError("Expected an official qwen4_exp config with text_config")
    c = copy.deepcopy(raw["text_config"])
    # Defaults from the upstream Qwen4ExpTextConfig (not present in model JSON).
    c.setdefault("norm_topk_prob", True)
    c.setdefault("seed", 1234)
    c["tie_word_embeddings"] = raw.get("tie_word_embeddings", False)
    c["_attn_implementation"] = "eager"
    if profile == "tiny":
        c.update(
            hidden_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=32, linear_value_head_dim=32,
            hc_lowrank=32, vocab_size=512, eos_token_id=511, bos_token_id=511,
            moe_intermediate_size=64, shared_expert_intermediate_size=64,
            num_experts=4, num_experts_per_tok=2, indexer_head_dim=32,
            indexer_budget=16, ple_embed_dim=128, ngram_vocab_size_base=257,
        )
    elif profile == "smoke":
        # Preserve official width, heads, expert FFN size, QSA budget and vocabulary.
        c.update(num_experts=16, ngram_vocab_size_base=20000)
    elif profile != "official":
        raise ValueError(f"Unknown profile: {profile}")
    if not 4 <= num_layers <= raw["text_config"]["num_hidden_layers"]:
        raise ValueError("Use 4..48 layers so at least one GDN/GDN/GDN/QSA cycle is retained")
    c["num_hidden_layers"] = num_layers
    c["layer_types"] = c["layer_types"][:num_layers]
    c["ple_layer_ids"] = [i for i in c["ple_layer_ids"] if i <= num_layers]
    if num_experts is not None:
        c["num_experts"] = num_experts
    if ngram_vocab_size is not None:
        c["ngram_vocab_size_base"] = ngram_vocab_size
    if c["num_experts"] < c["num_experts_per_tok"] or c["ngram_vocab_size_base"] <= 0:
        raise ValueError("Expert count must cover router TopK and N-gram table size must be positive")
    c["use_cache"] = False
    c["mtp_num_hidden_layers"] = 0
    c["mtp"] = None
    return c
