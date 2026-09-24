# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Correctness checks against the vendored upstream implementation."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hcu_megatron.bridge.models.qwen38 import reference as ref
from hcu_megatron.bridge.models.qwen38.config import load_text_config
from hcu_megatron.bridge.models.qwen38.model import TextBackbone, TrainableQSA, text_positions

CONFIG = Path(__file__).resolve().parents[2] / "examples/qwen38/config.official.json"


def tiny():
    return load_text_config(CONFIG, profile="tiny")


@pytest.mark.parametrize("length", [1, 3, 4, 7, 32])
def test_qsa_matches_upstream_outputs_and_attention_gradients(length):
    torch.manual_seed(193)
    c = SimpleNamespace(**tiny())
    oracle = ref.Qwen4ExpTextAttention(c, 3)
    actual = TrainableQSA(c, 3, tie_break="upstream")
    actual.load_state_dict(oracle.state_dict())
    x = torch.randn(2, length, c.hidden_size, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    positions = text_positions(c, torch.arange(length).expand(2, -1), x.dtype)
    mask = torch.zeros(2, 1, length, length)
    mask.masked_fill_(torch.ones(length, length, dtype=torch.bool).triu(1), float("-inf"))
    expected, _ = oracle(x, positions, mask)
    result, kl = actual(y, positions)
    torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
    expected.square().sum().backward()
    result.square().sum().backward()
    torch.testing.assert_close(y.grad, x.grad, rtol=5e-5, atol=3e-6)
    assert torch.isfinite(kl).all()
    assert kl.min() >= -1e-6


def test_qsa_indexer_is_trainable_and_deterministic_on_ties():
    torch.manual_seed(31)
    c = SimpleNamespace(**tiny())
    attention = TrainableQSA(c, 3)
    x = torch.randn(1, 32, c.hidden_size, requires_grad=True)
    positions = text_positions(c, torch.arange(32).unsqueeze(0), x.dtype)
    out, kl = attention(x, positions)
    kl.mean().backward()
    grad = attention.indexer.index_qk_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 0
    with torch.no_grad():
        attention.indexer.index_qk_proj.weight.zero_()
        first, _ = attention(x, positions)
        second, _ = attention(x, positions)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    # Independent tie oracle: select the earliest budget tokens and the current
    # incomplete block. Once four blocks are visible, later full blocks are skipped.
    class FirstBlocks(torch.nn.Module):
        def forward(self, hidden, pos, mask, cache):
            p = torch.arange(hidden.shape[1])
            complete = (p + 1) // c.indexer_compress_ratio
            allowed = (p[None] < c.indexer_budget) | (p[None] >= (complete * c.indexer_compress_ratio)[:, None])
            allowed &= p[None] <= p[:, None]
            return torch.where(allowed, 0.0, float("-inf"))[None, None]
    oracle = ref.Qwen4ExpTextAttention(c, 3)
    oracle.load_state_dict(attention.state_dict())
    oracle.indexer = FirstBlocks()
    mask = torch.zeros(1, 1, 32, 32).masked_fill(torch.ones(32, 32, dtype=torch.bool).triu(1), float("-inf"))
    expected, _ = oracle(x, positions, mask)
    torch.testing.assert_close(first, expected, atol=2e-6, rtol=2e-5)


def test_stage2_kl_matches_independent_block_teacher():
    torch.manual_seed(152)
    c = SimpleNamespace(**tiny())
    actual = TrainableQSA(c, 3)
    oracle = ref.Qwen4ExpTextAttention(c, 3)
    oracle.load_state_dict(actual.state_dict())
    x = torch.randn(1, 8, c.hidden_size)
    positions = text_positions(c, torch.arange(8).unsqueeze(0), x.dtype)
    mask = torch.zeros(1, 1, 8, 8).masked_fill(torch.ones(8, 8, dtype=torch.bool).triu(1), float("-inf"))
    _, weights = oracle(x, positions, mask)
    _, kl = actual(x, positions)
    teacher = weights[0, :, -1].detach().sum(0).reshape(2, 4).max(-1).values
    teacher /= teacher.sum()
    ix = actual.indexer
    projection = ix.index_qk_proj(x)
    raw_q = projection[:, -1:, :c.indexer_n_heads * c.indexer_head_dim].reshape(1, 1, 4, 32)
    raw_k = projection[:, :, c.indexer_n_heads * c.indexer_head_dim:].reshape(1, 2, 4, 32).mean(2)
    iq = ref.apply_rotary_pos_emb(ix.q_layernorm(raw_q), cos=positions[0][:, -1:], sin=positions[1][:, -1:], unsqueeze_dim=2)
    ik = ref.apply_rotary_pos_emb(ix.k_layernorm(raw_k).unsqueeze(2), cos=positions[0][:, ::4], sin=positions[1][:, ::4], unsqueeze_dim=2)
    scores = torch.stack([(iq[0, 0] * ik[0, j, 0]).sum(-1).relu().sum() for j in range(2)]) / (32 ** 0.5)
    expected = torch.nn.functional.kl_div(scores.log_softmax(0), teacher, reduction="sum")
    torch.testing.assert_close(kl[0, -1], expected)


def test_gdn_chunk_matches_recurrence():
    torch.manual_seed(93)
    q, k, v = [torch.randn(1, 11, 2, 8) for _ in range(3)]
    g = -torch.rand(1, 11, 2)
    beta = torch.rand(1, 11, 2)
    a, _ = ref.torch_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    b, _ = ref.torch_recurrent_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_ngram_eos_resets_context():
    c = SimpleNamespace(**tiny())
    table = ref.Qwen4ExpTextNGramEmbedding(c, c.ple_embed_dim, 1)
    tokens = torch.tensor([[3, 7, c.eos_token_id, 12, 15]])
    shifted = table._shift_right_ignore_eos(tokens, 1)
    assert shifted.tolist() == [[c.eos_token_id, 3, 7, c.eos_token_id, 12]]


def test_all_structures_have_gradients_and_recompute_agrees():
    torch.manual_seed(932)
    model = TextBackbone(tiny())
    recomputed = copy.deepcopy(model)
    recomputed.recompute = True
    tokens = torch.randint(0, 511, (1, 24))
    targets = torch.randint(0, 511, (1, 24))
    actual = model(tokens, labels=targets)
    expected = recomputed(tokens, labels=targets)
    torch.testing.assert_close(actual, expected)
    actual.mean().backward()
    expected.mean().backward()
    for (name, p), (_, q) in zip(model.named_parameters(), recomputed.named_parameters()):
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=1e-4)
    for fragment in ("index_qk_proj", "ngram_embedding.weight", "in_proj_qkv", "input_mix_weight_down", "gate.weight"):
        matches = [p.grad.abs().max() for n, p in model.named_parameters() if fragment in n]
        assert matches and max(matches) > 0, fragment


def test_invalid_reductions_fail():
    with pytest.raises(ValueError):
        load_text_config(CONFIG, num_layers=1)
    with pytest.raises(ValueError):
        load_text_config(CONFIG, num_experts=1)
