# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Trainable, short-context Qwen3.8 text reference backend.

Uses real QSA routes, GDN, gated residuals and N-gram features. QSA is evaluated
with dense masked GQA: this is a correctness baseline, NOT a fused sparse kernel.
Only unpacked causal text batches are supported. No inference cache or MTP.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import reference as ref


def text_positions(config, position_ids, dtype):
    """Text-only partial RoPE; all three multimodal coordinates coincide."""
    rope = config.rope_parameters
    if rope["rope_type"] != "default":
        raise ValueError("Only default partial RoPE is implemented")
    dim = int(config.head_dim * rope["partial_rotary_factor"])
    inv = 1.0 / rope["rope_theta"] ** (
        torch.arange(0, dim, 2, device=position_ids.device, dtype=torch.float32) / dim
    )
    angles = position_ids.float().unsqueeze(-1) * inv
    angles = torch.cat((angles, angles), -1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


class TrainableQSA(ref.Qwen4ExpTextAttention):
    """Vectorized reference QSA plus Stage-2 indexer supervision.

TopK ties are broken by ascending block ID. Teacher distributions are detached.
All score/mask tensors remain quadratic; max_sequence_length is enforced by the
model. The auxiliary KL follows the report's block max-pooling and normalization.
"""

    def __init__(self, config, layer_idx, *, tie_break="block_id"):
        super().__init__(config, layer_idx)
        if tie_break not in ("block_id", "upstream"):
            raise ValueError("tie_break must be block_id or upstream")
        self.tie_break = tie_break

    def forward(self, hidden_states, position_embeddings, **kwargs):
        b, s, _ = hidden_states.shape
        ix = self.indexer
        if ix.index_kv_heads != 1 or ix.token_budget % ix.compress_ratio:
            raise ValueError("QSA requires one indexer KV head and a block-aligned budget")
        r = ix.compress_ratio
        blocks = s // r
        qk = ix.index_qk_proj(hidden_states)
        iq, ik = qk.split((ix.index_n_heads * ix.index_head_dim, ix.index_head_dim), -1)
        iq = ix.q_layernorm(iq.reshape(b, s, ix.index_n_heads, ix.index_head_dim))
        cos, sin = position_embeddings
        iq = ref.apply_rotary_pos_emb(iq, cos=cos, sin=sin, unsqueeze_dim=2)
        positions = torch.arange(s, device=hidden_states.device)
        causal = positions[None, :] <= positions[:, None]
        token_blocks = positions // r
        complete_per_query = (positions + 1) // r
        # The tail contains only observed tokens in the query's incomplete block.
        tail = (positions[None, :] >= (complete_per_query * r)[:, None]) & causal
        if blocks:
            ik = ik[:, :blocks * r].reshape(b, blocks, r, ix.index_head_dim)
            ik = ix.k_layernorm(ik.float().mean(2).to(hidden_states.dtype))
            ik = ref.apply_rotary_pos_emb(
                ik.unsqueeze(2), cos=cos[:, :blocks * r:r],
                sin=sin[:, :blocks * r:r], unsqueeze_dim=2,
            ).squeeze(2)
            scores = torch.einsum("bshd,bnd->bshn", iq.float(), ik.float())
            scores = scores.relu().sum(2) / math.sqrt(ix.index_head_dim)
            visible = torch.arange(blocks, device=positions.device)[None, :] < complete_per_query[:, None]
            scores = scores.masked_fill(~visible, -1e30)
            k_blocks = min(ix.block_topk, blocks)
            if self.tie_break == "block_id":
                selected = torch.argsort(scores.detach(), dim=-1, descending=True, stable=True)[..., :k_blocks]
            else:
                # Match upstream's query-local topk, including ties. Vectorized
                # topk over masked future blocks can otherwise choose other ties.
                selected = torch.zeros((b, s, k_blocks), dtype=torch.long, device=scores.device)
                for query in range(s):
                    available = (query + 1) // r
                    take = min(ix.block_topk, available)
                    if take:
                        selected[:, query, :take] = scores[:, query, :available].topk(take, -1).indices
            valid_selection = torch.arange(k_blocks, device=scores.device)[None, None, :] < complete_per_query[None, :, None]
            block_mask = torch.zeros_like(scores, dtype=torch.int).scatter_add(-1, selected, valid_selection.expand(b, -1, -1).int()).bool() & visible
            selected_tokens = block_mask.gather(-1, token_blocks.clamp_max(blocks - 1).expand(b, s, s))
            selected_tokens = selected_tokens & (token_blocks < blocks)
            allowed = (selected_tokens | tail) & causal
        else:
            allowed = tail.expand(b, s, s)

        q, gate = self.q_proj(hidden_states).reshape(b, s, -1, 2 * self.head_dim).chunk(2, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).reshape(b, s, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden_states).reshape(b, s, -1, self.head_dim).transpose(1, 2)
        q, k = ref.apply_rotary_pos_emb(q, k, cos, sin)
        k = ref.repeat_kv(k, self.num_key_value_groups)
        v = ref.repeat_kv(v, self.num_key_value_groups)
        logits = (q.float() @ k.float().transpose(-1, -2)) * self.scaling
        probabilities = logits.masked_fill(~allowed[:, None], float("-inf")).softmax(-1)
        out = (probabilities.to(v.dtype) @ v).transpose(1, 2).reshape(b, s, -1)
        out = self.o_proj(out * gate.reshape(b, s, -1).sigmoid())

        if blocks:
            # Sum over teacher heads then normalize. Scaling by head count cancels
            # after block max-pooling and re-normalization on the selected blocks.
            teacher = probabilities.detach().sum(1)[..., :blocks * r]
            teacher = teacher.reshape(b, s, blocks, r).amax(-1) * block_mask
            teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-20)
            log_student = scores.masked_fill(~block_mask, -1e30).log_softmax(-1)
            kl = (teacher * (teacher.clamp_min(1e-20).log() - log_student)).sum(-1)
        else:
            kl = qk.sum(-1) * 0
        return out, kl


class ReferenceMoE(ref.Qwen4ExpTextSparseMoeBlock):
    def forward(self, hidden):
        b, s, d = hidden.shape
        flat = hidden.reshape(-1, d)
        logits, weights, indices = self.gate(flat)
        out = self.experts(flat, indices, weights)
        out = out + self.shared_expert_gate(flat).sigmoid() * self.shared_expert(flat)
        # Differentiable Switch-style load balancing; selection counts are fixed.
        counts = F.one_hot(indices, self.gate.num_experts).float().mean((0, 1))
        aux = self.gate.num_experts * (counts * logits.float().softmax(-1).mean(0)).sum()
        # Keep unused experts attached to the graph for MCore DDP's gradient hooks.
        out = out + (self.experts.gate_up_proj.reshape(-1)[0] + self.experts.down_proj.reshape(-1)[0]) * 0
        return out.reshape(b, s, d), aux


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.attention = (
            ref.Qwen4ExpTextGatedDeltaNet(config, layer_idx)
            if self.layer_type == "linear_attention" else TrainableQSA(config, layer_idx)
        )
        self.mlp = ReferenceMoE(config)
        self.attn_hyper_connection = ref.Qwen4ExpTextGatedResidual(config)
        self.mlp_hyper_connection = ref.Qwen4ExpTextGatedResidual(config)
        self.ple = None
        if layer_idx + 1 in config.ple_layer_ids:
            self.ple = ref.Qwen4ExpTextPLELayer(config, layer_idx, config.ple_layer_ids.index(layer_idx + 1))

    def forward(self, hidden, input_ids, cos, sin):
        if self.ple is not None:
            hidden = hidden + self.ple(hidden, input_ids, None)
        x, residual, gates = self.attn_hyper_connection(hidden)
        if self.layer_type == "linear_attention":
            x = self.attention(x)
            kl = x.sum(-1) * 0
        else:
            x, kl = self.attention(x, (cos, sin))
        hidden = residual + (x.unsqueeze(-2) * gates.unsqueeze(-1)).flatten(-2)
        x, residual, gates = self.mlp_hyper_connection(hidden)
        x, router_loss = self.mlp(x)
        hidden = residual + (x.unsqueeze(-2) * gates.unsqueeze(-1)).flatten(-2)
        return hidden, kl, router_loss


class TextBackbone(nn.Module):
    def __init__(self, config: dict, *, recompute: bool = False, max_sequence_length: int = 4096):
        super().__init__()
        self.text_config = SimpleNamespace(**config)
        c = self.text_config
        self.recompute = recompute
        self.max_sequence_length = max_sequence_length
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(c, i) for i in range(c.num_hidden_layers)])
        self.hyper_connection_mixer = ref.Qwen4ExpTextGatedResidual(c, use_combine=False)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.apply(self._initialize)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _initialize(self, module):
        std = self.text_config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, std=std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, ref.Qwen4ExpTextExperts):
            nn.init.normal_(module.gate_up_proj, std=std)
            nn.init.normal_(module.down_proj, std=std)
        elif isinstance(module, ref.Qwen4ExpTextTopKRouter):
            nn.init.normal_(module.weight, std=std)
        elif isinstance(module, ref.Qwen4ExpTextPLELayer):
            nn.init.zeros_(module.conv1d.weight)

    def forward(self, input_ids, position_ids=None, labels=None):
        b, s = input_ids.shape
        if s > self.max_sequence_length:
            raise ValueError("Sequence exceeds the configured dense-reference QSA limit")
        if position_ids is None:
            position_ids = torch.arange(s, device=input_ids.device).expand(b, -1)
        hidden = self.embed_tokens(input_ids)
        cos, sin = text_positions(self.text_config, position_ids, hidden.dtype)
        hidden = hidden.repeat(1, 1, self.text_config.hc_count)
        qsa_loss = hidden.new_zeros((b, s), dtype=torch.float32)
        router_loss = hidden.new_zeros((), dtype=torch.float32)
        for layer in self.layers:
            if self.recompute and self.training:
                hidden, kl, router = checkpoint(layer, hidden, input_ids, cos, sin, use_reentrant=False)
            else:
                hidden, kl, router = layer(hidden, input_ids, cos, sin)
            qsa_loss = qsa_loss + kl
            router_loss = router_loss + router
        hidden = self.hyper_connection_mixer(hidden)
        if labels is None:
            return self.lm_head(hidden)
        # Chunk CE to avoid allocating a full [batch, sequence, vocabulary] FP32 tensor.
        losses = []
        for start in range(0, s, 32):
            logits = self.lm_head(hidden[:, start:start + 32])
            losses.append(F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels[:, start:start + 32].reshape(-1), reduction="none",
            ).reshape(b, -1))
        lm_loss = torch.cat(losses, -1)
        return torch.stack((lm_loss, qsa_loss, router_loss.expand(b, s)))
