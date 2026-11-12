# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Pinned Qwen vision method overrides, bound only to HCU model instances."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
from hcu_megatron.bridge.models.qwen38_vl.vision_rope import (
    prepare_vision_rotary_cos_sin,
)
from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from torch.nn import functional as F


def _maybe_pad_vision_sequence_for_cuda_graph(
    hidden_states: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    seq_len: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad vision token tensors to ``max_seq_len`` for fixed-shape CUDA graphs.

    Args:
        hidden_states: ``[seq_len, hidden_size]``.
        rotary_pos_emb: ``[seq_len, 1, 1, dim]`` (same layout as after ``reshape``/``repeat`` in :meth:`Qwen3VLVisionModel.forward`).
        seq_len: Current sequence length (must match tensor leading size).
        max_seq_len: Target length for CUDA graph capture.

    Returns:
        Tuple of (padded hidden_states, padded rotary_pos_emb, new seq_len).

    Raises:
        ValueError: If ``seq_len`` exceeds ``max_seq_len``.
    """
    if seq_len > max_seq_len:
        raise ValueError(
            f"Vision input sequence length ({seq_len}) exceeds max_vision_cuda_graph_seq_length ({max_seq_len}). "
            f"Increase max_vision_cuda_graph_seq_length in config or disable vision CUDA graphs."
        )
    if seq_len < max_seq_len:
        pad_len = max_seq_len - seq_len
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len), value=0.0)
        rotary_pos_emb = F.pad(rotary_pos_emb, (0, 0, 0, 0, 0, 0, 0, pad_len), value=0.0)
        seq_len = max_seq_len
    return hidden_states, rotary_pos_emb, seq_len


def _vision_forward_packed_attention_setup(
    use_cuda_graph_padding: bool,
    hidden_states: torch.Tensor,
    original_seq_len: int,
    seq_len: int,
    grid_thw: torch.Tensor,
    build_packed_seq_params: Callable[[torch.Tensor], PackedSeqParams],
) -> tuple[Optional[PackedSeqParams], Optional[torch.Tensor]]:
    """Return ``(packed_seq_params, attention_mask)`` for vision encoder forward.

    When using CUDA graphs, packed sequence metadata (non-tensors) cannot be passed; use full
    attention on a fixed-length padded sequence and optionally an additive mask to ignore padding.

    Args:
        use_cuda_graph_padding: Whether vision CUDA graph padding path is active.
        hidden_states: Vision hidden states after adding the batch dimension, shape ``[S, 1, H]``.
        original_seq_len: Sequence length before padding.
        seq_len: Sequence length after optional padding (equals ``hidden_states`` leading size).
        grid_thw: Grid sizes per image/frame (used only when not using CUDA graph padding).
        build_packed_seq_params: Callback to build :class:`PackedSeqParams` from ``grid_thw``.

    Returns:
        ``packed_seq_params`` (``None`` when using CUDA graph padding) and ``attention_mask``
        (additive mask for padded CUDA graph runs, else ``None``).
    """
    if use_cuda_graph_padding:
        packed_seq_params = None
        if original_seq_len < seq_len:
            attention_mask = torch.ones(
                (1, 1, seq_len, seq_len), dtype=hidden_states.dtype, device=hidden_states.device
            )
            attention_mask[:, :, :, original_seq_len:] = 0
            attention_mask[:, :, original_seq_len:, :] = 0
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        else:
            attention_mask = None
        return packed_seq_params, attention_mask

    packed_seq_params = build_packed_seq_params(grid_thw)
    return packed_seq_params, None


def forward(
    self,
    hidden_states: Optional[torch.Tensor],
    grid_thw: torch.Tensor,
    inference_params: Optional[InferenceParams] = None,
    extra_block_kwargs: dict = None,
) -> torch.Tensor:
    """Forward function of the Qwen3 Vision Model. This function passes the input tensors
    through the embedding layer and then the transformer.

    Args:
        x (torch.Tensor): input image/video data of shape [n_tokens, n_dims]
        grid_thw (torch.Tensor): the size tensor indicates grid size of each image/frame
        packed_seq_params (PackedSeqParams): parameters to build attention mask in the backend

    Returns:
        x (torch.Tensor): output after final transformer block of shape [b, s, h].
    """
    assert grid_thw is not None
    assert self.input_tensor is None
    assert inference_params is None

    patch_embed = self.patch_embed
    if getattr(self, "patch_graph", None) is not None and not any(p.requires_grad for p in self.parameters()):
        patch_embed = self.patch_graph
    metadata = None
    if getattr(self.config, "vision_metadata_backend", "reference") == "prepared":
        # Read the whole grid before submitting patch convolution/GEMM,
        # avoiding repeated GPU scalar reads between patch and decoder.
        metadata = self.metadata_cache.get(
            grid_thw,
            merge_size=self.spatial_merge_size,
            grid_side=self.num_grid_per_side,
            device=self.pos_embed.weight.device,
            dtype=self.pos_embed.weight.dtype,
            rotary=self.rotary_pos_emb,
        )
        hidden_states = patch_embed(hidden_states)
        if hidden_states.shape[0] != metadata.total_tokens:
            raise ValueError("Vision grid token count does not match patch embedding output")
        hidden_states = (hidden_states + metadata.interpolate(self.pos_embed))[:, None]
        seq_len = original_seq_len = metadata.total_tokens
        use_cuda_graph_padding = False
        attention_mask = None
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=metadata.cu_seqlens,
            cu_seqlens_kv=metadata.cu_seqlens,
            qkv_format="thd",
            max_seqlen_q=metadata.max_seqlen,
            max_seqlen_kv=metadata.max_seqlen,
        )
        if self.config.vision_rope_backend == "reference":
            rotary_pos_emb = metadata.rotary_half[:, None, None, :].repeat(1, 1, 1, 2)
            rotary_pos_cos = rotary_pos_sin = None
        else:
            rotary_pos_emb = None
            rotary_pos_cos, rotary_pos_sin = metadata.cos, metadata.sin
    else:
        hidden_states = patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        seq_len, _ = hidden_states.size()

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, 1, 1, -1).repeat(1, 1, 1, 2)

        # Check if we need to pad for CUDA graphs
        use_cuda_graph_padding = self._uses_vision_cuda_graph()
        original_seq_len = seq_len
        if use_cuda_graph_padding:
            max_seq_len = self._get_max_vision_seq_length()
            hidden_states, rotary_pos_emb, seq_len = _maybe_pad_vision_sequence_for_cuda_graph(
                hidden_states, rotary_pos_emb, seq_len, max_seq_len
            )
        hidden_states = hidden_states[:, None]
        packed_seq_params, attention_mask = _vision_forward_packed_attention_setup(
            use_cuda_graph_padding=use_cuda_graph_padding,
            hidden_states=hidden_states,
            original_seq_len=original_seq_len,
            seq_len=seq_len,
            grid_thw=grid_thw,
            build_packed_seq_params=self.build_packed_seq_params,
        )
        rotary_pos_cos = rotary_pos_sin = None
        if self.config.vision_rope_backend != "reference":
            # Positions are fixed across layers, while Q/K change in each layer.
            # Keep tables local to this microbatch, including during recompute.
            rotary_pos_cos, rotary_pos_sin = prepare_vision_rotary_cos_sin(rotary_pos_emb)
            rotary_pos_emb = None
    hidden_states, deepstack_feature_lists = self.decoder(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        packed_seq_params=packed_seq_params,
        **(extra_block_kwargs or {}),
    )
    # Remove padding if we added it
    if use_cuda_graph_padding and original_seq_len < seq_len:
        hidden_states = hidden_states[:original_seq_len]
        # Unpad deepstack features - they go through a merger that reduces by spatial_merge_size^2
        # So their length is seq_len // (spatial_merge_size^2)
        original_merged_seq_len = original_seq_len // (self.spatial_merge_size**2)
        deepstack_feature_lists = [feat[:original_merged_seq_len] for feat in deepstack_feature_lists]
    hidden_states = self.merger(hidden_states)

    # Encodes images into continuous embeddings that can be forwarded to the language model.
    if metadata is None:
        split_sizes = (grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
        hidden_states = torch.split(hidden_states, split_sizes)
        hidden_states = torch.cat(hidden_states, dim=0)
    return hidden_states, deepstack_feature_lists


def clear_vision_patch_graph(self) -> dict | None:
    """Return lifetime statistics and release graph storage at training teardown."""
    if self.patch_graph is None:
        return None
    stats = self.patch_graph.stats()
    self.patch_graph.clear()
    return stats
