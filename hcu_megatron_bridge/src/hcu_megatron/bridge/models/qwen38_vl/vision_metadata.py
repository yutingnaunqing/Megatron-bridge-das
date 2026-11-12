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

"""Bounded, weight-independent geometry for Qwen's packed vision encoder."""

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PreparedVisionMetadata:
    """Device geometry in the actual spatial-merge token order, without model outputs."""

    indices: Tensor
    weights: Tensor
    order: Tensor
    rotary_half: Tensor
    cos: Tensor
    sin: Tensor
    cu_seqlens: Tensor
    max_seqlen: int
    total_tokens: int
    split_sizes: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        """Return owned device tensor storage bytes (views are not stored)."""
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.indices,
                self.weights,
                self.order,
                self.rotary_half,
                self.cos,
                self.sin,
                self.cu_seqlens,
            )
        )

    def interpolate(self, embedding: torch.nn.Embedding) -> Tensor:
        """Read current learned weights on every forward, retaining parameter gradients."""
        values = embedding(self.indices) * self.weights[:, :, None]
        interpolated = values[0] + values[1] + values[2] + values[3]
        return interpolated.index_select(0, self.order)


class VisionMetadataCache:
    """LRU cache keyed by complete ordered geometry, precision and rotary configuration.

    The caller supplies grids after all image/video reorder operations. Only
    geometric tensors are cached; learned embeddings and image contents are not.
    A miss and eviction never mutate previously returned tensors.
    """

    def __init__(self, *, max_entries: int, max_bytes: int) -> None:
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("Vision metadata cache limits must be nonnegative")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        """Release cached tensors; active autograd users retain their own references."""
        self.entries.clear()
        self.bytes = 0

    def get(
        self,
        grid_thw: Tensor,
        *,
        merge_size: int,
        grid_side: int,
        device: torch.device,
        dtype: torch.dtype,
        rotary: torch.nn.Module,
    ) -> PreparedVisionMetadata:
        """Read geometry once on CPU and prepare/cache token-aligned device tensors."""
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or grid_thw.dtype not in (torch.int32, torch.int64):
            raise ValueError("Vision grids must be an integer [images, 3] tensor")
        grids = tuple(tuple(row) for row in grid_thw.detach().cpu().tolist())
        if not grids or merge_size <= 0 or grid_side <= 0:
            raise ValueError("Prepared vision metadata requires nonempty grids and positive geometry")
        for frames, height, width in grids:
            if frames <= 0 or height <= 0 or width <= 0 or height % merge_size or width % merge_size:
                raise ValueError("Vision grids must be positive and spatially divisible by merge size")
        key = (grids, merge_size, grid_side, device, dtype, rotary.dim, rotary.theta)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        with torch.no_grad():
            result = self._prepare(grids, merge_size, grid_side, device, dtype, rotary)
        if self.max_entries and result.nbytes <= self.max_bytes:
            while self.entries and (
                len(self.entries) >= self.max_entries or self.bytes + result.nbytes > self.max_bytes
            ):
                _, evicted = self.entries.popitem(last=False)
                self.bytes -= evicted.nbytes
            self.entries[key] = result
            self.bytes += result.nbytes
        return result

    @staticmethod
    def _prepare(grids, merge_size, grid_side, device, dtype, rotary) -> PreparedVisionMetadata:
        indices, weights, orders, positions = [], [], [], []
        lengths, split_sizes = [], []
        spatial_offset = 0
        for frames, height, width in grids:
            # Match the reference's CPU FP32 linspace/weight arithmetic, then
            # convert weights once to the learned embedding's dtype.
            hs = torch.linspace(0, grid_side - 1, height, device="cpu", dtype=torch.float32)
            ws = torch.linspace(0, grid_side - 1, width, device="cpu", dtype=torch.float32)
            hf, wf = hs.int(), ws.int()
            hc, wc = (hf + 1).clamp(max=grid_side - 1), (wf + 1).clamp(max=grid_side - 1)
            dh, dw = hs - hf, ws - wf
            indices.append(
                torch.stack(
                    [
                        (a[:, None] * grid_side + b[None, :]).flatten()
                        for a, b in ((hf, wf), (hf, wc), (hc, wf), (hc, wc))
                    ]
                ).long()
            )
            weights.append(
                torch.stack(
                    [
                        (a[:, None] * b[None, :]).flatten()
                        for a, b in ((1 - dh, 1 - dw), (1 - dh, dw), (dh, 1 - dw), (dh, dw))
                    ]
                )
            )
            order = (
                torch.arange(height * width, device="cpu")
                .view(height // merge_size, merge_size, width // merge_size, merge_size)
                .permute(0, 2, 1, 3)
                .reshape(-1)
                .repeat(frames)
            )
            orders.append(order + spatial_offset)
            positions.append(torch.stack((order // width, order % width), dim=-1))
            spatial_offset += height * width
            lengths.extend([height * width] * frames)
            split_sizes.append(frames * height * width // merge_size**2)
        total = sum(lengths)
        if total > torch.iinfo(torch.int32).max:
            raise ValueError("Vision token count exceeds int32 packed-attention indices")
        frequency_table = rotary(max(max(h, w) for _, h, w in grids))
        half = frequency_table[torch.cat(positions).to(device=device)].flatten(1).contiguous()
        # rotary's inverse frequency buffer is model/device-local; reject stale
        # placement rather than implicitly moving it behind the caller's back.
        if half.device != device:
            raise ValueError("Vision rotary module and patch embedding must share a device")
        return PreparedVisionMetadata(
            indices=torch.cat(indices, dim=1).to(device=device),
            weights=torch.cat(weights, dim=1).to(device=device, dtype=dtype),
            order=torch.cat(orders).to(device=device),
            rotary_half=half,
            cos=half.float().cos().contiguous(),
            sin=half.float().sin().contiguous(),
            cu_seqlens=torch.tensor([0] + lengths, dtype=torch.int32, device="cpu")
            .cumsum(0, dtype=torch.int32)
            .to(device),
            max_seqlen=max(lengths),
            total_tokens=total,
            split_sizes=tuple(split_sizes),
        )
