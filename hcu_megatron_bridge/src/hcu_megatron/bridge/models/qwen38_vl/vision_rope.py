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

"""Microbatch-local, FP32 rotary tables for the Qwen vision encoder.

The flash backend reuses FlashAttention's half-split Triton rotary kernel.
Its public wrapper in some ROCm builds requires input and table dtypes to
match, although the kernel loads both into FP32. This adapter preserves the
vision reference's FP32 arithmetic without materializing FP32 Q/K tensors.
"""

from contextlib import nullcontext
from functools import lru_cache

import torch
from torch import Tensor


VISION_ROPE_BACKENDS = ("reference", "cached", "flash")


@lru_cache(maxsize=1)
def _get_flash_rotary_kernel():
    try:
        from flash_attn.ops.triton.rotary import rotary_kernel
    except ImportError as exc:
        raise RuntimeError("vision_rope_backend='flash' requires FlashAttention's Triton rotary kernel") from exc
    # Fail explicitly on a different kernel ABI rather than silently benchmarking
    # the eager path. The adapter targets the FA2 kernel used by the DTK build.
    required_args = {"BLOCK_K", "BLOCK_M", "rotary_dim", "stride_x_headdim"}
    if not required_args.issubset(rotary_kernel.arg_names):
        raise RuntimeError("Unsupported FlashAttention rotary kernel ABI; use vision_rope_backend='cached'")
    return rotary_kernel


def validate_vision_rope_backend(backend: str) -> None:
    """Validate the backend and, when requested, the optional FlashAttention ABI."""
    if backend not in VISION_ROPE_BACKENDS:
        raise ValueError(f"vision_rope_backend must be one of {VISION_ROPE_BACKENDS}, got {backend!r}")
    if backend == "flash":
        _get_flash_rotary_kernel()


def prepare_vision_rotary_cos_sin(freqs: Tensor) -> tuple[Tensor, Tensor]:
    """Build shared FP32 tables from duplicated, token-aligned vision angles.

    Args:
        freqs: Angles with shape ``[T, 1, 1, D]`` and duplicated halves.

    Returns:
        Contiguous cosine and sine tensors, each with shape ``[T, D / 2]``.
        They follow the actual packed token order, not per-frame offsets.
    """
    if freqs.ndim != 4 or freqs.shape[1:3] != (1, 1) or freqs.shape[-1] % 2:
        raise ValueError("Vision rotary angles must have shape [T, 1, 1, even_D]")
    if freqs.requires_grad:
        raise ValueError("Vision rotary position tables must not require gradients")
    angles = freqs[:, 0, 0, : freqs.shape[-1] // 2].float()
    return angles.cos().contiguous(), angles.sin().contiguous()


def _launch_flash_rotary(x: Tensor, cos: Tensor, sin: Tensor, *, conjugate: bool) -> Tensor:
    """Launch the optional FA2 kernel on ``[1, T, H, D]`` without dtype casts."""
    kernel = _get_flash_rotary_kernel()
    out = torch.empty_like(x)
    _, tokens, heads, dim = x.shape
    block_k = max(32, 1 << (dim - 1).bit_length())
    block_m = 8 if dim <= 64 else 4
    with torch.cuda.device(x.device) if x.is_cuda else nullcontext():
        kernel[((tokens + block_m - 1) // block_m, 1, heads)](
            out,
            x,
            cos,
            sin,
            None,  # No sequence resets: tables are already token-aligned.
            0,
            tokens,
            dim,
            tokens,
            *out.stride(),
            *x.stride(),
            BLOCK_K=block_k,
            IS_SEQLEN_OFFSETS_TENSOR=False,
            IS_VARLEN=False,
            INTERLEAVED=False,
            CONJUGATE=conjugate,
            BLOCK_M=block_m,
            # Match the reference's separate FP32 multiplies and add.
            enable_fp_fusion=False,
        )
    return out


class _FlashVisionRotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        ctx.save_for_backward(cos, sin)
        return _launch_flash_rotary(x, cos, sin, conjugate=False)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, None]:
        cos, sin = ctx.saved_tensors
        return _launch_flash_rotary(grad_output, cos, sin, conjugate=True), None, None


def apply_vision_rotary_pos_emb(t: Tensor, *, cos: Tensor, sin: Tensor, backend: str) -> Tensor:
    """Apply out-of-place, full-head half-split RoPE using shared FP32 tables.

    Args:
        t: Vision Q or K in ``[T, H, D]`` or ``[T, 1, H, D]`` layout.
        cos: FP32 cosine table, shape ``[T, D / 2]``.
        sin: FP32 sine table, shape ``[T, D / 2]``.
        backend: ``cached`` for PyTorch or ``flash`` for a fused FA2 kernel.

    Returns:
        Rotated Q or K with the original shape and dtype. Q/K storage is not
        modified, and input gradients use the inverse rotation.
    """
    if t.ndim not in (3, 4) or (t.ndim == 4 and t.shape[1] != 1):
        raise ValueError("Vision Q/K must have shape [T, H, D] or [T, 1, H, D]")
    if cos.ndim != 2 or cos.shape != sin.shape or cos.shape != (t.shape[0], t.shape[-1] // 2):
        raise ValueError("Vision cos/sin must match every token and the full, even head dimension")
    if t.shape[-1] % 2 or cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("Vision rotary requires an even head dimension and FP32 cos/sin")
    if cos.device != t.device or sin.device != t.device:
        raise ValueError("Vision Q/K and cos/sin must be on the same device")
    if cos.requires_grad or sin.requires_grad:
        raise ValueError("Vision rotary position tables must not require gradients")
    if backend == "cached":
        broadcast_shape = (t.shape[0],) + (1,) * (t.ndim - 2) + (cos.shape[-1],)
        c, s = cos.view(broadcast_shape), sin.view(broadcast_shape)
        first, second = t.float().chunk(2, dim=-1)
        return torch.cat((first * c - second * s, second * c + first * s), dim=-1).to(t.dtype)
    if backend != "flash":
        raise ValueError(f"Shared vision rotary expects backend='cached' or 'flash', got {backend!r}")
    if not t.is_cuda:
        raise ValueError("vision_rope_backend='flash' requires a CUDA/HIP device")
    if t.shape[-1] > 256:
        raise ValueError("FlashAttention vision rotary supports head dimensions up to 256")
    thd = t if t.ndim == 3 else t.squeeze(1)
    result = _FlashVisionRotary.apply(thd.unsqueeze(0), cos.contiguous(), sin.contiguous()).squeeze(0)
    return result if t.ndim == 3 else result.unsqueeze(1)
