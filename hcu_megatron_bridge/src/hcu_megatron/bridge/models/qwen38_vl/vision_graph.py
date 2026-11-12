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

"""Opt-in forward-only graphs for the unchanged frozen vision patch convolution."""

import logging
import time
from collections import Counter
from dataclasses import dataclass

import torch
from torch import Tensor, nn


logger = logging.getLogger(__name__)


@dataclass
class _PatchGraphEntry:
    graph: torch.cuda.CUDAGraph
    stream: torch.cuda.Stream
    static_input: Tensor
    static_output: Tensor
    last_use: torch.cuda.Event
    owned_bytes: int


class FrozenPatchGraph:
    """Bounded exact-shape capture, preserving Conv3d arithmetic and output ownership.

    This is deliberately not an nn.Module: the existing patch module remains the
    sole owner of weight/bias and state_dict names. Graph inputs/workspaces are
    runtime-only state. Capture contains no distributed collectives.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        token_counts: tuple[int, ...],
        max_entries: int,
        max_bytes: int,
        capture_calls: int,
        warmup_calls: int = 3,
        capture_error_mode: str = "global",
    ) -> None:
        if not token_counts or any(t <= 0 for t in token_counts):
            raise ValueError("Patch graphs require a nonempty positive exact-token whitelist")
        if min(max_entries, max_bytes, capture_calls, warmup_calls) <= 0:
            raise ValueError("Patch graph entry/byte/capture/warmup limits must be positive")
        if capture_error_mode not in ("global", "thread_local"):
            raise ValueError("Patch Graph capture mode must be global or thread_local")
        self.capture_error_mode = capture_error_mode
        self.module = module
        self.token_counts = frozenset(token_counts)
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.capture_calls = capture_calls
        self.warmup_calls = warmup_calls
        self.entries: dict[tuple, _PatchGraphEntry] = {}
        self.failed: dict[tuple, str] = {}
        self.calls = 0
        self.bytes = 0
        self.captures = 0
        self.replays = 0
        self.capture_ms = 0.0
        self.fallbacks: Counter = Counter()
        self._weight_stamp = None

    def clear(self) -> None:
        """Wait for graph consumers' copies, then release graph-owned storage."""
        for entry in self.entries.values():
            entry.last_use.synchronize()
            entry.graph.reset()
        self.entries.clear()
        self.failed.clear()
        self.bytes = 0
        self._weight_stamp = None

    def stats(self) -> dict:
        """Return observable coverage without claiming every call replayed."""
        return {
            "capture_error_mode": self.capture_error_mode,
            "calls": self.calls,
            "captures": self.captures,
            "replays": self.replays,
            "entries": len(self.entries),
            "owned_bytes_estimate": self.bytes,
            "capture_ms": self.capture_ms,
            "fallbacks": dict(self.fallbacks),
            "failed_signatures": len(self.failed),
        }

    def _stamp(self) -> tuple:
        return tuple(
            (p.data_ptr(), p._version, p.device, p.dtype, tuple(p.shape), p.stride()) for p in self.module.parameters()
        )

    def _ineligible_reason(self, x: Tensor) -> str | None:
        if not x.is_cuda:
            return "non_gpu"
        if x.requires_grad or any(p.requires_grad for p in self.module.parameters()):
            return "requires_grad"
        if torch.is_autocast_enabled("cuda"):
            return "autocast"
        if x.ndim != 2 or not x.is_contiguous():
            return "layout"
        if x.dtype not in (torch.float32, torch.bfloat16) or self.module.proj.weight.dtype != torch.bfloat16:
            return "dtype"
        if x.shape[0] not in self.token_counts:
            return "token_count"
        return None

    def _capture(self, x: Tensor) -> _PatchGraphEntry:
        logger.info(
            "Capturing frozen patch graph shape=%s dtype=%s mode=%s",
            tuple(x.shape),
            x.dtype,
            self.capture_error_mode,
        )
        before = torch.cuda.memory_reserved(x.device)
        static_input = torch.empty_like(x)
        static_input.copy_(x)
        current = torch.cuda.current_stream(x.device)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(current)
        with torch.no_grad(), torch.cuda.stream(stream):
            for _ in range(self.warmup_calls):
                self.module(static_input)
        current.wait_stream(stream)
        # G1 contains no TP/PP operations. Finish warmup and allocator work
        # before entering capture; never synchronize from inside the graph.
        torch.cuda.synchronize(x.device)
        graph = torch.cuda.CUDAGraph()
        # The graph contains only frozen forward work in this Python thread.
        # thread_local allows independent DataLoader pin-memory activity; it
        # retains capture checks in this thread and does not capture backward.
        with torch.no_grad(), torch.cuda.graph(graph, stream=stream, capture_error_mode=self.capture_error_mode):
            static_output = self.module(static_input)
        torch.cuda.synchronize(x.device)
        payload = (
            static_input.numel() * static_input.element_size() + static_output.numel() * static_output.element_size()
        )
        owned_bytes = max(payload, torch.cuda.memory_reserved(x.device) - before)
        last_use = torch.cuda.Event()
        last_use.record(current)
        return _PatchGraphEntry(graph, stream, static_input, static_output, last_use, owned_bytes)

    def __call__(self, x: Tensor) -> Tensor:
        """Replay eligible frozen inputs, returning a fresh consumer-owned output."""
        self.calls += 1
        reason = self._ineligible_reason(x)
        if reason is not None:
            self.fallbacks[reason] += 1
            return self.module(x)
        stamp = self._stamp()
        if self._weight_stamp is not None and stamp != self._weight_stamp:
            self.clear()
        self._weight_stamp = stamp
        with torch.cuda.device(x.device):
            key = (
                tuple(x.shape),
                x.stride(),
                x.dtype,
                x.device,
                stamp,
                torch.cuda.current_stream(x.device).cuda_stream,
            )
            entry = self.entries.get(key)
            if entry is None:
                if key in self.failed:
                    reason = "capture_failed"
                elif self.calls > self.capture_calls:
                    reason = "capture_window_closed"
                elif len(self.entries) >= self.max_entries:
                    reason = "entry_budget"
                elif self.bytes >= self.max_bytes:
                    reason = "byte_budget"
                else:
                    started = time.perf_counter()
                    try:
                        entry = self._capture(x)
                    except RuntimeError as exc:
                        message = str(exc)
                        fatal = any(
                            word in message.lower()
                            for word in ("illegal memory", "device-side assert", "illegal address")
                        )
                        if fatal or torch.cuda.is_current_stream_capturing():
                            raise
                        self.failed[key] = message
                        logger.warning("Frozen patch capture failed; eager fallback: %s", message)
                        reason = "capture_failed"
                    else:
                        self.capture_ms += (time.perf_counter() - started) * 1000
                        if self.bytes + entry.owned_bytes > self.max_bytes:
                            entry.last_use.synchronize()
                            entry.graph.reset()
                            self.failed[key] = "byte budget exceeded"
                            entry = None
                            reason = "byte_budget"
                        else:
                            self.entries[key] = entry
                            self.bytes += entry.owned_bytes
                            self.captures += 1
                            logger.info(
                                "Captured frozen patch graph shape=%s owned_bytes=%d",
                                tuple(x.shape),
                                entry.owned_bytes,
                            )
                if entry is None:
                    self.fallbacks[reason] += 1
                    return self.module(x)
            # Include both copy and clone in the path measured by callers.
            # The clone prevents subsequent microbatches overwriting features
            # retained by downstream trainable layers for weight gradients.
            entry.static_input.copy_(x)
            entry.graph.replay()
            result = entry.static_output.clone()
            entry.last_use.record(torch.cuda.current_stream(x.device))
            self.replays += 1
            return result
