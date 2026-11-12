# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Exact arithmetic, fallback and output-lifetime checks for the original patch graph."""

import __future__

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / "3rdparty/Megatron-Bridge/src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl"
spec = importlib.util.spec_from_file_location(
    "standalone_patch_graph", ROOT / "hcu_megatron_bridge/src/hcu_megatron/bridge/models/qwen38_vl/vision_graph.py"
)
graph_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = graph_module
spec.loader.exec_module(graph_module)
FrozenPatchGraph = graph_module.FrozenPatchGraph
source = ast.parse((MODEL / "utils.py").read_text())
node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3VLVisionPatchEmbed")
namespace = {"torch": torch, "nn": nn}
exec(  # noqa: S102 -- trusted local class only; avoid TE initialization during CPU collection.
    compile(
        ast.Module(body=[node], type_ignores=[]),
        str(MODEL / "utils.py"),
        "exec",
        flags=__future__.annotations.compiler_flag,
    ),
    namespace,
)
Patch = namespace["Qwen3VLVisionPatchEmbed"]
HCU = ROOT / "hcu_megatron_bridge/src/hcu_megatron/bridge/models/qwen38_vl"
pytestmark = pytest.mark.unit
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")


def patch(device="cuda", dtype=torch.bfloat16):
    return (
        Patch(SimpleNamespace(in_channels=3, temporal_patch_size=2, patch_size=16, hidden_size=1152))
        .to(device=device, dtype=dtype)
        .requires_grad_(False)
    )


def wrapper(module, **kwargs):
    options = dict(token_counts=(40, 44), max_entries=2, max_bytes=128 * 1024**2, capture_calls=20, warmup_calls=3)
    options.update(kwargs)
    return FrozenPatchGraph(module, **options)


def test_cpu_fallback_keeps_actual_convolution_and_gradients():
    module = patch("cpu", torch.float32)
    graph = wrapper(module)
    x = torch.randn(2, 1536, requires_grad=True)
    output = graph(x)
    expected = module(x)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert torch.equal(torch.autograd.grad(output.sum(), x)[0], torch.autograd.grad(expected.sum(), x)[0])
    assert graph.stats()["fallbacks"] == {"non_gpu": 1} and graph.stats()["captures"] == 0


@gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("capture_error_mode", ["global", "thread_local"])
def test_replay_changed_inputs_owns_outputs_and_interleaves_shapes(dtype, capture_error_mode):
    module = patch()
    graph = wrapper(module, capture_error_mode=capture_error_mode)
    try:
        x = torch.randn(40, 1536, device="cuda", dtype=dtype)
        y = torch.randn_like(x)
        with torch.no_grad():
            first = graph(x)
            saved = first.clone()
            second = graph(y)
            torch.testing.assert_close(first, module(x), atol=0, rtol=0)
            torch.testing.assert_close(second, module(y), atol=0, rtol=0)
            assert first.data_ptr() != second.data_ptr()
            assert not torch.equal(first, second) and torch.equal(first, saved)
            # An eager shape can reuse library workspaces; the cached graph
            # must still read the new input and preserve the old output.
            graph(torch.randn(48, 1536, device="cuda", dtype=dtype))
            assert torch.equal(graph(x), module(x))
        assert graph.captures == 1 and graph.replays == 3
        assert graph.fallbacks["token_count"] == 1
    finally:
        graph.clear()
    assert not graph.entries and graph.bytes == 0


@gpu
def test_multiple_forwards_then_downstream_backward_matches_eager():
    module = patch()
    graph = wrapper(module)
    expected_head = nn.Linear(1152, 16).cuda().bfloat16()
    actual_head = nn.Linear(1152, 16).cuda().bfloat16()
    actual_head.load_state_dict(expected_head.state_dict())
    try:
        inputs = [torch.randn(40, 1536, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        with torch.no_grad():
            expected_features = [module(x) for x in inputs]
        actual_features = [graph(x) for x in inputs]
        expected_loss = sum(expected_head(x).float().square().mean() for x in expected_features)
        actual_loss = sum(actual_head(x).float().square().mean() for x in actual_features)
        actual_loss.backward()
        expected_loss.backward()
        assert torch.equal(actual_loss, expected_loss)
        for a, b in zip(actual_head.parameters(), expected_head.parameters()):
            assert torch.equal(a.grad, b.grad)
        assert graph.captures == 1 and graph.replays == 3
    finally:
        graph.clear()


@gpu
def test_trainable_input_or_parameters_use_eager_autograd():
    module = patch()
    graph = wrapper(module)
    x = torch.randn(40, 1536, device="cuda", requires_grad=True)
    actual = graph(x)
    expected = module(x)
    grad = torch.randn_like(actual)
    assert torch.equal(torch.autograd.grad(actual, x, grad)[0], torch.autograd.grad(expected, x, grad)[0])
    module.requires_grad_(True)
    graph(x.detach()).sum().backward()
    assert module.proj.weight.grad is not None
    assert graph.captures == 0 and graph.fallbacks["requires_grad"] == 2


@gpu
def test_limits_closed_window_and_safe_capture_failure(monkeypatch):
    module = patch()
    x = torch.randn(40, 1536, device="cuda")
    graph = wrapper(module, max_entries=1)
    try:
        assert torch.equal(graph(x), module(x))
        other = torch.randn(44, 1536, device="cuda")
        assert torch.equal(graph(other), module(other))
        assert graph.fallbacks["entry_budget"] == 1
    finally:
        graph.clear()
    expired = wrapper(module, capture_calls=1)
    expired(torch.randn(48, 1536, device="cuda"))
    assert torch.equal(expired(x), module(x)) and expired.fallbacks["capture_window_closed"] == 1
    small = wrapper(module, max_bytes=1)
    try:
        assert torch.equal(small(x), module(x))
        assert small.captures == 0 and small.fallbacks["byte_budget"] == 1
    finally:
        small.clear()
    failed = wrapper(module)

    def fail(value):
        raise RuntimeError("synthetic safe capture failure")

    monkeypatch.setattr(failed, "_capture", fail)
    assert torch.equal(failed(x), module(x)) and torch.equal(failed(x), module(x))
    assert len(failed.failed) == 1 and failed.fallbacks["capture_failed"] == 2


@gpu
def test_changed_weight_invalidates_before_replay():
    module = patch()
    graph = wrapper(module)
    try:
        x = torch.randn(40, 1536, device="cuda")
        original = graph(x)
        saved = original.clone()
        with torch.no_grad():
            module.proj.bias.add_(0.125)
        actual = graph(x)
        assert torch.equal(actual, module(x)) and torch.equal(original, saved)
        assert not torch.equal(original, actual) and graph.captures == 2
    finally:
        graph.clear()


def test_relaxed_capture_policy_is_rejected():
    with pytest.raises(ValueError, match="global or thread_local"):
        wrapper(patch("cpu", torch.float32), capture_error_mode="relaxed")
