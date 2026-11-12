# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Compare prepared metadata against unchanged reference methods with Conv3d retained."""

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
HCU = ROOT / "hcu_megatron_bridge/src/hcu_megatron/bridge/models/qwen38_vl"
pytestmark = pytest.mark.unit


def _load_metadata():
    spec = importlib.util.spec_from_file_location(
        "standalone_vision_metadata",
        ROOT / "hcu_megatron_bridge/src/hcu_megatron/bridge/models/qwen38_vl/vision_metadata.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


metadata = _load_metadata()
namespace = {"torch": torch, "nn": nn}
vision = ast.parse((MODEL / "vision_model.py").read_text())
vision_class = next(n for n in vision.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3VLVisionModel")
methods = [
    n
    for n in vision_class.body
    if isinstance(n, ast.FunctionDef) and n.name in ("rot_pos_emb", "fast_pos_embed_interpolate")
]
exec(  # noqa: S102 -- compile only trusted local definitions, avoiding GPU-initializing imports.
    compile(
        ast.Module(body=methods, type_ignores=[]),
        str(MODEL / "vision_model.py"),
        "exec",
        flags=__future__.annotations.compiler_flag,
    ),
    namespace,
)


class Rotary(nn.Module):
    dim = 36
    theta = 10000.0

    def __init__(self, device):
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / (self.theta ** (torch.arange(0, 36, 2, device=device).float() / 36)),
        )

    def forward(self, length):
        return torch.outer(
            torch.arange(length, device=self.inv_freq.device, dtype=self.inv_freq.dtype),
            self.inv_freq,
        )


def _reference(device="cpu", dtype=torch.float32):
    return SimpleNamespace(
        spatial_merge_size=2,
        num_grid_per_side=8,
        config=SimpleNamespace(spatial_merge_size=2),
        pos_embed=nn.Embedding(64, 16, device=device, dtype=dtype),
        rotary_pos_emb=Rotary(device),
    )


def _get(cache, grid, model):
    return cache.get(
        grid,
        merge_size=2,
        grid_side=8,
        device=model.pos_embed.weight.device,
        dtype=model.pos_embed.weight.dtype,
        rotary=model.rotary_pos_emb,
    )


@pytest.mark.parametrize("grid", [[[1, 4, 6]], [[1, 4, 6], [2, 2, 4]], [[3, 6, 2], [1, 2, 8]]])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prepared_geometry_matches_reference_and_parameter_gradients(grid, device, dtype):
    torch.manual_seed(123)
    model = _reference(device, dtype)
    grids = torch.tensor(grid, device=device)
    cache = metadata.VisionMetadataCache(max_entries=4, max_bytes=4 * 1024**2)
    actual = _get(cache, grids, model)
    expected_pos = namespace["fast_pos_embed_interpolate"](model, grids)
    expected_angles = namespace["rot_pos_emb"](model, grids)
    torch.testing.assert_close(actual.rotary_half, expected_angles, atol=0, rtol=0)
    torch.testing.assert_close(actual.cos, expected_angles.cos(), atol=0, rtol=0)
    torch.testing.assert_close(actual.sin, expected_angles.sin(), atol=0, rtol=0)
    actual_pos = actual.interpolate(model.pos_embed)
    torch.testing.assert_close(actual_pos, expected_pos, atol=0, rtol=0)
    grad = torch.randn_like(actual_pos)
    a_grad = torch.autograd.grad(actual_pos, model.pos_embed.weight, grad)[0]
    e_grad = torch.autograd.grad(expected_pos, model.pos_embed.weight, grad)[0]
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(a_grad, e_grad, atol=tolerance, rtol=tolerance)
    lengths = [h * w for t, h, w in grid for _ in range(t)]
    expected_cu = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    torch.testing.assert_close(actual.cu_seqlens.cpu(), expected_cu)
    assert isinstance(actual.max_seqlen, int) and actual.max_seqlen == max(lengths)
    assert actual.total_tokens == sum(lengths)
    assert actual.split_sizes == tuple(t * h * w // 4 for t, h, w in grid)
    assert _get(cache, grids, model) is actual
    assert cache.hits == 1
    with torch.no_grad():
        model.pos_embed.weight.add_(1)
    assert not torch.equal(actual.interpolate(model.pos_embed), actual_pos)


def test_metadata_cache_keys_boundaries_eviction_disable_and_errors():
    model = _reference()
    cache = metadata.VisionMetadataCache(max_entries=1, max_bytes=1024**2)
    first = _get(cache, torch.tensor([[1, 4, 4]]), model)
    # Equal total tokens but different frame boundaries and H/W must not alias.
    second = _get(cache, torch.tensor([[2, 2, 4]]), model)
    assert first.total_tokens == second.total_tokens
    assert first.cu_seqlens.tolist() == [0, 16]
    assert second.cu_seqlens.tolist() == [0, 8, 16]
    assert len(cache.entries) == 1 and cache.bytes == second.nbytes
    assert _get(cache, torch.tensor([[1, 4, 4]]), model) is not first
    assert first.cu_seqlens.tolist() == [
        0,
        16,
    ]  # Eviction does not mutate retained entries.
    cache.clear()
    assert cache.bytes == 0 and not cache.entries
    for entries, size in [(0, 1024**2), (10, 1)]:
        disabled = metadata.VisionMetadataCache(max_entries=entries, max_bytes=size)
        _get(disabled, torch.tensor([[1, 4, 4]]), model)
        assert not disabled.entries and disabled.bytes == 0
    for grid in [[[1, 3, 4]], [[0, 4, 4]], []]:
        with pytest.raises(ValueError):
            _get(cache, torch.tensor(grid, dtype=torch.long), model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("backend", ["reference", "flash"])
def test_real_vision_metadata_parity(tmp_path, frozen, backend):
    import torch.distributed as dist
    from hcu_megatron.bridge.models.qwen38_vl.provider import HCUQwen35VLModelProvider as Qwen35VLModelProvider
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1)
    try:
        torch.cuda.set_device(0)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(1234)
        provider = Qwen35VLModelProvider(
            num_layers=64,
            hidden_size=5120,
            ffn_hidden_size=17408,
            num_attention_heads=24,
            sequence_parallel=False,
            vision_config=Qwen3_5VisionConfig(
                depth=2,
                hidden_size=144,
                intermediate_size=256,
                num_heads=2,
                out_hidden_size=128,
                num_position_embeddings=64,
                deepstack_visual_indexes=[],
            ),
            vision_rope_backend=backend,
        )
        pg = ProcessGroupCollection.use_mpu_process_groups()
        reference = build_module(provider.build_vision_encoder_spec(), pg_collection=pg).cuda().bfloat16().train()
        provider.vision_metadata_backend = "prepared"
        candidate = build_module(provider.build_vision_encoder_spec(), pg_collection=pg).cuda().bfloat16().train()
        candidate.load_state_dict(reference.state_dict())
        for model in [reference, candidate]:
            model.requires_grad_(not frozen)
        for grids in [
            [[1, 4, 6], [2, 2, 4]],
            [[2, 2, 4], [1, 4, 6]],
            [[1, 4, 6], [2, 2, 4]],
        ]:
            grid = torch.tensor(grids, device="cuda")
            reference.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            x = torch.randn(40, 1536, device="cuda", dtype=torch.bfloat16, requires_grad=not frozen)
            y = x.detach().clone().requires_grad_(not frozen)
            e, _ = reference(x, grid)
            a, _ = candidate(y, grid)
            torch.testing.assert_close(a, e, atol=1e-2, rtol=1e-2)
            if not frozen:
                grad = torch.randn_like(a)
                a.backward(grad)
                e.backward(grad)
                torch.testing.assert_close(y.grad, x.grad, atol=1e-2, rtol=1e-2)
                for name, param in candidate.named_parameters():
                    torch.testing.assert_close(
                        param.grad, dict(reference.named_parameters())[name].grad, atol=2e-2, rtol=2e-2
                    )
        assert candidate.metadata_cache.hits == 1 and candidate.metadata_cache.misses == 2
        assert not any("metadata_cache" in key for key in candidate.state_dict())
        candidate.float()
        assert not candidate.metadata_cache.entries
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_rotary_coefficients_remain_fp32_after_frequency_buffer_dtype_change():
    model = _reference(dtype=torch.bfloat16)
    model.rotary_pos_emb.bfloat16()
    cache = metadata.VisionMetadataCache(max_entries=4, max_bytes=1024**2)
    grid = torch.tensor([[1, 4, 6]])
    actual = _get(cache, grid, model)
    angles = namespace["rot_pos_emb"](model, grid)
    assert actual.rotary_half.dtype == angles.dtype
    assert actual.cos.dtype == actual.sin.dtype == torch.float32
    torch.testing.assert_close(actual.cos, angles.float().cos(), atol=0, rtol=0)
    torch.testing.assert_close(actual.sin, angles.float().sin(), atol=0, rtol=0)
