# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Migration-specific invariants and real-module integration checks."""

import ast
import hashlib
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]
HCU = ROOT / "hcu_megatron_bridge/src/hcu_megatron/bridge/models/qwen38_vl"
pytestmark = pytest.mark.unit


def test_upstream_files_match_adapter_compatibility_manifest():
    tree = ast.parse((HCU / "compatibility.py").read_text())
    expected = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_EXPECTED" for t in n.targets)
    )
    base = ROOT / "3rdparty/Megatron-Bridge/src/megatron/bridge/models/qwen_vl"
    for name, digest in ast.literal_eval(expected).items():
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == digest


def test_environment_does_not_copy_recipe_files_into_upstream():
    text = (ROOT / "env/env.sh").read_text()
    assert not any(line.lstrip().startswith("cp ") for line in text.splitlines())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an idle GPU")
def test_adapter_preserves_parameters_checkpoint_and_exact_outputs(tmp_path):
    import torch.distributed as dist
    from hcu_megatron.bridge.models.qwen38_vl.adapter import attach_vision_optimizations, detach_vision_optimizations
    from hcu_megatron.bridge.models.qwen38_vl.provider import HCUQwen35VLModelProvider
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    dist.init_process_group("nccl", init_method="file://" + str(tmp_path / "dist"), rank=0, world_size=1)
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        provider = HCUQwen35VLModelProvider(
            num_layers=2,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            vision_hcu_enabled=False,
            vision_config=Qwen3_5VisionConfig(
                depth=2,
                hidden_size=128,
                intermediate_size=256,
                num_heads=4,
                out_hidden_size=128,
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
                num_position_embeddings=64,
                deepstack_visual_indexes=[],
            ),
        )
        pg = ProcessGroupCollection.use_mpu_process_groups()
        model = build_module(provider.build_vision_encoder_spec(), pg_collection=pg).cuda().bfloat16().train()
        parameters = {n: id(v) for n, v in model.named_parameters()}
        checkpoint = {n: v.detach().clone() for n, v in model.state_dict().items()}
        sharded_keys = set(model.sharded_state_dict())
        original_forward = model.forward.__func__
        x = torch.randn(40, 1536, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        grid = torch.tensor([[1, 4, 6], [2, 2, 4]], device="cuda")
        expected, _ = model(x, grid)
        upstream = torch.randn_like(expected)
        expected.backward(upstream)
        expected_x_grad = x.grad.clone()
        expected_grads = {n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        x.grad = None
        provider.vision_hcu_enabled = True
        provider.vision_rope_backend = "flash"
        provider.vision_metadata_backend = "prepared"
        attach_vision_optimizations(model, provider)
        assert {n: id(v) for n, v in model.named_parameters()} == parameters
        assert set(model.state_dict()) == set(checkpoint)
        assert set(model.sharded_state_dict()) == sharded_keys
        model.load_state_dict(checkpoint, strict=True)
        actual, _ = model(x, grid)
        actual.backward(upstream)
        assert torch.equal(actual, expected)
        assert torch.equal(x.grad, expected_x_grad)
        for name, p in model.named_parameters():
            ref = expected_grads[name]
            assert p.grad is None if ref is None else torch.equal(p.grad, ref)
        with pytest.raises(RuntimeError, match="already attached"):
            attach_vision_optimizations(model, provider)
        assert model.metadata_cache.entries
        model.float()
        assert not model.metadata_cache.entries
        model.bfloat16()
        detach_vision_optimizations(model)
        assert model.forward.__func__ is original_forward
        assert not hasattr(model, "metadata_cache")
        assert {n: id(v) for n, v in model.named_parameters()} == parameters
        # Exercise the standalone provider/spec path as well as attach/detach.
        standalone = build_module(provider.build_vision_encoder_spec(), pg_collection=pg).cuda().bfloat16()
        assert standalone._hcu_vision_attached
        standalone.load_state_dict(checkpoint, strict=True)
        with torch.no_grad():
            assert torch.equal(standalone(x.detach(), grid)[0], expected)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_provider_conversion_preserves_dynamic_hf_fields_and_loading_hooks():
    from dataclasses import dataclass, fields

    @dataclass
    class Upstream:
        hidden_size: int = 128

    @dataclass
    class Local(Upstream):
        hf_text_config: object = None
        head_dim: int = 256
        hf_model_id: str = ""

    tree = ast.parse((HCU / "provider.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "from_upstream")
    namespace = {
        "fields": fields,
        "Qwen35VLModelProvider": Upstream,
        "Qwen35VLMoEModelProvider": type("Other", (), {}),
        "HCUQwen35VLModelProvider": Local,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "provider.py", "exec"), namespace)
    original = Upstream()
    text_config = object()
    # Reproduce the pinned upstream AutoBridge's post-construction assignments.
    original.hf_text_config = text_config
    original.head_dim = 72
    original.hf_model_id = "local-test"
    original._pre_wrap_hooks = [object()]
    result = namespace["from_upstream"](original)
    assert result.hf_text_config is text_config
    assert result.head_dim == 72
    assert result.hf_model_id == "local-test"
    assert result._pre_wrap_hooks == original._pre_wrap_hooks
    assert result._pre_wrap_hooks is not original._pre_wrap_hooks
    original.unrecognized_conversion_value = 1
    with pytest.raises(ValueError, match="Undeclared"):
        namespace["from_upstream"](original)
