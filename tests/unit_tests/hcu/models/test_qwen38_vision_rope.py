# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Standalone rotary checks, runnable without importing GPU-initializing TE.

Load the actual operator and pure MCore reference functions from this worktree.
CPU checks do not substitute a fake TE device or claim GPU integration coverage.
"""

import __future__

import ast
import importlib.util
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from torch.utils.checkpoint import checkpoint


ROOT = Path(__file__).resolve().parents[4]
BRIDGE = ROOT / "3rdparty/Megatron-Bridge"
MODEL = BRIDGE / "src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl"
SPEC = importlib.util.spec_from_file_location(
    "worktree_vision_rope", ROOT / "hcu_megatron_bridge/src/hcu_megatron/bridge/models/qwen38_vl/vision_rope.py"
)
rope = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rope)
HCU = ROOT / "hcu_megatron_bridge/src/hcu_megatron/bridge/models/qwen38_vl"
pytestmark = pytest.mark.unit


def _load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


reference = _load_functions(
    BRIDGE / "3rdparty/Megatron-LM/megatron/core/models/common/embeddings/rope_utils.py",
    {"_rotate_half", "_apply_rotary_pos_emb_bshd"},
    {"torch": torch, "Tensor": torch.Tensor, "Optional": Optional},
)["_apply_rotary_pos_emb_bshd"]


def _angles(tokens, device="cpu"):
    generator = torch.Generator().manual_seed(1234)
    half = torch.randn(tokens, 1, 1, 36, generator=generator).to(device)
    return torch.cat((half, half), dim=-1)


def _reference(x, angles):
    y = x[:, None] if x.ndim == 3 else x
    y = reference(y.float(), angles).to(x.dtype)
    return y.squeeze(1) if x.ndim == 3 else y


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [1, 7, 65])
@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_cached_matches_actual_mcore_output_and_input_gradient(dtype, tokens, thd, strided):
    torch.manual_seed(1234)
    storage = torch.randn(tokens, 3, 8, 72, dtype=dtype)
    x = storage[:, 0].detach().requires_grad_() if strided else storage[:, 0].clone().requires_grad_()
    if not thd:
        x = x.unsqueeze(1).detach().requires_grad_()
    original = x.detach().clone()
    angles = _angles(tokens)
    cos, sin = rope.prepare_vision_rotary_cos_sin(angles)
    expected = _reference(x, angles)
    actual = rope.apply_vision_rotary_pos_emb(x, cos=cos, sin=sin, backend="cached")
    grad = torch.randn_like(expected)
    actual_grad = torch.autograd.grad(actual, x, grad)[0]
    expected_grad = torch.autograd.grad(expected, x, grad)[0]
    tolerance = 1e-6 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(x, original, atol=0, rtol=0)
    assert actual.shape == x.shape and actual.dtype == dtype
    assert actual.data_ptr() != x.data_ptr()


def test_tables_are_fp32_token_aligned_and_microbatch_local():
    first = _angles(12)
    permutation = torch.tensor([8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7])
    cos, sin = rope.prepare_vision_rotary_cos_sin(first)
    reordered_cos, reordered_sin = rope.prepare_vision_rotary_cos_sin(first[permutation])
    torch.testing.assert_close(reordered_cos, cos[permutation], atol=0, rtol=0)
    torch.testing.assert_close(reordered_sin, sin[permutation], atol=0, rtol=0)
    changed_cos, _ = rope.prepare_vision_rotary_cos_sin(first + 0.7)
    assert not torch.equal(changed_cos, cos)
    assert cos.dtype == sin.dtype == torch.float32
    assert cos.shape == sin.shape == (12, 36)
    assert cos.is_contiguous() and sin.is_contiguous()


def test_trig_is_computed_once_for_27_layers_and_both_qk(monkeypatch):
    calls = {"cos": 0, "sin": 0}
    for name in calls:
        original = getattr(torch.Tensor, name)

        def counted(tensor, _name=name, _original=original):
            calls[_name] += 1
            return _original(tensor)

        monkeypatch.setattr(torch.Tensor, name, counted)
    cos, sin = rope.prepare_vision_rotary_cos_sin(_angles(7))
    x = torch.randn(7, 8, 72, dtype=torch.bfloat16)
    for _ in range(27):
        for _ in range(2):
            rope.apply_vision_rotary_pos_emb(x, cos=cos, sin=sin, backend="cached")
    assert calls == {"cos": 1, "sin": 1}


@pytest.mark.parametrize("backend", ["unknown", "FLASH", ""])
def test_invalid_backend_is_rejected(backend):
    with pytest.raises(ValueError, match="vision_rope_backend"):
        rope.validate_vision_rope_backend(backend)


def test_flash_does_not_silently_fallback_on_cpu():
    cos, sin = rope.prepare_vision_rotary_cos_sin(_angles(7))
    with pytest.raises(ValueError, match="CUDA/HIP"):
        rope.apply_vision_rotary_pos_emb(torch.randn(7, 8, 72), cos=cos, sin=sin, backend="flash")


def test_incompatible_flash_kernel_abi_is_rejected(monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules, "flash_attn.ops.triton.rotary", SimpleNamespace(rotary_kernel=SimpleNamespace(arg_names=[]))
    )
    rope._get_flash_rotary_kernel.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="ABI"):
            rope.validate_vision_rope_backend("flash")
    finally:
        rope._get_flash_rotary_kernel.cache_clear()


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="Run separately with TRITON_INTERPRET=1")
@pytest.mark.parametrize("tokens", [7, 65])
def test_actual_flash_kernel_with_cpu_interpreter(tokens):
    """Check real kernel indexing and backward; this is not a GPU compile/timing test."""
    # DTK's interpreter cast_impl passes rounding_mode=None for BF16 stores,
    # which truncates instead of GPU round-to-nearest-even. BF16 parity belongs
    # in the real-device test below; do not loosen its tolerance to mask this.
    dtype = torch.float32
    torch.manual_seed(1234)
    x = torch.randn(1, tokens, 3, 8, 72, dtype=dtype)[:, :, 0].detach().requires_grad_()
    angles = _angles(tokens)
    cos, sin = rope.prepare_vision_rotary_cos_sin(angles)
    actual = rope._FlashVisionRotary.apply(x, cos, sin)
    expected = _reference(x.squeeze(0), angles).unsqueeze(0)
    grad = torch.randn(1, tokens, 8, 144, dtype=dtype)[..., ::2]
    actual_grad = torch.autograd.grad(actual, x, grad)[0]
    expected_grad = torch.autograd.grad(expected, x, grad)[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("failure", ["batch", "length", "dtype", "trainable"])
def test_shared_table_contract(failure):
    x = torch.randn(7, 1, 8, 72)
    cos, sin = rope.prepare_vision_rotary_cos_sin(_angles(7))
    if failure == "batch":
        x = x.expand(7, 2, 8, 72)
    elif failure == "length":
        cos, sin = cos[:6], sin[:6]
    elif failure == "dtype":
        cos, sin = cos.bfloat16(), sin.bfloat16()
    else:
        cos.requires_grad_()
    with pytest.raises(ValueError):
        rope.apply_vision_rotary_pos_emb(x, cos=cos, sin=sin, backend="cached")


@pytest.mark.parametrize("method", ["uniform", "block"])
@pytest.mark.parametrize("fp8_checkpoint", [False, True])
def test_actual_vision_checkpoint_method_preserves_tables_and_gradients(method, fp8_checkpoint):
    """Run the modified method with CPU layers and real PyTorch recomputation."""
    tree = ast.parse((HCU / "block_methods.py").read_text())
    vision_class = tree
    function = next(
        node
        for node in vision_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_checkpointed_forward"
    )
    globals_ = {
        "Tensor": torch.Tensor,
        "Optional": Optional,
        "PackedSeqParams": object,
        "torch": torch,
        "nullcontext": nullcontext,
    }

    def mcore_checkpoint(fn, distribute, *args):
        return checkpoint(fn, *args, use_reentrant=True)

    def te_checkpoint(fn, distribute, rng, tp, *args):
        return checkpoint(fn, *args, use_reentrant=True)

    globals_.update(
        tensor_parallel=SimpleNamespace(
            checkpoint=mcore_checkpoint, random=SimpleNamespace(get_cuda_rng_tracker=None)
        ),
        te_checkpoint=te_checkpoint,
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(MODEL / "transformer_block.py"), "exec"), globals_)
    angles = _angles(7)
    cos, sin = rope.prepare_vision_rotary_cos_sin(angles)
    packed = object()
    calls = []

    class Layer:
        def __init__(self, index):
            self.layer_number = index + 1

        def __call__(self, **kwargs):
            # Reentrant checkpoint detaches tensor objects but preserves storage.
            assert kwargs["rotary_pos_cos"].data_ptr() == cos.data_ptr()
            assert kwargs["rotary_pos_sin"].data_ptr() == sin.data_ptr()
            assert kwargs["packed_seq_params"] is packed
            calls.append(self.layer_number)
            out = rope.apply_vision_rotary_pos_emb(kwargs["hidden_states"], cos=cos, sin=sin, backend="cached")
            return out, kwargs["context"]

    layers = [Layer(i) for i in range(3)]
    block = SimpleNamespace(
        config=SimpleNamespace(
            fp8=fp8_checkpoint, distribute_saved_activations=False, recompute_method=method, recompute_num_layers=2
        ),
        num_layers_per_pipeline_rank=3,
        deepstack_visual_indexes=[],
        tp_group=None,
        _get_layer=lambda i: layers[i],
    )
    x = torch.randn(7, 1, 8, 72, requires_grad=True)
    actual, features = globals_["_checkpointed_forward"](
        block, x, None, None, None, None, cos, sin, None, packed, False
    )
    actual.backward(torch.ones_like(actual))
    actual_grad = x.grad.clone()
    x.grad = None
    expected = x
    for _ in layers:
        expected = _reference(expected, angles)
    expected_grad = torch.autograd.grad(expected, x, torch.ones_like(expected))[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-6)
    assert features == [] and len(calls) > len(layers)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("backend", ["reference", "cached", "flash"])
def test_actual_attention_forward_preserves_metadata_and_language_path(packed, backend):
    """Exercise the actual attention method around small CPU QKV/projection doubles."""
    namespace = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "Optional": Optional,
        "Qwen3VLTransformerConfig": object,
        "_apply_rotary_pos_emb_bshd": reference,
    }
    _load_functions(
        MODEL / "rope.py", {"apply_rotary_pos_emb_absolute", "apply_rotary_pos_emb_thd_absolute"}, namespace
    )
    tree = ast.parse((HCU / "attention_methods.py").read_text())
    attention_class = tree
    function = next(
        node for node in attention_class.body if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    namespace.update(
        deprecate_inference_params=lambda current, deprecated: current,
        nvtx_range_push=lambda **kwargs: None,
        nvtx_range_pop=lambda **kwargs: None,
        DotProductAttention=type("LocalAttention", (), {}),
        apply_vision_rotary_pos_emb=rope.apply_vision_rotary_pos_emb,
    )
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(MODEL / "attention.py"),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    config = SimpleNamespace(
        no_rope_freq=None,
        flash_decode=False,
        enable_cuda_graph=False,
        attention_output_gate=False,
        vision_rope_backend=backend,
        rotary_interleaved=False,
        apply_rotary_pos_emb_in_fp32=True,
        apply_rope_fusion=False,
    )
    lengths = torch.tensor([0, 3, 7], dtype=torch.int32)
    metadata = (
        SimpleNamespace(
            qkv_format="thd",
            cu_seqlens_q=lengths,
            cu_seqlens_kv=lengths,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
        )
        if packed
        else None
    )
    angles = _angles(7)
    cos, sin = rope.prepare_vision_rotary_cos_sin(angles)
    x = torch.randn(7, 3, 8, 72, requires_grad=True)
    seen = []

    def core(q, k, v, mask, **kwargs):
        assert kwargs["packed_seq_params"] is metadata
        seen.append(q)
        result = q + 2 * k + 3 * v
        return result if packed else result.flatten(2)

    attention = SimpleNamespace(
        config=config,
        layer_number=1,
        training=True,
        checkpoint_core_attention=False,
        get_query_key_value_tensors=lambda hidden, key_value: (hidden[:, 0:1], hidden[:, 1:2], hidden[:, 2:3]),
        _adjust_key_value_for_inference=lambda ctx, q, k, v, emb, *args: (q, k, v, emb, None, None),
        core_attention=core,
        linear_proj=lambda context: (context, None),
    )
    # A provider's vision backend must not redirect language M-RoPE when raw
    # angles (and no shared tables) are supplied, even when it says 'flash'.
    raw, _ = namespace["forward"](attention, x, None, rotary_pos_emb=angles, packed_seq_params=metadata)
    if backend == "cached":
        cached, _ = namespace["forward"](
            attention,
            x,
            None,
            rotary_pos_cos=cos,
            rotary_pos_sin=sin,
            packed_seq_params=metadata,
        )
        torch.testing.assert_close(cached, raw, atol=1e-6, rtol=1e-6)
        expected_grad = torch.autograd.grad(raw, x, torch.ones_like(raw))[0]
        actual_grad = torch.autograd.grad(cached, x, torch.ones_like(cached))[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-6)
    assert seen[0].shape == ((7, 8, 72) if packed else (7, 1, 8, 72))


def test_actual_vision_forward_prepares_tables_once_per_microbatch():
    """Check model-entry preparation and packed ordering without constructing TE."""
    source = HCU / "vision_methods.py"
    tree = ast.parse(source.read_text())
    model_class = tree
    forward = next(node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    setup = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_vision_forward_packed_attention_setup"
    )
    preparations = []

    def prepare(angles):
        preparations.append(angles)
        return rope.prepare_vision_rotary_cos_sin(angles)

    namespace = {"torch": torch, "prepare_vision_rotary_cos_sin": prepare}
    exec(
        compile(
            ast.Module(body=[setup, forward], type_ignores=[]),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    angles = _angles(16)
    packed = object()
    grid = torch.tensor([[1, 2, 4], [1, 4, 2]])
    x = torch.randn(16, 72, requires_grad=True)

    def run(backend):
        def decoder(**kwargs):
            assert kwargs["packed_seq_params"] is packed
            hidden = kwargs["hidden_states"].unsqueeze(2)
            for _ in range(27):
                if backend == "reference":
                    assert kwargs["rotary_pos_cos"] is kwargs["rotary_pos_sin"] is None
                    hidden = _reference(hidden, kwargs["rotary_pos_emb"])
                else:
                    assert kwargs["rotary_pos_emb"] is None
                    hidden = rope.apply_vision_rotary_pos_emb(
                        hidden, cos=kwargs["rotary_pos_cos"], sin=kwargs["rotary_pos_sin"], backend=backend
                    )
            return hidden.squeeze(2), []

        model = SimpleNamespace(
            input_tensor=None,
            config=SimpleNamespace(vision_rope_backend=backend),
            spatial_merge_size=2,
            patch_embed=lambda data: data,
            fast_pos_embed_interpolate=lambda g: torch.zeros_like(x),
            rot_pos_emb=lambda g: angles[:, 0, 0, :36],
            _uses_vision_cuda_graph=lambda: False,
            build_packed_seq_params=lambda g: packed,
            decoder=decoder,
            merger=lambda hidden: hidden[::4, 0],
        )
        return namespace["forward"](model, x, grid)[0]

    expected = run("reference")
    assert preparations == []
    actual = run("cached")
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    expected_grad = torch.autograd.grad(expected, x, torch.ones_like(expected))[0]
    actual_grad = torch.autograd.grad(actual, x, torch.ones_like(actual))[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=3e-6)
    assert len(preparations) == 1
    angles.add_(0.4)
    assert not torch.equal(run("cached"), actual)
    assert len(preparations) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device unavailable")
@pytest.mark.parametrize("backend", ["cached", "flash"])
@pytest.mark.parametrize("recompute", [None, "uniform", "block"])
@pytest.mark.parametrize("frozen", [False, True])
def test_real_vision_encoder_parity(tmp_path, backend, recompute, frozen):
    """Exercise actual patch ordering, packed attention, merger and parameter grads."""
    import torch.distributed as dist
    from hcu_megatron.bridge.models.qwen38_vl.provider import HCUQwen35VLModelProvider as Qwen35VLModelProvider
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    assert not dist.is_initialized(), "Run this single-device test in its own pytest process"
    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        torch.cuda.set_device(0)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(1234)
        hf_vision = Qwen3_5VisionConfig(
            depth=2,
            hidden_size=144,
            intermediate_size=256,
            num_heads=2,
            out_hidden_size=128,
            num_position_embeddings=64,
            deepstack_visual_indexes=[],
        )
        provider = Qwen35VLModelProvider(
            num_layers=64,
            hidden_size=5120,
            ffn_hidden_size=17408,
            num_attention_heads=24,
            vision_config=hf_vision,
            sequence_parallel=False,
            recompute_granularity="full" if recompute else None,
            recompute_method=recompute,
            recompute_num_layers=1 if recompute else None,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        expected_model = (
            build_module(provider.build_vision_encoder_spec(), pg_collection=pg_collection).cuda().bfloat16().train()
        )
        provider.vision_rope_backend = backend
        actual_model = (
            build_module(provider.build_vision_encoder_spec(), pg_collection=pg_collection).cuda().bfloat16().train()
        )
        actual_model.load_state_dict(expected_model.state_dict())
        for model in (expected_model, actual_model):
            model.requires_grad_(not frozen)
        # Non-square image plus two frames: preserve three independent sequences.
        grid = torch.tensor([[1, 4, 6], [2, 2, 4]], device="cuda", dtype=torch.long)
        data = torch.randn(40, 1536, device="cuda", dtype=torch.bfloat16)
        expected_input = data.detach().clone().requires_grad_(not frozen)
        actual_input = data.detach().clone().requires_grad_(not frozen)
        expected, _ = expected_model(expected_input, grid)
        actual, _ = actual_model(actual_input, grid)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        if not frozen:
            grad = torch.randn_like(expected)
            expected.backward(grad)
            actual.backward(grad)
            torch.testing.assert_close(actual_input.grad, expected_input.grad, atol=1e-2, rtol=1e-2)
            for name, param in actual_model.named_parameters():
                torch.testing.assert_close(
                    param.grad, dict(expected_model.named_parameters())[name].grad, atol=2e-2, rtol=2e-2
                )
        else:
            assert all(param.grad is None for param in actual_model.parameters())
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens", [7, 65, 1568])
def test_real_flash_forward_backward_matches_reference(dtype, tokens):
    rope.validate_vision_rope_backend("flash")
    x = torch.randn(tokens, 3, 8, 72, device="cuda", dtype=dtype)[:, 0].detach().requires_grad_()
    original = x.detach().clone()
    angles = _angles(tokens, "cuda")
    cos, sin = rope.prepare_vision_rotary_cos_sin(angles)
    expected = _reference(x, angles)
    actual = rope.apply_vision_rotary_pos_emb(x, cos=cos, sin=sin, backend="flash")
    grad = torch.randn_like(x)
    actual_grad = torch.autograd.grad(actual, x, grad)[0]
    expected_grad = torch.autograd.grad(expected, x, grad)[0]
    tolerance = 1e-6 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(x, original, atol=0, rtol=0)
