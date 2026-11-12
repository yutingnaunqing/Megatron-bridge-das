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

"""Audit Conv3d vision outputs and gradients with fixed weights and inputs.

Uses one unchanged model and switches only opt-in computation backends. This
is numerical validation, not a timing benchmark or a downstream quality eval.
"""

import argparse
import hashlib
import io
import json
import logging
import os
import tarfile
from pathlib import Path

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    """Measure tensor-scale errors, retaining exact equality separately from tolerance."""
    a, r = actual.detach().float(), reference.detach().float()
    difference = a - r
    # Float64 reductions avoid inaccurate long FP32 dot products (including
    # cosine != 1 for identical multi-million-element vectors).
    r_square = r.square().sum(dtype=torch.float64)
    a_square = a.square().sum(dtype=torch.float64)
    d_square = difference.square().sum(dtype=torch.float64)
    denominator = (a_square * r_square).sqrt()
    cosine = (
        None if denominator.item() == 0 else ((a_square + r_square - d_square) / (2 * denominator)).clamp(-1, 1).item()
    )
    unequal = (actual != reference).sum().item()
    same_bits = torch.equal(actual.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8))
    return {
        "numel": actual.numel(),
        "dtype": str(actual.dtype),
        "bitwise_equal": same_bits,
        "unequal": unequal,
        "unequal_fraction": unequal / actual.numel(),
        "max_abs": difference.abs().max().item(),
        "rms_abs": (d_square / actual.numel()).sqrt().item(),
        "relative_l2": (d_square / r_square.clamp_min(1e-60)).sqrt().item(),
        "cosine": cosine,
        "reference_rms": (r_square / actual.numel()).sqrt().item(),
        "reference_max": r.abs().max().item(),
    }


def compare_gradients(actual: dict, reference: dict) -> dict:
    """Report each local TP parameter shard and aggregate squared error."""
    per_parameter = {}
    error_square, reference_square = 0.0, 0.0
    for name, expected in reference.items():
        value = actual[name]
        if expected is None or value is None:
            if value is not expected:
                raise AssertionError(f"Gradient presence differs for {name}")
            continue
        result = metrics(value, expected)
        per_parameter[name] = result
        error_square += result["rms_abs"] ** 2 * result["numel"]
        reference_square += result["reference_rms"] ** 2 * result["numel"]
    return {
        "all_bitwise_equal": all(row["bitwise_equal"] for row in per_parameter.values()),
        "global_relative_l2": (error_square / max(reference_square, 1e-60)) ** 0.5,
        "worst_parameter_relative_l2": max(per_parameter, key=lambda key: per_parameter[key]["relative_l2"]),
        "parameters": per_parameter,
    }


def real_pixels(hf_path: Path, shard: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Read the first real two-image sample with the project's restricted decoder."""
    from PIL import Image
    from transformers import AutoImageProcessor

    from megatron.bridge.utils.safe_pickle import safe_pickle_loads

    with tarfile.open(shard) as archive:
        member = next(member for member in archive if member.name.endswith(".jpgs"))
        raw = archive.extractfile(member).read()
    images = [Image.open(io.BytesIO(value)).convert("RGB") for value in safe_pickle_loads(raw)]
    processor = AutoImageProcessor.from_pretrained(hf_path, local_files_only=True)
    processed = processor(images=images, return_tensors="pt", min_pixels=200704, max_pixels=1003520)
    return (
        processed["pixel_values"],
        processed["image_grid_thw"],
        {
            "shard": str(shard),
            "member": member.name,
            "sample_sha256": hashlib.sha256(raw).hexdigest(),
            "image_sizes": [image.size for image in images],
            "min_pixels": 200704,
            "max_pixels": 1003520,
        },
    )


def main() -> int:
    """Execute paired forward and backward comparisons on the full configured ViT."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--patch-graph",
        action="store_true",
        help="Also validate frozen Conv3d Graph replay",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    from hcu_megatron.bridge.models.qwen38_vl.adapter import attach_vision_optimizations, detach_vision_optimizations
    from hcu_megatron.bridge.models.qwen38_vl.provider import HCUQwen35VLModelProvider as Qwen35VLModelProvider
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        hf = json.loads((args.hf_path / "config.json").read_text())
        provider = Qwen35VLModelProvider(
            vision_hcu_enabled=False,
            num_layers=64,
            hidden_size=5120,
            ffn_hidden_size=17408,
            num_attention_heads=24,
            sequence_parallel=False,
            tensor_model_parallel_size=dist.get_world_size(),
            vision_config=Qwen3_5VisionConfig(**hf["vision_config"]),
        )
        model = (
            build_module(
                provider.build_vision_encoder_spec(),
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )
            .cuda()
            .bfloat16()
            .train()
        )
        logger.info(
            "rank=%d built full ViT, depth=%d local_parameters=%d",
            rank,
            len(model.decoder.layers),
            sum(p.numel() for p in model.parameters()),
        )
        generator = torch.Generator(device="cuda").manual_seed(9876)
        pixel_values, real_grid, input_info = real_pixels(args.hf_path, args.shard)
        payload = {
            "rank": rank,
            "tp": dist.get_world_size(),
            "vision_config": hf["vision_config"],
            "weights": "fixed seeded initialization; one model, no optimizer updates or pretrained load",
            "input_info": input_info,
            "scenarios": {},
        }
        variants = {
            "reference": ("reference", "reference"),
            "reference_repeat": ("reference", "reference"),
            "rope_only": ("flash", "reference"),
            "rope_metadata": ("flash", "prepared"),
        }
        pairs = {
            "reference_repeat": ["reference"],
            "rope_only": ["reference"],
            "rope_metadata": ["rope_only", "reference"],
        }
        patch_graph = None
        if args.patch_graph:
            from hcu_megatron.bridge.models.qwen38_vl.vision_graph import (
                FrozenPatchGraph,
            )

            patch_graph = FrozenPatchGraph(
                model.patch_embed,
                token_counts=[pixel_values.shape[0], 40],
                max_entries=2,
                max_bytes=128 * 1024 * 1024,
                capture_calls=192,
                warmup_calls=3,
                capture_error_mode=provider.vision_graph_capture_error_mode,
            )
            variants["rope_metadata_graph"] = ("flash", "prepared")
            variants["rope_metadata_graph_repeat"] = ("flash", "prepared")
            pairs["rope_metadata_graph"] = ["rope_metadata"]
            pairs["rope_metadata_graph_repeat"] = ["rope_metadata"]
        scenarios = [
            (
                "real_images_frozen",
                pixel_values.to(device="cuda", dtype=torch.bfloat16),
                real_grid.cuda(),
                False,
            ),
            (
                "multiframe_trainable",
                torch.randn(40, 1536, generator=generator, device="cuda", dtype=torch.bfloat16),
                torch.tensor([[1, 4, 6], [2, 2, 4]], device="cuda"),
                True,
            ),
        ]
        for scenario, pixels, grid, backward in scenarios:
            logger.info("rank=%d scenario=%s tokens=%d", rank, scenario, pixels.shape[0])
            model.requires_grad_(backward)
            saved = {}
            rows = {
                "tokens": pixels.shape[0],
                "grids": grid.cpu().tolist(),
                "backward": backward,
                "comparisons": {},
            }
            upstream = torch.randn(
                pixels.shape[0] // 4,
                hf["vision_config"]["out_hidden_size"],
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for variant, (rope, metadata) in variants.items():
                if variant in ("reference", "reference_repeat"):
                    detach_vision_optimizations(model)
                elif not getattr(model, "_hcu_vision_attached", False):
                    provider.vision_hcu_enabled = True
                    attach_vision_optimizations(model, provider)
                if getattr(model, "_hcu_vision_attached", False):
                    model.patch_graph = patch_graph if "graph" in variant else None
                if getattr(model, "_hcu_vision_attached", False):
                    model.config.vision_rope_backend = rope
                    model.config.vision_metadata_backend = metadata
                    for layer in model.decoder.layers:
                        layer.self_attention.config.vision_rope_backend = rope
                model.zero_grad(set_to_none=True)
                stages = {}

                def capture(name):
                    def hook(module, inputs, output):
                        value = output[0] if isinstance(output, tuple) else output
                        stages[name] = value.detach().cpu().clone()

                    return hook

                # CPU snapshot hooks cannot run inside a GPU graph capture.
                handles = [] if "graph" in variant else [model.patch_embed.register_forward_hook(capture("patch"))]
                handles += [
                    layer.register_forward_hook(capture(f"layer_{i:02d}"))
                    for i, layer in enumerate(model.decoder.layers)
                ]
                x = pixels.detach().clone().requires_grad_(backward)
                try:
                    with torch.set_grad_enabled(backward):
                        output, _ = model(x, grid)
                    snapshot = {
                        "output": output.detach().cpu().clone(),
                        "stages": stages,
                    }
                    if backward:
                        output.backward(upstream)
                        snapshot["input_grad"] = x.grad.detach().cpu().clone()
                        snapshot["parameter_grads"] = {
                            name: None if value.grad is None else value.grad.detach().cpu().clone()
                            for name, value in model.named_parameters()
                        }
                    saved[variant] = snapshot
                    for baseline in pairs.get(variant, []):
                        ref = saved[baseline]
                        comparison = {
                            "output": metrics(snapshot["output"], ref["output"]),
                            "stages": {
                                key: metrics(value, ref["stages"][key]) for key, value in snapshot["stages"].items()
                            },
                        }
                        if backward:
                            comparison["input_grad"] = metrics(snapshot["input_grad"], ref["input_grad"])
                            comparison["parameter_grads"] = compare_gradients(
                                snapshot["parameter_grads"], ref["parameter_grads"]
                            )
                        rows["comparisons"][variant + "_vs_" + baseline] = comparison
                        logger.info(
                            "rank=%d %s %s vs %s output=%s",
                            rank,
                            scenario,
                            variant,
                            baseline,
                            comparison["output"],
                        )
                    payload["scenarios"][scenario] = rows
                    (args.output / f"rank-{rank}.json").write_text(json.dumps(payload, indent=2))
                    # Keep only the snapshots needed by later pair comparisons.
                    if variant == "reference_repeat":
                        del saved[variant]
                    del output, x
                finally:
                    for handle in handles:
                        handle.remove()
                dist.barrier()
        if patch_graph is not None:
            payload["patch_graph"] = patch_graph.stats()
            assert payload["patch_graph"]["captures"] > 0
            assert payload["patch_graph"]["replays"] >= 2
            patch_graph.clear()
        payload["status"] = "complete"
        (args.output / f"rank-{rank}.json").write_text(json.dumps(payload, indent=2))
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
