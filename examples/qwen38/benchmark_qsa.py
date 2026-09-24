#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Measure the Qwen3.8 reference QSA workload on one HIP device.

使用示例（在项目根目录执行）：
    HIP_VISIBLE_DEVICES=0 python -u examples/qwen38/benchmark_qsa.py \
        --profile smoke --sequence-lengths 128,2048 --warmup 2 --iterations 5 \
        --profile-ops --output-dir runs/qwen38/qsa_smoke

此脚本是 QSA 算子 bring-up 基线，并非融合稀疏 QSA kernel 的性能结论。它运行当前
稠密 mask 参考实现，为后续 HCU 算子提供固定形状、数值参考和前后对比命令。
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _path in (
    ROOT,
    ROOT / "hcu_megatron_bridge/src",
    ROOT / "3rdparty/Megatron-Bridge/src",
    ROOT / "3rdparty/Megatron-Bridge/3rdparty/Megatron-LM",
    ROOT / "3rdparty/Megatron-Energon/src",
):
    sys.path.insert(0, str(_path))

import torch

from hcu_megatron.bridge.models.qwen38.config import load_text_config
from hcu_megatron.bridge.models.qwen38.model import TrainableQSA, text_positions

LOGGER = logging.getLogger(__name__)


def parse_lengths(value: str) -> list[int]:
    """Parse a comma-separated, strictly positive sequence-length list."""
    try:
        lengths = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("sequence lengths must be comma-separated integers") from error
    if not lengths or any(length < 1 or length > 4096 for length in lengths):
        raise argparse.ArgumentTypeError("each sequence length must be in 1..4096")
    if len(set(lengths)) != len(lengths):
        raise argparse.ArgumentTypeError("sequence lengths must not repeat")
    return lengths


def workload_contract(config: dict[str, Any], *, batch_size: int, sequence_length: int) -> dict[str, Any]:
    """Return QSA tensor shapes and sparse-selection arithmetic for one input."""
    if batch_size < 1 or sequence_length < 1:
        raise ValueError("batch_size and sequence_length must be positive")
    compress = config["indexer_compress_ratio"]
    budget = config["indexer_budget"]
    q_heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    index_heads = config["indexer_n_heads"]
    index_dim = config["indexer_head_dim"]
    blocks = sequence_length // compress
    topk = min(budget // compress, blocks)
    positions = torch.arange(sequence_length, dtype=torch.int64)
    complete_blocks = (positions + 1) // compress
    selected_tokens = torch.minimum(complete_blocks * compress, torch.tensor(budget))
    selected_tokens += (positions + 1) - complete_blocks * compress
    causal_tokens = positions + 1
    dense_scores = batch_size * q_heads * sequence_length * sequence_length
    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "dtype": "bfloat16 inputs and weights; fp32 index scores, attention logits and probabilities",
        "indexer": {
            "projection": [batch_size, sequence_length, (index_heads + 1) * index_dim],
            "query": [batch_size, sequence_length, index_heads, index_dim],
            "compressed_key": [batch_size, blocks, index_dim],
            "score": [batch_size, sequence_length, blocks],
            "topk_blocks_per_query_max": topk,
            "topk_block_ids": [batch_size, sequence_length, topk],
        },
        "gqa": {
            "query_and_gate": [batch_size, q_heads, sequence_length, head_dim],
            "key_value_before_grouping": [batch_size, kv_heads, sequence_length, head_dim],
            "query_heads_per_kv_head": q_heads // kv_heads,
            "partial_rope_dimension": int(head_dim * config["rope_parameters"]["partial_rotary_factor"]),
        },
        "selection": {
            "budget_tokens": budget,
            "compress_ratio": compress,
            "selected_tokens_first_query": int(selected_tokens[0]),
            "selected_tokens_last_query": int(selected_tokens[-1]),
            "selected_tokens_per_query_mean": float(selected_tokens.float().mean()),
            "selected_to_causal_token_ratio": float(selected_tokens.sum() / causal_tokens.sum()),
        },
        "dense_reference": {
            "attention_logits_and_probabilities_shape_each": [batch_size, q_heads, sequence_length, sequence_length],
            "attention_logits_and_probabilities_fp32_gib_each": dense_scores * 4 / 2**30,
            "dense_boolean_mask_shape": [batch_size, sequence_length, sequence_length],
        },
    }


def run_once(
    module: TrainableQSA,
    inputs: torch.Tensor,
    positions: tuple[torch.Tensor, torch.Tensor],
    *,
    backward: bool,
) -> float:
    """Run one QSA forward pass and optionally its autograd backward pass."""
    if backward:
        output, kl = module(inputs, positions)
        loss = output.float().square().mean() + kl.float().mean()
        loss.backward()
        return float(loss.detach())
    with torch.no_grad():
        output, kl = module(inputs, positions)
        return float(output.float().square().mean() + kl.float().mean())


def benchmark_one(
    config: dict[str, Any],
    *,
    batch_size: int,
    sequence_length: int,
    warmup: int,
    iterations: int,
    backward: bool,
    seed: int,
) -> dict[str, Any]:
    """Run one steady-state QSA measurement and return serializable metrics."""
    torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    model_config = SimpleNamespace(**config)
    qsa_layer = next(index for index, kind in enumerate(config["layer_types"]) if kind == "full_attention")
    module = TrainableQSA(model_config, qsa_layer).to(device=device, dtype=torch.bfloat16).train(backward)
    inputs = torch.randn(
        batch_size, sequence_length, config["hidden_size"], device=device, dtype=torch.bfloat16,
        requires_grad=backward,
    )
    positions = text_positions(
        model_config,
        torch.arange(sequence_length, device=device).expand(batch_size, -1),
        inputs.dtype,
    )
    for _ in range(warmup):
        run_once(module, inputs, positions, backward=backward)
        module.zero_grad(set_to_none=True)
        inputs.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    loss = 0.0
    for _ in range(iterations):
        loss = run_once(module, inputs, positions, backward=backward)
        module.zero_grad(set_to_none=True)
        inputs.grad = None
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "contract": workload_contract(config, batch_size=batch_size, sequence_length=sequence_length),
        "mode": "forward_backward" if backward else "forward_only",
        "iterations": iterations,
        "loss_last_iteration": loss,
        "elapsed_ms_per_iteration": elapsed * 1000 / iterations,
        "tokens_per_second": batch_size * sequence_length * iterations / elapsed,
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "max_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }


def capture_operator_profile(
    config: dict[str, Any],
    *,
    batch_size: int,
    sequence_length: int,
    backward: bool,
    seed: int,
    row_limit: int,
) -> str:
    """Capture a one-iteration Torch operator summary when Kineto supports HIP."""
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        torch.manual_seed(seed)
        device = torch.device("cuda", torch.cuda.current_device())
        model_config = SimpleNamespace(**config)
        qsa_layer = next(index for index, kind in enumerate(config["layer_types"]) if kind == "full_attention")
        module = TrainableQSA(model_config, qsa_layer).to(device=device, dtype=torch.bfloat16).train(backward)
        inputs = torch.randn(
            batch_size, sequence_length, config["hidden_size"], device=device, dtype=torch.bfloat16,
            requires_grad=backward,
        )
        positions = text_positions(
            model_config,
            torch.arange(sequence_length, device=device).expand(batch_size, -1),
            inputs.dtype,
        )
        torch.cuda.synchronize()
        with profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            run_once(module, inputs, positions, backward=backward)
        torch.cuda.synchronize()
        return profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=row_limit)
    except Exception as error:  # HIP profiler availability varies by container release.
        return f"Torch operator profile unavailable: {type(error).__name__}: {error}"


def main() -> None:
    """Parse options, run the baseline and save its contract plus measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("tiny", "smoke", "official"), default="smoke")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.official.json"))
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-lengths", type=parse_lengths, default=parse_lengths("128,512,2048"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--profile-ops", action="store_true")
    parser.add_argument("--profile-rows", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.iterations < 1 or args.profile_rows < 1:
        parser.error("batch-size and iterations must be positive; warmup must be non-negative")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("QSA benchmark requires an available HIP PyTorch device")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or ROOT / "runs/qwen38" / f"qsa_{args.profile}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_text_config(args.config, profile=args.profile, num_layers=args.num_layers)
    if config["indexer_budget"] % config["indexer_compress_ratio"]:
        raise ValueError("QSA token budget must be divisible by its compression ratio")
    result: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "profile": args.profile,
        "config": {
            key: config[key]
            for key in (
                "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
                "indexer_n_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio",
            )
        },
        "measurements": [],
    }
    for sequence_length in args.sequence_lengths:
        LOGGER.info("Benchmarking QSA: batch=%d sequence=%d", args.batch_size, sequence_length)
        measurement = benchmark_one(
            config, batch_size=args.batch_size, sequence_length=sequence_length,
            warmup=args.warmup, iterations=args.iterations, backward=not args.forward_only, seed=args.seed,
        )
        result["measurements"].append(measurement)
        LOGGER.info(
            "sequence=%d mode=%s %.2f ms/iter %.1f token/s max allocated %.2f GiB",
            sequence_length, measurement["mode"], measurement["elapsed_ms_per_iteration"],
            measurement["tokens_per_second"], measurement["max_memory_allocated_gib"],
        )
    if args.profile_ops:
        result["torch_operator_profile"] = capture_operator_profile(
            config, batch_size=args.batch_size, sequence_length=max(args.sequence_lengths),
            backward=not args.forward_only, seed=args.seed, row_limit=args.profile_rows,
        )
    output_path = output_dir / "qsa_benchmark.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("Wrote QSA contract and baseline to %s", output_path)


if __name__ == "__main__":
    main()
