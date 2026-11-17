#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Estimate persistent BF16/FP32 Adam state per GPU; not total peak memory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from .convert_qwen38_vl_to_text import expected_shapes
except ImportError:
    from convert_qwen38_vl_to_text import expected_shapes


def estimate(config: dict, *, world_size: int, tp: int, pp: int,
             gpu_memory_mib: int = 65520) -> dict:
    """Count dense HF tensors, treating shared norm weights as TP replicas.

    Assumes uniform PP, untied embedding/head, CP1, FP32 gradients/master
    parameters/Adam moments, and the pinned MCore distributed optimizer.
    Activations, communication, padding, allocator and transient buffers are
    excluded. DP shards 12 bytes of state, not the persistent 6-byte buffers.
    """
    config = config.get("text_config", config)
    layers = config["num_hidden_layers"]
    if config.get("model_type") != "qwen3_5_text" or config.get("tie_word_embeddings", False):
        raise ValueError("Expected dense untied Qwen3.5 text config")
    if min(world_size, tp, pp) <= 0 or world_size % (tp * pp) or layers % pp:
        raise ValueError("Invalid world/TP/PP divisibility")
    for field in ("num_attention_heads", "num_key_value_heads", "linear_num_key_heads", "linear_num_value_heads"):
        if config[field] % tp:
            raise ValueError(f"TP must divide {field}")
    dp = world_size // (tp * pp)
    stages = [{"pp_rank": rank, "tp_sharded_parameters": 0, "tp_replicated_parameters": 0}
              for rank in range(pp)]
    mtp_layers = config.get("mtp_num_hidden_layers", 0)
    shapes = expected_shapes(config, layers, mtp_num_layers=mtp_layers)
    for key, shape in shapes.items():
        if key.startswith("model.layers."):
            stage = int(key.split(".")[2]) // (layers // pp)
        elif key == "model.embed_tokens.weight":
            stage = 0
        else:
            stage = pp - 1
        count = math.prod(shape)
        if key.endswith("norm.weight") or key in ("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"):
            stages[stage]["tp_replicated_parameters"] += count
        else:
            if count % tp:
                raise ValueError(f"Non-divisible tensor: {key}")
            stages[stage]["tp_sharded_parameters"] += count
    # The default MTP stage needs an input embedding when PP separates it from stage 0.
    embedding_replica = bool(mtp_layers and pp > 1)
    if embedding_replica:
        stages[-1]["tp_sharded_parameters"] += config["vocab_size"] * config["hidden_size"]
    bytes_per_parameter = 6 + 12 / dp
    for stage in stages:
        local = stage["tp_replicated_parameters"] + stage["tp_sharded_parameters"] // tp
        stage.update(local_parameters=local, state_only_gib=local * bytes_per_parameter / 1024**3,
                     remaining_before_activations_gib=gpu_memory_mib / 1024 - local * bytes_per_parameter / 1024**3)
    return {"world_size": world_size, "tp": tp, "pp": pp, "dp": dp, "layers": layers,
            "mtp_num_layers": mtp_layers, "mtp_embedding_replica_assumed": embedding_replica,
            "unique_parameters": sum(math.prod(shape) for shape in shapes.values()),
            "bytes_per_parameter": bytes_per_parameter, "gpu_capacity_gib": gpu_memory_mib / 1024,
            "max_state_only_gib": max(stage["state_only_gib"] for stage in stages), "stages": stages}


def main() -> None:
    """Write standard single/two-node comparisons from the local model config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads((args.model / "config.json").read_text())
    result = {"model": str(args.model.resolve()), "excludes": ["activations", "communication buffers", "temporary allocations", "allocator reserve", "checkpoint staging"],
              "cases": [estimate(config, world_size=world, tp=tp, pp=pp)
                        for world, tp, pp in ((8, 2, 4), (8, 4, 2), (16, 2, 4), (16, 4, 4))]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
