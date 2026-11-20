#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Stream Qwen3.5 dense VL safetensors into a standalone text checkpoint.

Only the Python standard library is needed. Tensor bytes are copied without
deserializing them; no vision tensor data is read. --dry-run reads headers only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import struct
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)
LM_PREFIX = "model.language_model."
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def read_header(path: Path) -> tuple[dict, int]:
    """Read and validate a safetensors header without reading tensor bodies."""
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Truncated safetensors: {path}")
        size = struct.unpack("<Q", size_bytes)[0]
        if not 0 < size <= min(100_000_000, path.stat().st_size - 8):
            raise ValueError(f"Invalid header length: {path}")
        header = json.loads(handle.read(size))
    cursor = 0
    entries = sorted((v for k, v in header.items() if k != "__metadata__"),
                     key=lambda v: v["data_offsets"][0])
    for value in entries:
        start, end = value["data_offsets"]
        if start != cursor or end < start:
            raise ValueError(f"Non-contiguous tensor offsets: {path}")
        cursor = end
    if 8 + size + cursor != path.stat().st_size:
        raise ValueError(f"File length disagrees with header: {path}")
    return header, 8 + size


def expected_shapes(config: dict, num_layers: int, *, mtp_num_layers: int = 0) -> dict[str, list[int]]:
    """Enumerate the supported dense, gated Qwen3.5 language state dict."""
    h, f, v = config["hidden_size"], config["intermediate_size"], config["vocab_size"]
    d = config["head_dim"]
    q, kv = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
    vd = config["linear_num_value_heads"] * config["linear_value_head_dim"]
    kd = config["linear_num_key_heads"] * config["linear_key_head_dim"]
    vh = config["linear_num_value_heads"]
    shapes = {"model.embed_tokens.weight": [v, h], "model.norm.weight": [h],
              "lm_head.weight": [v, h]}
    for layer, kind in enumerate(config["layer_types"][:num_layers]):
        common = {"input_layernorm.weight": [h], "post_attention_layernorm.weight": [h],
                  "mlp.gate_proj.weight": [f, h], "mlp.up_proj.weight": [f, h],
                  "mlp.down_proj.weight": [h, f]}
        if kind == "linear_attention":
            common.update({"linear_attn.A_log": [vh], "linear_attn.dt_bias": [vh],
                           "linear_attn.conv1d.weight": [2 * kd + vd, 1, config["linear_conv_kernel_dim"]],
                           "linear_attn.in_proj_a.weight": [vh, h],
                           "linear_attn.in_proj_b.weight": [vh, h],
                           "linear_attn.in_proj_qkv.weight": [2 * kd + vd, h],
                           "linear_attn.in_proj_z.weight": [vd, h],
                           "linear_attn.norm.weight": [config["linear_value_head_dim"]],
                           "linear_attn.out_proj.weight": [h, vd]})
        elif kind == "full_attention":
            common.update({"self_attn.q_proj.weight": [2 * q, h],
                           "self_attn.k_proj.weight": [kv, h], "self_attn.v_proj.weight": [kv, h],
                           "self_attn.o_proj.weight": [h, q],
                           "self_attn.q_norm.weight": [d], "self_attn.k_norm.weight": [d]})
        else:
            raise ValueError(f"Unsupported layer type: {kind}")
        shapes.update({f"model.layers.{layer}.{key}": value for key, value in common.items()})
    if mtp_num_layers not in (0, 1):
        raise ValueError("Only zero or one MTP layer is supported")
    if mtp_num_layers:
        attention = expected_shapes({**config, "layer_types": ["full_attention"]}, 1)
        prefix = "model.layers.0."
        shapes.update({"mtp.layers.0." + key[len(prefix):]: shape
                       for key, shape in attention.items() if key.startswith(prefix)})
        shapes.update({"mtp.fc.weight": [h, 2 * h], "mtp.norm.weight": [h],
                       "mtp.pre_fc_norm_embedding.weight": [h], "mtp.pre_fc_norm_hidden.weight": [h]})
    return shapes


def plan_conversion(source: Path, num_layers: int, *, mtp_num_layers: int = 0) -> tuple[dict, list[dict], list[dict]]:
    """Validate the source and return text config, retained and excluded tensors."""
    source = source.resolve()
    original = json.loads((source / "config.json").read_text())
    if original.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("Expected Qwen3_5ForConditionalGeneration source")
    config = dict(original["text_config"])
    if config.get("model_type") != "qwen3_5_text" or config.get("tie_word_embeddings", False):
        raise ValueError("Only dense, untied qwen3_5_text is supported")
    if config.get("attention_bias", False) or not config.get("attn_output_gate", True):
        raise ValueError("Expected bias-free, gated attention")
    total = config["num_hidden_layers"]
    if not 0 < num_layers <= total or num_layers % 4 or len(config["layer_types"]) != total:
        raise ValueError("Choose complete four-layer blocks within source layer count")
    if config["layer_types"] != (["linear_attention"] * 3 + ["full_attention"]) * (total // 4):
        raise ValueError("Expected repeated 3 GDN + 1 full-attention layout")
    expected = expected_shapes(config, num_layers, mtp_num_layers=mtp_num_layers)
    all_expected = expected_shapes(config, total)
    weight_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {}
    for filename in sorted(set(weight_map.values())):
        path = (source / filename).resolve()
        if path.parent != source or path.suffix != ".safetensors":
            raise ValueError(f"Unsafe shard path: {filename}")
        headers[filename] = read_header(path)
    retained, excluded = [], []
    actual_keys = {key for header, _ in headers.values() for key in header if key != "__metadata__"}
    if actual_keys != set(weight_map):
        raise ValueError("Index must cover every tensor in the source shards")
    for key, filename in sorted(weight_map.items()):
        header, base = headers[filename]
        if key not in header:
            raise ValueError(f"Index key missing from shard: {key}")
        mapped = "model." + key[len(LM_PREFIX):] if key.startswith(LM_PREFIX) else key
        if not key.startswith((LM_PREFIX, "model.visual.", "mtp.")) and key != "lm_head.weight":
            raise ValueError(f"Unexpected source prefix: {key}")
        if key.startswith(LM_PREFIX) and mapped not in all_expected:
            raise ValueError(f"Unexpected language key: {key}")
        reason = None
        if key.startswith("model.visual."):
            reason = "vision"
        elif key.startswith("mtp.") and not mtp_num_layers:
            reason = "mtp"
        elif mapped.startswith("model.layers.") and int(mapped.split(".")[2]) >= num_layers:
            if int(mapped.split(".")[2]) >= total:
                raise ValueError(f"Layer outside source config: {key}")
            reason = "truncated_layer"
        elif mapped not in expected:
            raise ValueError(f"Unexpected tensor key: {key}")
        if reason:
            excluded.append({"source_key": key, "reason": reason})
            continue
        entry = header[key]
        start, end = entry["data_offsets"]
        if entry["shape"] != expected[mapped]:
            raise ValueError(f"Shape mismatch: {key}: {entry['shape']} != {expected[mapped]}")
        if entry["dtype"] not in DTYPE_BYTES or end - start != math.prod(entry["shape"]) * DTYPE_BYTES[entry["dtype"]]:
            raise ValueError(f"Unsupported dtype or invalid tensor size: {key}")
        retained.append({"source_key": key, "key": mapped, "source_file": filename,
                         "source_offset": base + start, "bytes": end - start,
                         "shape": entry["shape"], "dtype": entry["dtype"]})
    missing = set(expected) - {item["key"] for item in retained}
    if missing:
        raise ValueError(f"Missing language tensors: {sorted(missing)}")
    config.update(architectures=["Qwen3_5ForCausalLM"], num_hidden_layers=num_layers,
                  layer_types=config["layer_types"][:num_layers], full_attention_interval=4,
                  mtp_num_hidden_layers=mtp_num_layers)
    return config, retained, excluded


def write_shard(source: Path, destination: Path, entries: list[dict]) -> None:
    """Copy selected byte ranges with bounded memory and record tensor hashes."""
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for item in entries:
        header[item["key"]] = {"dtype": item["dtype"], "shape": item["shape"],
                               "data_offsets": [offset, offset + item["bytes"]]}
        offset += item["bytes"]
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with destination.open("wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for item in entries:
            digest = hashlib.sha256()
            with (source / item["source_file"]).open("rb") as handle:
                handle.seek(item["source_offset"])
                remaining = item["bytes"]
                while remaining:
                    block = handle.read(min(remaining, 8 * 1024 * 1024))
                    if not block:
                        raise ValueError(f"Truncated source: {item['source_file']}")
                    output.write(block)
                    digest.update(block)
                    remaining -= len(block)
            item["sha256"] = digest.hexdigest()


def convert(source: Path, output: Path, num_layers: int, *, max_shard_bytes: int = 2 * 1024**3,
            dry_run: bool = False, mtp_num_layers: int = 0) -> dict:
    """Convert into a new directory; atomically publish only a completed result."""
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents or output.exists():
        raise ValueError("Output must be a new directory outside the source tree")
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    config, entries, excluded = plan_conversion(source, num_layers, mtp_num_layers=mtp_num_layers)
    manifest = {"source": str(source), "source_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
                "num_layers": num_layers, "mtp_num_layers": mtp_num_layers,
                "config": config, "tensors": entries, "excluded": excluded,
                "total_size": sum(item["bytes"] for item in entries),
                "validation": "source headers, key coverage, shapes and sizes; tensor hashes during conversion"}
    if dry_run:
        return manifest
    for required in ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
        if not (source / required).is_file():
            raise FileNotFoundError(source / required)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        shards: list[list[dict]] = [[]]
        size = 0
        for item in entries:
            if shards[-1] and size + item["bytes"] > max_shard_bytes:
                shards.append([])
                size = 0
            shards[-1].append(item)
            size += item["bytes"]
        weight_map = {}
        for index, shard in enumerate(shards, 1):
            filename = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
            write_shard(source, temporary / filename, shard)
            read_header(temporary / filename)
            weight_map.update({item["key"]: filename for item in shard})
        for filename in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                         "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"):
            if (source / filename).is_file():
                shutil.copy2(source / filename, temporary / filename)
        tokenizer_config = json.loads((temporary / "tokenizer_config.json").read_text())
        tokenizer = json.loads((temporary / "tokenizer.json").read_text())
        token_ids = {item["content"]: item["id"] for item in tokenizer.get("added_tokens", [])}
        for name in ("bos_token", "eos_token", "pad_token"):
            token = tokenizer_config.get(name)
            if isinstance(token, dict):
                token = token["content"]
            if token in token_ids:
                config[f"{name}_id"] = token_ids[token]
        generation = {key: config[key] for key in ("bos_token_id", "eos_token_id", "pad_token_id") if key in config}
        manifest["config"] = config
        documents = {"config.json": config, "generation_config.json": generation,
                     "model.safetensors.index.json": {"metadata": {"total_size": manifest["total_size"]}, "weight_map": weight_map},
                     "conversion_manifest.json": manifest}
        for name, value in documents.items():
            (temporary / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        temporary.chmod(0o755)
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return manifest


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-layers", required=True, type=int, choices=(4, 16, 64))
    parser.add_argument("--max-shard-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mtp-num-layers", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = convert(args.source, args.output, args.num_layers,
                     max_shard_bytes=args.max_shard_bytes, dry_run=args.dry_run,
                     mtp_num_layers=args.mtp_num_layers)
    logger.info("%s: layers=%d tensors=%d bytes=%d excluded=%d",
                "Header check" if args.dry_run else args.output, args.num_layers,
                len(result["tensors"]), result["total_size"], len(result["excluded"]))


if __name__ == "__main__":
    main()
