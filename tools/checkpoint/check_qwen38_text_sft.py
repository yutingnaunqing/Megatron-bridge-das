#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Validate text checkpoint, JSONL, topology and resume intent before launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from convert_qwen38_vl_to_text import DTYPE_BYTES, expected_shapes, read_header


def file_sha256(path: Path) -> str:
    """Hash full datasets with bounded host memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def check_environment(*, check_tokens: bool = False) -> dict:
    """Fail early on incompatible artifacts or unsafe checkpoint fallback."""
    model = Path(os.environ["LOCAL_HF_PATH"]).expanduser().resolve()
    data = Path(os.environ["DATA_PATH"]).expanduser().resolve()
    config = json.loads((model / "config.json").read_text())
    if config.get("architectures") != ["Qwen3_5ForCausalLM"] or config.get("model_type") != "qwen3_5_text":
        raise ValueError("LOCAL_HF_PATH must point at a converted text checkpoint")
    layers = config["num_hidden_layers"]
    if os.environ.get("EXPECTED_NUM_LAYERS") and layers != int(os.environ["EXPECTED_NUM_LAYERS"]):
        raise ValueError("Layer count mismatch: full-model entry requires its actual HF layers")
    mtp_layers = int(os.environ.get("MTP_NUM_LAYERS", "0"))
    mtp_loss_factor = float(os.environ.get("MTP_LOSS_SCALING_FACTOR", "0.1")) if mtp_layers else 0.0
    if mtp_layers not in (0, 1) or config.get("mtp_num_hidden_layers", 0) != mtp_layers:
        raise ValueError("MTP_NUM_LAYERS must match the converted checkpoint (0 or 1)")
    if mtp_layers and (not math.isfinite(mtp_loss_factor) or mtp_loss_factor <= 0):
        raise ValueError("MTP_LOSS_SCALING_FACTOR must be finite and positive")
    if layers not in (4, 16, 64) or "vision_config" in config:
        raise ValueError("Expected text-only 4/16/64-layer checkpoint")
    if config.get("layer_types") != (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4):
        raise ValueError("Expected complete hybrid blocks")
    expected = expected_shapes(config, layers, mtp_num_layers=mtp_layers)
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    if set(index) != set(expected):
        raise ValueError("Text checkpoint key coverage mismatch")
    headers = {}
    for name in set(index.values()):
        path = (model / name).resolve()
        if path.parent != model:
            raise ValueError("Unsafe shard path")
        headers[name], _ = read_header(path)
    for key, name in index.items():
        tensor = headers[name][key]
        if tensor["shape"] != expected[key] or tensor["dtype"] not in DTYPE_BYTES:
            raise ValueError(f"Shape/dtype mismatch: {key}")
        start, end = tensor["data_offsets"]
        if end - start != math.prod(tensor["shape"]) * DTYPE_BYTES[tensor["dtype"]]:
            raise ValueError(f"Tensor byte size mismatch: {key}")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if not (model / name).is_file():
            raise FileNotFoundError(model / name)
    tp, pp, cp = (int(os.environ[name]) for name in ("TP", "PP", "CP"))
    world = int(os.environ["NNODES"]) * int(os.environ["GPUS"])
    gbs, mbs = int(os.environ["GLOBAL_BATCH_SIZE"]), int(os.environ["MICRO_BATCH_SIZE"])
    if min(tp, pp, cp, world, gbs, mbs) < 1 or cp != 1 or mbs != 1:
        raise ValueError("Initial SFT requires positive topology/batch sizes, CP1 and MBS1")
    if world % (tp * pp * cp) or layers % pp:
        raise ValueError("WORLD_SIZE must be divisible by TP*PP*CP and layers by PP")
    for field in ("num_attention_heads", "num_key_value_heads", "linear_num_key_heads", "linear_num_value_heads", "hidden_size"):
        if config[field] % tp:
            raise ValueError(f"TP does not divide {field}")
    dp = world // (tp * pp * cp)
    if gbs % (mbs * dp):
        raise ValueError("GBS must be divisible by MBS*DP")
    steps, warmup = int(os.environ["TRAIN_ITERS"]), int(os.environ["WARMUP_ITERS"])
    if steps <= 0 or not 0 <= warmup < steps or int(os.environ["SEQ_LENGTH"]) <= 0:
        raise ValueError("Require positive sequence/steps and 0 <= warmup < steps")
    if int(os.environ["SAVE_INTERVAL"]) <= 0 or float(os.environ["LR"]) <= 0:
        raise ValueError("Require positive save interval/LR")
    if int(os.environ.get("EXIT_INTERVAL", "0")) < 0:
        raise ValueError("EXIT_INTERVAL must be nonnegative")
    offload = float(os.environ.get("OPTIMIZER_OFFLOAD_FRACTION", "0"))
    if not 0.0 <= offload <= 1.0:
        raise ValueError("OPTIMIZER_OFFLOAD_FRACTION must be between zero and one")
    recompute = os.environ.get("RECOMPUTE", "full" if layers == 64 else "none")
    if recompute not in ("full", "none"):
        raise ValueError("RECOMPUTE must be full or none")
    # Validate all provided rows even in dry-run; raw RL and vision data fail here.
    from data_preparation.prepare_qwen38_text_sft import messages_row

    checker = None
    if check_tokens:
        from data_preparation.prepare_qwen38_text_sft import make_token_checker

        checker = make_token_checker(model, int(os.environ["SEQ_LENGTH"]), os.environ["THINKING_MODE"])
    counts, hashes = {}, {}
    names = ["training"] + (["validation"] if int(os.environ["EVAL_ITERS"]) > 0 else [])
    for name in names:
        count = 0
        with (data / f"{name}.jsonl").open() as handle:
            for line in handle:
                if line.strip():
                    row = messages_row(json.loads(line))
                    if checker is not None and checker(row) is None:
                        raise ValueError(f"{name} row {count + 1}: too long or no supervised tokens; prepare with --check-tokens")
                    count += 1
        if not count:
            raise ValueError(f"Empty split: {name}")
        counts[name] = count
        hashes[name] = file_sha256(data / f"{name}.jsonl")
    resume = os.environ.get("RESUME_FROM")
    if resume:
        path = Path(resume).resolve()
        if not path.is_dir() or not any((path / name).is_file() for name in
                                        ("latest_train_state.pt", "latest_checkpointed_iteration.txt", "run_config.yaml")):
            raise ValueError("RESUME_FROM must be an existing Megatron checkpoint root or iteration")
        # Our saved sidecar forbids accidental cross-model, topology or data resume.
        run_dir = path.parent.parent if path.name.startswith("iter_") else path.parent
        previous = run_dir / "launch_manifest.json"
        if not previous.is_file():
            raise ValueError("Resume requires launch_manifest.json beside checkpoints; inspect imported checkpoints manually")
        old = json.loads(previous.read_text())
        for key, value in {"mtp_num_layers": mtp_layers, "mtp_loss_scaling_factor": mtp_loss_factor}.items():
            if old.get(key, 0) != value:
                raise ValueError(f"Resume mismatch: {key}")
        for key, value in {"layers": layers, "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
                           "tp": tp, "pp": pp, "dp": dp, "data": str(data),
                           "data_sha256": hashes, "seq_length": int(os.environ["SEQ_LENGTH"]), "gbs": gbs,
                           "thinking_mode": os.environ["THINKING_MODE"]}.items():
            if old.get(key) != value:
                raise ValueError(f"Resume mismatch: {key}")
    else:
        checkpoints = Path(os.environ["SAVE_DIR"]) / "checkpoints"
        if checkpoints.exists() and any(checkpoints.iterdir()):
            raise ValueError("Fresh initialization cannot reuse a nonempty checkpoint directory")
    return {"model": str(model), "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
            "data": str(data), "layers": layers, "world_size": world,
            "tp": tp, "pp": pp, "dp": dp, "seq_length": int(os.environ["SEQ_LENGTH"]),
            "gbs": gbs, "train_iters": steps, "thinking_mode": os.environ["THINKING_MODE"],
            "resume_from": resume, "split_rows": counts, "data_sha256": hashes,
            "recompute": recompute, "kernel_backend": "native",
            "mtp_num_layers": mtp_layers, "mtp_loss_scaling_factor": mtp_loss_factor,
            "optimizer_offload_fraction": offload}


def main() -> None:
    """Check inputs, optionally writing a launch manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--check-tokens", action="store_true")
    args = parser.parse_args()
    result = check_environment(check_tokens=args.check_tokens)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
