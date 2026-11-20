#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Prepare text messages JSONL without flattening chat or inventing solutions.

Accept local JSONL/parquet. Raw DAPO prompt/reward rows are rejected unless a
full teacher response column is explicitly supplied with --solution-field.
Optional token checks use the actual Bridge chat dataset/collator on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)
VISION_MARKERS = ("<|vision_start|>", "<|image_pad|>", "<|video_pad|>")


def chat_template_for_mode(template: str, mode: str) -> str:
    """Bind thinking explicitly without requiring a change to Bridge APIs."""
    if mode == "default":
        return template
    if mode not in ("enabled", "disabled"):
        raise ValueError("thinking mode must be default, enabled, or disabled")
    return "{% set enable_thinking = " + ("true" if mode == "enabled" else "false") + " %}\n" + template


def messages_row(row: dict, *, solution_field: str | None = None) -> dict:
    """Normalize a text conversation; never treat an RL reward as a solution."""
    messages = row.get("messages") or row.get("conversation") or row.get("conversations")
    if messages is None and "reward_model" in row:
        if not solution_field:
            raise ValueError("Raw DAPO rows contain final answers, not SFT solutions; supply --solution-field with verified teacher responses")
        solution = row.get(solution_field)
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError(f"Missing nonempty teacher response: {solution_field}")
        messages = list(row["prompt"]) + [{"role": "assistant", "content": solution}]
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Expected a messages conversation with a target assistant response")
    normalized = []
    aliases = {"human": "user", "gpt": "assistant"}
    for message in messages:
        role = message.get("role", message.get("from"))
        role = aliases.get(role, role)
        content = message.get("content", message.get("value"))
        if role not in ("system", "user", "assistant") or not isinstance(content, str) or not content.strip():
            raise ValueError("This initial text-only pipeline accepts nonempty system/user/assistant strings only")
        if any(marker in content for marker in VISION_MARKERS):
            raise ValueError("Vision placeholder in text-only data")
        normalized.append({"role": role, "content": content})
    if normalized[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in normalized):
        raise ValueError("Conversation must contain a user and end with an assistant target")
    return {"messages": normalized}


def input_rows(path: Path) -> Iterator[dict]:
    """Stream local files in deterministic order; parquet needs pyarrow."""
    files = sorted(path.glob("*.parquet")) + sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No JSONL/parquet files: {path}")
    for filename in files:
        if filename.suffix == ".parquet":
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(filename).iter_batches(batch_size=256):
                yield from batch.to_pylist()
        elif filename.suffix == ".jsonl":
            with filename.open() as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        else:
            raise ValueError(f"Unsupported input: {filename}")


def make_token_checker(model: Path, seq_length: int, thinking_mode: str):
    """Use the same tokenizer, dataset preprocessing and collator as training."""
    from megatron.bridge.data.datasets.gpt_sft import GPTSFTChatDataset
    from megatron.bridge.training.tokenizers.config import TokenizerConfig
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

    template = chat_template_for_mode((model / "chat_template.jinja").read_text(), thinking_mode)
    tokenizer = build_tokenizer(TokenizerConfig(tokenizer_type="HuggingFaceTokenizer",
                                               tokenizer_model=str(model), chat_template=template))
    # Construct the real dataset once; only its in-memory preprocessing/collation
    # is needed below. Release file mappings before removing the temporary files,
    # otherwise NFS may retain .nfs files and fail with "Directory not empty".
    with tempfile.TemporaryDirectory(prefix="qwen38-token-check-") as directory:
        path = Path(directory) / "training.jsonl"
        path.write_text(json.dumps({"messages": [{"role": "user", "content": "Hi"},
                                                {"role": "assistant", "content": "Hello"}]}) + "\n")
        dataset = GPTSFTChatDataset(file_path=str(path), tokenizer=tokenizer,
                                   max_seq_length=seq_length, memmap_workers=1,
                                   use_hf_tokenizer_chat_template=True, loss_mode="assistant",
                                   pad_to_max_length=True)
        dataset.indexed_dataset = None

    def check(row: dict) -> dict | None:
        processed = dataset._process_example(row)
        if len(processed["input_ids"]) > seq_length + 1:
            return None
        batch = dataset.collate_fn([processed])
        mask = batch["loss_mask"][0].bool()
        if not mask.any():
            return None
        return {"supervised_tokens": int(mask.sum()),
                "tokens": batch["tokens"][0].tolist(), "labels": batch["labels"][0].tolist(),
                "loss_mask": batch["loss_mask"][0].tolist(),
                "supervised_text": tokenizer.detokenize(batch["labels"][0][mask].tolist())}
    check.settings = {"model": str(model.resolve()), "seq_length": seq_length, "thinking_mode": thinking_mode}
    return check


def prepare(source: Path, output: Path, *, limit: int = 1024, validation_samples: int = 64,
            seed: int = 1234, solution_field: str | None = None, checker=None) -> dict:
    """Deduplicate before splitting and publish the complete prepared directory."""
    source, output = source.resolve(), output.resolve()
    if output.exists() or source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be a new directory outside the source tree")
    if limit <= 0 or validation_samples < 0:
        raise ValueError("limit must be positive and validation_samples nonnegative")
    rows, seen, audits = [], set(), []
    counts = {"read": 0, "duplicate": 0, "token_filtered": 0}
    for raw in input_rows(source):
        counts["read"] += 1
        row = messages_row(raw, solution_field=solution_field)
        # One target per prompt for the initial smoke dataset: alternative
        # solutions to the same prompt must not cross train/validation splits.
        identity = hashlib.sha256(json.dumps(row["messages"][:-1], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if identity in seen:
            counts["duplicate"] += 1
            continue
        seen.add(identity)
        if checker is not None:
            audit = checker(row)
            if audit is None:
                counts["token_filtered"] += 1
                continue
            if len(audits) < 3:
                audits.append({"id": identity, **audit})
        rows.append(row)
        if len(rows) >= limit + validation_samples:
            break
    if len(rows) <= validation_samples:
        raise ValueError("Not enough unique valid rows for train/validation split")
    random.Random(seed).shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        splits = {"training": rows[validation_samples:], "validation": rows[:validation_samples]}
        for name, split in splits.items():
            if split:
                (temporary / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split))
        manifest = {"source": str(source), "seed": seed, **counts,
                    "train": len(splits["training"]), "validation": len(splits["validation"]),
                    "token_checked": checker is not None, "token_settings": getattr(checker, "settings", None),
                    "solution_field": solution_field}
        (temporary / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if audits:
            (temporary / "token_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        # Container root may publish into a user-owned shared worktree.
        # mkdtemp's 0700 would hide the prepared artifacts from the host user.
        temporary.chmod(0o755)
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return manifest


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--solution-field")
    parser.add_argument("--check-tokens", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--thinking-mode", choices=("default", "enabled", "disabled"), default="default")
    args = parser.parse_args()
    if args.check_tokens and (args.model is None or args.seq_length <= 0):
        parser.error("--check-tokens requires --model and positive --seq-length")
    logging.basicConfig(level=logging.INFO)
    checker = make_token_checker(args.model, args.seq_length, args.thinking_mode) if args.check_tokens else None
    logger.info("%s", prepare(args.source, args.output, limit=args.limit,
                             validation_samples=args.validation_samples, seed=args.seed,
                             solution_field=args.solution_field, checker=checker))


if __name__ == "__main__":
    main()
