#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Prepare messages JSONL from official UltraChat Parquet SFT splits.

Preserve Arrow's native messages list. Keep official train_sft/test_sft
boundaries, and filter complete conversations using the actual Qwen tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_qwen38_text_sft import make_token_checker, messages_row

logger = logging.getLogger(__name__)


def split_files(source: Path, split: str) -> list[Path]:
    """Select a complete official Parquet shard set for one SFT split."""
    directory = source / "data" if (source / "data").is_dir() else source
    files = sorted(directory.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split} Parquet shards under {directory}")
    pattern = re.compile(rf"{split}-(\d{{5}})-of-(\d{{5}})-[0-9a-f]+\.parquet")
    matches = [pattern.fullmatch(path.name) for path in files]
    if not all(matches):
        raise ValueError(f"Unexpected official shard filename for {split}")
    totals = {int(match[2]) for match in matches}
    if len(totals) != 1:
        raise ValueError(f"Inconsistent shard totals for {split}")
    total = totals.pop()
    if len(files) != total or {int(match[1]) for match in matches} != set(range(total)):
        raise ValueError(f"Incomplete Parquet shard set for {split}")
    return files


def rows_from_files(files: list[Path]):
    """Stream native Arrow messages; do not reinterpret or rewrite text."""
    import pyarrow.parquet as pq

    for path in files:
        parquet = pq.ParquetFile(path)
        if set(parquet.schema_arrow.names) != {"prompt", "prompt_id", "messages"}:
            raise ValueError(f"Unexpected UltraChat columns: {path}")
        for batch in parquet.iter_batches(batch_size=256):
            yield from batch.to_pylist()


def prepare(source: Path, output: Path, *, max_train_samples: int = 1024,
            max_validation_samples: int = 64, checker=None) -> dict:
    """Publish messages JSONL with official split boundaries and filter counts."""
    source, output = source.resolve(), output.resolve()
    if output.exists() or source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be a new directory outside the source tree")
    if min(max_train_samples, max_validation_samples) < 0:
        raise ValueError("Sample limits must be nonnegative; zero means all")
    selections = {name: split_files(source, split) for name, split in
                  (("training", "train_sft"), ("validation", "test_sft"))}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    seen, audit_rows, counts = set(), [], {}
    try:
        for name, files in selections.items():
            limit = max_train_samples if name == "training" else max_validation_samples
            stats = {"read": 0, "written": 0, "invalid": 0, "duplicate_or_overlap": 0, "token_filtered": 0}
            with (temporary / f"{name}.jsonl").open("w") as handle:
                for raw in rows_from_files(files):
                    stats["read"] += 1
                    try:
                        row = messages_row(raw)
                    except (ValueError, TypeError) as error:
                        stats["invalid"] += 1
                        if stats["invalid"] <= 3:
                            logger.warning("%s row %d skipped: %s", name, stats["read"], error)
                        continue
                    first_user = next(m["content"] for m in row["messages"] if m["role"] == "user")
                    identity = hashlib.sha256(first_user.strip().encode()).hexdigest()
                    if identity in seen:
                        stats["duplicate_or_overlap"] += 1
                        continue
                    audit = checker(row) if checker is not None else None
                    if checker is not None and audit is None:
                        stats["token_filtered"] += 1
                        continue
                    seen.add(identity)
                    if audit is not None and len(audit_rows) < 3:
                        audit_rows.append({"split": name, "id": identity, **audit})
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["written"] += 1
                    if limit and stats["written"] >= limit:
                        break
            if not stats["written"]:
                raise ValueError(f"No usable examples in {name}")
            counts[name] = stats
            logger.info("%s: %s", name, stats)
        manifest = {"source": str(source), "official_splits": {"training": "train_sft", "validation": "test_sft"},
                    "selected_files": {name: [{"path": str(p), "bytes": p.stat().st_size} for p in files]
                                       for name, files in selections.items()},
                    "limits": {"training": max_train_samples, "validation": max_validation_samples},
                    "counts": counts, "token_checked": checker is not None,
                    "token_settings": getattr(checker, "settings", None),
                    "deduplication": "first user text, training first; no random resplit; generation splits excluded"}
        (temporary / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        if audit_rows:
            (temporary / "token_audit.json").write_text(json.dumps(audit_rows, ensure_ascii=False, indent=2) + "\n")
        temporary.chmod(0o755)
        if output.exists():
            raise FileExistsError(output)
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return manifest


def main() -> None:
    """Prepare a small tested subset, or all rows with both limits set to zero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-validation-samples", type=int, default=64)
    parser.add_argument("--check-tokens", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--thinking-mode", choices=("default", "enabled", "disabled"), default="disabled")
    args = parser.parse_args()
    if args.check_tokens and (args.model is None or args.seq_length <= 0):
        parser.error("--check-tokens requires --model and positive --seq-length")
    logging.basicConfig(level=logging.INFO)
    checker = make_token_checker(args.model, args.seq_length, args.thinking_mode) if args.check_tokens else None
    prepare(args.source, args.output, max_train_samples=args.max_train_samples,
            max_validation_samples=args.max_validation_samples, checker=checker)


if __name__ == "__main__":
    main()
