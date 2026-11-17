#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Compare local UltraChat Parquet bytes with official Hugging Face LFS hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import urllib.request
from pathlib import Path

REPO = "HuggingFaceH4/ultrachat_200k"
API = f"https://huggingface.co/api/datasets/{REPO}?blobs=true"
logger = logging.getLogger(__name__)


def verify(source: Path, metadata: dict) -> dict:
    """Hash all data shards, README and ancillary repository metadata."""
    if metadata.get("id") != REPO or not metadata.get("sha"):
        raise ValueError("Expected official dataset metadata with a pinned commit")
    expected = [item for item in metadata["siblings"] if item["rfilename"].endswith(".parquet")]
    if len(expected) != 8:
        raise ValueError("Expected eight official UltraChat shards")
    names = {item["rfilename"] for item in expected}
    actual = {str(path.relative_to(source)) for path in (source / "data").glob("*.parquet")}
    rows = []
    for item in expected + [item for item in metadata["siblings"] if item["rfilename"] in ("README.md", ".gitattributes")]:
        path = source / item["rfilename"]
        algorithm = "sha256" if "lfs" in item else "git_blob_sha1"
        expected_hash = item["lfs"]["sha256"] if "lfs" in item else item["blobId"]
        digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
        size = path.stat().st_size if path.is_file() else None
        if size is not None:
            if algorithm == "git_blob_sha1":
                digest.update(f"blob {size}\0".encode())
            with path.open("rb") as handle:
                while block := handle.read(8 * 1024 * 1024):
                    digest.update(block)
        actual_hash = digest.hexdigest() if size is not None else None
        match = size == item["size"] and actual_hash == expected_hash
        rows.append({"file": item["rfilename"], "size": size, "expected_size": item["size"],
                     "algorithm": algorithm, "hash": actual_hash, "expected_hash": expected_hash, "match": match})
        logger.info("%s: %s", item["rfilename"], "MATCH" if match else "MISMATCH")
    return {"repo": REPO, "commit": metadata["sha"], "metadata_url": API,
            "source": str(source.resolve()), "file_list_matches": names == actual,
            "data_and_readme_match": names == actual and all(row["match"] for row in rows if row["file"] != ".gitattributes"),
            "files": rows}


def main() -> None:
    """Fetch metadata only (no dataset download), then stream local hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="Previously fetched official API response")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.metadata:
        metadata = json.loads(args.metadata.read_text())
    else:
        with urllib.request.urlopen(API, timeout=45) as response:
            metadata = json.load(response)
    result = verify(args.source.resolve(), metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["data_and_readme_match"]:
        raise SystemExit("Official data/README comparison failed; inspect the report")


if __name__ == "__main__":
    main()
