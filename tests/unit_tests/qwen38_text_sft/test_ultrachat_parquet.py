"""Check native Parquet splits and byte-level official-source verification."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from data_preparation.prepare_ultrachat_sft import prepare, split_files
from tools.verify_ultrachat_parquet import REPO, verify

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


def conversation(question, answer="An answer."):
    return {"prompt": question, "prompt_id": question,
            "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}


def write_parquet(directory, split, index, total, rows):
    path = directory / f"{split}-{index:05d}-of-{total:05d}-0000000000000000.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


@unittest.skipUnless(pa is not None, "Run Parquet tests in the existing pyarrow container")
class ParquetTests(unittest.TestCase):
    def test_native_messages_text_and_official_split_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "raw/data"
            data.mkdir(parents=True)
            write_parquet(data, "train_sft", 0, 2, [conversation("Train\nquestion", "中文; literal \\n and \\theta")])
            write_parquet(data, "train_sft", 1, 2, [conversation("Duplicate"), conversation("Duplicate", "Other")])
            write_parquet(data, "test_sft", 0, 1, [conversation("Duplicate"), conversation("Validation")])
            write_parquet(data, "train_gen", 0, 1, [conversation("Excluded")])
            result = prepare(data.parent, root / "out", max_train_samples=0, max_validation_samples=0)
            self.assertEqual(result["counts"]["training"]["written"], 2)
            self.assertEqual(result["counts"]["validation"]["written"], 1)
            self.assertEqual(result["counts"]["validation"]["duplicate_or_overlap"], 1)
            training = [json.loads(line) for line in (root / "out/training.jsonl").read_text().splitlines()]
            self.assertEqual(training[0], {"messages": conversation("Train\nquestion", "中文; literal \\n and \\theta")["messages"]})
            self.assertNotIn("Excluded", str(training))
            with self.assertRaises(ValueError):
                prepare(data.parent, root / "out")

    def test_missing_shard_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_parquet(root, "train_sft", 0, 2, [conversation("Question")])
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                split_files(root, "train_sft")

    def test_token_filter_and_generation_only_directory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "raw"
            data.mkdir()
            write_parquet(data, "train_gen", 0, 1, [conversation("Excluded")])
            with self.assertRaises(FileNotFoundError):
                split_files(data, "train_sft")
            write_parquet(data, "train_sft", 0, 1, [conversation("Too long"), conversation("Keep")])
            write_parquet(data, "test_sft", 0, 1, [conversation("Validation")])
            result = prepare(data, root / "out", checker=lambda row: None if row["messages"][0]["content"] == "Too long" else {"supervised_tokens": 4})
            self.assertEqual(result["counts"]["training"]["token_filtered"], 1)
            self.assertTrue(result["token_checked"])


class IntegrityTests(unittest.TestCase):
    def test_hash_corruption_extra_file_and_ancillary_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            siblings = []
            for index in range(8):
                name = f"data/shard-{index}.parquet"
                content = b"PAR1" + bytes([index]) + b"PAR1"
                (root / name).write_bytes(content)
                siblings.append({"rfilename": name, "size": len(content), "lfs": {"sha256": hashlib.sha256(content).hexdigest()}})
            for name in ("README.md", ".gitattributes"):
                content = b"original"
                (root / name).write_bytes(content)
                siblings.append({"rfilename": name, "size": len(content), "blobId": hashlib.sha1(b"blob 8\0" + content).hexdigest()})
            metadata = {"id": REPO, "sha": "pinned-revision", "siblings": siblings}
            self.assertTrue(verify(root, metadata)["data_and_readme_match"])
            (root / ".gitattributes").write_text("local metadata")
            report = verify(root, metadata)
            self.assertTrue(report["data_and_readme_match"])
            self.assertFalse(report["files"][-1]["match"])
            (root / "data/shard-0.parquet").write_bytes(b"corrupt")
            self.assertFalse(verify(root, metadata)["data_and_readme_match"])
            (root / "data/extra.parquet").write_bytes(b"extra")
            self.assertFalse(verify(root, metadata)["file_list_matches"])


if __name__ == "__main__":
    unittest.main()
