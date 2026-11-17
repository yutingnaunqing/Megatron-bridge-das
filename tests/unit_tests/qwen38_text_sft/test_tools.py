"""CPU-only artifact tests; runnable with standard-library unittest."""

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data_preparation.prepare_qwen38_text_sft import chat_template_for_mode, make_token_checker, messages_row, prepare
from tools.checkpoint.convert_qwen38_vl_to_text import convert, expected_shapes, read_header

ROOT = Path(__file__).resolve().parents[3]


def make_source(path, *, mtp=False):
    config = {"model_type": "qwen3_5_text", "hidden_size": 8, "intermediate_size": 12,
              "vocab_size": 16, "head_dim": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
              "linear_num_key_heads": 1, "linear_num_value_heads": 2, "linear_key_head_dim": 2,
              "linear_value_head_dim": 2, "linear_conv_kernel_dim": 4,
              "num_hidden_layers": 8, "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
              "tie_word_embeddings": False, "bos_token_id": 0, "eos_token_id": 0}
    (path / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"],
                                                 "text_config": config, "vision_config": {}}))
    shapes = expected_shapes(config, 8, mtp_num_layers=int(mtp))
    shapes = {(key.replace("model.", "model.language_model.", 1) if key.startswith("model.") else key): shape
              for key, shape in shapes.items()}
    shapes.update({"model.visual.test": [4]})
    if not mtp:
        shapes["mtp.test"] = [4]
    header, body, cursor = {"__metadata__": {"format": "pt"}}, bytearray(), 0
    import math

    for index, (key, shape) in enumerate(shapes.items()):
        size = math.prod(shape) * 2
        header[key] = {"shape": shape, "dtype": "BF16", "data_offsets": [cursor, cursor + size]}
        body.extend(bytes([index % 255]) * size)
        cursor += size
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "model.safetensors" for key in shapes}}))
    (path / "tokenizer.json").write_text(json.dumps({"added_tokens": [{"id": 2, "content": "<|im_end|>"}]}))
    (path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))
    (path / "chat_template.jinja").write_text("{{ messages }}")


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        make_source(self.source)
        self.output = self.root / "text"

    def test_conversion_bytes_keys_shapes_hashes_and_source_unchanged(self):
        original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()}
        manifest = convert(self.source, self.output, 4, max_shard_bytes=1000)
        self.assertEqual(len(manifest["tensors"]), 56)
        self.assertEqual(manifest["config"]["layer_types"], ["linear_attention"] * 3 + ["full_attention"])
        self.assertEqual(manifest["config"]["eos_token_id"], 2)
        index = json.loads((self.output / "model.safetensors.index.json").read_text())["weight_map"]
        self.assertGreater(len(set(index.values())), 1)
        for entry in manifest["tensors"]:
            self.assertNotIn("language_model", entry["key"])
            header, base = read_header(self.output / index[entry["key"]])
            value = header[entry["key"]]
            start, end = value["data_offsets"]
            with (self.output / index[entry["key"]]).open("rb") as handle:
                handle.seek(base + start)
                actual = handle.read(end - start)
            with (self.source / entry["source_file"]).open("rb") as handle:
                handle.seek(entry["source_offset"])
                self.assertEqual(actual, handle.read(entry["bytes"]))
            self.assertEqual(hashlib.sha256(actual).hexdigest(), entry["sha256"])
        self.assertEqual(original, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()})
        self.assertEqual({item["reason"] for item in manifest["excluded"]}, {"vision", "mtp", "truncated_layer"})

    def test_dry_run_does_not_create_output(self):
        self.assertEqual(len(convert(self.source, self.output, 4, dry_run=True)["tensors"]), 56)
        self.assertFalse(self.output.exists())

    def test_mtp_conversion_retains_exact_tensors_and_language_hashes(self):
        make_source(self.source, mtp=True)
        baseline = convert(self.source, self.root / "baseline", 4)
        enabled = convert(self.source, self.output, 4, mtp_num_layers=1)
        self.assertEqual(enabled["config"]["mtp_num_hidden_layers"], 1)
        self.assertEqual(len(enabled["tensors"]), 71)
        mtp = [row for row in enabled["tensors"] if row["key"].startswith("mtp.")]
        self.assertEqual(len(mtp), 15)
        self.assertTrue(all(row["source_key"] == row["key"] for row in mtp))
        hashes = {row["key"]: row["sha256"] for row in enabled["tensors"]}
        self.assertEqual({row["key"]: row["sha256"] for row in baseline["tensors"]},
                         {key: value for key, value in hashes.items() if not key.startswith("mtp.")})
        index = json.loads((self.output / "model.safetensors.index.json").read_text())["weight_map"]
        for row in mtp:
            header, base = read_header(self.output / index[row["key"]])
            start, end = header[row["key"]]["data_offsets"]
            with (self.output / index[row["key"]]).open("rb") as handle:
                handle.seek(base + start)
                self.assertEqual(hashlib.sha256(handle.read(end - start)).hexdigest(), row["sha256"])

    def test_mtp_unknown_keys_and_mode_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected tensor"):
            convert(self.source, self.output, 4, mtp_num_layers=1)
        make_source(self.source, mtp=True)
        convert(self.source, self.output, 4, mtp_num_layers=1)
        data = self.root / "data"
        data.mkdir()
        (data / "training.jsonl").write_text(json.dumps({"messages": [
            {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}) + "\n")
        env = {**os.environ, "LOCAL_HF_PATH": str(self.output), "DATA_PATH": str(data),
               "TP": "1", "PP": "4", "GPUS": "4", "NNODES": "1", "SAVE_DIR": str(self.root / "run"),
               "PYTHON_BIN": sys.executable, "HOSTFILE": "", "RESUME_FROM": "",
               "EXPECTED_NUM_LAYERS": "4", "EVAL_ITERS": "0",
               "MTP_NUM_LAYERS": "1", "MTP_LOSS_SCALING_FACTOR": "0.1"}
        command = ["bash", str(ROOT / "examples/qwen/run_qwen38_text_sft.sh"), "preflight"]
        good = subprocess.run(command, env=env, text=True, capture_output=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        manifest = json.loads(good.stdout)
        self.assertEqual(manifest["mtp_num_layers"], 1)
        for overrides in ({"MTP_NUM_LAYERS": "0"}, {"MTP_NUM_LAYERS": "2"},
                          {"MTP_LOSS_SCALING_FACTOR": "nan"}):
            bad = subprocess.run(command, env={**env, **overrides}, capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
        run = self.root / "run"
        checkpoints = run / "checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "latest_checkpointed_iteration.txt").write_text("5\n")
        (run / "launch_manifest.json").write_text(json.dumps(manifest))
        resumed = subprocess.run(command, env={**env, "RESUME_FROM": str(checkpoints)}, capture_output=True)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        bad = subprocess.run(command, env={**env, "RESUME_FROM": str(checkpoints),
                                          "MTP_LOSS_SCALING_FACTOR": "0.2"}, capture_output=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn(b"Resume mismatch: mtp_loss_scaling_factor", bad.stderr)

    def test_source_existing_and_nested_output_rejected(self):
        for output in (self.source, self.source / "nested", self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                convert(self.source, output, 4)

    def test_missing_and_unknown_keys_rejected(self):
        path = self.source / "model.safetensors.index.json"
        index = json.loads(path.read_text())
        del index["weight_map"]["lm_head.weight"]
        path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "Index must cover|Missing language"):
            convert(self.source, self.output, 4)
        index["weight_map"]["unexpected.weight"] = "model.safetensors"
        path.write_text(json.dumps(index))
        with self.assertRaises(ValueError):
            convert(self.source, self.output, 4)

    def test_shape_mismatch_rejected_before_output(self):
        path = self.source / "config.json"
        config = json.loads(path.read_text())
        config["text_config"]["vocab_size"] += 1
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            convert(self.source, self.output, 4)
        self.assertFalse(self.output.exists())

    def test_missing_shard_and_bad_size_rejected(self):
        path = self.source / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "length"):
            convert(self.source, self.output, 4)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            convert(self.source, self.output, 4)

    def test_incomplete_layer_block_rejected(self):
        with self.assertRaises(ValueError):
            convert(self.source, self.output, 3)

    def test_failure_removes_partial_output(self):
        with patch("tools.checkpoint.convert_qwen38_vl_to_text.write_shard", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                convert(self.source, self.output, 4)
        self.assertFalse(self.output.exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["source"])

    def test_launcher_preflight_and_invalid_topology(self):
        convert(self.source, self.output, 4)
        data = self.root / "data"
        data.mkdir()
        (data / "training.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": "Hi"},
                                                                    {"role": "assistant", "content": "Hello"}]}) + "\n")
        env = {**os.environ, "LOCAL_HF_PATH": str(self.output), "DATA_PATH": str(data), "TP": "1",
               "PP": "4", "GPUS": "4", "NNODES": "1", "SAVE_DIR": str(self.root / "run"),
               "PYTHON_BIN": sys.executable, "HOSTFILE": "", "RESUME_FROM": "",
               "EXPECTED_NUM_LAYERS": "4", "EVAL_ITERS": "0"}
        command = ["bash", str(ROOT / "examples/qwen/run_qwen38_text_sft.sh"), "preflight"]
        result = subprocess.run(command, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["layers"], 4)
        self.assertEqual(json.loads(result.stdout)["kernel_backend"], "native")
        # Multi-node preflight is local and does not need a rendezvous address.
        dual = subprocess.run(command, env={**env, "NNODES": "2"}, text=True, capture_output=True)
        self.assertEqual(dual.returncode, 0, dual.stderr)
        self.assertEqual(json.loads(dual.stdout)["dp"], 2)
        # Resume retains the checked model, topology and dataset metadata.
        run = self.root / "run"
        checkpoints = run / "checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "latest_checkpointed_iteration.txt").write_text("5\n")
        manifest = json.loads(result.stdout)
        (run / "launch_manifest.json").write_text(json.dumps(manifest))
        resumed_env = {**env, "RESUME_FROM": str(checkpoints)}
        resumed = subprocess.run(command, env=resumed_env, text=True, capture_output=True)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        (checkpoints / "latest_checkpointed_iteration.txt").unlink()
        for overrides in ({"TP": "2"}, {"GLOBAL_BATCH_SIZE": "7", "GPUS": "8"},
                          {"RESUME_FROM": str(self.root / "absent")}, {"EXPECTED_NUM_LAYERS": "64"}):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(subprocess.run(command, env={**env, **overrides}, capture_output=True).returncode, 0)


class DataTests(unittest.TestCase):
    def test_token_checker_releases_files_before_temporary_cleanup(self):
        closed = []
        paths = []

        class IndexedFile:
            def __init__(self, path):
                self.handle = open(path)

            def __del__(self):
                self.handle.close()
                closed.append(True)

        class ChatDataset:
            def __init__(self, file_path, **kwargs):
                paths.append(Path(file_path))
                self.indexed_dataset = IndexedFile(file_path)

            def _process_example(self, row):
                raise RuntimeError("in-memory preprocessing reached")

        modules = {
            "megatron.bridge.data.datasets.gpt_sft": SimpleNamespace(GPTSFTChatDataset=ChatDataset),
            "megatron.bridge.training.tokenizers.config": SimpleNamespace(TokenizerConfig=lambda **kwargs: kwargs),
            "megatron.bridge.training.tokenizers.tokenizer": SimpleNamespace(build_tokenizer=lambda config: object()),
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            model = Path(directory)
            (model / "chat_template.jinja").write_text("{{ messages }}")
            checker = make_token_checker(model, 4096, "disabled")
            self.assertEqual(closed, [True])
            self.assertFalse(paths[0].parent.exists())
            with self.assertRaisesRegex(RuntimeError, "in-memory preprocessing reached"):
                checker({"messages": []})

    def test_raw_dapo_rejected_and_teacher_response_accepted(self):
        row = {"prompt": [{"role": "user", "content": "What is 1+1?"}],
               "reward_model": {"ground_truth": "2"}}
        with self.assertRaisesRegex(ValueError, "Raw DAPO"):
            messages_row(row)
        row["verified_solution"] = "Adding one and one gives two. Answer: 2"
        self.assertEqual(messages_row(row, solution_field="verified_solution")["messages"][-1]["content"], row["verified_solution"])

    def test_vision_and_missing_answer_rejected(self):
        for content in ("<|image_pad|>", [{"type": "image", "image": "abc"}]):
            with self.assertRaises(ValueError):
                messages_row({"messages": [{"role": "user", "content": content},
                                            {"role": "assistant", "content": "answer"}]})
        with self.assertRaises(ValueError):
            messages_row({"messages": [{"role": "user", "content": "Hi"}]})

    def test_thinking_template_binding(self):
        from jinja2 import Environment

        template = "{{ enable_thinking | default(true) }}"
        self.assertEqual(chat_template_for_mode(template, "default"), template)
        self.assertEqual(Environment().from_string(chat_template_for_mode(template, "disabled")).render().strip(), "False")
        with self.assertRaises(ValueError):
            chat_template_for_mode(template, "wrong")

    def test_deduplicate_before_split_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "raw.jsonl", root / "prepared"
            rows = [{"messages": [{"role": "user", "content": f"Q{i}"},
                                   {"role": "assistant", "content": f"A{i}"}]} for i in range(5)]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows[:2] + rows))
            manifest = prepare(source, output, limit=4, validation_samples=1)
            self.assertEqual(manifest["duplicate"], 2)
            train = set((output / "training.jsonl").read_text().splitlines())
            validation = set((output / "validation.jsonl").read_text().splitlines())
            self.assertEqual(len(train), 4)
            self.assertFalse(train & validation)
            with self.assertRaises(ValueError):
                prepare(source, output)


if __name__ == "__main__":
    unittest.main()
