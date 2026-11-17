"""Check worker exit evidence without loading Torch or the training environment."""

import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


class WorkerExitTests(unittest.TestCase):
    def run_worker(self, code):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = {
                "torch": SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)),
                "run_recipe": SimpleNamespace(main=lambda: sys.exit(code)),
                "megatron.bridge": SimpleNamespace(AutoBridge=SimpleNamespace(load_hf_weights=lambda: None)),
                "megatron.bridge.models.gpt_provider": SimpleNamespace(
                    GPTModelProvider=SimpleNamespace(provide=lambda: None)
                ),
            }
            env = {"SAVE_DIR": str(root), "RANK": "-1"}
            env.update({name: str(root / name) for name in
                        ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR")})
            with patch.dict(sys.modules, modules), patch.dict(os.environ, env):
                with self.assertRaises(SystemExit) as observed:
                    runpy.run_path(str(ROOT / "examples/qwen/qwen38_text_sft_worker.py"), run_name="__main__")
            self.assertEqual(observed.exception.code, code)
            return json.loads((root / "rank_-1_final.json").read_text())

    def test_normal_exit_interval_is_success(self):
        for code in (None, 0):
            with self.subTest(code=code):
                row = self.run_worker(code)
                self.assertTrue(row["success"])
                self.assertEqual(row["exit_code"], 0)

    def test_nonzero_exit_is_failure(self):
        row = self.run_worker(1)
        self.assertFalse(row["success"])
        self.assertEqual(row["exit_code"], 1)

    def test_inductor_cache_is_isolated_by_host_and_local_rank_in_both_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for worker in ("qwen38_text_sft_worker.py", "qwen38_vl_sft_worker.py"):
                seen = set()
                for host in ("node-a", "node-b"):
                    for rank in (0, 1):
                        with self.subTest(worker=worker, host=host, rank=rank):
                            observed = []

                            def runner():
                                observed.append(os.environ["TORCHINDUCTOR_CACHE_DIR"])
                                raise SystemExit(0)

                            modules = {
                                "torch": SimpleNamespace(cuda=SimpleNamespace(
                                    is_initialized=lambda: False, set_device=lambda index: None)),
                                "run_recipe": SimpleNamespace(main=runner),
                                "megatron.bridge": SimpleNamespace(AutoBridge=SimpleNamespace(load_hf_weights=lambda: None)),
                                "megatron.bridge.models.gpt_provider": SimpleNamespace(
                                    GPTModelProvider=SimpleNamespace(provide=lambda: None)),
                            }
                            env = {"SAVE_DIR": str(root), "RANK": "-1", "LOCAL_RANK": str(rank),
                                   "MTP_NUM_LAYERS": "0"}
                            env.update({key: str(root / key) for key in (
                                "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR")})
                            with patch.dict(sys.modules, modules), patch.dict(os.environ, env), \
                                    patch("socket.gethostname", return_value=host):
                                with self.assertRaises(SystemExit):
                                    runpy.run_path(str(ROOT / "examples/qwen" / worker), run_name="__main__")
                            expected = root / "TORCHINDUCTOR_CACHE_DIR" / host / f"rank-{rank}"
                            self.assertEqual(observed, [str(expected)])
                            self.assertTrue(expected.is_dir())
                            seen.update(observed)
                self.assertEqual(len(seen), 4)
