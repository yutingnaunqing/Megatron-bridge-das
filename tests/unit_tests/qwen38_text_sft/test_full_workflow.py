"""CPU-only full-model topology, memory and MPI launch checks."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.checkpoint.estimate_qwen38_text_sft_memory import estimate
from tools.checkpoint.convert_qwen38_vl_to_text import expected_shapes
import math

ROOT = Path(__file__).resolve().parents[3]


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.config = {"model_type": "qwen3_5_text", "hidden_size": 16, "intermediate_size": 32,
                       "vocab_size": 32, "head_dim": 4, "num_attention_heads": 4,
                       "num_key_value_heads": 4, "linear_num_key_heads": 4,
                       "linear_num_value_heads": 4, "linear_key_head_dim": 4,
                       "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4,
                       "num_hidden_layers": 64, "tie_word_embeddings": False,
                       "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16}

    def test_dp_shards_optimizer_states_but_not_entire_persistent_allocation(self):
        one = estimate(self.config, world_size=8, tp=2, pp=4)
        two = estimate(self.config, world_size=16, tp=2, pp=4)
        self.assertAlmostEqual(two["max_state_only_gib"] / one["max_state_only_gib"], 12 / 18)
        self.assertEqual(two["dp"], 2)

    def test_tp_replication_and_global_count(self):
        report = estimate(self.config, world_size=8, tp=4, pp=2)
        shapes = expected_shapes(self.config, 64)
        unique = sum(math.prod(shape) for shape in shapes.values())
        replicated = sum(math.prod(shape) for key, shape in shapes.items() if key.endswith("norm.weight"))
        self.assertEqual(report["unique_parameters"], unique)
        self.assertEqual(sum(stage["local_parameters"] for stage in report["stages"]) * 4,
                         unique + 3 * replicated)

    def test_invalid_topology_rejected(self):
        for world, tp, pp in ((7, 2, 4), (16, 2, 3), (8, 8, 1)):
            with self.subTest(world=world, tp=tp, pp=pp), self.assertRaises(ValueError):
                estimate(self.config, world_size=world, tp=tp, pp=pp)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for path in ("examples/qwen", "env", "bin"):
            (self.root / path).mkdir(parents=True)
        for name in ("run_qwen38_text_sft.sh", "run_qwen38_vl_sft_full.sh", "run_qwen38_text_sft_single_node.sh"):
            shutil.copy(ROOT / "examples/qwen" / name, self.root / "examples/qwen" / name)
        (self.root / "env/env.sh").write_text("")
        self.hostfile = self.root / "hosts"
        self.hostfile.write_text("node-a slots=8\nnode-b slots=8\n")
        fake_mpi = self.root / "bin/mpirun"
        fake_mpi.write_text("#!/usr/bin/env python3\nimport json,sys,os\nprint(json.dumps({'args':sys.argv[1:], 'env':{k:os.environ.get(k) for k in ['TP','PP','GLOBAL_BATCH_SIZE','EXPECTED_NUM_LAYERS','MASTER_ADDR']}}))\n")
        fake_mpi.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if k not in (
            "OMPI_COMM_WORLD_RANK", "NODE_RANK", "TP", "PP", "GLOBAL_BATCH_SIZE", "MASTER_ADDR")}
        self.env.update(PATH=str(self.root / "bin") + ":" + os.environ["PATH"],
                        NNODES="2", HOSTFILE=str(self.hostfile), DTK_ENV=str(self.root / "env/env.sh"),
                        SAVE_DIR=str(self.root / "output"))
        self.env["PRTE_TMPDIR_BASE"] = str(self.root / "prte")

    def run_launch(self):
        return subprocess.run(["bash", str(self.root / "examples/qwen/run_qwen38_text_sft.sh"), "sft"],
                              env=self.env, text=True, capture_output=True)

    def test_mpi_one_parent_per_node_and_full_model_environment(self):
        self.env.update(TP="4", PP="2", GLOBAL_BATCH_SIZE="64")
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        args = payload["args"]
        self.assertEqual(args[args.index("-np") + 1], "2")
        self.assertIn("ppr:1:node", args)
        self.assertEqual(Path(args[args.index("--hostfile") + 1]).read_text(),
                         "node-a slots=1\nnode-b slots=1\n")
        self.assertIn("-p 11463 -o BatchMode=yes -o ConnectTimeout=10", args)
        self.assertIn("EXPECTED_NUM_LAYERS", args)
        self.assertIn("RECOMPUTE", args)
        self.assertIn("OPTIMIZER_OFFLOAD_FRACTION", args)
        self.assertEqual(payload["env"], {"TP": "4", "PP": "2", "GLOBAL_BATCH_SIZE": "64",
                                          "EXPECTED_NUM_LAYERS": "64", "MASTER_ADDR": "node-a"})
        self.assertEqual("--allow-run-as-root" in args, os.geteuid() == 0)

    def test_explicit_master_address_preserved(self):
        self.env["MASTER_ADDR"] = "10.0.0.1"
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["env"]["MASTER_ADDR"], "10.0.0.1")

    def test_tp_overlap_setting_is_forwarded_to_both_nodes(self):
        for enabled in ("false", "true"):
            with self.subTest(enabled=enabled):
                self.env["TP_COMM_OVERLAP"] = enabled
                result = self.run_launch()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("TP_COMM_OVERLAP", json.loads(result.stdout)["args"])

    def test_uneven_pipeline_and_rope_settings_reach_mpi_children(self):
        keys = ["NUM_LAYERS_IN_FIRST_PIPELINE_STAGE", "NUM_LAYERS_IN_LAST_PIPELINE_STAGE",
                "APPLY_ROPE_FUSION"]
        (self.root / "bin/mpirun").write_text(
            "#!/usr/bin/env python3\nimport json,os,sys\n"
            f"keys={keys!r}\n"
            "print(json.dumps({'args':sys.argv[1:], 'env':{key:os.environ[key] for key in keys}}))\n")
        for first, last, fusion in (("20", "12", "1"), ("", "10", "false"), ("", "", "0")):
            with self.subTest(first=first, last=last, fusion=fusion):
                self.env.update(zip(keys, (first, last, fusion)))
                result = self.run_launch()
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["env"], dict(zip(keys, (
                    first, last, "true" if fusion == "1" else "false"))))
                for key in keys:
                    self.assertIn(key, payload["args"])

    def test_text_entry_without_vl_or_common_helpers(self):
        (self.root / "examples/qwen/run_qwen38_vl_sft_full.sh").unlink()
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["args"][-2:], [str(self.root / "examples/qwen/run_qwen38_text_sft.sh"), "sft"])

    def test_prepare_and_convert_are_local_in_the_unified_entry(self):
        fake_python = self.root / "bin/python-tool-probe"
        fake_python.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
        fake_python.chmod(0o755)
        env = {**self.env, "PYTHON_BIN": str(fake_python), "SEQ_LENGTH": "8192",
               "TRAIN_SAMPLES": "321", "VALIDATION_SAMPLES": "9", "EXPECTED_NUM_LAYERS": "16",
               "LOCAL_HF_PATH": str(self.root / "model"), "DATA_PATH": str(self.root / "data")}
        for mode in ("prepare", "convert"):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    ["bash", str(self.root / "examples/qwen/run_qwen38_text_sft.sh"), mode],
                    env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(result.stdout)
                if mode == "prepare":
                    self.assertTrue(args[0].endswith("prepare_ultrachat_sft.py"))
                    self.assertEqual(args[args.index("--seq-length") + 1], "8192")
                    self.assertEqual(args[args.index("--max-train-samples") + 1], "321")
                    self.assertEqual(args[args.index("--max-validation-samples") + 1], "9")
                else:
                    self.assertTrue(args[0].endswith("convert_qwen38_vl_to_text.py"))
                    self.assertEqual(args[args.index("--num-layers") + 1], "16")

    def test_single_node_entry_forces_local_16_layer_configuration(self):
        # The local entry must work without the other entries or a hostfile.
        (self.root / "examples/qwen/run_qwen38_text_sft.sh").unlink()
        (self.root / "examples/qwen/run_qwen38_vl_sft_full.sh").unlink()
        self.hostfile.unlink()
        fake_python = self.root / "bin/python-single-node-probe"
        fake_python.write_text(
            "#!/usr/bin/env python3\nimport json,os,sys\n"
            "print(json.dumps({'args':sys.argv[1:], 'env':{key:os.environ[key] for key in "
            "['NNODES','NODE_RANK','EXPECTED_NUM_LAYERS','LOCAL_HF_PATH','SEQ_LENGTH',"
            "'TP','PP','GLOBAL_BATCH_SIZE','RECOMPUTE']}}))\n")
        fake_python.chmod(0o755)
        result = subprocess.run(
            ["bash", str(self.root / "examples/qwen/run_qwen38_text_sft_single_node.sh"), "sft"],
            env={**self.env, "PYTHON_BIN": str(fake_python), "OMPI_COMM_WORLD_RANK": "1"},
            text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["env"], {
            "NNODES": "1", "NODE_RANK": "0", "EXPECTED_NUM_LAYERS": "16",
            "LOCAL_HF_PATH": "/public/home/yugeng/work/models/Qwen3.8-27B-text-16layers",
            "SEQ_LENGTH": "4096", "TP": "2", "PP": "4", "GLOBAL_BATCH_SIZE": "32",
            "RECOMPUTE": "none"})
        self.assertIn("--nnodes=1", payload["args"])
        self.assertIn("--node_rank=0", payload["args"])

    def test_profile_overrides_reach_all_three_worker_entries(self):
        fake_python = self.root / "bin/python-profile-probe"
        fake_python.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
        fake_python.chmod(0o755)
        env = {**self.env, "PYTHON_BIN": str(fake_python), "NNODES": "1", "NODE_RANK": "0",
               "PROFILE": "1", "PROFILE_STEP_START": "5", "PROFILE_STEP_END": "7",
               "PROFILE_RANKS": "[0,4]", "PROFILE_RECORD_SHAPES": "1", "PROFILE_WITH_STACK": "true"}
        for entry in ("run_qwen38_text_sft.sh", "run_qwen38_vl_sft_full.sh", "run_qwen38_text_sft_single_node.sh"):
            with self.subTest(entry=entry):
                result = subprocess.run(["bash", str(self.root / "examples/qwen" / entry), "dryrun"],
                                        env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(result.stdout)
                for option in ("profiling.use_pytorch_profiler=true", "profiling.use_nsys_profiler=false",
                               "profiling.profile_step_start=5", "profiling.profile_step_end=7",
                               "profiling.profile_ranks=[0,4]", "profiling.pytorch_profiler_collect_shapes=true",
                               "profiling.pytorch_profiler_collect_callstack=true"):
                    self.assertIn(option, args)

    def test_profile_settings_forwarded_by_both_mpi_entries(self):
        self.env.update(PROFILE="true", PROFILE_STEP_START="5", PROFILE_STEP_END="7",
                        PROFILE_RANKS="[0,8]", PROFILE_RECORD_SHAPES="false", PROFILE_WITH_STACK="false")
        for entry in ("run_qwen38_text_sft.sh", "run_qwen38_vl_sft_full.sh"):
            with self.subTest(entry=entry):
                result = subprocess.run(["bash", str(self.root / "examples/qwen" / entry), "sft"],
                                        env=self.env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(result.stdout)["args"]
                for key in ("PROFILE", "PROFILE_STEP_START", "PROFILE_STEP_END", "PROFILE_RANKS",
                            "PROFILE_RECORD_SHAPES", "PROFILE_WITH_STACK"):
                    self.assertIn(key, args)

    def test_mpi_output_is_logged_once_at_the_coordinator(self):
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        logs = list((Path(self.env["SAVE_DIR"]) / "train_logs").glob("mpi_*.log"))
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].read_text(), result.stdout)
        self.assertFalse(list(logs[0].parent.glob("node-*.log")))

    def test_mpi_failure_is_not_hidden_by_tee(self):
        (self.root / "bin/mpirun").write_text(
            "#!/usr/bin/env python3\nimport sys\nprint('MPI probe failed', file=sys.stderr)\nsys.exit(7)\n")
        result = self.run_launch()
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("MPI probe failed", result.stdout)
        logs = list((Path(self.env["SAVE_DIR"]) / "train_logs").glob("mpi_*.log"))
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].read_text(), result.stdout)

    def test_mpi_node_replaces_shell_with_agent_for_text_and_vl(self):
        fake_python = self.root / "bin/python-agent-probe"
        fake_python.write_text(
            "#!/usr/bin/env python3\nimport json,os,sys\n"
            "print(json.dumps({'pid':os.getpid(), 'args':sys.argv[1:]}))\n")
        fake_python.chmod(0o755)
        env = {**self.env, "PYTHON_BIN": str(fake_python), "OMPI_COMM_WORLD_RANK": "1"}
        for entry in ("run_qwen38_text_sft.sh", "run_qwen38_vl_sft_full.sh"):
            with self.subTest(entry=entry):
                with subprocess.Popen(
                    ["bash", str(self.root / "examples/qwen" / entry), "sft"],
                    env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
                    stdout, stderr = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 0, stderr)
                    payload = json.loads(stdout)
                    self.assertEqual(payload["pid"], process.pid)
                    self.assertEqual(payload["args"][:2], ["-m", "torch.distributed.run"])
                    self.assertIn("--node_rank=1", payload["args"])
        logs = Path(self.env["SAVE_DIR"]) / "train_logs"
        self.assertFalse(list(logs.glob("*.log")))

    def test_8192_performance_overrides_reach_mpi(self):
        (self.root / "bin/mpirun").write_text(
            "#!/usr/bin/env python3\nimport json,os\n"
            "print(json.dumps({key:os.environ[key] for key in "
            "['SEQ_LENGTH','DATA_PATH','GLOBAL_BATCH_SIZE','RECOMPUTE','TRAIN_ITERS','EVAL_ITERS','SAVE_INTERVAL']}))\n")
        result = subprocess.run(
            ["bash", str(self.root / "examples/qwen/run_qwen38_text_sft.sh"), "sft"],
            env={**self.env, "SEQ_LENGTH": "8192", "GLOBAL_BATCH_SIZE": "32", "RECOMPUTE": "full",
                 "TRAIN_ITERS": "50", "WARMUP_ITERS": "2", "EVAL_ITERS": "0", "SAVE_INTERVAL": "0",
                 "QWEN38_CACHE": str(self.root / "cache")}, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["SEQ_LENGTH"], "8192")
        self.assertTrue(payload["DATA_PATH"].endswith("seq8192-disabled"))
        self.assertEqual({key: payload[key] for key in (
            "GLOBAL_BATCH_SIZE", "RECOMPUTE", "TRAIN_ITERS", "EVAL_ITERS", "SAVE_INTERVAL")},
            {"GLOBAL_BATCH_SIZE": "32", "RECOMPUTE": "full", "TRAIN_ITERS": "50", "EVAL_ITERS": "0",
             "SAVE_INTERVAL": "0"})

    def test_training_and_dryrun_skip_all_automatic_preflight_checks(self):
        fake_python = self.root / "bin/python-probe"
        fake_python.write_text(
            "#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
        fake_python.chmod(0o755)
        output = Path(self.env["SAVE_DIR"])
        (output / "checkpoints").mkdir(parents=True)
        (output / "checkpoints/existing").write_text("existing checkpoint")
        manifest = output / "launch_manifest.json"
        manifest.write_text("previous run record\n")
        # Absent inputs and a nonempty checkpoint directory must not trigger the
        # launcher's former model/data/topology/resume/output preflight checks.
        env = {**self.env, "PYTHON_BIN": str(fake_python), "NNODES": "1",
               "NODE_RANK": "0", "LOCAL_HF_PATH": str(self.root / "absent-model"),
               "DATA_PATH": str(self.root / "absent-data"), "TP": "3", "PP": "5",
               "RESUME_FROM": str(self.root / "absent-resume")}
        for mode in ("sft", "dryrun"):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    ["bash", str(self.root / "examples/qwen/run_qwen38_text_sft.sh"), mode],
                    env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(result.stdout)
                worker = str(self.root / "examples/qwen/qwen38_text_sft_worker.py")
                self.assertIn(worker, args)
                self.assertFalse(any("check_qwen38_text_sft.py" in arg for arg in args))
                self.assertEqual("--dry-run" in args, mode == "dryrun")
                if mode == "sft":
                    self.assertEqual(args[:2], ["-m", "torch.distributed.run"])
                self.assertEqual(manifest.read_text(), "previous run record\n")
                self.assertFalse(list(output.glob("launch_manifest_node_*.json")))

    def test_custom_ssh_port_forwarded(self):
        self.env["MPI_SSH_PORT"] = "2222"
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-p 2222 -o BatchMode=yes -o ConnectTimeout=10", json.loads(result.stdout)["args"])

    def test_default_hostfile_infers_node_count(self):
        self.env.pop("HOSTFILE")
        self.env.pop("NNODES")
        (self.root / "examples/qwen/hostfile").write_text("# workers\nnode-a\nnode-b\n")
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)["args"]
        self.assertEqual(args[args.index("-np") + 1], "2")

    def test_prte_daemon_session_directory_is_explicit(self):
        result = self.run_launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)["args"]
        index = args.index("--prtemca")
        self.assertEqual(args[index + 1:index + 3],
                         ["prte_tmpdir_base", self.env["PRTE_TMPDIR_BASE"]])
        self.assertTrue(Path(self.env["PRTE_TMPDIR_BASE"]).is_dir())

    def test_vl_launch_is_independent_of_the_text_entry(self):
        (self.root / "examples/qwen/run_qwen38_text_sft.sh").unlink()
        self.env["MIN_PIXELS"] = "200704"
        result = subprocess.run(
            ["bash", str(self.root / "examples/qwen/run_qwen38_vl_sft_full.sh"), "sft"],
            env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)["args"]
        self.assertEqual(args[-2:], [str(self.root / "examples/qwen/run_qwen38_vl_sft_full.sh"), "sft"])
        self.assertIn("MIN_PIXELS", args)


if __name__ == "__main__":
    unittest.main()


class MTPMemoryTests(unittest.TestCase):
    def test_mtp_includes_tail_norm_replicas_and_input_embedding(self):
        config = {'model_type':'qwen3_5_text','hidden_size':16,'intermediate_size':32,
                  'vocab_size':32,'head_dim':4,'num_attention_heads':4,'num_key_value_heads':4,
                  'linear_num_key_heads':4,'linear_num_value_heads':4,'linear_key_head_dim':4,
                  'linear_value_head_dim':4,'linear_conv_kernel_dim':4,'num_hidden_layers':64,
                  'tie_word_embeddings':False,'layer_types':(['linear_attention']*3+['full_attention'])*16}
        baseline=estimate(config,world_size=16,tp=4,pp=4)
        enabled=estimate({**config,'mtp_num_hidden_layers':1},world_size=16,tp=4,pp=4)
        self.assertEqual(enabled['unique_parameters']-baseline['unique_parameters'],3416)
        self.assertTrue(enabled['mtp_embedding_replica_assumed'])
        self.assertEqual(enabled['stages'][0]['local_parameters'],baseline['stages'][0]['local_parameters'])
        self.assertEqual(enabled['stages'][-1]['local_parameters']-baseline['stages'][-1]['local_parameters'],1048)
