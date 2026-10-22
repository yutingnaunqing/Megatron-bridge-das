# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Check real multimodal bridge dispatch and dataset configuration offline."""

import importlib
import json

import pytest
from transformers import Qwen3_5Config


MODULE = "hcu_megatron.bridge.recipes.qwen38_vl.bw1000.qwen38_vl"


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    config = Qwen3_5Config()
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    config.text_config.num_hidden_layers = 64
    config.text_config.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    config.text_config.layer_types *= 16
    config.text_config.mtp_num_hidden_layers = 1
    config.vision_config.depth = 27
    config.save_pretrained(tmp_path)
    monkeypatch.setenv("LOCAL_HF_PATH", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SEQ_LENGTH", "1024")
    monkeypatch.setenv("MICRO_BATCH_SIZE", "1")
    monkeypatch.setenv("DATA_PATH", "/prepared/energon")
    return tmp_path


@pytest.mark.parametrize("suffix,backend", [("config", "EnergonDatasetConfig"),
                                            ("mock_config", "MockVLMSFTDatasetConfig")])
@pytest.mark.parametrize("offload", ["0", "0.25"])
def test_full_multimodal_provider(model_dir, monkeypatch, suffix, backend, offload):
    monkeypatch.setenv("OPTIMIZER_OFFLOAD_FRACTION", offload)
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider

    before = json.loads((model_dir / "config.json").read_text())
    module = importlib.import_module(MODULE)
    cfg = getattr(module, f"qwen38_vl_27b_pretrain_bw1000_bf16_{suffix}")()
    assert isinstance(cfg.model, Qwen35VLModelProvider)
    assert cfg.model.num_layers == 64
    assert cfg.model.vision_config.depth == 27
    assert cfg.model.position_embedding_type == "mrope"
    assert cfg.model.mtp_num_layers is None
    assert not cfg.model.freeze_language_model
    assert not cfg.model.freeze_vision_model
    assert not cfg.model.freeze_vision_projection
    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 2
    assert type(cfg.dataset).__name__ == backend
    assert cfg.model.seq_length == cfg.dataset.seq_length == 1024
    assert cfg.tokenizer.tokenizer_model == str(model_dir)
    if suffix == "config":
        assert cfg.dataset.task_encoder.hf_processor_path == str(model_dir)
        assert cfg.dataset.micro_batch_size == cfg.train.micro_batch_size
    else:
        assert cfg.dataset.hf_processor_path == str(model_dir)
        assert cfg.dataset.num_images == 1
    assert cfg.optimizer.optimizer_cpu_offload == (float(offload) > 0)
    assert cfg.optimizer.optimizer_offload_fraction == float(offload)
    assert cfg.checkpoint.load is None and cfg.checkpoint.save is None
    assert json.loads((model_dir / "config.json").read_text()) == before


def test_8gpu_smoke_preserves_hybrid_block_without_default_offload(model_dir, monkeypatch):
    monkeypatch.delenv("OPTIMIZER_OFFLOAD_FRACTION", raising=False)
    monkeypatch.delenv("TRAIN_ITERS", raising=False)
    before = json.loads((model_dir / "config.json").read_text())
    module = importlib.import_module("hcu_megatron.bridge.recipes.qwen38_vl")
    cfg = module.qwen38_vl_27b_pretrain_8gpu_smoke_bw1000_bf16_config()
    assert cfg.model.num_layers == cfg.model.hf_text_config.num_hidden_layers == 4
    assert cfg.model.hf_text_config.layer_types == ["linear_attention"] * 3 + ["full_attention"]
    assert cfg.model.linear_attention_freq == [1, 1, 1, 0]
    assert cfg.model.vision_config.depth == 27
    assert cfg.model.tensor_model_parallel_size * cfg.model.pipeline_model_parallel_size == 8
    assert type(cfg.dataset).__name__ == "MockVLMSFTDatasetConfig"
    assert not cfg.optimizer.optimizer_cpu_offload
    assert cfg.optimizer.optimizer_offload_fraction == 0
    assert cfg.train.train_iters == 5
    assert cfg.scheduler.lr_decay_iters is None
    assert json.loads((model_dir / "config.json").read_text()) == before
    full_cfg = module.qwen38_vl_27b_pretrain_bw1000_bf16_mock_config()
    assert full_cfg.model.num_layers == 64
    assert not full_cfg.optimizer.optimizer_cpu_offload
    assert full_cfg.optimizer.optimizer_offload_fraction == 0


def test_runner_selects_multimodal_forward_without_explicit_step(model_dir, monkeypatch):
    import run_recipe
    from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step

    captured = {}

    def capture_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_recipe, "run_config", capture_run)
    run_recipe.main([
        "--recipe", "qwen38_vl_27b_pretrain_bw1000_bf16_mock_config",
        "--mode", "pretrain", "--dry-run",
    ])
    assert captured["step_func"] is forward_step
    assert captured["mode"] == "pretrain"


@pytest.mark.parametrize("recipe_name", [
    "qwen38_vl_27b_pretrain_bw1000_bf16_config",
    "qwen38_vl_27b_pretrain_bw1000_bf16_mock_config",
    "qwen38_vl_27b_pretrain_8gpu_smoke_bw1000_bf16_config",
])
@pytest.mark.parametrize("decay_iters", [None, 100])
def test_runner_resolves_schedule_after_training_overrides(
    model_dir, monkeypatch, recipe_name, decay_iters,
):
    import run_recipe

    monkeypatch.setenv("TRAIN_ITERS", "5")
    captured = {}

    def capture_run(**kwargs):
        cfg = kwargs["config"]
        # Use the same scheduler resolution that ConfigContainer.validate calls.
        cfg._calculate_scheduler_steps()
        captured["config"] = cfg

    monkeypatch.setattr(run_recipe, "run_config", capture_run)
    args = ["--recipe", recipe_name, "--mode", "pretrain", "--dry-run",
            "train.train_iters=50"]
    if decay_iters is not None:
        args.append(f"scheduler.lr_decay_iters={decay_iters}")
    run_recipe.main(args)

    cfg = captured["config"]
    assert cfg.train.train_iters == 50
    assert cfg.scheduler.lr_decay_iters == (50 if decay_iters is None else decay_iters)
    assert cfg.scheduler.lr_decay_steps == cfg.scheduler.lr_decay_iters * cfg.train.global_batch_size
