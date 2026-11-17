"""Integration checks in the existing ROCm training Python environment (no GPU)."""

import importlib
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

# This ROCm image's Transformer Engine queries a GPU during import, even for
# configuration-only checks. Do not fake CUDA properties to hide that limit.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Transformer Engine import requires a visible GPU")

from transformers import Qwen3_5TextConfig

MODULE = "hcu_megatron.bridge.recipes.qwen38.bw1000.qwen38_sft"


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    for name in ("EXPECTED_NUM_LAYERS", "RECOMPUTE", "OPTIMIZER_OFFLOAD_FRACTION",
                 "TP", "PP", "GLOBAL_BATCH_SIZE", "TRAIN_ITERS", "WARMUP_ITERS",
                 "MTP_NUM_LAYERS", "MTP_LOSS_SCALING_FACTOR", "TP_COMM_OVERLAP"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "text"
    path.mkdir()
    config = Qwen3_5TextConfig(num_hidden_layers=16)
    config.architectures = ["Qwen3_5ForCausalLM"]
    config.layer_types = (["linear_attention"] * 3 + ["full_attention"]) * 4
    config.mtp_num_hidden_layers = 0
    config.save_pretrained(path)
    (path / "chat_template.jinja").write_text("{{ messages }}")
    monkeypatch.setenv("LOCAL_HF_PATH", str(path))
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("SAVE_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("RESUME_FROM", raising=False)
    return path


def test_real_bridge_and_recipe(model_dir, monkeypatch):
    from megatron.bridge.models.gpt_provider import GPTModelProvider

    module = importlib.import_module(MODULE)
    cfg = module.qwen38_text_27b_sft_bw1000_bf16_config()
    assert type(cfg.model) is GPTModelProvider
    assert cfg.model.num_layers == 16
    assert cfg.model.linear_attention_freq == [1, 1, 1, 0] * 4
    assert cfg.model.position_embedding_type == "rope"
    assert cfg.model.mtp_num_layers is None
    assert not hasattr(cfg.model, "vision_config")
    assert cfg.peft is None
    assert cfg.dataset.preprocessing.loss_mode == "assistant"
    assert not cfg.dataset.enable_offline_packing
    assert cfg.dataset.dataset_kwargs["pad_to_max_length"]
    assert cfg.model.recompute_granularity is None
    assert cfg.checkpoint.pretrained_checkpoint == str(model_dir)
    assert cfg.checkpoint.load is None
    assert cfg.checkpoint.save_optim and cfg.checkpoint.save_rng
    assert cfg.scheduler.use_checkpoint_opt_param_scheduler
    assert not cfg.scheduler.override_opt_param_scheduler
    monkeypatch.setenv("RESUME_FROM", "/saved/checkpoints")
    resumed = module.qwen38_text_27b_sft_bw1000_bf16_config()
    assert resumed.checkpoint.pretrained_checkpoint is None
    assert resumed.checkpoint.load == "/saved/checkpoints"
    assert not resumed.checkpoint.finetune
    assert resumed.checkpoint.load_optim and resumed.checkpoint.load_rng


@pytest.mark.parametrize("enabled", [False, True])
def test_tp_overlap_is_explicit_without_enabling_dp_overlap(model_dir, monkeypatch, enabled):
    monkeypatch.setenv("TP_COMM_OVERLAP", str(enabled).lower())
    module = importlib.import_module(MODULE)
    cfg = module.qwen38_text_27b_sft_bw1000_bf16_config()
    assert cfg.comm_overlap.tp_comm_overlap is enabled
    assert not cfg.comm_overlap.overlap_grad_reduce
    assert not cfg.comm_overlap.overlap_param_gather


def test_full_model_recompute_and_registration(model_dir, monkeypatch):
    path = model_dir / "config.json"
    config = json.loads(path.read_text())
    config["num_hidden_layers"] = 64
    config["layer_types"] *= 4
    path.write_text(json.dumps(config))
    family = importlib.import_module("hcu_megatron.bridge.recipes.qwen38")
    cfg = family.qwen38_text_27b_sft_bw1000_bf16_config()
    assert cfg.model.num_layers == 64
    assert cfg.model.recompute_granularity == "full"
    assert cfg.model.recompute_num_layers == 1
    assert cfg.model.recompute_method == "uniform"
    assert cfg.model.cuda_graph_impl == "none"
    assert not cfg.model.cpu_offloading
    assert not cfg.optimizer.optimizer_cpu_offload
    monkeypatch.setenv("OPTIMIZER_OFFLOAD_FRACTION", "0.5")
    cfg = family.qwen38_text_27b_sft_bw1000_bf16_config()
    assert cfg.optimizer.optimizer_cpu_offload
    assert cfg.optimizer.optimizer_offload_fraction == 0.5


def test_full_entry_rejects_reduced_checkpoint(model_dir, monkeypatch):
    monkeypatch.setenv("EXPECTED_NUM_LAYERS", "64")
    with pytest.raises(ValueError, match="layer count"):
        importlib.import_module(MODULE).qwen38_text_27b_sft_bw1000_bf16_config()


def test_vl_source_rejected(model_dir):
    path = model_dir / "config.json"
    config = json.loads(path.read_text())
    config["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Convert"):
        importlib.import_module(MODULE).qwen38_text_27b_sft_bw1000_bf16_config()


@pytest.mark.skipif(not os.environ.get("TEST_QWEN38_TOKENIZER_DIR"), reason="Set local official tokenizer directory")
def test_official_tokenizer_mask_and_collator():
    from data_preparation.prepare_qwen38_text_sft import make_token_checker

    path = Path(os.environ["TEST_QWEN38_TOKENIZER_DIR"])
    check = make_token_checker(path, 1024, "disabled")
    row = {"messages": [{"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "What is 2+2?"},
                        {"role": "assistant", "content": "The answer is 4."},
                        {"role": "user", "content": "What is 3+3?"},
                        {"role": "assistant", "content": "The answer is 6."}]}
    result = check(row)
    assert result is not None
    assert result["supervised_tokens"] > 0
    assert "answer is 4" in result["supervised_text"]
    assert "answer is 6" in result["supervised_text"]
    assert "What is" not in result["supervised_text"]
    assert "You are helpful" not in result["supervised_text"]
    assert len(result["tokens"]) == len(result["labels"]) == len(result["loss_mask"]) == 1024
    assert result["loss_mask"][-1] == 0
    huge = {"messages": [{"role": "user", "content": "long " * 4000},
                         {"role": "assistant", "content": "4"}]}
    assert check(huge) is None
