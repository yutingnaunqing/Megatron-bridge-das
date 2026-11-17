"""Optional CPU interoperability tests using installed HF and safetensors."""

import os
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

from test_tools import make_source
from data_preparation.prepare_qwen38_text_sft import chat_template_for_mode
from tools.checkpoint.convert_qwen38_vl_to_text import convert


def test_output_readable_by_safetensors_and_hf_config():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        make_source(source)
        manifest = convert(source, root / "text", 4)
        with safetensors.safe_open(root / "text/model-00001-of-00001.safetensors", framework="pt", device="cpu") as reader:
            assert set(reader.keys()) == {item["key"] for item in manifest["tensors"]}
            assert reader.get_tensor("model.embed_tokens.weight").shape == (16, 8)
            assert reader.get_tensor("model.layers.3.self_attn.q_proj.weight").shape == (16, 8)
        config = transformers.AutoConfig.from_pretrained(root / "text", local_files_only=True)
        assert config.model_type == "qwen3_5_text"
        assert config.architectures == ["Qwen3_5ForCausalLM"]
        assert config.num_hidden_layers == 4


@pytest.mark.skipif(not os.environ.get("TEST_QWEN38_TOKENIZER_DIR"), reason="Set official tokenizer directory")
def test_official_template_thinking_and_multiturn_on_cpu():
    model = Path(os.environ["TEST_QWEN38_TOKENIZER_DIR"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, local_files_only=True)
    original = (model / "chat_template.jinja").read_text()
    messages = [{"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4."},
                {"role": "user", "content": "And 3+3?"},
                {"role": "assistant", "content": "The answer is 6."}]
    tokenizer.chat_template = chat_template_for_mode(original, "disabled")
    disabled = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert "Reasoning effort is set to" not in disabled
    assert "The answer is 4." in disabled and "The answer is 6." in disabled
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    assert len(encoded) < 1024
    assert tokenizer.eos_token == "<|im_end|>"
    tokenizer.chat_template = chat_template_for_mode(original, "enabled")
    enabled = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert "Reasoning effort is set to" in enabled
