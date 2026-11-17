"""Exercise the real Bridge forward and MCore MTP mask/gradient paths."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.unit


@pytest.mark.parametrize('enabled', [False, True])
def test_forward_passes_original_mask_only_when_mtp_enabled(monkeypatch, enabled):
    from megatron.bridge.training import gpt_step as step

    mask = torch.tensor([[0., 0., 1., 1., 0., 0.]])
    tokens = torch.arange(6).reshape(1, 6)
    cfg = SimpleNamespace(mtp_num_layers=1 if enabled else None, seq_length=6)
    state = SimpleNamespace(cfg=SimpleNamespace(), timers=MagicMock(), straggler_timer=MagicMock())
    groups = SimpleNamespace(cp=SimpleNamespace(size=lambda: 1))
    model = MagicMock(return_value=torch.ones(1, 6))
    monkeypatch.setattr(step, 'get_model_config', lambda model: cfg)
    monkeypatch.setattr(step, 'get_pg_collection', lambda model: groups)
    monkeypatch.setattr(step, 'get_model_chunk_vp_stage', lambda model: None)
    monkeypatch.setattr(step, 'accumulate_flops_metadata', lambda *args, **kwargs: None)
    batch = lambda *args, **kwargs: (tokens, tokens + 1, mask, None, tokens, None)
    _, returned = step._forward_step_common(state, None, model, _get_batch_fn=batch)
    assert returned is mask
    if enabled:
        assert model.call_args.kwargs['loss_mask'] is mask
    else:
        assert 'loss_mask' not in model.call_args.kwargs


def assert_real_mtp_gradient_mask(labels, mask, monkeypatch):
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, process_mtp_loss

    monkeypatch.setattr(MTPLossAutoScaler, 'main_loss_backward_scale', torch.tensor(1.))
    original = mask.clone()
    raw_loss = torch.ones_like(mask, requires_grad=True)
    hidden = torch.ones(2 * labels.size(1), labels.size(0), 2, requires_grad=True)
    observed = []

    def loss_fn(targets, logits):
        observed.append(targets.clone())
        return raw_loss

    cfg = SimpleNamespace(mtp_num_layers=1, mtp_detach_heads=False,
                          mtp_loss_scaling_factor=.1, calculate_per_token_loss=False)
    output = process_mtp_loss(hidden, labels, mask,
                              lambda *args, **kwargs: (torch.zeros(labels.size(1), labels.size(0), 16), None),
                              None, None, False, loss_fn, cfg)
    output.sum().backward()
    shifted = torch.cat((mask[:, 1:], torch.zeros_like(mask[:, :1])), dim=1)
    shifted_labels = torch.cat((labels[:, 1:], torch.zeros_like(labels[:, :1])), dim=1)
    torch.testing.assert_close(observed[0], shifted_labels, rtol=0, atol=0)
    torch.testing.assert_close(raw_loss.grad, .1 * shifted / shifted.sum().clamp(min=1))
    torch.testing.assert_close(mask, original, rtol=0, atol=0)
    assert raw_loss.grad[:, -1].eq(0).all()
    assert raw_loss.grad[shifted == 0].eq(0).all()


@pytest.mark.parametrize('mask', [[0, 0, 1, 1, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1], [0] * 8])
def test_shifted_targets_user_padding_boundary_and_empty_mask(monkeypatch, mask):
    assert_real_mtp_gradient_mask(torch.arange(8).reshape(1, 8), torch.tensor([mask], dtype=torch.float32), monkeypatch)


@pytest.mark.parametrize('messages', [
    [{'role': 'system', 'content': '请简短回答。'}, {'role': 'user', 'content': '问题：一加一？'},
     {'role': 'assistant', 'content': '答案是二。'}],
    [{'role': 'user', 'content': 'What is two plus two?'}, {'role': 'assistant', 'content': 'Four.'},
     {'role': 'user', 'content': '问题：三加三？'}, {'role': 'assistant', 'content': '答案是六。'}],
])
def test_real_tokenizer_collator_mtp_supervision(monkeypatch, messages):
    from data_preparation.prepare_qwen38_text_sft import make_token_checker

    model = Path('/public/home/yugeng/work/models/Qwen3.8-27B-text-64layers')
    if not model.is_dir():
        pytest.skip('Official local tokenizer is required')
    checked = make_token_checker(model, 256, 'disabled')({'messages': messages})
    assert checked is not None and checked['supervised_tokens'] > 0
    assert '问题：' not in checked['supervised_text']
    assert 'What is' not in checked['supervised_text']
    assert checked['loss_mask'][-1] == 0
    assert_real_mtp_gradient_mask(torch.tensor([checked['labels']]),
                                  torch.tensor([checked['loss_mask']], dtype=torch.float32), monkeypatch)


def test_mtp_audit_checks_tp_import_and_accumulated_main_grad(monkeypatch, tmp_path):
    import json
    import sys
    from safetensors.torch import save_file

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples/qwen'))
    from qwen38_mtp_audit import MTPAudit, parallel_state

    monkeypatch.setattr(parallel_state, 'get_tensor_model_parallel_rank', lambda: 1)
    monkeypatch.setenv('LOCAL_HF_PATH', str(tmp_path))
    monkeypatch.setenv('MTP_AUDIT_INTERVAL', '100')
    monkeypatch.setenv('SAVE_INTERVAL', '250')
    monkeypatch.setenv('TRAIN_ITERS', '500')
    monkeypatch.delenv('RESUME_FROM', raising=False)
    source = {'mtp.fc.weight': torch.arange(256 * 512, dtype=torch.float32).reshape(256, 512) / 10000}
    for key in ('pre_fc_norm_embedding', 'pre_fc_norm_hidden', 'norm'):
        source[f'mtp.{key}.weight'] = torch.linspace(.1, .9, 256)
    save_file(source, tmp_path / 'model.safetensors')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {key: 'model.safetensors' for key in source}}))
    model = torch.nn.Module()
    model.mtp = torch.nn.Module()
    layer = torch.nn.Module()
    model.mtp.layers = torch.nn.ModuleList([layer])
    layer.eh_proj = torch.nn.Linear(512, 128, bias=False)
    with torch.no_grad():
        layer.eh_proj.weight.copy_(source['mtp.fc.weight'][128:])
        for attribute, key in (('enorm', 'pre_fc_norm_embedding'), ('hnorm', 'pre_fc_norm_hidden'),
                               ('final_layernorm', 'norm')):
            module = torch.nn.RMSNorm(256)
            module.weight.copy_(source[f'mtp.{key}.weight'])
            layer.add_module(attribute, module)
    context = SimpleNamespace(model=[model], state=SimpleNamespace(train_state=SimpleNamespace(step=0)))
    audit = MTPAudit(tmp_path, 1)
    audit.on_train_start(context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
        parameter.main_grad = parameter.grad.clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    audit.on_train_step_end(context)
    report = json.loads((tmp_path / 'mtp_rank_1_audit.json').read_text())
    assert len(report['hf_import_sample_checks']) == 4
    assert all(report['hf_import_sample_checks'].values())
    for stats in report['steps'][0]['parameters'].values():
        assert stats['gradient_finite'] and stats['gradient_norm'] > 0
        assert stats['changed_sample_values'] > 0
    for _ in range(249):
        audit.on_train_step_end(context)
    report = json.loads((tmp_path / 'mtp_rank_1_audit.json').read_text())
    assert [row['step'] for row in report['steps']] == [1, 100, 200, 250]
    monkeypatch.setenv('RESUME_FROM', 'checkpoints')
    context.state.train_state.step = 250
    resumed = MTPAudit(tmp_path, 1)
    resumed.on_train_start(context)
    for _ in range(250):
        resumed.on_train_step_end(context)
    report = json.loads((tmp_path / 'mtp_rank_1_audit.json').read_text())
    assert [row['step'] for row in report['steps']] == [251, 300, 400, 500]
