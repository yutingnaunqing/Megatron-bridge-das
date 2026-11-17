# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Record MTP imports, real accumulated gradients, and parameter updates."""
import importlib
import json
import os
from pathlib import Path

import torch
from megatron.core import parallel_state
from megatron.core.utils import unwrap_model
from megatron.bridge.training.callbacks import Callback
from safetensors import safe_open


class MTPAudit(Callback):
    """Audit representative MTP parameters without changing training gradients."""

    def __init__(self, output: Path, rank: int):
        self.output = output
        self.rank = rank
        self.parameters = {}
        self.initial = {}
        self.steps = []
        self.source_checks = {}
        self.start_step = 0
        self.completed_steps = 0
        self.audit_interval = int(os.environ.get("MTP_AUDIT_INTERVAL", "1"))
        self.audit_steps = {int(os.environ.get(key, "0")) for key in
                            ("SAVE_INTERVAL", "TRAIN_ITERS", "EXIT_INTERVAL")}

    def on_train_start(self, context):
        self.start_step = int(context.state.train_state.step)
        suffixes = {
            "eh_proj.weight": "mtp.fc.weight",
            "enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            "hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            "final_layernorm.weight": "mtp.norm.weight",
        }
        for chunk in context.model:
            model = unwrap_model(chunk)
            for name, parameter in model.named_parameters():
                if name.startswith("mtp.layers.0.") and name.removeprefix("mtp.layers.0.") in suffixes:
                    self.parameters[name] = parameter
        if not self.parameters:
            return
        for name, parameter in self.parameters.items():
            self.initial[name] = parameter.detach().flatten()[:256].float().cpu().clone()
        if not os.environ.get("RESUME_FROM"):
            source = Path(os.environ["LOCAL_HF_PATH"])
            index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            for name, parameter in self.parameters.items():
                key = suffixes[name.removeprefix("mtp.layers.0.")]
                with safe_open(source / index[key], framework="pt", device="cpu") as handle:
                    view = handle.get_slice(key)
                    if key == "mtp.fc.weight":
                        expected = view[tp_rank * parameter.shape[0], :256]
                    else:
                        expected = view[:256]
                self.source_checks[name] = torch.equal(self.initial[name], expected.float())
            if not all(self.source_checks.values()):
                raise ValueError(f"MTP HF import samples differ: {self.source_checks}")

    def on_train_step_end(self, context):
        if not self.parameters:
            return
        self.completed_steps += 1
        step = self.start_step + self.completed_steps
        if not (self.completed_steps == 1 or step % self.audit_interval == 0 or step in self.audit_steps):
            return
        stats = {}
        for name, parameter in self.parameters.items():
            gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                gradient = parameter.grad
            sample = parameter.detach().flatten()[:256].float().cpu()
            stats[name] = {
                "gradient_finite": gradient is not None and bool(torch.isfinite(gradient).all().item()),
                "gradient_norm": float(gradient.float().norm().item()) if gradient is not None else None,
                "changed_sample_values": int((sample != self.initial[name]).sum().item()),
                "max_sample_update": float((sample - self.initial[name]).abs().max().item()),
            }
        self.steps.append({"step": step, "parameters": stats})
        report = {"rank": self.rank, "source_checkpoint": os.environ.get("RESUME_FROM"),
                  "hf_import_sample_checks": self.source_checks, "steps": self.steps}
        (self.output / f"mtp_rank_{self.rank}_audit.json").write_text(json.dumps(report, indent=2) + "\n")


def enable_mtp_audit(output: Path, rank: int) -> None:
    """Attach the audit through Bridge's callback API in this SFT worker only."""
    module = importlib.import_module("megatron.bridge.training.finetune")
    original = module.pretrain

    def pretrain_with_audit(config, forward_step_func, callbacks=None):
        return original(config, forward_step_func,
                        callbacks=[*(callbacks or []), MTPAudit(output, rank)])

    module.pretrain = pretrain_with_audit
