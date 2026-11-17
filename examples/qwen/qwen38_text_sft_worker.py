#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bind the local GPU before TE import and retain per-rank SFT evidence."""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path


def main() -> None:
    """Execute the existing recipe runner with lightweight runtime auditing."""
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    rank = int(os.environ.get("RANK", "-1"))
    output = Path(os.environ["SAVE_DIR"])
    host = socket.gethostname()
    for key, suffix in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"),
                        ("MIOPEN_USER_DB_PATH", "miopen-db"), ("MIOPEN_CUSTOM_CACHE_DIR", "miopen-kernels")):
        base = Path(os.environ.get(key, f"/tmp/qwen38-text-sft/{suffix}"))
        cache = base / host / f"rank-{os.environ.get('LOCAL_RANK', '0')}"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(cache)
    import torch

    if rank >= 0:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    import run_recipe

    # run_recipe initializes the HCU adaptor before training modules are imported.
    if int(os.environ.get("MTP_NUM_LAYERS", "0")):
        from qwen38_mtp_audit import enable_mtp_audit

        enable_mtp_audit(output, rank)
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.gpt_provider import GPTModelProvider

    provide = GPTModelProvider.provide
    load_weights = AutoBridge.load_hf_weights

    def audited_provide(provider, *args, **kwargs):
        model = provide(provider, *args, **kwargs)
        named = list(model.named_parameters())
        forbidden = [name for name, _ in named if any(part in name.lower() for part in ("vision", "visual", "merger"))]
        frozen = [name for name, value in named if not value.requires_grad]
        mtp = getattr(model, "mtp", None)
        mtp_attention = ([type(layer.mtp_model_layer.self_attention).__name__ for layer in mtp.layers]
                         if mtp is not None else [])
        if any("DeltaNet" in kind for kind in mtp_attention):
            raise ValueError(f"MTP requires full attention, got {mtp_attention}")
        if forbidden or frozen:
            raise RuntimeError(f"Unexpected visual/frozen parameters: {forbidden}, {frozen}")
        row = {"rank": rank, "host": host, "model_class": type(model).__name__,
               "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
               "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
               "kernel_backend": "native",
               "local_parameters": sum(p.numel() for _, p in named),
               "frozen_parameters": frozen, "visual_parameters": forbidden,
               "mtp_num_layers": int(os.environ.get("MTP_NUM_LAYERS", "0")),
               "mtp_parameters": [name for name, _ in named if name.startswith("mtp.")],
               "mtp_attention_types": mtp_attention,
               "layer_numbers": [layer.layer_number for layer in model.decoder.layers],
               "attention_types": [type(layer.self_attention).__name__ for layer in model.decoder.layers]}
        (output / f"model_rank_{rank}.json").write_text(json.dumps(row, indent=2) + "\n")
        return model

    def audited_load(bridge, *args, **kwargs):
        result = load_weights(bridge, *args, **kwargs)
        (output / f"hf_loaded_rank_{rank}.json").write_text(json.dumps(
            {"rank": rank, "source": kwargs.get("hf_path"), "completed": True}, indent=2) + "\n")
        return result

    GPTModelProvider.provide = audited_provide
    AutoBridge.load_hf_weights = audited_load
    started = time.time()
    success = False
    exit_code = None
    try:
        run_recipe.main()
        success = True
        exit_code = 0
    except SystemExit as error:
        exit_code = 0 if error.code is None else error.code
        success = exit_code == 0
        raise
    finally:
        row = {"rank": rank, "host": host, "success": success, "started_unix": started,
               "finished_unix": time.time(), "exit_code": exit_code}
        if torch.cuda.is_initialized():
            row.update(max_allocated_bytes=torch.cuda.max_memory_allocated(),
                       max_reserved_bytes=torch.cuda.max_memory_reserved())
        (output / f"rank_{rank}_final.json").write_text(json.dumps(row, indent=2) + "\n")


if __name__ == "__main__":
    main()
