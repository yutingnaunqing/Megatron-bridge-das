#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bind the local GPU before importing the existing VL SFT training stack."""

import os
import socket
import sys
from pathlib import Path


def main() -> None:
    """Bind the GPU and run the native VL recipe."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    base = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp/qwen38-vl-sft/inductor"))
    cache = base / socket.gethostname() / f"rank-{os.environ.get('LOCAL_RANK', '0')}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    import torch

    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    import run_recipe

    run_recipe.main()


if __name__ == "__main__":
    main()
