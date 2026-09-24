#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Run the Qwen3.8 text reference recipe in an existing HCU environment.

使用示例（在项目根目录执行；通常优先使用 train_qwen38_flash_next_pretrain.sh）：
    HIP_VISIBLE_DEVICES=0 python -u examples/qwen38/run_qwen38.py \
        --profile tiny --num-layers 4 --seq-length 32 --train-iters 1 \
        --global-batch-size 1 --output-dir runs/qwen38/bridge_tiny_manual

传入 ``--dry-run`` 只解析并输出实际配置；该脚本属于 Bridge 参考实现，不是官方
Transformers 的直接测试入口。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT, ROOT / "hcu_megatron_bridge/src",
    ROOT / "3rdparty/Megatron-Bridge/src",
    ROOT / "3rdparty/Megatron-Bridge/3rdparty/Megatron-LM",
    ROOT / "3rdparty/Megatron-Energon/src",
):
    sys.path.insert(0, str(path))

from hcu_megatron.bridge.models.qwen38.config import load_text_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("tiny", "smoke", "official"), default="smoke")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.official.json"))
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-experts", type=int)
    p.add_argument("--ngram-vocab-size", type=int)
    p.add_argument("--seq-length", type=int, default=128)
    p.add_argument("--train-iters", type=int, default=3)
    p.add_argument("--global-batch-size", type=int, default=int(os.getenv("WORLD_SIZE", "1")))
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.seq_length <= 4096:
        p.error("Reference QSA is limited to 1..4096 tokens; it allocates dense attention tensors")
    if args.train_iters < 1 or args.global_batch_size < 1:
        p.error("train-iters and global-batch-size must be positive")
    world = int(os.getenv("WORLD_SIZE", "1"))
    if args.global_batch_size % world:
        p.error("global-batch-size must be divisible by world size for DP-only training")
    text_config = load_text_config(
        args.config, profile=args.profile, num_layers=args.num_layers,
        num_experts=args.num_experts, ngram_vocab_size=args.ngram_vocab_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        (args.output_dir / "text_config.json").write_text(json.dumps(text_config, indent=2) + "\n")
        manifest = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "backend": "dense-masked-QSA reference / native GDN / Megatron-Bridge DP",
                    "random_initialization": True, "mock_data": True, "world_size": world}
        manifest["commits"] = {}
        for name, path in (("project", ROOT), ("bridge", ROOT / "3rdparty/Megatron-Bridge"),
                           ("core", ROOT / "3rdparty/Megatron-Bridge/3rdparty/Megatron-LM")):
            result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
            manifest["commits"][name] = result.stdout.strip() if result.returncode == 0 else "unknown"
        (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logging.basicConfig(level=logging.INFO)
    logging.info("Qwen3.8 reference text backend: profile=%s layers=%s experts=%s ngram_base=%s; DP=%s",
                 args.profile, text_config["num_hidden_layers"], text_config["num_experts"],
                 text_config["ngram_vocab_size_base"], world)

    from hcu_megatron.bridge.models.qwen38.recipe import qwen38_flash_next_pretrain_config
    from hcu_megatron.bridge.models.qwen38.bridge import forward_step
    from megatron.bridge.training.pretrain import pretrain
    import torch
    import transformers

    if rank == 0:
        manifest["runtime"] = {
            "python": sys.version, "torch": torch.__version__,
            "hip": torch.version.hip, "transformers": transformers.__version__,
        }
        manifest["limitations"] = [
            "text only", "no MTP", "DP only", "no HF checkpoint import/export",
            "dense masked QSA", "deterministic block-ID TopK ties",
            "Adam optimizer (not the published Muon recipe)",
        ]
        (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    cfg = qwen38_flash_next_pretrain_config(
        text_config, seq_length=args.seq_length, train_iters=args.train_iters,
        global_batch_size=args.global_batch_size, recompute=args.recompute,
    )
    cfg.logger.tensorboard_dir = str(args.output_dir / "tb_logs")
    cfg.logger.save_config_filepath = str(args.output_dir / "resolved.yaml")
    if args.dry_run:
        if rank == 0:
            cfg.to_yaml(cfg.logger.save_config_filepath)
        logging.info("Dry run complete; no model was allocated")
        return
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("This HCU launcher requires a working HIP PyTorch device")
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device_memory = torch.cuda.get_device_properties(local_rank).total_memory
    # The actual prime-sized table is slightly larger than this lower bound.
    ngram_bytes = 2 * text_config["ngram_vocab_size_base"] * text_config["ple_embed_dim"]
    if text_config["ple_layer_ids"] and ngram_bytes >= device_memory:
        raise RuntimeError(
            f"N-gram weights alone require at least {ngram_bytes / 2**30:.1f} GiB per DP rank; "
            f"device has {device_memory / 2**30:.1f} GiB. This backend does not shard/offload "
            "the table. Choose an explicitly reduced --ngram-vocab-size or the smoke profile."
        )
    pretrain(config=cfg, forward_step_func=forward_step)
    result = {
        "status": "training_returned_successfully", "rank": rank,
        "requested_train_iters": args.train_iters,
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    (args.output_dir / f"result.rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    logging.info("Qwen3.8 reference training completed: %s", result)


if __name__ == "__main__":
    main()
