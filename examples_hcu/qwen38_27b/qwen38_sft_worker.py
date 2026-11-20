# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Bind each worker and run the dev Bridge finetuning loop."""

import argparse
import json
import logging
import os
import socket
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    """Resolve a Qwen SFT recipe and run or save its training configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("text", "vl"))
    parser.add_argument("--dry-run", type=int, metavar="WORLD_SIZE")
    parser.add_argument("--hf-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--mtp-layers", type=int, choices=(0, 1), default=0)
    parser.add_argument("--thinking-mode", choices=("default", "enabled", "disabled"), default="disabled")
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE")
    args = parser.parse_intermixed_args()
    logging.basicConfig(level=logging.INFO)

    host = socket.gethostname()
    local_rank = os.environ.get("LOCAL_RANK")
    for key, subdir in {
        "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "MIOPEN_USER_DB_PATH": "miopen-db",
        "MIOPEN_CUSTOM_CACHE_DIR": "miopen-kernels",
    }.items():
        cache = args.cache_dir / subdir / host / f"rank-{local_rank or 'dryrun'}"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(cache)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    transport = args.save_dir / "transport"
    transport.mkdir(exist_ok=True)
    os.environ["NCCL_DEBUG_FILE"] = str(transport / "nccl_%h_%p.log")

    import torch

    if local_rank is not None and args.dry_run is None:
        torch.cuda.set_device(int(local_rank))

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "training"))
    from recipe_runner import (
        apply_cli_overrides,
        apply_runtime_environment,
        load_forward_step,
        run_config,
        sync_finetuning_cp_invariants,
        sync_model_dataset_sequence_length,
    )

    from qwen38_sft_recipes import finalize_sft_config, qwen38_text_27b_sft_config, qwen38_vl_27b_sft_config

    hf_path = args.hf_path.expanduser().resolve(strict=True)
    data_path = args.data_path.expanduser().resolve(strict=True)
    if args.model == "text":
        config = qwen38_text_27b_sft_config(
            hf_path=hf_path,
            data_path=data_path,
            layers=args.layers,
            mtp_layers=args.mtp_layers,
            thinking_mode=args.thinking_mode,
        )
        step = "llm_step"
    else:
        if args.mtp_layers:
            parser.error("VL SFT uses MTP_NUM_LAYERS=0")
        config = qwen38_vl_27b_sft_config(hf_path=hf_path, data_path=data_path)
        step = "qwen3_vl_step"

    boolean_fields = {
        "comm_overlap.tp_comm_overlap",
        "logger.log_throughput",
        "model.apply_rope_fusion",
        "model.gdn_pre_gated_delta_rule_fusion",
        "profiling.pytorch_profiler_collect_callstack",
        "profiling.pytorch_profiler_collect_shapes",
    }
    overrides = []
    for override in args.overrides:
        key, separator, value = override.partition("=")
        if key in boolean_fields:
            value = {"1": "true", "0": "false"}.get(value, value.lower())
            if value not in {"true", "false"}:
                parser.error(f"{key} must be true/false or 1/0")
        overrides.append(f"{key}{separator}{value}")
    config = apply_cli_overrides(config, overrides)
    resume = args.resume_from.expanduser().resolve(strict=True) if args.resume_from else None
    finalize_sft_config(config, hf_path=hf_path, output=args.save_dir, resume=resume)
    config = apply_runtime_environment(config)
    config = sync_finetuning_cp_invariants(config, mode="finetune")
    config = sync_model_dataset_sequence_length(config)
    forward_step = load_forward_step(step, mode="finetune")

    success = False
    try:
        run_config(
            config=config,
            mode="finetune",
            step_func=forward_step,
            dryrun=args.dry_run is not None,
            dryrun_world_size=args.dry_run,
        )
        success = True
    finally:
        rank = int(os.environ.get("RANK", "-1"))
        result = {"rank": rank, "host": host, "success": success}
        if torch.cuda.is_initialized():
            result.update(
                max_allocated_bytes=torch.cuda.max_memory_allocated(),
                max_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        output = args.save_dir / f"rank_{rank}_final.json"
        try:
            output.write_text(json.dumps(result, indent=2) + "\n")
        except OSError:
            logger.exception("Could not write worker status to %s", output)
            if success:
                raise


if __name__ == "__main__":
    main()
