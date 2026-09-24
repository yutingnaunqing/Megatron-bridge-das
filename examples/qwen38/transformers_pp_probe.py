"""Probe whether native Transformers PP can forward the Qwen4Exp text model.

使用示例（在项目根目录、yg_qwen3.8 容器内执行）：
    export PYTHONPATH=/public/home/yugeng/work/transformers_qwen38/src:${PYTHONPATH}
    HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
    torchrun --standalone --nproc_per_node=2 examples/qwen38/transformers_pp_probe.py \
        --checkpoint runs/qwen38/tp8_tiny_checkpoint_tp8 \
        --output runs/qwen38/pp2_tiny_probe

This is a forward-only capability probe. It is expected to report a limitation
until Qwen4Exp's PLE IDs and gated residual streams are handled across stages.
"""

import argparse
import json
import os
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/qwen38/tp8_tiny_checkpoint_tp8"))
    parser.add_argument("--output", type=Path, default=Path("runs/qwen38/pp2_tiny_probe"))
    parser.add_argument("--seq-len", type=int, default=16)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    args.output.mkdir(parents=True, exist_ok=True)
    stage = "import"
    report = {"rank": rank, "world_size": world_size, "checkpoint": str(args.checkpoint)}
    try:
        import torch
        import torch.distributed as dist
        from transformers.distributed.configuration_utils import DistributedConfig
        import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen_modeling

        if world_size != 2:
            raise ValueError("Run this probe with --nproc_per_node=2")
        torch.cuda.set_device(local_rank)
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
            decorated = getattr(qwen_modeling, name)
            setattr(qwen_modeling, name, decorated.__wrapped__)

        stage = "load_pp"
        model = qwen_modeling.Qwen4ExpForCausalLM.from_pretrained(
            str(args.checkpoint),
            dtype=torch.bfloat16,
            attn_implementation="eager",
            experts_implementation="batched_mm",
            distributed_config=DistributedConfig(pp_size=2),
        )
        report["owned_parameter_count"] = sum(p.numel() for p in model.parameters())
        report["pp_rank"] = model._pp_stage.pp_rank
        report["ple_layer_ids"] = model.config.ple_layer_ids
        print(f"RANK {rank} LOAD_PP_OK owned_parameters={report['owned_parameter_count']}", flush=True)

        stage = "forward"
        input_ids = torch.ones((1, args.seq_len), dtype=torch.long, device=torch.device("cuda", local_rank))
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
            torch.cuda.synchronize()
        report["logits_shape"] = list(outputs.logits.shape)
        report["status"] = "PASS"
        print(f"RANK {rank} FORWARD_OK logits_shape={report['logits_shape']}", flush=True)
    except Exception:
        report.update(status="FAIL", failed_stage=stage, traceback=traceback.format_exc())
        print(f"RANK {rank} FAILED at {stage}\n{report['traceback']}", flush=True)
        raise
    finally:
        (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2))
        try:
            if "dist" in locals() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
