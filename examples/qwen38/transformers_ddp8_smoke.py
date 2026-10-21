"""Multi-DCU DDP smoke test for the upstream Qwen4Exp text model.

使用示例（在项目根目录执行）：
    export PYTHONPATH=/public/home/yugeng/work/transformers_qwen38/src:${PYTHONPATH}
    HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
    torchrun --standalone --nproc_per_node=8 examples/qwen38/transformers_ddp8_smoke.py \
        --profile tiny --dtype bfloat16 --attention eager --experts grouped_mm --gdn fla \
        --seq-len 128 --steps 2 --output runs/qwen38/ddp8_tiny_bf16

这是数据并行：每个 rank 保存一份完整缩小模型，反向后 DDP all-reduce 梯度。
它验证 DCU collective 与同步更新，不会把模型参数切分到多张卡。
"""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.official.json"))
    parser.add_argument("--profile", choices=["tiny", "width"], default="tiny")
    parser.add_argument("--layers", type=int, default=8, help="Must be a multiple of four to retain the 3-GDN/1-QSA pattern.")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--experts", choices=["eager", "batched_mm", "grouped_mm"], default="grouped_mm")
    parser.add_argument("--gdn", choices=["fla", "torch_reference"], default="fla")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--backend", default="nccl", choices=["nccl"])
    parser.add_argument("--output", type=Path, default=Path("runs/qwen38/ddp8_tiny_bf16_reference"))
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def build_config(args, config_cls):
    if args.layers < 4 or args.layers % 4:
        raise ValueError("--layers must be at least 4 and divisible by 4")
    if args.seq_len < 2 or args.local_batch_size < 1 or args.steps < 1:
        raise ValueError("seq-len >= 2, local-batch-size >= 1, and steps >= 1 are required")

    raw = json.loads(args.config.read_text()).get("text_config", {})
    layer_types = ["qwen_sparse_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(args.layers)]
    # PLE must be attached to a linear-attention layer. Using layers 2, 6, ...
    # exercises it once in every four-layer GDN/QSA block.
    ple_layer_ids = [index + 1 for index in range(args.layers) if (index + 1) % 4 == 2]
    raw.update(
        num_hidden_layers=args.layers,
        layer_types=layer_types,
        ple_layer_ids=ple_layer_ids,
        num_experts=16,
        num_experts_per_tok=10,
        ngram_vocab_size_base=20_000,
        use_cache=False,
        output_router_logits=True,
    )
    if args.profile == "tiny":
        raw.update(
            vocab_size=512,
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            hc_lowrank=32,
            ple_embed_dim=128,
            ngram_vocab_size_base=257,
            indexer_head_dim=32,
            indexer_budget=16,
            bos_token_id=511,
            eos_token_id=511,
            pad_token_id=0,
        )
        raw["rope_parameters"] = dict(raw["rope_parameters"], mrope_section=[1, 1, 2])
    else:
        # Retain the official hidden size, attention/GDN head dimensions and
        # top-10 routing. The vocabulary and N-gram table are reduced because
        # they consume memory but do not change the kernel paths under test.
        raw.update(vocab_size=8192, bos_token_id=8191, eos_token_id=8191, pad_token_id=0)

    config = config_cls.from_dict(raw)
    config._attn_implementation = args.attention
    return config


def force_gdn_torch_reference(module):
    """Select the upstream GDN reference functions in this process only."""
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        accelerated = getattr(module, name)
        reference = getattr(accelerated, "__wrapped__", None)
        if reference is None:
            raise RuntimeError(f"No PyTorch reference implementation found for {name}")
        setattr(module, name, reference)


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    stage = "imports"

    try:
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        import transformers
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
        import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling

        if world_size < 2:
            raise RuntimeError(f"DDP requires at least two ranks; torchrun reported WORLD_SIZE={world_size}")
        if not torch.cuda.is_available():
            raise RuntimeError("DCU unavailable to PyTorch")

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend=args.backend)

        def stage_print(name):
            nonlocal stage
            stage = name
            if rank == 0:
                print(f"STAGE {name}", flush=True)

        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        if args.gdn == "torch_reference":
            force_gdn_torch_reference(qwen4_exp_modeling)
        config = build_config(args, Qwen4ExpTextConfig)
        if rank == 0:
            config.to_json_file(str(args.output / "effective_config.json"))

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        stage_print("construct_model")
        model = qwen4_exp_modeling.Qwen4ExpForCausalLM(config)
        model.set_experts_implementation(args.experts)
        model = model.to(device=device, dtype=getattr(torch, args.dtype))
        # QSA's discrete Top-K selection leaves its indexer projections without
        # an LM-loss gradient. find_unused_parameters therefore must stay true.
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(args.seed + rank)
        inputs = torch.randint(
            1,
            config.vocab_size - 1,
            (args.local_batch_size, args.seq_len),
            device=device,
        )
        mask = torch.ones_like(inputs)

        stage_print("eval_forward")
        model.eval()
        with torch.no_grad():
            eval_output = model(input_ids=inputs, attention_mask=mask, use_cache=False)
            torch.cuda.synchronize(device)
            if not torch.isfinite(eval_output.logits).all().item():
                raise RuntimeError(f"Rank {rank}: non-finite eval logits")
        del eval_output

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        losses = []
        local_grad_count = 0
        local_missing = []
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            stage_print(f"step_{step}_train_forward_loss")
            output = model(
                input_ids=inputs,
                attention_mask=mask,
                labels=inputs.clone(),
                use_cache=False,
                output_router_logits=True,
            )
            torch.cuda.synchronize(device)
            if output.loss is None or not torch.isfinite(output.loss).item():
                raise RuntimeError(f"Rank {rank}: missing or non-finite loss")
            loss_for_report = output.loss.detach().float()
            dist.all_reduce(loss_for_report, op=dist.ReduceOp.SUM)
            loss_for_report /= world_size
            losses.append(loss_for_report.item())
            if rank == 0:
                print(f"GLOBAL_MEAN_LOSS {loss_for_report.item()}", flush=True)

            stage_print(f"step_{step}_backward")
            output.loss.backward()
            torch.cuda.synchronize(device)
            local_grad_count = 0
            local_missing = []
            for name, parameter in model.module.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    local_missing.append(name)
                elif parameter.grad is not None:
                    local_grad_count += 1
                    if not torch.isfinite(parameter.grad).all().item():
                        raise RuntimeError(f"Rank {rank}: non-finite gradient for {name}")

            stage_print(f"step_{step}_optimizer")
            optimizer.step()
            torch.cuda.synchronize(device)
            for name, parameter in model.module.named_parameters():
                if not torch.isfinite(parameter).all().item():
                    raise RuntimeError(f"Rank {rank}: non-finite parameter after update for {name}")
            del output

        peak_memory = torch.tensor(torch.cuda.max_memory_allocated(device), device=device, dtype=torch.int64)
        dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
        if rank == 0:
            report = {
                "status": "PASS",
                "environment": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "hip": torch.version.hip,
                    "transformers": transformers.__version__,
                    "transformers_path": transformers.__file__,
                },
                "distributed": {
                    "backend": args.backend,
                    "world_size": world_size,
                    "local_batch_size": args.local_batch_size,
                    "global_batch_size": args.local_batch_size * world_size,
                },
                "model": {
                    "profile": args.profile,
                    "layers": args.layers,
                    "parameters_per_rank": sum(parameter.numel() for parameter in model.module.parameters()),
                    "experts_implementation": model.module.config._experts_implementation,
                    "gdn_implementation": args.gdn,
                    "attention_implementation": args.attention,
                },
                "losses": losses,
                "parameters_without_grad": local_missing,
                "max_peak_allocated_gib_per_rank": peak_memory.item() / 2**30,
            }
            (args.output / "result.json").write_text(json.dumps(report, indent=2))
            print(f"PASS: {world_size}-rank DDP forward/loss/backward/all-reduce/update", flush=True)
        dist.barrier()
    except Exception:
        failure = traceback.format_exc()
        try:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / f"rank{rank}_error.txt").write_text(f"failed_stage={stage}\n{failure}")
        except Exception:
            pass
        print(f"RANK {rank} FAILED at {stage}\n{failure}", flush=True)
        raise
    finally:
        try:
            if "dist" in locals() and dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
