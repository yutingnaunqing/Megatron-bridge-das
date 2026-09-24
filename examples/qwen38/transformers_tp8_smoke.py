"""Eight-DCU native Transformers tensor-parallel (TP) smoke test for Qwen4Exp.

使用示例（在项目根目录执行）：
    export PYTHONPATH=/public/home/yugeng/work/transformers_qwen38/src:${PYTHONPATH}
    HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
    torchrun --standalone --nproc_per_node=8 examples/qwen38/transformers_tp8_smoke.py \
        --bootstrap-checkpoint --checkpoint runs/qwen38/tp8_tiny_checkpoint_tp8 \
        --dtype bfloat16 --attention eager --experts grouped_mm --gdn fla \
        --seq-len 32 --steps 1 --output runs/qwen38/tp8_tiny_bf16

    # 官方宽度，四层、单处 PLE，缩减专家数量和 N-gram 行数。
    HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
    torchrun --standalone --nproc_per_node=8 examples/qwen38/transformers_tp8_smoke.py \
        --profile width --bootstrap-checkpoint --checkpoint runs/qwen38/tp8_width4_checkpoint \
        --dtype bfloat16 --experts grouped_mm --gdn fla --seq-len 128 --steps 1 \
        --output runs/qwen38/tp8_width4_seq128

    # 8 层两处 PLE 时的诊断路径：复制 PLE 表，其余层仍使用 TP。
    HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
    torchrun --standalone --nproc_per_node=8 examples/qwen38/transformers_tp8_smoke.py \
        --checkpoint runs/qwen38/tp8_8layers_checkpoint --ple-tp replicated \
        --dtype bfloat16 --experts grouped_mm --gdn fla --seq-len 256 --steps 3 \
        --output runs/qwen38/tp8_8layers_seq256_steps3_ple_replicated

此脚本不包装 DDP。它以 ``DistributedConfig(tp_size=8, tp_plan="auto")``
在加载前按官方 TP plan 建立 DeviceMesh/DTensor 并切分权重。首次的
``--bootstrap-checkpoint`` 会按层数和 profile 创建缩小的随机 checkpoint；它验证 TP 加载、通信、
前向、反向和更新，不验证模型效果。
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
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/qwen38/tp8_tiny_checkpoint_tp8"))
    parser.add_argument(
        "--bootstrap-checkpoint",
        action="store_true",
        help="Have rank 0 create the reduced random-weight checkpoint if it does not already exist.",
    )
    parser.add_argument("--layers", type=int, default=4, help="Reduced checkpoint layers; must be divisible by four.")
    parser.add_argument("--profile", choices=["tiny", "width"], default="tiny", help="Tiny diagnostic shape or official-width reduced-layer shape.")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--experts", choices=["eager", "batched_mm", "grouped_mm"], default="grouped_mm")
    parser.add_argument("--gdn", choices=["fla", "torch_reference"], default="fla")
    parser.add_argument(
        "--ple-tp",
        choices=["official", "replicated"],
        default="official",
        help="Use the official PLE column shard, or replicate PLE to isolate its backward path.",
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-baseline-abs-error", type=float, default=0.2)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Do not load an extra unsharded rank-0 reference; required when the full checkpoint cannot fit on one DCU.",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/qwen38/tp8_tiny_bf16"))
    return parser.parse_args()


def build_reduced_config(args, config_cls):
    """Retain all Qwen4Exp substructures; width keeps official hidden/head dimensions."""
    if args.layers < 4 or args.layers % 4:
        raise ValueError("--layers must be at least 4 and divisible by 4")
    raw = json.loads(args.config.read_text()).get("text_config", {})
    layer_types = ["qwen_sparse_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(args.layers)]
    ple_layer_ids = [2] if args.profile == "width" else [index + 1 for index in range(args.layers) if (index + 1) % 4 == 2]
    raw.update(
        num_hidden_layers=args.layers,
        layer_types=layer_types,
        ple_layer_ids=ple_layer_ids,
        split_ngram_parts=1,
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
            # in_proj_a/b are [hidden_size, linear_num_value_heads]; TP=8 requires eight outputs.
            linear_num_value_heads=8,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            # One expert per TP rank makes the moe_tp_experts path unambiguous.
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
        # Keep official attention, GDN, hidden and PLE feature dimensions. Reduce
        # only the number of layers, experts, vocabulary and PLE hash rows so
        # a TP8 end-to-end test fits on the available DCUs.
        raw.update(
            vocab_size=8192,
            num_experts=16,
            ngram_vocab_size_base=20000,
            bos_token_id=8191,
            eos_token_id=8191,
            pad_token_id=0,
        )
    config = config_cls.from_dict(raw)
    config._attn_implementation = args.attention
    return config


def force_gdn_torch_reference(module):
    """Select the readable upstream GDN reference implementation in this process."""
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        accelerated = getattr(module, name)
        reference = getattr(accelerated, "__wrapped__", None)
        if reference is None:
            raise RuntimeError(f"No PyTorch reference implementation found for {name}")
        setattr(module, name, reference)


def local_tensor(tensor):
    """Return a rank-local tensor for DTensor parameters, gradients and outputs."""
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    stage = "imports"

    try:
        import torch
        import torch.distributed as dist
        import transformers
        from transformers.distributed.configuration_utils import DistributedConfig
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
        import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling

        if world_size != args.tp_size:
            raise RuntimeError(f"WORLD_SIZE={world_size}, but --tp-size={args.tp_size}; they must be equal for this test")
        if not torch.cuda.is_available():
            raise RuntimeError("DCU unavailable to PyTorch")
        if args.seq_len < 2 or args.local_batch_size < 1 or args.steps < 1:
            raise ValueError("seq-len >= 2, local-batch-size >= 1, and steps >= 1 are required")

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")

        def mark(name):
            nonlocal stage
            stage = name
            if rank == 0:
                print(f"STAGE {name}", flush=True)

        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        if args.gdn == "torch_reference":
            force_gdn_torch_reference(qwen4_exp_modeling)

        # A normal HF checkpoint is essential: from_pretrained applies TP before
        # loading it, so each rank reads only its own DTensor shard.  This local
        # checkpoint is intentionally random and validates execution, not quality.
        checkpoint_ready = (args.checkpoint / "config.json").is_file() and any(args.checkpoint.glob("*.safetensors"))
        if not checkpoint_ready:
            if not args.bootstrap_checkpoint:
                raise RuntimeError(
                    f"No checkpoint at {args.checkpoint}. Add --bootstrap-checkpoint for the reduced random test, "
                    "or pass a directory containing an official model checkpoint."
                )
            if rank == 0:
                mark("bootstrap_checkpoint")
                args.checkpoint.mkdir(parents=True, exist_ok=True)
                config = build_reduced_config(args, Qwen4ExpTextConfig)
                torch.manual_seed(args.seed)
                bootstrap = qwen4_exp_modeling.Qwen4ExpForCausalLM(config)
                bootstrap.set_experts_implementation(args.experts)
                bootstrap.save_pretrained(str(args.checkpoint), safe_serialization=True)
                del bootstrap
            dist.barrier()

        # TP ranks must see the same token batch.  There is one logical batch,
        # not an eight-times-larger DDP batch.
        checkpoint_config = Qwen4ExpTextConfig.from_pretrained(str(args.checkpoint))
        if args.profile == "width":
            expected = (2560, [2], 16, 20000)
            actual = (
                checkpoint_config.hidden_size,
                checkpoint_config.ple_layer_ids,
                checkpoint_config.num_experts,
                checkpoint_config.ngram_vocab_size_base,
            )
            if actual != expected:
                raise ValueError(
                    f"--profile width expects checkpoint (hidden, PLE layers, experts, ngram base)="
                    f"{expected}, got {actual}; choose a matching checkpoint directory"
                )
        torch.manual_seed(args.seed + 1)
        inputs = torch.randint(
            1,
            checkpoint_config.vocab_size - 1,
            (args.local_batch_size, args.seq_len),
            device=device,
        )
        mask = torch.ones_like(inputs)
        dtype = getattr(torch, args.dtype)

        # First establish a rank-0 unsharded reference.  This is a numerical
        # check on the TP result; it requires a reduced model that fits on one DCU.
        baseline_logits = None
        if rank == 0 and not args.skip_baseline:
            mark("baseline_unsharded_forward")
            baseline = qwen4_exp_modeling.Qwen4ExpForCausalLM.from_pretrained(
                str(args.checkpoint),
                dtype=dtype,
                attn_implementation=args.attention,
                experts_implementation=args.experts,
            ).to(device)
            baseline.eval()
            with torch.no_grad():
                baseline_logits = baseline(input_ids=inputs, attention_mask=mask, use_cache=False).logits.float()
            torch.cuda.synchronize(device)
            del baseline
            torch.cuda.empty_cache()
        dist.barrier()

        mark("tp_from_pretrained")
        if args.ple_tp == "official":
            tp_plan = "auto"
        else:
            # Diagnose the two-PLE-layer EmbeddingBackward shape mismatch.
            # The alternative only changes the PLE embedding TP rule.
            ple_rule = "layers.*.ple.ple_embedding.ngram_embedding"
            base_plan = checkpoint_config.base_model_tp_plan
            if not base_plan or ple_rule not in base_plan:
                raise RuntimeError("The expected official Qwen4Exp PLE TP rule is unavailable")
            tp_plan = {"lm_head": "colwise_gather_output"}
            tp_plan.update(
                {f"model.{name}": style for name, style in base_plan.items() if name != ple_rule}
            )
        distributed_config = DistributedConfig(tp_size=args.tp_size, tp_plan=tp_plan)
        model = qwen4_exp_modeling.Qwen4ExpForCausalLM.from_pretrained(
            str(args.checkpoint),
            dtype=dtype,
            attn_implementation=args.attention,
            experts_implementation=args.experts,
            distributed_config=distributed_config,
        )
        if model.tp_size != args.tp_size:
            raise RuntimeError(f"Expected TP size {args.tp_size}, got {model.tp_size}")

        torch.cuda.reset_peak_memory_stats(device)
        model.eval()
        mark("tp_eval_forward")
        with torch.no_grad():
            eval_output = model(input_ids=inputs, attention_mask=mask, use_cache=False)
            eval_logits = local_tensor(eval_output.logits).float()
            torch.cuda.synchronize(device)
            if not torch.isfinite(eval_logits).all().item():
                raise RuntimeError(f"Rank {rank}: non-finite TP eval logits")

        # lm_head's colwise_gather_output plan should give every TP rank the
        # same full logits.  Compare both cross-rank and with rank 0's baseline.
        rank0_logits = eval_logits.clone() if rank == 0 else torch.empty_like(eval_logits)
        dist.broadcast(rank0_logits, src=0)
        rank_delta = (eval_logits - rank0_logits).abs().max()
        dist.all_reduce(rank_delta, op=dist.ReduceOp.MAX)
        baseline_delta = None
        baseline_mean_abs_error = None
        baseline_rms_abs_error = None
        baseline_logit_rms = None
        if not args.skip_baseline:
            baseline_delta = torch.zeros((), device=device, dtype=torch.float32)
            if rank == 0:
                differences = eval_logits - baseline_logits
                baseline_delta = differences.abs().max()
                baseline_mean_abs_error = differences.abs().mean().item()
                baseline_rms_abs_error = differences.square().mean().sqrt().item()
                baseline_logit_rms = baseline_logits.square().mean().sqrt().item()
                print(
                    f"BASELINE_ERROR max_abs={baseline_delta.item():.6f} "
                    f"mean_abs={baseline_mean_abs_error:.6f} "
                    f"rms_abs={baseline_rms_abs_error:.6f} "
                    f"reference_logit_rms={baseline_logit_rms:.6f}",
                    flush=True,
                )
            dist.broadcast(baseline_delta, src=0)
            if baseline_delta.item() > args.max_baseline_abs_error:
                raise RuntimeError(
                    f"TP logits differ from the unsharded reference by {baseline_delta.item():.6f}, "
                    f"above --max-baseline-abs-error={args.max_baseline_abs_error}"
                )
        del eval_output, eval_logits, rank0_logits, baseline_logits

        model.train()
        # The official TP plan leaves some parameters replicated while it turns
        # selected projections into DTensors. AdamW's foreach path batches both
        # kinds in one operation, which PyTorch DTensor rejects. Per-parameter
        # AdamW keeps the identical update rule and supports this mixed layout.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
        losses = []
        missing_gradients = []
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            mark(f"step_{step}_train_forward_loss")
            output = model(
                input_ids=inputs,
                attention_mask=mask,
                labels=inputs.clone(),
                use_cache=False,
                output_router_logits=True,
            )
            loss = local_tensor(output.loss)
            torch.cuda.synchronize(device)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Rank {rank}: missing or non-finite TP loss")
            reported_loss = loss.detach().float().clone()
            dist.all_reduce(reported_loss, op=dist.ReduceOp.SUM)
            reported_loss /= world_size
            losses.append(reported_loss.item())
            if rank == 0:
                print(f"GLOBAL_MEAN_LOSS {reported_loss.item()}", flush=True)

            mark(f"step_{step}_backward")
            loss.backward()
            torch.cuda.synchronize(device)
            missing_gradients = []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if parameter.grad is None:
                    missing_gradients.append(name)
                    continue
                if not torch.isfinite(local_tensor(parameter.grad)).all().item():
                    raise RuntimeError(f"Rank {rank}: non-finite gradient for {name}")

            mark(f"step_{step}_optimizer")
            optimizer.step()
            torch.cuda.synchronize(device)
            for name, parameter in model.named_parameters():
                if not torch.isfinite(local_tensor(parameter)).all().item():
                    raise RuntimeError(f"Rank {rank}: non-finite parameter after update for {name}")
            del output

        local_parameter_numel = 0
        dtensor_parameter_numel = 0
        dtensor_parameter_count = 0
        for parameter in model.parameters():
            local = local_tensor(parameter)
            local_parameter_numel += local.numel()
            if local is not parameter:
                dtensor_parameter_count += 1
                dtensor_parameter_numel += local.numel()
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
                "parallelism": {
                    "type": "native Transformers tensor parallelism",
                    "tp_size": args.tp_size,
                    "world_size": world_size,
                    "logical_batch_size": args.local_batch_size,
                    "ddp_wrapped": False,
                    "tp_plan": model.tp_plan,
                },
                "model": {
                    "checkpoint": str(args.checkpoint),
                    "profile": args.profile,
                    "layers": model.config.num_hidden_layers,
                    "vocab_size": model.config.vocab_size,
                    "experts_implementation": model.config._experts_implementation,
                    "gdn_implementation": args.gdn,
                    "ple_tp_mode": args.ple_tp,
                    "attention_implementation": args.attention,
                    "local_parameter_numel_per_rank": local_parameter_numel,
                    "dtensor_parameter_count": dtensor_parameter_count,
                    "dtensor_local_numel_per_rank": dtensor_parameter_numel,
                    "parameters_without_grad": missing_gradients,
                },
                "checks": {
                    "max_cross_rank_logit_abs_error": rank_delta.item(),
                    "max_tp_vs_unsharded_logit_abs_error": baseline_delta.item() if baseline_delta is not None else None,
                    "mean_tp_vs_unsharded_logit_abs_error": baseline_mean_abs_error,
                    "rms_tp_vs_unsharded_logit_abs_error": baseline_rms_abs_error,
                    "reference_logit_rms": baseline_logit_rms,
                    "max_peak_allocated_gib_per_rank": peak_memory.item() / 2**30,
                },
                "losses": losses,
            }
            (args.output / "result.json").write_text(json.dumps(report, indent=2))
            print("PASS: 8-rank native Transformers TP forward/loss/backward/update", flush=True)
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
