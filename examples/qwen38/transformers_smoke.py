"""Exercise the upstream Transformers text model; no Megatron/custom model imports.

使用示例（在项目根目录执行）：
    export PYTHONPATH=/public/home/yugeng/work/transformers_qwen38/src:${PYTHONPATH}
    HIP_VISIBLE_DEVICES=0 python -u examples/qwen38/transformers_smoke.py \
        --profile tiny --device cuda --dtype bfloat16 --attention eager \
        --experts grouped_mm --gdn fla --seq-len 32 --steps 1 \
        --output runs/qwen38/hf_tiny_bf16

随机权重和缩小配置只验证执行路径，不验证 checkpoint 质量。需要含有
``transformers.models.qwen4_exp`` 的本地 Transformers 源码。
"""

import argparse
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.official.json"))
    parser.add_argument("--profile", choices=["tiny", "width"], default="tiny")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument(
        "--experts",
        choices=["eager", "batched_mm", "grouped_mm"],
        default="grouped_mm",
        help="Transformers MoE expert implementation; default preserves upstream grouped_mm dispatch.",
    )
    parser.add_argument(
        "--gdn",
        choices=["fla", "torch_reference"],
        default="fla",
        help="Gated DeltaNet implementation; torch_reference bypasses the optional FLA kernel package.",
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("runs/qwen38/transformers_smoke"))
    args = parser.parse_args()
    if args.seq_len < 2 or args.steps < 1:
        parser.error("seq-len must be >= 2 and steps >= 1")
    args.output.mkdir(parents=True, exist_ok=True)
    stage = "imports"
    report = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}

    def mark(name):
        nonlocal stage
        stage = name
        print(f"STAGE {stage}", flush=True)

    try:
        import torch
        import transformers

        report["environment"] = {
            "python": sys.version, "torch": torch.__version__, "hip": torch.version.hip,
            "transformers": transformers.__version__, "transformers_path": transformers.__file__,
        }
        report["gdn_implementation"] = args.gdn
        print(json.dumps(report["environment"], indent=2), flush=True)
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
        import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling

        Qwen4ExpForCausalLM = qwen4_exp_modeling.Qwen4ExpForCausalLM
        if args.gdn == "torch_reference":
            # The official module decorates these readable PyTorch reference
            # functions with an optional FLA implementation. functools.wraps
            # retains __wrapped__, so this selects the original functions
            # without changing the upstream source or model mathematics.
            for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
                accelerated = getattr(qwen4_exp_modeling, name)
                reference = getattr(accelerated, "__wrapped__", None)
                if reference is None:
                    raise RuntimeError(f"No PyTorch reference implementation found for {name}")
                setattr(qwen4_exp_modeling, name, reference)

        mark("config")
        raw = json.loads(args.config.read_text())
        raw = raw.get("text_config", raw)
        # Four layers retain GDN, PLE and QSA. Merely reducing layers leaves
        # the enormous n-gram table and each layer's 512 experts unchanged.
        raw.update(num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["qwen_sparse_attention"],
                   ple_layer_ids=[2], num_experts=16, num_experts_per_tok=10,
                   ngram_vocab_size_base=20000, use_cache=False, output_router_logits=True)
        if args.profile == "tiny":
            raw.update(vocab_size=512, hidden_size=128, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
                       linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                       moe_intermediate_size=64, shared_expert_intermediate_size=64,
                       num_experts=4, num_experts_per_tok=2, hc_lowrank=32,
                       ple_embed_dim=128, ngram_vocab_size_base=257,
                       indexer_head_dim=32, indexer_budget=16,
                       bos_token_id=511, eos_token_id=511, pad_token_id=0)
            raw["rope_parameters"] = dict(raw["rope_parameters"], mrope_section=[1, 1, 2])
        config = Qwen4ExpTextConfig.from_dict(raw)
        config._attn_implementation = args.attention
        config.to_json_file(str(args.output / "effective_config.json"))
        torch.manual_seed(1234)
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("DCU unavailable to PyTorch")
            report["device"] = torch.cuda.get_device_name(0)
            torch.cuda.reset_peak_memory_stats()

        def sync():
            if args.device == "cuda":
                torch.cuda.synchronize()

        mark("construct_model")
        model = Qwen4ExpForCausalLM(config)
        # This is the official Transformers dispatch interface. It changes only
        # the MoE experts' matrix multiplication implementation, not routing,
        # model weights, or the surrounding Qwen architecture.
        model.set_experts_implementation(args.experts)
        report["experts_implementation"] = model.config._experts_implementation
        mark("move_model")
        model = model.to(device=args.device, dtype=getattr(torch, args.dtype))
        report["model_class"] = f"{type(model).__module__}.{type(model).__name__}"
        report["parameters"] = sum(p.numel() for p in model.parameters())
        print(f"MODEL {report['model_class']} PARAMETERS {report['parameters']}", flush=True)
        inputs = torch.randint(1, config.vocab_size - 1, (1, args.seq_len), device=args.device)
        mask = torch.ones_like(inputs)
        model.eval()
        mark("eval_forward")
        with torch.no_grad():
            out = model(input_ids=inputs, attention_mask=mask, use_cache=False)
            sync()
            if not torch.isfinite(out.logits).all().item():
                raise RuntimeError("Non-finite eval logits")
        del out
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        report["losses"] = []
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            mark(f"step_{step}_train_forward_loss")
            # HF shifts labels internally. No custom auxiliary loss is added.
            out = model(input_ids=inputs, attention_mask=mask, labels=inputs.clone(),
                        use_cache=False, output_router_logits=True)
            sync()
            if out.loss is None or not torch.isfinite(out.loss).item():
                raise RuntimeError("Missing or non-finite loss")
            report["losses"].append(out.loss.item())
            print(f"LOSS {out.loss.item()}", flush=True)
            mark(f"step_{step}_backward")
            out.loss.backward()
            sync()
            missing = []
            grads = 0
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is None:
                    missing.append(name)
                if param.grad is not None:
                    grads += 1
                    if not torch.isfinite(param.grad).all().item():
                        raise RuntimeError(f"Non-finite gradient: {name}")
            if grads == 0:
                raise RuntimeError("No parameter gradients")
            report["parameters_without_grad"] = missing
            print(f"GRADIENT_TENSORS {grads} WITHOUT_GRAD {len(missing)}", flush=True)
            mark(f"step_{step}_optimizer")
            optimizer.step()
            sync()
            for name, param in model.named_parameters():
                if not torch.isfinite(param).all().item():
                    raise RuntimeError(f"Non-finite parameter after update: {name}")
            del out
        if args.device == "cuda":
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["status"] = "PASS"
        print("PASS: reduced official text model forward/loss/backward/update", flush=True)
    except Exception:
        report.update(status="FAIL", failed_stage=stage, traceback=traceback.format_exc())
        (args.output / "error.txt").write_text(report["traceback"])
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
