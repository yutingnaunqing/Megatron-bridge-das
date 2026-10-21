"""Qwen4Exp real-data smoke test: single/multi-node TP, DP, EP or hybrid TP+EP+DP.

--data: UTF-8 .txt, .jsonl ({"text": ...} or {"input_ids": [...]}) or
1-D integer .npy tokens. Text requires --tokenizer (local HF directory).
Profiles tiny/width reduce the model; official retains the full text config.
--layers optionally keeps the first N layers. Without --checkpoint weights are
random; with --checkpoint load local HF weights, optionally reducing layers.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import time
import traceback


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--text-key", default="text")
    p.add_argument("--repeat-data", action="store_true", help="Repeat a short dataset instead of failing on exhaustion")
    p.add_argument("--parallel", choices=["tp", "dp", "ep", "hybrid"], default="tp")
    p.add_argument("--tp-size", type=int)
    p.add_argument("--ep-size", type=int)
    p.add_argument("--dp-size", type=int, help="Hybrid: default WORLD_SIZE / (TP_SIZE * EP_SIZE)")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--use-cache", type=int, choices=[0, 1], default=1,
                   help="Public model default: 1. Each training batch gets a fresh cache; checkpointing disables it.")
    p.add_argument("--output-router-logits", type=int, choices=[0, 1], default=0,
                   help="Public model default: 0. Set 1 to include the HF router load-balancing auxiliary loss.")
    p.add_argument("--profile", choices=["tiny", "width", "official"], default="width")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.official.json"))
    p.add_argument("--checkpoint", type=Path, help="Existing local HF checkpoint; shapes come from its config")
    p.add_argument("--layers", type=int)
    for name in ("hidden-size", "num-experts", "num-experts-per-tok", "vocab-size",
                 "ngram-vocab-size-base", "ple-embed-dim", "moe-intermediate-size", "shared-expert-intermediate-size"):
        p.add_argument("--" + name, type=int)
    p.add_argument("--ple-layer-ids", help="Comma-separated one-indexed layers; empty string disables PLE")
    p.add_argument("--model-overrides", type=Path, help="Optional JSON config fields, e.g. attention/GDN head dimensions")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--local-batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    p.add_argument("--experts", choices=["grouped_mm", "batched_mm", "eager"], default="grouped_mm")
    p.add_argument("--gdn", choices=["fla", "torch_reference"], default="fla")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/qwen38/real_data_smoke"))
    a = p.parse_args(argv)
    if a.seq_len < 2 or a.local_batch_size < 1 or a.steps < 1 or (a.layers is not None and a.layers < 1):
        p.error("Require seq-len >= 2, batch >= 1, steps >= 1, layers >= 1")
    if not math.isfinite(a.lr) or a.lr <= 0:
        p.error("lr must be finite and positive")
    if not a.data.is_file():
        p.error(f"Data file does not exist: {a.data}")
    return a


def parallel_sizes(args, world_size):
    if args.parallel != "hybrid":
        sizes = (world_size, 1, 1) if args.parallel == "tp" else (1, world_size, 1) if args.parallel == "ep" else (1, 1, world_size)
        for value, expected in zip((args.tp_size, args.ep_size, args.dp_size), sizes):
            if value is not None and value != expected:
                raise ValueError("Custom TP/EP/DP sizes require --parallel hybrid")
        return sizes
    tp = args.tp_size if args.tp_size is not None else 1
    ep = args.ep_size if args.ep_size is not None else 1
    if tp < 1 or ep < 1 or world_size % (tp * ep):
        raise ValueError("WORLD_SIZE must be divisible by positive TP_SIZE * EP_SIZE")
    dp = args.dp_size if args.dp_size is not None else world_size // (tp * ep)
    if dp < 1 or tp * ep * dp != world_size:
        raise ValueError(f"WORLD_SIZE={world_size} must equal TP_SIZE={tp} * EP_SIZE={ep} * DP_SIZE={dp}")
    return tp, ep, dp


def build_config(args, config_cls, world_size):
    tp_size, ep_size, _ = parallel_sizes(args, world_size)
    path = args.checkpoint / "config.json" if args.checkpoint else args.config
    raw = json.loads(path.read_text())
    raw = dict(raw.get("text_config", raw))
    if not args.checkpoint and args.profile != "official":
        raw.update(num_hidden_layers=4, num_experts=16, ngram_vocab_size_base=20000,
                   ple_layer_ids=[2], split_ngram_parts=1)
        # Keep official vocabulary/token IDs so real tokenizer IDs remain valid.
        if args.profile == "tiny":
            size = ep_size
            raw.update(hidden_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                       linear_num_key_heads=2, linear_num_value_heads=max(8, tp_size),
                       linear_key_head_dim=32, linear_value_head_dim=32, moe_intermediate_size=64,
                       shared_expert_intermediate_size=64, num_experts=((8 + size - 1) // size) * size,
                       num_experts_per_tok=2, hc_lowrank=32,
                       ple_embed_dim=max(128, tp_size * 16),
                       ngram_vocab_size_base=257, indexer_head_dim=32, indexer_budget=16)
            raw["rope_parameters"] = dict(raw["rope_parameters"], mrope_section=[1, 1, 2])
    if args.model_overrides:
        raw.update(json.loads(args.model_overrides.read_text()))
    for name in ("hidden_size", "num_experts", "num_experts_per_tok", "vocab_size",
                 "ngram_vocab_size_base", "ple_embed_dim", "moe_intermediate_size", "shared_expert_intermediate_size"):
        value = getattr(args, name)
        if value is not None:
            raw[name] = value
    layers = args.layers if args.layers is not None else raw["num_hidden_layers"]
    types = raw.get("layer_types", [])
    raw.update(num_hidden_layers=layers,
        layer_types=[types[i] if i < len(types) else "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                     for i in range(layers)],
        ple_layer_ids=[i for i in raw.get("ple_layer_ids", []) if i <= layers],
        use_cache=bool(args.use_cache) and not args.gradient_checkpointing,
        output_router_logits=bool(args.output_router_logits))
    if args.ple_layer_ids is not None:
        raw["ple_layer_ids"] = [int(x) for x in args.ple_layer_ids.split(",") if x]
    if args.vocab_size is not None:
        raw.update(bos_token_id=args.vocab_size - 1, eos_token_id=args.vocab_size - 1, pad_token_id=0)
    config = config_cls.from_dict(raw)
    config._attn_implementation = args.attention
    for name in ("hidden_size", "num_experts", "vocab_size", "ngram_vocab_size_base", "ple_embed_dim",
                 "moe_intermediate_size", "shared_expert_intermediate_size"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 1 <= config.num_experts_per_tok <= config.num_experts:
        raise ValueError("num-experts-per-tok must be between 1 and num-experts")
    if any(i < 1 or i > layers for i in config.ple_layer_ids):
        raise ValueError("PLE layer IDs must be between 1 and layers")
    if args.checkpoint:
        original = json.loads(path.read_text()).get("text_config", json.loads(path.read_text()))
        if layers > original["num_hidden_layers"]:
            raise ValueError("Loaded checkpoint cannot add randomly initialized layers")
    if ep_size > 1 and config.num_experts % ep_size:
        raise ValueError(f"num_experts={config.num_experts} must be divisible by EP_SIZE={ep_size}")
    if tp_size > 1:
        dims = {"hidden_size": config.hidden_size, "vocab_size": config.vocab_size,
                "moe_intermediate_size": config.moe_intermediate_size,
                "shared_expert_intermediate_size": config.shared_expert_intermediate_size,
                "linear_num_value_heads": config.linear_num_value_heads,
                "indexer_projection": config.indexer_n_heads * config.indexer_head_dim}
        if config.ple_layer_ids:
            dims["PLE per-head features"] = config.ple_embed_dim // ((config.ngram_size - 1) * config.heads_per_ngram)
        for name, value in dims.items():
            if value % tp_size:
                raise ValueError(f"TP_SIZE={tp_size}: {name}={value} must be divisible by TP_SIZE")
    return config


def load_tokens(args, needed):
    import numpy as np
    if args.data.suffix == ".npy":
        tokens = np.load(args.data, mmap_mode="r", allow_pickle=False)
        if tokens.ndim != 1 or tokens.dtype.kind not in "iu":
            raise ValueError("Token .npy must be a 1-D integer array")
        return tokens
    if args.data.suffix not in (".txt", ".jsonl"):
        raise ValueError("Data must be .txt, .jsonl or .npy")
    tokenizer, tokens = None, []
    with args.data.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line) if args.data.suffix == ".jsonl" else {args.text_key: line.strip()}
            if "input_ids" in record:
                ids = record["input_ids"]
                if not isinstance(ids, list) or any(type(x) is not int for x in ids):
                    raise ValueError("JSONL input_ids must be an integer list")
            else:
                if tokenizer is None:
                    if args.tokenizer is None:
                        raise ValueError("Text input requires --tokenizer pointing to a local tokenizer directory")
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
                text = record[args.text_key]
                if not isinstance(text, str):
                    raise ValueError("Text field must be a string")
                ids = tokenizer.encode(text, add_special_tokens=False)
                if tokenizer.eos_token_id is not None:
                    ids.append(tokenizer.eos_token_id)
            tokens.extend(ids)
            if len(tokens) >= needed:
                break
    return np.asarray(tokens, dtype=np.int64)


def token_batch(tokens, args, step, rank, world_size, vocab_size, device):
    import numpy as np
    import torch
    count = args.seq_len * args.local_batch_size
    tp, ep, dp = parallel_sizes(args, world_size)
    dp_rank = rank // (tp * ep)
    batch_index = step * dp + dp_rank
    start = batch_index * count
    if args.repeat_data:
        data = tokens[np.arange(start, start + count) % len(tokens)]
    else:
        data = tokens[start:start + count]
    if len(data) != count:
        raise ValueError("Not enough tokens; provide more data or explicitly enable --repeat-data")
    if np.any(data < 0) or np.any(data >= vocab_size):
        raise ValueError("Token IDs exceed vocabulary; use the matching tokenizer and sufficient vocab-size")
    return torch.from_numpy(np.array(data, dtype=np.int64, copy=True)).reshape(args.local_batch_size, args.seq_len).to(device)


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def finite(value):
    import torch
    value = local_tensor(value).detach()
    # Avoid allocating a full-size boolean tensor for large embeddings.
    if value.ndim == 0:
        return bool(torch.isfinite(value).item())
    rows = max(1, 1048576 // max(1, value[0:1].numel()))
    return all(torch.isfinite(value[i:i + rows]).all().item() for i in range(0, value.shape[0], rows))


def hybrid_model_class(base_cls, mesh, device, tp_size, ep_size):
    """Keep HF shard-on-read loading, adding independent EP/TP axes locally.

    Dense layers use the official TP plan. Experts shard [expert, intermediate]
    over [EP, TP]. EP replicates tokens and reduces outputs (no all-to-all).
    No Transformers source files or global registries are modified.
    """
    import torch
    from torch.distributed.tensor import distribute_tensor, Shard
    from torch.distributed.tensor.placement_types import _StridedShard
    from transformers.distributed.tensor_parallel import (
        apply_tensor_parallelism, _use_local_dtensor_params, _AllReduceBackward, _AllReduceForward)

    tp_mesh, ep_mesh = mesh["tp"], mesh["ep"]
    expert_mesh = mesh["ep", "tp"]
    model_group = expert_mesh._flatten("model_parallel").get_group()

    def install_experts(experts):
        for name, placement in (("gate_up_proj", _StridedShard(dim=1, split_factor=2)),
                                ("down_proj", Shard(2))):
            param = experts._parameters[name]
            experts._parameters[name] = torch.nn.Parameter(
                distribute_tensor(param, expert_mesh, [Shard(0), placement], src_data_rank=None),
                requires_grad=param.requires_grad)
        experts.num_experts //= ep_size
        experts.intermediate_dim //= tp_size
        experts._is_expert_parallel = ep_size > 1
        original = experts.forward

        def forward(hidden, indices, scores):
            hidden = _AllReduceBackward.apply(hidden, model_group)
            # EP router already sums routing gradients on EP; sum TP partials here.
            if tp_size > 1:
                scores = _AllReduceBackward.apply(scores, tp_mesh.get_group())
            with _use_local_dtensor_params(experts):
                output = original(hidden, indices, scores)
            return _AllReduceForward.apply(output, model_group)
        experts.forward = forward

    class HybridQwen(base_cls):
        @classmethod
        def _can_set_experts_implementation(cls):
            # HF searches cls.__module__ for its expert-dispatch decorator.
            # Our launcher module defines no expert kernels; they are inherited
            # from Qwen. Delegate rather than relying on a rank-local cached flag.
            return base_cls._can_set_experts_implementation()

        @classmethod
        def prepare_distribute_model(cls, distributed_config, device_map=None):
            return distributed_config, {"": device}, mesh

        @classmethod
        def maybe_distribute_model(cls, model, distributed_config, device_mesh):
            model.config.distributed_config = distributed_config
            model._device_mesh = device_mesh
            model._tp_size = tp_size
            dense_plan = {k: v for k, v in model._tp_plan.items() if ".mlp.experts" not in k}
            ep_plan = dict(model._ep_plan)
            if tp_size > 1:
                model.tp_plan = dense_plan
                apply_tensor_parallelism(model, tp_mesh)
            if ep_size > 1:
                model.tp_plan = {k: v for k, v in ep_plan.items() if k.endswith(".mlp.gate")}
                apply_tensor_parallelism(model, ep_mesh)
            for name, module in model.named_modules():
                if name.endswith(".mlp.experts"):
                    install_experts(module)
            # Loader uses each DTensor's own mesh and placements, including 2-D experts.
            model._tp_plan = {**(dense_plan if tp_size > 1 else {}),
                              **{k: v for k, v in ep_plan.items() if ".mlp.experts" in k or ep_size > 1}}
            return model
    return HybridQwen


def average_dp_gradients(model, group, dp_size):
    """Average corresponding local shards across DP replicas, never over TP/EP."""
    import torch.distributed as dist
    if dp_size > 1:
        for param in model.parameters():
            if param.grad is not None:
                grad = local_tensor(param.grad)
                dist.all_reduce(grad, group=group)
                grad.div_(dp_size)


def main():
    args = parse_args()
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tp_size, ep_size, dp_size = parallel_sizes(args, world_size)
    stage = "imports"
    report = {"status": "FAIL"}
    started = time.perf_counter()
    try:
        import torch
        import torch.distributed as dist
        import transformers
        from torch.nn.parallel import DistributedDataParallel as DDP
        from transformers.distributed.configuration_utils import DistributedConfig
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
        import transformers.models.qwen4_exp.modeling_qwen4_exp as modeling
        device = torch.device(args.device, local_rank) if args.device == "cuda" else torch.device("cpu")
        if args.device == "cuda":
            torch.cuda.set_device(local_rank)
            torch.cuda.reset_peak_memory_stats(device)
        if world_size > 1:
            dist.init_process_group(backend="nccl" if args.device == "cuda" else "gloo")
        mesh = None
        if args.parallel == "hybrid" and world_size > 1:
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh(args.device, (dp_size, ep_size, tp_size), mesh_dim_names=("dp", "ep", "tp"))
        args.output.mkdir(parents=True, exist_ok=True)

        def barrier():
            if world_size > 1:
                dist.barrier()

        def sync():
            if args.device == "cuda":
                torch.cuda.synchronize(device)

        def mark(name):
            nonlocal stage
            stage = name
            if rank == 0:
                print(f"STAGE {name}", flush=True)

        if args.gdn == "torch_reference":
            for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
                decorated = getattr(modeling, name)
                setattr(modeling, name, getattr(decorated, "__wrapped__", decorated))
        mark("config_and_data")
        config = build_config(args, Qwen4ExpTextConfig, world_size)
        if rank == 0:
            if args.use_cache and args.gradient_checkpointing:
                print("[INFO] Gradient checkpointing requires use_cache=False during training.", flush=True)
            print(f"BACKENDS attention={args.attention} experts={args.experts} gdn={args.gdn} "
                  f"use_cache={config.use_cache} output_router_logits={config.output_router_logits}", flush=True)
        needed = args.steps * args.local_batch_size * args.seq_len * dp_size
        if rank == 0:
            print(f"PARALLEL TP={tp_size} EP={ep_size} DP={dp_size} world={world_size}", flush=True)
        tokens = load_tokens(args, needed)
        if not len(tokens) or (len(tokens) < needed and not args.repeat_data):
            raise ValueError(f"Need {needed} tokens, found {len(tokens)}; supply more data or --repeat-data")
        if rank == 0:
            config.to_json_file(str(args.output / "effective_config.json"))
            print(f"MODEL profile={args.profile} layers={config.num_hidden_layers} hidden={config.hidden_size} "
                  f"experts={config.num_experts} vocab={config.vocab_size} ngram_base={config.ngram_vocab_size_base}", flush=True)

        dtype = getattr(torch, args.dtype)
        def construct(target):
            previous = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                with torch.device(target):
                    result = modeling.Qwen4ExpForCausalLM(config)
            finally:
                torch.set_default_dtype(previous)
            result.set_experts_implementation(args.experts)
            return result.to(dtype=dtype)

        torch.manual_seed(args.seed)
        checkpoint = args.checkpoint
        sharded = tp_size * ep_size > 1
        if sharded and checkpoint is None:
            mark("random_checkpoint")
            checkpoint = args.output / "random_checkpoint"
            if rank == 0:
                if checkpoint.exists():
                    raise ValueError("Output already has random_checkpoint; use a new output directory")
                model = construct("cpu")
                model.save_pretrained(str(checkpoint), safe_serialization=True, max_shard_size="2GB")
                del model
            barrier()
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        mark("load_model")
        if checkpoint is not None:
            kwargs = dict(config=config, dtype=dtype, attn_implementation=args.attention,
                          experts_implementation=args.experts, local_files_only=True, output_loading_info=True)
            load_cls = modeling.Qwen4ExpForCausalLM
            if args.parallel == "hybrid" and sharded:
                load_cls = hybrid_model_class(load_cls, mesh, device, tp_size, ep_size)
                kwargs["distributed_config"] = DistributedConfig(tp_size=tp_size, tp_plan="auto")
            elif sharded:
                kwargs["distributed_config"] = DistributedConfig(tp_size=world_size, tp_plan="auto",
                    enable_expert_parallel=args.parallel == "ep")
            model, info = load_cls.from_pretrained(str(checkpoint), **kwargs)
            unexpected = []
            for key in info.get("unexpected_keys", []):
                layer = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
                if key.startswith(("visual.", "model.visual.", "mtp.", "model.mtp.")) or layer and int(layer.group(1)) >= config.num_hidden_layers:
                    continue
                unexpected.append(key)
            if info.get("missing_keys") or info.get("mismatched_keys") or unexpected or info.get("error_msgs"):
                raise RuntimeError(f"Incomplete checkpoint load: {info}")
            if not sharded:
                model.to(device)
        else:
            model = construct(device)
        if sharded and not any(local_tensor(p).numel() < p.numel() for p in model.parameters()):
            raise RuntimeError("Parallel plan did not actually shard weights")
        if args.parallel == "dp" and world_size > 1:
            model = DDP(model, device_ids=[local_rank] if args.device == "cuda" else None,
                        broadcast_buffers=False, find_unused_parameters=True)
        module = model.module if isinstance(model, DDP) else model
        if args.gradient_checkpointing:
            module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)
        losses, step_seconds, missing = [], [], []
        for step in range(args.steps):
            sync(); step_started = time.perf_counter()
            ids = token_batch(tokens, args, step, rank, world_size, config.vocab_size, device)
            optimizer.zero_grad(set_to_none=True)
            mark(f"step_{step}_forward")
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
                           use_cache=config.use_cache, output_router_logits=config.output_router_logits)
            loss = local_tensor(output.loss)
            if not finite(loss):
                raise RuntimeError("Non-finite loss")
            mean_loss = loss.detach().float().clone()
            if world_size > 1:
                dist.all_reduce(mean_loss); mean_loss /= world_size
            losses.append(mean_loss.item())
            mark(f"step_{step}_backward")
            loss.backward()
            if args.parallel == "hybrid" and dp_size > 1:
                average_dp_gradients(module, mesh["dp"].get_group(), dp_size)
            missing, grad_count = [], 0
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    missing.append(name)
                    if ".self_attn.indexer." not in name:
                        raise RuntimeError(f"Missing gradient: {name}")
                else:
                    grad_count += 1
                    if not finite(p.grad):
                        raise RuntimeError(f"Non-finite gradient: {name}")
            if not grad_count:
                raise RuntimeError("No parameter gradients")
            mark(f"step_{step}_update")
            optimizer.step()
            if any(not finite(p) for p in module.parameters()):
                raise RuntimeError("Non-finite parameter after update")
            sync(); step_seconds.append(time.perf_counter() - step_started)
            if rank == 0:
                print(f"STEP {step + 1}/{args.steps} loss={mean_loss.item():.6f} seconds={step_seconds[-1]:.3f} gradients={grad_count}", flush=True)
            del output
        metrics = {"rank": rank, "local_rank": local_rank,
            "dp_rank": rank // (tp_size * ep_size), "ep_rank": (rank // tp_size) % ep_size, "tp_rank": rank % tp_size,
            "local_parameter_numel": sum(local_tensor(p).numel() for p in module.parameters()),
            "parameters_without_grad": missing, "finite_gradient_tensors": grad_count,
            "step_seconds": step_seconds, "total_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if args.device == "cuda" else None}
        per_rank = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(per_rank, metrics)
        else:
            per_rank = [metrics]
        report = {"status": "PASS", "parallel": args.parallel, "world_size": world_size,
            "tp_size": tp_size, "ep_size": ep_size, "dp_size": dp_size,
            "dp_gradient_sync": "local_shard_allreduce" if args.parallel == "hybrid" else "DDP" if args.parallel == "dp" else "none",
            "optimizer_state_sharded_over_dp": False,
            "attention": args.attention, "experts": args.experts, "gdn": args.gdn,
            "use_cache": config.use_cache, "output_router_logits": config.output_router_logits,
            "global_batch_size": args.local_batch_size * dp_size,
            "data": str(args.data), "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "weight_source": "pretrained" if args.checkpoint else "random", "losses": losses, "per_rank": per_rank,
            "environment": {"torch": torch.__version__, "hip": torch.version.hip, "transformers": transformers.__version__}}
        if rank == 0:
            print(f"PASS: {args.parallel.upper()} {world_size} ranks, {args.steps} real-data training steps", flush=True)
        barrier()
    except Exception:
        report.update(status="FAIL", failed_stage=stage, traceback=traceback.format_exc())
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{rank}_error.txt").write_text(report["traceback"])
        raise
    finally:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "result.json").write_text(json.dumps(report, indent=2))
        if "dist" in locals() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
