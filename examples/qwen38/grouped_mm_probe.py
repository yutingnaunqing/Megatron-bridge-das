"""Minimal DCU probe for torch.nn.functional.grouped_mm.

使用示例（在项目根目录执行）：
    HIP_VISIBLE_DEVICES=0 python -u examples/qwen38/grouped_mm_probe.py \
        --m-per-group 16,16,16,16 --k 128 --n 128 --dtype bfloat16 --backward \
        --output runs/qwen38/grouped_mm_bf16

此脚本隔离 Transformers MoE 所用的 GroupedGEMM；不会构造 Qwen 模型或加载 checkpoint。
推荐先测 BF16。FP32 若出现 ``hipblaslt_groupedgemm: no valid algorithm found``，应单独
记录为 hipBLASLt GroupedGEMM 覆盖范围问题。
"""

import argparse
import json
from pathlib import Path
import traceback


def parse_m_per_group(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("m-per-group must be comma-separated integers") from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("m-per-group must contain one or more non-negative integers")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-per-group", type=parse_m_per_group, default="16,16,16,16")
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--backward", action="store_true", help="Also execute a scalar loss backward pass.")
    parser.add_argument("--output", type=Path, default=Path("runs/qwen38/grouped_mm_probe"))
    args = parser.parse_args()
    if args.k <= 0 or args.n <= 0:
        parser.error("k and n must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "arguments": {
            "m_per_group": args.m_per_group,
            "k": args.k,
            "n": args.n,
            "dtype": args.dtype,
            "device": args.device,
            "backward": args.backward,
        }
    }
    stage = "imports"
    try:
        import torch
        import torch.nn.functional as F

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DCU unavailable to PyTorch")
        device = torch.device(args.device)
        dtype = getattr(torch, args.dtype)
        groups = len(args.m_per_group)
        offsets = torch.tensor(args.m_per_group, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        total_m = sum(args.m_per_group)
        # grouped_mm contract: A=(sum(M_i), K), B=(groups, K, N),
        # offs=[M_0, M_0+M_1, ...]. Row ranges [start:end] use B[i].
        mat_a = torch.randn(total_m, args.k, device=device, dtype=dtype, requires_grad=args.backward)
        mat_b = torch.randn(groups, args.k, args.n, device=device, dtype=dtype, requires_grad=args.backward)
        report["environment"] = {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(device) if args.device == "cuda" else "cpu",
        }
        report["tensors"] = {
            "mat_a_shape": list(mat_a.shape),
            "mat_b_shape": list(mat_b.shape),
            "offsets": offsets.cpu().tolist(),
            "per_group_m": args.m_per_group,
            "dtype": str(dtype),
            "mat_a_contiguous": mat_a.is_contiguous(),
            "mat_b_contiguous": mat_b.is_contiguous(),
        }
        print(json.dumps(report, indent=2), flush=True)

        stage = "forward"
        output = F.grouped_mm(mat_a, mat_b, offs=offsets)
        if args.device == "cuda":
            torch.cuda.synchronize(device)
        if not torch.isfinite(output).all().item():
            raise RuntimeError("Non-finite grouped_mm output")
        report["output_shape"] = list(output.shape)

        if args.backward:
            stage = "backward"
            output.float().square().mean().backward()
            if args.device == "cuda":
                torch.cuda.synchronize(device)
            for name, tensor in (("mat_a", mat_a), ("mat_b", mat_b)):
                if tensor.grad is None or not torch.isfinite(tensor.grad).all().item():
                    raise RuntimeError(f"Missing or non-finite gradient for {name}")

        report["status"] = "PASS"
        print("PASS: grouped_mm", flush=True)
    except Exception:
        report.update(status="FAIL", failed_stage=stage, traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
