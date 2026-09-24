"""Isolate two unequal PLE embedding tables under eight-way DTensor column sharding.

使用示例（在项目根目录执行）：
    HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
    torchrun --standalone --nproc_per_node=8 examples/qwen38/ple_tp_probe.py
    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 \
        examples/qwen38/ple_tp_probe.py --device cpu

This uses the 8-layer tiny checkpoint's PLE shapes, 4864x8 and 6400x8, without
Qwen, Transformers, FLA, MoE, or a checkpoint. It checks whether the PyTorch
DTensor embedding backward alone can handle different row counts.
"""

import argparse
import os
import traceback


def main():
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend="nccl" if args.device == "cuda" else "gloo")
    try:
        mesh = init_device_mesh(args.device, (world_size,))
        torch.manual_seed(1234)
        weights = []
        outputs = []
        for rows in (4864, 6400):
            full_weight = torch.randn(rows, 8, dtype=torch.bfloat16, device=device)
            weight = torch.nn.Parameter(distribute_tensor(full_weight, mesh, [Shard(1)]))
            ids = torch.randint(0, rows, (1, 256, 16), device=device)
            ids_dt = DTensor.from_local(ids, mesh, [Replicate()], run_check=False)
            output = F.embedding(ids_dt, weight).redistribute(placements=[Replicate()]).to_local()
            weights.append(weight)
            outputs.append(output)
        loss = sum(output.float().square().mean() for output in outputs)
        loss.backward()
        if args.device == "cuda":
            torch.cuda.synchronize(device)
        for rows, weight in zip((4864, 6400), weights):
            grad = weight.grad
            if grad is None or not torch.isfinite(grad.to_local()).all().item():
                raise RuntimeError("Missing or non-finite gradient for {}-row table".format(rows))
            if rank == 0:
                print("TABLE {}x8: GRAD {}".format(rows, tuple(grad.shape)), flush=True)
        if rank == 0:
            print("PASS: two unequal DTensor embedding tables backward", flush=True)
        dist.barrier()
    except Exception:
        print("RANK {} FAILED\n{}".format(rank, traceback.format_exc()), flush=True)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
