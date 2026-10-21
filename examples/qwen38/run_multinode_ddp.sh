#!/usr/bin/env bash
# Qwen3.8-Flash-Next 官方 Transformers 文本模型：多节点 DDP 启动器。
#
# 运行位置：能 SSH 登录 hostfile 节点的宿主机；不在容器内运行本脚本。
# 用法：bash examples/qwen38/run_multinode_ddp.sh [hostfile] [--check]
#   不传 hostfile 时使用本脚本同目录的 hostfile。每行一个节点名/IP；空行、
#   注释和行尾的 slots=... 会忽略，重复节点只使用第一次出现的位置。
#   默认使用所有节点；NNODES 可指定只使用前 N 个节点。第一个节点
#   作为 torchrun rendezvous master；每个节点使用相同卡数。
#   --check 只检查 SSH、容器、host 网络、共享目录、依赖和可见 GPU，
#   会创建输出目录，但不会启动 DDP 模型计算。
#
# 先检查，再运行：
#   bash examples/qwen38/run_multinode_ddp.sh examples/qwen38/hostfile --check
#   bash examples/qwen38/run_multinode_ddp.sh examples/qwen38/hostfile
# 只使用 hostfile 前两台、每台 8 卡：
#   NNODES=2 GPUS_PER_NODE=8 \
#     bash examples/qwen38/run_multinode_ddp.sh examples/qwen38/hostfile
#
# 默认模型测试：width、4 层、BF16、eager attention、grouped_mm experts、
# FLA GDN、序列 32、每 rank batch 1、1 步。可先用参考算子排除加速
# kernel 问题：
#   EXPERTS=batched_mm GDN=torch_reference \
#     bash examples/qwen38/run_multinode_ddp.sh examples/qwen38/hostfile
# 进一步增加缩小模型的层数和序列长度：
#   PROFILE=width LAYERS=8 SEQ_LEN=128 STEPS=2 \
#     bash examples/qwen38/run_multinode_ddp.sh examples/qwen38/hostfile
#
# 可覆盖环境变量：
#   HOSTFILE、CONTAINER_BASE(默认 yg_qwen3.8)、PROJECT_ROOT、TRANSFORMERS_ROOT
#   NNODES(默认 hostfile 中的全部节点)、GPUS_PER_NODE(8)
#   MASTER_ADDR(首节点 IPv4)、MASTER_PORT(29538)
#   PROFILE(tiny|width)、LAYERS(4 的倍数)、DTYPE(bfloat16|float32)
#   ATTENTION(eager|sdpa)、EXPERTS(eager|batched_mm|grouped_mm)
#   GDN(fla|torch_reference)、SEQ_LEN(32)、LOCAL_BATCH_SIZE(1)、STEPS(1)
#   OUTPUT_DIR、RUN_ID、RUN_TIMEOUT(秒，默认 1800)
#
# 容器命名须与 create_containers_nodes_train.sh 相同：
#   <CONTAINER_BASE>_$(hostname)，且各容器使用 host 网络并共享
#   /public/home/yugeng/work。每个节点启动一个 torchrun，随后各自创建
#   GPUS_PER_NODE 个进程；总 rank 数 = 节点数 × GPUS_PER_NODE。
# 输出默认写入 basedate/qwen38_runs/ 下的独立目录：logs/ 是各节点日志，
# results/result.json 是全局 rank 0 的结果，失败时看 results/rank<N>_error.txt。
# 这是数据并行：每张卡持有完整缩小模型，不会把参数分到多卡。
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
hostfile="${HOSTFILE:-${script_dir}/hostfile}"
if (( $# > 0 )) && [[ "$1" != --check ]]; then
    hostfile="$1"
    shift
fi
check_only=false
if [[ "${1:-}" == --check ]]; then
    check_only=true
    shift
fi
if (( $# > 0 )) || [[ ! -r "$hostfile" ]]; then
    echo "Usage: $0 [hostfile] [--check]" >&2
    exit 2
fi

project_root="${PROJECT_ROOT:-/public/home/yugeng/work/Megatron-bridge-das}"
transformers_root="${TRANSFORMERS_ROOT:-/public/home/yugeng/work/transformers_qwen38}"
container_base="${CONTAINER_BASE:-yg_qwen3.8}"
gpus_per_node="${GPUS_PER_NODE:-8}"
master_port="${MASTER_PORT:-29538}"
run_timeout="${RUN_TIMEOUT:-1800}"
profile="${PROFILE:-width}"
layers="${LAYERS:-4}"
dtype="${DTYPE:-bfloat16}"
attention="${ATTENTION:-eager}"
experts="${EXPERTS:-grouped_mm}"
gdn="${GDN:-fla}"
seq_len="${SEQ_LEN:-32}"
local_batch_size="${LOCAL_BATCH_SIZE:-1}"
steps="${STEPS:-1}"

for value in "$gpus_per_node" "$master_port" "$run_timeout" "$layers" "$seq_len" "$local_batch_size" "$steps"; do
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "GPU、端口、超时及测试规模参数必须为正整数：$value" >&2
        exit 2
    fi
done
if (( gpus_per_node < 1 || master_port < 1 || master_port > 65535 || run_timeout < 1 || layers < 4 || layers % 4 != 0 || seq_len < 2 || local_batch_size < 1 || steps < 1 )); then
    echo "参数越界：检查 GPUS_PER_NODE、MASTER_PORT、RUN_TIMEOUT、LAYERS、SEQ_LEN、LOCAL_BATCH_SIZE 和 STEPS" >&2
    exit 2
fi
[[ "$profile" == tiny || "$profile" == width ]] || { echo "PROFILE 必须为 tiny 或 width" >&2; exit 2; }
[[ "$dtype" == bfloat16 || "$dtype" == float32 ]] || { echo "DTYPE 必须为 bfloat16 或 float32" >&2; exit 2; }
[[ "$attention" == eager || "$attention" == sdpa ]] || { echo "ATTENTION 必须为 eager 或 sdpa" >&2; exit 2; }
[[ "$experts" == batched_mm || "$experts" == grouped_mm || "$experts" == eager ]] || { echo "EXPERTS 不支持：$experts" >&2; exit 2; }
[[ "$gdn" == torch_reference || "$gdn" == fla ]] || { echo "GDN 必须为 torch_reference 或 fla" >&2; exit 2; }

mapfile -t all_nodes < <(awk '{ sub(/\r$/, "", $1) } $1 != "" && $1 !~ /^#/ && !seen[$1]++ { print $1 }' "$hostfile")
if (( ${#all_nodes[@]} == 0 )); then
    echo "hostfile 没有节点：$hostfile" >&2
    exit 2
fi
nnodes="${NNODES:-${#all_nodes[@]}}"
if ! [[ "$nnodes" =~ ^[0-9]+$ ]] || (( nnodes < 1 || nnodes > ${#all_nodes[@]} )); then
    echo "NNODES 必须是 1 到 ${#all_nodes[@]} 之间的整数" >&2
    exit 2
fi
nodes=("${all_nodes[@]:0:nnodes}")
for node in "${nodes[@]}"; do
    if ! [[ "$node" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "hostfile 包含不安全的节点名：$node" >&2
        exit 2
    fi
done
world_size=$(( nnodes * gpus_per_node ))
if (( world_size < 2 )); then
    echo "DDP 至少需要两个 rank" >&2
    exit 2
fi

master_addr="${MASTER_ADDR:-}"
if [[ -z "$master_addr" ]]; then
    master_addr="$(getent ahostsv4 "${nodes[0]}" | awk 'NR == 1 { print $1 }' || true)"
    master_addr="${master_addr:-${nodes[0]}}"
fi
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
output_dir="${OUTPUT_DIR:-/public/home/yugeng/work/basedate/qwen38_runs/multinode_ddp_${nnodes}nodes_${world_size}ranks_${run_id}}"
if [[ "$output_dir" != /* ]]; then
    output_dir="$project_root/$output_dir"
fi
if ! "$check_only" && [[ -e "$output_dir/results/result.json" ]]; then
    echo "输出目录已有 result.json，请设置新的 OUTPUT_DIR：$output_dir" >&2
    exit 2
fi
mkdir -p "$output_dir/logs" "$output_dir/results"
# The host user launches SSH, while root inside these containers is mapped to
# nobody on the shared filesystem. Both must be able to create result files.
chmod a+rwx "$output_dir" "$output_dir/logs" "$output_dir/results"
touch "$output_dir/.shared_mount_check"

printf '[INFO] nodes=%s GPUs/node=%s world_size=%s master=%s:%s\n' "$nnodes" "$gpus_per_node" "$world_size" "$master_addr" "$master_port"
printf '[INFO] container_base=%s profile=%s layers=%s dtype=%s experts=%s gdn=%s\n' "$container_base" "$profile" "$layers" "$dtype" "$experts" "$gdn"
printf '[INFO] output=%s\n' "$output_dir"

# Check every container before starting a torchrun process. The import also catches
# incompatible huggingface_hub/Transformers versions on individual nodes.
for node in "${nodes[@]}"; do
    printf '[CHECK] %s\n' "$node"
    printf -v remote_args '%q ' "$container_base" "$project_root" "$transformers_root" "$gpus_per_node" "$output_dir"
    ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$node" "bash -s -- $remote_args" <<'REMOTE_CHECK'
set -Eeuo pipefail
container="${1}_$(hostname)"
project_root="$2"
transformers_root="$3"
gpus="$4"
output_dir="$5"
state="$(docker inspect "$container" --format '{{.State.Running}} {{.HostConfig.NetworkMode}}')"
[[ "$state" == 'true host' ]] || { echo "容器 $container 未运行或未使用 host 网络：$state" >&2; exit 1; }
docker exec -i "$container" bash -s -- "$project_root" "$transformers_root" "$gpus" "$output_dir" <<'CONTAINER_CHECK'
set -Eeuo pipefail
project_root="$1"
transformers_root="$2"
gpus="$3"
output_dir="$4"
test -f "$project_root/examples/qwen38/transformers_ddp8_smoke.py"
test -f "$project_root/examples/qwen38/config.official.json"
test -f "$transformers_root/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py"
test -f "$output_dir/.shared_mount_check"
command -v torchrun >/dev/null
command -v timeout >/dev/null
export PYTHONPATH="$transformers_root/src:${PYTHONPATH:-}"
python - "$gpus" "$transformers_root" <<'PY'
import sys
import torch
import transformers
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
expected = int(sys.argv[1])
if torch.cuda.device_count() < expected:
    raise RuntimeError(f"Expected {expected} DCUs, found {torch.cuda.device_count()}")
if not transformers.__file__.startswith(sys.argv[2] + "/src/"):
    raise RuntimeError(f"Wrong Transformers source: {transformers.__file__}")
print(f"ready: {torch.cuda.device_count()} DCUs, torch={torch.__version__}, HIP={torch.version.hip}, transformers={transformers.__version__}, model={Qwen4ExpForCausalLM.__name__}", flush=True)
PY
CONTAINER_CHECK
REMOTE_CHECK
done

if "$check_only"; then
    echo '[PASS] 所有节点检查通过；未启动训练测试。'
    exit 0
fi

printf '%s\n' "hostfile=$hostfile" "nodes=${nodes[*]}" "world_size=$world_size" "master=$master_addr:$master_port" \
    "profile=$profile layers=$layers dtype=$dtype attention=$attention experts=$experts gdn=$gdn" \
    "seq_len=$seq_len local_batch_size=$local_batch_size steps=$steps" \
    > "$output_dir/launch.txt"

pids=()
for rank in "${!nodes[@]}"; do
    node="${nodes[$rank]}"
    log_file="$output_dir/logs/node${rank}_${node}.log"
    printf '[RUN] node_rank=%s host=%s log=%s\n' "$rank" "$node" "$log_file"
    printf -v remote_args '%q ' "$container_base" "$project_root" "$transformers_root" "$nnodes" "$rank" "$gpus_per_node" "$master_addr" "$master_port" "$output_dir/results" "$profile" "$layers" "$dtype" "$attention" "$experts" "$gdn" "$seq_len" "$local_batch_size" "$steps" "$run_timeout"
    ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$node" "bash -s -- $remote_args" > "$log_file" 2>&1 <<'REMOTE_RUN' &
set -Eeuo pipefail
container="${1}_$(hostname)"
shift
docker exec -i "$container" bash -s -- "$@" <<'CONTAINER_RUN'
set -Eeuo pipefail
project_root="$1"; transformers_root="$2"; nnodes="$3"; node_rank="$4"
gpus="$5"; master_addr="$6"; master_port="$7"; output_dir="$8"
profile="$9"; layers="${10}"; dtype="${11}"; attention="${12}"
experts="${13}"; gdn="${14}"; seq_len="${15}"; local_batch_size="${16}"
steps="${17}"; run_timeout="${18}"
cd "$project_root"
export PYTHONPATH="$transformers_root/src:${PYTHONPATH:-}"
export HIP_VISIBLE_DEVICES="$(seq -s, 0 "$((gpus - 1))")"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec timeout --signal=TERM --kill-after=30s "${run_timeout}s" \
    torchrun --nnodes="$nnodes" --nproc_per_node="$gpus" \
    --node_rank="$node_rank" --master_addr="$master_addr" --master_port="$master_port" \
    examples/qwen38/transformers_ddp8_smoke.py \
    --profile "$profile" --layers "$layers" --dtype "$dtype" \
    --attention "$attention" --experts "$experts" --gdn "$gdn" \
    --seq-len "$seq_len" --local-batch-size "$local_batch_size" --steps "$steps" \
    --output "$output_dir"
CONTAINER_RUN
REMOTE_RUN
    pids+=("$!")
done

failed=0
for rank in "${!nodes[@]}"; do
    if wait "${pids[$rank]}"; then
        printf '[DONE] %s\n' "${nodes[$rank]}"
    else
        printf '[FAIL] %s: %s\n' "${nodes[$rank]}" "$output_dir/logs/node${rank}_${nodes[$rank]}.log" >&2
        failed=1
    fi
done
if (( failed != 0 )) || [[ ! -f "$output_dir/results/result.json" ]]; then
    echo "[FAIL] 查看各节点日志和 $output_dir/results/rank<N>_error.txt" >&2
    exit 1
fi
python3 - "$output_dir/results/result.json" "$world_size" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
expected_size = int(sys.argv[2])
if result.get("status") != "PASS" or result.get("distributed", {}).get("world_size") != expected_size:
    raise SystemExit(f"[FAIL] result.json 未证明 {expected_size} rank DDP 测试成功")
print(f"[PASS] {expected_size} rank DDP，结果：{sys.argv[1]}")
PY
