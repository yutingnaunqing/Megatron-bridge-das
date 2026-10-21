#!/usr/bin/env bash
# Real-data training smoke test. All paths must be visible inside each container.
# DATA=/shared/train.txt TOKENIZER=/shared/tokenizer PARALLEL=tp NNODES=1 \
#   GPUS_PER_NODE=8 PROFILE=width LAYERS=4 STEPS=10 bash run_multinode_tp.sh hostfile
# PARALLEL=tp|dp|ep|hybrid; PROFILE=tiny|width|official. CHECKPOINT loads local HF weights.
# PROFILE=official with no LAYERS preserves the complete text model configuration.
# .txt/.jsonl text needs TOKENIZER; .jsonl input_ids and 1-D integer .npy do not.
# TP/EP use one shared logical batch; DP gives each rank different token batches.
# PARALLEL=hybrid TP_SIZE=8 EP_SIZE=4 DP_SIZE=2 NNODES=8 GPUS_PER_NODE=8
# uses 64 ranks. DP_SIZE defaults to WORLD_SIZE/(TP_SIZE*EP_SIZE).
# Hybrid experts shard both expert IDs (EP) and intermediate features (TP).
# DP averages local shard gradients; AdamW states remain replicated over DP.
# Public model defaults: USE_CACHE=1 OUTPUT_ROUTER_LOGITS=0.
# GRADIENT_CHECKPOINTING=1 disables cache; OUTPUT_ROUTER_LOGITS=1 adds HF router auxiliary loss.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
hostfile="${HOSTFILE:-$script_dir/hostfile}"
export DATA="${DATA:-/public/home/yugeng/work/dataset/oscar-en-10k/oscar-en-10k.jsonl}"
export TOKENIZER="${TOKENIZER-/public/home/yugeng/work/models/Qwen3.5-0.8B}"
if (( $# )) && [[ "$1" != --check ]]; then hostfile="$1"; shift; fi
check_only=false
if [[ "${1:-}" == --check ]]; then check_only=true; shift; fi
if (( $# )) || [[ ! -r "$hostfile" ]]; then echo "Usage: $0 [hostfile] [--check]" >&2; exit 2; fi
: "${DATA:?Set DATA to a real .txt, .jsonl or .npy file visible in every container}"
project_root="${PROJECT_ROOT:-/public/home/yugeng/work/Megatron-bridge-das}"
transformers_root="${TRANSFORMERS_ROOT:-/public/home/yugeng/work/transformers_qwen38}"
container_base="${CONTAINER_BASE:-yg_qwen3.8}"
nnodes="${NNODES:-1}"; gpus="${GPUS_PER_NODE:-8}"
default_parallel=tp
if [[ -v TP_SIZE || -v EP_SIZE || -v DP_SIZE ]]; then default_parallel=hybrid; fi
parallel="${PARALLEL:-$default_parallel}"; profile="${PROFILE:-width}"
master_port="${MASTER_PORT:-29539}"; run_timeout="${RUN_TIMEOUT:-3600}"
for value in "$nnodes" "$gpus" "$master_port" "$run_timeout"; do
    [[ "$value" =~ ^[0-9]+$ ]] && (( value > 0 )) || { echo "Invalid positive integer: $value" >&2; exit 2; }
done
(( master_port <= 65535 )) || { echo 'Invalid master port' >&2; exit 2; }
[[ "$parallel" == tp || "$parallel" == dp || "$parallel" == ep || "$parallel" == hybrid ]] || { echo 'PARALLEL=tp|dp|ep|hybrid' >&2; exit 2; }
[[ "$profile" == tiny || "$profile" == width || "$profile" == official ]] || { echo 'PROFILE=tiny|width|official' >&2; exit 2; }
mapfile -t hosts < <(awk '{sub(/\r$/, "", $1)} $1 != "" && $1 !~ /^#/ && !seen[$1]++ {print $1}' "$hostfile")
(( ${#hosts[@]} >= nnodes )) || { echo 'Not enough hosts in hostfile' >&2; exit 2; }
hosts=("${hosts[@]:0:nnodes}")
for host in "${hosts[@]}"; do
    [[ "$host" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "Invalid host: $host" >&2; exit 2; }
done
world_size=$((nnodes * gpus))
if [[ "$parallel" == hybrid ]]; then
    tp_size="${TP_SIZE:-1}"; ep_size="${EP_SIZE:-1}"
    for value in "$tp_size" "$ep_size" "${DP_SIZE:-1}"; do
        [[ "$value" =~ ^[0-9]+$ ]] && (( value > 0 )) || { echo "Invalid parallel size: $value" >&2; exit 2; }
    done
    (( world_size % (tp_size * ep_size) == 0 )) || { echo 'WORLD_SIZE must be divisible by TP_SIZE * EP_SIZE' >&2; exit 2; }
    dp_size="${DP_SIZE:-$((world_size / (tp_size * ep_size)))}"
    (( world_size == tp_size * ep_size * dp_size )) || { echo 'WORLD_SIZE must equal TP_SIZE * EP_SIZE * DP_SIZE' >&2; exit 2; }
else
    [[ ! -v TP_SIZE && ! -v EP_SIZE && ! -v DP_SIZE ]] || { echo 'Explicit TP_SIZE/EP_SIZE/DP_SIZE requires PARALLEL=hybrid' >&2; exit 2; }
    tp_size=1; ep_size=1; dp_size=1
    case "$parallel" in tp) tp_size="$world_size";; ep) ep_size="$world_size";; dp) dp_size="$world_size";; esac
fi
master_addr="${MASTER_ADDR:-$(getent ahostsv4 "${hosts[0]}" | awk 'NR==1 {print $1}' || true)}"
master_addr="${master_addr:-${hosts[0]}}"
output_dir="${OUTPUT_DIR:-/public/home/yugeng/work/basedate/qwen38_runs/${parallel}_${nnodes}nodes_${world_size}ranks_$(date +%Y%m%d_%H%M%S)_$$}"
[[ "$output_dir" == /* ]] || output_dir="$project_root/$output_dir"
if ! "$check_only" && [[ -e "$output_dir/results/result.json" || -e "$output_dir/results/random_checkpoint" ]]; then
    echo "Use a fresh OUTPUT_DIR: $output_dir" >&2; exit 2
fi
mkdir -p "$output_dir/logs" "$output_dir/results"
# Shared filesystem can map container root to nobody.
chmod a+rwx "$output_dir" "$output_dir/logs" "$output_dir/results"
touch "$output_dir/.shared_mount_check"
args=(--data "$DATA" --parallel "$parallel" --profile "$profile"
      --tp-size "$tp_size" --ep-size "$ep_size" --dp-size "$dp_size"
      --seq-len "${SEQ_LEN:-128}" --local-batch-size "${LOCAL_BATCH_SIZE:-1}"
      --steps "${STEPS:-2}" --lr "${LR:-1e-5}" --seed "${SEED:-1234}"
      --dtype "${DTYPE:-bfloat16}" --attention "${ATTENTION:-sdpa}"
      --experts "${EXPERTS:-grouped_mm}" --gdn "${GDN:-fla}"
      --use-cache "${USE_CACHE:-1}" --output-router-logits "${OUTPUT_ROUTER_LOGITS:-0}"
      --text-key "${TEXT_KEY:-text}" --output "$output_dir/results")
for name in TOKENIZER CHECKPOINT CONFIG LAYERS HIDDEN_SIZE NUM_EXPERTS NUM_EXPERTS_PER_TOK VOCAB_SIZE \
            NGRAM_VOCAB_SIZE_BASE PLE_EMBED_DIM MOE_INTERMEDIATE_SIZE SHARED_EXPERT_INTERMEDIATE_SIZE MODEL_OVERRIDES; do
    if [[ -n "${!name:-}" ]]; then
        flag="${name,,}"; args+=("--${flag//_/-}" "${!name}")
    fi
done
# An explicitly empty PLE_LAYER_IDS disables PLE.
[[ ! -v PLE_LAYER_IDS ]] || args+=(--ple-layer-ids "$PLE_LAYER_IDS")
[[ "${REPEAT_DATA:-0}" == 0 || "${REPEAT_DATA:-0}" == 1 ]] || { echo 'REPEAT_DATA=0|1' >&2; exit 2; }
[[ "${REPEAT_DATA:-0}" == 0 ]] || args+=(--repeat-data)
printf '[INFO] parallel=%s nodes=%s GPUs/node=%s world_size=%s master=%s:%s\n' "$parallel" "$nnodes" "$gpus" "$world_size" "$master_addr" "$master_port"
[[ "${GRADIENT_CHECKPOINTING:-0}" == 0 || "${GRADIENT_CHECKPOINTING:-0}" == 1 ]] || { echo 'GRADIENT_CHECKPOINTING=0|1' >&2; exit 2; }
[[ "${GRADIENT_CHECKPOINTING:-0}" == 0 ]] || args+=(--gradient-checkpointing)
printf '[INFO] TP_SIZE=%s EP_SIZE=%s DP_SIZE=%s global_batch=%s\n' "$tp_size" "$ep_size" "$dp_size" "$((dp_size * ${LOCAL_BATCH_SIZE:-1}))"
printf '[INFO] selected_hosts=%s profile=%s data=%s output=%s\n' "${hosts[*]}" "$profile" "$DATA" "$output_dir"
for host in "${hosts[@]}"; do
    printf '[CHECK] %s\n' "$host"
    printf -v remote_args '%q ' "$container_base" "$project_root" "$transformers_root" "$gpus" "$output_dir" "$world_size" "${args[@]}"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "bash -s -- $remote_args" <<'REMOTE_CHECK'
set -Eeuo pipefail
container="${1}_$(hostname)"; shift
[[ "$(docker inspect "$container" --format '{{.State.Running}} {{.HostConfig.NetworkMode}}')" == 'true host' ]]
docker exec -i "$container" bash -s -- "$@" <<'CONTAINER_CHECK'
set -Eeuo pipefail
project_root="$1"; transformers_root="$2"; gpus="$3"; output_dir="$4"; world_size="$5"; shift 5
cd "$project_root"
test -f "$output_dir/.shared_mount_check"
command -v torchrun >/dev/null; command -v timeout >/dev/null
export PYTHONPATH="$transformers_root/src:${PYTHONPATH:-}"
python - "$gpus" "$world_size" "$transformers_root" "$@" <<'PY'
import sys
import torch, transformers
from examples.qwen38.transformers_tp8_smoke import parse_args, build_config, load_tokens
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
args = parse_args(sys.argv[4:])
build_config(args, Qwen4ExpTextConfig, int(sys.argv[2]))
if not len(load_tokens(args, args.seq_len)):
    raise ValueError("Dataset has no tokens")
assert torch.cuda.device_count() >= int(sys.argv[1]), "Not enough visible DCUs"
assert transformers.__file__.startswith(sys.argv[3] + "/src/"), "Wrong Transformers source"
for path in (args.tokenizer, args.checkpoint):
    if path is not None and not path.is_dir():
        raise ValueError(f"Missing directory: {path}")
print(f"ready: {torch.cuda.device_count()} DCUs, torch={torch.__version__}, HIP={torch.version.hip}, transformers={transformers.__version__}")
PY
CONTAINER_CHECK
REMOTE_CHECK
done
if "$check_only"; then echo '[PASS] 环境和配置检查通过；未启动训练。'; exit 0; fi
printf '%q ' "${args[@]}" > "$output_dir/launch.txt"
pids=()
for rank in "${!hosts[@]}"; do
    log="$output_dir/logs/node${rank}_${hosts[$rank]}.log"
    printf '[RUN] node_rank=%s host=%s log=%s\n' "$rank" "${hosts[$rank]}" "$log"
    printf -v remote_args '%q ' "$container_base" "$project_root" "$transformers_root" "$nnodes" "$rank" "$gpus" "$master_addr" "$master_port" "$run_timeout" "${args[@]}"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${hosts[$rank]}" "bash -s -- $remote_args" > "$log" 2>&1 <<'REMOTE_RUN' &
set -Eeuo pipefail
container="${1}_$(hostname)"; shift
docker exec -i "$container" bash -s -- "$@" <<'CONTAINER_RUN'
set -Eeuo pipefail
project_root="$1"; transformers_root="$2"; nnodes="$3"; rank="$4"
gpus="$5"; master_addr="$6"; master_port="$7"; run_timeout="$8"; shift 8
cd "$project_root"
export PYTHONPATH="$transformers_root/src:${PYTHONPATH:-}"
export HIP_VISIBLE_DEVICES="$(seq -s, 0 "$((gpus - 1))")"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec timeout --signal=TERM --kill-after=30s "${run_timeout}s" \
    torchrun --nnodes="$nnodes" --nproc_per_node="$gpus" --node_rank="$rank" \
    --master_addr="$master_addr" --master_port="$master_port" \
    examples/qwen38/transformers_tp8_smoke.py "$@"
CONTAINER_RUN
REMOTE_RUN
    pids+=("$!")
done
failed=0
for rank in "${!hosts[@]}"; do
    if wait "${pids[$rank]}"; then printf '[DONE] %s\n' "${hosts[$rank]}";
    else printf '[FAIL] %s: %s/logs/node%s_%s.log\n' "${hosts[$rank]}" "$output_dir" "$rank" "${hosts[$rank]}" >&2; failed=1; fi
done
if (( failed != 0 )); then
    for error_file in "$output_dir"/results/rank*_error.txt; do
        [[ -f "$error_file" ]] || continue
        printf '[ERROR] %s\n' "$error_file" >&2
        tail -n 6 "$error_file" >&2
        break
    done
    exit 1
fi
python3 - "$output_dir/results/result.json" "$world_size" "$parallel" "$tp_size" "$ep_size" "$dp_size" <<'PY'
import json, sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text())
if (result.get("status") != "PASS" or result.get("world_size") != int(sys.argv[2])
        or result.get("parallel") != sys.argv[3] or len(result.get("per_rank", [])) != int(sys.argv[2])
        or [result.get(k) for k in ("tp_size", "ep_size", "dp_size")] != list(map(int, sys.argv[4:7]))):
    raise SystemExit("[FAIL] Invalid or incomplete training result")
print(f"[PASS] {sys.argv[3].upper()}，结果：{sys.argv[1]}")
PY
