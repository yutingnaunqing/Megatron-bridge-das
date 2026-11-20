#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
set -euo pipefail

current_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
megatron_path=$(cd -- "${current_dir}/../.." && pwd)
launch_dir=${PWD}
cd -- "${megatron_path}"
mode=${1:-dryrun}
dtk_env=${DTK_ENV:-/opt/dtk/env.sh}
bridge_env=${BRIDGE_ENV_SCRIPT:-${megatron_path}/examples_hcu/env.sh}
python_cmd=("${PYTHON_BIN:-python3}")
if [[ -z ${PYTHON_BIN:-} ]] && command -v uv >/dev/null 2>&1; then
    python_cmd=(uv run --no-sync python)
fi

export PYTHONPATH="${megatron_path}/src:${megatron_path}/3rdparty/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
tmp_dir=${QWEN38_TMPDIR:-/tmp/qwen38-sft}

source_hcu_environment() {
    if [[ -r ${dtk_env} ]]; then
        set +u
        source "${dtk_env}"
        set -u
    fi
    source "${bridge_env}"
    export TMPDIR="${tmp_dir}"
    mkdir -p "${TMPDIR}"
}

# ===================================== Model Inputs =====================================
models_dir=${MODELS_DIR:-${megatron_path%/*}/models}
layers=${EXPECTED_NUM_LAYERS:-64}
mtp_layers=${MTP_NUM_LAYERS:-0}
thinking_mode=${THINKING_MODE:-disabled}
suffix=""
[[ ${mtp_layers} == 0 ]] || suffix="-mtp${mtp_layers}"
hf_path=${LOCAL_HF_PATH:-${models_dir}/Qwen3.8-27B-text-${layers}layers${suffix}}

# ===================================== Data Config =====================================
datasets_dir=${DATASETS_DIR:-${megatron_path%/*}/datasets}
seq_length=${SEQ_LENGTH:-8192}
train_samples=${TRAIN_SAMPLES:-16384}
validation_samples=${VALIDATION_SAMPLES:-256}
dataset="ultrachat-qwen38-train${train_samples}-val${validation_samples}-seq${seq_length}-${thinking_mode}"
# dataset="ultrachat-qwen38-train${train_samples}-val${validation_samples}-seq${seq_length}-${thinking_mode}"
data_path=${DATA_PATH:-${datasets_dir}/${dataset}}
DATA_CONFIG=(
    "dataset.seq_length=${seq_length}"
    "dataset.num_workers=${NUM_WORKERS:-4}"
)

case "${mode}" in
    convert)
        exec "${python_cmd[@]}" "${current_dir}/convert_qwen38_vl_to_text.py" \
            --source "${SOURCE_HF_PATH:-${models_dir}/Qwen3.8-27B}" \
            --output "${hf_path}" --num-layers "${layers}" --mtp-num-layers "${mtp_layers}"
        ;;
    prepare)
        source_hcu_environment
        exec "${python_cmd[@]}" "${current_dir}/prepare_ultrachat_sft.py" \
            --source "${RAW_DATA_PATH:-${datasets_dir}/HuggingfaceH4_ultrachat_200k}" \
            --output "${data_path}" --max-train-samples "${train_samples}" \
            --max-validation-samples "${validation_samples}" --model "${hf_path}" \
            --seq-length "${seq_length}" --thinking-mode "${thinking_mode}" --check-tokens
        ;;
    dryrun|sft) ;;
    *) echo "Usage: $0 [dryrun|sft|convert|prepare] [HOSTFILE]" >&2; exit 2 ;;
esac

# ===================================== Train Config =====================================
gbs=${GLOBAL_BATCH_SIZE:-64}
mbs=${MICRO_BATCH_SIZE:-1}
TRAIN_CONFIG=(
    "train.train_iters=${TRAIN_ITERS:-500}"
    "train.global_batch_size=${gbs}"
    "train.micro_batch_size=${mbs}"
    "train.exit_interval=${EXIT_INTERVAL:-0}"
    "scheduler.lr_warmup_iters=${WARMUP_ITERS:-20}"
    "validation.eval_iters=${EVAL_ITERS:-8}"
    "validation.eval_interval=${EVAL_INTERVAL:-100}"
)

# ===================================== Optimizer Config =====================================
OPTIMIZER_CONFIG=(
    "optimizer.lr=${LR:-1e-5}"
    "optimizer.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION:-0}"
)

# ===================================== Parallel Config =====================================
gpus=${GPUS:-8}
tp=${TP:-2}
pp=${PP:-4}
cp=${CP:-2}
PARALLEL_CONFIG=(
    "model.tensor_model_parallel_size=${tp}"
    "model.pipeline_model_parallel_size=${pp}"
    "model.context_parallel_size=${cp}"
    "comm_overlap.tp_comm_overlap=${TP_COMM_OVERLAP:-false}"
)

# ===================================== Model Config =====================================
MODEL_CONFIG=(
    "model.gdn_pre_gated_delta_rule_fusion=${GDN_PRE_FUSION:-false}"
    "model.mtp_loss_scaling_factor=${MTP_LOSS_SCALING_FACTOR:-0.1}"
    "model.apply_rope_fusion=${APPLY_ROPE_FUSION:-false}"
    "model.num_layers_in_first_pipeline_stage=${NUM_LAYERS_IN_FIRST_PIPELINE_STAGE:-null}"
    "model.num_layers_in_last_pipeline_stage=${NUM_LAYERS_IN_LAST_PIPELINE_STAGE:-null}"
)
case "${RECOMPUTE:-full}" in
    full) MODEL_CONFIG+=(model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1) ;;
    none) MODEL_CONFIG+=(model.recompute_granularity=null model.recompute_method=null model.recompute_num_layers=null) ;;
    *) echo "RECOMPUTE must be full or none." >&2; exit 2 ;;
esac

# ===================================== Checkpoint & Logging Config =====================================
save_dir=$(realpath -m -- "${SAVE_DIR:-${megatron_path}/nemo_experiments/qwen38-text-2node/$(date +%Y%m%d-%H%M%S)-$$}")
cache_dir=$(realpath -m -- "${QWEN38_CACHE:-${megatron_path}/nemo_experiments/qwen38-sft-cache}")
CKPT_AND_LOGGING_CONFIG=(
    "checkpoint.save_interval=${SAVE_INTERVAL:-250}"
    "logger.log_throughput=${LOG_THROUGHPUT:-true}"
)

# ===================================== Profiler Config =====================================
PROFILING_CONFIG=()
profile=${PROFILE:-true}
case "${profile,,}" in
    true|1)
        PROFILING_CONFIG=(
            profiling.use_pytorch_profiler=true
            "profiling.profile_ranks=${PROFILE_RANKS:-[0,4,8,12]}"
            "profiling.profile_step_start=${PROFILE_STEP_START:-3}"
            "profiling.profile_step_end=${PROFILE_STEP_END:-4}"
            "profiling.pytorch_profiler_collect_shapes=${PROFILE_RECORD_SHAPES:-false}"
            "profiling.pytorch_profiler_collect_callstack=${PROFILE_WITH_STACK:-false}"
        )
        ;;
    false|0) ;;
    *) echo "PROFILE must be true/false or 1/0." >&2; exit 2 ;;
esac

# ===================================== Execution =====================================
if [[ ! -f ${hf_path}/config.json || ! -d ${data_path} ]]; then
    printf 'Missing model config or dataset directory.\n  Model: %s/config.json\n  Data:  %s\n' \
        "${hf_path}" "${data_path}" >&2
    exit 2
fi
if [[ ! -f ${megatron_path}/3rdparty/Megatron-LM/megatron/core/__init__.py ]]; then
    echo "Initialize the pinned Megatron-LM submodule before running SFT." >&2
    exit 2
fi

world_size=$((2 * gpus))
model_parallel=$((tp * pp))
if (( world_size % model_parallel != 0 || gbs % (world_size / model_parallel * mbs) != 0 )); then
    echo "Require world size divisible by TP*PP and GBS divisible by DP*MBS." >&2
    exit 2
fi

export NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
mkdir -p "${save_dir}/train_logs"
RECIPE_ARGS=(
    "${current_dir}/qwen38_sft_worker.py" text
    --hf-path "${hf_path}" --data-path "${data_path}" --save-dir "${save_dir}" --cache-dir "${cache_dir}"
    --layers "${layers}" --mtp-layers "${mtp_layers}" --thinking-mode "${thinking_mode}"
    "${DATA_CONFIG[@]}" "${TRAIN_CONFIG[@]}" "${OPTIMIZER_CONFIG[@]}" "${PARALLEL_CONFIG[@]}"
    "${MODEL_CONFIG[@]}" "${CKPT_AND_LOGGING_CONFIG[@]}" "${PROFILING_CONFIG[@]}"
)
if [[ -n ${RESUME_FROM:-} ]]; then
    RECIPE_ARGS+=(--resume-from "${RESUME_FROM}")
fi

if [[ ${mode} == dryrun ]]; then
    source_hcu_environment
    "${python_cmd[@]}" "${RECIPE_ARGS[@]}" --dry-run "${world_size}" 2>&1 | tee "${save_dir}/train_logs/dryrun.log"
    exit
fi

# Each MPI rank starts one torch distributed agent for the GPUs on that node.
hostfile=${2:-${QWEN38_HOSTFILE:-}}
if [[ -z ${hostfile} ]]; then
    echo "Pass a two-node hostfile or set QWEN38_HOSTFILE." >&2
    exit 2
fi
# Prefer the caller's path, with the repo-relative form as a fallback.
if [[ ${hostfile} != /* && -f ${launch_dir}/${hostfile} ]]; then
    hostfile=${launch_dir}/${hostfile}
fi
if [[ ! -f ${hostfile} || ! -r ${hostfile} ]]; then
    printf 'Cannot read hostfile %s (launch directory: %s, repository: %s).\n' \
        "${hostfile}" "${launch_dir}" "${megatron_path}" >&2
    exit 2
fi
mapfile -t hosts < <(awk 'NF && $1 !~ /^#/ {print $1}' "${hostfile}")
if [[ ${#hosts[@]} != 2 || ${hosts[0]} == "${hosts[1]}" ]]; then
    echo "The hostfile must contain exactly two distinct hosts." >&2
    exit 2
fi
slots=$(mktemp "${save_dir}/train_logs/hostfile.XXXXXX")
trap 'rm -f -- "${slots}"' EXIT
printf '%s slots=1\n' "${hosts[@]}" > "${slots}"

TORCHRUN_ARGS=(
    --nnodes=2 --nproc-per-node="${gpus}"
    --master-addr="${MASTER_ADDR:-${hosts[0]}}" --master-port="${MASTER_PORT:-29501}"
)
remote_cmd="$(printf '%q ' "${python_cmd[@]}" -m torch.distributed.run "${TORCHRUN_ARGS[@]}")"
remote_cmd+="--node-rank=\${OMPI_COMM_WORLD_RANK:?missing MPI rank} $(printf '%q ' "${RECIPE_ARGS[@]}")"
MPI_ARGS=(-np 2 --hostfile "${slots}" --map-by ppr:1:node --bind-to none --wdir "${megatron_path}"
    --mca plm_rsh_args "-p ${MPI_SSH_PORT:-${PORT:-11463}}")
if [[ $(id -u) == 0 ]]; then
    MPI_ARGS+=(--allow-run-as-root)
fi
for name in PATH LD_LIBRARY_PATH IB_TYPE NCCL_NET NCCL_DEBUG NCCL_DEBUG_SUBSYS \
    HF_HUB_OFFLINE TRANSFORMERS_OFFLINE PYTHONDONTWRITEBYTECODE PYTORCH_CUDA_ALLOC_CONF; do
    [[ -v ${name} ]] && MPI_ARGS+=(-x "${name}")
done
mpirun "${MPI_ARGS[@]}" bash -c "
    set -e
    [[ ! -r \$1 ]] || source \"\$1\"
    export PYTHONPATH=\"\$PWD/src:\$PWD/3rdparty/Megatron-LM\${PYTHONPATH:+:\$PYTHONPATH}\"
    source \"\$2\"
    export TMPDIR=\$3
    mkdir -p \"\${TMPDIR}\"
    exec ${remote_cmd}
" "${current_dir}/$(basename -- "$0")" "${dtk_env}" "${bridge_env}" "${tmp_dir}" 2>&1 | tee "${save_dir}/train_logs/mpi.log"
