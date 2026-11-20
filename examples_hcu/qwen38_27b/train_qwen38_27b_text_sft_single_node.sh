#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
set -euo pipefail

current_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
megatron_path=$(cd -- "${current_dir}/../.." && pwd)
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
layers=${EXPECTED_NUM_LAYERS:-16}
mtp_layers=${MTP_NUM_LAYERS:-0}
thinking_mode=${THINKING_MODE:-disabled}
suffix=""
[[ ${mtp_layers} == 0 ]] || suffix="-mtp${mtp_layers}"
hf_path=${LOCAL_HF_PATH:-${models_dir}/Qwen3.8-27B-text-${layers}layers${suffix}}

# ===================================== Data Config =====================================
datasets_dir=${DATASETS_DIR:-${megatron_path%/*}/datasets}
seq_length=${SEQ_LENGTH:-4096}
train_samples=${TRAIN_SAMPLES:-16384}
validation_samples=${VALIDATION_SAMPLES:-256}
dataset="ultrachat-qwen38-train${train_samples}-val${validation_samples}-seq${seq_length}-${thinking_mode}"
if [[ ${seq_length} == 1024 && ${thinking_mode} == disabled && ${train_samples} == 16384 && ${validation_samples} == 256 ]]; then
    dataset=ultrachat-qwen38-steady-train16384-val256-seq1024-disabled
fi
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
    *) echo "Usage: $0 [dryrun|sft|convert|prepare]" >&2; exit 2 ;;
esac

# Use full recomputation and CPU offload by default for the 64-layer checkpoint.
recompute=none
offload=0
if [[ ${layers} == 64 ]]; then
    recompute=full
    offload=0.5
fi

# ===================================== Train Config =====================================
gbs=${GLOBAL_BATCH_SIZE:-32}
mbs=${MICRO_BATCH_SIZE:-1}
TRAIN_CONFIG=(
    "train.train_iters=${TRAIN_ITERS:-50}"
    "train.global_batch_size=${gbs}"
    "train.micro_batch_size=${mbs}"
    "train.exit_interval=${EXIT_INTERVAL:-0}"
    "scheduler.lr_warmup_iters=${WARMUP_ITERS:-2}"
    "validation.eval_iters=${EVAL_ITERS:-0}"
    "validation.eval_interval=${EVAL_INTERVAL:-100}"
)

# ===================================== Optimizer Config =====================================
OPTIMIZER_CONFIG=(
    "optimizer.lr=${LR:-1e-5}"
    "optimizer.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION:-${offload}}"
)

# ===================================== Parallel Config =====================================
gpus=${GPUS:-8}
tp=${TP:-2}
pp=${PP:-4}
PARALLEL_CONFIG=(
    "model.tensor_model_parallel_size=${tp}"
    "model.pipeline_model_parallel_size=${pp}"
    model.context_parallel_size=1
    "comm_overlap.tp_comm_overlap=${TP_COMM_OVERLAP:-false}"
)

# ===================================== Model Config =====================================
export GDN_PRE_FUSION=${GDN_PRE_FUSION:-true}
MODEL_CONFIG=(
    "model.gdn_pre_gated_delta_rule_fusion=${GDN_PRE_FUSION}"
    "model.mtp_loss_scaling_factor=${MTP_LOSS_SCALING_FACTOR:-0.1}"
    "model.apply_rope_fusion=${APPLY_ROPE_FUSION:-true}"
    "model.num_layers_in_first_pipeline_stage=${NUM_LAYERS_IN_FIRST_PIPELINE_STAGE:-null}"
    "model.num_layers_in_last_pipeline_stage=${NUM_LAYERS_IN_LAST_PIPELINE_STAGE:-null}"
)
case "${RECOMPUTE:-${recompute}}" in
    full) MODEL_CONFIG+=(model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1) ;;
    none) MODEL_CONFIG+=(model.recompute_granularity=null model.recompute_method=null model.recompute_num_layers=null) ;;
    *) echo "RECOMPUTE must be full or none." >&2; exit 2 ;;
esac

# ===================================== Checkpoint & Logging Config =====================================
save_dir=$(realpath -m -- "${SAVE_DIR:-${megatron_path}/nemo_experiments/qwen38-text-1node-gdnpre${GDN_PRE_FUSION}/$(date +%Y%m%d-%H%M%S)-$$}")
cache_dir=$(realpath -m -- "${QWEN38_CACHE:-${megatron_path}/nemo_experiments/qwen38-sft-cache}")
CKPT_AND_LOGGING_CONFIG=(
    "checkpoint.save_interval=${SAVE_INTERVAL:-0}"
    "logger.log_throughput=${LOG_THROUGHPUT:-true}"
)

# ===================================== Profiler Config =====================================
PROFILING_CONFIG=()
profile=${PROFILE:-true}
case "${profile,,}" in
    true|1)
        PROFILING_CONFIG=(
            profiling.use_pytorch_profiler=true
            "profiling.profile_ranks=${PROFILE_RANKS:-[0,2,4,6]}"
            "profiling.profile_step_start=${PROFILE_STEP_START:-4}"
            "profiling.profile_step_end=${PROFILE_STEP_END:-5}"
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
world_size=${gpus}
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

source_hcu_environment
if [[ ${mode} == dryrun ]]; then
    "${python_cmd[@]}" "${RECIPE_ARGS[@]}" --dry-run "${world_size}" 2>&1 | tee "${save_dir}/train_logs/dryrun.log"
else
    "${python_cmd[@]}" -m torch.distributed.run --nnodes=1 --nproc-per-node="${gpus}" \
        --master-port="${MASTER_PORT:-29501}" "${RECIPE_ARGS[@]}" 2>&1 | tee "${save_dir}/train_logs/node-0.log"
fi
