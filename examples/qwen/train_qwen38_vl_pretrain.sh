#!/bin/bash

set -euo pipefail
CURRENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
MEGATRON_PATH=$( dirname "$( dirname "${CURRENT_DIR}" )" )

# default env
DTK_ENV=$(realpath -m "${DTK_ENV:-/opt/dtk/env.sh}")
source "${DTK_ENV}"
source "${MEGATRON_PATH}/env/env.sh"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-4}


# env.sh does not cover Megatron-Energon; add it when a local checkout is
# present.  Remembered separately because the remote shell of a multi-node run
# re-sources env.sh and would otherwise lose it.
ENERGON_SRC="${MEGATRON_PATH}/3rdparty/Megatron-Energon/src"
EXTRA_PYTHONPATH=""
if [[ -d "${ENERGON_SRC}" ]]; then
    export PYTHONPATH="${PYTHONPATH}:${ENERGON_SRC}"
    EXTRA_PYTHONPATH="${ENERGON_SRC}"
fi

# ===================================== Launch Mode =====================================
# Usage: <RECIPE> <dryrun|train> [HOSTFILE] [NODES] [KEY=VALUE ...]
if (( $# < 2 )); then
    echo "Usage: $0 <RECIPE> <dryrun|train> [HOSTFILE] [NODES] [KEY=VALUE ...]"
    echo "Recipes:"
    echo "  qwen38_vl_27b_pretrain_bw1000_bf16_config"
    echo "  qwen38_vl_27b_pretrain_bw1000_bf16_mock_config"
    echo "  qwen38_vl_27b_pretrain_8gpu_smoke_bw1000_bf16_config"
    exit 1
fi
RECIPE=$1
MODE=$2
shift 2

# [HOSTFILE] and [NODES] keep positions, and anything left over is a
# KEY=VALUE override forwarded to the runner.  A third argument containing '='
# is an override, not a hostfile, so both forms can be mixed freely.
HOSTFILE=""
NODES=""
if [[ $# -gt 0 && ${1} != *=* ]]; then
    HOSTFILE=${1}
    shift
    if [[ $# -gt 0 && ${1} =~ ^[0-9]+$ ]]; then
        NODES=${1}
        shift
    fi
fi
EXTRA_ARGS=("$@")

# The recipe name selects the dataset backend.
if [[ ${RECIPE} == *_mock_config || ${RECIPE} == *_8gpu_smoke_* ]]; then
    BACKEND=mock
else
    BACKEND=energon
fi

# ===================================== Multi-node Topology =====================================
GPUS=${GPUS:-8}
PORT=${SSH_PORT:-${PORT:-11463}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29539}

# Use the first NODES hosts, or all hosts when NODES is omitted.
HOSTS=()
if [[ -n ${HOSTFILE} ]]; then
    HOSTFILE=$(realpath -m "${HOSTFILE}")
    mapfile -t HOSTS < <(awk 'NF && $1 !~ /^#/ {print $1}' "${HOSTFILE}")
    NNODES=${NODES:-${#HOSTS[@]}}
    MASTER_ADDR=${HOSTS[0]}
else
    NNODES=${NODES:-1}
fi
WORLD_GPUS=$((NNODES * GPUS))

# ===================================== Model and Data Paths =====================================
# Model config, processor, and tokenizer share the same local directory.
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/public/home/yugeng/work/models/Qwen3.8-27B}
export DATA_PATH=${DATA_PATH:-}
LOCAL_HF_PATH=$(realpath -m "${LOCAL_HF_PATH}")
if [[ -n ${DATA_PATH} ]]; then
    DATA_PATH=$(realpath -m "${DATA_PATH}")
fi

# ===================================== Data Config =====================================
SEQ_LENGTH=${SEQ_LENGTH:-1024}
export SEQ_LENGTH

DATA_CONFIG=(dataset.seq_length=${SEQ_LENGTH})
if [[ ${BACKEND} == energon ]]; then
    DATA_CONFIG+=(dataset.path=${DATA_PATH})
fi

# ===================================== Train Config =====================================
TRAIN_ITERS=${TRAIN_ITERS:-5}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
# EnergonDatasetConfig.micro_batch_size must equal train.micro_batch_size, and
# the recipe sets both from this one variable; passing -mb would move only one.
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export TRAIN_ITERS GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE

TRAIN_CONFIG=(
    train.train_iters=${TRAIN_ITERS}
    train.global_batch_size=${GLOBAL_BATCH_SIZE}
    train.micro_batch_size=${MICRO_BATCH_SIZE}
    validation.eval_iters=0
)

# ===================================== Parallel Config =====================================
# TP4/PP2 also supports the four-language-layer 8gpu_smoke recipe.
TP=${TP:-4}
PP=${PP:-2}
CP=${CP:-1}
SP=false
(( TP <= 1 )) || SP=true

PARALLEL_CONFIG=(
    model.tensor_model_parallel_size=${TP}
    model.pipeline_model_parallel_size=${PP}
    model.context_parallel_size=${CP}
)

DP=$((WORLD_GPUS / (TP * PP * CP)))

# ================================ Optimizer Config =================================
# Set a nonzero fraction explicitly when CPU optimizer offload is needed.
export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0}

# ===================================== Model Config =====================================
MODEL_CONFIG=(
    model.seq_length=${SEQ_LENGTH}
    model.language_max_sequence_length=${SEQ_LENGTH}
    model.sequence_parallel=${SP}
)

# =================================== Mock Data Config ===================================
# Ignored by the energon backend; the recipe reads these from the environment.
MOCK_PROMPT=${MOCK_PROMPT:-Describe this image.}
MOCK_NUM_IMAGES=${MOCK_NUM_IMAGES:-1}
MOCK_NUM_BASE_EXAMPLES=${MOCK_NUM_BASE_EXAMPLES:-1000}
MOCK_RANDOM_SEED=${MOCK_RANDOM_SEED:-0}
MOCK_IMAGE_WIDTH=${MOCK_IMAGE_WIDTH:-256}
MOCK_IMAGE_HEIGHT=${MOCK_IMAGE_HEIGHT:-256}
export MOCK_PROMPT MOCK_NUM_IMAGES MOCK_NUM_BASE_EXAMPLES MOCK_RANDOM_SEED MOCK_IMAGE_WIDTH MOCK_IMAGE_HEIGHT

# =========================== Checkpoint & Logging Config =============================
SAVE_DIR=${CHECKPOINT_PATH:-${SAVE_DIR:-${CURRENT_DIR}/${RECIPE}_tp${TP}_pp${PP}_cp${CP}}}
SAVE_DIR=$(realpath -m "${SAVE_DIR}")
CHECKPOINT_DIR=${CHECKPOINT_DIR:-null}
LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-null}
if [[ ${CHECKPOINT_DIR} != null ]]; then
    CHECKPOINT_DIR=$(realpath -m "${CHECKPOINT_DIR}")
fi
if [[ ${LOAD_CHECKPOINT} != null ]]; then
    LOAD_CHECKPOINT=$(realpath -m "${LOAD_CHECKPOINT}")
fi
CONFIG_DUMP=${SAVE_DIR}/${RECIPE}.yaml

CKPT_AND_LOGGING_CONFIG=(
    checkpoint.load=${LOAD_CHECKPOINT}
    checkpoint.save=${CHECKPOINT_DIR}
    checkpoint.save_interval=100
    logger.log_interval=1
    logger.log_throughput=true
    logger.tensorboard_dir=${SAVE_DIR}/tb_logs
    logger.save_config_filepath=${CONFIG_DUMP}
)

# ===================================== Summary =====================================
mkdir -p "${SAVE_DIR}/train_logs"
echo "======================================"
echo "Qwen3.8-VL 27B pretrain"
echo "======================================"
echo "Mode:        ${MODE}"
echo "Recipe:      ${RECIPE}"
echo "Nodes:       ${NNODES}${HOSTFILE:+ (${HOSTFILE})}"
echo "GPUs:        ${GPUS}/node, ${WORLD_GPUS} total   (TP=${TP}, PP=${PP}, CP=${CP}, SP=${SP}, DP=${DP})"
echo "Iters:       ${TRAIN_ITERS}"
echo "GBS/MBS:     ${GLOBAL_BATCH_SIZE} / ${MICRO_BATCH_SIZE}"
echo "CPU offload: ${OPTIMIZER_OFFLOAD_FRACTION} (optimizer fraction)"
echo "Seq length:  ${SEQ_LENGTH}"
echo "Model:       ${LOCAL_HF_PATH}"
if [[ ${BACKEND} == energon ]]; then
    echo "Dataset:     energon (${DATA_PATH})"
else
    echo "Dataset:     mock (${MOCK_NUM_BASE_EXAMPLES} synthetic examples)"
fi
echo "Save dir:    ${SAVE_DIR}"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "======================================"

# ===================================== Execution =====================================
RECIPE_RUNNER=${RECIPE_RUNNER:-${MEGATRON_PATH}/run_recipe.py}
RECIPE_RUNNER=$(realpath -m "${RECIPE_RUNNER}")
RECIPE_ARGS=(
    "${RECIPE_RUNNER}"
    --recipe "${RECIPE}"
    --mode pretrain
    --step_func qwen3_vl_step
)
if [[ ${MODE} == dryrun ]]; then
    RECIPE_ARGS+=(--dry-run)
fi
RECIPE_ARGS+=(
    "${DATA_CONFIG[@]}"
    "${TRAIN_CONFIG[@]}"
    "${PARALLEL_CONFIG[@]}"
    "${MODEL_CONFIG[@]}"
    "${CKPT_AND_LOGGING_CONFIG[@]}"
)
# Trailing KEY=VALUE arguments override the defaults above.
RECIPE_ARGS+=("${EXTRA_ARGS[@]}")

cd "${CURRENT_DIR}"

case "${MODE}" in
    dryrun)
        python "${RECIPE_ARGS[@]}" 2>&1 | tee "${SAVE_DIR}/train_logs/dryrun.log"
        echo "======================================"
        echo "Dry run complete. Resolved config: ${CONFIG_DUMP}"
        echo "======================================"
        ;;

    train)
        TORCHRUN_ARGS=(
            --nnodes=${NNODES}
            --nproc_per_node=${GPUS}
            --master_addr=${MASTER_ADDR}
            --master_port=${MASTER_PORT}
        )

        if (( NNODES == 1 )); then
            torchrun "${TORCHRUN_ARGS[@]}" "${RECIPE_ARGS[@]}" 2>&1 \
                | tee "${SAVE_DIR}/train_logs/mpi_$(date +%F-%H%M).log"
        else
            # One mpirun process per node (slots=1); each starts its own torchrun
            # for that node's GPUs. The remote command is assembled as a string
            # because --node_rank must stay literal until it is evaluated on the
            # remote host, where mpirun sets OMPI_COMM_WORLD_RANK.
            HOSTFILE_SLOTS=$(mktemp "${CURRENT_DIR}/hostfile_slots.XXXXXX")
            trap 'rm -f "${HOSTFILE_SLOTS}"' EXIT
            printf '%s slots=1\n' "${HOSTS[@]:0:NNODES}" > "${HOSTFILE_SLOTS}"

            TORCHRUN_QUOTED="$(printf '%q ' "${TORCHRUN_ARGS[@]}")"
            RECIPE_QUOTED="$(printf '%q ' "${RECIPE_ARGS[@]}")"
            REMOTE_CMD="torchrun --node_rank=\${OMPI_COMM_WORLD_RANK} ${TORCHRUN_QUOTED} ${RECIPE_QUOTED}"

            # Load the vendor environment on each node and include local Energon sources.
            printf -v DTK_QUOTED '%q' "${DTK_ENV}"
            printf -v ENV_QUOTED '%q' "${MEGATRON_PATH}/env/env.sh"
            printf -v CURRENT_QUOTED '%q' "${CURRENT_DIR}"
            REMOTE_PREFIX="source ${DTK_QUOTED} && source ${ENV_QUOTED} && cd ${CURRENT_QUOTED}"
            if [[ -n ${EXTRA_PYTHONPATH} ]]; then
                REMOTE_PREFIX="${REMOTE_PREFIX} && export PYTHONPATH=\"\${PYTHONPATH}:${EXTRA_PYTHONPATH}\""
            fi

            # The recipe reads its model, data, batch, and mock knobs from the
            # environment, so every one of them has to cross to the other nodes.
            ENV_ARGS=()
            for name in PATH LD_LIBRARY_PATH LIBRARY_PATH ROCM_PATH HYHAL_PATH HF_HOME HF_HUB_CACHE TRITON_CACHE_DIR TRITON_PRINT_AUTOTUNING IB_TYPE NCCL_IB_HCA NCCL_SOCKET_IFNAME NCCL_IB_GID_INDEX NCCL_IB_DISABLE NCCL_TOPO_FILE RCCL_NET_PLANE SHCA_SHUT_UP_FWB RCCL_P2P_XHCL_CHANNEL_NUM HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; do
                [[ ! -v ${name} ]] || ENV_ARGS+=(-x "${name}")
            done
            mpirun -np ${NNODES} --hostfile ${HOSTFILE_SLOTS} \
                --allow-run-as-root \
                --bind-to none \
                --mca plm_rsh_no_tree_spawn 1 \
                --mca plm_rsh_args "-p ${PORT}" \
                "${ENV_ARGS[@]}" \
                -x HF_HUB_OFFLINE \
                -x TRANSFORMERS_OFFLINE \
                -x GPU_MAX_HW_QUEUES \
                -x SEQ_LENGTH \
                -x LOCAL_HF_PATH \
                -x DATA_PATH \
                -x TRAIN_ITERS \
                -x GLOBAL_BATCH_SIZE \
                -x MICRO_BATCH_SIZE \
                -x OPTIMIZER_OFFLOAD_FRACTION \
                -x MOCK_PROMPT \
                -x MOCK_NUM_IMAGES \
                -x MOCK_NUM_BASE_EXAMPLES \
                -x MOCK_RANDOM_SEED \
                -x MOCK_IMAGE_WIDTH \
                -x MOCK_IMAGE_HEIGHT \
                -x MASTER_ADDR=${MASTER_ADDR} \
                -x MASTER_PORT=${MASTER_PORT} \
                bash -c "
                ${REMOTE_PREFIX} && \
                ${REMOTE_CMD}" 2>&1 | tee "${SAVE_DIR}/train_logs/mpi_$(date +%F-%H%M).log"
        fi
        echo "======================================"
        echo "Training finished. Logs: ${SAVE_DIR}; checkpoint save: ${CHECKPOINT_DIR}"
        echo "======================================"
        ;;

    *)
        echo "ERROR: unknown mode '${MODE}'. Use 'dryrun' or 'train'."
        exit 1
        ;;
esac
