CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

# These variables should not be modified.
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export TORCH_CPP_LOG_LEVEL=fatal
export GLOG_minloglevel=3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export PYTHONPATH=${MEGATRON_PATH}/src:${MEGATRON_PATH}/3rdparty/Megatron-LM:${PYTHONPATH}
export TRITON_HOME=/tmp
export ROCSHMEM_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}

# These variables should be modified according to the environment of the machine you are using.
# Auto-detect the IB HCA family and pick the matching profile. Force a specific
# profile by exporting IB_TYPE=mlnx or IB_TYPE=shca before sourcing this file.

if [ -z "${IB_TYPE:-}" ]; then
  if [ -d /sys/class/infiniband ]; then
    _ib_devices=$(ls /sys/class/infiniband 2>/dev/null)
    case "$_ib_devices" in
      *mlx*) IB_TYPE=mlnx ;;
      *shca*) IB_TYPE=shca ;;
      *) IB_TYPE=mlnx
         echo "[env.sh] warning: no mlx*/shca* device under /sys/class/infiniband; defaulting IB_TYPE=mlnx" >&2 ;;
    esac
    unset _ib_devices
  else
    IB_TYPE=mlnx
    echo "[env.sh] warning: /sys/class/infiniband missing; defaulting IB_TYPE=mlnx" >&2
  fi
fi
export IB_TYPE

case "$IB_TYPE" in
  mlnx)
    export NCCL_IB_HCA=mlx5
    ;;
  shca)
    export NCCL_IB_HCA=shca
    export NCCL_PLUGIN_P2P=ib
    export NCCL_NET_PLUGIN=shca
    ;;
  *)
    echo "[env.sh] error: unknown IB_TYPE='$IB_TYPE' (expected mlnx or shca)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac