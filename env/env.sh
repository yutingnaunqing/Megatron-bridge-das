_HCU_ENV_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MEGATRON_PATH="$( dirname "${_HCU_ENV_DIR}" )"
unset _HCU_ENV_DIR

# HCU recipes are registered in-process by run_recipe.py; never overwrite upstream files.

# These variables should not be modified.
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export TORCH_CPP_LOG_LEVEL=fatal
export GLOG_minloglevel=3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=32
export NCCL_MIN_NCHANNELS=32
export NCCL_NET_GDR_LEVEL=4
export NCCL_NET_GDR_READ=1
export RCCL_SDMA_COPY_ENABLE=0
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATH}/hcu_megatron_bridge/src:${MEGATRON_PATH}/3rdparty/Megatron-Bridge/src:${MEGATRON_PATH}/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}
export TRITON_HOME=/tmp
export ROCSHMEM_IB_GID_INDEX=3

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
    export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1}
    ;;
  shca)
    export NCCL_IB_HCA=${NCCL_IB_HCA:-shca_0:1,shca_1:1,shca_2:1,shca_3:1}
    export NCCL_PXN_DISABLE=0
    export NCCL_PLUGIN_P2P=ib
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
    export NCCL_NET_PLUGIN=shca
    export RCCL_PXN_GPU_BALANCE=1
    export SHCA_DEBUG_MASK=0
    export SHCA_CMR_LOG_LEVEL=1
    export UCX_IB_NUM_PATHS=1

    export NCCL_NET_PLUGIN=/opt/rccl-rdma-sharp-plugins/lib/librccl-net-shca.so
    export NCCL_TOPO_FILE=/usr/local/built-in-508-topo-input-tj-default.xml
    # Per-machine tuning (e.g. 508-shca); enable as needed:
    export RCCL_NET_PLANE="shca_0,shca_3|shca_1,shca_2"
    export SHCA_SHUT_UP_FWB=0
    export RCCL_P2P_XHCL_CHANNEL_NUM=30
    ;;
  *)
    echo "[env.sh] error: unknown IB_TYPE='$IB_TYPE' (expected mlnx or shca)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
