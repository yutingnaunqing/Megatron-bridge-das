#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob
export LC_ALL=C

usage() {
    cat <<'EOF'
Usage: bash check_qwen38_host.sh [--storage DIR]... [--peer HOST] [--ssh-port PORT] [--data-peer IP]

Run as the training user on each host. With no --storage, inspect the current directory and /tmp.
--peer checks non-interactive SSH to the named host using existing known_hosts and credentials.
--data-peer prints the local route to the peer's data-plane IP (no packets sent).
Review WARN/SKIP lines and the raw output; exit 0 means collection finished, not training readiness.
EOF
}

storage_paths=()
peer=''
ssh_port=22
data_peer=''
while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --storage|--peer|--ssh-port|--data-peer)
            if (( $# < 2 )) || [[ -z $2 || $2 == --* ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --storage) storage_paths+=("$2") ;;
                --peer) peer=$2 ;;
                --ssh-port) ssh_port=$2 ;;
                --data-peer) data_peer=$2 ;;
            esac
            shift 2
            ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! $ssh_port =~ ^[0-9]{1,5}$ ]] || (( 10#$ssh_port < 1 || 10#$ssh_port > 65535 )); then
    printf 'Invalid SSH port: %s\n' "$ssh_port" >&2
    exit 2
fi
if [[ -n $peer && $peer == -* || -n $data_peer && $data_peer == -* ]]; then
    printf 'Peer values must not begin with a dash.\n' >&2
    exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
    printf 'This script requires timeout (GNU coreutils).\n' >&2
    exit 2
fi
if (( ${#storage_paths[@]} == 0 )); then
    storage_paths=("$PWD" /tmp)
fi

section() {
    printf '\n=== %s ===\n' "$1"
}

probe() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf '[SKIP] %s not in PATH\n' "$1"
        return 0
    fi
    printf '\n$'; printf ' %q' "$@"; printf '\n'
    local status=0
    timeout -k 2s 15s "$@" 2>&1 || status=$?
    if (( status != 0 )); then
        printf '[WARN] %s exited %s\n' "$1" "$status"
    fi
}

section 'Host / CPU / Memory'
probe hostname
probe date -u +%FT%TZ
probe id
probe uname -srmo
probe cat /etc/os-release
probe lscpu
probe free -h
printf 'limits: open_files=%s locked_memory_KiB=%s\n' "$(ulimit -n)" "$(ulimit -l)"
probe df -hT -- /dev/shm

section 'HCU / Driver / Device Access'
probe lspci -Dnn
probe modinfo -F version amdgpu
for version_file in /sys/module/amdgpu/version /opt/rocm/.info/version; do
    if [[ -r $version_file ]]; then probe cat "$version_file"; fi
done
if [[ ! -c /dev/kfd ]]; then printf '[WARN] /dev/kfd not visible\n'; fi
render_devices=(/dev/dri/renderD*)
if (( ${#render_devices[@]} == 0 )); then printf '[WARN] No /dev/dri/renderD* visible\n'; fi
for device in /dev/kfd "${render_devices[@]}" /dev/infiniband/uverbs*; do
    [[ -c $device ]] || continue
    probe ls -l -- "$device"
    if [[ ! -r $device || ! -w $device ]]; then
        printf '[WARN] Current user lacks read or write permission on %s\n' "$device"
    fi
done
if command -v hy-smi >/dev/null 2>&1; then
    probe hy-smi
elif command -v amd-smi >/dev/null 2>&1; then
    probe amd-smi list
    probe amd-smi static --vram
    probe amd-smi topology
else
    probe rocm-smi --showproductname --showmeminfo vram --showtopo
fi
probe rocminfo

section 'Network / RDMA'
probe ip -br address
probe rdma link show
probe ibv_devinfo
rdma_ports=(/sys/class/infiniband/*/ports/*)
if (( ${#rdma_ports[@]} == 0 )); then printf '[WARN] No RDMA ports visible in sysfs\n'; fi
for port_dir in "${rdma_ports[@]}"; do
    for attribute_file in "$port_dir"/{state,link_layer,rate}; do
        [[ -r $attribute_file ]] || continue
        printf '%s: ' "${attribute_file#/sys/class/infiniband/}"
        cat -- "$attribute_file" 2>&1 || printf '[WARN] Cannot read RDMA attribute\n'
    done
    for gid_file in "$port_dir"/gids/*; do
        if ! { IFS= read -r gid < "$gid_file"; } 2>/dev/null; then continue; fi
        [[ -n $gid && -n ${gid//[0:]/} ]] || continue
        gid_index=${gid_file##*/}
        gid_type=''; netdev=''
        { IFS= read -r gid_type < "$port_dir/gid_attrs/types/$gid_index"; } 2>/dev/null || true
        { IFS= read -r netdev < "$port_dir/gid_attrs/ndevs/$gid_index"; } 2>/dev/null || true
        printf '%s GID[%s]: %s type=%s netdev=%s\n' "${port_dir#/sys/class/infiniband/}" \
            "$gid_index" "$gid" "${gid_type:-unknown}" "${netdev:-n/a}"
    done
done
if [[ -n $data_peer ]]; then probe ip route get "$data_peer"; fi
if [[ -n $peer ]]; then
    probe ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=yes \
        -o UpdateHostKeys=no -o ClearAllForwardings=yes -o ForwardAgent=no -o ControlPath=none \
        -p "$ssh_port" -- "$peer" hostname
fi

section 'Storage / Mounts'
for storage_path in "${storage_paths[@]}"; do
    printf '\nPath: %s\n' "$storage_path"
    if [[ ! -d $storage_path ]]; then
        printf '[WARN] Directory not found or not traversable\n'
        continue
    fi
    probe findmnt -n -o TARGET,SOURCE,FSTYPE --target "$storage_path"
    probe df -hT -- "$storage_path"
    probe df -i -- "$storage_path"
    if [[ ! -r $storage_path || ! -w $storage_path || ! -x $storage_path ]]; then
        printf '[NOTE] Current user does not have full read/write/traverse permissions; compare with intended use.\n'
    fi
done

printf '\nCollection complete. Review WARN/SKIP and compare both hosts; this is not a training or bandwidth test.\n'
