#!/usr/bin/env bash
# Preserve the upstream gitlinks; apply the reviewed local Bridge changes.
set -euo pipefail
CURRENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${CURRENT_DIR}/../.." && pwd)
MODE=${1:---check}
case "${MODE}" in
    --check|--apply) ;;
    *) echo "Usage: $0 [--check|--apply]" >&2; exit 2 ;;
esac
BRIDGE=${ROOT}/3rdparty/Megatron-Bridge
MCORE=${BRIDGE}/3rdparty/Megatron-LM
[[ $(git -C "${BRIDGE}" rev-parse --show-toplevel) == "${BRIDGE}" ]]
[[ $(git -C "${MCORE}" rev-parse --show-toplevel) == "${MCORE}" ]]
[[ $(git -C "${BRIDGE}" rev-parse HEAD) == 51885cf132b2814188b6855c25a8588254274c2a ]]
[[ $(git -C "${MCORE}" rev-parse HEAD) == cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54 ]]
PENDING=()
for name in bridge-recipes.patch bridge-mtp-loss-mask.patch; do
    patch=${ROOT}/patches/qwen38-sft/${name}
    if git -C "${BRIDGE}" apply --reverse --check "${patch}" 2>/dev/null; then
        echo "Already applied: ${name}"
    elif git -C "${BRIDGE}" apply --check "${patch}"; then
        echo "Ready to apply: ${name}"
        PENDING+=("${patch}")
    else
        echo "Patch conflicts: ${name}; inspect local Bridge changes first." >&2
        exit 1
    fi
done
if [[ ${MODE} == --apply ]]; then
    for patch in "${PENDING[@]}"; do
        git -C "${BRIDGE}" apply "${patch}"
    done
fi
