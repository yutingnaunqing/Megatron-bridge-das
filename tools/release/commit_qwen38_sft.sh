#!/usr/bin/env bash
# Preview by default. Only --commit stages files and creates a local commit.
set -euo pipefail
CURRENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${CURRENT_DIR}/../.." && pwd)
cd "${ROOT}"
MODE=${1:---preview}
case "${MODE}" in
    --preview|--commit) ;;
    *) echo "Usage: $0 [--preview|--commit]" >&2; exit 2 ;;
esac
mapfile -t FILES < tools/release/qwen38_sft_files.txt
for file in "${FILES[@]}"; do
    [[ -f ${file} ]] || { echo "Missing file: ${file}" >&2; exit 1; }
done
bash tools/release/apply_qwen38_vendor_patches.sh --check
while IFS= read -r file; do
    case "${file}" in
        src/megatron/bridge/recipes/__init__.py|src/megatron/bridge/training/gpt_step.py) ;;
        *) echo "Vendor change missing from this snapshot: ${file}" >&2; exit 1 ;;
    esac
done < <(git -C 3rdparty/Megatron-Bridge diff --name-only HEAD)
if [[ -n $(git -C 3rdparty/Megatron-Bridge ls-files --others --exclude-standard) ||
      -n $(git -C 3rdparty/Megatron-Bridge/3rdparty/Megatron-LM status --porcelain) ]]; then
    echo "Uncaptured vendor files found; review them before committing." >&2
    exit 1
fi
# Detect edits made after the vendor snapshot was prepared.
for entry in 'bridge-recipes.patch:src/megatron/bridge/recipes/__init__.py' \
             'bridge-mtp-loss-mask.patch:src/megatron/bridge/training/gpt_step.py'; do
    name=${entry%%:*}
    file=${entry#*:}
    cmp <(git -C 3rdparty/Megatron-Bridge diff --binary HEAD -- "${file}") \
        "patches/qwen38-sft/${name}"
done
echo "Files selected for this commit:"
printf '  %s\n' "${FILES[@]}"
git diff --stat -- "${FILES[@]}"
if [[ ${MODE} == --preview ]]; then
    echo "Preview only; nothing staged or committed. Run with --commit after review."
    exit 0
fi
if ! git diff --cached --quiet; then
    echo "The index already contains staged changes. Review them before running this script." >&2
    exit 1
fi
git add -- "${FILES[@]}"
# Unified patches contain required single-space context lines.
git diff --cached --check -- . ':(exclude)patches/qwen38-sft/*.patch'
git diff --cached --stat
git commit -m "feat(qwen38): preserve native text and VL SFT workflows with portable Bridge patches"
