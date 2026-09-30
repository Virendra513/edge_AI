#!/usr/bin/env bash
# Copy the chosen GGUF variant into android/app/src/main/assets/models/model.gguf.
#
# Usage:
#   scripts/copy_model.sh Q2_K            # default
#   scripts/copy_model.sh Q4_K_M
#   scripts/copy_model.sh                 # defaults to Q2_K
#
# The Gradle build also runs this internally via the `copyModel` task
# (COPY_VARIANT property). This script is for one-off / CI use.

set -euo pipefail

VARIANT="${1:-Q4_K_M}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/../llama-3.2-1B-Instruct-gguf/Llama-3.2-1B-Instruct.${VARIANT}.gguf"
DST_DIR="${ROOT}/android/app/src/main/assets/models"
DST="${DST_DIR}/model.gguf"

if [[ ! -f "${SRC}" ]]; then
    echo "GGUF not found: ${SRC}"
    echo "Available variants:"
    ls -1 "${ROOT}/../llama-3.2-1B-Instruct-gguf/" | sed 's/^/  /'
    exit 1
fi

mkdir -p "${DST_DIR}"
cp -f "${SRC}" "${DST}"

echo "Staged ${SRC}"
echo "        -> ${DST}"
echo "Size: $(du -h "${DST}" | cut -f1)"