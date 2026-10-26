#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# 전체 600k 난이도 라벨링 (규칙채점 타입, judge 타입 제외). 재개 가능.
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"; conda activate verovlm
cd ${VERO_REPO}/curriculum
LOG=${VERO_REPO}/curriculum/full_labeling_$(date +%Y%m%d_%H%M%S).log
echo "START $(date)" | tee "$LOG"
python3 difficulty_label.py \
  --tp 2 --per-ability 0 --k 4 --chunk 2000 \
  --out ${VERO_REPO}/curriculum/difficulty_labels_full.jsonl 2>&1 | \
  stdbuf -oL grep -vE "Loading safetensors|it/s\]|Adding requests|Processed prompts|Repo card" | tee -a "$LOG"
echo "DONE $(date)" | tee -a "$LOG"
