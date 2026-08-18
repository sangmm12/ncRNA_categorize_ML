#!/usr/bin/env bash
set -euo pipefail
OUT="reviewer_revision_results_followup/binary_cRNA_ncRNA_visualization"
mkdir -p "$OUT" reviewer_revision_results_followup/logs
python scripts/reviewer_binary_visualization.py \
  --input ENA/ENA_deduped_NV1368_modified.csv \
  --outdir "$OUT" \
  --x_start 4 \
  --label_col Label \
  --length_col Length \
  --seq_col Sequence \
  --sample_per_class 5000 \
  --seed 42 \
  --do_umap 1 \
  | tee reviewer_revision_results_followup/logs/binary_cRNA_ncRNA_visualization.log
echo "[OK] binary visualization written to $OUT"
