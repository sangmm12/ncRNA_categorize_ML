#!/usr/bin/env bash
set -euo pipefail

# Run from /home/hgq/BioData/hgq/ncRNA
# conda activate tf
OUT="reviewer_revision_results_missing"
mkdir -p "$OUT"/{logs,tables,intermediate}

NV10="rnacentral/rnacentral_active_label_set10_2w_seed42_NV1368.csv"
RAW10="rnacentral/rnacentral_active_label_set10_2w_seed42.csv"

NV20="rnacentral/rnacentral_active_top20_2w_seed42_NV1368.csv"
BPE20="rnacentral/rnacentral_active_top20_2w_seed42_BPE1368.csv"

X_START10=5
X_START20=5

echo "[A] Better length-control analysis for label_set10: length-only, full ACNV, drop-basic, residualized-by-log-length."
python scripts/reviewer_length_controls_v2.py \
  --input "$NV10" \
  --outdir "$OUT/length_controls_v2_label10_seed42" \
  --x_start "$X_START10" \
  --label_col Label \
  --length_col Length \
  --seeds 42,43,44 \
  --max_rows_per_class 5000 \
  --test_size 0.2 \
  | tee "$OUT/logs/length_controls_v2_label10.log"

echo "[B] k-mer vs BPE vs fused representation on available top20 seed42 data."
if [[ -f "$NV20" && -f "$BPE20" ]]; then
  python scripts/reviewer_merge_train_fused_features_v2.py \
    --kmer "$NV20" \
    --bpe "$BPE20" \
    --outdir "$OUT/fused_top20_seed42" \
    --x_start "$X_START20" \
    --label_col Label \
    --max_rows_per_class 3000 \
    --seed 42 \
    | tee "$OUT/logs/fused_top20_seed42.log"
else
  echo "[WARN] Missing NV20 or BPE20. Please check paths:"
  echo "  $NV20"
  echo "  $BPE20"
fi

echo "[C] Prepare cluster-level split after CD-HIT."
echo "If cd-hit-est is available, first run:"
echo "  bash reviewer_revision_results/cdhit_seed42/run_cdhit_command.sh"
echo "Then run:"
echo "  python scripts/reviewer_cdhit_cluster_split.py --raw_csv $RAW10 --clstr reviewer_revision_results/cdhit_seed42/cdhit_id0.8.fa.clstr --outdir $OUT/cdhit_cluster_split_seed42 --label_col Label --test_size 0.2 --seed 42"

echo "[D] Summarize current completed reviewer controls into one table."
python scripts/reviewer_collect_revision_metrics.py \
  --base reviewer_revision_results \
  --missing "$OUT" \
  --out "$OUT/tables/reviewer_revision_metric_summary.csv" \
  | tee "$OUT/logs/collect_revision_metrics.log"

echo "[DONE] Additional analyses written to $OUT"
