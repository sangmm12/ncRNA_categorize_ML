#!/usr/bin/env bash
set -euo pipefail
OUT="reviewer_revision_results_followup"
mkdir -p "$OUT"/{logs,tables}

MAMMALIAN_NV="mammalian_multispecies/mammalian_train_dev_pre_NV1368.csv"
MULTISPECIES_NV="mammalian_multispecies/multispecies_train_dev_pre_NV1368_abs.csv"

if [[ -f "$MAMMALIAN_NV" ]]; then
  python scripts/reviewer_convex_hull_controls_v2.py --input "$MAMMALIAN_NV" --outdir "$OUT/convex_hull_mammalian" --x_start 5 --label_col Label --dims 24,88,344,1368 --sample_per_class 80 --n_resamples 5 --n_permutations 10 --seed 42 | tee "$OUT/logs/convex_hull_mammalian.log"
else
  echo "[WARN] Missing mammalian file: $MAMMALIAN_NV"
fi

if [[ -f "$MULTISPECIES_NV" ]]; then
  python scripts/reviewer_convex_hull_controls_v2.py --input "$MULTISPECIES_NV" --outdir "$OUT/convex_hull_multispecies" --x_start 5 --label_col Label --dims 24,88,344,1368 --sample_per_class 80 --n_resamples 5 --n_permutations 10 --seed 42 | tee "$OUT/logs/convex_hull_multispecies.log"
else
  echo "[WARN] Missing multispecies file: $MULTISPECIES_NV"
fi

NV10="rnacentral/rnacentral_active_label_set10_2w_seed42_NV1368.csv"
SPLIT="reviewer_revision_results_missing/cdhit_cluster_split_seed42/cluster_split_all_rows.csv"

if [[ -f "$NV10" && -f "$SPLIT" ]]; then
  python scripts/reviewer_train_on_cdhit_split_nv.py --nv_csv "$NV10" --split_csv "$SPLIT" --outdir "$OUT/cdhit_cluster_split_train_label10" --x_start 5 --label_col Label --split_col split --seed 42 | tee "$OUT/logs/cdhit_cluster_split_train.log"
else
  echo "[INFO] Skip cluster-split training because one file is missing:"
  echo "  $NV10"
  echo "  $SPLIT"
  echo "Run CD-HIT and reviewer_cdhit_cluster_split.py first."
fi
echo "[DONE] follow-up reviewer analyses written to $OUT"
