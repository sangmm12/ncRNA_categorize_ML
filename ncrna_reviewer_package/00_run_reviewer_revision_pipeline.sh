#!/usr/bin/env bash
set -euo pipefail

# Run from project root:
#   cd /home/hgq/BioData/hgq/ncRNA
#   conda activate tf
#   bash 00_run_reviewer_revision_pipeline.sh

OUT="reviewer_revision_results"
mkdir -p "$OUT"/{tables,figures,logs,intermediate}

# -------- USER-EDITABLE INPUTS --------
NV10="rnacentral/rnacentral_active_label_set10_2w_seed42_NV1368.csv"
BPE10=""
RAW10="rnacentral/rnacentral_active_label_set10_2w_seed42.csv"
X_START=5
LABEL_COL="Label"
SEQ_COL="Sequence"
LEN_COL="Length"
ID_COL="GeneName"
# --------------------------------------

echo "[1/6] Length controls"
python scripts/reviewer_length_controls.py \
  --input "$NV10" \
  --outdir "$OUT/length_controls_seed42" \
  --x_start "$X_START" \
  --label_col "$LABEL_COL" \
  --length_col "$LEN_COL" \
  --test_size 0.2 \
  --seeds 42,43,44 \
  --max_rows_per_class 5000 \
  | tee "$OUT/logs/length_controls.log"

echo "[2/6] High-dimensional controls"
python scripts/reviewer_highdim_controls.py \
  --input "$NV10" \
  --outdir "$OUT/highdim_controls_seed42" \
  --x_start "$X_START" \
  --label_col "$LABEL_COL" \
  --seeds 42,43,44 \
  --max_rows_per_class 4000 \
  --dims 24,88,344,1368 \
  --sample_sizes 200,500,1000,2000,4000 \
  | tee "$OUT/logs/highdim_controls.log"

echo "[3/6] Convex-hull feasibility controls"
python scripts/reviewer_convex_hull_controls.py \
  --input "$NV10" \
  --outdir "$OUT/convex_hull_controls_seed42" \
  --x_start "$X_START" \
  --label_col "$LABEL_COL" \
  --dims 24,88,344,1368 \
  --sample_per_class 80 \
  --n_permutations 20 \
  --seed 42 \
  | tee "$OUT/logs/convex_hull_controls.log"

if [[ -n "${BPE10}" && -f "${BPE10}" ]]; then
  echo "[4/6] Merge k-mer ACNV and BPE ACNV features"
  python scripts/reviewer_merge_train_fused_features.py \
    --kmer "$NV10" \
    --bpe "$BPE10" \
    --outdir "$OUT/fused_kmer_bpe_seed42" \
    --x_start "$X_START" \
    --label_col "$LABEL_COL" \
    --test_size 0.2 \
    --seed 42 \
    | tee "$OUT/logs/fused_kmer_bpe.log"
else
  echo "[4/6] Skip fused k-mer/BPE experiment: BPE10 is empty or missing."
fi

echo "[5/6] Export FASTA and CD-HIT commands for redundancy analysis"
python scripts/reviewer_prepare_cdhit.py \
  --input "$RAW10" \
  --outdir "$OUT/cdhit_seed42" \
  --id_col "$ID_COL" \
  --label_col "$LABEL_COL" \
  --seq_col "$SEQ_COL" \
  --length_col "$LEN_COL" \
  | tee "$OUT/logs/cdhit_prepare.log"

echo "[6/6] Write online platform model card and repo README scaffold"
python scripts/reviewer_write_docs.py \
  --outdir "$OUT/docs" \
  --model_name "ConvexHull-RNA online predictor" \
  --training_data "RNACentral label_set10 seed42, 5-mer ACNV/NV1368 for ncRNA subtype prediction; ENA balanced deduplicated cRNA/ncRNA data for binary MLP" \
  --model_variant "5-mer ACNV/NV1368 + XGBoost for ncRNA family; 5-mer ACNV/NV1368 + MLP for cRNA/ncRNA" \
  | tee "$OUT/logs/docs.log"

echo "[DONE] Reviewer revision outputs written to $OUT"
