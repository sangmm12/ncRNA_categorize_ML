#!/usr/bin/env bash
set -euo pipefail

# Create a clean GitHub-ready directory from the current project directory.
# Run from /home/hgq/BioData/hgq/ncRNA
# Output: ConvexHull-RNA-GitHub/

SRC="$(pwd)"
DEST="${1:-ConvexHull-RNA-GitHub}"

mkdir -p "$DEST"/{scripts,examples,docs,webapp/templates,models_placeholder,results_placeholder}

for f in \
  step1_create_NV.py \
  step1_create_NV_BPE.py \
  step2_training_XGB_singleTry_plus.py \
  step2_training_MLP_ENA_continue.py \
  rna_predict_pipeline_v3_1.py \
  00_run_reviewer_revision_pipeline.sh \
  00_run_missing_reviewer_analyses.sh \
  00_run_followup_reviewer_scripts.sh \
  00_run_binary_visualization.sh
do
  if [[ -f "$SRC/$f" ]]; then cp "$SRC/$f" "$DEST/"; fi
done

if [[ -d "$SRC/scripts" ]]; then
  cp "$SRC"/scripts/reviewer_*.py "$DEST/scripts/" 2>/dev/null || true
fi

if [[ -d "$SRC/webapp" ]]; then
  cp "$SRC/webapp/app.py" "$DEST/webapp/" 2>/dev/null || true
  cp -r "$SRC/webapp/templates" "$DEST/webapp/" 2>/dev/null || true
fi

if [[ -d "$SRC/webapp/examples" ]]; then
  find "$SRC/webapp/examples" -maxdepth 1 -type f -name "*.csv" -size -5M -exec cp {} "$DEST/examples/" \;
fi

mkdir -p "$DEST/docs"
cp docs_revision_package/*.md "$DEST/docs/" 2>/dev/null || true

if [[ -f "docs_revision_package/README_GITHUB.md" ]]; then
  cp docs_revision_package/README_GITHUB.md "$DEST/README.md"
fi

cat > "$DEST/requirements.txt" <<'REQ'
numpy
pandas
scipy
scikit-learn
xgboost
tensorflow
joblib
tqdm
matplotlib
seaborn
flask
umap-learn
REQ

cat > "$DEST/.gitignore" <<'GIT'
__pycache__/
*.pyc
*.pkl
*.h5
*.joblib
*.npy
*.npz
*.csv
*.fa
*.clstr
OUT/
INPUT/
results/
reviewer_revision_results/
reviewer_revision_results_missing/
reviewer_revision_results_followup/
GIT

cat > "$DEST/models_placeholder/README.md" <<'MD'
# Model files

Large trained models are not committed to GitHub.

Expected files:
- `best_models/ENA_binary_classification_model.h5`
- `best_models/ENA_label_encoder.pkl`
- `best_models_seed42_set10/rnacentral_best_xgb.pkl`
- `best_models_seed42_set10/rnacentral_label_encoder.pkl`

Please download them from the release/Zenodo link described in the manuscript.
MD

echo "[OK] GitHub-ready directory created at: $DEST"
echo "Next:"
echo "  cd $DEST"
echo "  git init"
echo "  git add ."
echo "  git commit -m 'Initial ConvexHull-RNA reproducibility package'"
