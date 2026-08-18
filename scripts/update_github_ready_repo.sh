#!/usr/bin/env bash
set -euo pipefail

# Run from /home/hgq/BioData/hgq/ncRNA
SRC="$(pwd)"
DEST="${1:-ConvexHull-RNA-GitHub}"

mkdir -p "$DEST"/{scripts,docs,examples,webapp/templates,models_placeholder}

# Core scripts
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
  if [[ -f "$SRC/$f" ]]; then
    cp -v "$SRC/$f" "$DEST/"
  else
    echo "[WARN] missing $f"
  fi
done

# Reviewer scripts
if [[ -d "$SRC/scripts" ]]; then
  cp -v "$SRC"/scripts/reviewer_*.py "$DEST/scripts/" 2>/dev/null || true
fi

# Webapp
if [[ -f "$SRC/webapp/app.py" ]]; then
  cp -v "$SRC/webapp/app.py" "$DEST/webapp/"
fi
if [[ -d "$SRC/webapp/templates" ]]; then
  cp -rv "$SRC/webapp/templates" "$DEST/webapp/"
fi

# Small examples only
if [[ -d "$SRC/webapp/examples" ]]; then
  find "$SRC/webapp/examples" -maxdepth 1 -type f -name "*.csv" -size -5M -exec cp -v {} "$DEST/examples/" \;
fi

# Docs from generated package
if [[ -d "$SRC/convexhull_rna_writing_and_github_package/docs_ready" ]]; then
  cp -v "$SRC"/convexhull_rna_writing_and_github_package/docs_ready/*.md "$DEST/docs/"
elif [[ -d "$SRC/docs_ready" ]]; then
  cp -v "$SRC"/docs_ready/*.md "$DEST/docs/"
fi

if [[ -f "$DEST/docs/README_GITHUB_FULL.md" ]]; then
  cp -v "$DEST/docs/README_GITHUB_FULL.md" "$DEST/README.md"
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

echo "[OK] Updated GitHub-ready repo: $DEST"
echo
echo "To commit:"
echo "  cd $DEST"
echo "  git config user.name \"Your Name\""
echo "  git config user.email \"your.email@example.com\""
echo "  git branch -M main"
echo "  git add ."
echo "  git commit -m \"Initial ConvexHull-RNA reproducibility package\""
