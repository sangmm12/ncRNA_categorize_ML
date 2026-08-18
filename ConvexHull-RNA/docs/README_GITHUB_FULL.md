# ConvexHull-RNA

ConvexHull-RNA is a sequence-only framework for RNA representation, ncRNA family classification, coding/non-coding screening, and geometric analysis using accumulated convex-hull nucleotide vector (ACNV) features.

## Installation

```bash
conda create -n convexhull_rna python=3.11 -y
conda activate convexhull_rna
pip install -r requirements.txt
```

If `umap-learn` is incompatible with the installed `scikit-learn`, use:

```bash
pip install "scikit-learn==1.5.2" "umap-learn==0.5.6"
```

## Input format

```text
GeneName,Label,Length,Sequence
URS0002635E5E,RNase_P_RNA,425,GGAGTAAA...
```

For unknown labels, use `unknown` or leave the field blank.

## Prediction

```bash
python rna_predict_pipeline_v3_1.py \
  --input examples/example_input.csv \
  --out_prefix OUT/example_run \
  --xgb_model best_models_seed42_set10/rnacentral_best_xgb.pkl \
  --xgb_label_encoder best_models_seed42_set10/rnacentral_label_encoder.pkl
```

Outputs:

```text
OUT/example_run.predictions.csv
OUT/example_run.summary.txt
```

## Online web server

```bash
cd webapp
python app.py
```

For production deployment, use gunicorn and nginx:

```bash
gunicorn -w 2 -b 127.0.0.1:8080 app:app
```

## Reviewer-control analyses

### Length controls

```bash
python scripts/reviewer_length_controls_v2.py \
  --input rnacentral/rnacentral_active_label_set10_2w_seed42_NV1368.csv \
  --outdir reviewer_revision_results_missing/length_controls_v2_label10_seed42 \
  --x_start 5 \
  --label_col Label \
  --length_col Length \
  --seeds 42,43,44
```

### Convex-hull controls

For mammalian/multispecies files:

```bash
python scripts/reviewer_convex_hull_controls_v2.py \
  --input mammalian_multispecies/mammalian_train_dev_pre_NV1368.csv \
  --outdir reviewer_revision_results_followup/convex_hull_mammalian_xstart3 \
  --x_start 3 \
  --label_col Label \
  --dims 24,88,344,1368
```

For RNACentral files:

```bash
python scripts/reviewer_convex_hull_controls_v2.py \
  --input rnacentral/rnacentral_active_top20_2w_seed42_NV1368.csv \
  --outdir reviewer_revision_results_followup/convex_hull_top20_seed42 \
  --x_start 5 \
  --label_col Label \
  --dims 24,88,344,1368
```

### Binary cRNA/ncRNA visualization

```bash
bash 00_run_binary_visualization.sh
```

The default UMAP uses a balanced subset of 10,000 ENA sequences: 5,000 cRNA and 5,000 ncRNA.

### k-mer/BPE/fusion

```bash
python scripts/reviewer_merge_train_fused_features_v2.py \
  --kmer rnacentral/rnacentral_active_top20_2w_seed42_NV1368.csv \
  --bpe rnacentral/rnacentral_active_top20_2w_seed42_BPE1368.csv \
  --outdir reviewer_revision_results_missing/fused_top20_seed42 \
  --x_start 5 \
  --label_col Label
```

### CD-HIT redundancy control

```bash
bash reviewer_revision_results/cdhit_seed42/run_cdhit_command.sh
```

After `.clstr` is generated:

```bash
python scripts/reviewer_cdhit_cluster_split.py \
  --raw_csv rnacentral/rnacentral_active_label_set10_2w_seed42.csv \
  --clstr reviewer_revision_results/cdhit_seed42/cdhit_id0.8.fa.clstr \
  --outdir reviewer_revision_results_missing/cdhit_cluster_split_seed42 \
  --label_col Label
```

## Large files and models

Large datasets and trained weights are not committed directly to GitHub. Use Zenodo, Figshare, institutional storage, or GitHub Releases for:

- trained `.pkl` / `.h5` models
- large RNACentral/ENA feature matrices
- CD-HIT FASTA and cluster files
- large prediction outputs
