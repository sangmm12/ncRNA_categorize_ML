# ConvexHull-RNA

ConvexHull-RNA is a sequence-only framework for RNA representation, ncRNA family classification, coding/non-coding screening and geometric analysis using accumulated convex-hull nucleotide vector (ACNV) features.

## Installation

```bash
conda create -n convexhull_rna python=3.11 -y
conda activate convexhull_rna
pip install numpy pandas scipy scikit-learn xgboost tensorflow joblib tqdm matplotlib seaborn flask umap-learn
```

## Input format

```text
GeneName,Label,Length,Sequence
URS0002635E5E,RNase_P_RNA,425,GGAGTAAA...
```

For unknown labels, use `unknown` or leave the field blank.

## Prediction pipeline

```bash
python rna_predict_pipeline_v3_1.py \
  --input examples/example_input.csv \
  --out_prefix OUT/example_run \
  --xgb_model best_models_seed42_set10/rnacentral_best_xgb.pkl \
  --xgb_label_encoder best_models_seed42_set10/rnacentral_label_encoder.pkl
```

Outputs:
- `OUT/example_run.predictions.csv`
- `OUT/example_run.summary.txt`

## Online web server

```bash
cd webapp
python app.py
```

For deployment, run Flask behind gunicorn and nginx:

```bash
gunicorn -w 2 -b 127.0.0.1:8080 app:app
```

## Reviewer-control examples

### Length controls

```bash
python scripts/reviewer_length_controls_v2.py \
  --input rnacentral/rnacentral_active_label_set10_2w_seed42_NV1368.csv \
  --outdir reviewer_revision_results_missing/length_controls_v2_label10_seed42 \
  --x_start 5 --label_col Label --length_col Length --seeds 42,43,44
```

### Convex-hull controls

```bash
python scripts/reviewer_convex_hull_controls_v2.py \
  --input mammalian_multispecies/mammalian_train_dev_pre_NV1368.csv \
  --outdir reviewer_revision_results_followup/convex_hull_mammalian_xstart3 \
  --x_start 3 --label_col Label --dims 24,88,344,1368
```

### Binary cRNA/ncRNA visualization

```bash
bash 00_run_binary_visualization.sh
```

The default UMAP uses 10,000 balanced samples: 5,000 cRNA and 5,000 ncRNA.

### k-mer/BPE/fusion

```bash
python scripts/reviewer_merge_train_fused_features_v2.py \
  --kmer rnacentral/rnacentral_active_top20_2w_seed42_NV1368.csv \
  --bpe rnacentral/rnacentral_active_top20_2w_seed42_BPE1368.csv \
  --outdir reviewer_revision_results_missing/fused_top20_seed42 \
  --x_start 5 --label_col Label
```

### CD-HIT redundancy control

```bash
bash reviewer_revision_results/cdhit_seed42/run_cdhit_command.sh
```

Then parse clusters and train on cluster split after `.clstr` is generated.
