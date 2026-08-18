#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write online model-card and repository README scaffolds."""
import argparse
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir',required=True); ap.add_argument('--model_name',default='ConvexHull-RNA'); ap.add_argument('--training_data',default=''); ap.add_argument('--model_variant',default=''); args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    model_card=f'''# {args.model_name}: online model card\n\n## Intended use\nThis web server provides sequence-only preliminary prediction for binary coding/non-coding RNA status and ncRNA family/subtype prediction. It is intended for exploratory annotation support and should not be treated as independent biological validation.\n\n## Deployed model variant\n{args.model_variant}\n\n## Training data\n{args.training_data}\n\n## Input format\nCSV text with header: GeneName,Label,Length,Sequence\n\n## Output\nThe server returns a summary report and a downloadable predictions CSV containing MLP binary label, probability of coding RNA, XGBoost ncRNA family label, and XGBoost family probability.\n\n## Caveats\nPredictions may be unreliable for sequences outside the training distribution. Database-derived labels can contain ambiguity or redundancy. High-confidence computational reassignment does not replace Rfam, structure, genomic-context, or manual curation.\n'''
    repo_readme='''# ConvexHull-RNA reproducibility README\n\n## Installation\n```bash\nconda create -n convexhull_rna python=3.11 -y\nconda activate convexhull_rna\npip install numpy pandas scipy scikit-learn xgboost tensorflow joblib tqdm matplotlib seaborn\n```\n\n## Basic pipeline\n1. Prepare input CSV with header: `GeneName,Label,Length,Sequence`.\n2. Create 5-mer ACNV/NV1368 features:\n```bash\npython step1_create_NV.py --input data/input.csv --output results/input_NV1368.csv\n```\n3. Train ncRNA family classifier:\n```bash\npython step2_training_XGB_singleTry_plus.py --input results/input_NV1368.csv --out results/train.log --x_start 5 --model_dir best_models\n```\n4. Predict using trained models:\n```bash\npython rna_predict_pipeline_v3_1.py --input examples/example.csv --out_prefix OUT/example\n```\n5. Reviewer-control analyses:\n```bash\nbash 00_run_reviewer_revision_pipeline.sh\n```\n'''
    (outdir/'ONLINE_MODEL_CARD.md').write_text(model_card,encoding='utf-8'); (outdir/'README_REPRODUCIBILITY.md').write_text(repo_readme,encoding='utf-8')
    print('[OK] wrote',outdir/'ONLINE_MODEL_CARD.md'); print('[OK] wrote',outdir/'README_REPRODUCIBILITY.md')
if __name__=='__main__': main()
