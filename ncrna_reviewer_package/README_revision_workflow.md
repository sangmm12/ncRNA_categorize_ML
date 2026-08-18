# ConvexHull-RNA reviewer revision package

This package implements the main experiments requested by the reviewers.

Run from:
  /home/hgq/BioData/hgq/ncRNA

Recommended:
  conda activate tf

Main workflows:
1. Length confounding controls
2. High-dimensional convex-hull controls
3. Random-feature and label-permutation controls
4. Sample-size sensitivity analysis
5. k-mer/BPE fused-feature experiment
6. Redundancy / CD-HIT split preparation
7. Online-platform model-card text and repository README scaffold

The scripts do not overwrite your original results. They write to:
  reviewer_revision_results/

Minimal command:
  bash 00_run_reviewer_revision_pipeline.sh

Edit the input paths at the top of 00_run_reviewer_revision_pipeline.sh.
