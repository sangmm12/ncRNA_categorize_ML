# Missing reviewer analyses after first run

This package adds three items not completed or not ideal in the first run:

1. Better length controls:
   - length-only baseline
   - full ACNV
   - full ACNV residualized by log(length)
   - drop first 8 basic count/position features
   - drop first 8 + residualized by log(length)

2. k-mer/BPE/fused experiment using available top20 seed42 files:
   - rnacentral_active_top20_2w_seed42_NV1368.csv
   - rnacentral_active_top20_2w_seed42_BPE1368.csv

3. CD-HIT cluster split parser:
   After running cd-hit-est, parse `.clstr` and produce train/test CSV split by cluster.

Usage:
```bash
cd /home/hgq/BioData/hgq/ncRNA
unzip ncrna_missing_reviewer_analyses.zip
cp -r ncrna_missing_reviewer_analyses/* .
conda activate tf
bash 00_run_missing_reviewer_analyses.sh
```
