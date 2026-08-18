# Follow-up reviewer scripts

Adds:
1. Convex-hull controls on the original seven-family benchmark files.
2. Training/evaluation on the CD-HIT cluster-level split after CD-HIT finishes.

Usage:
```bash
cd /home/hgq/BioData/hgq/ncRNA
unzip ncrna_followup_reviewer_scripts.zip
cp -r ncrna_followup_reviewer_scripts/* .
conda activate tf
bash 00_run_followup_reviewer_scripts.sh
```

If the mammalian/multispecies file paths differ, edit `00_run_followup_reviewer_scripts.sh`.
