# Binary cRNA/ncRNA visualization

Adds PCA/UMAP plots for the ENA binary cRNA/ncRNA ACNV dataset.

Usage:
```bash
cd /home/hgq/BioData/hgq/ncRNA
unzip ncrna_binary_visualization_scripts.zip
cp -r ncrna_binary_visualization_scripts/* .
conda activate tf
pip install umap-learn   # optional, only if UMAP is not installed
bash 00_run_binary_visualization.sh
```

Outputs:
`reviewer_revision_results_followup/binary_cRNA_ncRNA_visualization/`
