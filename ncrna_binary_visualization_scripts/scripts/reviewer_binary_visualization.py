#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def sample_per_class(df, label_col, n, seed):
    parts = []
    for _, g in df.groupby(label_col):
        parts.append(g.sample(n=min(len(g), n), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

def gc_content(seq):
    s = str(seq).upper().replace("U", "T")
    n = sum(ch in "ACGT" for ch in s)
    if n == 0:
        return np.nan
    return (s.count("G") + s.count("C")) / n

def draw_scatter(df, x, y, label_col, title, out_png, out_pdf):
    plt.figure(figsize=(7.2, 5.8))
    labels = sorted(df[label_col].astype(str).unique())
    for lab in labels:
        d = df[df[label_col].astype(str) == lab]
        plt.scatter(d[x], d[y], s=5, alpha=0.45, label=lab, linewidths=0)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend(markerscale=3, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=400)
    plt.savefig(out_pdf)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--x_start", type=int, default=4)
    ap.add_argument("--label_col", default="Label")
    ap.add_argument("--length_col", default="Length")
    ap.add_argument("--seq_col", default="Sequence")
    ap.add_argument("--sample_per_class", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--do_umap", type=int, default=1)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df0 = pd.read_csv(args.input)
    df0 = df0.dropna(subset=[args.label_col]).copy()
    df = sample_per_class(df0, args.label_col, args.sample_per_class, args.seed)

    feature_cols = list(df.columns[args.x_start:])
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    Xz = StandardScaler().fit_transform(X)

    meta_cols = [c for c in ["GeneName", args.label_col, args.length_col, args.seq_col] if c in df.columns]
    meta = df[meta_cols].copy()
    if args.seq_col in df.columns:
        meta["GC"] = df[args.seq_col].map(gc_content)
    if args.length_col in df.columns:
        meta[args.length_col] = pd.to_numeric(meta[args.length_col], errors="coerce")

    pca = PCA(n_components=10, random_state=args.seed)
    Z = pca.fit_transform(Xz)
    pca_df = meta.copy()
    for i in range(Z.shape[1]):
        pca_df[f"PC{i+1}"] = Z[:, i]
    pca_df.to_csv(outdir / "binary_pca.csv", index=False)
    pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_)
    }).to_csv(outdir / "binary_pca_explained_variance.csv", index=False)

    draw_scatter(pca_df, "PC1", "PC2", args.label_col,
                 "cRNA vs ncRNA in ACNV PCA space",
                 outdir / "binary_pca_scatter.png",
                 outdir / "binary_pca_scatter.pdf")

    summary_cols = []
    if args.length_col in meta.columns:
        summary_cols.append(args.length_col)
    if "GC" in meta.columns:
        summary_cols.append("GC")
    if summary_cols:
        meta.groupby(args.label_col)[summary_cols].describe().to_csv(outdir / "binary_length_gc_summary.csv")

    if args.do_umap:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                                metric="euclidean", random_state=args.seed)
            U = reducer.fit_transform(Xz)
            umap_df = meta.copy()
            umap_df["UMAP1"] = U[:, 0]
            umap_df["UMAP2"] = U[:, 1]
            umap_df.to_csv(outdir / "binary_umap.csv", index=False)
            draw_scatter(umap_df, "UMAP1", "UMAP2", args.label_col,
                         "cRNA vs ncRNA in ACNV UMAP space",
                         outdir / "binary_umap_scatter.png",
                         outdir / "binary_umap_scatter.pdf")
            print("[OK] UMAP completed.")
        except Exception as e:
            print("[WARN] UMAP skipped or failed:", repr(e))
            print("[HINT] install with: pip install umap-learn")

    print("[OK] wrote", outdir)
    print("[INFO] sampled rows:", len(df))
    print("[INFO] label counts:")
    print(df[args.label_col].value_counts().to_string())

if __name__ == "__main__":
    main()
