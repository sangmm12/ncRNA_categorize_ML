#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.ensemble import HistGradientBoostingClassifier

def sample_per_class(df, label_col, max_rows, seed):
    if max_rows is None or max_rows <= 0:
        return df
    parts = []
    for _, g in df.groupby(label_col):
        parts.append(g.sample(n=min(len(g), max_rows), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

def train_eval(X, y, seed, name, test_size=0.2):
    le = LabelEncoder()
    yy = le.fit_transform(pd.Series(y).astype(str))
    Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=test_size, stratify=yy, random_state=seed)
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, random_state=seed)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {
        "experiment": name,
        "n_rows": len(y),
        "n_train": len(ytr),
        "n_test": len(yte),
        "n_classes": len(le.classes_),
        "n_features": X.shape[1],
        "accuracy": accuracy_score(yte, pred),
        "macro_f1": f1_score(yte, pred, average="macro"),
        "weighted_f1": f1_score(yte, pred, average="weighted"),
        "mcc": matthews_corrcoef(yte, pred),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmer", required=True)
    ap.add_argument("--bpe", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--x_start", type=int, default=5)
    ap.add_argument("--label_col", default="Label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--max_rows_per_class", type=int, default=3000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] reading k-mer:", args.kmer)
    k = pd.read_csv(args.kmer)
    print("[INFO] reading BPE:", args.bpe)
    b = pd.read_csv(args.bpe)

    if len(k) != len(b):
        raise ValueError(f"Row counts differ: kmer={len(k)}, bpe={len(b)}. Use matched files or merge by GeneName first.")

    key_cols = [c for c in ["GeneName", args.label_col, "Length", "Sequence"] if c in k.columns and c in b.columns]
    for c in key_cols:
        if not (k[c].astype(str).values == b[c].astype(str).values).all():
            raise ValueError(f"Column {c} differs between k-mer and BPE files. Please align rows first.")

    meta = k.iloc[:, :args.x_start].copy()
    meta["_row_index"] = np.arange(len(k))
    meta = sample_per_class(meta, args.label_col, args.max_rows_per_class, args.seed)
    idx = meta["_row_index"].values

    kcols = list(k.columns[args.x_start:])
    bcols = list(b.columns[args.x_start:])

    Xk = k.loc[idx, kcols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    Xb = b.loc[idx, bcols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    y = k.loc[idx, args.label_col].astype(str).values
    Xf = np.hstack([Xk, Xb])

    rows = [
        train_eval(Xk, y, args.seed, "kmer_only_sampled", args.test_size),
        train_eval(Xb, y, args.seed, "bpe_only_sampled", args.test_size),
        train_eval(Xf, y, args.seed, "kmer_plus_bpe_concat_sampled", args.test_size),
    ]
    res = pd.DataFrame(rows)
    res.to_csv(outdir / "fused_feature_metrics.csv", index=False)
    print(res.to_string(index=False))
    print("[OK] wrote", outdir)

if __name__ == "__main__":
    main()
