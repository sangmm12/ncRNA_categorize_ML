#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, matthews_corrcoef
from sklearn.ensemble import HistGradientBoostingClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nv_csv", required=True)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--x_start", type=int, default=5)
    ap.add_argument("--label_col", default="Label")
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nv = pd.read_csv(args.nv_csv)
    nv = nv.reset_index().rename(columns={"index": "row_index"})
    split = pd.read_csv(args.split_csv)[["row_index", args.split_col, "cluster_id"]]
    m = nv.merge(split, on="row_index", how="inner")
    tr = m[m[args.split_col] == "train"].copy()
    te = m[m[args.split_col] == "test"].copy()
    if len(tr) == 0 or len(te) == 0:
        raise ValueError("Empty train or test split after merging.")

    feature_cols = list(nv.columns[args.x_start+1:])
    le = LabelEncoder()
    le.fit(m[args.label_col].astype(str))
    Xtr = tr[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    Xte = te[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    ytr = le.transform(tr[args.label_col].astype(str))
    yte = le.transform(te[args.label_col].astype(str))

    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, random_state=args.seed)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    metrics = {"n_train": len(ytr), "n_test": len(yte), "n_train_clusters": tr["cluster_id"].nunique(), "n_test_clusters": te["cluster_id"].nunique(), "n_classes": len(le.classes_), "accuracy": accuracy_score(yte, pred), "macro_f1": f1_score(yte, pred, average="macro"), "weighted_f1": f1_score(yte, pred, average="weighted"), "mcc": matthews_corrcoef(yte, pred)}
    pd.DataFrame([metrics]).to_csv(outdir / "cdhit_cluster_split_metrics.csv", index=False)
    report = classification_report(yte, pred, target_names=le.classes_, digits=4)
    (outdir / "cdhit_cluster_split_classification_report.txt").write_text(report, encoding="utf-8")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(report)
    print("[OK] wrote", outdir)

if __name__ == "__main__":
    main()
