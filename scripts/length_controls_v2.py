#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression

def parse_seeds(s):
    return [int(x) for x in str(s).split(",") if x.strip()]

def sample_per_class(df, label_col, max_rows, seed):
    if max_rows is None or max_rows <= 0:
        return df
    parts = []
    for _, g in df.groupby(label_col):
        parts.append(g.sample(n=min(len(g), max_rows), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

def residualize_by_log_length(X_train, X_test, len_train, len_test):
    ztr = np.log1p(np.asarray(len_train, dtype=float)).reshape(-1, 1)
    zte = np.log1p(np.asarray(len_test, dtype=float)).reshape(-1, 1)
    reg = LinearRegression(n_jobs=-1)
    reg.fit(ztr, X_train)
    return X_train - reg.predict(ztr), X_test - reg.predict(zte)

def eval_split(X, y, length, seed, name, test_size=0.2, residualize=False):
    le = LabelEncoder()
    yy = le.fit_transform(pd.Series(y).astype(str))
    Xtr, Xte, ytr, yte, ltr, lte = train_test_split(X, yy, length, test_size=test_size, stratify=yy, random_state=seed)
    if residualize:
        Xtr, Xte = residualize_by_log_length(Xtr, Xte, ltr, lte)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {
        "experiment": name,
        "seed": seed,
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
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--x_start", type=int, default=5)
    ap.add_argument("--label_col", default="Label")
    ap.add_argument("--length_col", default="Length")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--max_rows_per_class", type=int, default=5000)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--drop_first_n", type=int, default=8)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df0 = pd.read_csv(args.input).dropna(subset=[args.label_col, args.length_col]).copy()
    df0[args.length_col] = pd.to_numeric(df0[args.length_col], errors="coerce")
    df0 = df0.dropna(subset=[args.length_col])
    feature_cols = list(df0.columns[args.x_start:])
    df0.groupby(args.label_col)[args.length_col].describe().to_csv(outdir / "length_distribution_by_label.csv")

    rows = []
    for seed in parse_seeds(args.seeds):
        df = sample_per_class(df0, args.label_col, args.max_rows_per_class, seed)
        y = df[args.label_col].astype(str).values
        length = df[args.length_col].values
        X_len = df[[args.length_col]].values.astype(np.float32)
        rows.append(eval_split(X_len, y, length, seed, "length_only", args.test_size))

        X_full = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        rows.append(eval_split(X_full, y, length, seed, "full_acnv", args.test_size))

        rows.append(eval_split(X_full, y, length, seed, "full_acnv_residualized_by_log_length", args.test_size, residualize=True))

        if len(feature_cols) > args.drop_first_n:
            X_drop = df[feature_cols[args.drop_first_n:]].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
            rows.append(eval_split(X_drop, y, length, seed, f"drop_first_{args.drop_first_n}_basic_features", args.test_size))
            rows.append(eval_split(X_drop, y, length, seed, f"drop_first_{args.drop_first_n}_residualized_by_log_length", args.test_size, residualize=True))

    res = pd.DataFrame(rows)
    res.to_csv(outdir / "length_control_v2_metrics.csv", index=False)
    print(res.to_string(index=False))
    print("[OK] wrote", outdir)

if __name__ == "__main__":
    main()
