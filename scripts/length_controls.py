#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Length-confounding controls: length-only baseline, full ACNV, basic-feature ablation, length-matched subset."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

def parse_seeds(s):
    return [int(x) for x in str(s).split(',') if x.strip()]

def metric_row(y_true, y_pred, name, seed, n_train, n_test):
    return dict(experiment=name, seed=seed, n_train=n_train, n_test=n_test,
                accuracy=accuracy_score(y_true, y_pred),
                macro_f1=f1_score(y_true, y_pred, average='macro'),
                weighted_f1=f1_score(y_true, y_pred, average='weighted'),
                mcc=matthews_corrcoef(y_true, y_pred))

def sample_per_class(df, label_col, max_rows, seed):
    if max_rows is None or max_rows <= 0:
        return df
    parts = [g.sample(n=min(len(g), max_rows), random_state=seed) for _, g in df.groupby(label_col)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

def make_length_matched(df, label_col, length_col, seed, n_bins=20):
    d = df.copy()
    d['_len_bin'] = pd.qcut(d[length_col].rank(method='first'), q=n_bins, duplicates='drop')
    counts = d.groupby(['_len_bin', label_col]).size().unstack(fill_value=0)
    parts = []
    rng = np.random.default_rng(seed)
    for bin_id, row in counts.iterrows():
        min_n = int(row[row > 0].min()) if (row > 0).any() else 0
        if min_n <= 1:
            continue
        for lab in row.index:
            g = d[(d['_len_bin'] == bin_id) & (d[label_col] == lab)]
            if len(g) >= min_n:
                parts.append(g.sample(n=min_n, random_state=int(rng.integers(0, 1000000))))
    if not parts:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(parts).drop(columns=['_len_bin']).sample(frac=1, random_state=seed).reset_index(drop=True)

def train_eval(X, y, seed, name, test_size):
    le = LabelEncoder(); yy = le.fit_transform(pd.Series(y).astype(str))
    Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=test_size, random_state=seed, stratify=yy)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed)
    clf.fit(Xtr, ytr)
    return metric_row(yte, clf.predict(Xte), name, seed, len(ytr), len(yte))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True); ap.add_argument('--outdir', required=True)
    ap.add_argument('--x_start', type=int, default=5); ap.add_argument('--label_col', default='Label')
    ap.add_argument('--length_col', default='Length'); ap.add_argument('--test_size', type=float, default=0.2)
    ap.add_argument('--seeds', default='42,43,44'); ap.add_argument('--max_rows_per_class', type=int, default=5000)
    ap.add_argument('--drop_first_n', type=int, default=8)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input).dropna(subset=[args.label_col, args.length_col]).copy()
    df[args.length_col] = pd.to_numeric(df[args.length_col], errors='coerce')
    df = df.dropna(subset=[args.length_col])
    feature_cols = list(df.columns[args.x_start:])
    df.groupby(args.label_col)[args.length_col].describe().to_csv(outdir / 'length_distribution_by_label.csv')
    rows = []
    for seed in parse_seeds(args.seeds):
        d = sample_per_class(df, args.label_col, args.max_rows_per_class, seed)
        y = d[args.label_col]
        rows.append(train_eval(d[[args.length_col]].values, y, seed, 'length_only', args.test_size))
        X_full = d[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        rows.append(train_eval(X_full, y, seed, 'full_acnv', args.test_size))
        if len(feature_cols) > args.drop_first_n:
            X_drop = d[feature_cols[args.drop_first_n:]].apply(pd.to_numeric, errors='coerce').fillna(0).values
            rows.append(train_eval(X_drop, y, seed, f'drop_first_{args.drop_first_n}_basic_features', args.test_size))
        dm = make_length_matched(d, args.label_col, args.length_col, seed)
        if len(dm) and dm[args.label_col].nunique() >= 2:
            Xm = dm[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
            rows.append(train_eval(Xm, dm[args.label_col], seed, 'length_matched_full_acnv', args.test_size))
    res = pd.DataFrame(rows); res.to_csv(outdir / 'length_control_metrics.csv', index=False)
    print(res.to_string(index=False)); print('[OK] wrote', outdir)
if __name__ == '__main__': main()
