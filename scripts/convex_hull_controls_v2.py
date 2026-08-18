#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import linprog

def parse_int_list(s):
    return [int(x) for x in str(s).split(",") if x.strip()]

def sample_per_class(df, label_col, n, seed):
    parts = []
    for _, g in df.groupby(label_col):
        parts.append(g.sample(n=min(len(g), n), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

def hulls_intersect(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    nA, d = A.shape
    nB, dB = B.shape
    if d != dB:
        raise ValueError("Dimension mismatch")
    c = np.zeros(nA + nB)
    Aeq = np.zeros((d + 2, nA + nB))
    Aeq[:d, :nA] = A.T
    Aeq[:d, nA:] = -B.T
    Aeq[d, :nA] = 1.0
    Aeq[d + 1, nA:] = 1.0
    beq = np.zeros(d + 2)
    beq[d] = 1.0
    beq[d + 1] = 1.0
    bounds = [(0, None)] * (nA + nB)
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    return bool(res.success)

def disjoint_rate(X, y):
    labels = sorted(pd.unique(y))
    total = 0
    intersect = 0
    for a, b in combinations(labels, 2):
        ok = hulls_intersect(X[y == a], X[y == b])
        total += 1
        intersect += int(ok)
    return total, intersect, total - intersect, (total - intersect) / total if total else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--x_start", type=int, default=5)
    ap.add_argument("--label_col", default="Label")
    ap.add_argument("--dims", default="24,88,344,1368")
    ap.add_argument("--sample_per_class", type=int, default=80)
    ap.add_argument("--n_resamples", type=int, default=5)
    ap.add_argument("--n_permutations", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df0 = pd.read_csv(args.input).dropna(subset=[args.label_col]).copy()
    feature_cols = list(df0.columns[args.x_start:])
    rows = []
    for r in range(args.n_resamples):
        seed = args.seed + r
        df = sample_per_class(df0, args.label_col, args.sample_per_class, seed)
        X_all = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)
        y = df[args.label_col].astype(str).values
        for dim0 in parse_int_list(args.dims):
            dim = min(dim0, X_all.shape[1])
            X = X_all[:, :dim]
            for name, Xuse, yuse, it in [("observed", X, y, 0), ("random_gaussian_features", rng.normal(size=X.shape), y, 0)]:
                n_pairs, n_intersect, n_disjoint, rate = disjoint_rate(Xuse, yuse)
                rows.append({"dataset": Path(args.input).name, "experiment": name, "resample": r, "iteration": it, "dim": dim, "sample_per_class": args.sample_per_class, "n_classes": len(pd.unique(y)), "n_pairs": n_pairs, "n_intersect": n_intersect, "n_disjoint": n_disjoint, "disjoint_rate": rate})
            for p in range(args.n_permutations):
                yp = rng.permutation(y)
                n_pairs, n_intersect, n_disjoint, rate = disjoint_rate(X, yp)
                rows.append({"dataset": Path(args.input).name, "experiment": "permuted_labels", "resample": r, "iteration": p + 1, "dim": dim, "sample_per_class": args.sample_per_class, "n_classes": len(pd.unique(y)), "n_pairs": n_pairs, "n_intersect": n_intersect, "n_disjoint": n_disjoint, "disjoint_rate": rate})
    res = pd.DataFrame(rows)
    res.to_csv(outdir / "convex_hull_control_summary_v2.csv", index=False)
    print(res.groupby(["experiment", "dim"])["disjoint_rate"].agg(["mean", "std", "min", "max", "count"]).reset_index().to_string(index=False))
    print("[OK] wrote", outdir)

if __name__ == "__main__":
    main()
