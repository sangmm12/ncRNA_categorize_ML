#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def parse_clstr(path):
    rows = []
    cluster = None
    pat = re.compile(r">([^\.]+)\.\.\.")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                cluster = int(line.split()[-1])
            else:
                m = pat.search(line)
                if m:
                    cdhit_id = m.group(1)
                    row_index = int(cdhit_id.split("|", 1)[0])
                    rows.append({"row_index": row_index, "cluster_id": cluster, "cdhit_id": cdhit_id})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", required=True)
    ap.add_argument("--clstr", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label_col", default="Label")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.raw_csv)
    cl = parse_clstr(args.clstr)
    raw = raw.reset_index().rename(columns={"index": "row_index"})
    m = raw.merge(cl[["row_index", "cluster_id"]], on="row_index", how="inner")
    clu = m.groupby("cluster_id")[args.label_col].agg(lambda x: x.value_counts().idxmax()).reset_index()
    train_c, test_c = train_test_split(clu["cluster_id"].values, test_size=args.test_size,
                                       random_state=args.seed, stratify=clu[args.label_col].values)
    m["split"] = np.where(m["cluster_id"].isin(test_c), "test", "train")
    m.to_csv(outdir / "cluster_split_all_rows.csv", index=False)
    m[m["split"]=="train"].drop(columns=["row_index"]).to_csv(outdir / "train_cluster_split.csv", index=False)
    m[m["split"]=="test"].drop(columns=["row_index"]).to_csv(outdir / "test_cluster_split.csv", index=False)
    summary = m.groupby(["split", args.label_col]).size().reset_index(name="n")
    summary.to_csv(outdir / "cluster_split_label_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("[OK] wrote", outdir)

if __name__ == "__main__":
    main()
