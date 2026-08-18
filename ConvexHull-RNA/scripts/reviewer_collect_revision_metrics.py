#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd

def maybe_read(path, label):
    p = Path(path)
    if p.exists():
        df = pd.read_csv(p)
        df.insert(0, "source_table", label)
        return df
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="reviewer_revision_results")
    ap.add_argument("--missing", default="reviewer_revision_results_missing")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tables = [
        maybe_read(Path(args.base)/"length_controls_seed42/length_control_metrics.csv", "length_control_v1"),
        maybe_read(Path(args.base)/"highdim_controls_seed42/highdim_control_metrics.csv", "highdim_control"),
        maybe_read(Path(args.base)/"convex_hull_controls_seed42/convex_hull_control_summary.csv", "convex_hull_control"),
        maybe_read(Path(args.missing)/"length_controls_v2_label10_seed42/length_control_v2_metrics.csv", "length_control_v2"),
        maybe_read(Path(args.missing)/"fused_top20_seed42/fused_feature_metrics.csv", "fused_top20"),
    ]
    tables = [x for x in tables if x is not None]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if tables:
        pd.concat(tables, ignore_index=True, sort=False).to_csv(out, index=False)
        print("[OK] wrote", out)
    else:
        print("[WARN] no input metrics found.")

if __name__ == "__main__":
    main()
