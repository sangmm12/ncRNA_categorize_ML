#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rna_predict_pipeline.py
----------------------

Two-stage RNA classifier (NV1368 -> MLP binary coding/noncoding -> XGB ncRNA type).

Input CSV format (comma separated, header required):
GeneName,Label,Length,Sequence

- Label can be true label; if unknown, set to "unknow"/"unknown"/"" (empty).

Outputs:
1) <out_prefix>.predictions.csv   (row-wise predictions + probabilities)
2) <out_prefix>.summary.txt       (label distributions + metrics)

This script intentionally reuses the same NV preprocessing logic as step1_create_NV.py:
- U -> T, keep only VALID_CHARS, ignore 'N' as "not invalid" but not included in cleaned sequence.
- NV1368 constructed from NV_8 + NV_16 + NV_64 + NV_256 + NV_1024 with kernel_flag=1 ("abs").
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Stage-1 model (Keras) ---
from tensorflow.keras.models import load_model
import pickle

# --- Stage-2 model (XGBoost joblib) ---
from joblib import load as joblib_load


def _load_step1_nv_module(step1_path: str):
    """
    Import step1_create_NV.py as a module so we exactly reuse its NV implementation.
    This avoids subtle inconsistencies.
    """
    import importlib.util
    step1_path = os.path.abspath(step1_path)
    spec = importlib.util.spec_from_file_location("step1_create_NV", step1_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import step1 module from: {step1_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _normalize_unknown_label(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "unknow", "unknown", "na", "nan", "none", "null"}:
        return ""
    return s


def _infer_binary_truth(label: str, coding_labels: List[str]) -> Optional[str]:
    """
    If true label exists, map it to "CODING" or "NONCODING" for evaluating stage-1.
    - If label matches any coding_labels (case-insensitive), return "CODING".
    - Else return "NONCODING".
    - If label is empty, return None.
    """
    if not label:
        return None
    low = label.lower()
    coding_set = {x.lower() for x in coding_labels if x}
    return "CODING" if low in coding_set else "NONCODING"


def _safe_softmax_like(probs: np.ndarray) -> np.ndarray:
    """
    XGB predict_proba should already be normalized; this is just a safety net
    for weird edge cases.
    """
    probs = np.asarray(probs, dtype=float)
    s = probs.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return probs / s


def run(args: argparse.Namespace) -> None:
    in_csv = Path(args.input)
    out_prefix = Path(args.out_prefix)

    # --- Load input ---
    df = pd.read_csv(in_csv)
    required = {"GeneName", "Label", "Length", "Sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Label"] = df["Label"].apply(_normalize_unknown_label)
    df["Sequence"] = df["Sequence"].astype(str)

    # --- Load NV module (exact reuse) ---
    step1 = _load_step1_nv_module(args.step1_create_nv)

    # Ensure the step1 module has the required functions/vars.
    if not hasattr(step1, "clean_sequence") or not hasattr(step1, "process_seq") or not hasattr(step1, "VALID_CHARS"):
        raise RuntimeError(
            "step1_create_NV.py must define clean_sequence(), process_seq(), and VALID_CHARS "
            "(see your uploaded step1_create_NV.py)."
        )

    # --- Build NV features ---
    X = np.zeros((len(df), 1368), dtype=np.float32)
    bad_rows: List[int] = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="NV1368"):
        try:
            seq = row["Sequence"]
            cleaned = step1.clean_sequence(i, seq.upper(), step1.VALID_CHARS)
            nv = step1.process_seq(cleaned)
            if nv is None or len(nv) != 1368:
                raise ValueError(f"NV length != 1368 (got {None if nv is None else len(nv)})")
            X[i, :] = np.asarray(nv, dtype=np.float32)
        except Exception as e:
            bad_rows.append(i)
            X[i, :] = np.nan
            if args.verbose:
                print(f"[WARN] row={i} GeneName={row.get('GeneName','')} NV failed: {e}")

    # Drop bad rows for prediction; keep them in output with NA preds
    good_mask = ~np.isnan(X).any(axis=1)
    X_good = X[good_mask]
    df_good = df.loc[good_mask].reset_index(drop=False).rename(columns={"index": "_orig_index"})

    # --- Load Stage-1 model + label encoder ---
    mlp_model = load_model(args.mlp_model)
    with open(args.mlp_label_encoder, "rb") as f:
        mlp_le = pickle.load(f)

    if len(getattr(mlp_le, "classes_", [])) != 2:
        # Still allow running, but warn loudly — sigmoid binary assumes 2 classes.
        print(f"[WARN] MLP label encoder has {len(mlp_le.classes_)} classes: {list(mlp_le.classes_)}")
        print("       The training script uses sigmoid+bce, so it should be 2-class. Please double-check.")

    # Determine which class corresponds to code=1 (positive prob from sigmoid)
    try:
        pos_label = mlp_le.inverse_transform([1])[0]
        neg_label = mlp_le.inverse_transform([0])[0]
    except Exception:
        pos_label = str(mlp_le.classes_[1]) if len(mlp_le.classes_) > 1 else "POS"
        neg_label = str(mlp_le.classes_[0]) if len(mlp_le.classes_) > 0 else "NEG"

    # Predict stage-1
    p_pos = mlp_model.predict(X_good, verbose=0).reshape(-1)
    p_pos = np.clip(p_pos, 1e-8, 1.0 - 1e-8)
    pred1_code = (p_pos >= args.mlp_thr).astype(int)
    pred1_label = mlp_le.inverse_transform(pred1_code)

    # Decide which label means "CODING" for gating stage-2
    # By default, we treat the encoder's "positive" class as CODING if it matches any coding_labels; else treat "negative" as CODING.
    coding_labels = [x.strip() for x in args.coding_labels.split(",") if x.strip()]
    pos_is_coding = (str(pos_label).lower() in {x.lower() for x in coding_labels})
    coding_label_str = str(pos_label) if pos_is_coding else str(neg_label)
    noncoding_label_str = str(neg_label) if pos_is_coding else str(pos_label)

    # Map MLP prob to "prob_coding" consistently
    prob_coding = p_pos if pos_is_coding else (1.0 - p_pos)

    # --- Load Stage-2 XGB + encoder ---
    xgb_model = joblib_load(args.xgb_model)
    xgb_le = joblib_load(args.xgb_label_encoder)  # was saved by joblib.dump in training

    # Always run stage-2 XGB for ALL rows (good rows), regardless of stage-1 decision
    proba2_all = xgb_model.predict_proba(X_good)
    proba2_all = _safe_softmax_like(np.asarray(proba2_all))
    pred2_code_all = np.argmax(proba2_all, axis=1)
    pred2_label_all = xgb_le.inverse_transform(pred2_code_all)
    pred2_prob_all = proba2_all[np.arange(len(pred2_code_all)), pred2_code_all]

    xgb_pred_label = pred2_label_all.astype(object)
    xgb_pred_prob = pred2_prob_all.astype(float)

    # --- Final label ---

    # --- Assemble output (include bad rows too) ---
    out_df = df.copy()
    out_df["mlp_pred_label"] = ""
    out_df["mlp_prob_coding"] = np.nan
    out_df["xgb_pred_label"] = ""
    out_df["xgb_pred_prob"] = np.nan

    # write good rows
    good_orig_idx = df_good["_orig_index"].to_numpy()
    out_df.loc[good_orig_idx, "mlp_pred_label"] = pred1_label.astype(str)
    out_df.loc[good_orig_idx, "mlp_prob_coding"] = prob_coding
    out_df.loc[good_orig_idx, "xgb_pred_label"] = xgb_pred_label.astype(str)
    out_df.loc[good_orig_idx, "xgb_pred_prob"] = xgb_pred_prob

    # Mark NV failures
    if bad_rows:
        out_df.loc[bad_rows, "mlp_pred_label"] = "NV_FAILED"
        out_df.loc[bad_rows, "xgb_pred_label"] = "NV_FAILED"

    if bad_rows:
        out_df.loc[bad_rows, ""] = "NV_FAILED"

    # --- Metrics ---
    summary_lines: List[str] = []

    def _vc(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        s = series.astype(str).fillna("").str.strip()
        s = s[s != ""]
        cnt = s.value_counts(dropna=False)
        prop = (cnt / max(len(s), 1)).round(4)
        return cnt, prop

    summary_lines.append(f"Input: {in_csv}")
    summary_lines.append(f"Rows: {len(df)}")
    summary_lines.append(f"NV failures: {len(bad_rows)}")
    summary_lines.append(f"Stage-1 coding label assumed: {coding_label_str}")
    summary_lines.append(f"Stage-1 noncoding label assumed: {noncoding_label_str}")
    summary_lines.append(f"Stage-1 threshold (prob_coding): {args.mlp_thr}")

    # Distributions (good rows only)
    cnt1, prop1 = _vc(pd.Series(pred1_label.astype(str)))
    summary_lines.append("")
    summary_lines.append("[Stage-1] mlp_pred_label distribution (good rows):")
    for k in cnt1.index.tolist():
        summary_lines.append(f"  {k}	{int(cnt1[k])}	{float(prop1[k]):.4f}")

    summary_lines.append("")
    summary_lines.append("[Stage-1] prob_coding stats (good rows):")
    summary_lines.append(f"  mean={float(np.nanmean(prob_coding)):.6f}  std={float(np.nanstd(prob_coding)):.6f}  "
                         f"min={float(np.nanmin(prob_coding)):.6f}  max={float(np.nanmax(prob_coding)):.6f}")

    cnt2, prop2 = _vc(out_df.loc[good_orig_idx, "xgb_pred_label"])
    summary_lines.append("")
    summary_lines.append("[Stage-2] xgb_pred_label distribution (good rows, always predicted):")
    for k in cnt2.index.tolist():
        summary_lines.append(f"  {k}	{int(cnt2[k])}	{float(prop2[k]):.4f}")

    # --- Accuracy per your rule ---
    # If true Label is in {ncRNA, cRNA} (case-insensitive), evaluate using stage-1 mlp_pred_label vs Label (exact match).
    # Otherwise, evaluate using stage-2 xgb_pred_label vs Label (exact match).
    label_true = out_df["Label"].astype(str).fillna("").str.strip()
    label_low = label_true.str.lower()
    is_binary_truth = label_low.isin({"ncrna", "crna"})

    good_mask_series = out_df.index.isin(good_orig_idx) & (out_df["mlp_pred_label"].astype(str) != "NV_FAILED")

    # Stage-1 exact match on binary-truth rows
    mask1 = good_mask_series & (label_true.str.len() > 0) & is_binary_truth
    if mask1.any():
        acc1 = float((out_df.loc[mask1, "mlp_pred_label"].astype(str) == label_true[mask1]).mean())
        summary_lines.append("")
        summary_lines.append(f"Accuracy (binary truth: compare mlp_pred_label vs Label) = {acc1:.4f} (n={int(mask1.sum())})")
        # confusion for binary labels (treat cRNA as positive)
        y_true = np.where(label_low[mask1] == "crna", 1, 0)
        y_pred = np.where(out_df.loc[mask1, "mlp_pred_label"].astype(str).str.lower() == "crna", 1, 0)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        summary_lines.append("Confusion (binary truth; positive=cRNA):")
        summary_lines.append(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # Stage-2 exact match on non-binary truth rows
    mask2 = good_mask_series & (label_true.str.len() > 0) & (~is_binary_truth)
    if mask2.any():
        acc2 = float((out_df.loc[mask2, "xgb_pred_label"].astype(str) == label_true[mask2]).mean())
        summary_lines.append("")
        summary_lines.append(f"Accuracy (non-binary truth: compare xgb_pred_label vs Label) = {acc2:.4f} (n={int(mask2.sum())})")

        # Also report accuracy restricted to labels inside XGB class space (often cleaner)
        xgb_classes = set([str(x) for x in getattr(xgb_le, "classes_", [])])
        mask2_in = mask2 & (label_true.isin(xgb_classes))
        if mask2_in.any():
            acc2_in = float((out_df.loc[mask2_in, "xgb_pred_label"].astype(str) == label_true[mask2_in]).mean())
            summary_lines.append(f"  (restricted to Label in XGB classes) = {acc2_in:.4f} (n={int(mask2_in.sum())})")

    # Convenience: overall accuracy using the rule above (where Label exists)
    mask_any = good_mask_series & (label_true.str.len() > 0)
    if mask_any.any():
        pred_rule = np.where(is_binary_truth, out_df["mlp_pred_label"].astype(str), out_df["xgb_pred_label"].astype(str))
        acc_all = float((pred_rule[mask_any] == label_true[mask_any]).mean())
        summary_lines.append("")
        summary_lines.append(f"Overall accuracy (rule-based per-label type) = {acc_all:.4f} (n={int(mask_any.sum())})")

    # Save
    # Save
    # Save
    pred_csv = out_prefix.with_suffix(".predictions.csv")
    summ_txt = out_prefix.with_suffix(".summary.txt")
    out_df.to_csv(pred_csv, index=False)
    summ_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Print minimal console summary
    print("\n".join(summary_lines))
    print(f"[OK] wrote: {pred_csv}")
    print(f"[OK] wrote: {summ_txt}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Two-stage RNA prediction (NV1368 -> MLP -> XGB).")
    p.add_argument("--input", required=True, help="Input CSV: GeneName,Label,Length,Sequence")
    p.add_argument("--out_prefix", required=True, help="Output prefix (no extension)")

    # NV
    p.add_argument("--step1_create_nv", default="step1_create_NV.py",
                   help="Path to step1_create_NV.py (used as an importable module)")

    # Stage-1 MLP (coding vs noncoding)
    p.add_argument("--mlp_model", default="best_models/ENA_binary_classification_model.h5",
                   help="Keras .h5 model from step2_training_MLP_ENA_continue.py")
    p.add_argument("--mlp_label_encoder", default="best_models/ENA_label_encoder.pkl",
                   help="Pickle label encoder from step2_training_MLP_ENA_continue.py")
    p.add_argument("--mlp_thr", type=float, default=0.5, help="Threshold on prob_coding for stage-1 decision")

    # Stage-2 XGB (ncRNA label)
    p.add_argument("--xgb_model", default="best_models/rnacentral_best_xgb.pkl",
                   help="Joblib model from step2_training_XGB_singleTry_plus.py")
    p.add_argument("--xgb_label_encoder", default="best_models/rnacentral_label_encoder.pkl",
                   help="Joblib label encoder from step2_training_XGB_singleTry_plus.py")

    # Evaluation / gating
    p.add_argument("--coding_labels", default="codingRNA,coding,protein_coding,mRNA,CDS",
                   help="Comma-separated label strings that should be treated as CODING ground truth.")
    p.add_argument("--verbose", action="store_true", help="Print per-row NV failures")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
