#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-dimensional controls: true labels, permuted labels, matched random features, PCA, sample-size sensitivity."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier

def parse_int_list(s): return [int(x) for x in str(s).split(',') if x.strip()]
def sample_per_class(df, label_col, max_rows, seed):
    if not max_rows or max_rows <= 0: return df
    return pd.concat([g.sample(n=min(len(g), max_rows), random_state=seed) for _, g in df.groupby(label_col)]).sample(frac=1, random_state=seed).reset_index(drop=True)
def eval_once(X, y, seed, name):
    le = LabelEncoder(); yy = le.fit_transform(pd.Series(y).astype(str))
    Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.2, stratify=yy, random_state=seed)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed)
    clf.fit(Xtr, ytr); pred = clf.predict(Xte)
    return dict(experiment=name, seed=seed, n_train=len(ytr), n_test=len(yte), n_classes=len(le.classes_), n_features=X.shape[1], accuracy=accuracy_score(yte,pred), macro_f1=f1_score(yte,pred,average='macro'), weighted_f1=f1_score(yte,pred,average='weighted'), mcc=matthews_corrcoef(yte,pred))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--input', required=True); ap.add_argument('--outdir', required=True)
    ap.add_argument('--x_start', type=int, default=5); ap.add_argument('--label_col', default='Label')
    ap.add_argument('--seeds', default='42,43,44'); ap.add_argument('--dims', default='24,88,344,1368')
    ap.add_argument('--sample_sizes', default='200,500,1000,2000,4000'); ap.add_argument('--max_rows_per_class', type=int, default=4000)
    args = ap.parse_args(); outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df0 = pd.read_csv(args.input).dropna(subset=[args.label_col]).copy(); feature_cols=list(df0.columns[args.x_start:]); rows=[]
    for seed in parse_int_list(args.seeds):
        df=sample_per_class(df0,args.label_col,args.max_rows_per_class,seed)
        X_all=df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32); y=df[args.label_col].astype(str).values; rng=np.random.default_rng(seed)
        for dim in parse_int_list(args.dims):
            dim=min(dim,X_all.shape[1]); X_dim=X_all[:,:dim]
            rows.append(eval_once(X_dim,y,seed,f'true_features_first_{dim}'))
            rows.append(eval_once(X_dim,rng.permutation(y),seed,f'label_permutation_first_{dim}'))
            rows.append(eval_once(rng.normal(size=X_dim.shape).astype(np.float32),y,seed,f'random_gaussian_matched_{dim}'))
            if dim < X_all.shape[1] and X_all.shape[0] > dim + 5:
                Xp=PCA(n_components=dim,random_state=seed).fit_transform(StandardScaler().fit_transform(X_all))
                rows.append(eval_once(Xp,y,seed,f'pca_matched_{dim}'))
        for n in parse_int_list(args.sample_sizes):
            ds=pd.concat([g.sample(n=min(len(g),n),random_state=seed) for _,g in df.groupby(args.label_col)]).sample(frac=1,random_state=seed)
            Xs=ds[feature_cols].apply(pd.to_numeric,errors='coerce').fillna(0).values.astype(np.float32); ys=ds[args.label_col].astype(str).values
            rows.append(eval_once(Xs,ys,seed,f'sample_size_{n}_per_class_full'))
    res=pd.DataFrame(rows); res.to_csv(outdir/'highdim_control_metrics.csv',index=False); print(res.to_string(index=False)); print('[OK] wrote',outdir)
if __name__=='__main__': main()
