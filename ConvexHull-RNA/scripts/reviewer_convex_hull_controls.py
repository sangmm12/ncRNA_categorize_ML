#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convex-hull feasibility controls on sampled subsets."""
import argparse
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import linprog

def parse_int_list(s): return [int(x) for x in str(s).split(',') if x.strip()]
def sample_per_class(df,label_col,n,seed):
    return pd.concat([g.sample(n=min(len(g),n),random_state=seed) for _,g in df.groupby(label_col)]).sample(frac=1,random_state=seed).reset_index(drop=True)
def hulls_intersect(A,B):
    A=np.asarray(A,dtype=float); B=np.asarray(B,dtype=float); nA,d=A.shape; nB,dB=B.shape; assert d==dB
    c=np.zeros(nA+nB); Aeq=np.zeros((d+2,nA+nB)); Aeq[:d,:nA]=A.T; Aeq[:d,nA:]=-B.T; Aeq[d,:nA]=1; Aeq[d+1,nA:]=1
    beq=np.zeros(d+2); beq[d]=1; beq[d+1]=1
    res=linprog(c,A_eq=Aeq,b_eq=beq,bounds=[(0,None)]*(nA+nB),method='highs')
    return bool(res.success)
def disjoint_rate(X,y):
    labels=sorted(pd.unique(y)); total=0; inter=0; details=[]
    for a,b in combinations(labels,2):
        ok=hulls_intersect(X[y==a],X[y==b]); total+=1; inter+=int(ok); details.append(dict(label_a=a,label_b=b,intersect=ok))
    return dict(n_pairs=total,n_intersect=inter,n_disjoint=total-inter,disjoint_rate=(total-inter)/total if total else np.nan), pd.DataFrame(details)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--x_start',type=int,default=5)
    ap.add_argument('--label_col',default='Label'); ap.add_argument('--dims',default='24,88,344,1368'); ap.add_argument('--sample_per_class',type=int,default=80)
    ap.add_argument('--n_permutations',type=int,default=20); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(args.seed)
    df=sample_per_class(pd.read_csv(args.input).dropna(subset=[args.label_col]),args.label_col,args.sample_per_class,args.seed)
    feature_cols=list(df.columns[args.x_start:]); X_all=df[feature_cols].apply(pd.to_numeric,errors='coerce').fillna(0).values.astype(float); y=df[args.label_col].astype(str).values
    rows=[]; all_details=[]
    for dim in parse_int_list(args.dims):
        dim=min(dim,X_all.shape[1]); X=X_all[:,:dim]
        s,d=disjoint_rate(X,y); s.update(experiment='observed',dim=dim,iteration=0); rows.append(s); d['experiment']='observed'; d['dim']=dim; d['iteration']=0; all_details.append(d)
        Xr=rng.normal(size=X.shape); s,d=disjoint_rate(Xr,y); s.update(experiment='random_gaussian_features',dim=dim,iteration=0); rows.append(s); d['experiment']='random_gaussian_features'; d['dim']=dim; d['iteration']=0; all_details.append(d)
        for i in range(args.n_permutations):
            s,_=disjoint_rate(X,rng.permutation(y)); s.update(experiment='permuted_labels',dim=dim,iteration=i+1); rows.append(s)
    pd.DataFrame(rows).to_csv(outdir/'convex_hull_control_summary.csv',index=False)
    if all_details: pd.concat(all_details).to_csv(outdir/'convex_hull_pair_details.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print('[OK] wrote',outdir)
if __name__=='__main__': main()
