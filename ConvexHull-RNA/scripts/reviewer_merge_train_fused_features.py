#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train k-mer only, BPE only, and concatenated k-mer+BPE controls."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.ensemble import HistGradientBoostingClassifier

def train_eval(X,y,seed,name):
    le=LabelEncoder(); yy=le.fit_transform(pd.Series(y).astype(str)); Xtr,Xte,ytr,yte=train_test_split(X,yy,test_size=0.2,stratify=yy,random_state=seed)
    clf=HistGradientBoostingClassifier(max_iter=250,learning_rate=0.08,random_state=seed); clf.fit(Xtr,ytr); pred=clf.predict(Xte)
    return dict(experiment=name,n_features=X.shape[1],accuracy=accuracy_score(yte,pred),macro_f1=f1_score(yte,pred,average='macro'),weighted_f1=f1_score(yte,pred,average='weighted'),mcc=matthews_corrcoef(yte,pred))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--kmer',required=True); ap.add_argument('--bpe',required=True); ap.add_argument('--outdir',required=True)
    ap.add_argument('--x_start',type=int,default=5); ap.add_argument('--label_col',default='Label'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--test_size',type=float,default=0.2); args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True); k=pd.read_csv(args.kmer); b=pd.read_csv(args.bpe)
    key_cols=[c for c in ['GeneName',args.label_col,'Length','Sequence'] if c in k.columns and c in b.columns]
    if 'GeneName' in key_cols:
        m=k.merge(b,on=key_cols,suffixes=('_kmer','_bpe')); y=m[args.label_col]
        kcols=[c for c in m.columns if c.endswith('_kmer') and c.split('_kmer')[0].startswith('V')]; bcols=[c for c in m.columns if c.endswith('_bpe') and c.split('_bpe')[0].startswith('V')]
    else:
        if len(k)!=len(b): raise ValueError('No GeneName key and row counts differ; cannot merge safely.')
        y=k[args.label_col]; kcols=list(k.columns[args.x_start:]); bcols=list(b.columns[args.x_start:]); m=pd.concat([k[kcols].add_suffix('_kmer'),b[bcols].add_suffix('_bpe'),y.rename(args.label_col)],axis=1); kcols=[c for c in m.columns if c.endswith('_kmer')]; bcols=[c for c in m.columns if c.endswith('_bpe')]
    Xk=m[kcols].apply(pd.to_numeric,errors='coerce').fillna(0).values; Xb=m[bcols].apply(pd.to_numeric,errors='coerce').fillna(0).values; Xf=np.hstack([Xk,Xb])
    res=pd.DataFrame([train_eval(Xk,y,args.seed,'kmer_only'),train_eval(Xb,y,args.seed,'bpe_only'),train_eval(Xf,y,args.seed,'kmer_plus_bpe_concat')])
    res.to_csv(outdir/'fused_feature_metrics.csv',index=False); print(res.to_string(index=False)); print('[OK] wrote',outdir)
if __name__=='__main__': main()
