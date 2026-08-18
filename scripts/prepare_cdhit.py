#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare FASTA and CD-HIT command for redundancy controls."""
import argparse, re
from pathlib import Path
import pandas as pd

def clean_seq(s): return re.sub(r'[^ACGTUacgtu]','',str(s)).upper().replace('U','T')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--id_col',default='GeneName'); ap.add_argument('--label_col',default='Label'); ap.add_argument('--seq_col',default='Sequence'); ap.add_argument('--length_col',default='Length'); ap.add_argument('--identity',default='0.8'); args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True); df=pd.read_csv(args.input); fasta=outdir/'input_for_cdhit.fa'; meta=outdir/'input_for_cdhit_metadata.csv'; rec=[]
    with open(fasta,'w') as f:
        for i,row in df.iterrows():
            gid=str(row.get(args.id_col,f'seq{i}')).replace(' ','_'); lab=str(row.get(args.label_col,'NA')).replace(' ','_'); seq=clean_seq(row.get(args.seq_col,''))
            if not seq: continue
            sid=f'{i}|{gid}|{lab}'; f.write(f'>{sid}\n{seq}\n'); rec.append(dict(row_index=i,cdhit_id=sid,GeneName=gid,Label=lab,Length=len(seq)))
    pd.DataFrame(rec).to_csv(meta,index=False)
    cmd=f'cd-hit-est -i {fasta} -o {outdir}/cdhit_id{args.identity}.fa -c {args.identity} -n 5 -M 0 -T 16 -d 0'
    (outdir/'run_cdhit_command.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+cmd+'\n',encoding='utf-8')
    print('[OK] FASTA:',fasta); print('[OK] metadata:',meta); print('[NEXT] Run:'); print(cmd); print('[NEXT] Then split by clusters from:',f'{outdir}/cdhit_id{args.identity}.fa.clstr')
if __name__=='__main__': main()
