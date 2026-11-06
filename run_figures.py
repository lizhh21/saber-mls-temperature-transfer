#!/usr/bin/env python3
"""Rebuild main and SI figures entirely offline from the companion data archive."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from grlfig.data import Data
from grlfig.style import setup
from grlfig.main import figure1,figure2,figure3,figure4
from grlfig.si import all_si
from grlfig.verify import validate

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data',type=Path,default=ROOT.parent/'GRL_Figure_Data_v2')
    a.add_argument('--out',type=Path,default=ROOT/'build')
    a.add_argument('--figures',choices=['all','main','si','none'],default='all')
    a.add_argument('--dpi',type=int,default=600)
    args=a.parse_args()
    setup();d=Data(args.data);args.out.mkdir(parents=True,exist_ok=True)
    tables=args.out/'tables';tables.mkdir(exist_ok=True)
    monthly=d.monthly_pressure();hunga=d.hunga()
    monthly.to_csv(tables/'monthly_pressure_all_budgets.csv',index=False,float_format='%.15g')
    hunga.to_csv(tables/'Hunga_all_budgets_bands.csv',index=False,float_format='%.15g')
    d.w3_observed().to_csv(tables/'W3_recomputed_aligned_contrasts.csv',index=False,float_format='%.15g')
    report=validate(d,monthly,hunga)
    print('Numerical validation:',report['status'],report['numerical_values_checked'],'values',flush=True)
    out=args.out/'figures'
    if args.figures in ('all','main'):
        for n,fn in [(1,figure1),(2,figure2),(3,figure3),(4,figure4)]:
            print('Drawing Figure',n,flush=True)
            fn(d,out,args.dpi,monthly) if n==3 else fn(d,out,args.dpi)
    if args.figures in ('all','si'):
        print('Drawing SI S1–S12',flush=True);all_si(d,out,args.dpi,monthly,hunga)
    report['figure_selection']=args.figures;report['formats']=['png','pdf','svg'];report['png_dpi']=args.dpi
    generated=sorted(list(out.glob('*'))+list(tables.glob('*')))
    report['output_sha256']={str(p.relative_to(args.out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in generated if p.is_file()}
    (args.out/'BUILD_REPORT.json').write_text(json.dumps(report,indent=2))
    print('Completed:',args.out,flush=True)

if __name__=='__main__':main()
