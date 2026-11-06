"""Independent recomputation against input tables, not a claim of refitting models."""
import hashlib,json,platform
import numpy as np
import pandas as pd
import matplotlib
from .data import FRACTIONS

def validate(d,monthly,hunga):
    checks=[]
    def same(a,b,label,atol=5e-9):
        aa=np.asarray(a,dtype=float);bb=np.asarray(b,dtype=float)
        if aa.shape!=bb.shape: raise AssertionError((label,aa.shape,bb.shape))
        if not np.array_equal(np.isfinite(aa),np.isfinite(bb)): raise AssertionError(label+' finite support')
        valid=np.isfinite(aa);err=np.abs(aa[valid]-bb[valid])
        if (err>atol).any():raise AssertionError((label,float(err.max())))
        checks.append({'check':label,'values':int(valid.sum()),'max_abs_error':float(err.max()) if err.size else 0.})
    assert np.array_equal(d.m['days'],d.b['days'])
    assert np.array_equal(d.m['count'],d.b['count'])
    assert len(d.models)==18 and len(d.p)==60 and len(d.p[1:])==59
    assert np.all(np.diff(d.p)<0)
    same(d.p[1:],d.o['pressure_hpa'],'Pressure grid in model and MLS statistics')
    model_hash=hashlib.sha256((d.root/'statistics/model_daily.npz').read_bytes()).hexdigest()
    assert model_hash==d.b['model_statistics_sha256'].item()
    sha=d.root/'metadata/INPUT_V1_MANIFEST_SHA256.json'
    raw=json.loads(sha.read_text())
    # The original v1 manifest covers docs as well; only retained payloads are checked here.
    items=raw.items() if isinstance(raw,dict) else []
    hashed=0
    for name,expect in items:
        if not name.startswith(('data/','statistics/','metadata/')):continue
        target=d.root/name
        if isinstance(expect,dict): expect=expect.get('sha256')
        assert target.is_file(),name
        assert hashlib.sha256(target.read_bytes()).hexdigest()==expect,name
        hashed+=1
    assert hashed>20
    for filename in ['data/tables/main_rmse_summary.csv','data/tables/annual_and_band_summary.csv']:
        df=d.csv(filename)
        for _,r in df.iterrows():
            f=r.training_fraction;sc=str(r.scope);band=r.band
            for col,k in [('era5_last_rmse_k','era5_last'),('source_climatology_rmse_k','source_climatology'),('era5_offset_rmse_k','era5_offset'),('scratch_rmse_k','scratch'),('transfer_rmse_k','pretrained')]:
                same(d.score(f,k,sc,band),r[col],filename+' '+col)
            t=d.score(f,'pretrained',sc,band);s=d.score(f,'scratch',sc,band);b=d.score(f,'era5_offset',sc,band)
            same([t-s,t-b,100*(s-t)/s,100*(b-t)/b],[r.transfer_minus_scratch_k,r.transfer_minus_offset_k,r.transfer_rmse_reduction_vs_scratch_pct,r.transfer_rmse_reduction_vs_offset_pct],filename+' contrasts')
            same(d.m['count'][d.daymask(sc)][:,d.bandmask(band)].sum(),r.valid_cells,filename+' support',atol=0)
    pc=d.csv('data/tables/pressure_context_and_skill.csv')
    for (sc,f),dd in pc.groupby(['scope','training_fraction']):
        dd=dd[dd.primary_score_level].sort_values('z_star_index')
        vals={k:d.score(f,k,str(sc),profile=True) for k in ['pretrained','scratch','era5_last','source_climatology','era5_offset']}
        for col,k in [('transfer_rmse_k','pretrained'),('scratch_rmse_k','scratch'),('era5_last_rmse_k','era5_last'),('source_climatology_rmse_k','source_climatology'),('era5_offset_rmse_k','era5_offset')]:same(vals[k],dd[col].to_numpy(),'Pressure profiles '+col)
        same(vals['pretrained']-vals['scratch'],dd.transfer_minus_scratch_k.to_numpy(),'Pressure pretraining contrast')
        same(vals['pretrained']-vals['era5_offset'],dd.transfer_minus_offset_k.to_numpy(),'Pressure corrected-reference contrast')
    m=d.csv('data/tables/monthly_model_baseline_comparisons.csv')
    for _,r in m.iterrows():
        a=d.score(r.training_fraction,r.initialization,r.scope,r.band,r.metric)
        b=d.score(r.training_fraction,r.baseline,r.scope,r.band,r.metric)
        same([a,b,a-b],[r.model,r.baseline_score,r.delta_model_minus_baseline],'Monthly model baseline comparison')
        same(d.m['count'][d.daymask(r.scope)][:,d.bandmask(r.band)].sum(),r.n,'Monthly support',atol=0)
    ints=d.csv('data/evidence/original_paired_block_bootstrap.csv')
    for _,r in ints.iterrows():
        a=d.score(r.training_fraction,'pretrained',r.scope,r.band,r.metric)
        b=d.score(r.training_fraction,'scratch',r.scope,r.band,r.metric)
        same(a-b,r.delta_transfer_minus_scratch,'Bootstrap point estimates')
        assert r.ci95_low<=r.ci95_high
    h=d.csv('data/frozen/Hunga_verified_14day_intervals.csv')
    for _,r in h.iterrows():
        q=hunga[(hunga.training_fraction==.01)&(hunga.year==r.year)&(hunga.window==r.window)&(hunga.band=='21_79')].iloc[0]
        same([q.transfer_K,q.scratch_K,q.corrected_era5_K,q.transfer_minus_scratch_K,q.transfer_minus_corrected_era5_K],
             [r.transfer_k,r.scratch_k,r.offset_k,r.delta_scratch_k,r.delta_offset_k],'Hunga primary estimates')
        assert r.family9_low_k<=r.delta_scratch_k<=r.family9_high_k
    w=d.w3_observed();ww=d.csv('data/frozen/W3_observed_temperature_and_model_RMSE_by_pressure.csv')
    same(w.observed_temperature_contrast_K.to_numpy(),ww.MLS_2022_W3_minus_2020_2021_K.to_numpy(),'W3 independently recomputed observed contrast')
    same(w.transfer_minus_scratch_RMSE_K.to_numpy(),ww.v16_Transfer_minus_Scratch_pooled_band_RMSE_K.to_numpy(),'W3 independently recomputed model contrast')
    # Observational monthly means must use equal valid-day weights, not pooled profile weights.
    daily=np.divide(d.o['sum_K'],d.o['count'],out=np.full_like(d.o['sum_K'],np.nan),where=d.o['count']>0)
    dates=d.o['dates_yyyymmdd'];year=dates//10000;month=(dates//100)%100
    om=d.csv('data/frozen/MLS_monthly.csv')
    for (y,mo),dd in om.groupby(['year','month']):
        dd=dd.sort_values('zstar');part=daily[(year==y)&(month==mo)]
        cur=np.divide(np.nansum(part,axis=0),np.isfinite(part).sum(axis=0),out=np.full(part.shape[1],np.nan),where=np.isfinite(part).sum(axis=0)>0)
        ref=(np.nanmean(daily[(year==2020)&(month==mo)],axis=0)+np.nanmean(daily[(year==2021)&(month==mo)],axis=0))/2
        same(cur,dd.MLS_mean_K.to_numpy(),'Monthly observed mean')
        same(ref,dd.MLS_reference_2020_2021_K.to_numpy(),'Monthly reference mean')
        same(cur-ref,dd.MLS_difference_from_2020_2021_K.to_numpy(),'Monthly observed contrast')
    annual={}
    for b in [7,14]:
        sel=ints[(ints.band=='21_79')&(ints.metric=='rmse_k')&(ints.primary_annual_family9)&(ints.block_days==b)]
        assert len(sel)==9
        annual[str(b)]=int((sel.familywise95_high<0).sum())
        assert annual[str(b)]==4
    assert len(h)==9 and int((h.family9_high_k<0).sum())==8
    t=d.score(.01,'pretrained','2022_2024');s=d.score(.01,'scratch','2022_2024')
    return {'status':'PASS','input_payload_hashes_checked':hashed,'numerical_values_checked':sum(x['values'] for x in checks),
        'maximum_absolute_error':max(x['max_abs_error'] for x in checks),'pooled_1pct_transfer_rmse_K':t,'pooled_1pct_scratch_rmse_K':s,
        'pooled_1pct_reduction_percent':100*(s-t)/s,'annual_supported_out_of_9':annual,'Hunga_supported_out_of_9':8,
        'monthly_pressure_cells':len(monthly),'monthly_pressure_full_range_K':[float(monthly.transfer_minus_scratch_K.min()),float(monthly.transfer_minus_scratch_K.max())],
        'monthly_pressure_cells_outside_main_color_limits':int((monthly.transfer_minus_scratch_K.abs()>1).sum()),
        'bootstrap_recomputed':False,'model_training_or_inference_performed':False,'pvalues_created':False,
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,
        'check_groups':pd.DataFrame(checks).groupby('check').agg(values=('values','sum'),max_abs_error=('max_abs_error','max')).reset_index().to_dict('records')}
