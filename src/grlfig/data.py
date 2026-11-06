"""Aggregation rules. No training, inference, resampling, smoothing, or p-values."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

FRACTIONS = (.01,.1,1.)
BANDS = {'21_79':(21,79),'21_59':(21,59),'60_70':(60,70),'71_79':(71,79)}
SCOPES = ('2022','2023','2024','2025','2022_2024','2022_2025')

class Data:
    def __init__(self, root):
        self.root = Path(root)
        self.m = dict(np.load(self.root/'statistics/model_daily.npz',allow_pickle=False))
        self.b = dict(np.load(self.root/'statistics/baseline_daily.npz',allow_pickle=False))
        self.o = dict(np.load(self.root/'statistics/MLS_daily.npz',allow_pickle=False))
        self.models = json.loads(self.m['model_json'].item())
        self.baselines = json.loads(self.b['definitions_json'].item())
        self.days = self.m['days']
        self.year = np.array([int(d[:4]) for d in self.days])
        self.months = np.array([d[:7] for d in self.days])
        self.nodes = np.arange(20,80)
        self.p = self.csv('data/pressure/figure1b_pressure_stats.csv').pressure_hpa.to_numpy()
        self._cache={}

    def csv(self, name):
        return pd.read_csv(self.root/name)

    def daymask(self, scope):
        if len(str(scope))==7 and '-' in str(scope): return self.months==scope
        if '_' in str(scope):
            a,b = map(int,scope.split('_')); return (self.year>=a)&(self.year<=b)
        return self.year==int(scope)

    def windowmask(self, year, window):
        limits={'W1':('01-15','03-31'),'W2':('04-01','05-31'),'W3':('07-01','08-31')}
        lo,hi=limits[window]
        return (self.days>=f'{year}-{lo}') & (self.days<=f'{year}-{hi}')

    def bandmask(self, band):
        lo,hi=BANDS[band]; return (self.nodes>=lo)&(self.nodes<=hi)

    def indices(self, fraction, kind):
        if kind in ('pretrained','scratch'):
            return [i for i,m in enumerate(self.models) if m['training_fraction']==fraction and m['initialization']==kind]
        return [i for i,b in enumerate(self.baselines) if b['baseline']==kind and
                (b.get('training_fraction') is None or b['training_fraction']==fraction)]

    def seed_scores(self, fraction, kind, days, nodes, metric='rmse_k', profile=False):
        pack=self.m if kind in ('pretrained','scratch') else self.b
        idx=self.indices(fraction,kind)
        component={'rmse_k':0,'mae_k':1,'bias_k':2}[metric]
        vals=pack['sums'][idx][...,component][:,days][:,:,nodes].sum(axis=1)
        cnt=pack['count'][days][:,nodes].sum(axis=0)
        if not profile: vals=vals.sum(axis=-1);cnt=cnt.sum()
        with np.errstate(divide='ignore',invalid='ignore'): vals=vals/cnt
        return np.sqrt(vals) if metric=='rmse_k' else vals

    def score(self, fraction, kind, scope, band='21_79', metric='rmse_k', profile=False):
        key=(fraction,kind,scope,band,metric,profile)
        if key not in self._cache:
            self._cache[key]=self.seed_scores(fraction,kind,self.daymask(scope),self.bandmask(band),metric,profile).mean(axis=0)
        return self._cache[key]

    def monthly_pressure(self):
        rows=[]
        for f in FRACTIONS:
            for mo in np.unique(self.months):
                t=self.score(f,'pretrained',mo,profile=True)
                s=self.score(f,'scratch',mo,profile=True)
                b=self.score(f,'era5_offset',mo,profile=True)
                for z,p,tt,ss,bb in zip(self.nodes[1:],self.p[1:],t,s,b):
                    rows.append((f,mo,z,p,tt,ss,bb,tt-ss,tt-bb))
        return pd.DataFrame(rows,columns=['training_fraction','month','zstar','pressure_hpa','transfer_rmse_K','scratch_rmse_K','corrected_era5_rmse_K','transfer_minus_scratch_K','transfer_minus_corrected_era5_K'])

    def hunga(self):
        rows=[]
        for f in FRACTIONS:
            for y in (2022,2023,2024):
                for w in ('W1','W2','W3'):
                    dm=self.windowmask(y,w)
                    for band in BANDS:
                        t=self.seed_scores(f,'pretrained',dm,self.bandmask(band)).mean()
                        s=self.seed_scores(f,'scratch',dm,self.bandmask(band)).mean()
                        b=self.seed_scores(f,'era5_offset',dm,self.bandmask(band)).mean()
                        rows.append((f,y,w,band,t,s,b,t-s,t-b))
        return pd.DataFrame(rows,columns=['training_fraction','year','window','band','transfer_K','scratch_K','corrected_era5_K','transfer_minus_scratch_K','transfer_minus_corrected_era5_K'])

    def w3_observed(self):
        o=self.o; dates=o['dates_yyyymmdd'];pressure=o['pressure_hpa']
        with np.errstate(invalid='ignore',divide='ignore'): daily=o['sum_K']/o['count']
        def mean(y):
            return np.nanmean(daily[(dates>=y*10000+701)&(dates<=y*10000+831)],axis=0)
        anomaly=mean(2022)-(mean(2020)+mean(2021))/2
        rows=[]
        for lo,hi in [(10,50),(1,10),(.1,1),(.01,.1)]:
            select=(pressure>=lo)&(pressure<hi)
            model_select=(self.p>=lo)&(self.p<hi)&(self.nodes>=21)
            td=self.seed_scores(.01,'pretrained',self.windowmask(2022,'W3'),model_select)
            sd=self.seed_scores(.01,'scratch',self.windowmask(2022,'W3'),model_select)
            rows.append((f'{lo:g}-{hi:g} hPa',int(select.sum()),float(anomaly[select].mean()),float((td-sd).mean())))
        return pd.DataFrame(rows,columns=['pressure_band','pressure_nodes','observed_temperature_contrast_K','transfer_minus_scratch_RMSE_K'])
