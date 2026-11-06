"""SI S1–S12 from the same frozen sources, with their existing scientific inventory."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.lines import Line2D
from .data import FRACTIONS,BANDS
from .style import *

def finish(fig,out,n,dpi):save(fig,out,f'Figure_S{n}',dpi)
def boot(d): return d.csv('data/evidence/original_paired_block_bootstrap.csv')
def pairlegend(fig,y=.99):
    fig.legend(handles=[Line2D([0],[0],color=TEAL,marker='o',label='7-day blocks'),Line2D([0],[0],color=RUST,marker='s',label='14-day blocks')],loc='upper center',bbox_to_anchor=(.5,y),ncol=2,frameon=False)

def s1(d,out,dpi):
    fig=plt.figure(figsize=(7.2,6.0));g=fig.add_gridspec(2,3,left=.13,right=.97,bottom=.17,top=.87,height_ratios=[2.4,1],hspace=.7,wspace=.4)
    ints=boot(d); years=[2022,2023,2024,2025];bands=['21_59','60_70','71_79']
    for j,f in enumerate(FRACTIONS):
        ax=fig.add_subplot(g[0,j]);title(ax,chr(97+j),f'{int(f*100)}% training');zero(ax);clean(ax)
        for k,band in enumerate(bands):
            for i,year in enumerate(years):
                for block,color,off,m in [(7,TEAL,-.09,'o'),(14,RUST,.09,'s')]:
                    r=ints[(ints.training_fraction==f)&(ints.scope==str(year))&(ints.band==band)&(ints.metric=='rmse_k')&(ints.block_days==block)].iloc[0]
                    interval(ax,r.delta_transfer_minus_scratch,r.ci95_low,r.ci95_high,k*4+i+off,color,marker=m)
        ax.set_yticks(range(12),[f'{y} / {b.replace("_","–")}' for b in bands for y in years] if j==0 else [])
        ax.set_ylim(11.65,-.65);ax.set_xlabel('ΔRMSE (K)')
        for y in [3.5,7.5]:ax.axhline(y,color=GRID,lw=.7)
    ax=fig.add_subplot(g[1,:]);title(ax,'d','Pooled 2022–2024 by reference-node band')
    vals=np.array([[d.score(f,'pretrained','2022_2024',b)-d.score(f,'scratch','2022_2024',b) for f in FRACTIONS] for b in bands])
    im=ax.imshow(vals,cmap=CMAP,vmin=-.5,vmax=.5,aspect='auto');ax.set_xticks(range(3),['1%','10%','100%']);ax.set_yticks(range(3),[b.replace('_','–') for b in bands]);ax.set_ylabel('Reference nodes z*')
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{vals[i,j]:+.3f}',ha='center',va='center',fontsize=7)
    fig.colorbar(im,cax=fig.add_axes([.35,.085,.38,.012]),orientation='horizontal',label='ΔRMSE (K)')
    pairlegend(fig);fig.text(.13,.015,'Unadjusted exploratory intervals conditional on three fitted seed pairs. Negative favors Transfer.',fontsize=6.5,color=GREY)
    finish(fig,out,1,dpi)

def s2(d,out,dpi):
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.8));fig.subplots_adjust(left=.145,right=.98,top=.88,bottom=.13,hspace=.75,wspace=.45)
    ints=boot(d);scopes=['2022','2023','2024','2025','2022_2024']
    for row,metric in enumerate(['mae_k','bias_k']):
        for j,f in enumerate(FRACTIONS):
            ax=axes[row,j];title(ax,chr(97+row*3+j),f'{int(f*100)}% · '+('MAE' if row==0 else 'Signed bias'));zero(ax);clean(ax)
            for i,sc in enumerate(scopes):
                for block,col,off,m in [(7,TEAL,-.08,'o'),(14,RUST,.08,'s')]:
                    r=ints[(ints.training_fraction==f)&(ints.scope==sc)&(ints.band=='21_79')&(ints.metric==metric)&(ints.block_days==block)].iloc[0]
                    interval(ax,r.delta_transfer_minus_scratch,r.ci95_low,r.ci95_high,i+off,col,marker=m)
            ax.set_yticks(range(5),['2022','2023','2024','2025','Pooled 22–24'] if j==0 else []);ax.set_ylim(4.65,-.65);ax.set_xlabel('Transfer − Scratch (K)')
    pairlegend(fig);fig.text(.1,.03,'Ordinary exploratory intervals; a lower signed bias does not necessarily imply a smaller absolute bias.',fontsize=6.3,color=GREY);finish(fig,out,2,dpi)

def s3(d,out,dpi):
    fig=plt.figure(figsize=(7.2,6.25));g=fig.add_gridspec(2,3,left=.09,right=.98,top=.94,bottom=.14,hspace=.75,wspace=.45)
    df=d.csv('data/pressure/sensitivity_pressure_stats.csv');bal=d.csv('data/pressure/balanced_yearmonth_pressure_stats.csv')
    kinds=['all_pairs','year_2020','year_2021','distance_le100km_time_le1h','distance_le100km_time_le0p5h','complete_at_all60_pressures']
    names=['All pairs (1,028)','2020 (515)','2021 (513)','≤100 km, ≤1 h (140)','≤100 km, ≤0.5 h (66)','All 60 nodes valid (1,005)']
    colors=[INK,TEAL,RUST,PURPLE,GOLD,'#629e9b']
    for j,(key,lab) in enumerate([('bias_k','Mean difference (K)'),('rmse_k','RMS difference (K)'),('correlation','Pearson r')]):
        ax=fig.add_subplot(g[0,j]);title(ax,chr(97+j),lab);pressure(ax,labels=j==0);ax.set_ylim(60,.01);clean(ax)
        for k,name,col in zip(kinds,names,colors):
            dd=df[df.subset==k];ax.plot(dd[key],dd.pressure_hpa,color=col,lw=.8,label=name)
        if j<2:ax.plot(bal['equal_yearmonth_bias_k' if j==0 else 'equal_yearmonth_rmse_k'],bal.pressure_hpa,color=GREY,lw=1,ls=':',label='Equal year-month weight')
        ax.set_xlabel(lab)
        if j==0:ax.axvline(0,color=GREY,lw=.6)
    ax=fig.add_subplot(g[1,:2]);title(ax,'d','Month by pressure mean difference')
    ym=d.csv('data/pressure/yearmonth_pressure_stats.csv')
    vals=ym.pivot(index='z_star_km',columns=['year','month'],values='bias_k').sort_index().to_numpy()
    im=heat(ax,vals,d.p,limit=8);ax.set_ylim(60,.01);ax.set_xticks([.5,6.5,12.5,18.5,23.5],['Jan 2020','Jul','Jan 2021','Jul','Dec']);fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.06,pad=.2,label='SABER − MLS (K)',extend='both')
    ax=fig.add_subplot(g[1,2]);title(ax,'e','Valid pair support');pressure(ax,labels=False);ax.set_ylim(60,.01);clean(ax)
    for k,col in zip(kinds,colors):
        dd=df[df.subset==k];ax.plot(dd.n/dd.subset_pairs,dd.pressure_hpa,color=col,lw=.8)
    ax.set_xlabel('Valid / selected pairs');ax.set_xlim(.95,1.005)
    fig.legend(handles=[Line2D([0],[0],color=c,lw=1,label=n) for c,n in zip(colors,names)]+[Line2D([0],[0],color=GREY,ls=':',label='Equal year-month weight')],loc='center',bbox_to_anchor=(.51,.508),ncol=3,frameon=False,fontsize=6.1)
    finish(fig,out,3,dpi)

def s4(d,out,dpi):
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.8));fig.subplots_adjust(left=.1,right=.98,top=.94,bottom=.14,hspace=.75,wspace=.38)
    off=d.csv('data/evidence/calibration_parameters.csv');a=axes[0,0];title(a,'a','Training-only pressure offsets');pressure(a);a.set_ylim(60,.01);clean(a)
    for f,col in zip(FRACTIONS,[TEAL,PURPLE,RUST]):
        tmp=off[off.training_fraction==f].pivot(index='z_star_km',columns='seed',values='offset_mls_minus_era5_k').sort_index()
        a.plot(tmp.mean(axis=1),d.p,color=col,label=f'{int(f*100)}%');a.fill_betweenx(d.p,tmp.min(axis=1),tmp.max(axis=1),color=col,alpha=.12)
    a.set_xlabel('MLS − final-input ERA5 (K)');a.legend(frameon=False,fontsize=6.5)
    m=d.csv('data/tables/monthly_model_baseline_comparisons.csv');m=m[(m.band=='21_79')&(m.initialization=='pretrained')]
    a=axes[0,1];title(a,'b','Monthly reference errors');clean(a,'y')
    for k in ['era5_last','source_climatology']:
        dd=m[(m.training_fraction==.01)&(m.baseline==k)].sort_values('scope');a.plot(range(48),dd.baseline_score,color=COLORS[k],label=LABELS[k],lw=.85)
    for f,col in zip(FRACTIONS,[TEAL,PURPLE,RUST]):
        dd=m[(m.training_fraction==f)&(m.baseline=='era5_offset')].sort_values('scope');a.plot(range(48),dd.baseline_score,color=col,label=f'ERA5 + offset {int(f*100)}%',lw=.9)
    a.set_xticks([0,12,24,36,47],['2022','2023','2024','2025','Dec']);a.set_ylabel('RMSE (K)');a.set_ylim(3,9.6);a.legend(frameon=False,fontsize=5.7,loc='upper left',ncol=2)
    a=axes[1,0];title(a,'c','Monthly reference comparisons');clean(a)
    wc=d.csv('data/tables/monthly_main_win_counts.csv');wc=wc[wc.initialization=='pretrained']
    for j,k in enumerate(['era5_last','source_climatology','era5_offset']):
        vals=[wc[(wc.training_fraction==f)&(wc.baseline==k)].iloc[0].lower_rmse_months for f in FRACTIONS]
        a.barh(np.arange(3)+j*.21-.21,vals,height=.19,color=COLORS[k],label=LABELS[k])
    a.set_yticks(range(3),['1%','10%','100%']);a.invert_yaxis();a.set_xlim(0,50);a.set_xlabel('Months with lower Transfer RMSE / 48');a.legend(frameon=False,fontsize=6,loc='upper left',bbox_to_anchor=(0,1.32),ncol=2)
    a=axes[1,1];title(a,'d','Node losses against corrected ERA5');clean(a,'y')
    pc=d.csv('data/tables/pressure_context_and_skill.csv');vals=[]
    for f in FRACTIONS:
        dd=pc[(pc.scope=='2022_2024')&(pc.training_fraction==f)&(pc.primary_score_level)];vals.append(int((dd.transfer_minus_offset_k>0).sum()))
    a.bar(range(3),vals,color=[TEAL,PURPLE,RUST],width=.65)
    for j,n in enumerate(vals):a.text(j,n+.5,f'{n} / 59',ha='center',fontsize=8)
    a.set_xticks(range(3),['1%','10%','100%']);a.set_ylim(0,24);a.set_ylabel('Nodes with higher Transfer RMSE')
    fig.text(.1,.03,'Seed envelopes in (a) are descriptive; counts of months or nodes are not independent significance tests.',fontsize=6.3,color=GREY);finish(fig,out,4,dpi)

def s5(d,out,dpi):
    fig=plt.figure(figsize=(7.2,5.9));g=fig.add_gridspec(3,1,left=.09,right=.98,top=.94,bottom=.13,height_ratios=[1.65,.8,.7],hspace=.75)
    src=d.csv('data/evidence/used_mls_sources.csv');src['month']=src.utc_day.str[:7]
    cnt=src.groupby('month').evaluated_profiles.sum().reindex(sorted(np.unique(d.months)),fill_value=0)
    ax=fig.add_subplot(g[0]);title(ax,'a','Evaluation coverage by month');ax.bar(range(48),cnt,color=TEAL,width=.8);clean(ax,'y');ax.set_ylabel('Profiles');ax.set_xticks([0,12,24,36,47],['Jan 2022','Jan 2023','Jan 2024','Jan 2025','Dec']);ax.set_xlim(-.7,47.7)
    for i,y in enumerate([2022,2023,2024,2025]):ax.text(i*12+5.5,cnt.max()*1.05,f'{y}: {int(cnt.iloc[i*12:(i+1)*12].sum()):,}',ha='center',fontsize=7)
    ax.set_ylim(0,cnt.max()*1.22)
    ax=fig.add_subplot(g[1]);title(ax,'b','MLS versions and coverage')
    arr=np.zeros((4,366));code={'06-01':1,'06-02':2,'06-04':3}
    for _,r in src.iterrows():
        dt=pd.Timestamp(r.utc_day);arr[dt.year-2022,dt.dayofyear-1]=code[r.version]
    for i,y in enumerate([2022,2023,2024,2025]):
        if y!=2024:arr[i,365]=np.nan
    cm=ListedColormap(['#ecece8',TEAL,GOLD,'#19876f']);ax.imshow(np.ma.masked_invalid(arr),cmap=cm,vmin=0,vmax=3,aspect='auto',extent=(.5,366.5,3.5,-.5),interpolation='nearest')
    ax.set_yticks(range(4),['2022','2023','2024','2025']);ax.set_xticks([1,91,182,274,365]);ax.set_xlabel('Day of year')
    ax.legend(handles=[Line2D([0],[0],marker='s',ls='',color=c,label=l) for c,l in zip(cm.colors,['No observation','v06.01','v06.02','v06.04'])],ncol=4,frameon=False,loc='lower right',bbox_to_anchor=(1,1.04),fontsize=5.7,handlelength=1,columnspacing=.8)
    ax=fig.add_subplot(g[2]);title(ax,'c','Training and validation profile references');clean(ax)
    ax.barh(range(3),[145,1446,14463],color=RUST,height=.56,label='2020 training')
    ax.barh(range(3),[14417]*3,left=[145,1446,14463],color='#dae3e8',height=.56,label='2021 validation')
    ax.set_yticks(range(3),['1%','10%','100%']);ax.invert_yaxis();ax.set_xlabel('Profiles from distinct years');ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.7,1.1),fontsize=6.5)
    finish(fig,out,5,dpi)

def s6(d,out,dpi):
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.6));fig.subplots_adjust(left=.08,right=.98,top=.93,bottom=.18,hspace=.85,wspace=.2)
    m=d.csv('data/tables/monthly_model_baseline_comparisons.csv');m=m[(m.band=='21_79')&(m.initialization=='pretrained')]
    for row,base in enumerate(['scratch','era5_offset']):
        for j,f in enumerate(FRACTIONS):
            ax=axes[row,j]
            if base=='scratch':
                v=np.array([d.score(f,'pretrained',mo)-d.score(f,'scratch',mo) for mo in sorted(np.unique(d.months))]).reshape(4,12)
            else:
                dd=m[(m.baseline==base)&(m.training_fraction==f)].sort_values('scope');v=dd.delta_model_minus_baseline.to_numpy().reshape(4,12)
            im=ax.imshow(v,cmap=CMAP,vmin=-(1 if row==0 else 2.5),vmax=(1 if row==0 else 2.5),aspect='auto');title(ax,chr(97+row*3+j),f'{int(f*100)}% vs '+('Scratch' if row==0 else 'correction'))
            ax.set_yticks(range(4),['2022','2023','2024','2025'] if j==0 else []);ax.set_xticks([0,2,4,6,8,10],['J','M','M','J','S','N'])
            for y,x in zip(*np.where(v>0)):ax.text(x,y,'+',ha='center',va='center',fontsize=7,color=INK)
        cb=fig.colorbar(im,ax=axes[row,:].tolist(),orientation='horizontal',fraction=.045,pad=.16,aspect=40,extend='both');cb.set_label('Transfer − comparator RMSE (K)',fontsize=7)
    fig.text(.08,.035,'All 48 months retained. Plus signs mark positive point estimates without significance tests.',fontsize=6.4,color=GREY);finish(fig,out,6,dpi)

def s7(d,out,dpi):
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.6));fig.subplots_adjust(left=.09,right=.98,top=.78,bottom=.23,wspace=.22)
    ints=boot(d)
    for j,f in enumerate(FRACTIONS):
        ax=axes[j];title(ax,chr(97+j),f'{int(f*100)}% budget');zero(ax);clean(ax)
        for i,year in enumerate([2022,2023,2024,2025]):
            for block,col,off,m in [(7,TEAL,-.07,'o'),(14,RUST,.07,'s')]:
                r=ints[(ints.scope==str(year))&(ints.band=='21_79')&(ints.metric=='rmse_k')&(ints.training_fraction==f)&(ints.block_days==block)].iloc[0]
                lo,hi=(r.ci95_low,r.ci95_high) if year==2025 else (r.familywise95_low,r.familywise95_high)
                interval(ax,r.delta_transfer_minus_scratch,lo,hi,i+off,col,filled=year!=2025,marker=m)
        ax.set_xlim(-.65,.17);ax.set_yticks(range(4),['2022','2023','2024','2025'] if j==0 else []);ax.set_ylim(3.65,-.65);ax.set_xlabel('ΔRMSE (K)')
    pairlegend(fig);fig.text(.09,.07,'2022–2024: Bonferroni familywise intervals across nine annual comparisons.',fontsize=6.5,color=GREY)
    fig.text(.09,.025,'2025: ordinary exploratory intervals (hollow markers). Negative favors Transfer.',fontsize=6.5,color=GREY);finish(fig,out,7,dpi)

def s8(d,out,dpi):
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.2));fig.subplots_adjust(left=.13,right=.98,top=.88,bottom=.22,wspace=.45)
    h=d.csv('data/frozen/Hunga_verified_14day_intervals.csv')
    for j,prefix in enumerate(['pointwise95','family9']):
        ax=axes[j];title(ax,chr(97+j),'Pointwise 95% intervals' if j==0 else 'Nine-window adjusted intervals');zero(ax);clean(ax)
        for i,r in h.iterrows():
            interval(ax,r.delta_scratch_k,r[prefix+'_low_k'],r[prefix+'_high_k'],i,TEAL if r[prefix+'_high_k']<0 else RUST,filled=r[prefix+'_high_k']<0)
        ax.set_yticks(range(9),[f'{r.year}  {r.window}' for _,r in h.iterrows()]);ax.set_ylim(8.65,-.65);ax.set_xlim(-1,.1);ax.set_xlabel('Transfer − Scratch RMSE (K)')
    fig.text(.13,.08,'1% training · Three fixed seed pairs · 14-day blocks · 100,000 archived resamples',fontsize=6.5,color=GREY)
    fig.text(.13,.035,'W1: 15 Jan–31 Mar · W2: Apr–May · W3: Jul–Aug. Bootstrap is not rerun by this repository.',fontsize=6.4,color=GREY);finish(fig,out,8,dpi)

def s9(d,out,dpi,hunga):
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.45));fig.subplots_adjust(left=.1,right=.98,top=.9,bottom=.16,hspace=.65,wspace=.3)
    for j,band in enumerate(BANDS):
        ax=axes.flat[j];title(ax,chr(97+j),'Reference nodes z* '+band.replace('_','–'));clean(ax,'y');ax.axhline(0,color=GREY,lw=.7,ls='--')
        for f,col in zip(FRACTIONS,[TEAL,RUST,GOLD]):
            dd=hunga[(hunga.band==band)&(hunga.training_fraction==f)].sort_values(['year','window']);ax.plot(range(9),dd.transfer_minus_scratch_K,color=col,marker='o',ms=2.5,label=f'{int(f*100)}%')
        ax.set_xticks(range(9),['W1','W2','W3']*3);ax.tick_params(axis='x',labelsize=6);ax.set_ylabel('ΔRMSE (K)')
        for x in [2.5,5.5]:ax.axvline(x,color=GREY,lw=.6)
        for x,y in [(1,'2022'),(4,'2023'),(7,'2024')]:ax.text(x,-.25,y,transform=ax.get_xaxis_transform(),ha='center',fontsize=6.5)
    fig.legend(handles=[Line2D([0],[0],color=c,marker='o',label=f'{int(f*100)}%') for f,c in zip(FRACTIONS,[TEAL,RUST,GOLD])],loc='upper center',ncol=3,frameon=False)
    fig.text(.1,.02,'Exploratory Hunga window point estimates; negative favors Transfer. No bandwise significance claim.',fontsize=6.3,color=GREY);finish(fig,out,9,dpi)

def s10(d,out,dpi):
    dd=d.csv('data/frozen/MLS_monthly.csv');dd=dd[dd.year<=2024]
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.1));fig.subplots_adjust(left=.09,right=.87,top=.87,bottom=.22,wspace=.17)
    for j,y in enumerate([2022,2023,2024]):
        ax=axes[j];df=dd[dd.year==y].sort_values(['zstar','month']);v=df.pivot(index='zstar',columns='month',values='MLS_difference_from_2020_2021_K').to_numpy()
        im=heat(ax,v,d.p[1:],limit=5);ax.set_title(str(y),fontweight='bold');ax.set_xticks(np.arange(0,12,2)+.5,['Jan','Mar','May','Jul','Sep','Nov']);ax.set_xlabel('Calendar month');ax.tick_params(labelleft=j==0);ax.set_ylabel('Pressure (hPa)' if j==0 else '')
    cb=fig.colorbar(im,cax=fig.add_axes([.9,.22,.014,.65]),extend='both');cb.set_label('Observed MLS contrast (K)',fontsize=7)
    fig.text(.09,.035,'Native v06-01 diagnostic; matching-month 2020–2021 reference. Incomplete 2025 native record excluded.',fontsize=6.2,color=GREY);finish(fig,out,10,dpi)

def s11(d,out,dpi,monthly):
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.25),gridspec_kw={'width_ratios':[1,1.15]});fig.subplots_adjust(left=.08,right=.89,top=.88,bottom=.23,wspace=.37)
    ax=axes[0];title(ax,'a','Monthly prediction error · 1% training');clean(ax,'y')
    for k in ['pretrained','scratch','era5_offset']:
        ax.plot(range(36),[d.score(.01,k,mo) for mo in sorted(np.unique(d.months))[:36]],color=COLORS[k],lw=1,label=LABELS[k])
    ax.set_xticks([0,6,12,18,24,30,35],['Jan 22','Jul','Jan 23','Jul','Jan 24','Jul','Dec']);ax.tick_params(axis='x',labelrotation=35);ax.set_ylabel('RMSE (K)');ax.legend(frameon=False,fontsize=6)
    ax=axes[1];title(ax,'b','Monthly vertical error contrast')
    df=monthly[(monthly.training_fraction==.01)&(monthly.month<'2025')];v=df.pivot(index='zstar',columns='month',values='transfer_minus_scratch_K').to_numpy()
    im=heat(ax,v,d.p[1:],limit=3);ax.set_xticks([.5,6.5,12.5,18.5,24.5,30.5],['Jan 22','Jul','Jan 23','Jul','Jan 24','Jul']);ax.tick_params(axis='x',labelrotation=35)
    cb=fig.colorbar(im,cax=fig.add_axes([.925,.23,.013,.65]),extend='both');cb.set_label('Transfer − Scratch RMSE (K)',fontsize=7)
    finish(fig,out,11,dpi)

def s12(d,out,dpi):
    dd=d.csv('data/frozen/ERA5_monthly_native_levels.csv');fig,axes=plt.subplots(1,3,figsize=(7.2,3.1));fig.subplots_adjust(left=.09,right=.87,top=.87,bottom=.22,wspace=.17)
    for j,y in enumerate([2022,2023,2024]):
        ax=axes[j];df=dd[dd.year==y];v=df.pivot(index='model_level',columns='month',values='ERA5_departure_from_2012_2021_K').sort_index().to_numpy()
        im=ax.pcolormesh(np.arange(13),np.arange(.5,51.5),v,cmap=CMAP,vmin=-8,vmax=8);ax.set_ylim(50.5,.5);ax.set_title(str(y),fontweight='bold');ax.set_xticks(np.arange(0,12,2)+.5,['Jan','Mar','May','Jul','Sep','Nov']);ax.set_yticks([1,10,20,30,40,50]);ax.set_xlabel('Calendar month');ax.tick_params(labelleft=j==0)
        if j==0:ax.set_ylabel('ERA5 L137 model level\n(level 1 at top)')
    cb=fig.colorbar(im,cax=fig.add_axes([.9,.22,.014,.65]),extend='both');cb.set_label('Departure from 2012–2021 mean (K)',fontsize=7)
    fig.text(.09,.035,'Matching-calendar-month reference; native levels 1–50, not MLS pressure nodes.',fontsize=6.4,color=GREY);finish(fig,out,12,dpi)

def all_si(d,out,dpi,monthly,hunga):
    for fun in [s1,s2,s3,s4,s5,s6,s7,s8]:fun(d,out,dpi)
    s9(d,out,dpi,hunga);s10(d,out,dpi);s11(d,out,dpi,monthly);s12(d,out,dpi)
