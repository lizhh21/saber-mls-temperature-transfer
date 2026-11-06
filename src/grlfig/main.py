"""Four redesigned main figures. All scientific quantities come from archived data."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle
from matplotlib.lines import Line2D
from .data import FRACTIONS
from .style import *

ASSETS=Path(__file__).resolve().parents[2]/'assets'

def figure1(d,out,dpi):
    fig=plt.figure(figsize=(7.2,6.1))
    g=fig.add_gridspec(3,3,left=.075,right=.975,bottom=.06,top=.89,
        width_ratios=[2.85,1,1],height_ratios=[2.25,1.05,1.02],hspace=.52,wspace=.49)
    ax=fig.add_subplot(g[0,0]);title(ax,'a','Study region')
    ax.set_facecolor('#eaf1f3')
    for xy in json.loads((ASSETS/'archived_land_polygons.json').read_text()):
        xy=np.asarray(xy)
        if xy[:,0].max()<89 or xy[:,0].min()>131 or xy[:,1].max()<9 or xy[:,1].min()>32: continue
        ax.add_patch(Polygon(xy,facecolor='#d7e1d6',edgecolor='#91a69d',lw=.35,clip_on=True))
    ax.add_patch(Rectangle((91,11),38,19,facecolor=TEAL,alpha=.045,edgecolor='none'))
    ax.add_patch(Rectangle((91,11),38,19,facecolor='none',edgecolor=TEAL,lw=1.1))
    ax.set_xlim(89,131);ax.set_ylim(9,32);ax.set_aspect(1/np.cos(np.deg2rad(20.5)))
    ax.set_xticks([90,100,110,120,130],['90°E','100°E','110°E','120°E','130°E'])
    ax.set_yticks([10,20,30],['10°N','20°N','30°N'])
    ax.grid(color='white',lw=.5,alpha=.7)
    for x,y,t in [(111.5,26.5,'SOUTHERN ASIA'),(116.4,17.2,'South China\nSea'),(95,16.4,'Bay of\nBengal')]:
        ax.text(x,y,t,fontsize=6.6,ha='center',va='center')
    ax.text(.02,.02,'11–30°N  |  91–129°E',transform=ax.transAxes,fontsize=7,
        bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2})
    ax.text(.99,-.23,'Natural Earth land · 1:50m',transform=ax.transAxes,ha='right',fontsize=5.8,color=GREY)
    b=d.csv('data/pressure/figure1b_pressure_stats.csv')
    c=d.csv('data/pressure/figure1c_pressure_stats.csv')
    for j in [1,2]:
        ap=fig.add_subplot(g[:2,j]);pressure(ap);clean(ap)
        if j==1:
            title(ap,'b','SABER − MLS',pad=30)
            ap.axvline(0,color=GREY,lw=.6)
            ap.plot(b.bias_k,b.pressure_hpa,color=TEAL,label='Mean difference')
            ap.plot(b.rmse_k,b.pressure_hpa,color=RUST,label='RMS difference')
            ap.set_xlabel('Difference (K)');ap.set_xlim(-4.5,10.5);ap.set_xticks([-4,0,4,8])
            ap.legend(loc='lower left',bbox_to_anchor=(0,1.005),frameon=False,fontsize=5.5,handlelength=1.2,borderaxespad=0)
        else:
            title(ap,'c','Correlation',pad=30)
            ap.plot(c.correlation_raw,c.pressure_hpa,color=TEAL,label='Raw')
            ap.plot(c.correlation_yearmonth_demeaned,c.pressure_hpa,color=PURPLE,ls='--',label='Month demeaned')
            ap.set_xlim(.35,1);ap.set_xticks([.4,.6,.8,1]);ap.set_xlabel('Pearson r');ap.set_ylabel('')
            ap.legend(loc='lower left',bbox_to_anchor=(0,1.005),frameon=False,fontsize=5.3,handlelength=1.1,borderaxespad=0)
        # Includes the unscored 20th reference node as a development diagnostic.
        ap.set_ylim(60,.01)
    cov=d.csv('data/pressure/monthly_pairing_coverage.csv')
    cg=g[1,0].subgridspec(2,1,hspace=.15,height_ratios=[1.2,.8])
    a1=fig.add_subplot(cg[0]);title(a1,'d','Monthly coverage')
    x=np.arange(24)
    a1.plot(x,cov.mls_population,color=PURPLE,lw=1,label='MLS')
    a1.plot(x,cov.saber_population,color=TEAL,lw=1,label='SABER')
    a1.set_ylim(0,1450);a1.set_yticks([0,1000]);a1.set_ylabel('Profiles',fontsize=7)
    a1.tick_params(bottom=False,labelbottom=False);a1.set_xlim(-.6,23.6);clean(a1,'y')
    a1.legend(loc='upper right',bbox_to_anchor=(1.02,1.55),ncol=2,frameon=False,fontsize=5.8,handlelength=1.2,columnspacing=.7)
    a2=fig.add_subplot(cg[1],sharex=a1)
    a2.bar(x,cov.accepted_pairs,color=RUST,width=.72)
    a2.set_ylim(0,80);a2.set_yticks([0,60]);a2.set_ylabel('Pairs',fontsize=7)
    a2.set_xticks([0,6,12,18,23],['Jan 2020','Jul','Jan 2021','Jul','Dec'])
    a2.tick_params(axis='x',labelsize=6.2);clean(a2,'y')
    timeline=fig.add_subplot(g[2,:]);timeline.axis('off');title(timeline,'e','Observation use and target adaptation')
    blocks=[('SOURCE',TEAL,'2020–2021 SABER','13,759 profiles','Development and refit'),
            ('TRAIN',RUST,'2020 MLS','145 / 1,446 / 14,463','1% / 10% / 100%'),
            ('SELECT',PURPLE,'2021 MLS','14,417 profiles','Full validation year'),
            ('EVALUATE',INK,'2022–2025 MLS','56,930 profiles','All years retrospective')]
    for i,(h,col,t1,t2,t3) in enumerate(blocks):
        xx=i*.255
        timeline.plot([xx,xx+.23],[.94,.94],transform=timeline.transAxes,color=col,lw=2,solid_capstyle='butt')
        for yy,txt,fs,cl,weight in [(.78,h,7.3,col,'bold'),(.55,t1,7,INK,'normal'),(.32,t2,7,INK,'normal'),(.08,t3,6.3,GREY,'normal')]:
            timeline.text(xx,yy,txt,transform=timeline.transAxes,fontsize=fs,color=cl,fontweight=weight,va='center')
    fig.text(.075,.015,'1,028 accepted development pairs  ·  60 reference nodes; valid pairs vary by node (1,010–1,023).',fontsize=6.3,color=GREY)
    save(fig,out,'Figure_1',dpi)

def figure2(d,out,dpi):
    fig=plt.figure(figsize=(7.2,5.35))
    g=fig.add_gridspec(2,6,left=.09,right=.985,top=.89,bottom=.13,height_ratios=[1.18,1],hspace=.75,wspace=.85)
    a=fig.add_subplot(g[0,:3]);title(a,'a','Error versus target training budget');clean(a,'y')
    x=np.arange(3)
    for k in ['era5_last','source_climatology','era5_offset','scratch','pretrained']:
        vals=[d.score(f,k,'2022_2024') for f in FRACTIONS]
        a.plot(x,vals,color=COLORS[k],marker='o' if k in ('pretrained','scratch') else 'D',ms=4 if k in ('pretrained','scratch') else 2.8,
            lw=1.6 if k in ('pretrained','scratch') else .9,ls='--' if k in ('era5_last','source_climatology') else '-',label=LABELS[k])
    a.set_xticks(x,['1%','10%','100%']);a.set_ylabel('RMSE (K)');a.set_xlabel('MLS training budget');a.set_ylim(3.95,6.35)
    a.annotate('7.45% lower RMSE',xy=(0,d.score(.01,'pretrained','2022_2024')),xytext=(.6,4.75),color=TEAL,fontsize=7.3,
        arrowprops={'arrowstyle':'-','color':TEAL,'lw':.7},ha='left')
    b=fig.add_subplot(g[0,3:]);title(b,'b','Paired error contrasts');zero(b);clean(b)
    ints=d.csv('data/evidence/original_paired_block_bootstrap.csv')
    bs=d.csv('data/tables/baseline_interval_summary.csv')
    for i,f in enumerate(FRACTIONS):
        r=ints[(ints.scope=='2022_2024')&(ints.band=='21_79')&(ints.metric=='rmse_k')&(ints.block_days==14)&(ints.training_fraction==f)].iloc[0]
        interval(b,r.delta_transfer_minus_scratch,r.ci95_low,r.ci95_high,i-.10,TEAL,marker='o')
        r=bs[(bs.scope=='2022_2024')&(bs.band=='21_79')&(bs.initialization=='pretrained')&(bs.baseline=='era5_offset')&(bs.training_fraction==f)].iloc[0]
        interval(b,r.delta_model_minus_baseline,r.ci14_low,r.ci14_high,i+.10,PURPLE,marker='D')
    b.set_yticks(range(3),['1%','10%','100%']);b.set_ylim(3.3,-.55);b.set_xlim(-1.08,.055)
    b.set_xlabel('Transfer − comparator RMSE (K)')
    b.legend(handles=[Line2D([0],[0],color=TEAL,marker='o',label='vs Scratch'),Line2D([0],[0],color=PURPLE,marker='D',label='vs ERA5 + offset')],frameon=False,loc='lower left',fontsize=6.5)
    bands=['21_59','60_70','71_79']
    for j,f in enumerate(FRACTIONS):
        ax=fig.add_subplot(g[1,2*j:2*j+2]);title(ax,chr(99+j),f'{int(100*f)}% training');clean(ax)
        for i,band in enumerate(bands):
            vals={k:d.score(f,k,'2022_2024',band) for k in ['pretrained','scratch','era5_offset']}
            ax.plot([vals['pretrained'],vals['scratch']],[i,i],color='#a6b2b6',lw=2,zorder=1)
            ax.plot(vals['pretrained'],i,'o',color=TEAL,ms=5,zorder=3)
            ax.plot(vals['scratch'],i,'o',mfc='white',mec=RUST,ms=5,zorder=3)
            ax.plot(vals['era5_offset'],i,'D',color=PURPLE,ms=3.5,zorder=2)
        ax.set_ylim(2.5,-.5);ax.set_xlim(2.55,8.55);ax.set_xticks([3,4,5,6,7,8]);ax.set_xlabel('RMSE (K)')
        ax.set_yticks(range(3),['21–59','60–70','71–79'] if j==0 else [])
        if j==0: ax.set_ylabel('Reference nodes z*')
    fig.legend(handles=legend_models(['pretrained','scratch','era5_offset','era5_last','source_climatology']),loc='upper center',bbox_to_anchor=(.52,.995),ncol=5,frameon=False,columnspacing=1.25,handlelength=1.8,fontsize=6.8)
    fig.text(.09,.045,'Pooled 2022–2024  ·  Mean of three seed RMSEs after pooling SSE and valid counts within each seed.',fontsize=6.4,color=GREY)
    fig.text(.09,.018,'(b) Ordinary 95% intervals, 14-day blocks; exploratory and conditional on the fitted models.  Negative favors Transfer.',fontsize=6.1,color=GREY)
    save(fig,out,'Figure_2',dpi)

def figure3(d,out,dpi,monthly):
    fig=plt.figure(figsize=(7.2,6.2))
    g=fig.add_gridspec(3,2,left=.09,right=.975,top=.93,bottom=.18,width_ratios=[5.1,1.15],hspace=.35,wspace=.22)
    p=d.p[1:];mesh=None
    for i,f in enumerate(FRACTIONS):
        ax=fig.add_subplot(g[i,0]);ap=fig.add_subplot(g[i,1])
        v=monthly[monthly.training_fraction==f].pivot(index='zstar',columns='month',values='transfer_minus_scratch_K').sort_index().to_numpy()
        mesh=heat(ax,v,p,limit=1)
        title(ax,chr(97+i),f'{int(f*100)}% training',pad=7)
        ax.set_xlim(0,48);ax.set_xticks([.5,12.5,24.5,36.5],['2022','2023','2024','2025'])
        ax.set_xticks([6.5,18.5,30.5,42.5],minor=True)
        ax.tick_params(which='minor',length=2)
        for edge in [12,24]: ax.axvline(edge,color='white',lw=.8)
        ax.axvline(36,color=INK,lw=.8,ls=(0,(4,2)))
        if i<2: ax.set_xticklabels([])
        else:ax.set_xlabel('Calendar month')
        pressure(ap,labels=False);zero(ap);clean(ap);ap.set_xlim(-.95,.65);ap.set_xticks([-.8,0,.6]);ap.set_xlabel('ΔRMSE (K)' if i==2 else '')
        for scope,col,ls in [('2022_2024',TEAL,'-'),('2025',RUST,'--')]:
            vals=d.score(f,'pretrained',scope,profile=True)-d.score(f,'scratch',scope,profile=True)
            ap.plot(vals,p,color=col,ls=ls,lw=1.15)
        if i==0: ap.set_title('Pooled profiles',fontsize=8,pad=7)
    cbax=fig.add_axes([.23,.082,.48,.017]);cb=fig.colorbar(mesh,cax=cbax,orientation='horizontal',extend='both',ticks=[-1,-.5,0,.5,1]);cb.set_label('Transfer − Scratch RMSE (K)  |  Negative favors Transfer',fontsize=7,labelpad=3)
    fig.legend(handles=[Line2D([0],[0],color=TEAL,label='2022–2024'),Line2D([0],[0],color=RUST,ls='--',label='2025')],loc='upper right',bbox_to_anchor=(.985,.994),ncol=2,frameon=False,fontsize=6.5)
    fig.text(.09,.02,'Monthly cells use true log-pressure spacing; no interpolation or significance symbols.  Color limits ±1 K; end triangles indicate values beyond the limits.',fontsize=5.9,color=GREY)
    save(fig,out,'Figure_3',dpi)

def figure4(d,out,dpi):
    fig=plt.figure(figsize=(7.2,6.05))
    g=fig.add_gridspec(2,12,left=.09,right=.98,top=.925,bottom=.14,height_ratios=[.95,1.65],hspace=.65,wspace=.7)
    ints=d.csv('data/evidence/original_paired_block_bootstrap.csv')
    ints=ints[(ints.band=='21_79')&(ints.metric=='rmse_k')&(ints.block_days==14)]
    for j,f in enumerate(FRACTIONS):
        a=fig.add_subplot(g[0,4*j:4*j+4]);clean(a);zero(a)
        if j==0:title(a,'a','Annual comparisons',pad=23)
        a.text(.5,1.05,f'{int(f*100)}% training',transform=a.transAxes,ha='center',fontsize=8)
        for i,y in enumerate([2022,2023,2024]):
            r=ints[(ints.scope==str(y))&(ints.training_fraction==f)].iloc[0]
            yes=r.familywise95_high<0
            interval(a,r.delta_transfer_minus_scratch,r.familywise95_low,r.familywise95_high,i,TEAL if yes else GREY,filled=yes)
        a.set_xlim(-.65,.17);a.set_xticks([-.6,-.3,0]);a.set_yticks(range(3),['2022','2023','2024'] if j==0 else []);a.set_ylim(2.6,-.6)
        a.set_xlabel('ΔRMSE (K)')
    fig.text(.98,.98,'4/9 annual comparisons supported',ha='right',color=TEAL,fontsize=7.2,fontweight='bold')
    b=fig.add_subplot(g[1,:6]);title(b,'b','Hunga windows · 1% training');zero(b);clean(b)
    h=d.csv('data/frozen/Hunga_verified_14day_intervals.csv')
    for i,r in h.iterrows():
        col=TEAL if r.family9_high_k<0 else RUST
        interval(b,r.delta_scratch_k,r.family9_low_k,r.family9_high_k,i,col,filled=r.family9_high_k<0,marker='D' if r.window=='W3' and r.year==2022 else 'o')
    b.set_yticks(range(9),[f'{r.year}  {r.window}' for _,r in h.iterrows()]);b.set_ylim(8.65,-.7);b.set_xlim(-1,.11);b.set_xticks([-1,-.75,-.5,-.25,0]);b.set_xlabel('Transfer − Scratch RMSE (K)')
    b.get_yticklabels()[2].set_color(RUST)
    for yy in [2.5,5.5]:b.axhline(yy,color=GRID,lw=.8)
    b.text(.02,.02,'8/9 supported',transform=b.transAxes,fontsize=7,color=TEAL,fontweight='bold')
    side=g[1,7:].subgridspec(1,2,wspace=.34)
    c=fig.add_subplot(side[0]);e=fig.add_subplot(side[1],sharey=c)
    w=d.csv('data/frozen/W3_observed_temperature_and_model_RMSE_by_pressure.csv').iloc[::-1]
    for i,(_,r) in enumerate(w.iterrows()):
        for ax,col,v in [(c,'MLS_2022_W3_minus_2020_2021_K',r.MLS_2022_W3_minus_2020_2021_K),
                         (e,'v16_Transfer_minus_Scratch_pooled_band_RMSE_K',r.v16_Transfer_minus_Scratch_pooled_band_RMSE_K)]:
            co=TEAL if v<0 else RUST
            ax.plot([0,v],[i,i],color=co,lw=2)
            ax.plot(v,i,'o',color=co,ms=4)
            ax.annotate(f'{v:+.2f}' if ax is c else f'{v:+.3f}',(v,i),xytext=(0,9),textcoords='offset points',ha='center',fontsize=6.5,color=co)
    title(c,'c','MLS contrast',pad=25,fontsize=7.6);title(e,'d','Skill contrast',pad=25,fontsize=7.6)
    for ax in [c,e]:
        zero(ax);clean(ax);ax.set_ylim(3.6,-.6)
        ax.text(.5,1.04,'2022 July–August',transform=ax.transAxes,ha='center',fontsize=6.4,color=GREY)
    c.set_yticks(range(4),['0.01–0.1','0.1–1','1–10','10–50']);c.set_ylabel('Pressure band (hPa)');c.set_xlim(-3.8,2.4);c.set_xticks([-3,0,2]);c.set_xlabel('ΔTemperature (K)')
    e.tick_params(labelleft=False);e.set_xlim(-.64,.69);e.set_xticks([-.5,0,.5]);e.set_xlabel('ΔRMSE (K)')
    fig.text(.09,.073,'(a,b) 95% familywise intervals, 14-day blocks; separate families of nine comparisons.  Filled: upper interval bound < 0.',fontsize=6.15,color=GREY)
    fig.text(.09,.046,'W1: 15 Jan–31 Mar  ·  W2: Apr–May  ·  W3: Jul–Aug.  (c) Relative to 2020–2021 matching-window means.',fontsize=6.15,color=GREY)
    fig.text(.09,.019,'(c,d) Descriptive point estimates on aligned pressure bands.  Observed temperature contrasts and model error contrasts have different meanings.',fontsize=6.0,color=GREY)
    save(fig,out,'Figure_4',dpi)
