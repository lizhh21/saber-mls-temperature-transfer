"""Print-sized typography, shared encodings, and true log-pressure cells."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator,FuncFormatter
from matplotlib.lines import Line2D

INK='#263c49'; TEAL='#167e89'; RUST='#c46a4c'; PURPLE='#827099'; GOLD='#b5964c'; GREY='#8a959a'
PALE='#edf3f4'; GRID='#e3e8ea'
CMAP=LinearSegmentedColormap.from_list('skill', ['#17647b','#72a9b9','#c5dce2','#faf9f5','#e6c5b4','#cc8c6e','#af513e'],N=256)
CMAP.set_bad('#f0f0ed')
COLORS={'pretrained':TEAL,'scratch':RUST,'era5_offset':PURPLE,'era5_last':GREY,'source_climatology':GOLD}
LABELS={'pretrained':'Transfer','scratch':'Scratch','era5_offset':'ERA5 + offset','era5_last':'ERA5 persistence','source_climatology':'Source climatology'}

def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'text.color':INK,'axes.labelcolor':INK,
        'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':'#7f8c92','axes.linewidth':.6,
        'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.3,'lines.markersize':3.4,
        'xtick.major.width':.55,'ytick.major.width':.55,'xtick.major.size':2.8,'ytick.major.size':2.8,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})

def title(ax, letter, text, **kw):
    kw.setdefault('pad',10)
    ax.set_title(f'({letter})  {text}',loc='left',fontweight='bold',**kw)

def clean(ax,axis='x'):
    ax.set_axisbelow(True);ax.grid(axis=axis,color=GRID,lw=.5)

def pressure(ax,p=None,labels=True):
    ax.set_yscale('log');ax.set_ylim(50,.01)
    ax.yaxis.set_major_locator(FixedLocator([.01,.1,1,10,50]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
    ax.minorticks_off()
    if labels: ax.set_ylabel('Pressure (hPa)')
    else: ax.tick_params(labelleft=False)

def pedges(p):
    p=np.asarray(p);m=np.sqrt(p[1:]*p[:-1]);return np.r_[p[0]**2/m[0],m,p[-1]**2/m[-1]]

def heat(ax,values,p,limit=1,xedges=None):
    xedges=np.arange(values.shape[1]+1) if xedges is None else xedges
    mesh=ax.pcolormesh(xedges,pedges(p),np.ma.masked_invalid(values),cmap=CMAP,vmin=-limit,vmax=limit,shading='flat',rasterized=False)
    pressure(ax);return mesh

def zero(ax): ax.axvline(0,color=GREY,lw=.8,ls=(0,(3,2)),zorder=1)

def interval(ax,x,lo,hi,y,color=TEAL,filled=True,marker='o',lw=1.2):
    ax.plot([lo,hi],[y,y],color=color,lw=lw,zorder=2)
    ax.plot([lo,lo],[y-.05,y+.05],color=color,lw=.6)
    ax.plot([hi,hi],[y-.05,y+.05],color=color,lw=.6)
    ax.plot(x,y,marker=marker,ms=4,mec=color,mfc=color if filled else 'white',mew=.8,zorder=3)

def legend_models(kinds):
    return [Line2D([0],[0],color=COLORS[k],marker='o' if k in ('pretrained','scratch') else 'D',ms=3.5,lw=1,label=LABELS[k]) for k in kinds]

def save(fig,out,name,dpi=600):
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'{name}.{ext}',dpi=dpi,metadata={'Creator':'SABER–MLS figure repository v2'} if ext=='pdf' else None)
    plt.close(fig)
