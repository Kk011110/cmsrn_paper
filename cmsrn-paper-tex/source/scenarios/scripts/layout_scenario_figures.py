"""Re-layout the author-supplied scenario statistics at 160 mm manuscript width.
All baseline rows and signed effects retained. No predictions are rerun.
"""
from pathlib import Path
import csv, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'source/scenarios/data'
OUT=ROOT/'figures/scenarios'; OUT.mkdir(parents=True,exist_ok=True)
QA=ROOT/'source/scenarios/layout-qa'; QA.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'source/hyperparameters/figure-code'))
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,
'axes.spines.top':False,'axes.spines.right':False,'xtick.major.width':.6,'ytick.major.width':.6,
'legend.frameon':False,'savefig.facecolor':'white'})
def rows(name):
    with (DATA/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
perf=rows('performance_h6.csv'); serr=rows('serr_h6.csv'); bcms=rows('bcms_horizons.csv')
models=['HA','SVR','VAR','DCRNN','STGCN','GWNet','STNorm','DSTAGNN','PDFormer','STWave','PatchTST','TimeMixer','STDN','MetaDG','HyperD','RAST']
panels=[('chengdu','inflow'),('chengdu','outflow'),('shanghai','inflow'),('shanghai','outflow')]
def title(ax,k,city,flow):
    # Resolve the caption after the tick labels have been rendered.
    ax._panel_caption=f'({chr(97+k)}) {city.capitalize()} · {flow}'
def save(fig,name,axes,extra=()):
    axes=list(axes)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    captions=[]
    for ax in axes:
        bounds=ax.get_tightbbox(renderer)
        position=ax.get_position()
        y=(bounds.y0-4*fig.dpi/72)/fig.bbox.height
        captions.append(fig.text((position.x0+position.x1)/2,y,ax._panel_caption,
                                 ha='center',va='top',fontsize=9,fontweight='bold'))
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=QA/(name+'.alignment.json'),
                                      axes=axes,exclude_axes=extra,strict=True)
    placement=[]
    for ax,caption in zip(axes,captions):
        plot=ax.get_window_extent(renderer)
        content=ax.get_tightbbox(renderer)
        label=caption.get_window_extent(renderer)
        center_error=abs((label.x0+label.x1-plot.x0-plot.x1)/2)*72/fig.dpi
        gap=(content.y0-label.y1)*72/fig.dpi
        assert center_error<.1 and gap>=3.9, 'Panel caption must be centered below its own content.'
        assert label.y0>=0 and label.x0>=0 and label.x1<=fig.bbox.width, 'Panel caption is clipped.'
        placement.append({'label':caption.get_text(),'center_error_pt':center_error,
                          'gap_below_content_pt':gap,
                          'bbox_pt':[v*72/fig.dpi for v in label.extents]})
    (QA/(name+'.captions.json')).write_text(json.dumps({'width_mm':float(fig.get_figwidth()*25.4),
        'height_mm':float(fig.get_figheight()*25.4),'panel_titles':placement},indent=2),encoding='utf-8')
    # Preserve exact physical canvas; no bbox_inches='tight' rescaling.
    fig.savefig(OUT/(name+'.pdf'))
    fig.savefig(OUT/(name+'.svg'))
    fig.savefig(OUT/(name+'.png'),dpi=300)
    plt.close(fig)

width_mm=160

def performance(absolute=False):
    fig,axes=plt.subplots(2,2,figsize=(width_mm/25.4,146/25.4))
    fig.subplots_adjust(left=.13,right=.99,top=141/146,bottom=19/146,wspace=.40,hspace=12/55)
    mods=models+['Full'] if absolute else models
    vmax=max(float(r['mae']) for r in perf) if absolute else max(abs(float(r['improvement_pct'])) for r in perf)
    norm=Normalize(0,vmax) if absolute else Normalize(-vmax,vmax)
    cmap=plt.get_cmap('YlGnBu' if absolute else 'RdBu')
    for k,((city,flow),ax) in enumerate(zip(panels,axes.ravel())):
        lookup={(r['model'],int(r['group_id'])):r for r in perf if r['city']==city and r['flow']==flow}
        a=np.array([[float(lookup[m,g]['mae' if absolute else 'improvement_pct']) for g in range(4)] for m in mods])
        im=ax.imshow(a,cmap=cmap,norm=norm,aspect='auto')
        ax.set_xticks(range(4),['Low','Moderate','High','Extreme'],fontsize=7.3)
        ax.tick_params(axis='x',top=True,labeltop=True,bottom=False,labelbottom=False,length=0,pad=3)
        ax.set_yticks(range(len(mods)),['CMSRN' if m=='Full' else m for m in mods],fontsize=7.1)
        ax.tick_params(axis='y',length=0,pad=3)
        ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,len(mods),1),minor=True)
        ax.grid(which='minor',color='white',linewidth=.4);ax.tick_params(which='minor',length=0)
        for (i,j),v in np.ndenumerate(a):
            rgb=cmap(norm(v))[:3]; lum=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
            ax.text(j,i,f'{v:.2f}' if absolute else f'{v:+.2f}',ha='center',va='center',fontsize=7.3,color='white' if lum<.48 else '#222222')
        for sp in ax.spines.values():sp.set_visible(False)
        if not absolute:
            ax.set_ylim(17.6,-.5)
            ax.add_patch(Rectangle((-.5,15.65),4,1.2,color='#eef2f5',ec='none',zorder=-1))
            for g in range(4):
                r=lookup['Full',g]
                ax.text(g,16.3,f"{float(r['mae']):.3f}",ha='center',va='center',fontweight='bold',fontsize=7.3)
                ax.text(g,17.25,f"{int(r['n']):,}",ha='center',va='center',fontsize=6.7)
            ax.text(-.57,16.3,'CMSRN MAE',ha='right',va='center',fontsize=6.7)
            ax.text(-.57,17.25,'Valid n',ha='right',va='center',fontsize=6.7)
        else:
            ax.axhline(15.5,color='#444444',lw=.65)
            ax.get_yticklabels()[-1].set_fontweight('bold')
        title(ax,k,city,flow)
    cax=fig.add_axes([.25,8/146,.62,2.5/146]);cb=fig.colorbar(im,cax=cax,orientation='horizontal')
    cb.ax.tick_params(labelsize=7,length=2,pad=2)
    cb.set_label('MAE (original flow units)' if absolute else 'CMSRN MAE improvement over baseline (%)',fontsize=8,labelpad=3)
    save(fig,'performance_h6_absolute_mae_paper' if absolute else 'performance_h6_paper',axes.ravel(),[cax])

def serrplot():
    fig,axes=plt.subplots(2,2,figsize=(width_mm/25.4,88/25.4))
    fig.subplots_adjust(left=.105,right=.99,top=78/88,bottom=14/88,wspace=.23,hspace=16/24)
    cols=['#386da8','#d88937']
    for k,((city,flow),ax) in enumerate(zip(panels,axes.ravel())):
        rr={(r['subgroup'],r['branch']):r for r in serr if r['city']==city and r['flow']==flow}
        for b,(branch,col) in enumerate(zip(['Self','Delay'],cols)):
            ys=[float(rr[g,branch]['delta_mae']) for g in ['Slow','Rapid']]
            xx=np.arange(2)+(b-.5)*.28
            ax.bar(xx,ys,.28,color=col,label=f'Remove {branch.lower()} residual')
            for x,y in zip(xx,ys):ax.text(x,y+.005,f'+{y:.4f}',ha='center',va='bottom',fontsize=7)
        ax.set_xticks([0,1],[f"{g} change\nn = {int(rr[g,'Self']['n']):,}" for g in ['Slow','Rapid']],fontsize=7.5)
        ax.set_ylim(0,.175);ax.set_xlim(-.48,1.48);ax.set_yticks([0,.05,.10,.15])
        ax.tick_params(axis='y',labelsize=7);ax.tick_params(axis='x',length=0,pad=5)
        ax.grid(False)
        if k%2==0:ax.set_ylabel('ΔMAE (removed − CMSRN)',fontsize=8)
        title(ax,k,city,flow)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,1),ncol=2,fontsize=8)
    save(fig,'serr_sensitivity_h6_paper',axes.ravel())

def bcmsplot(scope):
    fig,axes=plt.subplots(4,1,figsize=(width_mm/25.4,128/25.4))
    fig.subplots_adjust(left=.18,right=.99,top=123/128,bottom=18/128,hspace=10.5/18.375)
    vals=[float(r['delta_mae'])*1000 for r in bcms if r['scope']==scope]
    vmax=max(abs(v) for v in vals);norm=Normalize(-vmax,vmax);cmap=plt.get_cmap('RdBu')
    for k,((city,flow),ax) in enumerate(zip(panels,axes)):
        rr={(r['background_group'],int(r['horizon'])):r for r in bcms if r['city']==city and r['flow']==flow and r['scope']==scope}
        groups=['Low','Medium','High'];a=np.array([[float(rr[g,h]['delta_mae'])*1000 for h in range(1,13)] for g in groups])
        im=ax.imshow(a,cmap=cmap,norm=norm,aspect='auto')
        ax.set_xticks(range(12),[f'H{h}' for h in range(1,13)],fontsize=7.2)
        ax.tick_params(axis='x',top=True,labeltop=True,bottom=False,labelbottom=False,length=0,pad=3)
        labs=[]
        for g in groups:
            counts=[int(rr[g,h]['n']) for h in range(1,13)]
            labs.append(f'{g}\nn = {min(counts)/1000:.1f}k–{max(counts)/1000:.1f}k')
        ax.set_yticks(range(3),labs,fontsize=6.9);ax.tick_params(axis='y',length=0,pad=3)
        ax.set_xticks(np.arange(-.5,12,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
        ax.grid(which='minor',color='white',lw=.5);ax.tick_params(which='minor',length=0)
        for (i,j),v in np.ndenumerate(a):
            rgb=cmap(norm(v))[:3];lum=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
            ax.text(j,i,f'{v:+.3f}',ha='center',va='center',fontsize=6.8,color='white' if lum<.48 else '#222222')
        for sp in ax.spines.values():sp.set_visible(False)
        title(ax,k,city,flow)
    fig.text(.015,.53,'Candidate-background difference',rotation=90,rotation_mode='anchor',va='center',fontsize=8)
    cax=fig.add_axes([.27,8/128,.62,2.5/128]);cb=fig.colorbar(im,cax=cax,orientation='horizontal')
    cb.ax.tick_params(labelsize=7,length=2,pad=2)
    cb.set_label('ΔMAE (×10⁻³; BCMS removed − CMSRN)',fontsize=8,labelpad=3)
    name='bcms_sensitivity_horizons'+('_all_test' if scope=='all_test' else '')+'_paper'
    save(fig,name,axes,[cax])
if __name__=='__main__':
    performance();performance(True);serrplot();bcmsplot('high_deviation');bcmsplot('all_test')
    print('Exported 5 scenario figures in PDF/SVG/PNG; rendered alignment gates passed.')
