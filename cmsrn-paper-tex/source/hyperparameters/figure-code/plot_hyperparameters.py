from pathlib import Path
import argparse,csv,json,hashlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from audit_panel_alignment import require_matplotlib_panel_alignment

C={'mae':'#00008B','rmse':'#C15D00','best':'#10B981'}
PARAMS=['s0.gate_hidden_dim','bgst.dim','s0.alpha_init','bgst.beta_init']
LABELS=['SERR gate hidden dim','BCMS representation dim',r'SERR residual init $\alpha_0$',r'BCMS residual init $\beta_0$']
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'mathtext.fontset':'stix','font.size':8.5,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8,'axes.linewidth':.75,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'legend.fontsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})

def legend(fig):
 h=[Patch(facecolor=C['mae'],label='MAE'),Patch(facecolor=C['rmse'],label='RMSE'),Line2D([],[],color=C['best'],marker='*',markersize=7,linestyle='none',label='Best per metric')]
 fig.legend(handles=h,loc='upper center',bbox_to_anchor=(.535,1),ncol=3,handlelength=1.5,columnspacing=1.6,handletextpad=.55,borderaxespad=.15)

def panel(ax,rows,param,channel,letter):
 data=sorted([r for r in rows if r['parameter']==param and r['channel']==channel],key=lambda r:float(r['value']))
 assert len(data)==5,(param,channel)
 x=np.arange(5);w=.31
 ax.set_axisbelow(True);ax.grid(False)
 for metric,offset in [('mae',-w/2),('rmse',w/2)]:
  vals=np.array([float(r[metric]) for r in data]);ax.bar(x+offset,vals,width=w,color=C[metric],linewidth=0)
  best=int(np.argmin(vals));assert data[best]['best_'+metric]=='True'
  ax.plot(x[best]+offset,vals[best]+.052,marker='*',markersize=6.6,markeredgewidth=0,color=C['best'],linestyle='none',clip_on=False)
  for xx,val in zip(x+offset,vals):
   ax.text(xx,val+.17,f'{val:.4f}',rotation=90,rotation_mode='anchor',ha='left',va='center',color=C[metric],fontsize=7.3)
 ax.set_xlim(-.58,4.58);ax.set_ylim(0,1.95);ax.set_yticks([0,.5,1,1.5]);ax.set_xticks(x,[format(float(r['value']),'g') for r in data])
 for tick,row in zip(ax.get_xticklabels(),data):tick.set_fontweight('bold' if row['is_reference']=='True' else 'normal')
 ax.tick_params(axis='both',length=2.8,width=.7,pad=2.2)
 label=f'({letter}) '+LABELS[PARAMS.index(param)]
 if channel!='joint':label+='\nChengdu '+channel
 ax.set_xlabel(label,labelpad=3,linespacing=1.15)
 ax.set_label(letter)
 return data

def export(fig,out,audit,mode,plotted):
 stem=f'chengdu-{mode}-hyperparameter-paper'
 fig.canvas.draw()
 report=require_matplotlib_panel_alignment(fig,json_out=audit/(stem+'.alignment.json'),overlay_svg=audit/(stem+'.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
 fig.savefig(out/f'hyperparameter-{mode}.pdf')
 fig.savefig(out/f'hyperparameter-{mode}.svg')
 fig.savefig(out/f'hyperparameter-{mode}.png',dpi=600)
 manifest={'width_mm':fig.get_figwidth()*25.4,'height_mm':fig.get_figheight()*25.4,'rows':len(plotted),'values':2*len(plotted),'display_decimals':4,'geometry':'Full supplied precision','alignment':report['verdict'],'uncertainty':'One run per configuration; no error bars','data':[{k:r[k] for k in ['parameter','value','channel','mae','rmse','run_id']} for r in plotted]}
 (audit/(stem+'.plot-data.json')).write_text(json.dumps(manifest,indent=2),encoding='utf8');plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--audit-out',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 audit=a.audit_out or a.out;audit.mkdir(parents=True,exist_ok=True)
 with a.csv.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 assert len(rows)==60
 fig,axs=plt.subplots(2,2,figsize=(160/25.4,106/25.4));fig.subplots_adjust(left=.095,right=.99,bottom=12/106,top=97/106,wspace=10/66.6,hspace=13/36);legend(fig);fig.text(.019,.50,'Joint validation error',rotation=90,rotation_mode='anchor',ha='center',va='center',fontsize=9.5)
 plotted=[]
 for ax,param,letter in zip(axs.flat,PARAMS,'abcd'):plotted.extend(panel(ax,rows,param,'joint',letter))
 export(fig,a.out,audit,'joint',plotted)
 fig,axs=plt.subplots(4,2,figsize=(160/25.4,184/25.4));fig.subplots_adjust(left=.095,right=.99,bottom=17/184,top=175/184,wspace=10/66.6,hspace=14/29);legend(fig)
 fig.text(.019,.48,'Validation error',rotation=90,rotation_mode='anchor',ha='center',va='center',fontsize=9.5)
 plotted=[]
 for i,param in enumerate(PARAMS):
  for j,channel in enumerate(['inflow','outflow']):plotted.extend(panel(axs[i,j],rows,param,channel,'abcdefgh'[2*i+j]))
 export(fig,a.out,audit,'flow',plotted)
 (audit/'source-data.sha256').write_text(hashlib.sha256(a.csv.read_bytes()).hexdigest()+'\n',encoding='ascii')
 print('Rendered 60 data rows and 120 values.')
if __name__=='__main__':main()
