"""Scientific figure of all real source traces; no invented model comparison."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'reports/chen2024_real_data'
data=np.load(folder/'full_published_traces.npz',allow_pickle=False)
stats=json.loads((folder/'all_cell_metrics.json').read_text(encoding='utf-8'))
assert stats['model_run_count']==0
t=data['relative_index_seconds'];stim=data['cell_1_stimulus_Rstar_per_s']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','savefig.facecolor':'white'})
fig,axes=plt.subplots(7,2,figsize=(16,17),sharex='col',gridspec_kw={'width_ratios':[2.2,1]})
for col in range(2):
    ax=axes[0,col];ax.plot(t,stim,color='#2d4e70',lw=.8,label='Published stimulus (unaltered)')
    ax.axhline(50000,color='#bd3434',ls='--',lw=1,label='Locked module upper bound')
    ax.axhline(0,color='#dd8d20',ls=':',lw=1)
    ax.fill_between(t,0,stim,where=stim<0,color='#dd8d20',alpha=.6)
    ax.fill_between(t,50000,stim,where=stim>=50000,color='#bd3434',alpha=.25)
    ax.set_ylabel('Stimulus (R*/s)');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    for index in range(1,7):
        ax=axes[index,col];ax.plot(t,data[f'cell_{index}_response_normalized'],color='#202a35',lw=.9)
        ax.set_ylim(-1.08,-.25);ax.grid(alpha=.15)
        ax.set_ylabel('Normalized current\n(unitless)')
        ax.text(.015,.96,f'PrimateConeResponses{{{index}}}',transform=ax.transAxes,va='top',fontsize=10,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
    axes[6,col].set_xlabel('Relative sample-index time (s); dt=0.0001 s from README')
    axes[0,col].set_title('Complete published range, all 80,000 samples' if col==0 else 'Same first 0.5 s for every cell')
    axes[0,col].set_xlim((0,t[-1]) if col==0 else (0,.5))
axes[0,0].legend(loc='upper right',fontsize=9)
fig.suptitle('Chen2024 public source data: all six primate cone repeat means\nFixed-model comparison blocked by input semantics and missing state/time history',fontsize=17,y=.985)
fig.text(.06,.018,'Measured source means, NOT synthetic. No pA ground truth, model prediction, fit, time shift or noise-ceiling claim.\nNegative rate: 1,026/80,000; >=50,000 R*/s: 12,538/80,000 per cell. Cell labels are array ordinals, not biological IDs.',fontsize=11)
fig.subplots_adjust(left=.115,right=.98,top=.935,bottom=.075,hspace=.22,wspace=.25)
fig.savefig(folder/'all_cells_source_audit.png',dpi=160)
fig.savefig(folder/'all_cells_source_audit.pdf')
print(json.dumps({'figure':'reports/chen2024_real_data/all_cells_source_audit.png','all_cells':6,'complete_samples_per_cell':80000,'comparison':'BLOCKED; no model line invented'}))
