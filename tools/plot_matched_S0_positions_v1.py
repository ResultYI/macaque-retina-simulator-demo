"""Standalone figures from saved CSV only; no model imports or forward run."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    out=parser.parse_args().directory
    with (out/'ALL_TYPE_TIME_COURSES.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    types=list(dict.fromkeys(row['type'] for row in rows))
    columns=[('deltaV_local_baseline_mV','Delta V from local baseline (mV)'),
        ('delta_rate_local_baseline_Hz','Delta source rate from local baseline (Hz)'),
        ('gE_nS_per_member','gE / represented member (nS)'),('gI_nS_per_member','gI / represented member (nS)')]
    names={'midget_RGC_ON':'Midget ON','midget_RGC_OFF':'Midget OFF','parasol_RGC_ON':'Parasol ON',
        'parasol_RGC_OFF':'Parasol OFF','small_bistratified_S_ON':'S-bistratified (mixed)'}
    fig,axes=plt.subplots(len(types),len(columns),figsize=(17,12),sharex=True,constrained_layout=True)
    for i,pop in enumerate(types):
        for position,color,style,label in [('near_S0','#1764a0','-','Near patch 0.06431 deg'),
                ('far_S0','#bd6524','--','Far target patch 4.726306 deg')]:
            data=[row for row in rows if row['type']==pop and row['position']==position]
            t=np.array([float(row['time_seconds'])*1000 for row in data])
            for j,(key,title) in enumerate(columns):
                ax=axes[i,j];ax.plot(t,[float(row[key]) for row in data],color=color,ls=style,lw=1.3,label=label)
                ax.grid(alpha=.18);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
                if i==0:ax.set_title(title,fontsize=10)
                if j==0:ax.set_ylabel(names[pop],fontsize=9)
                if i==len(types)-1:ax.set_xlabel('Time (ms)')
        for j in range(len(columns)):
            ax=axes[i,j]
            for left,right,shade in [(50,200,'#dceaf4'),(350,500,'#fae5d4'),(600,750,'#e3eedb')]:
                ax.axvspan(left,right,color=shade,alpha=.25,zorder=-1)
        if i==0:axes[i,0].legend(fontsize=7)
    fig.suptitle('Same frozen S0 rules, independently initialized local models\n'
        'Internal Cone release x0.9 / x1.1; Rod port x2. RF ranges and members tabulated separately.\n'
        'No calibrated external contrast or physiological eccentricity function inferred.',fontsize=12)
    for suffix in ['png','svg','pdf']:
        path=out/('matched_S0_functional_curves.'+suffix)
        if path.exists():raise FileExistsError('Preserve prior figures: '+str(path))
        fig.savefig(path,dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
