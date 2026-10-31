"""Static scientific plots from saved candidate data; no simulation or rescaling."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(r'D:\macaque retinal simulator')
OUT=ROOT/'reports/retinotopic_rf_revision/SPATIAL_POOLING_CANDIDATE_V1'


def readcsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def main():
    data=readcsv(OUT/'analysis_v3/TIME_COURSES.csv');directory=OUT/'figures';directory.mkdir()
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fields=[('diffuse_BC','V_mV','BC voltage (mV)'),('diffuse_BC','terminal_release','BC terminal release'),
        ('parasol_RGC','gE_nS_per_member','Parasol gE (nS / represented member)'),
        ('parasol_RGC','V_mV','Parasol voltage (mV)'),('parasol_RGC','expected_rate_Hz','Expected1ms count rate (Hz / group object)')]
    for difference in [False,True]:
        fig,axes=plt.subplots(5,2,figsize=(11,12),sharex=True,layout='constrained')
        for column,location in enumerate(['near','far']):
            for row,(prefix,field,label) in enumerate(fields):
                ax=axes[row,column]
                for polarity,color in [('ON','#2369a6'),('OFF','#d57424')]:
                    series={}
                    for arm in ['legacy','candidate']:
                        records=[r for r in data if r['location']==location and r['arm']==arm and r['type']==prefix+'_'+polarity and r['field']==field]
                        t=np.array([float(r['time_seconds']) for r in records]);series[arm]=np.array([float(r['value']) for r in records])
                    if difference:ax.plot(t*1000,series['candidate']-series['legacy'],color=color,label=polarity)
                    else:
                        for arm,style in [('legacy','--'),('candidate','-')]:ax.plot(t*1000,series[arm],style,color=color,label=polarity+' '+arm,linewidth=1.1)
                for a,b,color in [(50,200,'#cce7f5'),(350,500,'#ffebcf'),(600,750,'#e4e4e4')]:ax.axvspan(a,b,color=color,alpha=.35,zorder=-1)
                ax.set_ylabel(('New - old: ' if difference else '')+label);ax.grid(alpha=.15)
                if row==0:ax.set_title(location+(' (0.06431deg patch)' if location=='near' else ' (4.72631deg patch)'));ax.legend(ncol=2,fontsize=8)
                if row==4:ax.set_xlabel('Time (ms)')
        fig.suptitle('Engineering pooling candidate: '+('direct new-old differences' if difference else 'core representative responses')+'\nFrozen held S0; release x0.9 / x1.1, Rod x2. No physiological RF calibration.',fontsize=11)
        stem='new_old_differences' if difference else 'paired_responses'
        fig.savefig(directory/(stem+'.png'),dpi=180);fig.savefig(directory/(stem+'.svg'));plt.close(fig)
    rows=readcsv(OUT/'ROW_COMPARISON.csv');routes=list(dict.fromkeys(r['route'] for r in rows))
    fig,axes=plt.subplots(4,2,figsize=(11,10),layout='constrained')
    for i,route in enumerate(routes):
        for j,location in enumerate(['near','far']):
            selected=[r for r in rows if r['route']==route and r['location']==location and r['core']=='True']
            degree=np.array([int(r['candidate_inputs']) for r in selected]);ax=axes[i,j]
            low,high=degree.min(),degree.max();ax.hist(degree,bins=np.arange(low-.5,high+1.5),color='#497e90',alpha=.8)
            ax.axvline(8,color='#b1493a',linestyle='--',label='Legacy degree8')
            ax.set_title(location+': '+route,fontsize=9);ax.set_xlabel('Selected original presynaptic IDs / target row');ax.set_ylabel('Core fine target rows');ax.legend(fontsize=8)
    fig.suptitle('Actual structural convergence: all eligible sources within the engineering radius\nNot ribbon counts, dendritic morphologies, or measured functional RFs.',fontsize=11)
    fig.savefig(directory/'input_distributions.png',dpi=180);fig.savefig(directory/'input_distributions.svg');plt.close(fig)
    print('SAVED3_PNG3_SVG')


if __name__=='__main__':main()
