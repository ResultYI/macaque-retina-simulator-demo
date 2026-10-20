"""One consolidated static figure from saved B0-B2 arrays; no new simulation."""
from pathlib import Path
import argparse,json
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True,type=Path);a=p.parse_args();d=a.case
    r=json.loads((d/'result.json').read_bytes())
    if r['status']!='COMPLETE_B0_B1_AND_BOUNDED_B2':raise ValueError('Completed batch analysis required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Segoe UI','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    keys=['midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON']
    labels=['midget ON','midget OFF','parasol ON','parasol OFF','S / LM']
    rows=[r['decisions'][k] for k in keys];x=np.arange(5);blue='#286b9b';orange='#c36b2b';green='#2f8e76'
    fig,axes=plt.subplots(2,3,figsize=(16,9));fig.subplots_adjust(top=.86,bottom=.12,left=.065,right=.98,wspace=.31,hspace=.43)
    fig.suptitle('RGC source-transfer candidates | actual complete-network B0-B2 tests',x=.065,y=.975,ha='left',fontsize=17)
    fig.text(.065,.925,'42 components / 20,422 cells | Per-type candidates, no default adoption | Unchanged geometry, outer/rod models and feedback',color='#455568',fontsize=10)
    ax=axes[0,0]
    ax.bar(x-.17,[q['old_gray_expected_count_rate_Hz'] for q in rows],.32,color=blue,label='old')
    ax.bar(x+.17,[q['candidate_gray_expected_count_rate_Hz'] for q in rows],.32,color=orange,label='candidate')
    ax.set_xticks(x,labels,rotation=17,ha='right');ax.set(title='A  Matched-gray expected discrete count rates',ylabel='Expected count / second per cell')
    ax.legend(fontsize=8);ax.grid(alpha=.2,axis='y')
    ax=axes[0,1]
    ax.bar(x-.17,[q['bright_expected_delta_Hz'] for q in rows],.32,color=orange,label='bright +10%')
    ax.bar(x+.17,[q['dark_expected_delta_Hz'] for q in rows],.32,color=blue,label='dark -10%')
    ax.axhline(0,color='#8895a3',lw=.8);ax.set_xticks(x,labels,rotation=17,ha='right')
    ax.set(title='B  Candidate fullfield responses minus gray',ylabel='Change of expected count rate (Hz)');ax.legend(fontsize=8);ax.grid(alpha=.2,axis='y')
    z=np.load(d/'paired_probe_response_arrays.npz',allow_pickle=False);t=np.arange(500)
    ax=axes[0,2]
    for name,label,color,style in [('center05','center 5%',blue,':'),('center20','center 20%',blue,'-'),
        ('annulus20','annulus 20%',green,'-'),('full20','all cones 20%',orange,'-')]:
        ax.plot(t,z[name+'.expected_delta_Hz'].mean(axis=0),color=color,ls=style,label=label)
    ax.axhline(0,color='#8895a3',lw=.8);ax.set(title='C  ON-parasol spatial probes',xlabel='Probe time (ms)',ylabel='Expected count rate minus gray (Hz)')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    with np.load(d/'spatial/full20/stimulus/traces.npz',allow_pickle=False) as q:stim={k:q[k] for k in q.files}
    with np.load(d/'parasol_RGC_OFF/matched_gray/traces.npz',allow_pickle=False) as q:gray={k:q[k] for k in q.files}
    cid='parasol_RGC_ON';vb=gray[cid+'.voltage'];e=(0-vb)*(stim[cid+'.gE']-gray[cid+'.gE'])*1e12
    i=(-.08-vb)*(stim[cid+'.gI']-gray[cid+'.gI'])*1e12
    ax=axes[1,0];ax.plot(np.arange(501),e.mean(axis=0),color=orange,label='excitatory drive')
    ax.plot(np.arange(501),i.mean(axis=0),color=blue,label='inhibitory drive');ax.plot(np.arange(501),(e+i).mean(axis=0),color='#3d4753',lw=.8,label='sum')
    ax.axhline(0,color='#8895a3',lw=.8);ax.set(title='D  ON-parasol current increments: all cones 20%',xlabel='Probe time (ms)',ylabel='Current increment at gray V (pA)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,1]
    for name,color in [('S20',green),('LM20',orange)]:ax.plot(t,z[name+'.expected_delta_Hz'].mean(axis=0),color=color,label=name+' cone-input mask')
    ax.axhline(0,color='#8895a3',lw=.8);ax.set(title='E  Small-bistratified candidate: spectral masks',xlabel='Probe time (ms)',ylabel='Expected count rate minus gray (Hz)')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,2];ax.axis('off');full=r['weak_response_probes']['full20'];chain=full['chain'];mem=full['membrane']['phases']['increment']
    cells=[['cone release (q/aI)',f"{chain['outer_native.release']['mean_bright_delta']:+.6f}",'native drive'],
        ['ON diffuse BC release',f"{chain['diffuse_BC_ON.activation']['mean_bright_delta']:+.6f}",'effective q'],
        ['RGC excitation',f"{mem['delta_gE_pS']:+.4f}",'pS'],
        ['RGC voltage',f"{mem['mean_voltage_delta_mV']:+.6f}",'mV'],
        ['RGC expected count rate',f"{full['bright_expected_delta_Hz']:+.6f}",'Hz']]
    table=ax.table(cellText=cells,colLabels=['Measured stage','Bright minus gray','Unit'],loc='center',cellLoc='left',colWidths=[.48,.33,.19]);table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.9)
    ax.set_title('F  Recorded differential chain (all cones 20%)',pad=17)
    for panel in (axes[0,2],axes[1,0],axes[1,1]):
        for edge in [100,200,300,400]:panel.axvline(edge,color='#b9c1ca',lw=.6,alpha=.5)
    fig.text(.065,.065,'Count expectations exactly marginalize the retained three binary history taps on actual 1 ms bin-start V; the saved first count stays fixed. Actual one-seed counts are separate in CSV.',fontsize=9,color='#455568')
    fig.text(.065,.039,'A/B: independent consumer fullfield steps; C-F: declared cone RF-position masks with rod background held. No calibrated optical-scene or matched physiological amplitude claim.',fontsize=9,color='#455568')
    fig.text(.065,.014,'Each new type: fresh 100 ms wash-in + 500 ms stimulus + own 500 ms gray. Spatial branches use their own matching prepared checkpoint. Short preparation is not claimed steady.',fontsize=9,color='#455568')
    fig.savefig(d/'pro_rgc_b0_b2.png',dpi=170,facecolor='white');plt.close(fig)
    print(d/'pro_rgc_b0_b2.png')


if __name__=='__main__':main()
