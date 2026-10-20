"""Plot saved actual full-network source-transfer comparison; no model run."""
from pathlib import Path
import argparse
import json
import numpy as np


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True)
    args=p.parse_args();d=args.case;r=json.loads((d/'result.json').read_bytes())
    if not r['status'].startswith('COMPLETE_'):raise ValueError('Completed actual network comparison required')
    matched=json.loads((d/'matched_control_result.json').read_bytes())
    if matched['status']!='COMPLETE_PAIRED_GRAY_CONTROLS':raise ValueError('Completed matched gray controls required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Segoe UI','font.size':10,'axes.titlesize':11,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.formatter.useoffset':False})
    arrays={};expected={};deltas={};gray={}
    for name in ['old','candidate']:
        with np.load(d/name/'stimulus/traces.npz',allow_pickle=False) as z:arrays[name]={k:z[k] for k in z.files}
        with np.load(d/name/'exact_expected_count_rates.npz',allow_pickle=False) as z:expected[name]={k:z[k] for k in z.files}
        with np.load(d/name/'matched_gray/matched_response_deltas.npz',allow_pickle=False) as z:deltas[name]={k:z[k] for k in z.files}
        with np.load(d/name/'matched_gray/traces.npz',allow_pickle=False) as z:gray[name]={k:z[k] for k in z.files}
    cid='parasol_RGC_ON';t=(arrays['old']['time']-.1)*1000
    oldcolor='#246b9c';newcolor='#c46c2a'
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    fig.subplots_adjust(top=.85,bottom=.13,left=.065,right=.98,wspace=.3,hspace=.43)
    fig.suptitle('ON-parasol literature-transfer candidate | actual full-network test',x=.065,y=.98,ha='left',fontsize=18)
    fig.text(.065,.928,'42 components / 20,422 cells / 58,155 dynamic scalars | Separate cold starts + 100 ms wash-in | No default adoption',fontsize=11,color='#455568')
    ax=axes[0,0]
    ax.step(np.arange(6)*100,np.array([0,.1,0,-.1,0,0]),where='post',color='#299b86',lw=2)
    ax.set(title='A  Fullfield root-input steps',xlabel='Stimulus time (ms)',ylabel='Relative change in each consumer unit',ylim=(-.14,.14))
    ax.set_yticks([-.1,0,.1],['-10%','gray','+10%']);ax.grid(alpha=.2)
    ax.text(.5,.96,'Engineering input; no ISET video capture',transform=ax.transAxes,ha='center',va='top',fontsize=8)
    ax=axes[0,1]
    for name,color,ls in [('old',oldcolor,'-'),('candidate',newcolor,':')]:
        ax.plot(t,1000*arrays[name][cid+'.voltage'].mean(axis=0),color=color,ls=ls,label=name)
    ax.plot(t,1000*gray['old'][cid+'.voltage'].mean(axis=0),color='#7b8794',lw=.9,ls='--',label='matched gray')
    ax.set(title='B  ON-parasol membrane voltage',xlabel='Stimulus time (ms)',ylabel='Mean voltage (mV)')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[0,2]
    for name,color in [('old',oldcolor),('candidate',newcolor)]:
        ax.plot(t,arrays[name][cid+'.conditional_rate'].mean(axis=0),color=color,label=name,lw=1)
    ax.set(title='C  Recorded conditional hazard',xlabel='Stimulus time (ms)',ylabel='Conditional hazard (Hz)')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,0]
    for name,color in [('old',oldcolor),('candidate',newcolor)]:
        ax.plot(t[:-1],deltas[name][cid+'.expected_rate_delta_Hz'].mean(axis=0),color=color,label=name)
    ax.axhline(0,c='#8d99a6',lw=.8)
    ax.set(title='D  Response after matched-gray subtraction',xlabel='Stimulus time (ms)',ylabel='Change of expected count rate (Hz)')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax.text(.02,.04,'Original history and dt retained',transform=ax.transAxes,fontsize=8)
    ax=axes[1,1];x=np.arange(5);labels=['gray','bright','gray','dark','gray']
    rows={(q['variant'],q['component'],q['phase']):q for q in r['phase_comparison']}
    phases=['gray_before','increment','gray_middle','decrement','gray_after']
    for name,color,dx in [('old',oldcolor,-.18),('candidate',newcolor,.18)]:
        ys=[rows[name,cid,ph]['expected_binary_count_rate_Hz'] for ph in phases]
        obs=[rows[name,cid,ph]['one_seed_observed_rate_Hz'] for ph in phases]
        ax.bar(x+dx,ys,.34,color=color,alpha=.8,label=name+' exact mean')
        ax.scatter(x+dx,obs,marker='x',c='#263340',s=24,zorder=4,label='one-seed counts' if name=='old' else None)
    ax.set_xticks(x,labels);ax.set(title='E  Phase rates: expectation and actual counts',ylabel='Counts / second per cell')
    ax.legend(fontsize=8);ax.grid(alpha=.2,axis='y')
    ax=axes[1,2]
    fams=['midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON']
    short=['midget ON','midget OFF','parasol ON','parasol OFF','S ON']
    for name,color,dx in [('old',oldcolor,-.18),('candidate',newcolor,.18)]:
        bright=[matched['responses'][name][c]['phases']['increment']['mean_expected_count_rate_delta_Hz'] for c in fams]
        ax.bar(np.arange(5)+dx,bright,.34,color=color,label=name)
    ax.axhline(0,c='#8d99a6',lw=.8);ax.set_xticks(np.arange(5),short,rotation=20,ha='right')
    ax.set(title='F  Bright response after gray subtraction',ylabel='Change of phase mean count rate (Hz)')
    ax.legend(fontsize=8);ax.grid(alpha=.2,axis='y')
    for ax in axes.flat:
        if ax not in (axes[1,1],axes[1,2]):
            for edge in [100,200,300,400]:ax.axvline(edge,c='#b6bfc8',lw=.55,alpha=.5)
    fig.text(.065,.065,'Only ON-parasol alpha/mu/beta: 150/-70/5 to 90/-53/1.67 (Hz,mV,mV). Project C/gL/EL/reversals, 3-bin history and dt=1 ms are unchanged.',fontsize=9,color='#455568')
    fig.text(.065,.039,'Each variant: 0.1 s preparation + 0.5 s steps + 0.5 s paired gray control, with all coupled components. Independent td / R*/s inputs; no calibrated optical-scene claim.',fontsize=9,color='#455568')
    fig.text(.065,.014,'D/F subtract matched gray: each control restores only its own parameter-matched prepared state. Count expectation uses actually advanced V. Short preparation is not claimed steady; no empirical calibration.',fontsize=9,color='#455568')
    for suffix in ['png','svg']:fig.savefig(d/('on_parasol_candidate_network.'+suffix),dpi=180,facecolor='white')
    plt.close(fig);print(d/'on_parasol_candidate_network.png')
