"""Single saved-array figure for recovery, one-cell masks and rod propagation."""
from pathlib import Path
import argparse,json
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);a=p.parse_args();d=a.case
    r=json.loads((d/'result.json').read_bytes());z=np.load(d/'followup_response_arrays.npz',allow_pickle=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Segoe UI','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9));fig.subplots_adjust(top=.85,bottom=.12,left=.09,right=.97,wspace=.28,hspace=.42)
    blue='#286b9b';orange='#c36b2b';green='#2f8e76';cid='parasol_RGC_ON'
    fig.suptitle('Unchanged-network follow-up | recovery, single-cell surround and rod chain',x=.09,y=.975,ha='left',fontsize=16)
    fig.text(.09,.925,'No parameter/default, reference, wiring or spatial-support changes | Own exact checkpoints | No MATLAB or physiological fit',fontsize=10,color='#455568')
    ax=axes[0,0];v=z[cid+'.recovery_expected_delta_Hz'];t=np.arange(v.shape[1]);ax.plot(t,v.mean(axis=0),color=blue,label='ON-parasol candidate')
    ax.axhline(0,color='#8895a3',lw=.8);ax.axvline(500,color=orange,ls='--',lw=1,label='new gray continuation')
    ax.axvspan(900,1000,color=green,alpha=.1,label='final 100 ms')
    ax.set(title='A  Actual V trajectory, exact count expectation',xlabel='Time since original stimulus start (ms)',ylabel='Expected count rate minus own gray (Hz)')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[0,1];focus=r['single_cell']['cell_center']['geometry']['focus_cell_index']
    for key,label,color in [('cell_center','center: direct excitatory footprint',orange),('cell_annulus','ring: direct cones excluded',blue)]:
        a0=z[key+'.expected_delta_Hz'];ax.plot(np.arange(a0.shape[1]),a0[focus],label=label,color=color)
    ax.axhline(0,color='#8895a3',lw=.8);ax.set(title='B  Same actual RGC cell: center / annulus',xlabel='Stimulus time (ms)',ylabel='Expected count rate minus gray (Hz)')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    g=r['single_cell']['cell_center']['geometry'];ax.text(.02,.04,f"Center radius {g['center_radius_mm']*1000:.2f} um; ring to {g['ring_outer_radius_mm']*1000:.2f} um",transform=ax.transAxes,fontsize=8)
    ax=axes[1,0];w=r['small_bistratified_workpoint'];b=list(w['branches'].values());vals=[x['actual_conductance_nS']['mean'] for x in b]
    ax.bar([0,1],vals,color=[green,orange],width=.55);ax.set_xticks([0,1],['S-ON BC route','LM-OFF BC route'])
    ax.set(title='C  Small-bistratified actual gray excitation',ylabel='Conductance (nS)');ax.set_ylim(0,.8);ax.grid(alpha=.2,axis='y')
    ax.text(.02,.95,f"Distinct branch sum = {w['total_excitation_nS']['mean']:.4f} nS\nActual V = {w['actual_voltage_mV']['mean']:.4f} mV\nNo divide-by-two or gain adjustment",transform=ax.transAxes,va='top',fontsize=9)
    ax=axes[1,1];time=z['rod_time_seconds']*1000
    for key,label,color,style in [('rod_membrane.voltage','rod V (inverted)',blue,'-'),
        ('rod_BC.voltage','RBC V',orange,'-'),('AII_AC.voltage','AII V',green,'-'),
        ('midget_BC_ON.voltage','ON midget BC V','#8064a2','--'),('midget_BC_OFF.voltage','OFF midget BC V (inverted)','#a15454',':')]:
        delta=z['rod.'+key+'.delta'].mean(axis=0);peak=max(float(abs(delta).max()),1e-30)
        sign=-1 if key in ('rod_membrane.voltage','midget_BC_OFF.voltage') else 1
        ax.plot(time,sign*delta/peak,color=color,ls=style,label=label)
    ax.axvspan(50,150,color='#b9c1ca',alpha=.18);ax.axhline(0,color='#8895a3',lw=.8)
    ax.set(title='D  Rod-only flash: 300 ms polarity window',xlabel='Probe time (ms)',ylabel='Signed response / each observed absolute maximum')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.text(.09,.066,'A/B use actual 1 ms bin-start V and the retained 3-tap history. A keeps the original first-bin draw and propagates history across the saved boundary.',fontsize=9,color='#455568')
    fig.text(.09,.04,'D: rod 5 to 6 R*/s for 100 ms; cones held. Normalization shows relative propagation only; the response peak is not resolved in this window.',fontsize=9,color='#455568')
    fig.text(.09,.014,'Rod fixed-dark-anchor change is pending a reference-contract decision. Existing per-background anchor is retained. These are engineering functional checks.',fontsize=9,color='#455568')
    fig.savefig(d/'pro_followup.png',dpi=170,facecolor='white');plt.close(fig);print(d/'pro_followup.png')


if __name__=='__main__':main()
