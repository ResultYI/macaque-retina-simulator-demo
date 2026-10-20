"""Plot saved isolated source-reference calculations only; run no model."""
from pathlib import Path
import argparse
import json
import numpy as np


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    args=parser.parse_args();d=args.case
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((d/'result.json').read_bytes())
    z=np.load(d/'reference_traces.npz',allow_pickle=False)
    plt.rcParams.update({'font.family':'Segoe UI','font.size':10,'axes.titlesize':11,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    orange='#c96c26';blue='#246b9c';green='#299b86';purple='#855cb4'
    fig.subplots_adjust(top=.85,bottom=.13,left=.065,right=.98,wspace=.28,hspace=.42)
    fig.suptitle('Independent ON-parasol CBEM reference',fontsize=19,x=.065,ha='left',y=.98)
    fig.text(.065,.928,'Source equations re-expressed in NumPy/SciPy | No animal fit | Production model unchanged',fontsize=11,color='#455568')
    ax=axes[0,0]
    v=z['V_grid_mV']
    ax.semilogy(v,z['curve_paper_Hz'],color=orange,label='Paper / 2022 mapping: 90, -53, 1.67')
    ax.semilogy(v,z['curve_matlab_init_Hz'],color=purple,ls=':',label='2020 MATLAB initialization: beta=2.222')
    ax.semilogy(v,z['curve_project_Hz'],color=blue,label='Project: 150, -70, 5')
    ax.scatter([-65,-65],[r['anchors'][0]['author_zero_history_hazard_Hz'],r['anchors'][0]['project_zero_history_hazard_Hz']],c=[orange,blue],s=28,zorder=4)
    ax.axvline(-65,color='#aab2bb',lw=.8)
    ax.set(title='A  Same voltage, zero spike history',xlabel='Membrane voltage (mV)',ylabel='Conditional hazard (Hz)',ylim=(1e-5,2000))
    ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.18,which='major')
    ax=axes[0,1];t=z['t_start_s']*1000;gs=z['gs_over_C_per_s']
    ax.step(t,gs[:,0],where='post',color=orange,label='Excitation gE/C')
    ax.step(t,gs[:,1],where='post',color=green,label='Inhibition gI/C',ls='--')
    ax.set(title='B  Explicit synthetic conductance clamp',xlabel='Time (ms)',ylabel='Normalized conductance (1/s)',ylim=(-3,48))
    ax.legend(fontsize=8);ax.grid(alpha=.18)
    for j,text in enumerate(['Leak','E','E + I','Recovery','Equal E/I']):ax.text(j*100+50,45,text,ha='center',va='center',fontsize=8)
    ax=axes[0,2];te=z['t_end_s']*1000
    ax.plot(te,z['paper_V_mV'][1:],color=orange,label='Paper / MATLAB voltage algorithm')
    ax.plot(te,z['python_literal_V_mV'],color=purple,ls=':',label='2022 source literal indexing')
    ax.plot(te,z['project_V_mV'][1:],color=blue,label='Project membrane, same g/C')
    ax.set(title='C  Original membrane conditions retained',xlabel='Interval end (ms)',ylabel='Voltage (mV)')
    ax.legend(fontsize=8,loc='upper right');ax.grid(alpha=.18)
    ax=axes[1,0];tc=z['common_t_start_s']*1000
    ax.plot(t,z['paper_zero_history_hazard_Hz'],color=orange,label='Paper conditions, h=0')
    ax.plot(tc,z['project_no_history_hazard_Hz'],color=blue,label='Project conditions, h=0')
    ax.set(title='D  Voltage-to-hazard dynamics',xlabel='Bin start (ms)',ylabel='Conditional hazard (Hz)')
    ax.legend(fontsize=8);ax.grid(alpha=.18)
    ax=axes[1,1]
    author=z['paper_zero_history_hazard_Hz'];project=z['project_no_history_hazard_Hz']
    ax.plot(t,-np.expm1(-author*.0001)/.0001,color=orange,label='Paper h=0 | 0.1 ms')
    ax.plot(tc,-np.expm1(-project*.001)/.001,color=blue,ls=':',label='Project h=0 | 1 ms')
    ax.plot(tc,z['project_expected_count_rate_Hz'],color=blue,label='Project original h | exact mean')
    ax.set(title='E  Binary occupancy rate differs from hazard',xlabel='Bin start (ms)',ylabel='Expected binary counts / second')
    ax.legend(fontsize=8);ax.grid(alpha=.18)
    ax=axes[1,2]
    delta=z['python_literal_V_mV']-z['paper_V_mV'][1:]
    ax.plot(te,delta,color=purple,label='2022 literal minus paper equation')
    ax.axhline(0,color='#8d99a6',lw=.8)
    ax.set(title='F  Source voltage-indexing difference',xlabel='Interval end (ms)',ylabel='Voltage difference (mV)')
    ax.legend(fontsize=8);ax.grid(alpha=.18)
    ax.text(.98,.07,'First bin: -1.188 mV\nChange of input: previous-bin decay',ha='right',va='bottom',transform=ax.transAxes,fontsize=9,color='#55456d')
    fig.text(.065,.064,'Paper membrane: EL=-60 mV, EE=0, EI=-80 mV, gL/C=200 /s. Project: EL=-65 mV, gL/C=33.33 /s; physical C=60 pF, gL=2 nS.',fontsize=9,color='#455568')
    fig.text(.065,.039,'No author-fitted history or stimulus filters were available: author h=0 is an explicit partial reference. Inputs are synthetic g/C clamps, not visual recordings.',fontsize=9,color='#455568')
    fig.text(.065,.014,'JAX notebook and MATLAB program were not executed. Literal-source and equation routes are preserved separately; these results do not establish physiological calibration.',fontsize=9,color='#455568')
    for suffix in ['png','svg']:
        fig.savefig(d/('on_parasol_author_reference.'+suffix),dpi=180,facecolor='white')
    plt.close(fig)
    print(d/'on_parasol_author_reference.png')
