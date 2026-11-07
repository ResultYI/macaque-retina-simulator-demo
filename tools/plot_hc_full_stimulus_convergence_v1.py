"""Saved-output figures for full stimulus and bounded spatial comparisons."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();out=args.run
    result=json.loads((out/'RESULT.json').read_bytes());root=Path(__file__).resolve().parents[1]
    ref=result['reference_domain'];names=[v['name'] for v in result['plans'] if (out/v['name']/'B_biexponential.npz').exists()]
    data={name:{c:np.load(out/name/(c+'.npz')) for c in ['B_biexponential','C_feedback_off']} for name in names}
    colors={'small':'#365b99','medium':'#d26a2e','large':'#40806b'};t=data['small']['B_biexponential']['t_seconds']*1000
    fig,axes=plt.subplots(4,2,figsize=(11,11),sharex=True,constrained_layout=True)
    for panel,ax in enumerate(axes.ravel()):
        for name in names:
            z=data[name]['B_biexponential'];trace=z['H1_target'] if panel==0 else z['release_core'][panel-1]
            ax.plot(t,trace-trace[0],label=name,color=colors[name],lw=1.5)
        ax.set_title('H1 parent2891' if panel==0 else 'Core Cone '+str(panel),fontsize=10)
        ax.set_ylabel('Delta H1, native units' if panel==0 else 'Delta release, native units')
        ax.grid(alpha=.2);ax.axvspan(20,120,color='gray',alpha=.08)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useOffset=False)
    axes[0,0].legend();axes[-1,0].set_xlabel('Time (ms)');axes[-1,1].set_xlabel('Time (ms)')
    fig.suptitle('Complete annular input: nested-domain responses\nPlots subtract each baseline; acceptance compares absolute traces including baselines')
    for ext in ['png','pdf']:fig.savefig(out/('SPATIAL_RESPONSES.'+ext),dpi=160)
    plt.close(fig)
    v6=np.load(root/'reports/CONE_HC_MINIMAL_REAL_DIAGNOSTIC_20261102_v6/run_02/B_biexponential_TRAJECTORY.npz')
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,constrained_layout=True)
    for key,ax,index in [('H1_target',axes[0,0],None),('release_core',axes[0,1],4),('pooled_release_target',axes[1,0],None),('feedback_core',axes[1,1],4)]:
        for label,z,color in [('v6 partial stimulus',v6,'#777777'),('Complete stimulus, same small domain',data['small']['B_biexponential'],'#365b99')]:
            trace=z[key] if index is None else z[key][index]
            ax.plot(t,trace-trace[0],label=label,color=color)
        ax.set_title(key+(' (Cone parent156368)' if index is not None else ''));ax.set_ylabel('Change, native units');ax.grid(alpha=.2);ax.set_xlabel('Time (ms)')
    axes[0,0].legend(fontsize=8);fig.suptitle('Effect of applying the previously omitted annular input\nAll external stimulated Cone states evolve under fixed HC feedback')
    for ext in ['png','pdf']:fig.savefig(out/('FULL_STIMULUS_VS_V6.'+ext),dpi=160)
    plt.close(fig)
    z=data['small']['B_biexponential'];fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for name in names:
        z=data[name]['B_biexponential'];axes[0].plot(t,z['exterior_stimulated_input_target'],label=name,color=colors[name]);axes[1].plot(t,z['exterior_stimulated_input_delta_target'],label=name,color=colors[name])
    axes[0].set_title('Exterior stimulated contribution, including DC');axes[1].set_title('Exterior stimulated contribution, delta')
    for ax in axes:ax.set_xlabel('Time (ms)');ax.set_ylabel('A-weighted release, native units');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Exact original A weights; fixed-feedback peripheral Cone transduction')
    for ext in ['png','pdf']:fig.savefig(out/('EXTERIOR_INPUT.'+ext),dpi=160)
    plt.close(fig)
    final=data[ref];conditions=['B_biexponential','C_feedback_off']
    if (out/ref/'A_uniform.npz').exists():final['A_uniform']=np.load(out/ref/'A_uniform.npz');conditions.insert(0,'A_uniform')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for c in conditions:
        z=final[c];axes[0].plot(t,z['H1_target']-z['H1_target'][0],label=c)
        axes[1].plot(t,z['release_core'][4]-z['release_core'][4,0],label=c)
    axes[0].set_title('H1 parent2891');axes[1].set_title('Cone parent156368')
    for ax in axes:ax.set_xlabel('Time (ms)');ax.set_ylabel('Response from own baseline, native units');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Mechanism comparison in '+ref+' domain; '+result['status'])
    for ext in ['png','pdf']:fig.savefig(out/('MECHANISM_COMPARISON.'+ext),dpi=160)
    plt.close(fig)
    print('Four PNG/PDF figure pairs saved')


if __name__=='__main__':main()
