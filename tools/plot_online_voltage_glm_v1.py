"""Plot saved engineering verification arrays only; no simulation or fitting."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def main(folder: Path) -> None:
    result=json.loads((folder/'RESULT.json').read_text(encoding='utf-8'))
    if not result['passed']:
        raise ValueError('Completed verification required')
    target=folder/'figures'
    target.mkdir(exist_ok=False)
    with np.load(folder/'PHYSICAL_g1.npz',allow_pickle=False) as source:
        physical={name:source[name].copy() for name in source.files}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    t=physical['time_source_s']
    fig,axes=plt.subplots(4,1,figsize=(11,9),sharex=True,constrained_layout=True)
    axes[0].plot(t,physical['V_V']*1e3,color='#274c77')
    axes[0].set_ylabel('V (mV)')
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    for key,label,color in [('gE_S','gE','#d17a22'),('gI_S','gI','#577590')]:
        axes[1].plot(t,physical[key]*1e9,label=label,color=color)
    axes[1].set_ylabel('Conductance (nS)');axes[1].legend(loc='upper right')
    for key,label in [('I_L_A','IL'),('I_E_A','IE'),('I_I_A','II')]:
        axes[2].plot(t,physical[key]*1e12,label=label,linewidth=1)
    axes[2].set_ylabel('Current into group (pA)');axes[2].legend(loc='upper right',ncol=3)
    axes[3].plot(t,physical['tau_eff_s']*1e3,color='#6a4c93')
    axes[3].set_ylabel('Effective tau (ms)');axes[3].set_xlabel('Time from acquisition start (s)')
    axes[3].yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    for axis in axes:axis.set_xlim(t[0],t[-1]);axis.grid(alpha=.18)
    fig.suptitle('Existing target 8166 representative group (2 members): shared training input g1\n'
                 'Cached passive membrane; derived currents; gA=0; no reset or spike feedback')
    save(fig,target/'membrane_and_currents')
    fig,axes=plt.subplots(4,3,figsize=(14,10),sharex=True,constrained_layout=True)
    for col,cell in enumerate((2,3,4)):
        with np.load(folder/f'Cell{cell}_READABLE_TRACE.npz',allow_pickle=False) as z:
            trace={key:z[key].copy() for key in z.files}
        t=trace['time_bin_start_s']
        axes[0,col].plot(t,trace['D_dimensionless'],label='D',linewidth=.8)
        axes[0,col].plot(t,trace['H_path0_dimensionless'],label='H, path 0',linewidth=.8)
        axes[0,col].set_title(f'Cell {cell}: independent frozen V_GLM30')
        axes[1,col].plot(t,trace['lambda_path0_Hz'],label='lambda, path 0',linewidth=.8)
        axes[1,col].plot(t,trace['probability_path0']/.001,label='p / bin dt, path 0',linewidth=.8)
        axes[2,col].plot(t,trace['original_free_probability']/.001,label='Original',linewidth=1.6)
        axes[2,col].plot(t,trace['online_free_probability']/.001,label='Online',linewidth=.8,linestyle='--')
        events=t[trace['sampled_bin_path0'].astype(bool)]
        axes[3,col].vlines(events,0,1,color='#274c77',linewidth=.6)
        axes[3,col].set_ylim(-.1,1.1);axes[3,col].set_yticks([0,1]);axes[3,col].set_xlabel('Time (s)')
        for row in range(4):axes[row,col].set_xlim(0,71000*.0001);axes[row,col].grid(alpha=.18)
    labels=['Statistical predictor\n(dimensionless)','Conditional intensity /\nexpected bin-count rate (Hz)',
            'Free expected bin-count\nrate (Hz), 4096 paths','Sampled occupied bin\n(path 0)']
    for row,label in enumerate(labels):axes[row,0].set_ylabel(label)
    for row in range(3):axes[row,2].legend(loc='upper right',fontsize=8)
    fig.suptitle('Online equivalent stateful readout, full g1 acquisition (no new fit)\n'
                 'Each path is a statistical trial, not a group member; bin events are not exact action-potential times')
    save(fig,target/'online_readout_states')


def save(fig,stem: Path) -> None:
    for extension in ('png','pdf','svg'):
        fig.savefig(stem.with_suffix('.'+extension),dpi=170)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder)
