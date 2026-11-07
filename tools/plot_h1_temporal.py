"""Plot saved Table 2 comparison only; never simulate or refit."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/CONE_HC_FINAL_IMPLEMENTATION_2026-11-02'


def main():
    m = json.loads((OUT/'CALIBRATION_METRICS.json').read_bytes())
    if m['status'] != 'COMPLETE':
        raise RuntimeError('Completed calibration required')
    z = np.load(OUT/'H1_TEMPORAL_TRAJECTORIES.npz', allow_pickle=False)
    import csv
    with (ROOT/'data/source_data/h1_temporal_targets.csv').open(encoding='utf-8') as stream:
        targets = list(csv.DictReader(stream))
    plt.rcParams.update({'font.size':10, 'axes.spines.right':False, 'axes.spines.top':False})
    fig, axes = plt.subplots(3,2,figsize=(13,12),layout='constrained')
    colors = ['#27699a','#d27a28','#27866b']
    for col, pulse in enumerate((10,100)):
        obs = [r for r in targets if int(r['pulse_ms']) == pulse]
        x = np.arange(3)
        axes[0,col].errorbar(x,[float(r['peak_latency_ms']) for r in obs],
            yerr=[float(r['peak_latency_sd_ms']) for r in obs],fmt='ko',capsize=4,
            label='Measured population mean +/- SD')
        for label,style,color in [('baseline','--','#777777'),('candidate','-','#27699a')]:
            rows = [r for r in m[label]['conditions'] if r['pulse_ms']==pulse]
            axes[0,col].plot(x,[r['predicted_peak_latency_ms'] for r in rows],style+'o',
                color=color,label=f"Model tau_h={m[label]['tau_h_ms']:.3f} ms")
            axes[1,col].plot(x,[r['predicted_amplitude_ratio_to_100td'] for r in rows],style+'o',
                color=color,label=label)
            for k,bg in enumerate((100,10,1)):
                key=f'{label}_{bg}td_{pulse}ms_'
                ref=next(r['amplitude_native'] for r in rows if r['background_td']==100)
                axes[2,col].plot(z[key+'observed_time_seconds']*1000,
                    z[key+'observed_h1_delta_native']/ref,style,color=colors[k],
                    alpha=.75 if label=='baseline' else 1.,
                    label=f'{bg} td: {label}')
        axes[1,col].plot(x,[float(r['amplitude_mV'])/float(obs[0]['amplitude_mV']) for r in obs],
            'ko',label='Measured ratio of population means')
        for ax in axes[:2,col]:
            ax.set_xticks(x,['100 td\ntrain','10 td\ntrain','1 td\nheld out'])
            ax.grid(alpha=.2); ax.legend(fontsize=8)
        axes[0,col].set(title=f'{pulse} ms positive pulse, Weber contrast 19',ylabel='Peak latency (ms; includes 2.82 ms readout delay)')
        axes[1,col].set(ylabel='Peak amplitude / 100 td peak amplitude')
        axes[2,col].axvspan(2.82,pulse+2.82,color='#eeeeee',zorder=-1)
        axes[2,col].set(xlim=(0,250),xlabel='Time after light onset (ms)',
            ylabel='Simulated H1 change / 100 td peak magnitude',
            title='Simulated curves only; no measured cell trajectories')
        axes[2,col].grid(alpha=.2); axes[2,col].legend(fontsize=8,ncol=2)
    fig.suptitle('Smith et al. (2001), Table 2 vs homogeneous native Cone-H1 model\n'
        f"{m['decision']}; spatial initialization / full-chain integration BLOCKED",fontsize=14)
    fig.savefig(OUT/'H1_TEMPORAL_COMPARISON.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
