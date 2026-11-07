"""Plot saved native-unit trajectories only; never execute a model."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args(); out = args.run
    result = json.loads((out/'RESULT.json').read_bytes())
    if result['status'] != 'PASS_CONDITIONAL_REAL_GRAPH_DIAGNOSTIC':
        raise RuntimeError('A completed diagnostic is required for these result plots')
    values = {name: np.load(out/(name+'_TRAJECTORY.npz'), allow_pickle=False)
              for name in ('A_uniform', 'B_biexponential', 'C_feedback_off')}
    geo = np.load(out/'GEOMETRY.npz', allow_pickle=False)
    t = values['A_uniform']['t_seconds']*1000
    names = {'A_uniform': 'Uniform + local feedback', 'B_biexponential': 'Biexponential + local feedback',
             'C_feedback_off': 'Biexponential, feedback off'}
    colors = {'A_uniform': '#385b9c', 'B_biexponential': '#d65a31', 'C_feedback_off': '#43856b'}
    figures = []
    for stem, pair, title in [
            ('KERNEL_COMPARISON', ('A_uniform', 'B_biexponential'), 'Uniform vs biexponential: response from each compatible baseline'),
            ('FEEDBACK_COMPARISON', ('B_biexponential', 'C_feedback_off'), 'Feedback on vs off: response from separate compatible baselines')]:
        fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True, constrained_layout=True)
        for panel, ax in enumerate(axes.ravel()):
            for condition in pair:
                trace = values[condition]['H1_target'] if panel == 0 else values[condition]['release_core'][panel-1]
                ax.plot(t, trace-trace[0], lw=1.6, color=colors[condition],
                        ls='-' if condition == pair[0] else '--', label=names[condition])
            ax.axvspan(20, 120, color='#d7d7d7', alpha=.22)
            ax.set_title('H1 parent row 2891' if panel == 0 else 'Cone parent row '+str(geo['core_cone_parent_rows'][panel-1]), fontsize=10)
            ax.set_ylabel('Delta H1 (native units)' if panel == 0 else 'Delta release (native units)')
            ax.grid(alpha=.2); ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useOffset=False)
            ax.set_xlim(0, 120)
        for ax in axes[-1]: ax.set_xlabel('Time (ms)')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(title+'\n321 source cones / 249 existing H1; held exterior; one registered zero B', fontsize=12)
        for ext in ('png', 'pdf'): fig.savefig(out/(stem+'.'+ext), dpi=160)
        plt.close(fig); figures.append(stem+'.png')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    for condition, value in values.items():
        kw = dict(label=names[condition], color=colors[condition], lw=1.5)
        axes[0, 0].plot(t, value['H1_target'], **kw)
        axes[0, 1].plot(t, value['release_core'][0], **kw)
        axes[1, 0].plot(t, value['pooled_release_target'], **kw)
        axes[1, 1].plot(t, value['feedback_core'][0], **kw)
    for ax, title, ylabel in zip(axes.ravel(), ['Target H1', 'Cone '+str(geo['core_cone_parent_rows'][0]),
            'Target H1 spatial input', 'Same cone feedback input'],
            ['H1 (native units)', 'Release (native units)', 'A release + held background', 'B H1 (native units)']):
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel('Time (ms)'); ax.grid(alpha=.2)
        ax.axvspan(20, 120, color='#d7d7d7', alpha=.22); ax.set_xlim(0, 120)
        ax.ticklabel_format(axis='y', useOffset=False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Absolute trajectories and distinct equilibrium working points\nRaw native units; no amplitude normalization', fontsize=12)
    for ext in ('png', 'pdf'): fig.savefig(out/('ABSOLUTE_WORKING_POINTS.'+ext), dpi=160)
    plt.close(fig); figures.append('ABSOLUTE_WORKING_POINTS.png')
    (out/'PLOTS.json').write_text(json.dumps(dict(figures=figures, source='saved NPZ only',
        amplitude_normalization=False, response_definition='trace minus its own pre-stimulus baseline for comparison panels'), indent=2), encoding='utf-8')
    print('\n'.join(str(out/p) for p in figures))


if __name__ == '__main__': main()
