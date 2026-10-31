"""Standalone scientific figures from saved two-position curves only."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args(); out = args.directory
    with (out / 'ALL_POSITION_TYPE_TIME_COURSES.csv').open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    types = list(dict.fromkeys(r['type'] for r in rows))
    columns = [('deltaV_local_background_mV', 'Delta V from local background (mV)'),
               ('delta_rate_local_background_Hz', 'Delta source rate from local background (Hz)'),
               ('gE_per_represented_member_nS', 'gE / represented member (nS)'),
               ('gI_per_represented_member_nS', 'gI / represented member (nS)')]
    fig, axes = plt.subplots(len(types), len(columns), figsize=(17, 12), sharex=True, constrained_layout=True)
    for i, pop in enumerate(types):
        for position, color, style in [('near', '#1764a0', '-'), ('reference473', '#bd6524', '--')]:
            data = [r for r in rows if r['type'] == pop and r['position'] == position]
            t = np.array([float(r['time_seconds']) * 1000 for r in data])
            for j, (key, title) in enumerate(columns):
                ax = axes[i, j]
                ax.plot(t, [float(r[key]) for r in data], color=color, ls=style, lw=1.3,
                        label='0.06431 deg' if position == 'near' else '4.73 deg (saved)')
                for left, right, shade in [(50, 200, '#dceaf4'), (350, 500, '#fae5d4'), (600, 750, '#e3eedb')]:
                    ax.axvspan(left, right, color=shade, alpha=.22, zorder=-1)
                ax.grid(alpha=.18)
                if i == 0: ax.set_title(title, fontsize=10)
                if j == 0: ax.set_ylabel(pop.replace('_RGC', '').replace('small_bistratified_S_ON', 'S-bistratified'), fontsize=9)
                if i == len(types) - 1: ax.set_xlabel('Time (ms)')
                if i == 0 and j == 0: ax.legend(fontsize=8)
    fig.suptitle('Two local functional probes: fixed parameters, independent backgrounds\n'
        'Cone release x0.9 / x1.1; Rod input x2. No calibrated eccentricity function or matched physical background claim.', fontsize=13)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(out / ('two_position_functional_curves.' + ext), dpi=170)
    plt.close(fig)


if __name__ == '__main__': main()
