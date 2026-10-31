"""Plot saved scientific data only; may use a separate installed plotting Python."""
import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coupled', action='store_true')
    args = parser.parse_args()
    base = args.output.resolve()
    out = base / 'source_coupled_v1/analysis' if args.coupled else base
    if (out / 'functional_curves.png').exists():
        raise FileExistsError('Preserve existing figures')
    result = json.loads((out / 'RESULT.json').read_bytes())
    rows = json.loads((out / 'TYPE_COMPLETION.json').read_bytes())
    packet = base / 'source_coupled_v1/standard/RESPONSE.npz' if args.coupled else out / 'assay_standard.npz'
    z = dict(np.load(packet, allow_pickle=False))
    records = list(csv.DictReader((out / 'ALL_TYPE_MEAN_TIME_COURSES.csv').open(encoding='utf8')))
    fig, axes = plt.subplots(5, 3, figsize=(14, 13), sharex=True, constrained_layout=True)
    aliases = ['Midget ON', 'Midget OFF', 'Parasol ON', 'Parasol OFF', 'Small bistratified S-ON']
    for i, row in enumerate(rows):
        data = [r for r in records if r['type'] == row['type']]
        col = lambda name: np.array([float(r[name]) for r in data])
        t = col('time_seconds') * 1000
        axes[i, 0].plot(t, col('deltaV_from_background_V') * 1000, color='#263845', lw=1.5)
        if i == 4:
            axes[i, 0].plot(t, (col('selective_S_V_V') - col('selective_S_V_V')[0]) * 1000,
                            color='#ad4f2b', lw=1.4, label='Selective S-BC input')
            axes[i, 0].legend(fontsize=8)
        for field, color, label in [('gE_S', '#176c8a', 'Excitation'), ('gI_S', '#b44744', 'Inhibition')]:
            v = col(field)
            axes[i, 1].plot(t, (v - v[0]) * 1e9, color=color, label=label, lw=1.4)
        axes[i, 2].plot(t, col('source_post_tick_rate_Hz'), color='#176c8a', lw=1.5)
        if i == 4:
            axes[i, 2].plot(t, col('selective_S_rate_Hz'), color='#ad4f2b', lw=1.3)
        for ax in axes[i]:
            ax.axvspan(50, 200, color='#e7edf1', alpha=.8)
            ax.axvspan(350, 500, color='#f5e8db', alpha=.8)
            ax.axvspan(600, 750, color='#e4eedc', alpha=.7)
            ax.grid(axis='y', color='#dddddd', lw=.6)
            ax.set_xlim(0, 900)
            ax.spines[['top', 'right']].set_visible(False)
        axes[i, 0].set_ylabel(aliases[i] + '\nDelta V (mV)')
        axes[i, 1].set_ylabel('Delta conductance (nS)')
        axes[i, 2].set_ylabel('Original expected rate (Hz)')
    axes[0, 0].set_title('Membrane response versus constant background')
    axes[0, 1].set_title('Original E/I conductance dynamics')
    axes[0, 2].set_title('Retained binary-history output')
    axes[0, 1].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Time (ms)')
    label = 'original coupled inner source assay' if args.coupled else 'conditional original-equation BC replay'
    fig.suptitle('RGC functional v1: ' + label + '\n'
                 '5 types / 35 groups; no fitted coefficients or new connections', fontsize=14)
    fig.savefig(out / 'functional_curves.png', dpi=170)
    fig.savefig(out / 'functional_curves.svg')
    fig.savefig(out / 'functional_curves.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for ax, row, alias in zip(axes, rows, aliases):
        p = row['type']
        ax.scatter(z[p + '__event_time_seconds'] * 1000, z[p + '__event_group_index'],
                   marker='|', s=32, linewidths=.7, color='#263845')
        ax.set_ylabel(alias + '\nGroup index', fontsize=8)
        ax.set_ylim(-.7, row['groups'] - .3)
        ax.set_yticks(range(row['groups']))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color='#dddddd', lw=.6)
    axes[-1].set_xlabel('Time (ms); native first-event time inside each 1 ms bin')
    axes[-1].set_xlim(0, 900)
    fig.suptitle('Actual original PCG64 sampled group-object events\n'
                 'One object per computed group; no multiplication or replication of fine-cell spikes')
    fig.savefig(out / 'native_group_spikes.png', dpi=170)
    fig.savefig(out / 'native_group_spikes.svg')
    plt.close(fig)
    table = ''.join('<tr>' + ''.join('<td>' + html.escape(str(row[k])) + '</td>' for k in
        ('type', 'groups', 'status', 'baseline_source_rate_Hz', 'common_decrease_deltaV_mV',
         'common_increase_deltaV_mV', 'actual_sampled_object_events')) + '</tr>' for row in rows)
    scope = ('Main curves integrate all6305 original coupled inner states with the retained100us/1ms clocks. '
             'Controls use conditional driven equations; original Cone/HC background remains held.' if args.coupled
             else 'Main curves use1ms interpolation of original BC-cache input and the retained AC network.')
    links = '../..' if args.coupled else '.'
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>RGC functional v1</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 16px;line-height:1.5}}
img{{max-width:100%;height:auto}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border:1px solid #ddd}}
code{{background:#f1f3f5;padding:2px 4px}}</style><h1>RGC functional v1 — local conditional result</h1>
<p>{html.escape(result['status'])}. Numerical reference: {result['max_numerical_error_over_budget']:.6g} of original allocation.</p>
<p>{scope} Whole-retina equivalence, physiological calibration, optical spectral isolation and cross-eccentricity calibration remain open.</p>
<table><tr><th>Type</th><th>Groups</th><th>Status</th><th>Baseline Hz</th><th>Release decrease ΔV mV</th><th>Release increase ΔV mV</th><th>Actual events</th></tr>{table}</table>
<img src="functional_curves.png" alt="Five RGC types conductance, membrane and expected output curves">
<p>Shaded epochs: native Cone release ×0.9 (50–200 ms), ×1.1 (350–500 ms), and original rod light ×2 (600–750 ms).
These are layer-input assays, not optical contrast calibration. Orange S traces use a selective BC-terminal condition.
Small-bistratified common-input net polarity remains mixed/OFF-like.</p>
<img src="native_group_spikes.png" alt="Actual native group-object spike raster">
<p>Rate curves retain the original post-tick exported convention. Exact pre-bin expected binary counts and actual draws are stored separately;
the endpoint has no sampled future bin. Type curves use member-weighted descriptive means; event totals count sampled group objects only.</p>
<p><a href="TYPE_COMPLETION.csv">Five-type completion table</a> · <a href="ALL_TYPE_MEAN_TIME_COURSES.csv">Curve data CSV</a> ·
<a href="NUMERICAL_REFERENCE.csv">Numerical reference</a> · <a href="{links}/PARAMETERS.json">Frozen parameter ledger</a> ·
<a href="SOURCE_ENDPOINT_COMPARISON.csv">Saved full-inner endpoint comparison</a> · <a href="{links}/RESOURCE_LEDGER.json">Resource ledger</a></p>
<p><code>FIT_READY=false</code> · <code>background_qualified=false</code> · <code>production_adopted=false</code>.</p></html>'''
    (out / 'index.html').write_text(page, encoding='utf8')
    print(json.dumps(dict(figures=['functional_curves.png', 'native_group_spikes.png'],
                          report='index.html', model_rerun=False)))


if __name__ == '__main__':
    main()
