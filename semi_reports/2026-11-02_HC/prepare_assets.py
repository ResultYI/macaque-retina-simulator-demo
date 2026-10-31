"""Prepare handout figures from saved coordinates and a cropped original table."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT / 'reports/group_meeting_HC_20261102'
SOURCE = ROOT / 'data/source_packages/20261012/hc/Wassle_2000.pdf'


def source_table():
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import RectangleObject
    page = PdfReader(SOURCE).pages[11]
    box = (56, 557, 304, 737)
    page.cropbox = RectangleObject(box)
    page.mediabox = RectangleObject(box)
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({'/Title': 'Wassle et al. 2000, p.602 Table 2 (original crop)'})
    with (HERE / 'wassle2000_table2_original.pdf').open('wb') as f:
        writer.write(f)
    provenance = {
        'source': str(SOURCE),
        'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'printed_page': 602,
        'pdf_page_one_based': 12,
        'crop_box_pdf_points': list(box),
        'operation': 'crop original PDF only; no redraw or numeric change',
    }
    (HERE / 'asset_sources.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def distribution():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    data = np.load(REPORT / 'data/structure_0_6deg.npz', allow_pickle=False)
    xy = data['xy_mm'] / float(data['mm_per_degree'])
    types = data['subtype']
    assert {t: int((types == t).sum()) for t in ('H1', 'H2')} == {'H1': 82456, 'H2': 20597}
    assert np.linalg.norm(xy, axis=1).max() <= 6 + 1e-12
    model = np.genfromtxt(REPORT / 'data/density_model.csv', delimiter=',', names=True)
    annuli = np.genfromtxt(REPORT / 'data/density_annuli.csv', delimiter=',', names=True)
    colors = {'H1': '#176B9A', 'H2': '#C65D1C'}
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), gridspec_kw={'width_ratios': [1, 1, 1.13]})
    fig.subplots_adjust(left=.045, right=.99, bottom=.15, top=.77, wspace=.34)
    for ax, t in zip(axes[:2], ('H1', 'H2')):
        points = xy[types == t]
        ax.scatter(points[:, 0], points[:, 1], s=.15, color=colors[t],
                   alpha=.7, rasterized=True, linewidths=0)
        for radius in (2, 4, 6):
            ax.add_patch(Circle((0, 0), radius, fill=False, lw=.6,
                                ls='-' if radius == 6 else '--', color='#727D87'))
        ax.set(xlim=(-6.25, 6.25), ylim=(-6.25, 6.25), aspect='equal',
               xticks=(-6, -3, 0, 3, 6), yticks=(-6, -3, 0, 3, 6),
               xlabel='x (deg)', ylabel='y (deg)')
        ax.set_title(f'{t}: {len(points):,} cells', color=colors[t], pad=10)
    ax = axes[2]
    for t in ('H1', 'H2'):
        ax.plot(model['eccentricity_deg'], model[t + '_cells_mm2'] / 1000,
                color=colors[t], lw=1.8, label=f'{t} model')
        ax.scatter((annuli['lower_deg'] + annuli['upper_deg']) / 2,
                   annuli[t + '_cells_mm2'] / 1000, color=colors[t], facecolors='none',
                   s=16, lw=.8, label=f'{t} annuli')
    ax.axvline(.6 / .223, color='#6E7680', ls=':', lw=1)
    ax.set(xlim=(0, 6), ylim=(0, 20.5), yticks=(0, 5, 10, 15, 20),
           xlabel='Eccentricity (deg)', ylabel=r'Density ($10^3$ cells/mm$^2$)')
    sec = ax.secondary_xaxis('top', functions=(lambda v: v * .223, lambda v: v / .223))
    sec.set_xlabel('Retinal distance (mm)', labelpad=6)
    ax.legend(frameon=False, fontsize=8, loc='center right')
    fig.savefig(HERE / 'hc_distribution_density.pdf', dpi=300, bbox_inches='tight', pad_inches=.04)
    fig.savefig(HERE / 'hc_distribution_density.png', dpi=220, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def contact_sizes():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'pdf.fonttype':42, 'axes.spines.top':False, 'axes.spines.right':False})
    colors = {'H1':'#176B9A', 'H2':'#C65D1C'}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.8))
    for ax, maximum, scale, xlabel, filename in [(axes[0],1.338,.223,'Eccentricity (deg)','hc_size_curve.csv'),
                                      (axes[1],6.,1.,'Physical eccentricity (mm)','hc_size_full_curve.csv')]:
        data = np.genfromtxt(HERE/filename, delimiter=',', names=True)
        radii = data['eccentricity_mm']
        ax.axvspan(.5/scale,1./scale,color='#E6E6E6',alpha=.8,zorder=0)
        if maximum > 5:
            ax.axvspan(5.,6.,color='#E6E6E6',alpha=.8,zorder=0)
        for kind in ('H1','H2'):
            ax.plot(radii/scale, data[kind+'_diameter_um'],color=colors[kind],label=kind,lw=1.8)
        ax.set(xlim=(0,maximum/scale),xlabel=xlabel,ylabel='Equivalent contact diameter (um)')
        ax.legend(frameon=False)
    fig.tight_layout(w_pad=2)
    fig.savefig(HERE/'hc_size_connectivity.pdf',bbox_inches='tight',pad_inches=.04)
    fig.savefig(HERE/'hc_size_connectivity.png',dpi=220,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)


def response_probe():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    receipt = json.loads((HERE/'hc_response_validation.json').read_bytes())
    path = ROOT/receipt['response_data']['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['response_data']['sha256']
    with np.load(path,allow_pickle=False) as z:
        times, kinds, counts = z['time_s']*1000, z['HC_types'], z['HC_member_counts']
        traces = {name:z[name+'_HC_delta'].copy() for name in ('legacy','eccentric')}
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'pdf.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(10.5,2.6))
    for ax, kind, color in zip(axes,('H1','H2'),('#176B9A','#C65D1C')):
        mask = kinds == kind
        for name, style, label in [('legacy','--','Previous contacts'),('eccentric','-','Eccentric contact footprint')]:
            average = traces[name][:,mask]@counts[mask]/counts[mask].sum()
            ax.plot(times,average,ls=style,color=color,label=label,lw=1.7)
        ax.set(xlim=(0,50),xlabel='Time after light step (ms)',ylabel='HC change (native units)',title=kind)
        ax.legend(frameon=False,fontsize=8)
    fig.tight_layout(w_pad=2)
    fig.savefig(HERE/'hc_response_probe.pdf',bbox_inches='tight',pad_inches=.04)
    fig.savefig(HERE/'hc_response_probe.png',dpi=220,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)


if __name__ == '__main__':
    if sys.argv[1:] == ['table']:
        source_table()
    elif sys.argv[1:] == ['plot']:
        distribution()
    elif sys.argv[1:] == ['size']:
        contact_sizes()
    elif sys.argv[1:] == ['response']:
        response_probe()
    else:
        raise SystemExit('Use table with bundled Python; plot/size/response with D:/anaconda/python.exe')
