"""Prepare handout figures from saved coordinates and a cropped original table."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT / 'reports/group_meeting_HC_20261102'
SOURCE = ROOT / 'data/source_packages/20261012/hc/Wassle_2000.pdf'
CURRENT_RUN = ROOT / 'reports/HC_VISUAL6DEG_SUPPORT_REPAIR_20261101_v1'


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
    receipt = json.loads((CURRENT_RUN / 'BUILD_RESULT.json').read_bytes())
    assert receipt['status'] == 'GEOMETRY_READY'
    data = np.load(CURRENT_RUN / 'HC_GEOMETRY.npz', allow_pickle=False)
    assert str(data['current_hc_identity']) == receipt['current_hc_identity']
    xy = data['xy_mm'] / float(data['mm_per_degree'])
    types = data['subtype']
    assert {t: int((types == t).sum()) for t in ('H1', 'H2')} == receipt['counts']
    all_counts = {t: int((types == t).sum()) for t in ('H1', 'H2')}
    observation = np.linalg.norm(xy, axis=1) <= 6
    xy, types = xy[observation], types[observation]
    summary = dict(run_directory=str(CURRENT_RUN), anatomical_plot_radius_mm=1.338,
        support_radius_mm=receipt['finite_HC_radius_mm'], full_support_counts=all_counts,
        plotted_counts={t:int((types == t).sum()) for t in ('H1','H2')},
        current_hc_identity=receipt['current_hc_identity'])
    (HERE / 'hc_distribution_summary.json').write_bytes(
        (json.dumps(summary,indent=2)+'\n').encode('utf-8'))
    selection = json.loads((ROOT / 'configurations/hc_density_geometry_current.json').read_bytes())
    selected_id = hashlib.sha256(json.dumps(selection, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert selected_id == receipt['current_hc_identity']
    config_path = ROOT / selection['bindings']['parameters']['path']
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == selection['bindings']['parameters']['sha256']
    config = json.loads(config_path.read_bytes())
    coefficients = config['density']['coefficients']
    model = np.zeros(601, dtype=[('eccentricity_deg',float),('H1_cells_mm2',float),('H2_cells_mm2',float)])
    model['eccentricity_deg'] = np.linspace(0,6,len(model))
    rho = sum(v['c']*np.exp(v['lambda_per_mm']*model['eccentricity_deg']*.223) for v in coefficients)
    model['H1_cells_mm2'], model['H2_cells_mm2'] = .8*rho, .2*rho
    edges = np.linspace(0,6,25)
    areas = np.pi*.223**2*np.diff(edges**2)
    annuli = np.zeros(24,dtype=[('lower_deg',float),('upper_deg',float),('H1_cells_mm2',float),('H2_cells_mm2',float)])
    annuli['lower_deg'], annuli['upper_deg'] = edges[:-1],edges[1:]
    for t in ('H1','H2'):
        annuli[t+'_cells_mm2'] = np.histogram(np.linalg.norm(xy[types==t],axis=1),edges)[0]/areas
    for name,table in [('hc_density_model.csv',model),('hc_density_annuli.csv',annuli)]:
        np.savetxt(HERE/name,np.column_stack([table[k] for k in table.dtype.names]),delimiter=',',header=','.join(table.dtype.names),comments='')
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


def contact_size_data():
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from retina_sim.hc_joint_v2 import JointV2HCDensityProfile
    profile = JointV2HCDensityProfile(ROOT)
    samples_deg = np.array([0., 3., 6.])
    sample_r = np.array([0., 3. * .223, profile.support_radius_mm])
    radii = np.unique(np.r_[np.linspace(0., profile.support_radius_mm, 1201),
                            profile.config['geometry']['central_join']['interval_mm'], sample_r])
    values = {t: profile.contact_radius_mm(radii, t) * 1000 for t in ('H1', 'H2')}
    assert all(np.isfinite(v).all() and (v > 0).all() for v in values.values())
    assert np.all(values['H2'] > values['H1'])
    central = radii <= profile.config['geometry']['central_join']['interval_mm'][0]
    for t in values:
        assert np.allclose(values[t][central], profile.config['geometry']['central_radii_mm'][t] * 1000,
                           rtol=0, atol=1e-12)
    csv = HERE / 'hc_contact_radius_model.csv'
    np.savetxt(csv, np.column_stack([radii / .223, radii, values['H1'], values['H2']]),
               delimiter=',', comments='',
               header='eccentricity_deg,retinal_distance_mm,H1_radius_um,H2_radius_um')
    metadata = {
        'quantity': 'Direct Cone-input contact radius; not a measured functional RF',
        'current_hc_identity': profile.current_selection['identity'],
        'source_bindings': {k: profile.current_selection['bindings'][k]
                            for k in ('hc_joint_v2.py', 'parameters')},
        'function': 'JointV2HCDensityProfile.contact_radius_mm',
        'domain_mm': [0., profile.support_radius_mm],
        'mm_per_degree': .223,
        'csv_sha256': hashlib.sha256(csv.read_bytes()).hexdigest(),
        'samples_degrees': samples_deg.tolist(),
        'samples_radius_um': {t: (profile.contact_radius_mm(sample_r, t) * 1000).tolist()
                              for t in values},
        'radius_min_max_um': {t: [float(v.min()), float(v.max())] for t, v in values.items()},
        'central_join_mm': profile.config['geometry']['central_join']['interval_mm'],
        'simulation_rerun': False,
    }
    (HERE / 'hc_contact_radius_model.json').write_bytes(
        (json.dumps(metadata, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'current_hc_identity': metadata['current_hc_identity'],
                      'samples_radius_um': metadata['samples_radius_um']}))


def contact_sizes():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    metadata = json.loads((HERE / 'hc_contact_radius_model.json').read_bytes())
    csv = HERE / 'hc_contact_radius_model.csv'
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == metadata['csv_sha256']
    selected = json.loads((ROOT / 'configurations/hc_density_geometry_current.json').read_bytes())
    selected_id = hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(',', ':'),
                                           allow_nan=False).encode()).hexdigest()
    assert selected_id == metadata['current_hc_identity']
    for binding in metadata['source_bindings'].values():
        assert hashlib.sha256((ROOT / binding['path']).read_bytes()).hexdigest() == binding['sha256']
    data = np.genfromtxt(csv, delimiter=',', names=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                         'pdf.fonttype':42, 'axes.spines.top':False, 'axes.spines.right':False})
    colors = {'H1':'#176B9A', 'H2':'#C65D1C'}
    fig, (ax, circles) = plt.subplots(1, 2, figsize=(8.5, 3.1),
                                     gridspec_kw={'width_ratios': [1.15, 1]})
    for t in colors:
        values = data[t + '_radius_um']
        ax.plot(data['eccentricity_deg'], values, color=colors[t], label=t, lw=2)
        ax.annotate(f'{values[0]:.2f}', (0, values[0]), xytext=(5, 6),
                    textcoords='offset points', color=colors[t], fontsize=10)
        ax.scatter([6], [values[-1]], color=colors[t], s=16, zorder=3)
        ax.annotate(f'{values[-1]:.2f}', (6, values[-1]), xytext=(-3, 6), ha='right',
                    textcoords='offset points', color=colors[t], fontsize=10)
    ax.set(xlim=(0, 6.05), ylim=(0, 17), xticks=range(7), yticks=(0, 4, 8, 12, 16),
           xlabel='HC soma eccentricity (deg)', ylabel=r'Contact radius ($\mu$m)')
    ax.legend(frameon=False, loc='lower left', ncol=2)
    ax.grid(axis='y', color='#D9DEE2', lw=.6)
    sec = ax.secondary_xaxis('top', functions=(lambda v: v * .223, lambda v: v / .223))
    sec.set_xlabel('Retinal distance (mm)', labelpad=7)
    sec.set_xticks([0, .3, .6, .9, 1.2])
    for x, ecc in zip((0, 40, 80), metadata['samples_degrees']):
        circles.text(x, 20, f'{ecc:g}°', ha='center', va='bottom')
    for t, y in [('H1', 0), ('H2', -38)]:
        circles.text(-23, y, t, color=colors[t], ha='right', va='center', fontweight='bold')
        for x, radius in zip((0, 40, 80), metadata['samples_radius_um'][t]):
            circles.add_patch(Circle((x, y), radius, facecolor=colors[t] + '24',
                                      edgecolor=colors[t], lw=1.5))
            circles.text(x, y, f'{radius:.2f}', ha='center', va='center', fontsize=9.5)
    circles.plot([60, 80], [-59, -59], color='#303940', lw=2)
    circles.text(70, -62, r'20 $\mu$m', ha='center', va='top', fontsize=9)
    circles.set(xlim=(-34, 99), ylim=(-71, 29), aspect='equal')
    circles.set_title(r'Same-scale input circles; radius in $\mu$m', fontsize=10.5, pad=9)
    circles.axis('off')
    fig.tight_layout(w_pad=1.5)
    fig.savefig(HERE / 'hc_contact_radius_comparison.pdf', bbox_inches='tight', pad_inches=.04)
    fig.savefig(HERE / 'hc_contact_radius_comparison.png', dpi=220, bbox_inches='tight', pad_inches=.04)
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
    elif sys.argv[1:] == ['size-data']:
        contact_size_data()
    elif sys.argv[1:] == ['response']:
        response_probe()
    else:
        raise SystemExit('Use size-data with project Python; table with bundled Python; plot/size/response with D:/anaconda/python.exe')
