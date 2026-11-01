"""Plot an existing local movie response; never advance or alter the model."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAVED = ROOT / 'reports/group_meeting_HC_20261102'
VIDEO = ROOT / 'reports/cone_video_113_4s_eccentric_aperture_0_6deg_20261029_v1'
BASE = ROOT / 'reports/external_video_two_positions_4s_20261029_v1'
TARGET = 1.0


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract():
    from scipy.io import loadmat

    evidence = json.loads((SAVED / 'EVIDENCE.json').read_bytes())
    source_video = VIDEO / 'source_rgb.mat'
    preview_identity = json.loads((BASE / 'PREVIEW_IDENTITY.json').read_bytes())
    assert sha(source_video) == preview_identity['original_source_rgb_sha256']
    response_path = SAVED / 'data/far_responses.npz'
    with np.load(response_path, allow_pickle=False) as data:
        time, side = data['time_seconds'], data['side']
        indices = np.flatnonzero((np.abs(time - TARGET) < 1e-12) & (side == 'left'))
        assert indices.size, 'Need the recorded left limit at the exposure end'
        index = int(indices[-1])
        snapshot_time = float(time[index])
        assert data['hc'].shape == (437, len(time))
        center, xy = data['center_deg'], data['HC__xy']
        h0, h = data['hc'][:, 0], data['hc'][:, index]
        delta = h - h0
        types, members = data['HC__subtypes'], data['HC__members']
        ids = data['HC__ids']
        assert np.isfinite(delta).all() and np.max(np.abs(delta)) > 0
    source_chunk = BASE / 'far_run_v1/RESPONSE_THROUGH_1000ms.npz'
    expected = evidence['sources'][str(source_chunk.relative_to(ROOT))]
    assert sha(source_chunk) == expected
    with np.load(source_chunk, allow_pickle=False) as raw:
        j = np.flatnonzero((np.abs(raw['time_seconds'] - snapshot_time) < 1e-12)
                          & (raw['side'] == 'left'))[-1]
        assert np.array_equal(raw['ports__HC:Vh'][:, j], h)
        assert np.array_equal(raw['ports__HC:Vh'][:, 0], h0)
    bank = loadmat(source_video, variable_names=['source_rgb'])['source_rgb']
    assert bank.shape == (360, 640, 3, 120) and bank.dtype == np.uint8
    frame = int(round(TARGET * 30)) - 1
    rgb = bank[:, :, :, frame]
    np.savez_compressed(HERE / 'hc_video_response_preview.npz',
                        input_rgb=rgb, xy_deg=xy, center_deg=center, types=types,
                        members=members, group_ids=ids, h0=h0, h=h, delta_h=delta,
                        time_seconds=snapshot_time, source_frame_index=frame)
    provenance = dict(
        operation='Saved-results plotting only; no simulation or parameter changes',
        run_id=evidence['run_id'], initialized_model=evidence['positions']['far']['source_model_id'],
        connection_version='Existing 2026-10-29 movie connectivity; predates eccentric-contact revision',
        connection_not_latest_eccentric_revision=True,
        snapshot_seconds=snapshot_time, response_side='left', source_frame_index=frame,
        exposure_interval_seconds=[frame / 30, (frame + 1) / 30],
        source_rgb=dict(path=str(source_video), sha256=sha(source_video)),
        source_chunk=dict(path=str(source_chunk), sha256=expected),
        extracted_responses=dict(path=str(response_path), sha256=sha(response_path)),
        scope='Existing far local representative groups only; no full-field response or interpolation',
        center_deg=center.tolist(),
        center_eccentricity_deg=float(np.linalg.norm(center)),
        groups={t: int(np.sum(types == t)) for t in ('H1', 'H2')},
        unit='vH_native_voltage; uncalibrated to physical mV',
        quantity='delta_h = h(t) - h(0)',
        spatial_interpolation=False, members_expanded_as_independent_cells=False,
        source_port='ports__HC:Vh; continuous HC state, no observation delay',
        source_frame_verified_against_saved_exposure=True,
        FIT_READY=False, physiological_validation=False)
    (HERE / 'hc_video_response_preview.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('Verified saved t=1 s response and original stimulus frame 29', flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with np.load(HERE / 'hc_video_response_preview.npz', allow_pickle=False) as data:
        rgb, xy, center, types, delta = [data[k] for k in
            ('input_rgb', 'xy_deg', 'center_deg', 'types', 'delta_h')]
    assert len(xy) == len(types) == len(delta) == 437
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': True,
                         'axes.spines.right': True})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    fig.subplots_adjust(left=.055, right=.92, top=.84, bottom=.19, wspace=.39)
    ax = axes[0]
    ax.imshow(rgb, extent=(-6, 6, -6, 6), origin='upper', interpolation='nearest')
    ax.plot(center[0], center[1], marker='+', ms=10, mew=1.5, color='#ffea00')
    ax.annotate('HC patch', center, xytext=(2, -3), color='white', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='white', lw=1))
    ax.set(title='A  Input frame (t = 1.00 s)', xlabel='x (deg)', ylabel='y (deg)',
           xlim=(-6, 6), ylim=(-6, 6), xticks=(-6, 0, 6), yticks=(-6, 0, 6))
    relative = xy - center
    limit = float(np.max(np.abs(delta)))
    assert limit > 0 and np.isfinite(limit)
    low, high = relative.min(0) - .1, relative.max(0) + .1
    mid = (low + high) / 2
    half = max(high - low) / 2
    for ax, typ, label in zip(axes[1:], ('H1', 'H2'), ('B', 'C')):
        sel = types == typ
        points = ax.scatter(*relative[sel].T, c=delta[sel], s=24, cmap='RdBu_r',
                            vmin=-limit, vmax=limit, linewidths=0, rasterized=True)
        ax.plot(0, 0, marker='+', color='black', ms=6, mew=.8)
        ax.set(title=f'{label}  {typ} response ({int(sel.sum())} groups)',
               xlabel='relative x (deg)', ylabel='relative y (deg)', aspect='equal',
               xlim=(mid[0]-half, mid[0]+half), ylim=(mid[1]-half, mid[1]+half))
    colorbar = fig.colorbar(points, cax=fig.add_axes([.94, .23, .014, .53]))
    colorbar.set_label(r'$\Delta h$ (native units)')
    fig.text(.055, .035, 'Far local patch (4.73 deg) | Saved 2026-10-29 movie network | '
             'No spatial interpolation', fontsize=9, color='#555555')
    for ext in ('png', 'pdf'):
        fig.savefig(HERE / ('hc_video_response_preview.' + ext), dpi=300)
    plt.close(fig)
    print('HC preview rendered from saved data', flush=True)


def extract_fullfield():
    from scipy.io import loadmat
    sys.path.insert(0,str(ROOT))
    from retina_sim.disk_tables import FrozenDiskTable
    from retina_sim.spatial_group_model import _verify_artifact
    from tools.build_eccentric_hc import load_matrix

    run = ROOT / 'reports/HC_FULLFIELD_FIRST_FRAME_0_6DEG_20261101_v1'
    result = json.loads((run/'RESULT.json').read_bytes())
    contract = json.loads((run/'CONTRACT.json').read_bytes())
    assert result['status'] == 'COMPLETE_VALIDATED' and result['accepted']
    candidate_path=ROOT/'reports/retinotopic_rf_revision/native_validation_v1/CANDIDATE_MODEL.json'
    assert sha(candidate_path)==contract['identity']['candidate_sha256']
    candidate=json.loads(candidate_path.read_bytes())['fine_source_model']
    cones=FrozenDiskTable(ROOT,candidate['tables']['cones'])
    hcs=FrozenDiskTable(ROOT,candidate['tables']['HC'])
    A=load_matrix(candidate['native_A'])
    mass=np.asarray(A.sum(axis=1)).ravel()
    has_input=mass>0
    # Contact weights define a display position, independent of stimulus or response.
    center_mm=np.divide(A@cones.coordinates['rf'],mass[:,None],
        out=np.full((hcs.count,2),np.nan),where=has_input[:,None])
    center_deg=center_mm/.223
    selected=has_input & (np.linalg.norm(center_deg,axis=1)<=6.)
    assert int((~has_input).sum())==29
    source_rows=np.flatnonzero(selected)
    parent=ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2'
    parent,_,parent_identity=_verify_artifact(parent)
    assert parent_identity==contract['identity']['parent_artifact']
    mapping=np.load(parent/'populations/HC/fine_to_group.npy',allow_pickle=False)
    group_counts=np.load(parent/'populations/HC/counts.npy',allow_pickle=False)
    with np.load(run/'GEOMETRY_AND_INPUT.npz',allow_pickle=False) as d:
        active=d['active_HC_parent_rows']
    inverse=np.full(len(group_counts),-1,dtype=int)
    inverse[active]=np.arange(len(active))
    assert np.all(mapping[source_rows]>=0)
    group_rows=inverse[mapping[source_rows]]
    assert np.all(group_rows>=0)
    response_path=run/'STANDARD.npz'
    assert sha(response_path)==result['receipts']['standard']['response_sha256']
    with np.load(response_path,allow_pickle=False) as d:
        delta_by_group=d['HC_delta'][:,-1]
        assert float(d['time_seconds'][-1])==1/30
    delta=delta_by_group[group_rows]
    soma_deg=hcs.coordinates['soma'][source_rows]/.223
    outside_anatomical_window=int(np.sum(np.linalg.norm(soma_deg,axis=1)>6.))
    assert outside_anatomical_window>0 and np.isfinite(delta).all()
    arrays=dict(xy_deg=center_deg[selected],soma_xy_deg=soma_deg,HC_fine_rows=source_rows,
        delta_h=delta,supported=np.ones(len(source_rows),dtype=bool),
        HC_types=hcs.extras['subtype'][source_rows],group_active_rows=group_rows,
        duration_seconds=1/30,source_frame_index=0,
        color_limit=float(np.max(np.abs(delta_by_group))))
    display = loadmat(VIDEO/'input.mat',variable_names=['rgb_frames'])['rgb_frames']
    assert display.shape[:3] == (360,360,3)
    arrays['input_rgb'] = display[:,:,:,0]
    np.savez_compressed(HERE/'hc_fullfield_response_preview.npz',**arrays)
    provenance = dict(run_directory=str(run), result_sha256=sha(run/'RESULT.json'),
        contract_sha256=sha(run/'CONTRACT.json'),network=contract['geometry'],
        numerical=dict(maximum_all_state_budget_ratio=result['maximum_all_state_budget_ratio'],
            maximum_HC_standard_tight_difference=result['maximum_HC_standard_tight_difference']),
        snapshot_seconds=float(arrays['duration_seconds']),source_frame_index=0,
        input_image='Frozen cropped rgb_frames[:,:,:,0] from original ISET input.mat',
        quantity='HC delta_h = h(1/30 s) - h(0); all types displayed together',
        unit='vH_native_voltage; uncalibrated to physical mV',
        displayed_HC=len(source_rows),unsupported_HC=29, spatial_interpolation=False,
        coordinate_frame='functional_visual_field',
        position_definition='Fine HC direct-cone weighted RF centroid; display approximation, not functional RF fit',
        position_equation='x_display_j = sum_i A_fine_ji*x_cone_RF_i / sum_i A_fine_ji',
        position_weights='Frozen complete fine-cell Cone-to-HC contact matrix; independent of image and response',
        position_matrix=candidate['native_A'],
        selection='All finite-network HC with defined direct-input centroid inside the 0-6 degree visual field',
        newly_included_from_outside_anatomical_6deg=outside_anatomical_window,
        unsupported_position_policy='29 zero-input HC have no defined input centroid; omitted from this visual-field view and retained in anatomical data',
        representation='Saved representative HC state placed at each fine HC direct-input visual-field centroid',
        raw_response_sha256=sha(response_path),
        color_limit=arrays['color_limit'],color_limit_rule='Maximum absolute HC change over the same saved complete finite network',
        response_values_changed=False,new_simulation=False,
        latest_eccentric_contacts=True, latest_fine_before_group_typed_feedback=True,
        simulation_scope='Complete finite Cone-HC network; image drives all captured 0-6 degree cones; outside light remains at reference',
        complete_4s_movie=False, physiological_validation=False,
        plotting_source_sha256=sha(Path(__file__)))
    (HERE/'hc_fullfield_response_preview.json').write_text(
        json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(f'Visual-field view: {len(source_rows)} HC; {outside_anatomical_window} were outside the anatomical 6-degree window',flush=True)


def plot_fullfield():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    with np.load(HERE/'hc_fullfield_response_preview.npz',allow_pickle=False) as data:
        rgb,xy,delta,supported=[data[k] for k in ('input_rgb','xy_deg','delta_h','supported')]
        limit=float(data['color_limit'])
    assert len(xy)==len(delta) and supported.all()
    assert np.linalg.norm(xy,axis=1).max()<=6.
    assert limit>0 and np.isfinite(limit)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.25))
    fig.subplots_adjust(left=.065,right=.875,bottom=.16,top=.88,wspace=.32)
    axes[0].imshow(rgb,extent=(-6,6,-6,6),origin='upper',interpolation='nearest')
    axes[0].set_title('A  Input frame')
    points=axes[1].scatter(*xy[supported].T,c=delta[supported],s=.32,
        cmap='RdBu_r',vmin=-limit,vmax=limit,linewidths=0,rasterized=True)
    axes[1].add_patch(Circle((0,0),6,fill=False,lw=.6,color='#666666'))
    axes[1].set_title('B  HC response (t = 33.3 ms)')
    for ax in axes:
        ax.set(xlim=(-6,6),ylim=(-6,6),aspect='equal',xlabel='visual-field x (deg)',ylabel='visual-field y (deg)',
               xticks=(-6,-3,0,3,6),yticks=(-6,-3,0,3,6))
    colorbar=fig.colorbar(points,cax=fig.add_axes([.905,.22,.015,.57]))
    colorbar.set_label(r'$\Delta h$ (native units)')
    fig.text(.065,.035,f'0-6 deg visual field | {len(xy):,} HC | Positions: direct-cone input centroids',
             fontsize=9,color='#555555')
    for ext in ('png','pdf'):
        fig.savefig(HERE/('hc_fullfield_response_preview.'+ext),dpi=400)
    plt.close(fig)
    print('Combined full-field HC preview rendered',flush=True)


if __name__ == '__main__':
    {'extract':extract,'plot':plot,'extract-fullfield':extract_fullfield,
     'plot-fullfield':plot_fullfield}[sys.argv[1]]()
