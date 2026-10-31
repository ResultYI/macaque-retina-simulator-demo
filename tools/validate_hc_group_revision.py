"""Native Cone/H1/H2 probe on the existing far patch with fresh backgrounds."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.disk_graph_storage import file_sha, save_json
from retina_sim.h1_h2_effective import global_operators, joint_equilibrium, matrix_id
from retina_sim.hc_lateral_candidate import LateralHCCandidate
from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
from retina_sim.spatial_group_graph import aggregate_edges, mean_matrix
from retina_sim.spatial_group_model import GroupModelAssembly, _verify_artifact
from retina_sim.vh2005 import SOURCE_VALUES
from tools.build_eccentric_hc import load_matrix, DIRECTORY as SOURCE_DIRECTORY
from retina_sim.hc_group_projection import project_hc_contacts
from retina_sim.disk_tables import FrozenDiskTable
from tools.durable_research_runtime_v1 import DurableRun
import os, traceback
DIRECTORY = Path('reports/retinotopic_rf_revision/native_validation_v1')
HANDOUT = ROOT/DIRECTORY


PREPARED = ROOT/'reports/position_S0_far_originaltarget_20261029_resume_v2/prepared'
BACKGROUND = PREPARED.parent/'background/BACKGROUND.npz'


def main():
    started = time.perf_counter()
    HANDOUT.mkdir(parents=True,exist_ok=True)
    with (HANDOUT/'STARTED.json').open('x',encoding='utf8') as stream:
        json.dump(dict(pid=os.getpid(),source_sha256=file_sha(__file__),no_auto_retry=True),stream)
    journal=DurableRun(HANDOUT/'runtime','HC_group_projection_native_probe')
    memory = MemoryGuard(24*1024**3, 8*1024**3)
    def guard():
        journal.heartbeat(phase='new_background_or_native_probe')
        if time.perf_counter()-started > 300:
            raise TimeoutError('Bounded native probe exceeded 300 seconds')
        if not memory.check():
            raise MemoryError(str((memory.stop_reason,memory.last)))
    guard()
    parent_dir, manifest, parent_identity = _verify_artifact(
        ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2')
    def array(name):
        return np.load(parent_dir/name, allow_pickle=False, mmap_mode='r')
    populations = {}
    for name in ('cones', 'HC'):
        populations[name] = SimpleNamespace(**{k:array('populations/'+name+'/'+k+'.npy') for k in
            ('fine_to_group','counts','centroid_mm','group_ids')})
        populations[name].group_count = len(populations[name].counts)
    full = SimpleNamespace(manifest=manifest, artifact_sha256=parent_identity,
        populations=populations, array=array, _path=lambda name:parent_dir/name)
    full.matrix = lambda name:GroupModelAssembly.matrix(full,name)
    cfg = json.loads((PREPARED/'local_static/selection_identity.json').read_bytes())
    with np.load(PREPARED/'local_static/selection.npz', allow_pickle=False) as z:
        c, h = z['population__cones'], z['population__HC']
    with np.load(BACKGROUND, allow_pickle=False) as z:
        light = z['native_background'].copy()  # No saved state is reused.
    types = {}
    for name in ('cones', 'HC'):
        meta = full.manifest['populations'][name]
        key = full.array('populations/'+name+'/keys.npy')
        types[name] = np.asarray(meta['subtype_labels'])[key[:,meta['key_columns'].index('subtype')]]
    cp, hp = full.populations['cones'], full.populations['HC']
    operator = json.loads((ROOT/SOURCE_DIRECTORY/'operator.json').read_bytes())
    original_model=json.loads((ROOT/'configurations/atlas_hc_eccentric_size_w2000_v1.json').read_bytes())
    fine_cones=FrozenDiskTable(ROOT,original_model['tables']['cones'])
    fine_HC=FrozenDiskTable(ROOT,original_model['tables']['HC'])
    for name,table in [('cones',fine_cones),('HC',fine_HC)]:
        if full.manifest['populations'][name]['source_table'] != original_model['tables'][name]:
            raise ContractError('Parent source ID order differs from fixed fine tables')
        mapping=full.populations[name].fine_to_group;valid=mapping>=0
        if not np.array_equal(table.extras['subtype'].astype('U')[valid],types[name][mapping[valid]]):
            raise ContractError('Actual source subtypes differ from group member labels')
    raw=load_matrix(operator['raw_contacts'])
    projection=project_hc_contacts(raw,fine_cones.extras['subtype'].astype('U1'),fine_HC.extras['subtype'].astype('U2'),
        cp.fine_to_group,hp.fine_to_group,cp.counts,hp.counts)
    new_A,new_B=projection['A'],projection['typed_B']
    previous_A=sparse.load_npz(ROOT/original_model['group_native_operators']['A']['path'])
    difference_A=new_A-previous_A
    if difference_A.nnz and np.max(abs(difference_A.data))>2e-14:
        raise ContractError('This correction must retain existing Cone-to-HC input aggregation')
    new_A=previous_A.copy().tocsr()  # Preserve the exact existing summation order and Q identity.
    expected=sparse.load_npz(ROOT/'reports/retinotopic_rf_revision/candidate_operators/typed_B_fine_normalized_before_grouping.npz')
    difference_B=new_B-expected
    if difference_B.nnz and np.max(abs(difference_B.data))>2e-14:
        raise ContractError('Reusable correction differs from the independently diagnosed fine-before-group operator')
    global_h1,global_h2=np.flatnonzero(types['HC']=='H1'),np.flatnonzero(types['HC']=='H2')
    global_light = full.array('input/group_background_native_td.npy').copy()
    global_light[c] = light
    params = dict(tau_seconds=SOURCE_VALUES['tau_h']/1000,
                  input_scale=SOURCE_VALUES['g_t'], feedback_scale=1.)
    htypes, ctypes = types['HC'][h], types['cones'][c]
    h1, h2 = np.flatnonzero(htypes == 'H1'), np.flatnonzero(htypes == 'H2')
    populations = {}
    for name, original, rows in [('cones', cp, c), ('HC', hp, h)]:
        populations[name] = SimpleNamespace(group_count=len(rows), group_ids=original.group_ids[rows],
            counts=original.counts[rows], recording_auxiliary_group_rows=np.array([], dtype=int),
            recording_auxiliary_fine_rows=np.array([], dtype=int))
    center = np.asarray(cfg['configuration']['rf_center_deg'])
    spot_radius = cfg['configuration']['patch_geometry']['core_radius_deg']
    spot = np.linalg.norm(cp.centroid_mm[c]/.223-center, axis=1) <= spot_radius
    if not spot.any():
        raise ContractError('Existing patch core has no cone group')
    initial, responses, receipts = {}, {}, {}
    observations = np.linspace(0, .05, 101)
    for label, ga, gb in [('corrected', new_A, new_B)]:
        print('Preparing fresh native background:', label, flush=True)
        guard()
        if params['input_scale'] != SOURCE_VALUES['g_t'] or params['feedback_scale'] != 1.:
            raise ContractError('Shared-gain background shortcut requires the unchanged eta1 parameters')
        # shortcut: omit the redundant CSC assignment only for equal gains and
        # unit feedback; the guard rejects future subtype-specific settings.
        eq = joint_equilibrium(ga, gb, global_h1, np.array([], dtype=int),
                               global_light, params, guard=guard)
        A, B = ga[h][:,c].tocsr(), gb[c][:,h].tocsr()
        if np.any(np.asarray(ga[h].sum(1)).ravel() <= 0):
            raise ContractError('Selected local HC has no input; its native pooled adaptation is undefined')
        matrices = {'routes/cones_to_HC/mean_weights.npz':A, 'routes/HC_to_cones/mean_weights.npz':B}
        identity = dict(parent_artifact=full.artifact_sha256, profile=dict(fine_contacts=operator['profile'],group_projection=file_sha(ROOT/'retina_sim/hc_group_projection.py')),
                        A=matrix_id(ga), B=matrix_id(gb), cone_rows=c, HC_rows=h)
        assembly = SimpleNamespace(populations=populations, matrix=lambda name, m=matrices:m[name],
            operators=dict(A1=A[h1], A2=A[h2], B1=B[:,h1], B2=B[:,h2]),
            types={'cones':ctypes, 'HC':htypes}, h1_rows=h1, h2_rows=h2,
            h2_parameters=params, layout=dict(scope='native_only_existing_far_patch', counts=[len(c),len(h1),len(h2)]),
            connection_identity=identity, artifact_sha256=canonical_hash(identity))
        native = LateralHCCandidate(assembly, coupling_rate_per_second=50., S_efficacy_ratio=1.)
        native.boundary = dict(A_release=ga[h]@eq['release']-A@eq['release'][c],
            A_Vbar=ga[h]@eq['cone']['Vbar']-A@eq['cone']['Vbar'][c],
            B_h=gb[c]@eq['h']-B@eq['h'][h])
        values = [eq['cone'][n][c] for n in ('X','C','Vis','g_i','Vbar')]
        values += [eq['w'][np.searchsorted(global_h1,h[h1])]]*2
        values += [eq['h'][h], np.array([], dtype=float)]
        packed = np.concatenate(values)
        baseline = AnalyticREAnchor(0., light, light, light, .051, 0)
        packed, background_receipt = native.equilibrate_native(packed, baseline, 0.)
        if background_receipt['maximum_full_native_rhs'] > 1e-7:
            raise ContractError('New background is not stationary in the actual native RHS')
        initial[label] = packed.copy()
        stepped = light.copy(); stepped[spot] *= 1.1
        stimulus = AnalyticREAnchor(0., light, light, stepped, .051, 1)
        def integrate(rtol, atol, step):
            def rhs(t, y):
                guard()
                return native.rhs(y,t,stimulus)
            result = solve_ivp(rhs, (0., .05), packed, method='DOP853', rtol=rtol,
                               atol=atol, max_step=step, t_eval=observations)
            if not result.success or not np.isfinite(result.y).all():
                raise ContractError('Native probe failed: '+result.message)
            return result
        solution = integrate(1e-6, 1e-9, .0001)
        delta = solution.y[native.slices['Vh']].T-packed[native.slices['Vh']]
        responses[label] = delta
        receipts[label] = dict(native_source_version=native.source_version,
            global_joint_residual=eq['residual'], fresh_local_background=background_receipt,
            initial_state_sha256=canonical_hash(packed), local_A=matrix_id(A), local_B=matrix_id(B),
            local_A_nnz=A.nnz, local_B_nnz=B.nnz, zero_local_feedback_rows=int(np.sum(np.asarray(B.sum(1)).ravel()==0)),
            globally_unsupported_local_cone_groups=int(np.sum(np.asarray(gb[c].sum(1)).ravel()==0)),
            coupling_Q_sha256={k:matrix_id(v) for k,v in native.Q.items()},
            nfev=solution.nfev, max_abs_HC_delta=float(abs(delta).max()))
        if label == 'corrected':
            tight = integrate(1e-7, 1e-10, .00005)
            tight_delta = tight.y[native.slices['Vh']].T-packed[native.slices['Vh']]
            receipts[label]['tighter_reference_max_abs_HC_difference'] = float(abs(tight_delta-delta).max())
        print('Native probe complete:', label, flush=True)
    with np.load(ROOT/SOURCE_DIRECTORY/'native_response_probe.npz',allow_pickle=False) as saved:
        if not (np.array_equal(saved['time_s'],observations) and np.array_equal(saved['cone_parent_rows'],c)
                and np.array_equal(saved['HC_parent_rows'],h)):
            raise ContractError('Existing comparison has a different protocol/row identity')
        difference=float(abs(responses['corrected']-saved['eccentric_HC_delta']).max())
    numerical=receipts['corrected']['tighter_reference_max_abs_HC_difference']
    fraction=projection['cone_supported_member_fraction'][c]
    previous_report=json.loads((ROOT/'semi_reports/2026-11-02_HC/hc_response_validation.json').read_bytes())
    Q_unchanged=receipts['corrected']['coupling_Q_sha256']==previous_report['receipts']['eccentric']['coupling_Q_sha256']
    if not Q_unchanged:raise ContractError('HC homotypic coupling changed outside the projection correction')
    output = dict(schema='HC_fine_before_group_native_probe_v2',
        implementation={'path':Path(__file__).relative_to(ROOT).as_posix(), 'sha256':file_sha(__file__)},
        source_implementations={name:file_sha(ROOT/name) for name in
            ('retina_sim/h1_h2_effective.py','retina_sim/hc_lateral_candidate.py','retina_sim/vh2005.py')},
        selection={'path':(PREPARED/'local_static/selection_identity.json').relative_to(ROOT).as_posix(),
                   'sha256':file_sha(PREPARED/'local_static/selection_identity.json')},
        optical_background={'path':BACKGROUND.relative_to(ROOT).as_posix(), 'sha256':file_sha(BACKGROUND)},
        parent_artifact=full.artifact_sha256, new_connection_profile=operator['profile'],
        quantities=dict(HC_contact_footprint='literature-constrained direct cone-pedicle support',
                        HC_functional_RF='emergent direct input plus homotypic dynamics; not fitted in this probe'),
        center_deg=center.tolist(), center_eccentricity_deg=float(np.linalg.norm(center)),
        cone_groups=len(c), H1_groups=len(h1), H2_groups=len(h2),
        stimulus=dict(kind='spatial light step on the original patch core', radius_deg=spot_radius, contrast=.1, driven_cone_groups=int(spot.sum()), duration_s=.05),
        equations='existing LateralHCCandidate native Cone/H1/H2 equations',
        coupling_rate_per_second=50., S_efficacy_ratio=1., new_Gaussian_filter=False,
        boundary='newly solved global direct-input background held outside local Cone/HC subset; local homotypic graph sealed as before',
        background_recomputed_per_connection=True, old_state_reused=False, receipts=receipts,
        max_abs_difference_in_HC_delta=difference, response_difference_resolved_above_numerics=bool(difference>max(1e-8,10*numerical)),
        numerical=dict(method='DOP853', rtol=1e-6, atol=1e-9, max_step_s=.0001,
                       tighter_rtol=1e-7, tighter_atol=1e-10, tighter_max_step_s=.00005),
        native_probe_passed=True, physiological_validation=False, complete_RF_fit=False,
        downstream_initialized=False, old_video_overwritten=False,
        wall_seconds=time.perf_counter()-started, peak_rss_bytes=memory.peak_rss_bytes)
    output['member_support']=dict(partial_halo_groups=int(np.sum((fraction>0)&(fraction<1))),
        unsupported_halo_groups=int(np.sum(fraction==0)),core_partial=int(np.sum(spot&(fraction<1))),
        Q_unchanged=Q_unchanged,scope='Existing incomplete graph retained; unsupported halo is not biological zero; no physiological qualification')
    output['projection_implementation']=dict(path='retina_sim/hc_group_projection.py',sha256=file_sha(ROOT/'retina_sim/hc_group_projection.py'))
    np.savez_compressed(ROOT/DIRECTORY/'native_response_probe.npz',time_s=observations,
        corrected_HC_delta=responses['corrected'],corrected_initial=initial['corrected'],
        HC_types=htypes,HC_member_counts=hp.counts[h],cone_parent_rows=c,HC_parent_rows=h,
        cone_supported_member_fraction=fraction)
    np.savez_compressed(ROOT/DIRECTORY/'fresh_native_initial_state.npz',packed=initial['corrected'],
        cone_parent_rows=c,HC_parent_rows=h,background_light=light,
        A_release_boundary=native.boundary['A_release'],A_Vbar_boundary=native.boundary['A_Vbar'],B_h_boundary=native.boundary['B_h'])
    sparse.save_npz(ROOT/DIRECTORY/'group_cone_to_HC_A.npz',new_A)
    sparse.save_npz(ROOT/DIRECTORY/'group_HC_to_cone_typed_B.npz',new_B)
    output['response_data'] = {'path':(DIRECTORY/'native_response_probe.npz').as_posix(),
                               'sha256':file_sha(ROOT/DIRECTORY/'native_response_probe.npz')}
    output['group_native_operators'] = {name:{'path':(DIRECTORY/filename).as_posix(),
        'sha256':file_sha(ROOT/DIRECTORY/filename)} for name,filename in
        [('A','group_cone_to_HC_A.npz'),('typed_B','group_HC_to_cone_typed_B.npz')]}
    save_json(HANDOUT/'hc_response_validation.json', output)
    save_json(HANDOUT/'CANDIDATE_MODEL.json',dict(schema='HC_group_projection_candidate_v2',
        fine_source_model=original_model,group_native_operators=output['group_native_operators'],
        source=file_sha(ROOT/'retina_sim/hc_group_projection.py'),native_response=output['response_data'],
        fresh_initial_state_sha256=file_sha(HANDOUT/'fresh_native_initial_state.npz'),
        initializer_scope='Native only; not a complete downstream checkpoint',
        FIT_READY=False,background_qualified=False,production_adopted=False))
    journal.finish('PASS',phase='complete')
    print(json.dumps(output,indent=2),flush=True)


if __name__ == '__main__':
    try:main()
    except BaseException as exc:
        HANDOUT.mkdir(parents=True,exist_ok=True)
        save_json(HANDOUT/'FAILURE.json',dict(error=str(exc),traceback=traceback.format_exc(),no_auto_retry=True))
        raise
