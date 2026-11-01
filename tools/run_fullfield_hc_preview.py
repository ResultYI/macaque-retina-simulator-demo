"""Full finite Cone-HC network driven by the complete existing first exposure."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.atlas import global_cell_id
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
from retina_sim.contracts import canonical_hash
from retina_sim.disk_graph_storage import file_sha
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.h1_h2_effective import H1H2Native, csr, joint_equilibrium, matrix_id
from retina_sim.hc_lateral_candidate import LateralHCCandidate, homotypic_operator
from retina_sim.spatial_group_model import _verify_artifact
from retina_sim.vh2005 import SOURCE_VALUES
from tools.durable_research_runtime_v1 import DurableRun, atomic_json, inspect_run, utc

OUTPUT = ROOT / 'reports/HC_FULLFIELD_FIRST_FRAME_0_6DEG_20261101_v1'
VIDEO = ROOT / 'reports/cone_video_113_4s_eccentric_aperture_0_6deg_20261029_v1'
CANDIDATE = ROOT / 'reports/retinotopic_rf_revision/native_validation_v1/CANDIDATE_MODEL.json'
SOURCE_FILES = ('retina_sim/vh2005.py', 'retina_sim/h1_h2_effective.py',
                'retina_sim/hc_lateral_candidate.py', 'retina_sim/hc_group_projection.py',
                'retina_sim/cone_hc_reduced_native.py', 'retina_sim/group_native_physics.py')


def member_mean(captured_values, group_rows, counts, exterior=100.):
    """Keep uncaptured members at their declared reference; never renormalize a crop."""
    return exterior + np.bincount(group_rows, weights=np.asarray(captured_values)-exterior,
                                 minlength=len(counts)) / counts


def make_assembly(A, B, cone_counts, hc_counts, cone_types, hc_types, identity):
    def population(counts, name):
        return SimpleNamespace(group_count=len(counts), counts=counts,
            group_ids=np.array([f'{name}:{i}' for i in range(len(counts))]),
            recording_auxiliary_group_rows=np.array([], dtype=int),
            recording_auxiliary_fine_rows=np.array([], dtype=int))
    h1, h2 = (np.flatnonzero(hc_types == kind) for kind in ('H1', 'H2'))
    matrices = {'routes/cones_to_HC/mean_weights.npz': A,
                'routes/HC_to_cones/mean_weights.npz': B}
    return SimpleNamespace(populations={'cones': population(cone_counts, 'cone'),
        'HC': population(hc_counts, 'HC')}, matrix=lambda name: matrices[name],
        operators=dict(A1=A[h1], A2=A[h2], B1=B[:, h1], B2=B[:, h2]),
        types={'cones': cone_types, 'HC': hc_types}, h1_rows=h1, h2_rows=h2,
        h2_parameters=dict(tau_seconds=.020, input_scale=SOURCE_VALUES['g_t'], feedback_scale=1.),
        layout=dict(scope='complete_finite_active_Cone_HC_network', cone_groups=A.shape[1],
                    HC_groups=A.shape[0]), connection_identity=identity,
        artifact_sha256=canonical_hash(identity))


class SparseFullFieldNative(LateralHCCandidate):
    """Existing native equations with sparse block assembly for a large complete graph."""
    def __init__(self, assembly):
        H1H2Native.__init__(self, assembly)
        self.coupling_rate, self.S_efficacy_ratio = 50., 1.
        self.Q, self.overlap, self.topology_normalizer = {}, {}, {}
        rows, cols, values = [], [], []
        for kind, indices in (('H1', self.h1_rows), ('H2', self.h2_rows)):
            q, overlap, normalizer = homotypic_operator(self.A[indices],
                assembly.populations['HC'].counts[indices])
            self.Q[kind], self.overlap[kind], self.topology_normalizer[kind] = q, overlap, normalizer
            coo = q.tocoo()
            rows.append(indices[coo.row]); cols.append(indices[coo.col]); values.append(coo.data)
        # Sparse assembly avoids an enormous rectangular selector assignment.
        self.Qfull = csr(sparse.coo_matrix((np.concatenate(values),
            (np.concatenate(rows), np.concatenate(cols))), shape=(self.nh, self.nh)))
        self.coupling_degree = self.Qfull.diagonal()
        self.coupling_neighbor = csr(sparse.diags(self.coupling_degree)-self.Qfull)
        self.identity.update(full_field_adapter=dict(coupling_rate_per_second=50.,
            S_efficacy_ratio=1., Q={k: matrix_id(v) for k, v in self.Q.items()},
            implementation_sha256=file_sha(__file__), equations='inherited LateralHCCandidate'))
        self.source_version = canonical_hash(self.identity)


def run(resume_preflight=False):
    runtime_name = 'runtime'
    if OUTPUT.exists():
        if not resume_preflight:
            raise FileExistsError('Existing run retained; inspect ownership and terminal receipt before continuation')
        failure = json.loads((OUTPUT/'FAILURE.json').read_bytes())
        previous_runtimes = sorted(p for p in OUTPUT.glob('runtime*') if p.is_dir())
        ownership = [inspect_run(path) for path in previous_runtimes]
        assert failure['phase'] == 'preflight' and not (OUTPUT/'CONTRACT.json').exists()
        assert all(o['decision'] == 'TERMINAL_RECORDED' for o in ownership)
        assert not any(p['state'] in ('ALIVE','UNKNOWN') for o in ownership for p in o['observations'])
        attempt = len(previous_runtimes)+1
        (OUTPUT/f'PREFLIGHT_FAILURE_BEFORE_{attempt}.json').write_bytes((OUTPUT/'FAILURE.json').read_bytes())
        atomic_json(OUTPUT/f'PREFLIGHT_RECOVERY_{attempt}.json',dict(ownership=ownership,
            reason='Verify source producer ID serialization and retain an explicit canonical-ID crosswalk',
            no_neural_simulation_started=True, no_scientific_contract_changed=True))
        runtime_name = f'runtime_preflight_recovery_{attempt}'
    else:
        OUTPUT.mkdir()
    journal = DurableRun(OUTPUT / runtime_name, 'complete_first_frame_Cone_HC_network')
    memory = MemoryGuard(24*1024**3, 8*1024**3)
    started, checked = time.perf_counter(), -1e9
    phase = 'preflight'
    def guard(force=False):
        nonlocal checked
        now = time.perf_counter()
        if now-started > 1800:
            raise TimeoutError('Full-field preview exceeded its 1800 s bound')
        if force or now-checked > .5:
            if not memory.check():
                raise MemoryError(str((memory.stop_reason, memory.last)))
            journal.heartbeat(phase=phase, memory=memory.last)
            checked = now
    try:
        guard(True)
        frozen_sources = {name: file_sha(ROOT/name) for name in SOURCE_FILES}
        candidate = json.loads(CANDIDATE.read_bytes())
        parent, manifest, parent_id = _verify_artifact(
            ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2')
        def array(name):
            return np.load(parent/name, allow_pickle=False, mmap_mode='r')
        tables = {name: FrozenDiskTable(ROOT, binding)
                  for name, binding in candidate['fine_source_model']['tables'].items()}
        populations, types = {}, {}
        for name in ('cones', 'HC'):
            populations[name] = {key: array(f'populations/{name}/{key}.npy')
                for key in ('counts', 'centroid_mm', 'fine_to_group', 'group_ids')}
            meta = manifest['populations'][name]
            keys = array(f'populations/{name}/keys.npy')
            types[name] = np.asarray(meta['subtype_labels'])[keys[:, meta['key_columns'].index('subtype')]]
            assert meta['source_table'] == candidate['fine_source_model']['tables'][name]
        bindings = candidate['group_native_operators']
        for binding in bindings.values():
            assert file_sha(ROOT/binding['path']) == binding['sha256']
        complete_A = sparse.load_npz(ROOT/bindings['A']['path']).tocsr()
        complete_B = sparse.load_npz(ROOT/bindings['typed_B']['path']).tocsr()
        active = np.asarray(complete_A.sum(1)).ravel() > 0
        assert complete_B[:, ~active].nnz == 0
        active_rows = np.flatnonzero(active)
        A, B = complete_A[active].tocsr(), complete_B[:, active].tocsr()
        assert A[types['HC'][active] == 'H1'][:, types['cones'] == 'S'].nnz == 0
        assert B[types['cones'] != 'S'][:, types['HC'][active] == 'H2'].nnz == 0
        assert B[types['cones'] == 'S'][:, types['HC'][active] == 'H1'].nnz == 0
        capture_path = VIDEO/'first_frame/capture.mat'
        capture = loadmat(capture_path)
        capture_meta = json.loads(capture['manifest_json'].item())
        completion = json.loads((VIDEO/'first_frame/capture_complete.json').read_bytes())
        assert capture_meta['request_signature'] == completion['request_signature']
        with np.load(VIDEO/'geometry.npz', allow_pickle=False) as g:
            source_rows, source_types = g['source_table_rows'], g['types']
            assert len(source_rows) == len(np.unique(source_rows)) == 190498
            assert np.array_equal(g['xy_mm'], tables['cones'].coordinates['rf'][source_rows])
            assert np.array_equal(source_types, tables['cones'].extras['subtype'][source_rows].astype('U1'))
            metadata = tables['cones'].metadata
            identities = [global_cell_id(metadata['atlas_id'], metadata['atlas_version'],
                metadata['source_version'], metadata['population'], bytes(tables['cones'].keys[row]).decode('ascii'))
                for row in source_rows]
            producer_ids = [global_cell_id(metadata['atlas_id'],metadata['atlas_version'],
                metadata['source_version'],metadata['population'],str(tables['cones'].keys[row]))
                for row in source_rows]
            assert np.array_equal(g['cell_ids'],producer_ids)
            geometry_record=json.loads((VIDEO/'geometry.json').read_bytes())
            assert geometry_record['frozen_table_manifest_sha256']==candidate['fine_source_model']['tables']['cones']['sha256']
            np.savez_compressed(OUTPUT/'OPTICAL_CELL_ID_CROSSWALK.npz',source_table_rows=source_rows,
                optical_ids=g['cell_ids'],canonical_ids=np.asarray(identities))
        group_rows = populations['cones']['fine_to_group'][source_rows]
        assert np.all(group_rows >= 0)
        assert np.array_equal(types['cones'][group_rows], source_types)
        bridge_path = ROOT/'reports/external_video_two_positions_4s_20261029_v1/far/BINDING.json'
        bridge = json.loads(bridge_path.read_bytes())['conversion']
        alpha = np.array([bridge['alpha_rstar_s_per_reference_td_LMS'][t] for t in source_types])
        duration = float(np.diff(capture['window_edges_seconds'].ravel())[0])
        assert duration == 1/30 and capture['expected_counts'].shape == (1,190498)
        counts = populations['cones']['counts']
        background = member_mean(capture['background_counts'].ravel()/duration/alpha, group_rows, counts)
        light = member_mean(capture['expected_counts'].ravel()/duration/alpha, group_rows, counts)
        assert np.isfinite(background).all() and np.all(background > 0) and np.all(light >= 0)
        identity = dict(parent_artifact=parent_id, candidate_sha256=file_sha(CANDIDATE),
            complete_A=matrix_id(complete_A), complete_typed_B=matrix_id(complete_B),
            active_HC_parent_rows=active_rows.tolist(), capture_sha256=file_sha(capture_path),
            bridge_sha256=file_sha(bridge_path), source_hashes=frozen_sources,
            optical_identity_crosswalk=dict(path='OPTICAL_CELL_ID_CROSSWALK.npz',
                sha256=file_sha(OUTPUT/'OPTICAL_CELL_ID_CROSSWALK.npz'),
                rule='same frozen source-table row, exact RF position and subtype; producer IDs hash str(ASCII bytes)'))
        assembly = make_assembly(A, B, counts, populations['HC']['counts'][active],
                                 types['cones'], types['HC'][active], identity)
        assembly.populations['cones'].group_ids = populations['cones']['group_ids']
        assembly.populations['HC'].group_ids = populations['HC']['group_ids'][active]
        native = SparseFullFieldNative(assembly)
        native.boundary = dict(A_release=np.zeros(native.nh), A_Vbar=np.zeros(native.nh), B_h=np.zeros(native.nc))
        xyz = tables['HC'].coordinates['soma']
        owner = np.linalg.norm(xyz,axis=1) <= 6*.223
        fine_groups = populations['HC']['fine_to_group'][owner]
        inverse = np.full(len(active), -1, dtype=int); inverse[active] = np.arange(active.sum())
        supported = fine_groups >= 0
        owner_groups = np.full(len(fine_groups), -1, dtype=int)
        owner_groups[supported] = inverse[fine_groups[supported]]
        supported = owner_groups >= 0
        assert owner.sum() == 103053 and (~supported).sum() == 29
        contract = dict(schema='full_finite_Cone_HC_first_exposure_v1', started_utc=utc(),
            implementation_sha256=file_sha(__file__), identity=identity,
            geometry=dict(cone_groups=native.nc, active_HC_groups=native.nh,
                full_fine_cones=tables['cones'].count, full_fine_HC=tables['HC'].count,
                visually_driven_cones=len(source_rows), displayed_HC=int(owner.sum()),
                unsupported_displayed_HC=int((~supported).sum())),
            stimulus=dict(duration_seconds=duration, source_frame_index=0, image_radius_degrees=6,
                exterior_light_reference_equivalent_Td=100., captured_background='per-cell frozen gray catches / fixed LMS alpha',
                group_input='All-member mean once; uncaptured members retain 100 Td reference'),
            dynamics='All finite active groups evolve jointly, including exterior groups; no held neural boundary',
            display='One HC map; existing types retained internally; fine coordinates inherit their existing representative state',
            unsupported_policy='29 disconnected owner HC explicitly masked, no invented partners or pooled adaptation',
            numerical=dict(method='DOP853',standard=dict(rtol=1e-6,atol=1e-9,max_step_s=1e-4),
                tight=dict(rtol=1e-7,atol=1e-10,max_step_s=5e-5),
                background_max_abs_rhs=1e-7, max_HC_comparison_difference=1e-6,
                all_state_comparison_budget='1e-9 + 1e-6*max(abs(standard),abs(tight)); max ratio <=1'),
            physiological_validation=False,FIT_READY=False,production_adopted=False)
        atomic_json(OUTPUT/'CONTRACT.json',contract)
        np.savez_compressed(OUTPUT/'GEOMETRY_AND_INPUT.npz', HC_xy_deg=xyz[owner]/.223,
            HC_types=tables['HC'].extras['subtype'][owner], HC_fine_rows=np.flatnonzero(owner),
            HC_group_active_rows=owner_groups, supported=supported, active_HC_parent_rows=active_rows,
            background=background, light=light)
        print(json.dumps(contract['geometry']), flush=True)
        print('Solving new complete-network background', flush=True)
        phase = 'global_direct_background'; guard(True)
        # shortcut: equal gains and unit feedback make the H2 CSC rescaling an identity.
        assert assembly.h2_parameters == dict(tau_seconds=.020,input_scale=SOURCE_VALUES['g_t'],feedback_scale=1.)
        eq = joint_equilibrium(A,B,native.h1_rows,np.array([],dtype=int), background,
                               assembly.h2_parameters,guard=guard)
        packed = np.concatenate([eq['cone'][name] for name in ('X','C','Vis','g_i','Vbar')]
                                + [eq['w'],eq['z2'],eq['h'],np.array([],dtype=float)])
        baseline = AnalyticREAnchor(0.,background,background,background,duration,0)
        phase = 'complete_network_lateral_background'; guard(True)
        packed, bg_receipt = native.equilibrate_native(packed,baseline,0.)
        guard(True)
        assert bg_receipt['maximum_full_native_rhs'] <= 1e-7
        np.savez_compressed(OUTPUT/'BACKGROUND.npz', y=packed, R=background,E=background)
        atomic_json(OUTPUT/'BACKGROUND.json',dict(bg_receipt, direct_residual=eq['residual'],
            initial_state_sha256=canonical_hash(packed),native_source_version=native.source_version,
            Q_sha256={k:matrix_id(v) for k,v in native.Q.items()}))
        print('Background ready; maximum RHS',bg_receipt['maximum_full_native_rhs'],flush=True)
        del eq
        stimulus = AnalyticREAnchor(0.,background,background,light,duration,1)
        observations = np.r_[np.arange(0,.033,.001),duration]
        results, receipts = {}, {}
        for label,rtol,atol,step in [('standard',1e-6,1e-9,1e-4),('tight',1e-7,1e-10,5e-5)]:
            phase = label; begin=time.perf_counter(); guard(True)
            print('Integrating complete network:',label,flush=True)
            def rhs(t,y):
                guard(); return native.rhs(y,t,stimulus)
            solution=solve_ivp(rhs,(0.,duration),packed,method='DOP853',rtol=rtol,atol=atol,
                               max_step=step,t_eval=observations)
            assert solution.success and np.array_equal(solution.t,observations) and np.isfinite(solution.y).all()
            results[label]=solution.y
            np.savez_compressed(OUTPUT/(label.upper()+'.npz'),time_seconds=solution.t,
                y=solution.y,HC=solution.y[native.slices['Vh']],
                HC_delta=solution.y[native.slices['Vh']]-packed[native.slices['Vh'],None])
            receipts[label]=dict(nfev=solution.nfev,wall_seconds=time.perf_counter()-begin,
                endpoint_state_sha256=canonical_hash(solution.y[:,-1]),response_sha256=file_sha(OUTPUT/(label.upper()+'.npz')))
            atomic_json(OUTPUT/(label.upper()+'_RECEIPT.json'),receipts[label])
            print(label,'finished:',receipts[label],flush=True)
        difference=np.abs(results['standard']-results['tight'])
        scale=1e-9+1e-6*np.maximum(np.abs(results['standard']),np.abs(results['tight']))
        ratio=float(np.max(difference/scale))
        hc_error=float(np.max(difference[native.slices['Vh']]))
        accepted=ratio<=1. and hc_error<=1e-6
        delta=results['standard'][native.slices['Vh'],-1]-packed[native.slices['Vh']]
        owner_delta=np.full(len(owner_groups),np.nan);owner_delta[supported]=delta[owner_groups[supported]]
        np.savez_compressed(OUTPUT/'HC_PREVIEW.npz',xy_deg=xyz[owner]/.223,delta_h=owner_delta,
            supported=supported,HC_types=tables['HC'].extras['subtype'][owner],
            group_active_rows=owner_groups,duration_seconds=duration,source_frame_index=0)
        assert all(file_sha(ROOT/name)==value for name,value in frozen_sources.items())
        result=dict(status='COMPLETE_VALIDATED' if accepted else 'NUMERICAL_GATE_FAILED',
            accepted=accepted,started_utc=contract['started_utc'],ended_utc=utc(),
            wall_seconds=time.perf_counter()-started,peak_rss_bytes=memory.peak_rss_bytes,
            geometry=contract['geometry'],receipts=receipts,maximum_all_state_budget_ratio=ratio,
            maximum_HC_standard_tight_difference=hc_error,max_abs_display_HC_delta=float(np.nanmax(abs(owner_delta))),
            no_subsampling=True,old_results_unchanged=True,source_hashes_unchanged=True,
            complete_4s_movie=False,physiological_validation=False,FIT_READY=False)
        atomic_json(OUTPUT/'RESULT.json',result)
        journal.finish(result['status'],result='RESULT.json',accepted=accepted)
        if not accepted: raise RuntimeError('The fixed numerical comparison gate failed')
        print(json.dumps(result),flush=True)
    except BaseException as error:
        atomic_json(OUTPUT/'FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),
            phase=phase,elapsed_seconds=time.perf_counter()-started,memory=memory.last,automatic_retry=False))
        if not (OUTPUT/runtime_name/'terminal.json').exists():journal.finish('FAILED',error=repr(error),phase=phase)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-preflight',action='store_true')
    run(parser.parse_args().resume_preflight)
