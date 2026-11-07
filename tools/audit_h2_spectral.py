"""Read saved eccentric-HC responses and audit weights and actual graph ancestors.

No trajectory advancement, parameter fitting, optical computation or model edits.
"""
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import time
import traceback

import numpy as np
from scipy import sparse
from scipy.special import expit

from retina_sim.assembly import RetinaBuilder
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.hc_eccentric import digest, checked_path
from retina_sim.hc_group_projection import _normalize_complete_rows
from retina_sim.h1_h2_effective import matrix_id
from retina_sim.spatial_group_model import BC, RGC
from retina_sim.vh2005 import SOURCE_VALUES
from retina_platform.response_data import ancestors
from tools.prepare_hc_eccentric import write
from tools.run_hc_minimal_real_diagnostic_v1 import peak_memory_bytes


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/CONE_HC_FINAL_IMPLEMENTATION_2026-11-02'
CONFIG = 'configurations/research/cone_hc_eccentric.json'


def csv_write(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    values = np.asarray(values)
    return dict(count=len(values), sum=float(values.sum()), mean=float(values.mean()) if len(values) else None,
                min=float(values.min()) if len(values) else None,
                q25=float(np.quantile(values, .25)) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                q75=float(np.quantile(values, .75)) if len(values) else None,
                max=float(values.max()) if len(values) else None)


def weights(model):
    m = model.members
    source = ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/executions/assemble_native_01'
    receipt = json.loads((source/'NATIVE_ASSEMBLY_RESULT.json').read_bytes())
    cones = FrozenDiskTable(ROOT, receipt['cone_table'])
    types = cones.extras['subtype'].astype('U1')
    raw = sparse.load_npz(source/'RAW_HC_BY_CONE.npz').tocsr()
    uniform, _ = _normalize_complete_rows(raw)
    h2_fine = np.flatnonzero(m['hc_types']=='H2')
    full_fine = uniform[m['hc_parent_rows'][h2_fine]].tocsr()
    fine_dynamic = sparse.load_npz(OUT/'eccentric_assets/A_fine_dynamic.npz').tocsr()[h2_fine]
    np.testing.assert_allclose((full_fine[:, m['cone_parent_rows']]-fine_dynamic).data, 0., atol=2e-14, rtol=0)
    row_lookup = {int(g): i for i, g in enumerate(model.h2_rows)}
    hg = np.array([row_lookup[int(g)] for g in m['hc_group'][h2_fine]])
    mean_hc = sparse.csr_matrix((1/m['hc_counts'][m['hc_group'][h2_fine]],
                                (hg, np.arange(len(hg)))), shape=(len(model.h2_rows), len(hg)))
    complete = (mean_hc@full_fine).tocsr()
    dynamic = complete[:, m['cone_parent_rows']].tocsr()
    cone_projection = sparse.csr_matrix((np.ones(len(m['cone_group'])),
        (np.arange(len(m['cone_group'])), m['cone_group'])), shape=(len(m['cone_group']), model.A.shape[1]))
    A2 = model.A[model.h2_rows].tocsr()
    np.testing.assert_allclose((dynamic@cone_projection-A2).data, 0., rtol=0, atol=2e-14)
    np.testing.assert_allclose(np.asarray(complete.sum(1)).ravel(), 1., rtol=0, atol=2e-14)
    np.testing.assert_allclose(np.asarray(A2.sum(1)).ravel()+model.mass['A_group_held'][model.h2_rows], 1., rtol=0, atol=2e-14)
    group_types = model.assembly.types['cones']
    np.testing.assert_array_equal(group_types[m['cone_group']], m['cone_types'])
    records = []
    for kind, matrix, labels, indices in (
            ('runtime_H2_group', dynamic, m['cone_types'], model.h2_rows),
            ('complete_support_H2_group', complete, types, model.h2_rows),
            ('complete_support_fine_H2', full_fine, types, m['hc_parent_rows'][h2_fine])):
        for j, row in enumerate(indices):
            start, stop = matrix.indptr[j:j+2]
            cols, values = matrix.indices[start:stop], matrix.data[start:stop]
            record = dict(scope=kind, HC_row=int(row), represented_HC_members=int(m['hc_counts'][row]) if 'group' in kind else 1)
            for label in ('L', 'M', 'S', 'LM'):
                stats = distribution(values[np.isin(labels[cols], list(label))])
                record.update({label+'_'+key: value for key, value in stats.items()})
            record['row_mass'] = float(values.sum())
            records.append(record)
    csv_write(OUT/'H2_CONNECTION_STATISTICS.csv', records)
    coo = complete.tocoo()
    np.savez_compressed(OUT/'H2_CONNECTION_WEIGHTS.npz', H2_group_rows=model.h2_rows,
        edge_H2_group_row=model.h2_rows[coo.row], source_cone_parent_row=coo.col,
        source_cone_type=types[coo.col], effective_single_cone_weight=coo.data,
        dynamic_source=np.isin(coo.col, m['cone_parent_rows']),
        source_cone_keys=cones.keys[np.unique(coo.col)], source_key_parent_rows=np.unique(coo.col),
        fine_H2_parent_rows=m['hc_parent_rows'][h2_fine], fine_H2_runtime_group_rows=m['hc_group'][h2_fine])
    # Counts of unique connected fine Cone members, not group count or contact count.
    incidence = dynamic.copy(); incidence.data[:] = 1.
    connected_member_counts = (incidence@cone_projection).tocsr()
    raw_h2 = raw[m['hc_parent_rows'][h2_fine]].tocsr()
    uniform_per_fine_row = all(np.ptp(raw_h2.data[raw_h2.indptr[j]:raw_h2.indptr[j+1]])==0 for j in range(raw_h2.shape[0]))
    return A2, connected_member_counts, records, dict(
        runtime_H2_groups=len(model.h2_rows), represented_fine_H2=len(h2_fine),
        runtime_A2_identity=matrix_id(A2), complete_single_cone_operator_identity=matrix_id(complete),
        original_raw_contact_values_equal_within_each_H2=uniform_per_fine_row,
        single_cone_semantics='For a runtime H2 group, average the fine HC rows, then retain each source fine Cone separately; never divide a group edge by all group members',
        dynamic_vs_complete='Runtime A is the dynamic crop; remaining complete-support mass is held and has zero stimulus modulation')


def responses(model, A2, counts, prior):
    trace_path = OUT/'eccentric_acceptance/H1_H2_NATIVE_TRAJECTORIES.npz'
    z = np.load(trace_path, allow_pickle=False)
    ct = model.assembly.types['cones']
    n = model.native
    p = SOURCE_VALUES
    sample_time = z['S__time_seconds']
    saved = dict(time_seconds=sample_time, H2_group_rows=model.h2_rows,
                 cone_group_types=ct, cone_group_counts=model.members['cone_counts'])
    records = []; input_records = []; errors = {}
    representative = int(prior['H2_SPECTRAL_CONSTRAINT']['representative_H2_group'])
    rep = int(np.flatnonzero(model.h2_rows==representative).item())
    for label in ('S', 'L', 'M', 'LM'):
        times = z[label+'__time_seconds']; state = z[label+'__states']
        release = z[label+'__cone_release']; H2 = z[label+'__H2']
        np.testing.assert_array_equal(times, sample_time)
        np.testing.assert_array_equal(state[:, 0], model.y0)
        np.testing.assert_array_equal(H2, state[n.slices['Vh']][model.h2_rows])
        light = model.stimulus(model.input['onset_seconds'], label)
        expected = model.light*(1+model.input['contrast']*np.isin(ct, list(label)))
        np.testing.assert_array_equal(light, expected)
        np.testing.assert_array_equal(state[:, times<=model.input['onset_seconds']], z['S__states'][:, times<=model.input['onset_seconds']])
        bh = model.B@state[n.slices['Vh']]+n.boundary['B_h'][:, None]
        recomputed = expit((state[n.slices['Vis']]-bh-p['V_k'])/p['V_n'])/(state[n.slices['Vbar']]/p['V_I'])**p['mu']
        np.testing.assert_allclose(recomputed, release, rtol=3e-14, atol=1e-14)
        np.testing.assert_allclose(model.A@release+n.boundary['A_release'][:,None], z[label+'__A_release'], rtol=3e-14, atol=1e-14)
        dh = H2-H2[:, [0]]; dr = release-release[:, [0]]
        saved[label+'__H2_delta'] = dh
        saved[label+'__cone_release_delta'] = dr
        saved[label+'__light_on_td'] = light
        saved[label+'__initial_state'] = state[:, 0]
        contribution = {}; unweighted = {}
        for source_type in ('L', 'M', 'S', 'LM'):
            mask = np.isin(ct, list(source_type))
            contribution[source_type] = A2[:, mask]@dr[mask]
            unweighted[source_type] = counts[:, mask]@dr[mask]
            saved[label+'__weighted_release_'+source_type] = contribution[source_type]
            saved[label+'__sum_connected_fine_release_'+source_type] = unweighted[source_type]
        np.testing.assert_allclose(sum(contribution[k] for k in ('L','M','S')), A2@dr, rtol=3e-14, atol=1e-14)
        for g in range(len(ct)):
            input_records.append(dict(condition=label, cone_group=g, cone_type=str(ct[g]),
                fine_members=int(model.members['cone_counts'][g]), stimulated=bool(ct[g] in label),
                background_td=float(model.light[g]), on_td=float(light[g]), contrast=float(light[g]/model.light[g]-1),
                baseline_release=float(release[g,0]), min_release_delta=float(dr[g].min()),
                max_release_delta=float(dr[g].max()), peak_abs_release_delta=float(abs(dr[g]).max())))
        for j, row in enumerate(model.h2_rows):
            drive = contribution[label][j]
            total = unweighted[label][j]
            members = float(counts[j, np.isin(ct,list(label))].sum())
            peak = float(abs(dh[j]).max())
            weighted_peak = float(abs(drive).max())
            total_peak = float(abs(total).max())
            mean_peak = total_peak/members if members else 0.
            record = dict(condition=label, H2_group=int(row), connected_stimulated_fine_cones=int(members),
                stimulated_A_weight=float(A2[j,np.isin(ct,list(label))].sum()),
                H2_min_delta=float(dh[j].min()), H2_max_delta=float(dh[j].max()),
                H2_peak_abs=peak, H2_peak_time_seconds=float(times[np.argmax(abs(dh[j]))]),
                sum_release_peak_abs=total_peak, mean_release_peak_abs=mean_peak,
                weighted_release_peak_abs=weighted_peak,
                response_per_mean_release=peak/mean_peak if mean_peak>1e-14 else None,
                response_per_summed_release=peak/total_peak if total_peak>1e-14 else None,
                transfer_per_weighted_release=peak/weighted_peak if weighted_peak>1e-14 else None)
            for source_type in ('L','M','S'):
                value=contribution[source_type][j]
                record[source_type+'_weighted_release_min']=float(value.min())
                record[source_type+'_weighted_release_max']=float(value.max())
                record[source_type+'_weighted_release_peak_abs']=float(abs(value).max())
            records.append(record)
        errors[label]=dict(release_reconstruction=float(np.max(abs(recomputed-release))),
                           prior_H2_peak=abs(float(abs(dh[rep]).max())-prior['H2_SPECTRAL_CONSTRAINT']['responses'][label]['H2_peak_abs']))
        assert errors[label]['prior_H2_peak'] < 1e-14
    csv_write(OUT/'H2_ISOLATION_RESPONSE_METRICS.csv', records)
    csv_write(OUT/'H2_CONE_INPUT_MODULATION.csv', input_records)
    np.savez_compressed(OUT/'H2_ISOLATION_DECOMPOSITION.npz', **saved)
    reps={r['condition']:r for r in records if r['H2_group']==representative}
    ratios={}
    for other in ('L','M','LM'):
        ratios[other]={key: reps['S'][key]/reps[other][key] if reps['S'][key] is not None and reps[other][key] not in (None,0) else None
            for key in ('H2_peak_abs','stimulated_A_weight','mean_release_peak_abs','response_per_mean_release','response_per_summed_release','transfer_per_weighted_release')}
    return dict(representative_H2_group=representative, representative=reps, S_over_comparison=ratios,
        arithmetic_checks=errors, same_initial_state=True, same_time_grid=True, same_H2_identity=True,
        background_td=1000., on_contrast=.01, onset_seconds=.02, end_seconds=.12,
        saved_trajectory_samples=len(sample_time), additional_dynamic_runs=0,
        waveform_comparability='Comparable native-Td class-isolating diagnostics; equal per-Cone fractional input, not equal total weighted release',
        selectivity_metric='METRIC_NOT_COMPARABLE',
        reason='Previous peak(S)/peak(all LM) confounds convergence, effective weights, release responses and H2 coupling; it is not a per-Cone S-preference test',
        physical_stimulus_limit='No matched physical spectral radiance or cone-catch isolation calibration is present; no new conversion parameters introduced',
        peak_definition='max(abs(value(t)-value(0))) on common saved 0..120 ms grid; not a transfer-function fit',
        normalization_caution='response_per_summed_release describes the same connected fine-Cone pool; transfer_per_weighted_release divides out A and cannot demonstrate S selectivity. Feedback and homotypic coupling remain active.')


def directed_graph(model):
    a=model.assembly
    inner=a.runtime_factories['inner'](a, model.seed['inner_parameters'])
    incoming=defaultdict(list); adjacency=defaultdict(list); edge_records=[]; route_records=[]
    def add(name, pre, post, matrix, provenance):
        matrix=matrix.tocsr(copy=True); matrix.eliminate_zeros()
        assert np.isfinite(matrix.data).all()
        incoming[post].append((pre,matrix.indptr,matrix.indices))
        coo=matrix.tocoo()
        route_records.append(dict(name=name, pre=pre, post=post, edges=matrix.nnz,
            matrix_identity=matrix_id(matrix), provenance=provenance))
        for row,col,value in zip(coo.row,coo.col,coo.data):
            source=(pre,int(col)); target=(post,int(row))
            adjacency[source].append((target,name))
            edge_records.append(dict(route=name,pre=pre,pre_row=int(col),post=post,post_row=int(row),coefficient=float(value)))
    add('actual_A_release','cones','HC',model.A,'Bound current native A; H1 uses release and Vbar; H2 uses release')
    add('actual_B_feedback','HC','cones',model.B,'Current local feedback affects Cone release')
    q=-model.native.Qfull.copy();q.setdiag(0);q.eliminate_zeros()
    add('H2_homotypic','HC','HC',q*model.native.coupling_rate,'Current nonzero homotypic off-diagonal rate; no H1/H2 heterotypic edge')
    channels={c['pre']+'_to_'+c['pool']:c for c in inner.channels}
    for route in inner.routes:
        if route.kind!='inner_chemical': continue
        matrix=inner.mean[route.name]
        if route.pre=='cones' and route.post in BC:
            assert np.any(np.asarray(inner.p[route.post]['kernel'])!=0)
            assert np.all(inner.par(route.post,'native_voltage_gain',positive=True)>0)
        if route.name in channels:
            matrix=matrix*channels[route.name]['G_BA_S']
        if route.post in RGC:
            gain=np.broadcast_to(model.seed['inner_parameters']['route_gains'][route.name],(inner.counts[route.post],))
            matrix=sparse.diags(gain*inner.members[route.post])@matrix
        add(route.name,route.pre,route.post,matrix,'Runtime inner mean with actual AC input/RGC receptor gain; zero coefficients omitted')
    for component,c in inner.contacts.items():
        matrix=sparse.csr_matrix((c['coefficient'],(c['post'],c['pre'])),
            shape=(inner.counts[c['route'].post],inner.counts[c['route'].pre]))
        add(component,c['route'].pre,c['route'].post,matrix,'Actual contact coefficient S, including AII to OFF BC glycine')
    gap=a.matrix('gap/laplacian_S.npz')
    for pre,ps in inner.gap_slices.items():
        for post,qs in inner.gap_slices.items():
            matrix=-gap[qs,ps].copy()
            if pre==post: matrix.setdiag(0); matrix.eliminate_zeros()
            add('gap:'+pre+'->'+post,pre,post,matrix,'Actual off-diagonal gap Laplacian; held boundary has no dynamic ancestor edge')
    def paths(starts):
        prior={s:None for s in starts};queue=deque(starts)
        while queue:
            node=queue.popleft()
            for target,route in adjacency[node]:
                if target not in prior:prior[target]=(node,route);queue.append(target)
        return prior
    h2_paths=paths([('HC',int(i)) for i in model.h2_rows])
    s_paths=paths([('cones',int(i)) for i in np.flatnonzero(a.types['cones']=='S')])
    son_paths=paths([('S_ON_BC',i) for i in range(a.populations['S_ON_BC'].group_count)])
    def witness(target):
        if target not in h2_paths:return []
        result=[];node=target
        while h2_paths[node] is not None:
            previous,route=h2_paths[node]
            result.append(dict(pre=list(previous),post=list(node),route=route));node=previous
        return list(reversed(result))
    records=[];proofs={}
    for population in RGC:
        for row,gid in enumerate(a.populations[population].group_ids):
            target=(population,row)
            seen,_=ancestors(a.populations,incoming,target)
            h2=np.intersect1d(seen['HC'],model.h2_rows)
            assert bool(len(h2)) == (target in h2_paths)
            proof=witness(target);proofs[str(gid)]=proof
            records.append(dict(population=population,local_row=row,group_id=str(gid),
                parent_row=int(a.parent_population_rows[population][row]),
                H2_directed_path=bool(len(h2)), H2_ancestor_count=len(h2),
                H1_ancestor_count=len(np.intersect1d(seen['HC'],model.h1_rows)),
                S_cone_ancestor_count=int(np.sum(a.types['cones'][seen['cones']]=='S')),
                LM_cone_ancestor_count=int(np.sum(a.types['cones'][seen['cones']]!='S')),
                S_ON_BC_ancestor_count=len(seen['S_ON_BC']), S_path=(target in s_paths),
                S_ON_BC_path=(target in son_paths),
                H2_gate='H2_RELEVANT_UNRESOLVED' if len(h2) else 'NOT_BLOCKED_BY_H2_ALONE',
                all_fit_prerequisites_met=False,
                non_H2_limits='Current 1000 Td diagnostic is not the registered movie input/background; central spatial and held-boundary qualification absent; protocol-specific target binding/cache checks still required'))
    csv_write(OUT/'H2_RGC_GRAPH_DEPENDENCIES.csv',records)
    csv_write(OUT/'H2_DIRECTED_GRAPH_EDGES.csv',edge_records)
    summary={pop:dict(groups=sum(r['population']==pop for r in records),
        H2_dependent=sum(r['population']==pop and r['H2_directed_path'] for r in records),
        S_dependent=sum(r['population']==pop and r['S_path'] for r in records),
        S_ON_BC_dependent=sum(r['population']==pop and r['S_ON_BC_path'] for r in records)) for pop in RGC}
    return records,dict(model_identity=model.identity,routes=route_records,
        graph_identity=canonical_hash(edge_records), edge_count=len(edge_records),summary=summary,
        path_witnesses=proofs, traversal_crosscheck='Forward multi-source BFS agrees with independent existing backward ancestors traversal for every current RGC',
        scope='Exact nonzero structural influence in the current dynamic graph; does not assert response magnitude or physiological truth; held constants are not dynamic ancestors')


def target_readiness(model, dependencies):
    registry_path=ROOT/'reports/RGC_DATASET_TARGET_MATRIX_V1/TARGET_MATRIX.json'
    index_path=ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/RESULT_INDEX.json'
    binding_path=ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/joint_gray_background/TARGET_INDEX_REPAIR.json'
    registry=json.loads(registry_path.read_bytes())
    index=json.loads(index_path.read_bytes())
    binding=json.loads(binding_path.read_bytes())
    bycell={r['cell']:r for r in index['tasks']}
    byid={r['group_id']:r for r in dependencies}
    rows=[]
    for entry in registry['rows']:
        cell=entry['cell_id'];gid=entry.get('neural_owner')
        is_gin=entry['dataset']=='Schottdorf_Lee2021_GIN_xage77'
        if is_gin and cell=='69#6':gid=binding['actual_group_id']
        dep=byid.get(gid)
        blockers=[]
        if dep is None:blockers.append('TARGET_NOT_BOUND_IN_CURRENT_GRAPH')
        if not entry.get('actual_recording_available'):blockers.append('RAW_RECORD_UNAVAILABLE')
        if is_gin and cell!='69#6':blockers.append('TARGET_SPECIFIC_INPUT_AND_MODEL_REGISTRATION_UNRESOLVED')
        if is_gin and cell=='69#6':
            assert dep is not None and dep['parent_row']==binding['target_parent_row'] and dep['local_row']==binding['local_row']
            blockers.extend(['CURRENT_ECCENTRIC_1000TD_DIAGNOSTIC_DIFFERS_FROM_REGISTERED_MOVIE_MODEL',
                             'NO_COMPLETE_REGISTERED_TRAIN_DEV_CACHE_FOR_THIS_MODEL_IDENTITY'])
        if not is_gin:
            blockers.extend(['HISTORICAL_PROXY_MODEL_IDENTITY_NOT_TRANSFERABLE','ENGINEERING_NATIVE_INPUT_AND_ECCENTRICITY_PROXY'])
            if cell=='Cell_6':blockers.append('PRIOR_FIXED_DEVELOPMENT_INCONCLUSIVE_G5_GATE_NOT_PASSED')
        blockers.extend(['CENTRAL_SPATIAL_PHYSIOLOGY_UNQUALIFIED','HELD_BOUNDARY_SPATIAL_CONVERGENCE_UNQUALIFIED'])
        status=bycell.get(cell,{}).get('status',entry['fit_status']) if is_gin else entry['fit_status']
        rows.append(dict(dataset=entry['dataset'],target=cell,model_type=entry['mapped_model_type'],
            registered_group_id=gid,current_group_found=dep is not None,
            H2_directed_path=dep['H2_directed_path'] if dep is not None else 'UNKNOWN_UNBOUND_TARGET',
            depends_on_unresolved_S_H2=dep['H2_directed_path'] if dep is not None else 'UNKNOWN_UNBOUND_TARGET',
            H2_gate=dep['H2_gate'] if dep is not None else 'CANNOT_TRANSFER_TYPE_LEVEL_REACHABILITY',
            recorded_status_snapshot=status,non_H2_blockers=';'.join(blockers),
            all_existing_protocol_prerequisites_met=False,global_FIT_READY_changed=False))
    csv_write(OUT/'H2_REGISTERED_TARGET_READINESS.csv',rows)
    return dict(rows=rows, registry_sha256=digest(registry_path),index_snapshot_sha256=digest(index_path),
        target_binding_sha256=digest(binding_path),
        note='Only exact IDs can inherit current-graph reachability; unbound targets are unknown, never classified H2-free from cell-type names. No experiment target or global FIT_READY was modified.')


def main():
    output=OUT/'H2_SPECTRAL_AUDIT.json'
    if output.exists():raise FileExistsError('Existing completed audit retained')
    start=time.perf_counter()
    result=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),dynamic_runs=0,optimizer_calls=0)
    paths=[ROOT/CONFIG,OUT/'eccentric_acceptance/RESULT.json',OUT/'eccentric_acceptance/H1_H2_NATIVE_TRAJECTORIES.npz',
           OUT/'CALIBRATION_METRICS.json',ROOT/'configurations/research/BC_PHYSIOLOGY_2026-11-02.json',
           ROOT/'retina_sim/h1_h2_effective.py',ROOT/'retina_sim/hc_lateral_candidate.py',
           ROOT/'retina_sim/cone_hc_eccentric.py',ROOT/'retina_sim/group_inner_dynamics.py',
           ROOT/'retina_sim/ac_vs_model.py',ROOT/'retina_sim/vh2005.py',ROOT/'tools/accept_hc_eccentric.py']
    paths+=list((OUT/'eccentric_assets').glob('*'))
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths if p.is_file()}
    try:
        prior=json.loads((OUT/'eccentric_acceptance/RESULT.json').read_bytes())
        model=RetinaBuilder.build_research_cone_hc(ROOT,CONFIG)
        assert model.identity==prior['model_identity']
        assert matrix_id(model.A)==prior['matrices']['A_identity'] and matrix_id(model.B)==prior['matrices']['B_identity']
        assert model.calibration==prior['calibration']
        print('Verified current Builder identity; no trajectory run',flush=True)
        A2,counts,stats,weight_meta=weights(model)
        result['weights']=weight_meta
        rep=prior['H2_SPECTRAL_CONSTRAINT']['representative_H2_group']
        result['representative_weights']=[r for r in stats if r['scope'].endswith('H2_group') and r['HC_row']==rep]
        result['comparison']=responses(model,A2,counts,prior)
        print('Decomposed saved S/L/M/LM input and H2 trajectories',flush=True)
        dependencies,graph=directed_graph(model)
        result['directed_graph']=graph
        result['registered_targets']=target_readiness(model,dependencies)
        for name,expected in hashes.items():checked_path(ROOT,name,expected)
        result.update(status='COMPLETE',conclusion='H2_PREVIOUS_METRIC_INVALID',
            previous_selectivity_failure_withdrawn=True,physiological_validation_passed=False,
            remaining_H2_limit='No demonstrated per-Cone S advantage or calibrated physical isolation; type-neutral original H2 gain/time dynamics retained. Weight-normalized transfer is not S selectivity.',
            source_files=hashes,model_identity=model.identity,H1_calibration_unchanged=model.calibration,
            acceptance=dict(current_matrix_identity=True,full_and_dynamic_weight_sum_projection=True,
                matched_saved_background_time_identity=True,release_and_response_reconstructed=True,
                directed_graph_bidirectional_traversal_agrees=True,protected_files_unchanged=True),
            global_FIT_READY_changed=False,parameter_changes=False,
            H2_input_variable='dimensionless source Cone release = sigmoid((Vis-Bh-Vk)/Vn)/(Vbar/VI)^mu; not membrane voltage directly',
            H2_equation='dh2/dt=(151.1*(A2@release+held_release)-h2)/0.020 - coupling_rate*Q2@h2',
            H2_extra_type_gain=False,H2_coupling_rate_per_second=model.native.coupling_rate,
            group_types_pure=True,additional_isolation_test_needed=False)
    except Exception as error:
        result.update(status='FAILED_RETAINED',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        result.update(ended_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-start,
                      peak_working_set_bytes=peak_memory_bytes(),implementation_sha256=digest(Path(__file__)))
        write(output,result)
    print(json.dumps({k:result[k] for k in ('status','conclusion','wall_seconds','peak_working_set_bytes')}),flush=True)


if __name__=='__main__':main()
