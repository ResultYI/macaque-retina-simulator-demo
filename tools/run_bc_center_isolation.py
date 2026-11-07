"""One center-isolated surround supplement; reuse all original control trajectories."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import os
import json
import time
import traceback

import numpy as np

from tools import run_bc_center_surround as original
from retina_sim.assembly import RetinaBuilder
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.h1_h2_effective import matrix_id
from retina_sim.hc_eccentric import digest
from retina_sim.parameter_system import data as json_data

ROOT, OUT = original.ROOT, original.OUT
LABEL = 'annulus_center_isolated'
METRICS = OUT/'CENTER_ISOLATION_METRICS.json'
STATES = OUT/'BC_ANNULUS_CENTER_ISOLATED_STATES.npz'
FIGURE = OUT/'BC_CENTER_ISOLATION_COMPARISON.png'
IDENTITY = 'b3b8d74ad9656620fac19bdf1da699c9beec5c88307efc8aa59e3cf406fd1c10'


def isolate_mask(xy, center, radius, groups, counts, candidate):
    protected = np.linalg.norm(xy-center, axis=1) <= radius
    protected_groups = np.unique(groups[protected])
    mask = candidate & ~np.isin(groups, protected_groups)
    fraction = np.bincount(groups, weights=mask.astype(float), minlength=len(counts))/counts
    return protected, protected_groups, mask, fraction


def require(condition, message):
    if not condition:
        raise ContractError(message)


def result_identity(arrays, model_identity, protocol):
    content={name:dict(dtype=value.dtype.str,shape=value.shape,
                      sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
             for name,value in arrays.items() if name!='metadata_json'}
    return canonical_hash(dict(model=model_identity,condition=LABEL,arrays=content,protocol=protocol))


def save_states(path, arrays, metrics):
    payload=dict(arrays,metadata_json=np.array(json.dumps(json_data(metrics),ensure_ascii=False)))
    if not isinstance(path,(str,Path)):
        np.savez_compressed(path,**payload)
        return
    path=Path(path)
    pending=path.with_suffix(path.suffix+'.writing')
    with pending.open('xb') as stream:
        np.savez_compressed(stream,**payload)
        stream.flush()
        os.fsync(stream.fileno())
    with np.load(pending,allow_pickle=False) as saved:
        require(set(saved.files)==set(payload),'Saved trajectory keys differ')
        for key,value in payload.items():
            np.testing.assert_array_equal(saved[key],value,err_msg='Disk roundtrip: '+key)
    os.replace(pending,path)


def static_preflight(model, old, saved):
    members = model.upstream.members
    for key, member_key in [('cone_source_rows','cone_parent_rows'), ('cone_group_rows','cone_group'),
                            ('cone_group_counts','cone_counts'), ('cone_IS_mm','cone_IS_mm'),
                            ('cone_pedicle_mm','cone_pedicle_mm'), ('cone_global_ids','cone_keys'),
                            ('HC_fine_source_rows','hc_parent_rows'), ('HC_fine_group_rows','hc_group'),
                            ('HC_fine_anchor_mm','hc_anchor_mm')]:
        np.testing.assert_array_equal(saved[key], members[member_key], err_msg=key)
    np.testing.assert_array_equal(saved['HC_group_ids'], model.assembly.populations['HC'].group_ids)
    A, B = model.upstream.native.A, model.upstream.native.B
    require({k:matrix_id(getattr(model.upstream.native,k)) for k in ('A','B')} == old['operator_hashes'],
            'Frozen A/B operators changed')
    require(canonical_hash(model.seed['inner_parameters']) == old['model_parameter_hash'], 'Parameters changed')
    xy, groups, counts = saved['cone_IS_mm'], saved['cone_group_rows'], saved['cone_group_counts']
    geom = old['geometry']; center = np.asarray(geom['center_IS_mm'])
    radius = geom['annulus_inner_radius_mm']
    protected, protected_groups, mask, fraction = isolate_mask(
        xy, center, radius, groups, counts, saved['annulus__fine_mask'])
    source = FrozenDiskTable(ROOT, model.assembly.populations['cones'].source_table)
    full_xy = source.coordinates['rf']; parent = saved['cone_source_rows']
    np.testing.assert_array_equal(full_xy[parent], xy)
    np.testing.assert_array_equal(source.coordinates['opl_pedicle'][parent], saved['cone_pedicle_mm'])
    full_distances = np.linalg.norm(full_xy-center, axis=1)
    candidate_full = (full_distances >= radius) & (full_distances < geom['annulus_outer_radius_mm'])
    dynamic = np.zeros(len(full_xy), dtype=bool); dynamic[parent] = True
    require(not np.any(candidate_full & ~dynamic), 'Original annulus includes held-only source')
    require(not np.any((full_distances <= radius) & ~dynamic), 'Protection disk has unsaved members')
    np.testing.assert_array_equal(candidate_full[parent], saved['annulus__fine_mask'])
    np.testing.assert_array_equal(np.bincount(groups, minlength=len(counts)), counts)
    seed, protocol = original.prepared_inputs(model, fraction)
    light0, light1 = seed['native_inputs'][0], seed['native_inputs'][1]
    require(np.all(light0 == 1000.), 'Background differs from frozen 1000 native-Td')
    actual_delta = light1-light0
    require(np.all(fraction[protected_groups] == 0.) and np.all(actual_delta[groups[protected]] == 0.),
            'Center protection failed after Runner input projection')
    shared = np.flatnonzero((saved['center__group_fraction'] > 0) & (actual_delta != 0))
    require(not len(shared), 'New surround and original center share nonzero groups')
    require(np.any(actual_delta != 0), 'NOT_SUPPORTED: no surviving surround groups')
    require(protocol['reference'] == old['conditions']['annulus']['solver'], 'Solver changed')
    paths = {}
    for name, cell in geom['cells'].items():
        meta = model.assembly.bc_cells[name]; row = cell['row']
        require(str(meta['global_cell_ids'][row]) == cell['global_id'], 'Target identity changed')
        take = meta['input_post_rows'] == row
        np.testing.assert_array_equal(meta['input_cone_source_rows'][take], cell['direct_cone_source_rows'])
        np.testing.assert_array_equal(meta['input_weights'][take], cell['weights'])
        direct = np.asarray(cell['direct_cone_group_rows'])
        if cell['mandatory']:
            require(np.all(actual_delta[direct] == 0), 'Target direct input not isolated')
        W = model.assembly.matrix('routes/cones_to_'+name+'/mean_weights.npz')[row]
        feedback = (W@B).tocsr()
        expected_h = np.asarray(cell['HC_dependencies']['group_rows'])
        actual_order, expected_order = np.argsort(feedback.indices), np.argsort(expected_h)
        np.testing.assert_array_equal(feedback.indices[actual_order], expected_h[expected_order])
        np.testing.assert_array_equal(feedback.data[actual_order],
                                     np.asarray(cell['HC_dependencies']['BC_weighted_B_coefficients'])[expected_order])
        drive = np.asarray(A@fraction).ravel()
        h = feedback.indices; mass = feedback.data * drive[h]
        paths[name] = dict(HC_group_rows=h, HC_group_ids=saved['HC_group_ids'][h],
            HC_types=saved['HC_types'][h], BC_weighted_B_coefficients=feedback.data,
            illuminated_A_fraction=drive[h], path_mass_by_HC_group=mass,
            total_path_mass=float(mass.sum()), represented_B_mass=float(feedback.sum()),
            direct_input_delta_native_Td=actual_delta[direct],
            by_type={kind:dict(HC_rows=h[(saved['HC_types'][h] == kind) & (mass > 0)],
                              path_mass=float(mass[saved['HC_types'][h] == kind].sum())) for kind in ('H1','H2')},
            structure_qualified=bool(mass.sum() > 0 and abs(float(feedback.sum())-1) < 1e-12))
    def coverage(take):
        d = xy[take]-center
        angles = np.mod(np.arctan2(d[:,1], d[:,0]), 2*np.pi)
        return dict(member_count=int(take.sum()), radial_range_um=(1000*np.linalg.norm(d,axis=1)[[np.argmin(np.linalg.norm(d,axis=1)),np.argmax(np.linalg.norm(d,axis=1))]]).tolist(),
                    angular_30_degree_bin_counts=np.histogram(angles,bins=np.linspace(0,2*np.pi,13))[0])
    audit = dict(CENTER_INPUT_ISOLATION='PASS', protection_radius_mm=radius,
        protected_cone_count=int(protected.sum()), protected_group_count=len(protected_groups),
        excluded_originally_illuminated_group_count=int(np.sum(saved['annulus__group_fraction'][protected_groups] > 0)),
        original_cone_count=int(saved['annulus__fine_mask'].sum()), isolated_cone_count=int(mask.sum()),
        original_nonzero_group_count=int(np.count_nonzero(saved['annulus__group_fraction'])),
        isolated_nonzero_group_count=int(np.count_nonzero(actual_delta)),
        removed_cone_count=int(saved['annulus__fine_mask'].sum()-mask.sum()),
        center_contaminated_member_count=int(np.count_nonzero(actual_delta[groups[protected]])),
        maximum_center_input_delta_native_Td=float(np.max(abs(actual_delta[groups[protected]]),initial=0)),
        shared_nonzero_center_groups=shared, physical_selected_coverage=coverage(mask),
        projected_member_coverage=coverage(fraction[groups] > 0),
        integrated_member_fraction=float(np.dot(counts, fraction)),
        dynamic_support='PASS_ALL_STIMULATED_MEMBERS_AND_GROUPS_DYNAMIC', held_only_stimulated_cones=0,
        boundary=geom['boundary'], HC_paths=paths,
        geometry='Center-isolated surround with missing annular members; not a continuous pure annulus',
        coverage_note='Discrete IS positions and projected member coverage, not calibrated retinal receptive-field area; no renormalization')
    arrays = {k:v for k,v in saved.items() if k.startswith(('cone_','HC_'))}
    arrays.update({LABEL+'__fine_mask':mask, LABEL+'__group_fraction':fraction,
                   'protection__fine_mask':protected, 'protection__group_rows':protected_groups,
                   'runner_native_input_epochs_Td':np.stack([seed['native_inputs'][i] for i in range(3)]),
                   'runner_epoch_start_seconds':np.array([0.,original.ONSET,original.OFFSET])})
    return audit, arrays, fraction


def initial_preflight(model, old, saved, fraction):
    seed, protocol = original.prepared_inputs(model, fraction)
    arm = original.SpatialStepRunner(model.assembly, seed, protocol)
    try:
        first = arm.snapshot(0., 'right', queue_node=True)
        state_hash = canonical_hash(dict(states=first['states'], spikes=arm.spike_states))
        require(all(v['initial_state_and_RNG_hash'] == state_hash for v in old['conditions'].values()),
                'Initial complete state/RGC RNG hash incompatible with old experiment')
        values = original.sample(first, old['geometry']['cells'])
        for label in original.CONDITIONS:
            for key, value in values.items():
                np.testing.assert_array_equal(value, saved[label+'__'+key][...,0], err_msg=label+' '+key)
        return dict(status='PASS_EXACT', initial_state_and_RNG_hash=state_hash,
                    compared_sample_fields=len(values), conditions_compared=list(original.CONDITIONS),
                    method='One Builder background reconstruction; no saved complete t=0 checkpoint. All state/RGC RNG hash and all saved t=0 readouts match exactly. No time integration during preflight.')
    finally:
        arm.close()


def analyze(new, old, saved, audit):
    times = saved['background__time_seconds']; result = {}
    np.testing.assert_array_equal(new[LABEL+'__time_seconds'], times)
    window = (np.rint((times-original.ONSET)*1000) >= 20) & (np.rint((times-original.ONSET)*1000) <= 100)
    def delta(key):
        value, control = new[LABEL+'__'+key], saved['background__'+key]
        return (value-value[...,0,None])-(control-control[...,0,None])
    def summary(v):
        return dict(maximum_absolute_change=float(np.max(abs(v),initial=0)),
                    mean_20_100ms=float(np.mean(v[...,window])),
                    maximum_absolute_fixed_window_mean=float(np.max(abs(np.mean(v[...,window],axis=-1)),initial=0)))
    for name, cell in old['geometry']['cells'].items():
        previous = old['responses'][name]; threshold = previous['resolution_threshold_V']
        stats = original.response_statistics(new[LABEL+'__'+name+'__V'],times,
                                            saved['background__'+name+'__V'], threshold)
        center = previous['center']['mean_20_100ms_V']; surround = stats['mean_20_100ms_V']
        resolved = abs(center) > threshold and abs(surround) > threshold
        q = -surround/center if resolved else None
        qualified = audit['HC_paths'][name]['structure_qualified'] and resolved
        support = bool(q is not None and q > 0 and previous['center_polarity_matches'])
        status = ('支持' if support else '不支持') if qualified else '不可判读'
        if not cell['mandatory']:
            status = '探索性记录，不参与四类资格'
        direct = np.unique(cell['direct_cone_group_rows']); h = np.asarray(cell['HC_dependencies']['group_rows'])
        propagation = {'target_direct_light_delta_native_Td':audit['HC_paths'][name]['direct_input_delta_native_Td']}
        for field in ('cones__Bh','cones__release','cones__Vis'):
            propagation[field] = dict(group_rows=direct, **summary(delta(field)[direct]))
        for kind in ('H1','H2'):
            rows = h[saved['HC_types'][h] == kind]
            propagation[kind] = dict(group_rows=rows, present=bool(len(rows)))
            if len(rows):
                propagation[kind].update(Vh=summary(delta('HC__Vh')[rows]), pooled_aI=summary(delta('HC__pooled_aI')[rows]))
        for field in ('drive','generator_activation','V','terminal_release'):
            propagation['BC__'+field] = summary(delta(name+'__'+field))
        result[name] = dict(global_id=cell['global_id'], mandatory=cell['mandatory'],
            center=previous['center'], original_annulus=previous['annulus'], isolated=stats,
            resolution_threshold_V=threshold, original_Q=previous['Q_fixed_window'], Q_isolated=q,
            Q_status='defined' if resolved else 'undefined_unresolved_response',
            isolated_minus_original_mean_V=surround-previous['annulus']['mean_20_100ms_V'],
            CENTER_SURROUND_PHENOTYPE=status, propagation=propagation)
    return result


def plot(new, saved, old):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(15,8.5))
    times = (saved['background__time_seconds']-original.ONSET)*1000
    for ax, name in zip(axes.flat, original.TYPES):
        control = saved['background__'+name+'__V']; drift = control-control[0]
        for label, source, color in [('center',saved,'#2456a6'),('annulus',saved,'#d88931'),(LABEL,new,'#09846a')]:
            v = source[label+'__'+name+'__V']
            ax.plot(times,(v-v[0]-drift)*1e6,label=label,color=color)
        ax.axvspan(0,150,color='gray',alpha=.07); ax.axvspan(20,100,color='green',alpha=.06)
        ax.axhline(0,color='black',lw=.5); ax.grid(alpha=.15)
        ax.set(xlabel='Time from onset (ms)',ylabel='Delta BC voltage (uV)',
               title=name+(' (exploratory)' if name=='S_ON_BC' else ''))
    axes[0,0].legend(fontsize=7)
    ax=axes[1,2]; xy=saved['cone_IS_mm']; center=np.asarray(old['geometry']['center_IS_mm'])
    xy=(xy-center)*1000; keep=new[LABEL+'__fine_mask']; removed=saved['annulus__fine_mask'] & ~keep
    active=new[LABEL+'__group_fraction'][saved['cone_group_rows']] > 0
    ax.scatter(xy[active,0],xy[active,1],s=16,facecolors='none',edgecolors='#888888',linewidths=.5,label='projected active members')
    ax.scatter(xy[removed,0],xy[removed,1],s=7,color='#d88931',label='removed from annulus')
    ax.scatter(xy[keep,0],xy[keep,1],s=6,color='#09846a',label='isolated selected members')
    ax.add_patch(plt.Circle((0,0),old['geometry']['annulus_inner_radius_mm']*1000,fill=False,color='#2456a6',ls='--'))
    ax.set_aspect('equal'); ax.set(xlabel='IS x from center (um)',ylabel='IS y from center (um)',title='Actual member mask and group projection')
    ax.legend(fontsize=6,loc='upper left')
    fig.suptitle('Frozen finite BC model | center-isolated surround supplement\nFixed 5% step; fixed 20-100 ms window; qualitative phenotype only',fontsize=12)
    fig.tight_layout(); fig.savefig(FIGURE,dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-preflight',action='store_true')
    parser.add_argument('--authorized-single-rerun',action='store_true')
    args=parser.parse_args()
    require(not STATES.exists() and not FIGURE.exists(), 'Supplement output already exists; no repeated simulation')
    prior=None
    if args.authorized_single_rerun:
        require(not args.resume_preflight,'Choose one explicit recovery mode')
        prior=json.loads(METRICS.read_bytes())
        require(prior['status']=='POST_INTEGRATION_PERSISTENCE_FAILED_TRAJECTORY_UNRECOVERABLE'
                and prior['integration_attempt_count']==1,'Additional attempt already consumed or incompatible prior record')
        for name,h in prior['delivery_verification'].items():
            require(digest(OUT/name)==h,'Previously verified artifact changed: '+name)
    elif args.resume_preflight:
        prior=json.loads(METRICS.read_bytes())
        require(prior['status']=='FAILED_RETAINED_NO_AUTOMATIC_RERUN' and 'integration_started_utc' not in prior,
                'Only a failed pre-integration check can be resumed')
    else:
        with METRICS.open('x',encoding='utf8') as f:
            f.write('{}\n')
    start=time.perf_counter(); guard=original.MemoryGuard()
    metrics=dict(status='PREPARING',started_utc=datetime.now(timezone.utc).isoformat(),condition=LABEL,
                 FIT_READY=False,production_qualified=False,model_frozen=True)
    if args.authorized_single_rerun:
        metrics['previous_failed_attempt']=prior
        metrics['previous_metrics_sha256']=digest(METRICS)
        metrics['authorization']='User explicitly authorized one additional annulus_center_isolated run after persistence verification; no model or analysis changes.'
        metrics['integration_attempt_count']=2
        metrics['additional_authorized_attempt_count']=1
        metrics['builder_construction_count_this_attempt']=1
    elif prior:
        metrics['prior_preflight_failure']={k:prior[k] for k in ('started_utc','ended_utc','wall_seconds',
            'peak_working_set_bytes','source_code_sha256','error','traceback') if k in prior}
        metrics['prior_preflight_failure']['integration_started']=False
        metrics['builder_construction_count_including_failed_preflight']=2
        metrics['preflight_repair']='Compare sparse HC edges by row identity, independent of CSR storage order; no model/operator change'
    arrays={}
    try:
        old=json.loads((OUT/'PHYSIOLOGY_METRICS.json').read_bytes())
        with np.load(OUT/'BC_CENTER_SURROUND_STATES.npz',allow_pickle=False) as z:
            saved={k:z[k] for k in z.files}
        frozen=dict(old['protected_files_before'])
        for name in ('PHYSIOLOGY_METRICS.json','BC_CENTER_SURROUND_STATES.npz','BC_CENTER_SURROUND_RESPONSES.png'):
            frozen[str((OUT/name).relative_to(ROOT))]=digest(OUT/name)
        if args.authorized_single_rerun:
            for name in ('CENTER_ISOLATION_INPUTS.npz','CENTER_ISOLATION_INPUT_MASK.png'):
                frozen[str((OUT/name).relative_to(ROOT))]=prior['delivery_verification'][name]
        require(digest(Path(original.__file__)) == old['analysis_code_sha256'], 'Original experiment helper changed')
        frozen[str(Path(original.__file__).relative_to(ROOT))]=digest(Path(original.__file__))
        for name,h in frozen.items():
            require(digest(ROOT/name)==h,'Protected source changed: '+name)
        metrics['protected_files_before']=frozen
        require(guard.check(),str(guard.stop_reason))
        model=RetinaBuilder.build_research_bc(ROOT,'configurations/research/BC_macaque_fullchain.json')
        require(model.identity==old['model_identity']==IDENTITY,'Scientific model identity changed')
        metrics.update(model_identity=model.identity,configuration='configurations/research/BC_macaque_fullchain.json',
                       protocol=old['protocol'],cells=old['geometry']['cells'],operator_hashes=old['operator_hashes'],
                       model_parameter_hash=old['model_parameter_hash'],source_code_sha256=digest(Path(__file__)))
        audit,arrays,fraction=static_preflight(model,old,saved)
        if args.authorized_single_rerun:
            with np.load(OUT/'CENTER_ISOLATION_INPUTS.npz',allow_pickle=False) as z:
                verified_inputs={k:z[k] for k in z.files if k!='metadata_json'}
            require(result_identity(verified_inputs,model.identity,old['protocol'])==prior['static_input_identity'],
                    'Previously verified input identity changed')
            require(set(arrays)==set(verified_inputs),'Input record keys changed')
            for key,value in arrays.items():
                np.testing.assert_array_equal(value,verified_inputs[key],err_msg='Frozen input: '+key)
            require(canonical_hash(old['geometry']['cells'])==canonical_hash(prior['cells']),'Selected target records changed')
            arrays=verified_inputs
            fraction=arrays[LABEL+'__group_fraction']
            metrics['static_input_identity']=prior['static_input_identity']
            metrics['verified_input_reuse']='Loaded prior CENTER_ISOLATION_INPUTS.npz byte-for-byte; independently rechecked against Builder; no reselection or new mask'
        metrics['static_qualification']=audit
        metrics['initial_state_comparability']=initial_preflight(model,old,saved,fraction)
        metrics['status']='STATIC_AND_INITIAL_STATE_QUALIFIED_BEFORE_INTEGRATION'
        metrics['preparation_seconds']=time.perf_counter()-start
        original.write(METRICS,metrics)
        print('Preflight PASS: exact initial states/RNG and readouts; isolated members/groups',audit['isolated_cone_count'],audit['isolated_nonzero_group_count'],flush=True)
        print('HC paths',json.dumps({n:v['total_path_mass'] for n,v in audit['HC_paths'].items()}),flush=True)
        metrics['integration_started_utc']=datetime.now(timezone.utc).isoformat()
        original.write(METRICS,metrics)
        new,status=original.run_condition(model,fraction,old['geometry']['cells'],LABEL,guard)
        arrays.update(new); metrics['condition_result']=status
        metrics['status']='INTEGRATION_COMPLETED_ANALYSIS_PENDING'
        metrics.update(wall_seconds_before_persistence=time.perf_counter()-start,
                       peak_working_set_bytes=original.peak_memory_bytes())
        save_states(STATES,arrays,metrics)
        metrics['trajectory_saved_before_analysis']=dict(sha256=digest(STATES),
            saved_utc=datetime.now(timezone.utc).isoformat(),roundtrip_all_arrays_exact=True)
        original.write(METRICS,metrics)
        print('Trajectory durable and verified on disk before analysis:',STATES.name,flush=True)
        require(status['initial_state_and_RNG_hash']==metrics['initial_state_comparability']['initial_state_and_RNG_hash'], 'Runner initial hash changed')
        metrics['responses']=analyze(arrays,old,saved,audit)
        metrics['CENTER_INPUT_ISOLATION']=audit['CENTER_INPUT_ISOLATION']
        metrics['PHYSIOLOGICAL_EVIDENCE']='Finite frozen Cone-HC-BC model at approximately 3.7 degrees under actual input and held exterior boundary; per-cell qualitative phenotype only. No RF-size/gain calibration, HC ablation, spatial convergence, temporal refit, or central 0-6 degree validation.'
        metrics['original_experiment_qualification']=old['input_projection_audit']['strict_pure_annulus_qualification']
        metrics['area_confound']='Surround lost illuminated members; Q changes cannot be attributed solely to decontamination.'
        metrics['reused_assets']=['Original background, center and annulus trajectories','Frozen model parameters and connectivity','Original geometry and cell IDs','Original solver, clock, threshold and analysis windows']
        metrics['protected_files_unchanged']=all(digest(ROOT/name)==h for name,h in frozen.items())
        require(metrics['protected_files_unchanged'],'Protected files changed during supplement')
        metrics['status']='ONE_ISOLATED_SURROUND_COMPLETED'
    except Exception as error:
        metrics.update(status='FAILED_RETAINED_NO_AUTOMATIC_RERUN',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        metrics.update(wall_seconds=time.perf_counter()-start,peak_working_set_bytes=original.peak_memory_bytes(),
                       ended_utc=datetime.now(timezone.utc).isoformat())
        original.write(METRICS,metrics)
        if LABEL+'__time_seconds' in arrays:
            metrics['result_identity']=result_identity(arrays,metrics['model_identity'],metrics['protocol'])
            if not STATES.exists():
                save_states(STATES,arrays,metrics)
        original.write(METRICS,metrics)
    plot(arrays,saved,old)
    print(metrics['status'], metrics['wall_seconds'],flush=True)
    for name,row in metrics['responses'].items():
        print(name,row['isolated']['mean_20_100ms_V'],row['Q_isolated'],row['CENTER_SURROUND_PHENOTYPE'],flush=True)


if __name__=='__main__':
    main()
