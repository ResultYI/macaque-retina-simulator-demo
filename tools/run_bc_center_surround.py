"""One preregistered local spot/annulus experiment on the frozen independent BC model."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree

from retina_sim.assembly import RetinaBuilder
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.bc_independent import DEFAULT_CONFIGURATION
from retina_sim.cone_hc_eccentric import EccentricHCFullChainRunner
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.group_inner_dynamics import BC_MECH
from retina_sim.hc_eccentric import digest
from retina_sim.h1_h2_effective import matrix_id
from retina_sim.parameter_system import data as json_data
from tools.validate_h1_h2_effective import tick
from tools.run_hc_minimal_real_diagnostic_v1 import peak_memory_bytes

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'reports/BC_PHYSIOLOGY_IMPLEMENTATION_2026-11-02'
OUT = PARENT/'center_surround'
TYPES = ('midget_BC_ON','midget_BC_OFF','diffuse_BC_ON','diffuse_BC_OFF','S_ON_BC')
CONDITIONS = ('background','center','annulus')
ONSET, OFFSET, STOP = .01, .16, .26
CONTRAST = .05
SOURCE = 'https://www.cns.nyu.edu/~tony/vns/readings/dacey-etal-2000.pdf'


def write(path, value):
    path.write_text(json.dumps(json_data(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


class SpatialStepRunner(EccentricHCFullChainRunner):
    """Only input epoch lookup changes: 150 ms is not an integer 150-Hz frame count."""
    def frame_at(self,t,side='right'):
        if not np.isfinite(t) or side not in ('left','right') or not 0 <= t <= STOP:
            raise ContractError('Spatial step requires a finite protocol time and exact side')
        return int(np.searchsorted([ONSET,OFFSET],t,side=side))

    def frame_end(self,index):
        if index not in (0,1,2):raise ContractError('Only three explicit step epochs exist')
        return (ONSET,OFFSET,STOP+.01)[index]


def geometry(model):
    assembly=model.assembly;members=model.upstream.members
    cone=FrozenDiskTable(ROOT,assembly.populations['cones'].source_table)
    full_is=cone.coordinates['rf']
    parent_rows=members['cone_parent_rows'];groups=members['cone_group']
    np.testing.assert_array_equal(full_is[parent_rows],members['cone_IS_mm'])
    np.testing.assert_array_equal(cone.coordinates['opl_pedicle'][parent_rows],members['cone_pedicle_mm'])
    # Selection uses geometry only, before any response is computed.
    median=np.median(members['cone_IS_mm'],axis=0)
    meta=assembly.bc_cells['midget_BC_ON']
    anchor=int(np.argmin(np.linalg.norm(meta['functional_ancestry_mm']-median,axis=1)))
    center=meta['functional_ancestry_mm'][anchor].copy()
    selected={};direct=[];direct_groups=[];arrays={}
    lookup={int(row):int(group) for row,group in zip(parent_rows,groups)}
    for n in TYPES:
        meta=assembly.bc_cells[n]
        row=int(np.argmin(np.linalg.norm(meta['functional_ancestry_mm']-center,axis=1)))
        take=meta['input_post_rows']==row
        sources=meta['input_cone_source_rows'][take];weights=meta['input_weights'][take]
        cg=np.array([lookup[int(c)] for c in sources],dtype=np.int64)
        if n!='S_ON_BC':direct.extend(sources);direct_groups.extend(cg)
        selected[n]=dict(row=row,global_id=str(meta['global_cell_ids'][row]),
            eccentricity_deg=float(meta['eccentricity_deg'][row]),
            RF_mm=meta['functional_ancestry_mm'][row],OPL_anchor_mm=meta['input_anchor_OPL_mm'][row],
            original_group_row=int(meta['original_group_rows'][row]),
            direct_cone_ids=cone.rows(sources).global_cell_ids,direct_cone_source_rows=sources,
            direct_cone_IS_mm=full_is[sources],direct_cone_pedicle_mm=cone.coordinates['opl_pedicle'][sources],
            weights=weights,count=len(sources),direct_cone_group_rows=cg,
            mandatory=n!='S_ON_BC')
    direct=np.unique(direct);direct_groups=np.unique(direct_groups)
    group_extent=np.isin(groups,direct_groups)
    radius_group=float(np.max(np.linalg.norm(members['cone_IS_mm'][group_extent]-center,axis=1)))
    spacing=float(np.median(cKDTree(members['cone_IS_mm']).query(full_is[direct],k=2)[0][:,1]))
    excluded=np.ones(len(full_is),dtype=bool);excluded[parent_rows]=False
    distances=np.linalg.norm(full_is-center,axis=1)
    clearance=float(np.min(distances[excluded]))
    rc,ri,ro=radius_group+spacing,radius_group+2*spacing,.8*clearance
    if not 0<rc<ri<ro:raise ContractError('No complete geometric annulus fits the unchanged dynamic source')
    masks={'background':np.zeros(len(full_is),dtype=bool),'center':distances<rc,
           'annulus':(distances>=ri)&(distances<ro)}
    if np.any(masks['center']&masks['annulus']):raise ContractError('Spatial masks overlap')
    fractions={};stimuli={}
    A,B=model.upstream.native.A,model.upstream.native.B
    for label,mask in masks.items():
        if np.any(mask&excluded):raise ContractError('Stimulus contains held-only Cone input')
        fraction=np.bincount(groups,weights=mask[parent_rows].astype(float),minlength=A.shape[1])/members['cone_counts']
        fractions[label]=fraction
        hit=np.flatnonzero(mask)
        stimuli[label]=dict(fine_cone_count=len(hit),nonzero_groups=int(np.count_nonzero(fraction)),
            mixed_groups=int(np.sum((fraction>0)&(fraction<1))),missing_dynamic_cones=0,
            direct_group_fraction=fraction[direct_groups],true_cone_ids=cone.rows(hit).global_cell_ids,
            true_cone_source_rows=hit,
            projected_extra_member_count=int(np.sum((fraction[groups]>0)&~mask[parent_rows])))
        arrays[label+'__fine_mask']=mask[parent_rows]
        arrays[label+'__group_fraction']=fraction
    if not np.all(fractions['center'][direct_groups]==1) or np.any(fractions['annulus'][direct_groups]):
        raise ContractError('Grouped stimulus leaks onto the direct target or misses its center')
    htypes=np.full(A.shape[0],'H1');htypes[model.upstream.h2_rows]='H2'
    held=1-np.asarray(A.sum(1)).ravel()
    for n,row in selected.items():
        W=assembly.matrix('routes/cones_to_'+n+'/mean_weights.npz')[row['row']]
        feedback=(W@B).tocsr(); h=feedback.indices
        row['HC_dependencies']=dict(group_rows=h,group_ids=assembly.populations['HC'].group_ids[h],
            types=htypes[h],BC_weighted_B_coefficients=feedback.data,
            A_held_fraction=held[h],B_supported_mass=float(feedback.sum()),
            annulus_path_mass=float(np.asarray(feedback@A@fractions['annulus']).item()),
            center_path_mass=float(np.asarray(feedback@A@fractions['center']).item()),
            weighted_held_A_mass=float(np.asarray(feedback@held).item()),
            grouped_upstream=True)
        if row['mandatory'] and (abs(float(feedback.sum())-1)>1e-12 or row['HC_dependencies']['annulus_path_mass']<=0):
            raise ContractError('Mandatory target lacks represented Cone-HC-Cone annular feedback')
        row['center_direct_weighted_fraction']=float(np.asarray(W@fractions['center']).item())
        row['annulus_direct_weighted_fraction']=float(np.asarray(W@fractions['annulus']).item())
    arrays.update(cone_source_rows=parent_rows,cone_group_rows=groups,cone_group_counts=members['cone_counts'],
        cone_global_ids=members['cone_keys'],cone_IS_mm=members['cone_IS_mm'],
        cone_pedicle_mm=members['cone_pedicle_mm'],HC_group_ids=assembly.populations['HC'].group_ids,
        HC_types=htypes,HC_fine_source_rows=members['hc_parent_rows'],HC_fine_group_rows=members['hc_group'],
        HC_fine_anchor_mm=members['hc_anchor_mm'])
    description=dict(center_IS_mm=center,center_radius_mm=rc,annulus_inner_radius_mm=ri,annulus_outer_radius_mm=ro,
        nearest_excluded_cone_distance_mm=clearance,median_direct_cone_spacing_mm=spacing,
        direct_group_max_radius_mm=radius_group,selection='nearest ON midget to median dynamic IS; nearest other BC types to that anchor',
        size_rule='spot=all mandatory direct-group members plus one median Cone spacing; annulus inner=plus two spacings; outer=0.8*nearest excluded Cone distance',
        fixed_before_response=True,coordinate_mapping='Existing frozen RF coordinate equals saved Cone IS; OPL used only for synaptic distances',
        stimulus_projection='Original member arithmetic mean; no group-centroid mask or per-condition renormalization',
        boundary='All masked source Cones dynamic; all direct target feedback represented. Long-range unilluminated exterior remains held; no recursive whole-retina convergence claim',
        cells=selected,stimuli=stimuli)
    return description,fractions,arrays


def prepared_inputs(model,fraction):
    seed=deepcopy(model.seed);protocol=deepcopy(model.protocol)
    light=np.asarray(seed['input_reference']['native_background_td'])
    protocol.update(stop_seconds=STOP,input_frame_indices=[0,1,2],checkpoint_seconds=STOP,
        spatial_step_epochs=[0.,ONSET,OFFSET,STOP],input_schedule='explicit_relative_Td_step_not_movie_frames')
    rod=np.asarray(model.seed['rod_inputs'][min(model.seed['rod_inputs'])]).copy()
    seed['native_inputs']={0:light.copy(),1:light*(1+CONTRAST*fraction),2:light.copy()}
    seed['rod_inputs']={i:rod.copy() for i in range(3)}
    return seed,protocol


def sample(snapshot,cells):
    result={}
    for n,row in cells.items():
        i=row['row']
        for field in ('history_0','history_1','history_2','activation','V','terminal_release','A'):
            result[n+'__'+field]=float(snapshot['states'][n,BC_MECH,field][i])
        for field in ('drive','generator_activation','unadapted_release','effective_release'):
            result[n+'__'+field]=float(snapshot['ports'][n,field][i])
    for pop,field in (('cones','release'),('cones','Vis'),('cones','Bh'),('cones','R'),('cones','E'),
                       ('HC','Vh'),('HC','pooled_aI'),('AII_AC','voltage')):
        result[pop+'__'+field]=snapshot['ports'][pop,field].copy()
    return result


def run_condition(model,fraction,cells,label,guard):
    seed,protocol=prepared_inputs(model,fraction);start=time.perf_counter()
    arm=SpatialStepRunner(model.assembly,seed,protocol)
    rows=[];times=[]
    try:
        first=arm.snapshot(0.,'right',queue_node=True)
        rows.append(sample(first,cells));times.append(0.)
        initial_hash=canonical_hash(dict(states=first['states'],spikes=arm.spike_states))
        arm.initial_commit();arm.ack(0.)
        for i in range(1,2601):
            _,s=tick(arm,i*.0001,i)
            arm.inner.validate(arm.inner_state)
            if i%10==0:
                values=sample(s,cells)
                if any(not np.isfinite(v).all() for v in values.values()):raise ContractError('Nonfinite experiment readout')
                rows.append(values);times.append(i*.0001)
            if i%250==0:
                if not guard.check():raise ContractError('Memory guard: '+str(guard.stop_reason))
                if time.perf_counter()-start>900:raise ContractError('900 second local condition protection budget exceeded')
                print(f'{label}: {i/10:.0f} ms, {time.perf_counter()-start:.1f} s wall',flush=True)
        result={label+'__'+k:np.stack([r[k] for r in rows],axis=-1) for k in rows[0]}
        result[label+'__time_seconds']=np.array(times)
        status=dict(status='COMPLETED',simulated_seconds=STOP,stimulus_seconds=OFFSET-ONSET,
            wall_seconds=time.perf_counter()-start,initial_state_and_RNG_hash=initial_hash,
            input_identity=canonical_hash(seed['native_inputs']),runner_identity=arm.identity,
            model_parameters_hash=canonical_hash(seed['inner_parameters']),
            solver=protocol['reference'],native_state_unit='vH_native_voltage; not physical mV',
            observations=len(times),physiology_parameters_changed=False)
        return result,status
    finally:arm.close()


def response_statistics(values,times,control,threshold):
    delta=np.asarray(values)-values[0]
    corrected=delta-(np.asarray(control)-control[0])
    ms=np.rint((times-ONSET)*1000).astype(int)
    active=(ms>=0)&(ms<=150);window=(ms>=20)&(ms<=100)
    indices=np.flatnonzero(active);peak=int(indices[np.argmax(np.abs(corrected[active]))])
    mean=float(corrected[window].mean())
    polarity=lambda v:'unresolved' if abs(v)<=threshold else 'depolarizing' if v>0 else 'hyperpolarizing'
    return dict(peak_signed_V=float(corrected[peak]),peak_absolute_V=float(abs(corrected[peak])),
        peak_latency_ms=float(ms[peak]),peak_polarity=polarity(corrected[peak]),
        mean_20_100ms_V=mean,mean_polarity=polarity(mean),
        raw_mean_20_100ms_V=float(delta[window].mean()),
        raw_peak_absolute_V=float(abs(delta[active]).max()),
        value_at_150ms_V=float(corrected[np.flatnonzero(ms==150)[0]]),
        value_after_100ms_recovery_V=float(corrected[-1]))


def analyze(arrays,cells):
    result={};t=arrays['background__time_seconds']
    for n,row in cells.items():
        control=arrays['background__'+n+'__V'];drift=float(abs(control-control[0]).max())
        threshold=max(10*drift,1e-9+1e-6*abs(float(control[0])))
        a=response_statistics(arrays['center__'+n+'__V'],t,control,threshold)
        b=response_statistics(arrays['annulus__'+n+'__V'],t,control,threshold)
        resolved=abs(a['mean_20_100ms_V'])>threshold and abs(b['mean_20_100ms_V'])>threshold
        q=-b['mean_20_100ms_V']/a['mean_20_100ms_V'] if resolved else None
        expected=('hyperpolarizing' if n.endswith('_OFF') else 'depolarizing') if row['mandatory'] else None
        result[n]=dict(background_voltage_V=float(control[0]),background_max_drift_V=drift,
            resolution_threshold_V=threshold,center=a,annulus=b,Q_fixed_window=q,
            Q_status='defined' if resolved else 'undefined_unresolved_response',
            center_expected_polarity=expected,center_polarity_matches=a['mean_polarity']==expected if row['mandatory'] else None,
            antagonism_fixed_window=bool(q>0) if q is not None and row['mandatory'] else None,
            required_test=row['mandatory'])
        for label in ('center','annulus'):
            traces={}
            group=np.unique(row['direct_cone_group_rows'])
            hc=row['HC_dependencies']['group_rows']
            for name,idx in (('cones__release',group),('cones__Bh',group),('HC__Vh',hc)):
                v=arrays[label+'__'+name][idx];base=arrays['background__'+name][idx]
                change=(v-v[:,[0]])-(base-base[:,[0]])
                traces[name]=dict(maximum_absolute_control_subtracted_change=float(abs(change).max()),group_rows=idx)
            result[n][label]['upstream_propagation']=traces
    return result


def projection_audit(arrays):
    """Member-disjoint masks need not remain disjoint after frozen group averaging."""
    g=arrays['cone_group_rows'];c=arrays['center__group_fraction'];a=arrays['annulus__group_fraction']
    cm=arrays['center__fine_mask'];am=arrays['annulus__fine_mask']
    count=int(np.sum(cm&(a[g]>0)))
    return dict(member_masks_disjoint=not bool(np.any(cm&am)),
        shared_nonzero_groups=np.flatnonzero((c>0)&(a>0)),
        center_mask_members_receiving_annulus_group_drive=count,
        annulus_mask_members_receiving_center_group_drive=int(np.sum(am&(c[g]>0))),
        maximum_annulus_relative_drive_inside_center=float(CONTRAST*np.max(a[g[cm]],initial=0.)),
        ideal_center_region_kept_at_background=count==0,
        strict_pure_annulus_qualification='PASS' if count==0 else 'NOT_QUALIFIED_GROUP_PROJECTION_CONTAMINATION',
        explanation='Four target BC direct Cone groups remain unilluminated in the annulus, but other center-region members share input with illuminated group members. An opposite response is a conditional grouped-model result, not a clean ideal-annulus physiological pass.',
        minimal_future_input_fix='Before any future response, exclude every group intersecting the entire center disk from the annulus; derive a group-safe inner radius and verify it still fits the existing dynamic source. No parameter/connection change or automatic rerun.')


def plot(arrays,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    t=(arrays['background__time_seconds']-ONSET)*1000
    for ax,n in zip(axes.flat,TYPES):
        control=arrays['background__'+n+'__V'];drift=control-control[0]
        for label,color in (('center','#2155a3'),('annulus','#d87923')):
            v=arrays[label+'__'+n+'__V'];ax.plot(t,(v-v[0]-drift)*1e6,label=label,color=color)
        ax.plot(t,drift*1e6,label='background drift',color='gray',ls=':')
        ax.axvspan(0,150,color='gray',alpha=.08);ax.axvspan(20,100,color='green',alpha=.05)
        ax.axhline(0,color='black',lw=.5);ax.set(xlabel='Time from onset (ms)',ylabel='Delta V (uV)',
            title=n+(' (exploratory)' if n=='S_ON_BC' else ''));ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8)
    ax=axes[1,2];xy=arrays['cone_IS_mm'];c=metrics['geometry']['center_IS_mm']
    ax.scatter(xy[:,0],xy[:,1],s=2,color='lightgray')
    for label,color in (('center','#2155a3'),('annulus','#d87923')):
        take=arrays[label+'__fine_mask'];ax.scatter(xy[take,0],xy[take,1],s=4,color=color,label=label)
    ax.scatter(*c,marker='+',color='black',s=50);ax.set_aspect('equal');ax.set(title='Fixed masks on actual Cone IS positions',xlabel='IS x (mm)',ylabel='IS y (mm)')
    fig.suptitle('Frozen BC center-surround diagnostic | 5% relative native-Td step\nGroup averaging contaminates the center region: conditional response, not a pure-annulus pass',fontsize=12)
    fig.tight_layout();fig.savefig(OUT/'BC_CENTER_SURROUND_RESPONSES.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    if args.analyze_only:
        metrics=json.loads((OUT/'PHYSIOLOGY_METRICS.json').read_bytes())
        with np.load(OUT/'BC_CENTER_SURROUND_STATES.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        metrics['responses']=analyze(arrays,metrics['geometry']['cells'])
        metrics['input_projection_audit']=projection_audit(arrays)
        metrics['analysis_code_sha256']=digest(Path(__file__))
        metrics['analysis_note']='Saved arrays only; added whole-center group-overlap audit and marked S-ON as exploratory without a physiological gate. No stimuli, response windows, thresholds or simulated values changed.'
        plot(arrays,metrics)
        arrays['metadata_json']=np.array(json.dumps(json_data(metrics),ensure_ascii=False))
        np.savez_compressed(OUT/'BC_CENTER_SURROUND_STATES.npz',**arrays)
        write(OUT/'PHYSIOLOGY_METRICS.json',metrics);return
    OUT.mkdir(exist_ok=True)
    path=OUT/'PHYSIOLOGY_METRICS.json'
    with path.open('x',encoding='utf8') as f:f.write('{}\n')
    started=time.perf_counter();arrays={};guard=MemoryGuard()
    metrics=dict(status='PREPARING',started_utc=datetime.now(timezone.utc).isoformat(),conditions={},
        source=dict(url=SOURCE,doi='10.1016/S0042-6989(00)00039-0',locators=['Methods 2.2-2.3','Results 3.2','Table 1'],
            observations='Peripheral primate ON/OFF cone bipolar cells reverse spot polarity under annular light; cross-region qualitative comparator only',
            region='8 to >14mm, not central 0-6 degrees',
            protocol_difference='Paper uses flicker/gratings and 100% modulation; this experiment is a 5% single step',
            primary_full_text_verified=True),
        protocol=dict(onset_seconds=ONSET,offset_seconds=OFFSET,stop_seconds=STOP,contrast=CONTRAST,
            BC_recording_interval_seconds=.001,background='Frozen 1000 native-Td input per group; not a new spectral calibration',
            baseline='Each condition t=0; identical source states',
            analysis='All conditions use own baseline delta; center and annulus subtract the same time-matched background delta',
            peak_window_ms=[0,150],mean_window_ms=[20,100],
            resolution='max(10*maximum background voltage drift, original solver atol + rtol*abs(V0)); engineering resolution screen, not measured noise/error bound',
            Q='Negative ratio of the two control-subtracted 20-100ms means; null if either is unresolved; no area or gain interpretation',
            elapsed_tail_is_not_steady_state=True),
        protection=dict(max_rss_bytes=guard.max_rss_bytes,min_available_bytes=guard.min_available_bytes,condition_wall_limit_seconds=900),
        FIT_READY=False,production_qualified=False,new_parameters_calibrated=False)
    try:
        if not guard.check():raise ContractError(guard.stop_reason)
        accepted=json.loads((PARENT/'CALIBRATION_METRICS.json').read_bytes())
        frozen=dict(accepted['protected_sources'],**accepted['delivery_verification']['files'])
        for name,h in frozen.items():
            if digest(ROOT/name)!=h:raise ContractError('Frozen source changed before experiment: '+name)
        metrics['protected_files_before']=frozen
        model=RetinaBuilder.build_research_bc(ROOT,DEFAULT_CONFIGURATION)
        expected=accepted['initialization_order_verification']['current_source_model_identity']
        if model.identity!=expected:raise ContractError('Explicit current independent BC identity changed')
        metrics['model_identity']=model.identity;metrics['configuration']=DEFAULT_CONFIGURATION
        metrics['geometry'],fractions,arrays=geometry(model)
        metrics['geometry']['identity']=canonical_hash(metrics['geometry'])
        metrics['input_projection_audit']=projection_audit(arrays)
        metrics['preparation_seconds']=time.perf_counter()-started
        metrics['source_code_sha256']=digest(Path(__file__))
        metrics['model_parameter_hash']=canonical_hash(model.seed['inner_parameters'])
        metrics['operator_hashes']={k:matrix_id(getattr(model.upstream.native,k)) for k in ('A','B')}
        metrics['background_reuse']='Current Builder reconstructs the accepted algebraic background once; all conditions reuse exact state copies. No persisted complete t=0 checkpoint was available; no optical or 50ms acceptance rerun.'
        arrays['time_from_onset_seconds']=np.arange(261)*.001-ONSET
        metrics['status']='DESIGN_FIXED_BEFORE_RESPONSES';write(path,metrics)
        print('Fixed geometry:',json.dumps(json_data({k:metrics['geometry'][k] for k in ('center_IS_mm','center_radius_mm','annulus_inner_radius_mm','annulus_outer_radius_mm')})),flush=True)
        initial_hash=None
        for label in CONDITIONS:
            values,status=run_condition(model,fractions[label],metrics['geometry']['cells'],label,guard)
            if initial_hash is None:initial_hash=status['initial_state_and_RNG_hash']
            if initial_hash!=status['initial_state_and_RNG_hash']:raise ContractError('Conditions did not start at identical states')
            arrays.update(values);metrics['conditions'][label]=status
            arrays['metadata_json']=np.array(json.dumps(json_data(metrics),ensure_ascii=False))
            np.savez_compressed(OUT/'BC_CENTER_SURROUND_STATES.npz',**arrays)
            write(path,metrics)
        metrics['responses']=analyze(arrays,metrics['geometry']['cells'])
        metrics['status']='THREE_CONDITIONS_COMPLETED_CONDITIONAL_SPATIAL_MODEL'
        plot(arrays,metrics)
        metrics['protected_files_unchanged']=all(digest(ROOT/name)==h for name,h in frozen.items())
        if not metrics['protected_files_unchanged']:raise ContractError('Protected source changed during experiment')
    except Exception as error:
        metrics.update(status='FAILED_RETAINED',error=str(error),traceback=traceback.format_exc());raise
    finally:
        metrics.update(wall_seconds=time.perf_counter()-started,peak_working_set_bytes=peak_memory_bytes(),
                       ended_utc=datetime.now(timezone.utc).isoformat())
        write(path,metrics)
        if arrays:
            arrays['metadata_json']=np.array(json.dumps(json_data(metrics),ensure_ascii=False))
            np.savez_compressed(OUT/'BC_CENTER_SURROUND_STATES.npz',**arrays)
    print(metrics['status'],metrics['wall_seconds'],flush=True)


if __name__=='__main__':main()
