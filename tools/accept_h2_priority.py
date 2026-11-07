"""One matched L/M/S acceptance group, with selected exterior S input completed."""
from copy import copy
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import time
import traceback
import argparse
import io
import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

from retina_sim.assembly import RetinaBuilder
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.h2_priority import (DEFAULT_CONFIGURATION, complete_priority, PriorityAssembly,
                                  EccentricHCFullChainRunner)
from retina_sim.hc_eccentric import digest, checked_path
from retina_sim.h1_h2_effective import matrix_id
from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
from retina_sim.group_time_storage import load_bundle
from retina_sim.vh2005 import SOURCE_VALUES
from tools.accept_hc_eccentric import fullchain, reject
from tools.prepare_hc_eccentric import write
from tools.run_hc_minimal_real_diagnostic_v1 import peak_memory_bytes

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/CONE_HC_FINAL_IMPLEMENTATION_2026-11-02'
OUT=REPORT/'h2_priority_acceptance'


def uniform_control(model,label,cache):
    base=model.uniform_base
    control=copy(base)
    control.assembly=PriorityAssembly(base.assembly.parent,base.A,base.B,base.native.Q['H2'],
        base.calibration,base.spatial_receipt,base.config)
    control.assembly.priority_identity=canonical_hash(dict(uniform=base.identity,matched_exterior=model.exterior_members))
    control.assembly.exterior_mass=model.priority_mass['uniform_selected_exterior_group_mass']
    control.assembly.exterior_cache=cache;control.assembly.isolation_label=label
    control.assembly.native_boundary={k:v.copy() for k,v in base.native.boundary.items()}
    control.native=control.assembly.native_factory(control.assembly)
    control.identity=canonical_hash(dict(frozen_uniform=base.identity,label=label,exterior=cache.identity))
    return control


def weight_checks(model):
    folder=model.priority_directory
    spec=model.priority_receipt['specification']
    fixture=sparse.csr_matrix([[2.,1.,3.],[4.,2.,0.],[0.,0.,0.]])
    weighted=complete_priority(fixture,np.array(['L','M','S']),spec)
    np.testing.assert_allclose(weighted.data,[2/33,1/33,30/33,2/3,1/3],atol=1e-15,rtol=0)
    assert weighted[2].nnz==0 and weighted[1,2]==0
    old=sparse.load_npz(folder/'H2_uniform_fine_complete.npz').tocsr()
    new=sparse.load_npz(folder/'H2_A_fine_complete.npz').tocsr()
    types=model.priority_mass['all_cone_types'];g=np.where(types=='S',10.,1.)
    expected=old.multiply(g[None,:]).tocsr()
    denominator=np.asarray(expected.sum(1)).ravel()
    expected=sparse.diags(np.divide(1.,denominator,out=np.zeros_like(denominator),where=denominator>0))@expected
    np.testing.assert_allclose((new-expected).data,0.,rtol=0,atol=2e-14)
    old.sort_indices();new.sort_indices()
    np.testing.assert_array_equal(old.indptr,new.indptr);np.testing.assert_array_equal(old.indices,new.indices)
    both=[];without=[]
    for j in range(new.shape[0]):
        lo,hi=new.indptr[j:j+2];ct=types[new.indices[lo:hi]]
        if np.any(ct=='S') and np.any(ct!='S'):
            ratios=new.data[lo:hi]/old.data[lo:hi]
            both.append(float(ratios[ct=='S'][0]/ratios[ct!='S'][0]))
        if not np.any(ct=='S'):without.append(j)
    np.testing.assert_allclose(both,10.,rtol=0,atol=2e-14)
    np.testing.assert_allclose((new[without]-old[without]).data,0.,rtol=0,atol=2e-14)
    if (model.A[model.h1_rows]-model.uniform_base.A[model.h1_rows]).nnz:
        raise ContractError('H1 matrix changed')
    assert model.A[model.h1_rows][:,model.assembly.types['cones']=='S'].nnz==0
    assert matrix_id(model.B)==matrix_id(model.uniform_base.B)
    assert matrix_id(model.native.Q['H2'])==matrix_id(model.uniform_base.native.Q['H2'])
    assert model.calibration==model.uniform_base.calibration
    assert dict(model.native.parameters)==dict(model.uniform_base.native.parameters)
    assert dict(model.native.effective_h1_parameters)==dict(model.uniform_base.native.effective_h1_parameters)
    assert canonical_hash(model.seed['inner_parameters'])==canonical_hash(model.uniform_base.seed['inner_parameters'])
    rows=[]
    ci=model.members['cone_parent_rows'];dyn=np.zeros(len(types),bool);dyn[ci]=True
    for variant in ('uniform','priority'):
        group=sparse.load_npz(folder/('H2_uniform_group_complete.npz' if variant=='uniform' else 'H2_A_group_complete.npz')).tocsr()
        for row in model.h2_rows:
            start,stop=group.indptr[row:row+2];indices=group.indices[start:stop];values=group.data[start:stop]
            record=dict(model=variant,H2_group=int(row),HC_members=int(model.members['hc_counts'][row]))
            for label in ('L','M','S','LM'):
                chosen=np.isin(types[indices],list(label));weights=values[chosen]
                record.update({label+'_count':int(chosen.sum()),label+'_weight':float(weights.sum()),
                    label+'_mean_weight':float(weights.mean()) if len(weights) else None,
                    label+'_min_weight':float(weights.min()) if len(weights) else None,
                    label+'_max_weight':float(weights.max()) if len(weights) else None,
                    label+'_dynamic_count':int(np.sum(chosen&dyn[indices])),
                    label+'_dynamic_weight':float(values[chosen&dyn[indices]].sum())})
            rows.append(record)
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    weight_path=OUT/'H2_WEIGHTS_BEFORE_AFTER.csv'
    encoded=stream.getvalue().encode('utf-8-sig')
    if weight_path.exists():
        if weight_path.read_bytes()!=encoded:raise ContractError('Retained weight statistics changed during repair')
    else:
        with weight_path.open('xb') as target:target.write(encoded)
    return dict(formula_max_abs_error=float(np.max(abs((new-expected).data),initial=0.)),
        comparable_fine_H2_rows=len(both),S_relative_enhancement_min=min(both),S_relative_enhancement_max=max(both),
        no_S_rows_preserved=len(without),nonuniform_raw_fixture_passed=True,H1_A_and_S_exclusion_preserved=True,
        B_exactly_preserved=True,Q_H2_exactly_preserved=True,time_parameters_preserved=True,
        downstream_parameters_preserved=True,representative=[r for r in rows if r['H2_group']==18])


def trace(model,label,cache,priority_model):
    native=model.native;tgrid=np.linspace(0.,.12,601)
    state=[];times=[];y=model.y0.copy();R=model.light.copy();E=R.copy();nfev=0
    for epoch,(start,stop,light) in enumerate(((0.,.02,model.light),(.02,.12,model.stimulus(.02,label)))):
        anchor=AnalyticREAnchor(start,R,E,light,stop,epoch)
        solution=solve_ivp(lambda t,v:native.rhs(v,t,anchor),(start,stop),y,method='DOP853',
                           rtol=1e-7,atol=1e-10,max_step=.0002,dense_output=True)
        if not solution.success:raise ContractError(solution.message)
        take=(tgrid>=start)&(tgrid<=stop)
        if times:take&=tgrid>times[-1]
        for t in tgrid[take]:
            value=solution.sol(t);native._validate(value,anchor,float(t))
            state.append(value);times.append(float(t))
        y=solution.y[:,-1];R,E=anchor.at(stop);nfev+=solution.nfev
    states=np.column_stack(state);times=np.array(times);p=SOURCE_VALUES
    h=states[native.slices['Vh']]
    from scipy.special import expit
    feedback=model.B@h+native.boundary['B_h'][:,None]
    release=expit((states[native.slices['Vis']]-feedback-p['V_k'])/p['V_n'])/(states[native.slices['Vbar']]/p['V_I'])**p['mu']
    actual_boundary=np.column_stack([native.boundary_at(t)['A_release'] for t in times])
    ar=model.A@release+actual_boundary
    full=sparse.load_npz(priority_model.priority_directory/('H2_A_group_complete.npz' if model.A is priority_model.A else 'H2_uniform_group_complete.npz')).tocsr()
    rep=priority_model.config['representative_H2_group']
    sources=priority_model.exterior_members['representative_all_source_rows']
    types=priority_model.priority_mass['all_cone_types'][sources]
    source_release=[];source_feedback=[];source_light=[]
    lookup={int(row):int(group) for row,group in zip(model.members['cone_parent_rows'],model.members['cone_group'])}
    for row,kind in zip(sources,types):
        if int(row) in lookup:
            group=lookup[int(row)];source_release.append(release[group]);source_feedback.append(feedback[group])
        else:
            if kind!='S' or row not in priority_model.exterior_members['cone_parent_rows']:
                raise ContractError('Uncovered stimulated source of representative H2')
            source_release.append(np.array([cache.at(t)[0] if label=='S' else cache.background_release for t in times]))
            source_feedback.append(np.full(len(times),cache.feedback))
        source_light.append(np.where((times>.02)&(times<=.12)&(kind==label),1010.,1000.))
    source_release=np.array(source_release);source_feedback=np.array(source_feedback)
    source_light=np.array(source_light)
    weights=np.array([float(full[rep,int(row)]) for row in sources])
    np.testing.assert_allclose(weights@source_release,ar[rep],rtol=2e-14,atol=2e-14)
    dh=h[rep]-h[rep,0];dr=source_release-source_release[:,[0]]
    selected=types==label
    if not np.all(np.max(abs(dr[selected]),axis=1)>1e-6):
        raise ContractError('A stimulated connected Cone did not receive an effective time-varying drive')
    assert np.all(source_light[selected][:,times>.02]==1010.)
    record=dict(condition=label,connected_cones=int(selected.sum()),total_weight=float(weights[selected].sum()),
        mean_weight=float(weights[selected].mean()),min_H2_delta=float(dh.min()),max_H2_delta=float(dh.max()),
        peak_abs_H2=float(abs(dh).max()),peak_time_seconds=float(times[np.argmax(abs(dh))]),
        hyperpolarizing=bool(dh.min() < -1e-8 and dh.max()<1e-8),
        all_stimulated_sources_dynamic=True,source_release_peak_range=[float(np.max(abs(dr[selected]),axis=1).min()),float(np.max(abs(dr[selected]),axis=1).max())],
        native_rhs_evaluations=nfev,model_identity=model.identity)
    return dict(time_seconds=times,states=states,H2=h[model.h2_rows],cone_release=release,
        H2_A_release=ar[model.h2_rows],B_Vh=feedback,representative_H2=h[rep],
        source_cone_parent_rows=sources,source_cone_types=types,source_weights=weights,
        source_release=source_release,source_feedback=source_feedback,source_light_td=source_light),record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-failed',action='store_true')
    args=parser.parse_args()
    start=time.perf_counter()
    if args.resume_failed:
        previous=json.loads((OUT/'RESULT.json').read_bytes())
        if previous.get('status')!='FAILED_RETAINED':raise ContractError('Only the failed acceptance may resume')
        with (OUT/'INPUT_BRANCH_FAILURE.json').open('xb') as stream:stream.write((OUT/'RESULT.json').read_bytes())
    else:OUT.mkdir(exist_ok=False)
    result=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat())
    saved={}
    try:
        model=RetinaBuilder.build_research_cone_hc(ROOT,DEFAULT_CONFIGURATION)
        protected=dict(model.uniform_base.spatial_receipt['source_files'])
        protected.update(model.priority_receipt['source_files'])
        result['weights']=weight_checks(model)
        result['background']=dict(max_native_rhs=model.background_rhs,diagnostics=model.background_diagnostics,
                                  new_model_identity=model.identity,uniform_identity=model.uniform_base.identity)
        cache=model.exterior_cache()
        result['exterior']=dict(identity=cache.identity,source_members=model.exterior_members,
            solver=cache.solver,nfev=cache.solution.nfev,feedback_held_native=cache.feedback,
            policy=model.priority_receipt['exterior_policy'])
        result['responses']={};conditions={}
        for label in ('L','M','S'):
            weighted=model.isolation(label,cache);conditions[label]=weighted
            baseline=uniform_control(model,label,cache)
            np.testing.assert_allclose(weighted.y0,baseline.y0,rtol=0,atol=2e-12)
            for variant,current in (('uniform',baseline),('priority',weighted)):
                arrays,summary=trace(current,label,cache,model)
                result['responses'][variant+'_'+label]=summary
                saved.update({variant+'__'+label+'__'+key:value for key,value in arrays.items()})
                print(variant+' '+label+' peak='+str(summary['peak_abs_H2']),flush=True)
        np.savez_compressed(OUT/'H2_LMS_RESPONSES.npz',**saved)
        if not all(row['hyperpolarizing'] for row in result['responses'].values()):
            raise ContractError('Matched positive L/M/S inputs did not preserve same response polarity')
        result['FULLCHAIN_INTERFACE']=fullchain(conditions['S'],OUT)
        historical=load_bundle(REPORT/'eccentric_acceptance/fullchain_checkpoint',expected_kind='complete_model_checkpoint')['payload']
        reject(lambda:EccentricHCFullChainRunner.from_checkpoint(conditions['S'].assembly,conditions['S'].seed,conditions['S'].protocol,historical))
        for path,expected in protected.items():checked_path(ROOT,path,expected)
        result.update(status='COMPLETE',H2_INPUT_WEIGHT_IMPLEMENTED=True,H2_ENGINEERING_ACCEPTANCE='PASS',
            H2_SPECTRAL_PHYSIOLOGY_CONSISTENCY='QUALITATIVE_INPUT_PRIORITY_IMPLEMENTED_AMPLITUDE_CONSISTENCY_NOT_CALIBRATED',
            FIT_READY_changed=False,old_weight_checkpoint_rejected=True,
            uniform_control='Frozen original uniform A2 with same completed two-S exterior stimulus; prior partial-input traces retained separately',
            H1_refit=False,parameters_tuned=False,protected_files=protected,
            implementation_sha256=digest(Path(__file__)))
    except Exception as error:
        result.update(status='FAILED_RETAINED',error=str(error),traceback=traceback.format_exc());raise
    finally:
        result.update(wall_seconds=time.perf_counter()-start,peak_working_set_bytes=peak_memory_bytes(),
                      ended_utc=datetime.now(timezone.utc).isoformat())
        write(OUT/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ('status','wall_seconds','peak_working_set_bytes')}),flush=True)


if __name__=='__main__':main()
