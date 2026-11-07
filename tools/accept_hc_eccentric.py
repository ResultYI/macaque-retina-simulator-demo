"""One explicit source-bound HC acceptance group and one minimal full-chain run.

This script is not called by preparation, import or Builder construction.
It must only be executed after the user authorizes running the written code.
"""
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import time
import traceback

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

from retina_sim.assembly import RetinaBuilder
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.cone_hc_eccentric import EccentricHCFullChainRunner,DEFAULT_CONFIGURATION
from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
from retina_sim.hc_eccentric import project_members,digest
from retina_sim.hc_group_projection import _member_mean
from retina_sim.h1_h2_effective import H1H2Native,matrix_id
from retina_sim.group_time_storage import save_bundle,load_bundle
from retina_sim.spatial_group_model import BC,RGC
from retina_sim.vh2005 import SOURCE_VALUES
from tools.prepare_hc_eccentric import write
from tools.run_hc_minimal_real_diagnostic_v1 import peak_memory_bytes
from tools.validate_h1_h2_effective import tick


ROOT=Path(__file__).resolve().parents[1]


def reject(call):
    try: call()
    except ContractError: return
    raise ContractError('Invalid identity or unsupported execution was accepted')


def matrix_checks(model):
    folder=model.repo_root/model.config['assets_directory'];m=model.members
    fine=sparse.load_npz(folder/'A_fine_dynamic.npz')
    grouped,held=project_members(fine,model.mass['A_fine_held'],m['cone_group'],m['hc_group'],
                                 m['cone_counts'],m['hc_counts'])
    np.testing.assert_allclose((grouped-model.A).toarray(),0.,rtol=0,atol=2e-14)
    np.testing.assert_allclose(held,model.mass['A_group_held'],rtol=0,atol=2e-14)
    fineB=sparse.load_npz(folder/'B_fine_dynamic.npz')
    groupedB=_member_mean(fineB,m['cone_group'],m['hc_group'],m['cone_counts'],model.B.shape)
    np.testing.assert_allclose((groupedB-model.B).toarray(),0.,rtol=0,atol=2e-14)
    if fine[m['hc_types']=='H1'][:,m['cone_types']=='S'].nnz:
        raise ContractError('Fine H1 matrix contains S input')
    with np.load(folder/'H1_COMPLETE_ROW_WITNESSES.npz',allow_pickle=False) as z:
        rows=z['fine_rows'].copy();checks=[]
        if len(rows)!=2 or len(np.unique(m['hc_group'][rows]))!=2:
            raise ContractError('Two distinct eccentric H1 groups required')
        if m['lambda_short_mm'][rows[0]]==m['lambda_short_mm'][rows[1]] or m['lambda_long_mm'][rows[0]]==m['lambda_long_mm'][rows[1]]:
            raise ContractError('Selected H1 spatial scales are identical')
        official=np.array([int(k.decode('ascii').split(':')[-1]) for k in m['cone_keys']])
        for j in rows:
            prefix='row_'+str(j)+'__'
            distance=z[prefix+'distance_mm']
            np.testing.assert_allclose(distance,np.linalg.norm(z[prefix+'cone_pedicle_mm']-m['hc_anchor_mm'][j],axis=1),rtol=0,atol=1e-14)
            short=np.exp(-distance/m['lambda_short_mm'][j]);short/=short.sum()
            long=np.exp(-distance/m['lambda_long_mm'][j]);long/=long.sum()
            weights=.15*short+.85*long
            np.testing.assert_allclose(weights,z[prefix+'A'],rtol=3e-14,atol=1e-17)
            if np.any(z[prefix+'cone_types']=='S'):raise ContractError('H1 witness has an illegal source')
            lookup=dict(zip(z[prefix+'official_source_rows'],weights))
            expected=np.array([lookup.get(k,0.) for k in official])
            error=float(np.max(abs(fine[j].toarray()[0]-expected)))
            if error>2e-14:raise ContractError('Dynamic crop differs from full-source normalization')
            checks.append(dict(fine_row=int(j),parent_row=int(m['hc_parent_rows'][j]),
                group=int(m['hc_group'][j]),eccentricity_mm=float(m['eccentricity_mm'][j]),
                lambda_short_mm=float(m['lambda_short_mm'][j]),lambda_long_mm=float(m['lambda_long_mm'][j]),
                support_radius_mm=float(m['support_radius_mm'][j]),full_LM_sources=len(weights),
                full_mass=float(weights.sum()),formula_max_abs_error=error))
    return dict(selected_H1=checks,A_identity=matrix_id(model.A),B_identity=matrix_id(model.B),
        fine_member_projection=True,H1_S_excluded=True,crop_renormalized=False,
        H2_Q_identity=matrix_id(model.native.Q['H2']),H2_parameters=dict(model.native.parameters),
        effective_H1_parameters=dict(model.native.effective_h1_parameters))


def native_trace(model,label):
    native=model.native;spec=model.input;onset=float(spec['onset_seconds']);stop=float(spec['offset_seconds'])
    selected=None if label=='annulus' else label
    times=np.linspace(0.,stop,int(round(stop/.0002))+1)
    packed=[];stamps=[];epochs=[];y=model.y0.copy();R=model.light.copy();E=R.copy();nfev=0
    for epoch,(start,end,light) in enumerate(((0.,onset,model.light),(onset,stop,model.stimulus(onset,selected)))):
        anchor=AnalyticREAnchor(start,R,E,light,end,epoch)
        solution=solve_ivp(lambda t,v:native.rhs(v,t,anchor),(start,end),y,
            method='DOP853',rtol=1e-7,atol=1e-10,max_step=.0002,dense_output=True)
        if not solution.success:raise ContractError(solution.message)
        take=(times>=start)&(times<=end)
        if stamps:take&=times>stamps[-1]
        for t in times[take]:
            state=solution.sol(t);native._validate(state,anchor,float(t))
            packed.append(state);stamps.append(float(t));epochs.append(anchor)
        y=solution.y[:,-1];R,E=anchor.at(end);nfev+=solution.nfev
    state=np.column_stack(packed);outputs=[]
    for t,y,anchor in zip(stamps,state.T,epochs):outputs.append(native.evaluate(y,t,anchor).outputs)
    release=np.column_stack([o['release'] for o in outputs])
    voltage=state[native.slices['Vh']]
    feedback=model.B@voltage+native.boundary['B_h'][:,None]
    vbar=model.A@state[native.slices['Vbar']]+native.boundary['A_Vbar'][:,None]
    ar=model.A@release+native.boundary['A_release'][:,None]
    np.testing.assert_allclose(feedback,np.column_stack([o['Bh'] for o in outputs]),rtol=2e-14,atol=1e-14)
    np.testing.assert_allclose((vbar[model.h1_rows]/SOURCE_VALUES['V_I'])**SOURCE_VALUES['mu'],
        np.column_stack([o['pooled_aI'][model.h1_rows] for o in outputs]),rtol=2e-14,atol=1e-14)
    # Compare the inherited source coefficient with the one permitted H1 change.
    y=state[:,-1];anchor=epochs[-1]
    sa,sb,_=H1H2Native._coefficients(native,y,stamps[-1],anchor)
    aa,bb,_=native._coefficients(y,stamps[-1],anchor)
    scale=20./model.calibration['tau_h_ms']
    np.testing.assert_allclose(aa[7][model.h1_rows],sa[7][model.h1_rows]*scale,rtol=0,atol=0)
    np.testing.assert_allclose(bb[7][model.h1_rows],sb[7][model.h1_rows]*scale,rtol=0,atol=0)
    for i in (0,1,2,3,4,5,6,8):
        np.testing.assert_array_equal(aa[i],sa[i]);np.testing.assert_array_equal(bb[i],sb[i])
    if not all(np.isfinite(v).all() for v in (state,release,feedback,vbar,ar)):
        raise ContractError('Nonfinite native trajectory')
    return dict(time_seconds=np.array(stamps),states=state,H1=voltage[model.h1_rows],
        H2=voltage[model.h2_rows],cone_release=release,B_Vh=feedback,A_Vbar=vbar,A_release=ar),nfev


def spectral_checks(model,traces):
    H1=model.h1_rows;H2=model.h2_rows;types=model.assembly.types['cones']
    A2=model.A[H2]
    smass=np.asarray(A2[:,types=='S'].sum(1)).ravel()
    lmmass=np.asarray(A2[:,types!='S'].sum(1)).ravel()
    eligible=np.flatnonzero((smass>0)&(lmmass>0))
    if not len(eligible):raise ContractError('No existing H2 representative has both S and LM input')
    h2=int(eligible[0]);report={}
    for label in ('L','M','S','LM'):
        h1=traces[label]['H1']-traces[label]['H1'][:,[0]]
        h=traces[label]['H2'][h2]-traces[label]['H2'][h2,0]
        report[label]=dict(H1_min_delta=float(h1.min()),H1_max_abs_delta=float(abs(h1).max()),
            H2_min_delta=float(h.min()),H2_final_delta=float(h[-1]),H2_peak_abs=float(abs(h).max()))
    compatible=(report['S']['H1_max_abs_delta']<=1e-8
        and all(report[k]['H1_min_delta'] < -1e-8 for k in ('L','M'))
        and all(report[k]['H2_min_delta'] < -1e-8 for k in ('S','L','M','LM'))
        and report['S']['H2_peak_abs']>report['LM']['H2_peak_abs'])
    return dict(status='QUALITATIVELY_COMPATIBLE' if compatible else 'H2_SPECTRAL_CONSTRAINT_UNRESOLVED',
        representative_H2_group=int(H2[h2]),group_id=str(model.assembly.populations['HC'].group_ids[H2[h2]]),
        retained_S_weight=float(smass[h2]),retained_LM_weight=float(lmmass[h2]),responses=report,
        input='equal fractional native Td increments, existing class labels; not an exact LED spectral replication',
        source_rule='Dacey1996 doi:10.1126/science.271.5249.656; existing project H1/H2 source record',
        stronger_S_comparison='S peak magnitude versus joint LM at the same increment',
        S_gain_added=False,physiological_prediction_qualified=False)


def fullchain(model,out):
    arm=model.runner();restored=None;rows=[];error=0.;p=model.protocol
    origin=p['clock_origin_seconds'];steps=int(round((p['stop_seconds']-origin)/.0001))
    mid=steps//2;frozen=canonical_hash(model.seed['inner_parameters'])
    clock=time.perf_counter()
    try:
        initial=arm.snapshot(arm.start,'right',queue_node=True);rows.append(initial)
        arm.initial_commit();arm.ack(arm.start)
        for k in range(1,steps+1):
            t=origin+k*.0001
            _,right=tick(arm,t,k)
            arm.inner.validate(arm.inner_state)
            if any(not np.isfinite(v).all() for section in ('states','rhs','ports') for v in right[section].values()):
                raise ContractError('Nonfinite full-chain state, RHS or port')
            if k%10==0:rows.append(right)
            if k==mid:
                identity=dict(model=model.identity,execution=arm.identity)
                save_bundle(out/'fullchain_checkpoint',kind='complete_model_checkpoint',identity=identity,
                    payload=arm.checkpoint_payload(),qualification=dict(default_adopted=False,research_only=True,fit_eligible=False,physiological_calibration=False))
                payload=load_bundle(out/'fullchain_checkpoint',expected_kind='complete_model_checkpoint',expected_identity=identity)['payload']
                restored=EccentricHCFullChainRunner.from_checkpoint(model.assembly,model.seed,p,payload)
                bad=deepcopy(payload);bad['controller']['H1_tau_h_ms']=20. if model.calibration['tau_h_ms']!=20. else 40.
                reject(lambda:EccentricHCFullChainRunner.from_checkpoint(model.assembly,model.seed,p,bad))
                reject(lambda:EccentricHCFullChainRunner.from_checkpoint(model.assembly,model.seed,p,{'controller':{}}))
            elif k>mid:
                _,other=tick(restored,t,k)
                for section in ('states','rhs','ports'):
                    if set(right[section])!=set(other[section]):raise ContractError('Restored state/port registry changed')
                    for key,value in right[section].items():
                        error=max(error,float(np.max(abs(value-other[section][key]),initial=0.)))
                if canonical_hash(arm.spike_states)!=canonical_hash(restored.spike_states):
                    raise ContractError('RGC probability history/RNG changed on restore')
        if error!=0. or canonical_hash(model.seed['inner_parameters'])!=frozen:
            raise ContractError('Checkpoint continuation or downstream parameter identity differs')
        saved={'time_seconds':np.array([r['time_seconds'] for r in rows])}
        for section in ('states','ports'):
            for key in rows[-1][section]:
                saved[section+'__'+'__'.join(key)]=np.column_stack([r[section][key] for r in rows])
        np.savez_compressed(out/'FULLCHAIN_STATES.npz',**saved)
        required=[('cones','release'),('HC','h1'),('HC','h2'),('AII_AC','voltage')]
        required += [(pop,name) for pop in BC for name in ('voltage','effective_release')]
        required += [(pop,name) for pop in RGC for name in ('voltage','gE','gI','expected_rate_Hz')]
        if not any(key[0] in arm.inner.ac_populations for key in rows[-1]['states']):
            raise ContractError('AC state readouts absent')
        if any(k not in rows[-1]['ports'] for k in required):raise ContractError('Required full-chain readout absent')
        return dict(status='CONDITIONAL_PASS',duration_seconds=p['stop_seconds']-arm.start,
            wall_seconds=time.perf_counter()-clock,checkpoint_restore_max_abs_error=error,
            source_model=model.identity,native_identity=arm.native_physics.source_version,
            H1_tau_h_ms=arm.native_physics.tau_h_ms,H2_parameters=dict(arm.native_physics.parameters),
            BC_parameter_hash=frozen,reference_BC_identity=model.reference_bc_identity,
            RGC_history_RNG_equal=True,old_checkpoint_rejected=True,metrics=arm.metrics,
            port_units={':'.join(k):v for k,v in rows[-1]['port_units'].items()},
            limitations='Finite held exterior; central spatial physiology and whole-network convergence not qualified')
    finally:
        arm.close()
        if restored is not None:restored.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',default=DEFAULT_CONFIGURATION)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/CONE_HC_FINAL_IMPLEMENTATION_2026-11-02/eccentric_acceptance')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ContractError('Acceptance output must stay in workspace')
    out.mkdir(parents=True,exist_ok=False)
    clock=time.perf_counter();result=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat())
    try:
        model=RetinaBuilder.build_research_cone_hc(ROOT,args.configuration)
        result['model_identity']=model.identity
        result['matrices']=matrix_checks(model)
        result['background']=dict(max_native_rhs=model.background_rhs,diagnostics=model.background_diagnostics,
            identity=model.seed['joint_background_initialization'],new_state=True)
        result['calibration']=model.calibration
        traces={};calls={};saved={}
        for label in ('annulus','L','M','S','LM'):
            traces[label],calls[label]=native_trace(model,label)
            saved.update({label+'__'+k:v for k,v in traces[label].items()})
        np.savez_compressed(out/'H1_H2_NATIVE_TRAJECTORIES.npz',**saved)
        result['native_rhs_evaluations']=calls
        result['H2_SPECTRAL_CONSTRAINT']=spectral_checks(model,traces)
        # Only after the initialization and local source tests have completed.
        result['FULLCHAIN_INTEGRATION']=fullchain(model,out)
        result.update(status='CONDITIONAL_ENGINEERING_ACCEPTANCE_PASS',HC_INITIALIZATION='COMPLETE',
            H1_TEMPORAL_CALIBRATION=model.calibration['decision'],
            next_stage_fit_ready=False,central_spatial_physiology_qualified=False,
            source_files=model.spatial_receipt['source_files'])
    except Exception as error:
        result.update(status='FAILED_RETAINED',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        result.update(ended_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-clock,
                      peak_working_set_bytes=peak_memory_bytes())
        write(out/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ('status','wall_seconds','peak_working_set_bytes')}),flush=True)


if __name__=='__main__':main()
