"""One real-source Builder/Runner acceptance case for the final functional kernel."""
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import argparse
import csv
import hashlib
import json
import time
import traceback
import numpy as np
from scipy import sparse
from retina_sim.assembly import RetinaBuilder
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.cone_hc_spatial import ConeHCSpatialRunner,FUNCTIONAL_KERNEL
from retina_sim.group_time_storage import save_bundle,load_bundle
from retina_sim.h1_h2_effective import matrix_id
from retina_sim.hc_support_closure import native_dependency_closure,require_closed_native
from retina_sim.vh2005 import SOURCE_VALUES
from tools.validate_cone_hc_engineering_integration import downstream
from tools.run_hc_minimal_real_diagnostic_v1 import peak_memory_bytes


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def rejects(call):
    try:call()
    except ContractError:return True
    raise ContractError('Unsupported mode/checkpoint/structure was silently admitted')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=out.parent;clock=time.perf_counter()
    result=dict(started_utc=datetime.now(timezone.utc).isoformat(),status='RUNNING',acceptance_case='real_H1_2891_final_functional_Builder_Runner')
    try:
        before=json.loads((report/'BEFORE_SHA256.json').read_bytes())
        protected=json.loads((report/'PROTECTED_FILES.json').read_bytes())
        for name in protected:
            if sha(root/name)!=before[name]:raise ContractError('Protected model/config changed: '+name)
        build_clock=time.perf_counter();model=RetinaBuilder.build_research_cone_hc(root)
        assert model.kernel_definition==FUNCTIONAL_KERNEL and model.config['spatial_mode']=='biexponential'
        assert model.config['execution_mode']=='local_cpu'
        result['build_wall_seconds']=time.perf_counter()-build_clock
        target=int(model.geometry['target_hc_local']);assets=model.repo_root/model.config['assets']['spatial_receipt']['path']
        receipt=model.spatial_receipt;adir=assets.parent
        row=np.load(adir/'TARGET_COMPLETE_ROW.npz',allow_pickle=False)
        d=np.linalg.norm(row['cone_pedicle_mm']-row['target_hc_anchor_mm'],axis=1)
        lm=(row['cone_types']!='S')&(d<=receipt['support_radius_mm'])
        s=np.zeros(len(d));l=np.zeros(len(d))
        s[lm]=np.exp(-d[lm]/.020);s/=s.sum()
        l[lm]=np.exp(-d[lm]/.300);l/=l.sum()
        expected=.15*s+.85*l
        np.testing.assert_allclose(row['A'],expected,rtol=2e-14,atol=1e-17)
        dynamic=row['dynamic_source_rows']
        np.testing.assert_allclose(model.A[target].toarray()[0],expected[dynamic],rtol=2e-14,atol=1e-17)
        np.testing.assert_allclose(model.held_mass[target],expected.sum()-expected[dynamic].sum(),rtol=0,atol=2e-14)
        np.testing.assert_allclose(np.asarray(model.A.sum(1)).ravel()+model.held_mass,1.,rtol=0,atol=2e-14)
        short=sparse.load_npz(adir/'short_dynamic.npz');long=sparse.load_npz(adir/'long_dynamic.npz')
        residual=model.A-(.15*short+.85*long)
        assert np.max(abs(residual.data),initial=0.)<2e-14
        raw=sparse.load_npz((root/model.config['assets']['source_receipt']['path']).parent/'RAW_HC_BY_CONE.npz').tocsr()
        direct=model.geometry['core_local'];noncontact=np.ones(model.A.shape[1],bool);noncontact[direct]=False
        assert np.any(short[target,noncontact].data>0)
        np.testing.assert_array_equal(raw[int(model.geometry['hc_parent_rows'][target])].indices,model.geometry['core_cone_parent_rows'])
        B=sparse.load_npz(root/model.config['assets']['B_local']['path'])
        assert matrix_id(model.B)==matrix_id(B)
        np.testing.assert_array_equal(model.geometry['cone_parent_rows'][model.binding['unsupported_feedback']],[166128])
        # The strict gate must continue to reject the registered research approximation.
        closure=native_dependency_closure(model.A,model.B,np.arange(model.A.shape[1]),np.arange(model.A.shape[0]),
            hc_support_complete=model.held_mass==0,cone_feedback_complete=~model.binding['unsupported_feedback'],
            cone_input_available=np.ones(model.A.shape[1],bool))
        strict_rejected=rejects(lambda:require_closed_native(closure))
        for mode in ('cuda','group_gpu','unsupported'):
            rejects(lambda mode=mode:RetinaBuilder.build_research_cone_hc(root,execution_mode=mode))
            rejects(lambda mode=mode:model.runner(execution_mode=mode))
        result['matrix']=dict(A_identity=matrix_id(model.A),B_identity=matrix_id(model.B),
            shape=list(model.A.shape),nnz=model.A.nnz,full_target_LM_count=int(lm.sum()),
            full_target_sum=float(expected.sum()),short_noncontact_dynamic_mass=float(.15*short[target,noncontact].sum()),
            short_noncontact_dynamic_count=int(np.count_nonzero(short[target,noncontact].data)),
            short_held_mass=float(1-short[target].sum()),long_held_mass=float(1-long[target].sum()),
            max_formula_error=float(abs(model.A[target].toarray()[0]-expected[dynamic]).max()),
            B_exact=True,H2_rule='independent uniform; exercised by downstream native H2 rows',
            registered_unsupported=[166128],strict_structure_rejected=strict_rejected,
            no_silent_fallback=True,anatomical_contacts_added=0)
        result['model']=dict(identity=model.identity,native_identity=model.native.source_version,
            state_layout=model.state_layout,parameters=dict(SOURCE_VALUES),background_max_rhs=model.background_rhs,
            configuration=model.config,kernel_receipt_sha256=sha(assets))

        runner=model.runner();first=runner.advance(.06,np.arange(31)*.002)
        save_bundle(out/'native_checkpoint',kind='cone_hc_spatial_checkpoint',identity=dict(model=model.identity),
            payload=runner.checkpoint_payload(),qualification=dict(default_adopted=False,fit_eligible=False,research_only=True))
        loaded=load_bundle(out/'native_checkpoint',expected_kind='cone_hc_spatial_checkpoint',expected_identity=dict(model=model.identity))
        restored=ConeHCSpatialRunner.from_checkpoint(model,loaded['payload'])
        bad=deepcopy(loaded['payload']);bad['model_identity']='uniform-or-wrong-A'
        rejects(lambda:ConeHCSpatialRunner.from_checkpoint(model,bad))
        tail=runner.advance(.12,np.arange(31,61)*.002)
        restored_tail=restored.advance(.12,np.arange(31,61)*.002)
        error={k:float(np.max(abs(tail[k]-restored_tail[k]))) for k in tail}
        assert max(error.values())<1e-7
        data={k:np.concatenate((first[k],tail[k]),axis=1 if first[k].ndim==2 else 0) for k in first}
        assert all(np.isfinite(v).all() for v in data.values())
        assert data['y'].dtype==np.float64
        assert abs(data['H1_target']-data['H1_target'][0]).max()>1e-6
        assert abs(data['release_core']-data['release_core'][:,0,None]).max()>1e-8
        params=SOURCE_VALUES
        feedback=B@data['H1_all']+model.native.boundary['B_h'][:,None]
        vbar=model.A@data['y'][model.native.slices['Vbar']]+model.native.boundary['A_Vbar'][:,None]
        np.testing.assert_allclose(data['pooled_aI_target'],(vbar[target]/params['V_I'])**params['mu'],rtol=2e-14,atol=1e-14)
        np.testing.assert_allclose(data['feedback_core'],feedback[direct],rtol=2e-14,atol=1e-14)
        for t,y in zip(data['t_seconds'],data['y'].T):
            segment=next(q for q in runner.segments if q[0]<=t<=q[1])
            model.native._validate(y,segment[3],t)
        np.savez_compressed(out/'H1_CONE_TRAJECTORY.npz',**data)
        np.savez_compressed(out/'MASS_LEDGER.npz',**model.binding['mass_ledger'])
        with (out/'H1_CONE_KEY_TRACES.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['time_s','H1_native','H1_delta','pooled_release','pooled_Vbar','pooled_aI']+[f'Cone_{int(r)}_release' for r in model.geometry['core_cone_parent_rows']])
            for i,t in enumerate(data['t_seconds']):writer.writerow([t,data['H1_target'][i],data['H1_target'][i]-data['H1_target'][0],data['pooled_release_target'][i],data['pooled_Vbar_target'][i],data['pooled_aI_target'][i],*data['release_core'][:,i]])
        result['native_run']=dict(simulated_seconds=.12,samples=len(data['t_seconds']),metrics=runner.metrics,
            checkpoint_branch_metrics=restored.metrics,restore_errors=error,all_finite=True,state_domain_valid=True,
            H1_baseline=float(data['H1_target'][0]),H1_delta_final=float(data['H1_target'][-1]-data['H1_target'][0]),
            release_delta_final=(data['release_core'][:,-1]-data['release_core'][:,0]).tolist(),
            pooled_aI_range=[float(data['pooled_aI_target'].min()),float(data['pooled_aI_target'].max())],
            spatial_searches_in_time_steps=0,weight_builds_in_time_steps=0,
            qualification='native-scale conditional diagnostic; not physiological RF validation')

        # Restore an actual historical checkpoint under its unchanged explicit frozen config.
        historical=RetinaBuilder.build_research_cone_hc(root,'configurations/research/cone_hc_biexponential.json')
        old_dir=root/'reports/CONE_HC_ENGINEERING_INTEGRATION_2026-11-02/interface_validated_run/native_checkpoint'
        old_checkpoint=load_bundle(old_dir,expected_kind='cone_hc_spatial_checkpoint')['payload']
        old_runner=ConeHCSpatialRunner.from_checkpoint(historical,old_checkpoint)
        np.testing.assert_array_equal(old_runner.y,old_checkpoint['y'])
        rejects(lambda:ConeHCSpatialRunner.from_checkpoint(model,old_checkpoint))
        frozen_uniform=RetinaBuilder.build_research_cone_hc(root,'configurations/research/cone_hc_uniform.json')
        assert frozen_uniform.config['spatial_mode']=='uniform'
        assert matrix_id(frozen_uniform.A)==matrix_id(sparse.load_npz(root/frozen_uniform.config['assets']['A_uniform']['path']))
        result['compatibility']=dict(historical_checkpoint_loaded=True,historical_time=old_runner.time,
            cross_kernel_checkpoint_rejected=True,frozen_uniform_matrix_exact=True,
            final_state_layout_unchanged=model.state_layout==historical.state_layout)
        result['downstream']=downstream(model,runner,out,source_start_seconds=.06)
        for name in protected:
            if sha(root/name)!=before[name]:raise ContractError('Protected source changed during acceptance: '+name)
        result.update(status='CONE_HC_FINAL_IMPLEMENTATION_ACCEPTED',protected_sources_unchanged=True,
            spatial_convergence='HISTORICAL_FAIL_RETAINED; FINAL_KERNEL_NOT_SPATIALLY_QUALIFIED',
            physiological_prediction_qualified=False,new_supported_local_default='biexponential_functional',
            frozen_historical_default='uniform',acceptance_cases_executed=1)
    except Exception as exc:
        result.update(status='FAILED_RETAINED',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        result.update(ended_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-clock,
                      peak_working_set_bytes=peak_memory_bytes())
        (out/'RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','wall_seconds','peak_working_set_bytes')},indent=2),flush=True)


if __name__=='__main__':main()
