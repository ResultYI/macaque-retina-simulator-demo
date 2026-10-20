"""Approved fixed-dark rod reference: actual backgrounds and one short flash pair.

No fitted parameters, optical capture, active membrane, wiring, or reset of
the reference at runtime. Coupled background adaptation is not certified here.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,time
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_rgc_type_candidate import (RunConfig,atlas_population_network,HybridRunner,
    constant_background_channels,RecordingPlan,MemoryGuard,record_segment,save,sha,
    seed_background_state,balance_rgc_initial_voltages,SolverConfig)
from retina_sim.contracts import ContractError,Quantity,canonical_hash
from retina_sim.network import InputBundle
from retina_sim.rod_pathway import rod_source_values


def values(c):return {r.parameter_id:np.asarray(r.value.values) for r in c.parameters.records}
def stats(a):
    a=np.asarray(a);return dict(mean=float(a.mean()),minimum=float(a.min()),maximum=float(a.max()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-startup',action='store_true');a=p.parse_args()
    d=a.output.resolve()
    if not d.is_relative_to(ROOT):raise ContractError('Project output required')
    if a.resume_startup:
        prior=json.loads((d/'result.json').read_bytes())
        if prior.get('new_neural_seconds')!=0. or not prior.get('error','').startswith('Recording requires at least') or any(d.rglob('chunk_*.npz')):
            raise ContractError('Only the pre-advance recording-budget failure can use startup recovery')
        save(d/'startup_recording_budget_failure.json',prior)
    d.mkdir(parents=True,exist_ok=a.resume_startup);started=time.perf_counter();guard=MemoryGuard(4*1024**3,2*1024**3)
    report=dict(status='RUNNING',matlab_launches=0,fit_performed=False,physiology_calibrated=False,
        new_neural_seconds=0.,backgrounds={},reference='fixed dark anchor 0A; no per-background reanchor',
        fully_adapted=False,graph_or_active_channels_changed=False)
    save(d/'result.json',report)
    def boundary(*unused):
        if not guard.check() or time.perf_counter()-started>1500 or (d/'STOP').exists():raise ContractError('Rod short regression boundary: 1500s/4GiB/STOP')
    class Runner(HybridRunner):
        def advance(self,*args,**kwargs):
            boundary();kwargs.setdefault('state_observer',boundary);return super().advance(*args,**kwargs)
    solver=SolverConfig(method='DOP853',rtol=1e-7,atol=1e-10,max_step=.0001,input_interpolation='zero_order_hold')
    try:
        for rate in (1.,20.,5.):
            cfg=RunConfig.project_default(ROOT);light=replace(cfg.light_context(),rod_background_rstar_s=rate)
            compiled,_,audit=atlas_population_network(ROOT,duration=.5,light_context=light,
                execution_profile=cfg.execution_profile,configuration_path=cfg.circuit_configuration_path,
                owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,seed=cfg.seed)
            assert len(compiled.components)==34
            assert dict(compiled.components['rod'].native_model.p)==rod_source_values(ROOT)
            runner=Runner(compiled,compute_backend='python');state=runner.initialize(solver)
            bg=constant_background_channels(compiled,light,.5,interpolation='zero_order_hold')
            y=runner._flat(state);incoming,out=runner.model.evaluate(0.,y,InputBundle({k:v.at(0.) for k,v in bg.items()}))
            c=compiled.components['rod_membrane'];rp=values(c);b=runner.model.local_blocks('rod_membrane',y)
            I=incoming['rod_membrane']['current'].values
            np.testing.assert_array_equal(I,out['rod','current'].values*1e-12)
            np.testing.assert_array_equal(rp['current_anchor'],0.)
            np.testing.assert_allclose(b[0],rp['EL']-I/rp['gL'],atol=1e-15,rtol=0)
            np.testing.assert_allclose(b[1],expit((b[0]-rp['release_half'])/rp['release_slope']),atol=1e-15,rtol=0)
            rhs=c.derivative(0.,b,incoming['rod_membrane'])
            assert max(float(abs(q.values).max()) for q in rhs)<1e-10
            contact=compiled.components['individual_rod_RBC_contacts'];cb=runner.model.local_blocks('individual_rod_RBC_contacts',y)
            cp=values(contact);release=incoming['individual_rod_RBC_contacts']['release'].values
            np.testing.assert_array_equal(cp['drive0'],release)
            np.testing.assert_allclose(cb[0],expit(-cp['hill']*np.log(release/cp['K'])),rtol=0,atol=1e-15)
            assert float(abs(contact.derivative(0.,cb,incoming['individual_rod_RBC_contacts'])[0].values).max())<1e-10
            row=dict(anchor_A=stats(rp['current_anchor']),current_pA=stats(I*1e12),initial_voltage_mV=stats(b[0]*1000),
                initial_release=stats(b[1]),initial_TRPM1_gate=stats(cb[0]),
                maximum_rod_membrane_RHS=float(max(abs(q.values).max() for q in rhs)),
                source_group_unchanged=True,pA_to_A_applied_once=True,TRPM1_initial_state_uses_actual_release=True,
                frozen_geometry_verified=audit['atlas_runtime']['frozen_typed_geometry'])
            report['backgrounds'][str(rate)]=row
            if rate!=5.:
                plan=RecordingPlan(states=(),ports=('rod.current','rod_membrane.voltage','rod_membrane.release'),max_bytes=64*1024**2)
                end,result=runner.advance(state,bg,.005,sample_times=[0.,.005],recording_plan=plan)
                for port in ('rod.current','rod_membrane.voltage','rod_membrane.release'):
                    v=result.outputs_with_units[port].values
                    np.testing.assert_allclose(v[:,0],v[:,-1],rtol=0,atol=2e-12)
                row['full_network_5ms_constant_input_pass']=True;report['new_neural_seconds']+=.005
            else:
                seed=seed_background_state(runner,state,bg);state,_=balance_rgc_initial_voltages(runner,seed.state,bg)
                ports=[]
                for cid in ('rod','rod_current_SI','rod_membrane','individual_rod_RBC_contacts','rod_BC','AII_AC','RBC_AII_AMPA','AII_FMB_glycine','AII_DB1_glycine'):
                    ports.extend(cid+'.'+key for key in compiled.components[cid].outputs)
                ports+=['midget_RGC_ON.voltage','midget_RGC_OFF.voltage']
                plan=RecordingPlan(states=(),ports=tuple(ports),max_bytes=128*1024**2)
                state,_,_=record_segment(runner,state,bg,.1,d/'wash_in',plan,label='FIXED_DARK_ROD_WASH')
                report['new_neural_seconds']+=.1;prepared=runner.checkpoint(state)
                (d/'prepared_checkpoint.json').write_text(prepared,encoding='utf8');traces={}
                for name in ('flash','gray'):
                    state=runner.restore(prepared);assert runner.checkpoint(state)==prepared;channels=dict(bg)
                    if name=='flash':
                        ch=bg['rod_light'];base=ch.at(.1)
                        def evaluate(t):return Quantity(base.values*(1.2 if .15<=t<.25 else 1.),base.unit,base.meaning,base.evidence_tag)
                        channels['rod_light']=replace(ch,evaluator=evaluate,breakpoints_seconds=(.15,.25),
                            input_contract=dict(ch.input_contract,declared_constant_background=False,rod_only=True,cone_input_held=True,
                                engineering_increment_fraction=.2,calibrated_visual_contrast=False))
                    state,trace,meta=record_segment(runner,state,channels,.4,d/name,plan,label='FIXED_DARK_ROD_'+name.upper())
                    (d/name/'final_checkpoint.json').write_text(runner.checkpoint(state),encoding='utf8');traces[name]=trace
                    report['new_neural_seconds']+=.3;save(d/'result.json',report)
                directions={};arrays={};signs={'rod.current':1,'rod_membrane.voltage':-1,'rod_membrane.release':-1,
                    'individual_rod_RBC_contacts.gate':1,'rod_BC.voltage':1,'rod_BC.release':1,'AII_AC.voltage':1,
                    'AII_AC.release':1,'RBC_AII_AMPA.conductance':1,'AII_FMB_glycine.conductance':1,'AII_DB1_glycine.conductance':1,
                    'midget_RGC_ON.voltage':1,'midget_RGC_OFF.voltage':-1}
                for port,sign in signs.items():
                    delta=traces['flash'][port]-traces['gray'][port];np.testing.assert_allclose(delta[:,:50],0.,rtol=0,atol=2e-12)
                    mean=delta.mean(axis=0);peak=int(np.argmax(abs(mean)));tail=float(mean[200:].mean())
                    directions[port]=dict(mean_delta_last100ms=tail,peak_signed_delta=float(mean[peak]),peak_time_after_flash_on_seconds=float(traces['flash']['time'][peak]-.15),
                        expected_direction_sign=sign,observed_expected_direction=bool(sign*tail>0))
                    arrays[port+'.delta']=delta
                np.savez_compressed(d/'flash_gray_deltas.npz',time=traces['flash']['time'],**arrays)
                comparison={}
                for cid in ('rod_membrane','rod_BC','AII_AC','midget_RGC_ON','midget_RGC_OFF'):
                    port=cid+'.voltage';v=traces['gray'][port];scale=1e-10+1e-7*float(abs(v).max())
                    peak=abs(directions[port]['peak_signed_delta'])
                    comparison[cid]=dict(peak_mean_voltage_delta_V=peak,requested_atol_plus_rtol_scale_V=scale,
                        ratio=peak/scale,gray_voltage_mV=float(v[:,-1].mean()*1000),diagnostic_only_not_global_error_bound=True)
                report.update(rod_chain_directions=directions,all_expected_directions=all(v['observed_expected_direction'] for v in directions.values()),
                    voltage_numerical_scale_diagnostic=comparison,
                    downstream_amplitude_status='Directions agree in recorded differences; small AII/RGC voltage deltas are below the requested local tolerance scale, so this short run does not certify resolved neural output or physiological amplitude',
                    claim='Fixed reference, actual-background initial states, units and short numerical integration pass; downstream sign agreement alone is not an amplitude or sensitivity validation',
                    response_scope='300ms after100ms own wash; relative engineering direction only, no complete peak/recovery or amplitude sufficiency claim',
                    assembly_hash=compiled.assembly_hash,accepted_RGC_types=[v['component_id'] for v in compiled.accepted_rgc_output_defaults])
            save(d/'result.json',report)
        voltages=[report['backgrounds'][str(x)]['initial_voltage_mV']['mean'] for x in (1.,5.,20.)]
        releases=[report['backgrounds'][str(x)]['initial_release']['mean'] for x in (1.,5.,20.)]
        assert voltages[0]>voltages[1]>voltages[2] and releases[0]>releases[1]>releases[2]
        report.update(status='PASS_FIXED_DARK_REFERENCE_SHORT_REGRESSION' if report['all_expected_directions'] else 'COMPLETE_FIXED_DARK_REFERENCE_DOWNSTREAM_DIRECTION_FAILURE',cross_background_DC_retained=True,
            downstream_direction_regression_pass=report['all_expected_directions'],wall_seconds=time.perf_counter()-started,
            maximum_RSS_bytes=guard.peak_rss_bytes,policy_sha256=sha(ROOT/'configurations/rod_fixed_dark_reference_v1.json'))
        save(d/'result.json',report);print(json.dumps({k:v for k,v in report.items() if k!='backgrounds'}),flush=True)
    except Exception as exc:
        report.update(status='STOPPED_EXACT_BLOCKER',error_type=type(exc).__name__,error=str(exc),wall_seconds=time.perf_counter()-started)
        save(d/'result.json',report);raise


if __name__=='__main__':main()
