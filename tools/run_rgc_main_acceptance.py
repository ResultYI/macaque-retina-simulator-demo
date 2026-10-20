"""One type's own candidate checks on the actual shared-main BC/AC route.

Every functional branch advances the full unchanged network. Replay of the
unmodified samplers is solely a strict regression on actual recorded V.
"""
from pathlib import Path
from dataclasses import replace
import argparse,copy,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_rgc_type_candidate import (RunConfig,atlas_population_network,HybridRunner,constant_background_channels,
    RecordingPlan,MemoryGuard,record_segment,chain_ports,save,sha,seed_background_state,balance_rgc_initial_voltages,SolverConfig,RGCS,load_arrays)
from retina_sim.source_reference_candidate import read_source_candidate,CROSS_TYPE_TARGETS
from retina_sim.calibration_modules import merge_parameter_package
from retina_sim.contracts import ContractError,Quantity,canonical_hash
from retina_sim.rgc_expectation import exact_history_rates,validate_sampler


def channels_for(compiled,bg,cid,kind,fraction,stop):
    pop=compiled.populations[cid].cells;xy=pop.rf_position.values;focus=int(np.argmin(np.linalg.norm(xy-xy.mean(axis=0),axis=1)))
    gid=pop.cell_id[focus];center=xy[focus];direct=set();routes={}
    for edge in compiled.spec.links:
        if edge.target_component!=cid or edge.target_port!='excitation':continue
        adapter=edge.source_component
        source=next(l for l in compiled.spec.links if l.target_component==adapter and l.target_port=='activation')
        bc=source.source_component
        selected={a for a,b,w in zip(source.connections.pre_cell_id,source.connections.post_cell_id,source.connections.weight) if b==gid and w>0}
        link=next(l for l in compiled.spec.links if l.target_component==bc and l.target_port=='drive')
        cone_ids={a for a,b,w in zip(link.connections.pre_cell_id,link.connections.post_cell_id,link.connections.weight) if b in selected and w>0}
        if link.source_component!='outer_native':
            raise ContractError('Current-main probe expects its retained direct cone→history BC drive; no implicit route substitution')
        routes[bc]=dict(BC_ids=sorted(selected),cone_ids=sorted(cone_ids));direct.update(cone_ids)
    cones=compiled.populations['cones'].cells;index={g:j for j,g in enumerate(cones.cell_id)}
    if not direct or not direct<=set(index):raise ContractError('Complete actual direct cone footprint required')
    distance=np.linalg.norm(cones.rf_position.values-center,axis=1)
    radius=max(.015,1.05*max(distance[index[g]] for g in direct));outer=radius+.03
    points=cones.rf_position.values
    if kind.startswith('ring') and (np.any(center-outer<points.min(axis=0)) or np.any(center+outer>points.max(axis=0))):
        raise ContractError('Fixed graph does not cover this cell ring; no spatial expansion or edge compensation')
    mask=(distance>radius)&(distance<=outer) if kind=='ring' else distance<=radius
    if kind in ('S','ring_S'):mask&=np.asarray(cones.subtype)=='S'
    if kind=='LM':mask&=np.isin(cones.subtype,('L','M'))
    if kind=='ring_S':mask=(distance>radius)&(distance<=outer)&(np.asarray(cones.subtype)=='S')
    if not mask.any():raise ContractError('Actual selected cone mask is empty')
    if kind.startswith('ring') and any(mask[index[g]] for g in direct):raise ContractError('Direct cone leaked into surround mask')
    edges=np.array([round(.1+j*.05,10) for j in range(6)]);fractions=[0.,fraction,0.,-fraction,0.]
    contract=dict(component=cid,focus_cell_index=focus,focus_cell_id=gid,focus_RF_mm=center.tolist(),
        center_radius_mm=float(radius),ring_outer_radius_mm=float(outer),actual_direct_routes=routes,
        mask_kind=kind,active_cone_ids=[g for g,on in zip(cones.cell_id,mask) if on],
        fractions=fractions,phase_seconds=.05,extended_gray_after_seconds=.25,
        calibrated_optical_contrast=False,rod_background_held=True,geometry_or_connections_changed=False)
    result={}
    for name,ch in bg.items():
        ext=next(e for e in compiled.spec.external_inputs if e.channel==name);base=ch.at(0.)
        if ext.target_component!='outer_native':result[name]=ch;continue
        def evaluate(t,base=base):
            j=min(max(int(np.searchsorted(edges,t,side='right')-1),0),4)
            return Quantity(base.values*(1+fractions[j]*mask),base.unit,base.meaning,base.evidence_tag)
        result[name]=replace(ch,evaluator=evaluate,breakpoints_seconds=tuple(edges[1:-1]),
            input_contract=dict(ch.input_contract,declared_constant_background=False,engineering_cell_probe=contract))
    return result,contract


def sampler_regression(original,trace,prepared_aux,actual_state,target,*,includes_first_draw):
    checked={}
    for cid in RGCS:
        if cid==target:continue
        n=original.components[cid].native_model;aux=copy.deepcopy(prepared_aux[cid]);nt=len(trace['time'])
        first=0 if includes_first_draw else 1
        if not includes_first_draw:np.testing.assert_array_equal(aux['conditional_rate_Hz'],trace[cid+'.conditional_rate'][:,0])
        for j in range(first,nt):
            aux,_=n.sample_spikes(float(trace['time'][j]),trace[cid+'.voltage'][:,j],aux)
            np.testing.assert_array_equal(aux['conditional_rate_Hz'],trace[cid+'.conditional_rate'][:,j])
        current=actual_state.spike_states[cid]
        for key in ('history','last_bin_counts','rng_state','conditional_rate_Hz'):
            assert canonical_hash(aux[key])==canonical_hash(current[key]),(cid,key)
        checked[cid]=dict(all_native_bin_conditional_intensities_bitwise_equal=True,
            final_history_counts_RNG_bitwise_equal=True,scope='Original unchanged sampler on actual whole-network V; regression, not functional replacement')
    return checked


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--component',choices=CROSS_TYPE_TARGETS,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-startup',action='store_true')
    a=p.parse_args();d=a.output.resolve()
    if a.resume_startup:
        prior=json.loads((d/'result.json').read_bytes())
        if prior.get('new_neural_seconds')!=0. or not prior.get('error','').startswith('CIRCUIT_GEOMETRY:') or any(d.rglob('chunk_*.npz')):
            raise ContractError('Startup recovery requires proof of no neural advance and the repaired profile-label geometry check')
        save(d/'startup_geometry_label_failure.json',prior)
    d.mkdir(parents=True,exist_ok=a.resume_startup);started=time.perf_counter()
    report=dict(status='ASSEMBLY',component=a.component,new_neural_seconds=0.,matlab_launches=0,fit_performed=False,
        production_adoption=False,rod_reference_changed=False,main_route='retained 34-component HistoryBipolarTerminal',branches={})
    save(d/'result.json',report);guard=MemoryGuard(4*1024**3,2*1024**3)
    def boundary(*unused):
        if not guard.check() or time.perf_counter()-started>2400 or (d/'STOP').exists():raise ContractError('Per-type acceptance reached 2400s/4GiB/STOP boundary')
    class Runner(HybridRunner):
        def advance(self,*args,**kwargs):boundary();kwargs.setdefault('state_observer',boundary);return super().advance(*args,**kwargs)
    try:
        cfg=RunConfig.project_default(ROOT)
        original,_,audit=atlas_population_network(ROOT,duration=.85,light_context=cfg.light_context(),execution_profile=cfg.execution_profile,
            configuration_path=cfg.circuit_configuration_path,owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,
            max_spatial_cells=cfg.max_spatial_cells,seed=cfg.seed)
        assert len(original.components)==34 and original.components['diffuse_BC_ON'].mechanism=='HistoryBipolarTerminal'
        recipe,path=read_source_candidate(ROOT,ROOT/'configurations'/(a.component+'_literature_candidate_v1.json'))
        compiled=merge_parameter_package(original,audit['registry'],recipe,applicability=recipe['applicability'],allowed_fields=['alpha','mu','beta'],proposal_path=path)
        assert canonical_hash(original.spec.links)==canonical_hash(compiled.spec.links)
        assert canonical_hash(original.spec.gap_junctions)==canonical_hash(compiled.spec.gap_junctions)
        solver=SolverConfig(method='DOP853',rtol=1e-7,atol=1e-10,max_step=.0001,input_interpolation='zero_order_hold')
        bg=constant_background_channels(compiled,cfg.light_context(),.85,interpolation='zero_order_hold')
        runner=Runner(compiled,compute_backend='python');state=runner.initialize(solver)
        old_runner=HybridRunner(original,compute_backend='python');old_initial=old_runner.initialize(solver)
        old_aux=copy.deepcopy(old_initial.spike_states)
        seed=seed_background_state(runner,state,bg);state,_=balance_rgc_initial_voltages(runner,seed.state,bg)
        plan=RecordingPlan(states=(),ports=chain_ports(compiled),max_bytes=128*1024**2)
        state,warm,meta=record_segment(runner,state,bg,.1,d/'wash_in',plan,label=a.component+'_OWN_WASH_IN')
        sampler_regression(original,warm,old_aux,state,a.component,includes_first_draw=True)
        prepared=runner.checkpoint(state);(d/'prepared_checkpoint.json').write_text(prepared,encoding='utf8');aux=json.loads(prepared)['payload']['spike_states']
        n=validate_sampler(compiled,a.component)
        snapshots={cid:{k:np.asarray(getattr(compiled.components[cid].native_model,k)).tolist() for k in ('alpha','mu','beta','C','gL','EL','EE','EI','history_kernel','dt')} for cid in RGCS}
        save(d/'actual_RGC_parameters.json',snapshots)
        specs=[('center05','S' if a.component=='small_bistratified_S_ON' else 'center',.05,.35),
               ('center20','S' if a.component=='small_bistratified_S_ON' else 'center',.2,.85),
               ('ring20','ring_S' if a.component=='small_bistratified_S_ON' else 'ring',.2,.35)]
        if a.component=='small_bistratified_S_ON':specs.append(('LM20','LM',.2,.35))
        specs.append(('gray',None,0.,.85));rates={};traces={};geometries={};regression={}
        report.update(status='OWN_BRANCHES_RUNNING',assembly_hash=compiled.assembly_hash,base_main_assembly_hash=original.assembly_hash,
            recipe_sha256=sha(path),counts=dict(components=len(compiled.components),cells=sum(len(p.cells.cell_id) for p in compiled.populations.values())),
            changed_fields=['alpha','mu','beta'],new_neural_seconds=.1)
        save(d/'result.json',report)
        blocked={}
        for name,kind,fraction,stop in specs:
            state=runner.restore(prepared);assert runner.checkpoint(state)==prepared
            try:channels,geometry=channels_for(compiled,bg,a.component,kind,fraction,stop) if kind else (bg,None)
            except ContractError as error:
                if kind and kind.startswith('ring'):
                    blocked[name]=str(error);report['branches'][name]=dict(status='BLOCKED_FIXED_SUPPORT',reason=str(error));save(d/'result.json',report);continue
                raise
            if geometry:save(d/(name+'_geometry.json'),geometry);geometries[name]=geometry
            state,trace,segment=record_segment(runner,state,channels,stop,d/name,plan,label=a.component+'_'+name)
            (d/name/'final_checkpoint.json').write_text(runner.checkpoint(state),encoding='utf8')
            regression[name]=sampler_regression(original,trace,aux,state,a.component,includes_first_draw=False)
            r=exact_history_rates(n,trace[a.component+'.voltage'],trace['time'],aux[a.component])
            rates[name]=r['expected_binary_count_rate_Hz'];traces[name]=trace
            np.savez_compressed(d/name/'four_rate_expectations.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
            report['branches'][name]=dict(neural_seconds=stop-.1,wall_seconds=segment['wall_seconds'],all_finite=True,own_checkpoint_exact=True)
            report['new_neural_seconds']+=stop-.1;save(d/'result.json',report)
        responses={};arrays={}
        for name in rates:
            if name=='gray':continue
            nt=rates[name].shape[1];delta=rates[name]-rates['gray'][:,:nt];focus=geometries[name]['focus_cell_index']
            dv=1000*(traces[name][a.component+'.voltage']-traces['gray'][a.component+'.voltage'][:,:nt+1])
            np.testing.assert_allclose(delta[:,:50],0.,rtol=0,atol=1e-8)
            responses[name]=dict(focus_cell_index=focus,focus_cell_id=geometries[name]['focus_cell_id'],
                bright_expected_delta_Hz=float(delta[focus,50:100].mean()),dark_expected_delta_Hz=float(delta[focus,150:200].mean()),
                peak_abs_cell_expected_delta_Hz=float(abs(delta).max()),last100ms_max_abs_cell_expected_delta_Hz=float(abs(delta[:,-100:]).max()),
                peak_abs_cell_delta_V_mV=float(abs(dv).max()))
            arrays[name+'.expected_delta_Hz']=delta;arrays[name+'.voltage_delta_mV']=dv
        sign=-1 if a.component.endswith('_OFF') else 1
        checks=dict(mapping=True,bright_dark=bool(sign*responses['center20']['bright_expected_delta_Hz']>0 and sign*responses['center20']['dark_expected_delta_Hz']<0),
            contrast_ordering=bool(abs(responses['center20']['bright_expected_delta_Hz'])>abs(responses['center05']['bright_expected_delta_Hz']) and abs(responses['center20']['dark_expected_delta_Hz'])>abs(responses['center05']['dark_expected_delta_Hz'])),
            own_mapping_recovery=bool(responses['center20']['last100ms_max_abs_cell_expected_delta_Hz']<=max(.01,.05*responses['center20']['peak_abs_cell_expected_delta_Hz'])),
            single_cell_spatial_or_spectral=bool('ring20' in responses and sign*responses['ring20']['bright_expected_delta_Hz']<0 and sign*responses['ring20']['dark_expected_delta_Hz']>0))
        if 'LM20' in responses:
            checks['single_cell_spatial_or_spectral']=bool(checks['single_cell_spatial_or_spectral'] and responses['LM20']['bright_expected_delta_Hz']<0 and responses['LM20']['dark_expected_delta_Hz']>0)
        report.update(status='COMPLETE_OWN_TYPE_MAIN_ROUTE_ACCEPTANCE',passed_checks=checks,all_checks_passed=all(checks.values()),responses=responses,
            gray_expected_count_rate_Hz=float(rates['gray'].mean()),unmodified_sampler_regression=regression,blocked_checks=blocked,
            physiology_calibrated=False,fully_adapted=False,engineering_recovery_rule='last100ms max cell residual <= max0.01Hz or5%peak; not physiological standard',
            wall_seconds=time.perf_counter()-started,maximum_RSS_bytes=guard.peak_rss_bytes,
            decision='PASS_SCOPED_ENGINEERING_ADOPTION' if all(checks.values()) else 'KEEP_CANDIDATE_FAILED_CHECKS')
        np.savez_compressed(d/'response_arrays.npz',**arrays);save(d/'result.json',report)
        print(json.dumps(dict(component=a.component,status=report['status'],checks=checks,responses=responses,gray_rate_Hz=report['gray_expected_count_rate_Hz'])),flush=True)
    except Exception as e:
        report.update(status='STOPPED_EXACT_BLOCKER',error_type=type(e).__name__,error=str(e),wall_seconds=time.perf_counter()-started)
        save(d/'result.json',report);raise


if __name__=='__main__':main()
