"""One paired900ms held-Cone-release assay using unchanged inner equations."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from types import FunctionType,SimpleNamespace
import hashlib,json,os,sys,time,traceback
import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.cross_position_context_v1 import load_background
from tools.run_cross_position_rgc_v1 import LocalBC
from tools.durable_research_runtime_v1 import DurableRun
from retina_sim.h1_h2_background import initialize_joint_background
from retina_sim.rgc_functional_v1 import RGCFunctionalV1
from retina_sim.spatial_group_model import BC,RGC
from retina_sim.group_inner_dynamics import BC_MECH
from retina_sim.group_time_monitor import frozen_budget
from retina_sim.group_time_storage import save_bundle
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.contracts import canonical_hash

OUT=ROOT/'reports/retinotopic_rf_revision/SPATIAL_POOLING_CANDIDATE_V1'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
    from retina_sim.parameter_system import data
    with Path(path).open('x',encoding='utf8') as f:json.dump(data(value),f,ensure_ascii=False,indent=2,allow_nan=False)
def progress(value):
    path=OUT/'FUNCTION_PROGRESS.json';temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=True),encoding='utf8');temporary.replace(path)
def file_sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def initialize(assembly,parent,original_factory,release,location,arm,guard):
    initialization_start=time.perf_counter();resource_guard=guard
    def guard():
        resource_guard()
        if time.perf_counter()-initialization_start>170:raise TimeoutError('Original170s initialization guard')
    model_matrices={};identities={}
    if arm=='candidate':
        for source in read(OUT/'STRUCTURE_RESULT.json')['routes']:
            if source['location']!=location:continue
            if not source['dynamic_local_closure']:raise RuntimeError('BLOCKED_GEOMETRY_SUPPORT: new chemical exterior dependency')
            p=OUT/'structure'/location/source['route']/'group_candidate_local.npz'
            model_matrices[source['route']]=sparse.load_npz(p);identities[source['route']]=file_sha(p)
    def factory(a,parameters):
        model=original_factory(a,parameters)
        for name,matrix in model_matrices.items():
            if matrix.shape!=model.mean[name].shape:raise ValueError('Candidate local layout differs')
            model.mean[name]=matrix
        model.identity=canonical_hash(dict(parent_inner=model.identity,spatial_candidate='RETINOTOPIC_GEOMETRY_ANCHORED_POOLING_V1',
            location=location,arm=arm,matrices=identities,physiology_parameters_unchanged=True))
        return model
    assembly.runtime_factories=dict(assembly.runtime_factories,inner=factory)
    def binding(a,seed,root,protocol):
        descriptor=dict(schema='geometry-anchored-pooling-interface-binding-1',location=location,arm=arm,
            source_parent_seed=canonical_hash(parent),matrices=identities,parameters=canonical_hash(seed['inner_parameters']),
            initial_state=canonical_hash(seed['states']),spike_prehistory=canonical_hash(seed['spike_states']),
            FIT_READY=False,background_qualified=False,production_adopted=False)
        return SimpleNamespace(identity=canonical_hash(descriptor),descriptor=descriptor)
    private=dict(initialize_joint_background.__globals__)
    private.update(H1H2Native=lambda a:assembly.lateral_native_factory(a,bound=False),bind_parameters=binding,
        MODEL='ENGINEERING_SPATIAL_POOLING_CANDIDATE',INITIALIZATION='geometry_anchored_pooling_independent_inner_equilibrium_v1')
    initializer=FunctionType(initialize_joint_background.__code__,private,'initialize_spatial_pair',initialize_joint_background.__defaults__,initialize_joint_background.__closure__)
    initializer.__kwdefaults__=dict(initialize_joint_background.__kwdefaults__ or {})
    protocol=dict(schema='cross-position-held-background-initialization-1',clock_origin_seconds=0.,input_origin_seconds=0.,stop_seconds=.9,BC_RGC_period_seconds=.001,rod_step_seconds=.0001,initial_time_seconds=0.,recording_step_seconds=.001)
    guard();seed,b=initializer(assembly,parent,ROOT,protocol,guard=guard)
    assert canonical_hash(seed['inner_parameters'])==canonical_hash(parent['inner_parameters'])
    model=factory(assembly,seed['inner_parameters']);state={k:seed['states'][k].copy() for k in model.state_keys}
    outputs=model.outputs(0.,state,release)
    # The initializer must have used this exact frozen native background.
    for pop in BC:
        route=next(r for r in model.routes if r.pre=='cones' and r.post==pop)
        expected=model.mean[route.name]@release-model.par(pop,'baseline')
        for j in range(3):np.testing.assert_allclose(state[pop,BC_MECH,'history_'+str(j)],expected,rtol=0,atol=1e-12)
    after,_,_=model.commit_ticks(0.,state,cone_release=release,bc_tick=True)
    history_error=max(float(np.max(abs(after[k]-state[k]),initial=0)) for k in state if k[1]==BC_MECH and k[2].startswith('history_'))
    if history_error>1e-12:raise ValueError('Independent initialization/history mismatch')
    rhs=model.rhs(0.,state,release);normalized=[]
    for block in model.blocks:
        k=(block.population,block.mechanism,block.name);budget=frozen_budget(state[k],block.unit)[2]
        normalized.extend(np.abs(rhs[k])*.0001/budget)
    max_initial=float(np.max(normalized,initial=0))
    if max_initial>1:raise ValueError('Independent background exceeds frozen state budget: '+str(max_initial))
    destination=OUT/'function_v2'/location/arm;destination.mkdir(parents=True)
    save_bundle(destination/'initial_seed',kind='source_construction_seed',identity=dict(binding=b.identity,model=model.identity),payload=seed,
        qualification=dict(default_adopted=False,fit_eligible=False,FIT_READY=False,background_qualified=False,production_adopted=False,scope='Inner paired assay only; frozen original S0 ConeHC held; not full chain checkpoint'))
    write(destination/'INITIALIZATION.json',dict(binding=b.identity,model_identity=model.identity,matrices=identities,
        all_inner_states_independently_reconstructed=True,outer_S0_reused_without_changes=True,
        upstream_HC='Existing per-position S0 baseline; latest native-only HC spatial revision not silently adopted',
        same_exact_initializer_code=True,initializer_source_sha256=file_sha(ROOT/'retina_sim/h1_h2_background.py'),
        initial_RHS_one100us_step_over_frozen_budget=max_initial,BC_history_error=history_error,
        parameter_sha256=canonical_hash(seed['inner_parameters']),old_seed_overwritten=False))
    return model,seed,destination


def simulate(assembly,model,seed,release,destination,location,arm,precision,memory,journal):
    out=destination/precision;out.mkdir();started=time.perf_counter();nfev=0
    def guard():
        if time.perf_counter()-started>820:raise TimeoutError('Original coupled820s process guard')
        if not memory.check():raise MemoryError(str((memory.stop_reason,memory.last)))
    layer=object.__new__(RGCFunctionalV1);layer.model=model;layer.populations=tuple(p for p in RGC if model.counts[p])
    bc=LocalBC(model,seed,release);layer.ac=SimpleNamespace(bc=bc,assembly=assembly);layer.origin=bc.origin;layer.fields=layer._fields()
    for pop in BC:
        for name,unit in [('voltage','V'),('terminal_release','dimensionless')]:layer.fields[pop+'__'+name]=dict(population=pop,unit=unit,count=model.counts[pop],meaning='Original BC state; unchanged equation')
    # Existing RGC observe_ports checks its own exact50-field contract.
    all_fields=layer.fields;layer.fields={k:v for k,v in all_fields.items() if not k.startswith(tuple(p+'__' for p in BC))}
    state=layer.initial();auxiliary=layer.initial_spike_states();saved={f:[] for f in all_fields}
    laws={p:[] for p in layer.populations};events={p:[] for p in layer.populations};sampled={p:[] for p in layer.populations}
    rtol,atol,step=(1e-10,1e-13,.000025) if precision=='tight' else (1e-6,1e-9,.0001)
    write(out/'IDENTITY.json',dict(model=model.identity,initial_seed_sha256=canonical_hash(seed),fields=all_fields,
        location=location,arm=arm,precision=precision,rtol=rtol,atol=atol,max_step=step,
        nstate=model.nstate,state_fields={'__'.join((b.population,b.mechanism,b.name)):dict(unit=b.unit,count=b.count) for b in model.blocks},
        protocol_source='tools/run_cross_position_rgc_v1.py::LocalBC.inputs; original900ms,0.1ms event clocks,1ms recordings',
        Cone_release_windows_ms=[[50,200,.9],[350,500,1.1]],Rod_window_ms=[600,750,2.],
        no_new_equations=True,members_applied_once=True,spike_events_are_group_objects=True,FIT_READY=False,background_qualified=False,production_adopted=False))
    def capture(index,cone):
        values=layer.observe(bc.origin+index*.0001,state,cone)
        for pop in BC:
            values[pop+'__voltage']=state[pop,BC_MECH,'V'];values[pop+'__terminal_release']=state[pop,BC_MECH,'terminal_release']
        for name,value in values.items():saved[name].append(np.array(value,copy=True))
        for p in layer.populations:laws[p].append(np.stack([state[p,'RGC_expected_history','history_probability_'+str(j)] for j in range(8)],axis=1))
    def packet():
        result=dict(time_seconds=np.arange(len(next(iter(saved.values()))))*.001,**{f:np.asarray(v).T for f,v in saved.items()})
        for p in layer.populations:
            result[p+'__history_law_after_tick']=np.asarray(laws[p]).transpose(1,0,2)
            result[p+'__sampled_bin_counts']=np.asarray(sampled[p]).T
            result[p+'__event_time_seconds']=np.asarray([e['time_seconds']-bc.origin for e in events[p]])
            result[p+'__event_group_index']=np.asarray([e['cell_index'] for e in events[p]],dtype=np.int64)
        return result
    for index in range(9000):
        if index%100==0:guard()
        cone,rod=bc.inputs(index);tick=index%10==0
        state,following,drawn=layer.commit(index,state,cone,rod,auxiliary if tick else None)
        if tick:
            auxiliary=following
            for p in layer.populations:sampled[p].append(auxiliary[p]['last_bin_counts']);events[p]+=drawn[p]
            capture(index,cone)
        left,right=bc.origin+index*.0001,bc.origin+(index+1)*.0001
        sol=solve_ivp(layer.rhs,(left,right),model.pack(state),method='DOP853',rtol=rtol,atol=atol,max_step=step,first_step=min(step,right-left),t_eval=[right])
        if not sol.success:raise ValueError(sol.message)
        state=model.unpack(sol.y[:,-1]);nfev+=sol.nfev
        if (index+1)%1000==0:
            current=dict(location=location,arm=arm,precision=precision,completed_ms=(index+1)/10,wall_seconds=time.perf_counter()-started,pid=os.getpid())
            progress(current);journal.heartbeat(phase=location+'_'+arm+'_'+precision);print(json.dumps(current),flush=True)
            if (index+1)%3000==0:np.savez_compressed(out/f'PARTIAL_{(index+1)//10:04d}.npz',**packet())
    index=9000;cone,rod=bc.inputs(index);state=bc.endpoint(index,state,cone,rod);capture(index,cone)
    data=packet();np.savez_compressed(out/'RESPONSE.npz',**data)
    np.savez_compressed(out/'STATE_END.npz',inner_flat=model.pack(state),**{'__'.join(k):v for k,v in state.items()})
    write(out/'END_HISTORY.json',auxiliary)
    write(out/'RESOURCE.json',dict(status='PASS',wall_seconds=time.perf_counter()-started,nfev=nfev,peak_rss_bytes=memory.peak_rss_bytes,no_automatic_retry=True))


def main():
    started=time.perf_counter();journal=DurableRun(OUT/'function_runtime_v2','spatial_pooling_paired_probe');memory=MemoryGuard(24*1024**3,8*1024**3)
    source=read(ROOT/'reports/external_video_two_positions_4s_20261029_v1/PREPARATION.json')['bindings']
    def guard():
        if not memory.check():raise MemoryError(str((memory.stop_reason,memory.last)))
    try:
        for location in ['near','far']:
            progress(dict(location=location,phase='loading_existing_S0',pid=os.getpid()));guard()
            assembly,parent,_=load_background(Path(source[location]['prepared']),Path(source[location]['background']))
            with np.load(Path(source[location]['background'])/'BACKGROUND.npz',allow_pickle=False) as z:release=z['cone_release'].copy()
            original_factory=assembly.runtime_factories['inner']
            for arm in ['legacy','candidate']:
                phase_start=time.perf_counter()
                model,seed,destination=initialize(assembly,parent,original_factory,release,location,arm,guard)
                write(destination/'INITIALIZATION_RESOURCE.json',dict(wall_seconds=time.perf_counter()-phase_start,peak_rss_bytes=memory.peak_rss_bytes))
                for precision in ['standard','tight']:simulate(assembly,model,seed,release,destination,location,arm,precision,memory,journal)
        write(OUT/'FUNCTION_RESULT.json',dict(status='PAIRED_RUNS_COMPLETE_PENDING_ANALYSIS',wall_seconds=time.perf_counter()-started,peak_rss_bytes=memory.peak_rss_bytes,
            runs=8,original_protocol_reused=True,no_model_or_gain_changes=True,FIT_READY=False,background_qualified=False,production_adopted=False))
        journal.finish('PASS',phase='paired_runs_complete')
    except BaseException as e:
        write(OUT/'FUNCTION_FAILURE_ATTEMPT2.json',dict(error=str(e),traceback=traceback.format_exc(),no_automatic_retry=True))
        journal.finish('FAILED',phase='stopped');raise


if __name__=='__main__':main()
