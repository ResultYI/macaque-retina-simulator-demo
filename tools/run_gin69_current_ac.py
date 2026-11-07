"""Continuous current-model GIN69 execution and the existing frozen voltage readout."""
from copy import deepcopy
from pathlib import Path
from types import MethodType, FunctionType
import argparse
import json
import os
import time
import traceback

import numpy as np

from retina_sim.aii_functional import FunctionalACRunner, S_KEY, V_KEY
from retina_sim.ac_vs_compiled import CompiledACVSInnerRHS
from retina_sim.ac_vs_model import DirectACVSRHS
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_time_storage import load_bundle, save_bundle
from retina_sim.group_inner_dynamics import RGC_MECH, BC_MECH
from retina_sim.spatial_group_graph import digest
from tools.continue_bc_ac_calibration import ROOT, OUT, DEST, read, guard, RetinaBuilder
from tools.cell6_raw_runner_v1 import Cell6Runner, pack_without_state_cell
from tools.gin69_frame_index_support_v2 import IndexedGIN69Runner
from tools.validate_ac_vs_timescale import calendar
from tools.durable_research_runtime_v1 import atomic_json


class CurrentRunner(FunctionalACRunner):
    frame_at = IndexedGIN69Runner.frame_at
    configure_native_dense = Cell6Runner.configure_native_dense
    configure_inner_dense = Cell6Runner.configure_inner_dense

    def rod_sample_at(self,t,side='right'):
        if t==self.stop and side=='left':
            origin=self.protocol['clock_origin_seconds']
            sampled=origin+(int(round((t-origin)/.0001))-1)*.0001
            return sampled,self.frame_at(sampled,'right')
        return super().rod_sample_at(t,side)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.numeric_profile = dict(inner_rtol=1e-6,inner_atol=1e-9)
        self.configure_native_dense(False)
        self.configure_inner_dense()
        self.inner.pack = MethodType(pack_without_state_cell,self.inner)
        candidate = CompiledACVSInnerRHS(self.inner,repo_root=ROOT)
        reference = DirectACVSRHS(self.inner)
        initial = self.inner.pack(self.inner_state)
        errors = []
        for sign in (0.,-1.,1.):
            value = initial.copy()
            for key,section in self.inner.slices.items():
                if key[-1]=='V':
                    value[section] += sign*.001
                elif key==S_KEY:
                    value[section] = np.clip(value[section]+sign*.01,0.,1.)
            before = value.copy()
            expected,actual = reference.rhs_flat(0.,value),candidate.rhs_flat(0.,value)
            np.testing.assert_array_equal(value,before)
            np.testing.assert_allclose(actual,expected,rtol=5e-14,atol=2e-13)
            errors.append(float(np.max(abs(actual-expected))))
        self.compiled = self.inner.compiled_rhs = candidate
        self.compilation_evidence = dict(max_RHS_error=max(errors),probe_states=3,
            rtol=5e-14,atol=2e-13,implementation_sha256=digest(ROOT/'retina_sim/ac_vs_compiled.py'))
        self.identity = canonical_hash(dict(parent=self.identity,compiled=candidate.identity,
            dense='existing_Cell6_increment_FP64',source=digest(Path(__file__))))


def load_current():
    bg = read(DEST/'BACKGROUND.json')
    if digest(DEST/'seed/manifest.json')!=bg['seed_manifest_sha256']:
        raise ContractError('Current source seed changed')
    model = RetinaBuilder.build_research_ac(ROOT)
    if model.identity!=bg['model_identity']:
        raise ContractError('Frozen current AC model changed')
    seed = load_bundle(DEST/'seed',expected_kind='source_construction_seed',expected_identity=bg['identity'])['payload']
    if canonical_hash(model.seed['inner_parameters'])!=canonical_hash(seed['inner_parameters']):
        raise ContractError('Source rebind changed frozen parameters')
    if canonical_hash(model.assembly.aii_mapping)!=bg['AII_mapping_sha256']:
        raise ContractError('Source rebind changed AII gains')
    original_factory = model.assembly.runtime_factories['native']
    def native_factory(assembly,*,bound=True,**kwargs):
        native = original_factory(assembly,bound=False,**kwargs)
        if bound:
            native.boundary = seed['h1_h2_model']['boundary']
        return native
    model.assembly.runtime_factories = dict(model.assembly.runtime_factories,native=native_factory)
    model.seed = seed
    protocol = deepcopy(model.protocol)
    protocol.update(stop_seconds=45.,input_frame_indices=list(range(6750)),checkpoint_seconds=.5)
    return model,protocol,bg


def queue(arm,t,side):
    evaluation,_,_ = arm._native_evaluation(t,side)
    if not np.isfinite(evaluation.rhs).all():
        raise ContractError('Nonfinite native derivative')
    arm._delay_append(t,evaluation.outputs['Vh'])


def advance(arm,end,flags,*,full=False):
    arm.ensure_native(end,'left'); arm.advance_inner(arm.inner_time,end)
    side = 'left' if flags['frame'] or end==arm.stop else 'right'
    if full:
        arm.snapshot(end,side,queue_node=True)
    else:
        queue(arm,end,side)
    if flags['rod'] or flags['bc']:
        arm.commit(end,rod=flags['rod'],bc=flags['bc'],rgc=flags['bc'],
            rod_sample_time=max(0.,end-.0001+1e-13) if flags['rod'] else None)
    side = 'left' if end==arm.stop else 'right'
    if full:
        arm.snapshot(end,side,queue_node=True)
    else:
        queue(arm,end,side)
    arm.inner.validate(arm.inner_state)
    gate = arm.inner_state[S_KEY]
    if np.any(gate<0) or np.any(gate>1):
        raise ContractError('AII output gate left [0,1]')
    arm.ack(end)


def equivalence(arm,model,protocol):
    """Check snapshot omission and complete continuation on the actual new source."""
    started = time.perf_counter()
    queue(arm,0.,'right'); arm.initial_commit(); queue(arm,0.,'right'); arm.ack(0.)
    saved = arm.checkpoint_payload()
    other = CurrentRunner.from_checkpoint(model.assembly,model.seed,protocol,saved,guard=guard)
    greatest = 0.
    try:
        for end,flags in calendar(protocol):
            if end>.002:
                break
            advance(arm,end,flags)
            advance(other,end,flags,full=True)
            for key,value in arm.inner_state.items():
                greatest = max(greatest,float(np.max(abs(value-other.inner_state[key]),initial=0.)))
        left,right = arm.checkpoint_payload(),other.checkpoint_payload()
        categories = ('native','inner','spike_states','clocks','input_position','observation_delay','readout_cursor')
        for key in categories:
            if canonical_hash(left[key])!=canonical_hash(right[key]):
                raise ContractError('Compact observation or restoration changed '+key)
        if greatest!=0:
            raise ContractError('Compact observation changed physical trajectory')
        return dict(physical_max_abs_difference=greatest,exact_categories=list(categories),
            source_seconds=.002,wall_seconds=time.perf_counter()-started,compiled=arm.compilation_evidence)
    finally:
        other.close()


def run():
    started = time.perf_counter()
    result_path = DEST/'NEURAL_RESULT.json'
    if result_path.exists() and read(result_path)['status']=='COMPLETE':
        raise FileExistsError('Completed current trajectory retained')
    model,protocol,bg = load_current()
    pop = 'parasol_RGC_OFF'
    rows = np.flatnonzero(model.assembly.parent_population_rows[pop]==7730)
    if len(rows)!=1 or model.assembly.populations[pop].counts[rows[0]]!=1:
        raise ContractError('Frozen 69#6 target row/member identity differs')
    row = int(rows[0])
    arrays = {name:np.empty(450001) for name in ('V','gE','gI')}
    expected = np.empty(45000)
    recorded = 0
    arm = CurrentRunner(model.assembly,model.seed,protocol,guard=guard)
    def record(t):
        nonlocal recorded
        j = round(t*10000)
        if j!=recorded:
            raise ContractError('Native voltage sample omitted or repeated')
        state = arm.inner_state
        values = arm.inner.outputs(t,state)
        for name,port in [('V','voltage'),('gE','gE'),('gI','gI')]:
            arrays[name][j] = values[pop,port][row]
        if j%10==0 and j<450000:
            expected[j//10] = values[pop,'expected_rate_Hz'][row]
        recorded += 1
    try:
        # Retain the first nodes while proving exact complete checkpoint recovery.
        queue(arm,0.,'right'); arm.initial_commit(); queue(arm,0.,'right'); arm.ack(0.); record(0.)
        saved = arm.checkpoint_payload()
        other = CurrentRunner.from_checkpoint(model.assembly,model.seed,protocol,saved,guard=guard)
        checks = dict(compiled=arm.compilation_evidence)
        for end,flags in calendar(protocol):
            if end<=arm.inner_time:
                continue
            advance(arm,end,flags)
            if other is not None:
                advance(other,end,flags,full=True)
                for key,value in arm.inner_state.items():
                    np.testing.assert_array_equal(value,other.inner_state[key])
                if end>=.002:
                    left,right = arm.checkpoint_payload(),other.checkpoint_payload()
                    categories = ('native','inner','spike_states','clocks','input_position','observation_delay','readout_cursor')
                    for key in categories:
                        if canonical_hash(left[key])!=canonical_hash(right[key]):
                            raise ContractError('Compact observation/checkpoint changed '+key)
                    checks.update(exact_categories=list(categories),source_seconds=end,physical_max_abs_difference=0.)
                    atomic_json(DEST/'EXECUTION_PARITY.json',checks)
                    other.close(); other=None
            if flags['rod'] or end==45.:
                record(end)
                j = round(end*10000)
                if j%5000==0:
                    progress = dict(status='RUNNING',pid=os.getpid(),source_seconds=end,target_seconds=45.,
                        recorded=recorded,wall_seconds=time.perf_counter()-started,source_identity=bg['identity'])
                    atomic_json(DEST/'PROGRESS.json',progress); print(json.dumps(progress),flush=True)
                    arm.metrics['inner_counters'].clear()
                if j%50000==0:
                    checkpoint = DEST/('CHECKPOINT_'+str(j))
                    payload = arm.checkpoint_payload(recording=dict(recorded=recorded,arrays={k:v[:recorded].copy() for k,v in arrays.items()},
                        expected=expected[:j//10].copy()))
                    save_bundle(checkpoint,kind='validation_evidence',identity=bg['identity'],payload=payload,
                        qualification=dict(default_adopted=False,fit_eligible=False))
                if end in (30.,45.):
                    path = DEST/f'PREFIX_{int(end)}s.npz'
                    np.savez_compressed(path,time_source_s=np.arange(j+1)*.0001,
                        **{k:v[:recorded] for k,v in arrays.items()},expected_rate_base_Hz=expected[:j//10])
                    atomic_json(DEST/f'PREFIX_{int(end)}s.json',dict(status='COMPLETE_CONTINUOUS_PREFIX',
                        seconds=end,sha256=digest(path),background_identity=bg['identity'],
                        target_group=7730,source_cell='69#6',members=1,split_sha256=bg['split_sha256']))
        if recorded!=450001 or not all(np.isfinite(v).all() for v in arrays.values()):
            raise ContractError('Current continuous trajectory incomplete')
        atomic_json(result_path,dict(status='COMPLETE',source_seconds=45.,wall_seconds=time.perf_counter()-started,
            identity=bg['identity'],model_identity=bg['model_identity'],target_row=row,recorded=recorded,
            frozen_upstream=True,AC_calibrated=False,sealed_accessed=False,checks=checks,
            source_sha256=digest(Path(__file__)),FIT_READY=False))
    finally:
        arm.close()


def source_case(seconds):
    from tools.gin69_voltage_fit_support_v2 import counts_prefix, model
    neural = read(DEST/'NEURAL_RESULT.json')
    if neural['status']!='COMPLETE':
        raise ContractError('Only completed current-model trajectory admitted')
    receipt = read(DEST/f'PREFIX_{seconds}s.json'); path = DEST/f'PREFIX_{seconds}s.npz'
    if digest(path)!=receipt['sha256'] or receipt['split_sha256']!=digest(OUT/'SPLIT_CONTRACT.json'):
        raise ContractError('Current source prefix or frozen split changed')
    with np.load(path) as z:
        voltage = z['V'].copy()
        np.testing.assert_array_equal(z['time_source_s'],np.arange(seconds*10000+1)*.0001)
    counts,evidence = counts_prefix(seconds*1000)
    return dict(counts=counts,stimulus_features=model.stimulus_features(voltage[:-1]),
        history_features=model.history_features(counts)),dict(neural_array_sha256=receipt['sha256'],neural_identity=neural,
        observations=evidence,voltage_unit='V',source_field='V',pre0='constantV0; zero readout spike history as inherited contract')


def fit():
    from tools.gin69_voltage_fit_support_v2 import execute
    split = DEST/'SPLIT_CONTRACT.json'
    if split.exists():
        assert digest(split)==digest(OUT/'SPLIT_CONTRACT.json')
    else:
        split.write_bytes((OUT/'SPLIT_CONTRACT.json').read_bytes())
    namespace = dict(execute.__globals__,OUT=DEST,source_case=source_case)
    function = FunctionType(execute.__code__,namespace,execute.__name__,execute.__defaults__,execute.__closure__)
    for mode in ('train','evaluate'):
        output = DEST/mode; output.mkdir(exist_ok=True)
        function(output,guard,mode)


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode',choices=['run','fit'])
    args = parser.parse_args()
    began = time.perf_counter()
    owner = DEST/('OWNER_'+args.mode+'.json')
    if owner.exists() and read(owner).get('status')=='RUNNING':
        raise RuntimeError('Existing owner must be verified before recovery')
    atomic_json(owner,dict(status='RUNNING',pid=os.getpid(),mode=args.mode))
    try:
        (run if args.mode=='run' else fit)()
    except BaseException:
        atomic_json(owner,dict(status='FAILED',pid=os.getpid(),mode=args.mode,
            error=traceback.format_exc(),wall_seconds=time.perf_counter()-began))
        raise
    else:
        atomic_json(owner,dict(status='EXITED_0',pid=os.getpid(),mode=args.mode,wall_seconds=time.perf_counter()-began))
