"""CPU-only acceptance-contract regressions; never initialize a real model/GPU."""
from pathlib import Path
import json

import numpy as np
import pytest

from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_scc_protocol import (AcceptanceFailure, BATCH_ID, BudgetLedger, LIMITS,
    REPORT_PARENT, RESOURCE_LIMITS, comparison_status, finite_nonnegative, load_protocol,
    performance_result, read_json, resource_check, write_json)
from tools.validate_group_scc_radau import recording_content_hash
from tools.validate_group_time_candidate import clock_events, sample_grid


def output(root, name):
    return root/REPORT_PARENT/name


def test_real_contract_keeps_full_window_clocks_and_frozen_budget():
    root = Path(__file__).resolve().parents[1]
    p = load_protocol(root)
    assert p['batch_id'] == BATCH_ID
    assert p['candidate']['method'] == 'SCC_RADAU_IIA5_TWO_HALF_HERMITE5'
    assert p['candidate']['maximum_subdivision'] == 6
    assert 'degree' not in p['candidate']
    assert len(clock_events(p)) == 80
    assert (4.997, True, True) in clock_events(p)
    assert (5.0, True, True) in clock_events(p)
    assert len(sample_grid(4.993, 5.001, .0005)) == 17
    assert p['reference']['method'] == 'joint_DOP853'


def test_necessary_source_repair_retains_all_three_budget_totals(tmp_path):
    with BudgetLedger(tmp_path) as ledger:
        ledger.begin(output(tmp_path, 'first'), 'paired', {'source': 'v1'})
        ledger.charge('cold', 21)
        ledger.charge('paired', 102)
        ledger.finish('INCONCLUSIVE')
    with BudgetLedger(tmp_path) as ledger:
        ledger.begin(output(tmp_path, 'fixed'), 'paired', {'source': 'v2'})
        ledger.charge('cold', 3)
        ledger.charge('paired', 7)
        ledger.finish('INCONCLUSIVE')
        assert ledger.data['used'] == dict(cold=24, feasibility=0, paired=109)
        assert len(ledger.data['attempts']) == 2


def test_same_source_code_failure_cannot_buy_another_attempt(tmp_path):
    with BudgetLedger(tmp_path) as ledger:
        ledger.begin(output(tmp_path, 'broken'), 'feasibility', {'source': 'v1'})
        ledger.charge('feasibility', 1)
        ledger.finish('INCONCLUSIVE')
        with pytest.raises(ContractError, match='changed actual source'):
            ledger.begin(output(tmp_path, 'retry'), 'feasibility', {'source': 'v1'})
        assert not output(tmp_path, 'retry').exists()


def test_only_qualified_timing_recheck_can_repeat_unchanged_source(tmp_path):
    with BudgetLedger(tmp_path) as ledger:
        ledger.begin(output(tmp_path, 'timing1'), 'paired', {'source': 'v1'})
        ledger.charge('paired', 110)
        ledger.finish('INCONCLUSIVE', retry_kind='timing_recheck')
        ledger.begin(output(tmp_path, 'timing2'), 'paired', {'source': 'v1'})
        ledger.charge('paired', 105)
        ledger.finish('PERFORMANCE_BELOW_10X')
        assert ledger.data['used']['paired'] == 215


def test_resource_stop_blocks_new_folder_new_source_and_other_phase(tmp_path):
    with BudgetLedger(tmp_path) as ledger:
        ledger.begin(output(tmp_path, 'first'), 'feasibility', {'source': 'v1'})
        with pytest.raises(AcceptanceFailure, match='wall budget'):
            ledger.charge('feasibility', 180)
        ledger.finish('RESOURCE_FAIL')
        assert (tmp_path/REPORT_PARENT/'REQUEST_STOP').is_file()
        with pytest.raises(AcceptanceFailure):
            ledger.begin(output(tmp_path, 'changed'), 'paired', {'source': 'v2'})
    with pytest.raises(AcceptanceFailure, match='stopped'):
        with BudgetLedger(tmp_path):
            pass


@pytest.mark.parametrize('category', ['cold', 'paired'])
def test_other_budget_boundaries_stop_exactly_at_limit(tmp_path, category):
    with BudgetLedger(tmp_path) as ledger:
        ledger.begin(output(tmp_path, 'first'), 'paired', {})
        ledger.charge(category, LIMITS[category]-.01)
        with pytest.raises(AcceptanceFailure):
            ledger.charge(category, .01)
        assert ledger.data['stopped']
        ledger.finish('RESOURCE_FAIL')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, True, '1', None])
def test_nonfinite_and_ambiguous_budget_inputs_rejected(value):
    with pytest.raises(ContractError):
        finite_nonnegative(value, 'duration')


def test_persistent_ledger_hash_prevents_silent_counter_reset(tmp_path):
    with BudgetLedger(tmp_path) as ledger:
        ledger.begin(output(tmp_path, 'first'), 'paired', {})
        ledger.charge('paired', 100)
        ledger.finish('INCONCLUSIVE')
        path = ledger.path
    value = read_json(path)
    value['used']['paired'] = 0
    write_json(path, value)
    with pytest.raises(ContractError, match='integrity'):
        with BudgetLedger(tmp_path):
            pass


def test_lock_cannot_be_stolen_even_when_pid_is_stale(tmp_path):
    parent = tmp_path/REPORT_PARENT
    parent.mkdir(parents=True)
    write_json(parent/'budget_ledger.lock', {'pid': 999999999, 'batch_id': BATCH_ID})
    with pytest.raises(ContractError, match='owned or orphaned'):
        with BudgetLedger(tmp_path):
            pass
    assert (parent/'budget_ledger.lock').is_file()


def test_concurrent_owner_cannot_remove_first_owner_lock(tmp_path):
    with BudgetLedger(tmp_path) as first:
        with pytest.raises(ContractError, match='owned or orphaned'):
            with BudgetLedger(tmp_path):
                pass
        assert first.lock.exists()
    assert not first.lock.exists()


def test_orphaned_active_attempt_cannot_be_reset(tmp_path):
    with BudgetLedger(tmp_path) as ledger:
        ledger.begin(output(tmp_path, 'crashed'), 'paired', {})
        ledger.charge('paired', 1)
    with pytest.raises(ContractError, match='Orphaned'):
        with BudgetLedger(tmp_path):
            pass


def test_existing_evidence_without_ledger_refuses_new_budget(tmp_path):
    prior = output(tmp_path, 'prior')
    prior.mkdir(parents=True)
    write_json(prior/'job.json', {'old': True})
    with pytest.raises(ContractError, match='refusing budget reset'):
        with BudgetLedger(tmp_path):
            pass


def test_output_must_be_fresh_immediate_child(tmp_path):
    with BudgetLedger(tmp_path) as ledger:
        for path in (tmp_path/'elsewhere', output(tmp_path, 'nested')/'deeper'):
            with pytest.raises(ContractError, match='immediate child'):
                ledger.begin(path, 'paired', {})
        first = output(tmp_path, 'first')
        ledger.begin(first, 'paired', {})
        ledger.finish('INCONCLUSIVE')
        with pytest.raises(ContractError, match='Fresh immediate child'):
            ledger.begin(first, 'paired', {'changed': True})


def test_nonfinite_json_stop_document_is_never_trusted(tmp_path):
    path = tmp_path/'REQUEST_STOP'
    path.write_text('{"seconds":NaN}', encoding='utf8')
    with pytest.raises(ContractError, match='Nonfinite JSON'):
        read_json(path)


def test_physical_gpu_reserve_and_old_ram_limits():
    G = 1024**3
    resource_check(rss=16*G, available=4*G, gpu_total=8*G, gpu_used=7.5*G)
    for values in [dict(rss=16*G+1, available=4*G), dict(rss=0, available=4*G-1),
                   dict(rss=0, available=4*G, gpu_total=8*G, gpu_used=7.5*G+1),
                   dict(rss=0, available=4*G, gpu_total=24*G, gpu_used=12*G+1)]:
        with pytest.raises(AcceptanceFailure) as error:
            resource_check(**values)
        assert error.value.status == 'RESOURCE_FAIL'
    assert RESOURCE_LIMITS['max_whole_GPU_bytes'] == 12*G


@pytest.mark.parametrize('gate,expected', [('states.cones.V','NUMERICAL_FAIL'),
    ('domain','DOMAIN_FAIL'), ('monitor.domain','DOMAIN_FAIL'), ('draw.midget_RGC.u','EVENT_FAIL'),
    ('aux','EVENT_FAIL'), ('stamp','EVENT_FAIL')])
def test_scientific_failures_are_not_performance_failures(gate, expected):
    assert comparison_status(dict(status='FAIL', issues=[dict(gate=gate)])) == expected
    assert comparison_status(dict(status='INCONCLUSIVE', issues=[dict(gate='typed_budget')])) == 'INCONCLUSIVE'


def test_real_net_improvement_below_ten_is_adoptable():
    result = performance_result([dict(reference=100, radau=50)])
    assert result['status'] == 'PERFORMANCE_BELOW_10X'
    assert result['speedup'] == 2
    assert result['adoptable']


def test_timing_noise_needs_evidence_and_does_not_change_adoption_threshold():
    assert performance_result([dict(reference=100, radau=99)])['reason'] == 'TIMING_RECHECK_REQUIRED'
    repeated = performance_result([dict(reference=100, radau=98), dict(reference=101, radau=98.5)])
    assert repeated['adoptable'] and repeated['speedup'] < 1.1
    noisy = performance_result([dict(reference=100, radau=99), dict(reference=98, radau=97)])
    assert noisy['status'] == 'INCONCLUSIVE'
    assert performance_result([dict(reference=100, radau=101)])['status'] == 'NO_SPEEDUP'


def test_recording_hash_uses_values_not_zip_container(tmp_path):
    for branch in ('a','b'):
        for folder in ('windows','observations','events'):
            (tmp_path/branch/folder).mkdir(parents=True)
    values = dict(time=np.array([1.,2.]), values=np.array([[.1,.2]],dtype=np.float64))
    np.savez(tmp_path/'a/windows/window_0000.npz', **values)
    np.savez_compressed(tmp_path/'b/windows/window_0000.npz', **values)
    assert recording_content_hash(tmp_path/'a') == recording_content_hash(tmp_path/'b')
    values['values'][0,1] += 1e-12
    np.savez(tmp_path/'b/windows/window_0000.npz', **values)
    assert recording_content_hash(tmp_path/'a') != recording_content_hash(tmp_path/'b')

def test_restore_oom_is_resource_stop_not_restore_retry():
    from tools.validate_group_scc_radau import _failure_status
    assert _failure_status(RuntimeError('CUDA out of memory'), 'RESTORE_LOAD') == 'RESOURCE_FAIL'
    assert _failure_status(ContractError('payload differs'), 'RESTORE_LOAD') == 'RESTORE_FAIL'


@pytest.mark.parametrize('status_denials',[0,1,'permanent'])
def test_supervisor_executes_only_a_small_cpu_fixture_and_closes_ledger(tmp_path, monkeypatch, status_denials):
    import tools.validate_group_scc_radau as harness
    fake = tmp_path/'fake_cpu_worker.py'
    source_root = Path(__file__).resolve().parents[1]
    fake.write_text('import sys\nsys.path.insert(0,'+repr(str(source_root))+')\n'+'''import json, pathlib, time
from tools.validate_group_scc_radau import worker_handshake, read_json
out = pathlib.Path(sys.argv[sys.argv.index('--output')+1])
worker_handshake(out,read_json(out/'job.json'),'uninterrupted')
(out/'status.json').write_text(json.dumps({'hot_started_unix':time.time()}))
time.sleep(.05)
(out/'uninterrupted_result.json').write_text(json.dumps({'dense_status':'PASS','hot_seconds':{'reference':1.,'radau':.5}}))
''', encoding='utf8')
    monkeypatch.setattr(harness, 'ROOT', tmp_path)
    monkeypatch.setattr(harness, '__file__', str(fake))
    monkeypatch.setattr(harness, 'scientific_source_preflight', lambda _: {'status':'PASS'})
    monkeypatch.setattr(harness, 'load_protocol', lambda _, candidate='radau': {'synthetic_cpu_only': True})
    monkeypatch.setattr(harness, 'identity_for', lambda _: {'synthetic_cpu_only': True,'candidate':'radau','execution_class':'B'})
    monkeypatch.setattr(harness, 'resource_check', lambda **_: None)
    original_read_bytes=Path.read_bytes
    remaining=[status_denials]
    def transient_sharing(path):
        if path.name=='status.json' and remaining[0]:
            if remaining[0]!='permanent':
                remaining[0]-=1
            raise PermissionError('synthetic Windows sharing violation')
        return original_read_bytes(path)
    monkeypatch.setattr(Path,'read_bytes',transient_sharing)
    if status_denials=='permanent':
        with pytest.raises(PermissionError,match='sharing violation'):
            harness.supervise(output(tmp_path,'fixture'),'feasibility')
        ledger=read_json(tmp_path/REPORT_PARENT/'budget_ledger.json')
        assert ledger['active'] is None and ledger['attempts'][0]['status']=='INCONCLUSIVE'
        assert ledger['used']['cold']>0 and ledger['used']['feasibility']>=.5
        assert not (output(tmp_path,'fixture')/'result.json').exists()
        assert (output(tmp_path,'fixture')/'REQUEST_STOP').exists()
        return
    result = harness.supervise(output(tmp_path, 'fixture'), 'feasibility')
    assert result['status'] == 'FEASIBILITY_PASS'
    assert remaining[0]==0
    ledger = read_json(tmp_path/REPORT_PARENT/'budget_ledger.json')
    assert ledger['active'] is None
    assert ledger['attempts'][0]['status'] == 'FEASIBILITY_PASS'
    assert ledger['used']['feasibility'] > 0
    receipt = read_json(output(tmp_path,'fixture')/'uninterrupted_worker_ownership.json')
    assert receipt['worker']['pid'] != receipt['launcher']['pid']
    assert receipt['job_kill_on_close']
    measured = read_json(output(tmp_path,'fixture')/'uninterrupted_supervisor_resources.json')
    assert measured['measured_process'] == receipt['worker']
    assert measured['actual_worker_bound']

def test_plain_text_stop_cannot_be_evaded_after_source_change(tmp_path):
    with BudgetLedger(tmp_path) as ledger:
        first = output(tmp_path, 'stopped')
        ledger.begin(first, 'paired', {'source': 1})
        ledger.finish('INCONCLUSIVE')
        (first/'REQUEST_STOP').write_text('user stop', encoding='utf8')
        with pytest.raises(AcceptanceFailure, match='STOP blocks'):
            ledger.begin(output(tmp_path, 'evade'), 'paired', {'source': 2})
        assert ledger.data['stopped']


def test_owned_code_error_stop_allows_only_changed_source_with_same_budget(tmp_path):
    with BudgetLedger(tmp_path) as ledger:
        first = output(tmp_path, 'broken')
        token = ledger.begin(first, 'paired', {'source': 1})
        ledger.charge('paired', 20)
        write_json(first/'REQUEST_STOP', dict(status='INCONCLUSIVE', run_id=token, message='ordinary code error'))
        ledger.finish('INCONCLUSIVE')
        ledger.begin(output(tmp_path, 'fixed'), 'paired', {'source': 2})
        assert ledger.data['used']['paired'] == 20
        ledger.finish('INCONCLUSIVE')

def test_union_observation_preserves_reference_own_mandatory_schedule():
    from types import SimpleNamespace as NS
    from tools.validate_group_scc_radau import mandatory_callback_points
    reference = NS(native_intervals=[NS(curves={'n':NS(segments=[NS(t1=.3),NS(t1=1.)])})])
    candidate = NS(native_intervals=[NS(curves={'n':NS(segments=[NS(t1=.25),NS(t1=.75)])})])
    fixed = {.5, 0., 1.}
    ref = mandatory_callback_points(reference, NS(accepted_times=[.2,1.]), fixed, 0., 1.)
    cand = mandatory_callback_points(candidate, NS(accepted_times=[.1,.2,.4]), fixed, 0., 1.)
    assert ref == {.2,.3,.5}
    assert cand == {.1,.2,.25,.4,.5,.75}
    assert (ref|cand)-ref == {.1,.25,.4,.75}
    assert 0. not in ref|cand and 1. not in ref|cand


def test_candidate_only_reference_mirrors_cannot_create_false_speedup():
    from tools.validate_group_scc_radau import paired_timing_summary
    phases = {'reference':dict(native_integration=40., monitor=10.),
              'radau':dict(native_integration=45., monitor=15.)}
    result = paired_timing_summary(phases, dict(snapshot_and_bridge=70., monitor=30.))
    assert result['raw_union_speedup'] == 2.5
    assert result['conservative_speedup'] == 50/60
    assert performance_result([result['hot_seconds']])['status'] == 'NO_SPEEDUP'
    assert result['reference_mirror_QA_seconds'] == dict(snapshot_and_bridge=70., monitor=30.)
    assert result['hot_seconds']['radau'] == result['raw_union_hot_seconds']['radau'] == 60.

@pytest.mark.parametrize('stop_method', ['terminate','close'])
def test_real_venv_worker_job_rejects_unowned_pid_and_kills_descendants(tmp_path, stop_method):
    import os
    import subprocess
    import sys
    import time
    from tools.validate_group_scc_radau import WindowsWorkerJob, process_identity, process_alive
    source_root = Path(__file__).resolve().parents[1]
    supervisor = process_identity(os.getpid())
    token = 'cpu-only-process-tree-regression'
    write_json(tmp_path/'job.json', dict(token=token, supervisor_process=supervisor))
    worker = tmp_path/'owned_cpu_worker.py'
    worker.write_text('import sys\nsys.path.insert(0,'+repr(str(source_root))+')\n'+'''
import json, os, pathlib, subprocess, time
from tools.validate_group_scc_radau import worker_handshake, read_json
out=pathlib.Path(sys.argv[1])
worker_handshake(out,read_json(out/'job.json'),'uninterrupted')
allocation=bytearray(48*1024**2)
child_code="import json,os,pathlib,time;pathlib.Path("+repr(str(out/'grandchild.json'))+").write_text(json.dumps({'pid':os.getpid(),'parent':os.getppid()}));time.sleep(60)"
child=subprocess.Popen([sys._base_executable,'-B','-c',child_code],creationflags=subprocess.CREATE_NO_WINDOW)
(out/'allocated.json').write_text(json.dumps({'pid':os.getpid(),'bytes':len(allocation)}))
time.sleep(60)
''',encoding='utf8')
    owner=WindowsWorkerJob()
    process=None
    try:
        process=subprocess.Popen([sys.executable,'-B',str(worker),str(tmp_path)],
            stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
        owner.bind_launcher(process)
        deadline=time.monotonic()+10
        while not (tmp_path/'uninterrupted_worker_handshake.json').exists():
            assert process.poll() is None, process.stderr.read().decode(errors='replace')
            assert time.monotonic()<deadline
            time.sleep(.02)
        handshake=read_json(tmp_path/'uninterrupted_worker_handshake.json')
        assert handshake['worker']['pid'] != process.pid
        assert handshake['parent_pid'] == process.pid
        with pytest.raises(ContractError,match='launcher'):
            owner.accept(dict(handshake,worker=supervisor,ancestry=[supervisor]),token=token,
                label='uninterrupted',supervisor=supervisor)
        with pytest.raises(ContractError,match='token'):
            owner.accept(handshake,token='wrong',label='uninterrupted',supervisor=supervisor)
        receipt=owner.accept(handshake,token=token,label='uninterrupted',supervisor=supervisor)
        write_json(tmp_path/'uninterrupted_worker_ownership.json',receipt)
        while not (tmp_path/'grandchild.json').exists() or not (tmp_path/'allocated.json').exists():
            assert process.poll() is None
            assert time.monotonic()<deadline
            time.sleep(.02)
        grandchild=process_identity(read_json(tmp_path/'grandchild.json')['pid'])
        actual=handshake['worker']
        measured=owner.resources(process._handle)
        assert measured['measured_process']==actual
        assert measured['peak_rss_bytes']>48*1024**2
        assert process_alive(actual['pid'],actual['creation_ticks'])
        assert process_alive(grandchild['pid'],grandchild['creation_ticks'])
        getattr(owner,stop_method)()
        process.wait(timeout=5)
        deadline=time.monotonic()+3
        while any(process_alive(p['pid'],p['creation_ticks']) for p in (actual,grandchild)):
            assert time.monotonic()<deadline
            time.sleep(.02)
        assert not process_alive(process.pid,owner.launcher['creation_ticks'])
    finally:
        owner.terminate(); owner.close()
        if process is not None:
            process.wait(timeout=5)
            if process.stderr is not None:
                process.stderr.close()

def _gpu_row(pid, engine, utilization, status=0):
    return dict(name='pid_'+str(pid)+'_luid_0x0_0x1_phys_0_eng_0_engtype_'+engine,
        utilization_percent=utilization,counter_status=status)


def test_gpu_activity_excludes_owned_compute_and_accepts_idle_desktop_contexts():
    from tools.validate_group_scc_radau import classify_gpu_activity
    rows=[_gpu_row(7,'Compute_0',90.),_gpu_row(8,'3D',0.),_gpu_row(9,'Copy',0.)]
    value=classify_gpu_activity(rows,{7})
    assert value['status']=='PASS'
    assert value['owned_active_engines']==1
    assert value['active_external_engines']==[]
    assert classify_gpu_activity([_gpu_row(8,'CUDA',.001)],{7})['status']=='CONTAMINATED'


def test_fractional_graphics_is_recorded_without_becoming_compute_contamination():
    from tools.validate_group_scc_radau import classify_gpu_activity, gpu_activity_summary
    value=classify_gpu_activity([_gpu_row(8,'3D',.001),_gpu_row(9,'Copy',.002)],{7})
    assert value['status']=='PASS'
    assert value['external_graphics_aggregate_percent']==.003
    samples=[dict(value,label=label,interval_seconds=1.) for label in ('before','during','after')]
    assert gpu_activity_summary(samples)['status']=='PASS'
    assert performance_result([dict(reference=100,radau=99)])['reason']=='TIMING_RECHECK_REQUIRED'
    busy=classify_gpu_activity([_gpu_row(8,'3D',2.)],{7})
    samples[1]=dict(busy,label='during',interval_seconds=10.)
    assert gpu_activity_summary(samples)['status']=='GRAPHICS_RECHECK_REQUIRED'


def test_unknown_or_unavailable_gpu_activity_cannot_claim_clean_timing():
    from tools.validate_group_scc_radau import classify_gpu_activity, gpu_activity_summary
    assert classify_gpu_activity([],{7})['status']=='INCONCLUSIVE'
    assert classify_gpu_activity([_gpu_row(8,'unknown',1.)],{7})['status']=='INCONCLUSIVE'
    assert classify_gpu_activity([_gpu_row(8,'3D',0.,0xc0000bc6)],{7})['status']=='INCONCLUSIVE'
    assert gpu_activity_summary([dict(status='PASS',label='before',interval_seconds=1.)])['status']=='INCONCLUSIVE'


def test_cold_rhs_warmup_compiles_once_without_advancing_or_sampling():
    from types import SimpleNamespace as NS
    from tools.validate_group_scc_radau import warm_exact_initial_rhs
    class Arm:
        start=4.993
        gpu_rhs=None
        def __init__(self):
            self.inner_state=np.array([1.,2.],dtype=np.float64)
            self.inner=NS(pack=lambda state:state.copy())
            self.rng=np.random.Generator(np.random.PCG64(41))
            self.compiled=NS(execution_profile={'calls':0},rhs_flat=self.rhs)
        def rhs(self,t,y):
            assert t==self.start
            self.compiled.execution_profile['calls']+=1
            return -y
        def checkpoint_payload(self):
            return dict(state=self.inner_state.copy(),time=self.start,rng=self.rng.bit_generator.state,events=[])
    arm=Arm()
    before=canonical_hash(arm.checkpoint_payload())
    result=warm_exact_initial_rhs(arm)
    assert arm.compiled.execution_profile['calls']==1
    assert canonical_hash(arm.checkpoint_payload())==before
    assert result['state_event_rng_unchanged']
    assert result['production_output_reused'] is False
    assert result['rhs_call_wall_seconds']>=0
    assert result['compiled_profile_before']['calls']==0
    assert result['compiled_profile_after']['calls']==1


def test_cold_rhs_warmup_rejects_hidden_state_mutation():
    from types import SimpleNamespace as NS
    from tools.validate_group_scc_radau import warm_exact_initial_rhs
    state=np.array([1.],dtype=np.float64)
    def bad_rhs(t,y):
        state[0]+=1
        return y
    arm=NS(start=4.993,gpu_rhs=None,inner_state=state,inner=NS(pack=lambda value:value.copy()),
        compiled=NS(execution_profile={},rhs_flat=bad_rhs),checkpoint_payload=lambda:dict(state=state.copy()))
    with pytest.raises(AcceptanceFailure,match='changed state/event/RNG'):
        warm_exact_initial_rhs(arm)

def test_missing_frozen_addendum_is_identified_with_exact_original_hash(tmp_path):
    from retina_sim.group_scc_protocol import scientific_source_preflight
    from retina_sim.resident_derived_ports import ADDENDUM_FILENAME, ADDENDUM_SHA256
    result=scientific_source_preflight(tmp_path)
    row=next(r for r in result['issues'] if r['path'].endswith(ADDENDUM_FILENAME))
    assert result['status']=='INCONCLUSIVE'
    assert row['status']=='MISSING_FROZEN_SCIENTIFIC_DEPENDENCY'
    assert row['expected_sha256']==ADDENDUM_SHA256=='21e2cc8ad4a00ea3b9d4f331109a4c11c88f793334653cd4008addaa3026f735'
    assert row['actual_sha256'] is None
    assert result['model_forward_started'] is False and result['cuda_initialized'] is False
    assert result['adoptable'] is False


def test_fabricated_or_modified_addendum_is_never_accepted(tmp_path):
    from retina_sim.group_scc_protocol import scientific_source_preflight
    from retina_sim.resident_derived_ports import ADDENDUM_FILENAME
    folder=tmp_path/'configurations/numerical_equivalence'
    folder.mkdir(parents=True)
    (folder/ADDENDUM_FILENAME).write_text('{}',encoding='utf8')
    result=scientific_source_preflight(tmp_path)
    row=next(r for r in result['issues'] if r['path'].endswith(ADDENDUM_FILENAME))
    assert row['status']=='FROZEN_HASH_MISMATCH'
    assert row['actual_sha256']!=row['expected_sha256']
    assert result['status']=='INCONCLUSIVE'


def test_missing_dependency_stops_before_ledger_child_or_expensive_preparation(tmp_path,monkeypatch):
    import tools.validate_group_scc_radau as harness
    monkeypatch.setattr(harness,'ROOT',tmp_path)
    def forbidden(*args,**kwargs):
        pytest.fail('Missing scientific dependency reached runtime preparation')
    monkeypatch.setattr(harness,'BudgetLedger',forbidden)
    monkeypatch.setattr(harness,'load_protocol',forbidden)
    monkeypatch.setattr(harness.subprocess,'Popen',forbidden)
    result=harness.supervise(output(tmp_path,'must_not_start'),'feasibility')
    assert result['status']=='INCONCLUSIVE' and result['stage']=='SOURCE_PREFLIGHT'
    assert result['budget_charged'] is False
    assert not (tmp_path/REPORT_PARENT).exists()

def test_status_read_retry_does_not_ignore_json_parse_failures(tmp_path,monkeypatch):
    path=tmp_path/'status.json'
    path.write_text('{broken JSON',encoding='utf8')
    original=Path.read_bytes
    calls=[]
    def counted(p):
        calls.append(p)
        return original(p)
    monkeypatch.setattr(Path,'read_bytes',counted)
    with pytest.raises(json.JSONDecodeError):
        read_json(path,permission_retry_seconds=.25)
    assert calls==[path]


def test_budget_split_includes_status_retry_time_and_uncertain_stop_cleanup():
    from tools.validate_group_scc_radau import charge_budget_interval
    class Ledger:
        def __init__(self):self.used=dict(cold=0.,feasibility=0.,paired=0.)
        def charge(self,kind,elapsed):self.used[kind]+=elapsed
    ledger=Ledger()
    charge_budget_interval(ledger,'feasibility',100.,102.,101.5)
    assert ledger.used['cold']==1.5 and ledger.used['feasibility']==.5
    charge_budget_interval(ledger,'feasibility',102.,102.25,101.5,uncertain=True)
    assert ledger.used['feasibility']==.75
    unknown=Ledger()
    charge_budget_interval(unknown,'paired',100.,100.25,None,uncertain=True)
    charge_budget_interval(unknown,'paired',100.25,100.5,None,uncertain=True)
    assert unknown.used['cold']==.5 and unknown.used['paired']==.5

def test_gpu_dop853_contract_preserves_reference_radau_and_original_scientific_budget():
    from retina_sim.group_scc_protocol import GPU_DOP853_CONTRACT, FALLBACK_GATE, FALLBACK_ORIGIN, BUDGET_SHA256
    from tools.validate_group_scc_radau import identity_for
    root=Path(__file__).resolve().parents[1]
    original=load_protocol(root)
    radau_before=canonical_hash(original)
    fallback=load_protocol(root,'gpu_dop853')
    assert fallback['reference']==original['reference']
    assert fallback['candidate']==dict(method='joint_DOP853',rtol=1e-6,atol=1e-9,max_step=.0001,fast_max_step=.0001)
    assert fallback['execution_contract']==GPU_DOP853_CONTRACT
    assert fallback['paired_entry_gate']==FALLBACK_GATE
    assert fallback['fallback_origin']==FALLBACK_ORIGIN
    assert fallback['numerical_budget']==original['numerical_budget']
    assert fallback['numerical_budget']['sha256']==BUDGET_SHA256
    assert fallback['budget_limits_seconds']==LIMITS
    assert fallback['resource_limits']==RESOURCE_LIMITS
    assert clock_events(fallback)==clock_events(original)
    assert canonical_hash(load_protocol(root))==radau_before
    a,b=identity_for(original),identity_for(fallback)
    assert a['candidate']=='radau' and a['execution_class']=='B'
    assert b['candidate']=='gpu_dop853' and b['execution_class']=='A'
    assert a!=b
    assert all(FALLBACK_ORIGIN['output']+'/'+name in b['historical_pins'] for name in FALLBACK_ORIGIN['evidence_sha256'])


def test_gpu_dop853_factory_job_and_checkpoint_do_not_alias_radau():
    from tools.validate_group_scc_radau import arm_execution_options, bind_worker_selection, checkpoint_identity
    marker=object()
    ref=arm_execution_options('gpu_dop853','reference',marker,marker,marker)
    gpu=arm_execution_options('gpu_dop853','gpu_dop853',marker,marker,marker)
    assert ref['mode']=='reference' and ref['shared_plans'] is None
    assert gpu['mode']=='gpu_dop853' and gpu['shared_plans'] is marker
    assert ref['torch'] is gpu['torch'] is marker
    identity={'candidate':'gpu_dop853','execution_class':'A'}
    job=dict(candidate='gpu_dop853',phase='paired',identity=identity)
    bind_worker_selection(job,'gpu_dop853','paired')
    with pytest.raises(ContractError,match='immutable job'):
        bind_worker_selection(job,'radau','paired')
    with pytest.raises(ContractError,match='new feasibility'):
        bind_worker_selection(dict(job,phase='feasibility'),'gpu_dop853','feasibility')
    assert checkpoint_identity(identity,'gpu_dop853')==dict(identity,mode='gpu_dop853')
    assert checkpoint_identity(identity,'reference')==dict(identity,mode='reference')
    with pytest.raises(ContractError,match='Checkpoint arm'):
        checkpoint_identity(identity,'radau')
    with pytest.raises(ContractError,match='Arm mode'):
        arm_execution_options('gpu_dop853','radau',marker,marker,marker)


@pytest.mark.parametrize('candidate,phase',[('gpu_dop853','feasibility'),('unknown','paired')])
def test_invalid_fallback_phase_fails_before_any_ledger_or_child(tmp_path,monkeypatch,candidate,phase):
    import tools.validate_group_scc_radau as h
    monkeypatch.setattr(h,'ROOT',tmp_path)
    monkeypatch.setattr(h,'scientific_source_preflight',lambda _:pytest.fail('must reject before preflight'))
    with pytest.raises(ContractError):
        h.supervise(output(tmp_path,'forbidden'),phase,candidate)
    assert not (tmp_path/REPORT_PARENT).exists()


def _fallback_lineage_fixture(tmp_path):
    from shutil import copyfile
    from retina_sim.group_scc_protocol import FALLBACK_ORIGIN
    root=Path(__file__).resolve().parents[1]
    p=load_protocol(root,'gpu_dop853')
    origin=tmp_path/FALLBACK_ORIGIN['output']
    origin.mkdir(parents=True)
    for name in FALLBACK_ORIGIN['evidence_sha256']:
        copyfile(root/FALLBACK_ORIGIN['output']/name,origin/name)
    ledger=dict(attempts=[dict(run_id=FALLBACK_ORIGIN['run_id'],status='NUMERICAL_FAIL',
        phase='feasibility',identity=FALLBACK_ORIGIN['source_identity'],output=str(origin))])
    return p,ledger,origin


@pytest.mark.parametrize('altered',['failure.json','REQUEST_STOP','prospective_identity.json','ledger'])
def test_fallback_requires_exact_preserved_radau_lineage(tmp_path,altered):
    from retina_sim.group_scc_protocol import verify_fallback_origin
    p,ledger,origin=_fallback_lineage_fixture(tmp_path)
    verify_fallback_origin(tmp_path,p,ledger)
    if altered=='ledger':
        ledger['attempts'][0]['status']='RESOURCE_FAIL'
    else:
        (origin/altered).write_text('{}',encoding='utf8')
    with pytest.raises(ContractError):
        verify_fallback_origin(tmp_path,p,ledger)


def _entry_receipt(started=100.,completed=101.):
    from tools.validate_group_scc_radau import enforce_paired_entry_gate
    p={'start_seconds':4.993}
    identity={'candidate':'gpu_dop853'}
    receipt=dict(status='PASS',run_id='token',candidate='gpu_dop853',identity=canonical_hash(identity),
        hot_started_unix=started,completed_unix=completed,time_seconds=4.9931,
        numerical_status='PASS',domain_status='PASS',event_status='PASS',monitor_status='PASS',dense_status='PASS',
        recording_and_acknowledgement_complete=True,complete_coverage=True,charge_category='paired')
    return enforce_paired_entry_gate,identity,p,receipt


def test_integral_entry_gate_has_no_new_clock_or_feasibility_allowance():
    gate,identity,p,receipt=_entry_receipt()
    assert not gate('gpu_dop853','token',identity,p,None,100.)
    assert not gate('gpu_dop853','token',identity,p,100.,279.999)
    with pytest.raises(AcceptanceFailure,match='wall limit'):
        gate('gpu_dop853','token',identity,p,100.,280.)
    assert gate('gpu_dop853','token',identity,p,100.,1300.,receipt)
    with pytest.raises(ContractError):
        gate('gpu_dop853','token',identity,p,102.,1300.,receipt)
    receipt['completed_unix']=280.
    with pytest.raises(AcceptanceFailure,match='wall limit'):
        gate('gpu_dop853','token',identity,p,100.,280.,receipt)


@pytest.mark.parametrize('key,value',[('candidate','radau'),('run_id','other'),('identity','other'),
    ('dense_status','INCONCLUSIVE'),('complete_coverage',False),('event_status','FAIL'),
    ('monitor_status','INCONCLUSIVE'),('recording_and_acknowledgement_complete',False),
    ('time_seconds',4.99305),('charge_category','feasibility')])
def test_integral_entry_gate_rejects_incomplete_or_switched_receipts(key,value):
    gate,identity,p,receipt=_entry_receipt()
    receipt[key]=value
    with pytest.raises(ContractError):
        gate('gpu_dop853','token',identity,p,100.,102.,receipt)


def test_fallback_timing_uses_actual_candidate_key():
    from tools.validate_group_scc_radau import paired_timing_summary
    phases={'reference':{'inner_integration':2.},'gpu_dop853':{'inner_integration':1.}}
    report=paired_timing_summary(phases,{'monitor':.3},'gpu_dop853')
    assert report['hot_seconds']=={'reference':2.,'gpu_dop853':1.}
    assert report['conservative_speedup']==2.
    assert report['raw_union_speedup']==2.3
    assert performance_result([report['hot_seconds']],'gpu_dop853')['status']=='PERFORMANCE_BELOW_10X'


@pytest.mark.parametrize('gate_timeout',[False,True])
def test_synthetic_fallback_pair_and_restore_share_existing_ledger(tmp_path,monkeypatch,gate_timeout):
    import tools.validate_group_scc_radau as h
    from retina_sim.group_scc_protocol import FALLBACK_ORIGIN
    root=Path(__file__).resolve().parents[1]
    fake=tmp_path/'fake_fallback_cpu_worker.py'
    fake.write_text('import sys\nsys.path.insert(0,'+repr(str(root))+')\n'+r'''import pathlib,time,json
from tools.validate_group_scc_radau import worker_handshake,read_json,write_json,bind_worker_selection
from retina_sim.contracts import canonical_hash
out=pathlib.Path(sys.argv[sys.argv.index('--output')+1])
job=read_json(out/'job.json')
candidate=sys.argv[sys.argv.index('--candidate')+1]
phase=sys.argv[sys.argv.index('--phase')+1]
bind_worker_selection(job,candidate,phase)
restore='--restore' in sys.argv
worker_handshake(out,job,'restored' if restore else 'uninterrupted')
if restore:
    saved=read_json(out/'session_checkpoint.json')
    assert saved['candidate']==candidate
    write_json(out/'status.json',{'hot_started_unix':saved['started']})
    write_json(out/'restore_result.json',{'status':'RESTORE_PASS','candidate':candidate})
else:
    started=time.time()
    write_json(out/'status.json',{'hot_started_unix':started})
    if read_json(out/'protocol.json')['synthetic_timeout']:
        time.sleep(2.)
        raise RuntimeError('Supervisor should have stopped this fixture')
    receipt=dict(status='PASS',run_id=job['token'],candidate=candidate,identity=canonical_hash(job['identity']),
        hot_started_unix=started,completed_unix=time.time(),time_seconds=4.9931,
        numerical_status='PASS',domain_status='PASS',event_status='PASS',monitor_status='PASS',dense_status='PASS',
        recording_and_acknowledgement_complete=True,complete_coverage=True,charge_category='paired')
    write_json(out/'paired_entry_gate.json',receipt)
    write_json(out/'session_checkpoint.json',dict(candidate=candidate,started=started))
    write_json(out/'uninterrupted_oracle.json',{})
    write_json(out/'uninterrupted_result.json',dict(candidate=candidate,dense_status='PASS',
        hot_seconds={'reference':2.,candidate:1.},raw_union_hot_seconds={'reference':2.,candidate:1.},
        timing_environment_status='PASS',hardware={'synthetic_cpu_only':True}))
''',encoding='utf8')
    with BudgetLedger(tmp_path) as ledger:
        ledger.begin(output(tmp_path,'old_radau'),'feasibility',{'old':True})
        ledger.charge('cold',11.)
        ledger.charge('feasibility',174.)
        ledger.finish('NUMERICAL_FAIL')
    monkeypatch.setattr(h,'ROOT',tmp_path)
    monkeypatch.setattr(h,'__file__',str(fake))
    monkeypatch.setattr(h,'scientific_source_preflight',lambda _: {'status':'PASS'})
    monkeypatch.setattr(h,'load_protocol',lambda _,candidate: {'start_seconds':4.993,'synthetic_timeout':gate_timeout})
    monkeypatch.setattr(h,'identity_for',lambda _: {'candidate':'gpu_dop853','execution_class':'A','synthetic_cpu_only':True})
    monkeypatch.setattr(h,'verify_fallback_origin',lambda *args:None)
    monkeypatch.setattr(h,'resource_check',lambda **kwargs:None)
    if gate_timeout:
        monkeypatch.setattr(h,'FALLBACK_GATE',dict(h.FALLBACK_GATE,wall_limit_seconds=.03))
        with pytest.raises(AcceptanceFailure,match='first100us wall limit'):
            h.supervise(output(tmp_path,'fallback'),'paired','gpu_dop853')
    else:
        result=h.supervise(output(tmp_path,'fallback'),'paired','gpu_dop853')
        assert result['restore_status']=='RESTORE_PASS'
        assert result['hot_seconds']=={'reference':2.,'gpu_dop853':1.}
    ledger=read_json(tmp_path/REPORT_PARENT/'budget_ledger.json')
    assert ledger['used']['feasibility']==174.
    assert ledger['used']['cold']>=11.
    assert ledger['used']['paired']>0.
    assert ledger['active'] is None
    assert ledger['limits']==LIMITS
    assert ledger['stopped']==gate_timeout
    if gate_timeout:
        assert ledger['attempts'][-1]['status']=='RESOURCE_FAIL'
        assert (tmp_path/REPORT_PARENT/'REQUEST_STOP').exists()
    else:
        job=read_json(output(tmp_path,'fallback')/'job.json')
        assert job['candidate']==job['identity']['candidate']=='gpu_dop853'
        receipt=read_json(output(tmp_path,'fallback')/'restored_worker_ownership.json')
        assert receipt['job_kill_on_close']


@pytest.mark.parametrize('timing_status',['PASS','CONTAMINATED','INCONCLUSIVE','GRAPHICS_RECHECK_REQUIRED',None])
def test_repeat_aggregation_requires_prior_clean_timing(timing_status):
    from tools.validate_group_scc_radau import eligible_timing_repeat
    prior=dict(scientific_status='PASS',hardware={'GPU':'synthetic'},restore_status='RESTORE_PASS',
        timing_environment_status=timing_status)
    assert eligible_timing_repeat(prior,{'GPU':'synthetic'}) == (timing_status=='PASS')
    assert not eligible_timing_repeat(prior,{'GPU':'different'})
    assert not eligible_timing_repeat(dict(prior,restore_status='PENDING'),{'GPU':'synthetic'})


def test_environment_identity_binds_scipy_and_numba_without_cuda():
    from types import SimpleNamespace
    import scipy
    import numba
    from tools.validate_group_scc_radau import hardware_identity
    fake=SimpleNamespace(__version__='synthetic',version=SimpleNamespace(cuda='synthetic'),
        cuda=SimpleNamespace(get_device_properties=lambda _:SimpleNamespace(uuid='fixture',total_memory=1),
                             get_device_name=lambda _:'fixture'))
    identity=hardware_identity(fake)
    assert identity['scipy_version']==scipy.__version__
    assert identity['numba_version']==numba.__version__
    assert identity==hardware_identity(fake)
