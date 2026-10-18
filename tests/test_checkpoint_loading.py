"""Strict layers, mutable isolation and continuation for the parsed decoder."""
from pathlib import Path
import copy,json,types
import numpy as np
import pytest
from examples.full_network_fixture import full_fixture
from retina_sim.background import BackgroundPolicy
from retina_sim.background_checkpoint import BackgroundCheckpointStore
from retina_sim.contracts import ContractError,SolverConfig,canonical_hash
from retina_sim.engine import restore,restore_payload
from retina_sim.hybrid import HybridRunner
from retina_sim.runtime_profiles import LightContext

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def case(tmp_path_factory):
    compiled,channels,_=full_fixture(ROOT);runner=HybridRunner(compiled,compute_backend='python')
    state=runner.initialize(SolverConfig(input_interpolation='analytic'))
    store=BackgroundCheckpointStore(ROOT,runner,state,LightContext(),BackgroundPolicy())
    state.input_position['format_probe']=[-0.,1e-307,1e300,'测试',True,None]
    path=tmp_path_factory.mktemp('parsed_checkpoint')/'original.json';store.save(path,state)
    return store,state,path,channels


def repair_outer(value,dynamic=True,retina=True):
    p=value['checkpoint'];d=p['payload']
    if dynamic:d['dynamic']['payload_hash']=canonical_hash(d['dynamic']['state'])
    if retina:p['payload_hash']=canonical_hash(d)
    value['complete_checkpoint_hash']=canonical_hash(p)
    value['auxiliary_time_rng_history_hash']=canonical_hash({k:d[k] for k in ('grid_origin_seconds','sample_index',
        'component_tick_indices','spike_states','delay_states','link_delay_states','input_position')})
    value['report_hash']=canonical_hash(value.get('report'))


def test_all_fields_and_continuation_equal_without_mutating_source(case):
    store,initial,path,channels=case;source=path.read_bytes();before=store.runner.checkpoint(initial)
    old,old_value=store.load(path,decode_mode='legacy');new,value=store.load(path,decode_mode='parsed')
    assert old_value==value and store.runner.checkpoint(old)==store.runner.checkpoint(new)==before
    assert store.last_load_procedure['procedure']=='parsed_strict_checkpoint_v1'
    assert store.last_load_timings['file_read_seconds']>=0
    first,a=store.runner.advance(old,channels,.0013);second,b=store.runner.advance(new,channels,.0013)
    assert store.runner.checkpoint(first)==store.runner.checkpoint(second)
    for key,q in a.outputs_with_units.items():np.testing.assert_array_equal(q.values,b.outputs_with_units[key].values)
    assert path.read_bytes()==source and store.runner.checkpoint(initial)==before


def test_returned_envelope_and_state_have_no_mutable_aliases(case):
    store,_,path,_=case;state,value=store.load(path,decode_mode='parsed');before=store.runner.checkpoint(state)
    payload=value['checkpoint']['payload'];payload['input_position']['format_probe'][0]=123.
    payload['dynamic']['state']['ordered_state_arrays'][0][0]=99.
    next(iter(payload['spike_states'].values()))['history'][0][0]=31.
    next(iter(payload['delay_states'].values()))['history_values'][0][0]=77.
    assert store.runner.checkpoint(state)==before
    state.input_position['format_probe'][1]=321.
    assert payload['input_position']['format_probe'][1]!=321.


@pytest.mark.parametrize('layer',['binding','complete','retina','dynamic','model','context','shape','absolute_time','report','convergence'])
def test_every_validation_layer_still_rejects(case,tmp_path,layer):
    store,_,path,_=case;value=json.loads(path.read_bytes());p=value['checkpoint'];d=p['payload'];raw=d['dynamic']['state']
    if layer=='binding':value['binding_hash']='incorrect'
    elif layer=='complete':raw['ordered_state_arrays'][0][0]+=1
    elif layer=='retina':d['sample_index']+=1;repair_outer(value,retina=False)
    elif layer=='dynamic':raw['time_seconds']+=.1;repair_outer(value,dynamic=False)
    elif layer=='model':raw['model_hash']='different';repair_outer(value)
    elif layer=='context':d['input_position']['phase']='media_stimulus';repair_outer(value)
    elif layer=='shape':raw['ordered_state_arrays'][0].append(1.);repair_outer(value)
    elif layer=='absolute_time':value['absolute_time_seconds']+=1.
    else:
        value['report']={'status':'NOT_REACHED'}
        if layer=='convergence':value['report_hash']=canonical_hash(value['report'])
    target=tmp_path/(layer+'.json');target.write_text(json.dumps(value),encoding='utf-8')
    for mode in ('legacy','parsed'):
        with pytest.raises((ContractError,ValueError,TypeError,KeyError)):
            store.load(target,decode_mode=mode,require_converged=layer in ('report','convergence'))


def test_real_geometry_check_is_still_called_first(case,monkeypatch):
    store,_,path,_=case
    def changed():raise ContractError('Geometry changed')
    monkeypatch.setattr(store,'_require_current_geometry',changed)
    for mode in ('legacy','parsed'):
        with pytest.raises(ContractError,match='Geometry changed'):store.load(path,decode_mode=mode)


def test_custom_restore_hook_keeps_original_api(case,monkeypatch):
    store,_,path,_=case;original=store.runner.restore;calls=[]
    def custom(self,*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    monkeypatch.setattr(store.runner,'restore',types.MethodType(custom,store.runner))
    state,_=store.load(path)
    assert calls==[True] and store.last_load_procedure['procedure']=='legacy_JSON_roundtrip'
    with pytest.raises(ContractError,match='Custom restore'):store.load(path,decode_mode='parsed')


def test_parsed_path_skips_only_the_two_whole_document_json_bridges(case,monkeypatch):
    store,_,path,_=case;original=json.dumps
    def guarded(value,*args,**kwargs):
        if isinstance(value,dict) and ({'version','payload','payload_hash'}<=value.keys() or
            {'checkpoint_version','state','payload_hash'}<=value.keys()):
            raise AssertionError('Unnecessary whole-document JSON bridge')
        return original(value,*args,**kwargs)
    monkeypatch.setattr(json,'dumps',guarded)
    store.load(path,decode_mode='parsed')
    with pytest.raises(AssertionError,match='JSON bridge'):store.load(path,decode_mode='legacy')


def test_low_level_parsed_payload_keeps_its_document(case):
    store,_,path,_=case;payload=json.loads(path.read_bytes())['checkpoint']['payload']['dynamic']
    expected={key:payload['state'][key] for key in ('state_spec_hash','graph_hash','parameter_hash','model_hash','model_id','model_version')}
    before=copy.deepcopy(payload);left=restore(json.dumps(payload),expected);right=restore_payload(payload,expected)
    assert payload==before
    for a,b in zip(left.ordered_state_arrays,right.ordered_state_arrays):np.testing.assert_array_equal(a.view(np.uint64),b.view(np.uint64))


def test_explicit_environment_rollback(case,monkeypatch):
    store,_,path,_=case;monkeypatch.setenv('RETINA_CHECKPOINT_DECODER','legacy')
    store.load(path);assert store.last_load_procedure['procedure']=='legacy_JSON_roundtrip'
    store.load(path,decode_mode='parsed');assert store.last_load_procedure['procedure']=='parsed_strict_checkpoint_v1'
    monkeypatch.setenv('RETINA_CHECKPOINT_DECODER','unchecked')
    with pytest.raises(ContractError,match='Unknown checkpoint'):store.load(path)
