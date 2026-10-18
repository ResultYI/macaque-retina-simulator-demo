"""Actual legacy-loader and whole-file compatibility, without new neural cases."""
from pathlib import Path
import json
import numpy as np
import pytest
from examples.full_network_fixture import full_fixture
from retina_sim.background import BackgroundPolicy
from retina_sim.background_checkpoint import BackgroundCheckpointStore
from retina_sim.checkpoint_streaming import save_background_streamed,_legacy_hash
from retina_sim.contracts import ContractError,SolverConfig,canonical_hash
from retina_sim.hybrid import HybridRunner
from retina_sim.runtime_profiles import LightContext

ROOT=Path(__file__).resolve().parents[1]


def setup():
    compiled,_,_=full_fixture(ROOT);runner=HybridRunner(compiled)
    state=runner.initialize(SolverConfig())
    state.input_position['format_probe']=[-0.,1e-307,1e300,'中文',True,None]
    store=BackgroundCheckpointStore(ROOT,runner,state,LightContext(),BackgroundPolicy())
    return store,state


def test_same_bytes_and_unmodified_strict_loader_restores_every_field(tmp_path):
    store,state=setup();before=store.runner.checkpoint(state)
    old=tmp_path/'old.json';new=tmp_path/'streamed.json'
    store.save(old,state,report=None,origin_seconds=0.,safety={'measured':True})
    receipt=save_background_streamed(store,new,state,report=None,origin_seconds=0.,safety={'measured':True})
    assert old.read_bytes()==new.read_bytes()
    restored,value=store.load(new)
    assert store.runner.checkpoint(restored)==before==store.runner.checkpoint(state)
    assert receipt['complete_checkpoint_hash']==value['complete_checkpoint_hash']


def test_corrupted_array_is_rejected_by_the_existing_loader(tmp_path):
    store,state=setup();path=tmp_path/'streamed.json';save_background_streamed(store,path,state)
    payload=json.loads(path.read_bytes());payload['checkpoint']['payload']['dynamic']['state']['ordered_state_arrays'][0][0]+=1
    path.write_text(json.dumps(payload),encoding='utf-8')
    with pytest.raises(ContractError):store.load(path)


def test_nonfinite_report_failure_preserves_previous_complete_checkpoint(tmp_path):
    store,state=setup();path=tmp_path/'streamed.json';save_background_streamed(store,path,state);before=path.read_bytes()
    with pytest.raises(ValueError):save_background_streamed(store,path,state,report={'invalid':float('nan')})
    assert path.read_bytes()==before


def test_array_chunk_boundaries_match_legacy_plain_json_hash():
    array=np.arange(17005,dtype=float)*.001
    array[8191]=-0.;array[8192]=1e-307
    assert _legacy_hash({'array':array})==canonical_hash({'array':array.tolist()})
