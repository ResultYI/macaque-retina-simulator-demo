"""Order, isolation, invalidation and real event/trace equivalence."""
from pathlib import Path
import numpy as np
import pytest
from examples.full_network_fixture import full_fixture
from retina_sim.contracts import SolverConfig,canonical_hash
from retina_sim.hybrid import HybridRunner
from retina_sim.recording import RecordingPlan
from retina_sim.recording_indices import IndexedRecordingPlan

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('selected',[None,(),('b','a','a','missing'),frozenset(('c','a'))])
def test_order_duplicates_and_return_isolation(selected):
    old=RecordingPlan(cell_ids=selected);new=IndexedRecordingPlan(old);ids=('a','b','c','a')
    expected=old.indices(ids);first=new.indices(ids);second=new.indices(ids)
    np.testing.assert_array_equal(first,expected);np.testing.assert_array_equal(second,expected)
    if len(first):first[0]=999
    np.testing.assert_array_equal(new.indices(ids),expected)


def test_mutable_layout_and_selection_never_reuse_stale_indices():
    ids=['a','b'];old=RecordingPlan(cell_ids=('a',));new=IndexedRecordingPlan(old)
    np.testing.assert_array_equal(new.indices(ids),[0]);ids.reverse()
    np.testing.assert_array_equal(new.indices(ids),[1])
    selected=['a'];new=IndexedRecordingPlan(RecordingPlan(cell_ids=selected));layout=('a','b')
    np.testing.assert_array_equal(new.indices(layout),[0]);selected[0]='b'
    np.testing.assert_array_equal(new.indices(layout),[1])


def test_cache_is_bounded_and_layout_identity_is_retained():
    new=IndexedRecordingPlan(RecordingPlan(cell_ids=('a',)));layouts=[]
    for i in range(80):
        ids=tuple(['a',str(i)]);layouts.append(ids);np.testing.assert_array_equal(new.indices(ids),[0])
    assert len(new._index_cache)==64
    for key,(ids,value) in new._index_cache.items():assert id(ids)==key and not value.flags.writeable


def test_full_trace_units_ids_metadata_clocks_rng_and_history_equal():
    compiled,channels,_=full_fixture(ROOT);runner=HybridRunner(compiled,compute_backend='python')
    baseline=runner.initialize(SolverConfig(rtol=1e-6,atol=1e-9,max_step=.0001))
    plan=RecordingPlan(ports=('outer.release','RGC_midget_ON.voltage'),states=(),cell_ids=('cones:0','cones:1','RGC_midget_ON:0'))
    fast=IndexedRecordingPlan(plan);times=np.array([0.,.00013,.00047,.001,.0013])
    left,a=runner.advance(baseline,channels,.0013,sample_times=times,recording_plan=plan)
    right,b=runner.advance(baseline,channels,.0013,sample_times=times,recording_plan=fast)
    assert runner.checkpoint(left)==runner.checkpoint(right)
    assert canonical_hash(a.metadata)==canonical_hash(b.metadata)
    assert a.outputs_with_units.keys()==b.outputs_with_units.keys()
    for key,q in a.outputs_with_units.items():
        other=b.outputs_with_units[key]
        assert (q.unit,q.meaning,q.evidence_tag,q.coordinate_frame)==(other.unit,other.meaning,other.evidence_tag,other.coordinate_frame)
        np.testing.assert_array_equal(q.values,other.values)
    assert fast.procedure_identity['statistics']['hits']>0
