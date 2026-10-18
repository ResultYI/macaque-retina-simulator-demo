"""Compiled RHS boundary, isolation, fallback and checkpoint regressions."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from retina_sim.assembly import RetinaBuilder
from retina_sim.contracts import ContractError,Quantity,RunMode
from retina_sim.hybrid import HybridRunner
from retina_sim.network import InputBundle,JointNetwork
from examples.full_network_fixture import full_fixture

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def outer():
    c,ch,a=full_fixture(ROOT);s=a['spec']
    spec=replace(s,populations=s.populations[:2],components=s.components[:1],links=())
    compiled=RetinaBuilder(a['registry']).build(spec,RunMode.NUMERIC_FIXTURE)
    return compiled,ch

@pytest.fixture(scope='module')
def models(outer):
    c,ch=outer;reference=JointNetwork(c,compute_backend='python');compiled=JointNetwork(c,compute_backend='numba')
    y=np.concatenate([q.values for q in reference.initial_conditions()]);b=InputBundle({k:v.at(0) for k,v in ch.items()})
    return reference,compiled,y,b

def test_rhs_state_isolation_and_reference(models):
    py,jit,y,b=models;z=y.copy();z[0]*=1.01;z[18]*=1.001
    first=None
    for state in (y,z,y):
        a=py.rhs_flat(0,state,b,py.parameters,py.graph);v=jit.rhs_flat(0,state,b,jit.parameters,jit.graph)
        np.testing.assert_allclose(a,v,rtol=0,atol=2e-10)
        if first is None:first=v.copy()
    np.testing.assert_array_equal(first,v)
    v[:]=np.nan
    assert np.all(np.isfinite(y))
    np.testing.assert_array_equal(first,jit.rhs_flat(0,y,b,jit.parameters,jit.graph))

@pytest.mark.parametrize('bad',['NaN','float32','shape','unit','meaning','evidence','corrupted_quantity','identity'])
def test_public_boundary_rejections(models,bad):
    py,jit,y,b=models;y=y.copy();channels=dict(b.channels);parameters=jit.parameters
    if bad=='NaN':y[0]=np.nan
    elif bad=='float32':y=y.astype(np.float32)
    elif bad=='shape':y=y[:-1]
    elif bad=='identity':parameters=object()
    else:
        key=next(iter(channels));q=channels[key];values=q.values.copy()
        unit='A' if bad=='unit' else q.unit;meaning='wrong' if bad=='meaning' else q.meaning
        evidence='wrong' if bad=='evidence' else q.evidence_tag
        channels[key]=Quantity(values,unit,meaning,evidence,q.coordinate_frame)
        if bad=='corrupted_quantity':
            channels[key].values.setflags(write=True);channels[key].values[0]=np.nan;channels[key].values.setflags(write=False)
    with pytest.raises(ContractError):jit.rhs_flat(0,y,InputBundle(channels),parameters,jit.graph)

def test_unsupported_closure_reports_fallback(outer):
    c,_=outer;blocks=(replace(c.output_order[0],cyclic=True),)+c.output_order[1:]
    c=replace(c,output_order=blocks)
    auto=JointNetwork(c,compute_backend='auto')
    assert auto.compute_metadata['backend']=='python_reference'
    assert 'Algebraic closure' in auto.compute_metadata['fallback_reason']
    with pytest.raises(ContractError,match='unavailable'):JointNetwork(c,compute_backend='numba')

def test_source_revision_must_be_reviewed(outer,monkeypatch):
    from retina_sim import compiled_kernel
    monkeypatch.setitem(compiled_kernel.IMPLEMENTATIONS,'vh2005.py','unreviewed')
    auto=JointNetwork(outer[0]);assert auto.compute_metadata['backend']=='python_reference'
    assert 'reviewed' in auto.compute_metadata['fallback_reason']

def test_checkpoint_backend_identity_and_continuation(outer):
    c,ch=outer;jit=HybridRunner(c,compute_backend='numba');py=HybridRunner(c,compute_backend='python')
    state=jit.initialize();text=jit.checkpoint(state)
    assert jit.checkpoint(jit.restore(text))==text
    with pytest.raises(ContractError):py.restore(text)
    first,_=jit.advance(state,ch,.00035)
    restored=jit.restore(jit.checkpoint(first));a,_=jit.advance(first,ch,.00065);b,_=jit.advance(restored,ch,.00065)
    assert jit.checkpoint(a)==jit.checkpoint(b)


def test_fused_native_rows_match_original_instruction_order(outer,monkeypatch):
    from retina_sim.compiled_kernel import Plan
    compiled,channels=outer
    original=JointNetwork(compiled,compute_backend='numba')
    reference=JointNetwork(compiled,compute_backend='python')
    monkeypatch.setattr(Plan,'native_fusion_minimum_cells',0)
    fused=JointNetwork(compiled,compute_backend='numba')
    assert fused.compute_metadata['backend']=='numba_fused_native_rows_v1'
    y=np.concatenate([q.values for q in original.initial_conditions()])
    rng=np.random.default_rng(91421);perturbed=y*rng.uniform(.995,1.005,len(y))
    first=None
    for values in (y,perturbed,y):
        bundle=InputBundle({key:channel.at(.00023) for key,channel in channels.items()})
        expected=original.rhs_flat(.00023,values,bundle,original.parameters,original.graph)
        actual=fused.rhs_flat(.00023,values,bundle,fused.parameters,fused.graph)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_allclose(actual,reference.rhs_flat(.00023,values,bundle,reference.parameters,reference.graph),rtol=0,atol=2e-10)
        if first is None:first=actual.copy()
    np.testing.assert_array_equal(first,actual)
