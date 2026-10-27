"""Isolated source-teacher contracts; no physiological claims or production fit."""
from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np
import pytest

from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.hc_teacher_calibration import (
    DerivedHomogeneousTeacher, SOURCE_FILES, FREQUENCIES, load_targets,
    verify_source_files, normalized, frequency_response, linearize,
    make_homogeneous_source, teacher_loss,
)
from retina_sim.vh2005 import NativeConeH1, SOURCE_VALUES

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/source_data/h1_model_teacher_v1'
HASH = canonical_hash({'test':'explicit_protocol_lineage'})


@pytest.fixture(scope='module')
def source():
    return make_homogeneous_source(ROOT,1000.)


def derived(source,tau):
    return DerivedHomogeneousTeacher.create(source,tau,HASH,source.model_hash)


def constructor(source,parameters):
    return NativeConeH1(parameters, source.cones, source.hcs, source.A, source.B,
        source.initial_conditions()[0],pooling_policy=source.pooling_policy,
        geometry_frame=source.geometry_frame,observation_interval_seconds=.0001)


def test_exact_source_bytes_and_108_complex_targets():
    receipt=verify_source_files(SOURCE)
    assert len(receipt)==4
    targets,rows=load_targets(SOURCE)
    assert len(rows)==108 and len(targets)==9
    assert all(len(t)==len(FREQUENCIES) for t in targets.values())
    assert all(np.all(np.isfinite(normalized(t))) for t in targets.values())


def test_mismatched_hash_is_rejected_without_repair(tmp_path):
    for name in SOURCE_FILES:
        (tmp_path/name).write_bytes((SOURCE/name).read_bytes())
    file=tmp_path/'smith2001_fig2_model_derived_targets.csv'
    file.write_bytes(file.read_bytes()+b'\n')
    with pytest.raises(ContractError,match='actual'):
        verify_source_files(tmp_path)


def test_source_lock_rejects_changed_tau_and_derived_profile_identity(source):
    records=tuple(replace(r,value=30.) if r.parameter_id=='VH2005_tau_h' else r for r in source.parameters.records)
    with pytest.raises(ContractError,match='mix/tune'):
        constructor(source,replace(source.parameters,records=records))
    profile=replace(source.parameters.profile,profile_id='VH2005_derived_HC_teacher_tau_h_v1')
    with pytest.raises(ContractError,match='Exact'):
        constructor(source,replace(source.parameters,profile=profile))
    with pytest.raises(TypeError):source.p['tau_h']=30.


def test_only_tau_h_derivative_changes_and_dynamic_pooled_aI_remains(source):
    before=dict(source.p)
    baseline,candidate=derived(source,20),derived(source,40)
    y=baseline.equilibrium.copy();y[-1]-=1.
    a,b=baseline.rhs(0,y,1000),candidate.rhs(0,y,1000)
    np.testing.assert_array_equal(a[:-1],b[:-1])
    assert b[-1]==pytest.approx(a[-1]/2,rel=1e-14)
    expected=1000*(y[8]-y[9])/(40*(y[6]/SOURCE_VALUES['V_I'])**SOURCE_VALUES['mu'])
    assert b[-1]==pytest.approx(expected,rel=1e-14)
    y[6]*=1.1
    changed=candidate.rhs(0,y,1000)
    assert changed[-1]==pytest.approx(b[-1]/1.1**SOURCE_VALUES['mu'])
    assert dict(source.p)==before==SOURCE_VALUES
    assert baseline.profile.source_parameter_qualified is False
    assert candidate.candidate_hash!=baseline.candidate_hash
    schema=json.loads((ROOT/'schemas/hc_derived_teacher_profile_v1.schema.json').read_text(encoding='utf8'))
    assert set(asdict(candidate.profile))==set(schema['required'])==set(schema['properties'])
    with pytest.raises(ContractError,match='schema/identity'):
        replace(candidate.profile,source_parameter_qualified=True)


@pytest.mark.parametrize('tau',[9.,41.,float('nan')])
def test_single_parameter_local_bounds(source,tau):
    with pytest.raises(ContractError,match='outside'):derived(source,tau)


def test_normalization_rejects_zero_and_removes_constant_sign_without_extra_sample():
    with pytest.raises(ContractError,match='no epsilon'):normalized([0.,1.])
    transfer=np.arange(1,13)+1j
    assert teacher_loss(transfer,-10*transfer)==pytest.approx(0,abs=1e-29)
    changed=transfer.copy();changed[-1]+=1
    assert teacher_loss(changed,transfer)==pytest.approx(abs(1/transfer[0])**2/11)


def test_full_state_jacobian_scale_stable_poles_and_exactly_one_readout_delay(source):
    student=derived(source,20)
    transfer,details=frequency_response(student)
    other,_=frequency_response(student,1e-6)
    assert max(abs(normalized(transfer)-normalized(other)))<5e-5
    assert details['maximum_pole_real_per_second']<0
    jac,drive,poles=linearize(student)
    assert jac.shape==(10,10) and len(poles)==10
    state=np.linalg.solve(2j*np.pi*FREQUENCIES[7]*np.eye(10)-jac,drive)
    assert transfer[7]/state[-1]==pytest.approx(np.exp(-2j*np.pi*20*.00282),rel=1e-12)
    assert max(abs(transfer-frequency_response(derived(source,40))[0]))>1e-5
    assert np.linalg.solve(-jac,drive)[-1]<0


def test_positive_background_and_complete_state_required(source):
    with pytest.raises(ContractError,match='Complete'):derived(source,20).rhs(0,[1],1000)
    with pytest.raises(ContractError,match='positive'):
        linearize(derived(make_homogeneous_source(ROOT,0),20))
