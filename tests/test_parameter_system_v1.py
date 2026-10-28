from dataclasses import replace
import numpy as np
import pytest

from retina_sim.contracts import ContractError
from retina_sim.parameter_system import (
    RunContext, PopulationIndex, Evidence, Applicability, AggregationRule,
    ParameterRecord, IntrinsicParameterRegistry, ParameterResolver, CalibrationRecord)


@pytest.fixture
def context():
    return RunContext("macaque_effective_source","unknown",{"calibrated":False},
        {"policy_id":"B","luminance":10.,"luminance_unit":"td"},{"V":"V"},
        {"id":"input"},{"rod":.0001,"RGC":.001},{"id":"local"},42)


@pytest.fixture
def pop():
    return PopulationIndex("RGC",("g1","g2"),("ON","ON"),np.array([2,3]),"coarse_group",
        np.array([[1.,0.],[2.,0.]]),"mm",np.array([1.,2.]),np.zeros(2),(None,None),
        ("soma","soma"),"artifact",coordinate_frame="functional_visual_field")


def record(value=60e-12,**kwargs):
    base=dict(parameter_id="C",mechanism="passive",cell_types=("ON",),quantity="C",
        value_spec={"rule":"type_constant","value":value,"layout":"broadcast"},unit="F",
        applicability=Applicability(),evidence=Evidence("project_approximation","old_profile","profile/C"),
        binding_subject="RGC",aggregation=AggregationRule("extensive"))
    base.update(kwargs);return ParameterRecord(**base)


def test_broadcast_and_no_fake_eccentricity_curve(context,pop):
    resolver=ParameterResolver(IntrinsicParameterRegistry([record()]),context,{"RGC":pop})
    x,_=resolver.resolve("RGC","C","F")
    assert x.dtype==np.float64 and not x.flags.writeable
    np.testing.assert_array_equal(x,[60e-12,60e-12])


def test_once_only_extensive_and_heterogeneous_sum():
    rule=AggregationRule("extensive")
    labels=dict(cell_types=["ON","ON"],backgrounds=["B","B"])
    np.testing.assert_array_equal(rule.aggregate([60e-12,60e-12],[0,2],**labels),[2*60e-12])
    np.testing.assert_array_equal(rule.aggregate([10e-12,20e-12],[0,2],**labels),[30e-12])
    with pytest.raises(ContractError):
        AggregationRule("already_aggregated_contact").aggregate([1.,1.],[0,2])


def test_reversal_not_multiplied_and_leak_weighted():
    rule=AggregationRule("leak_weighted_reversal",assumptions=("equipotential",))
    labels=dict(cell_types=["ON","ON"],backgrounds=["B","B"])
    np.testing.assert_array_equal(rule.aggregate([-.065,-.065],[0,2],leak=[1.,2.],**labels),[-.065])
    np.testing.assert_allclose(rule.aggregate([-.06,-.08],[0,2],leak=[1.,3.],**labels),[-.075])
    with pytest.raises(ContractError):AggregationRule("leak_weighted_reversal").aggregate([1.,2.],[0,2])


def test_nonlinear_mixed_type_and_background_refused():
    with pytest.raises(ContractError):AggregationRule().aggregate([.1,.2],[0,2])
    with pytest.raises(ContractError):AggregationRule("extensive").aggregate([1.,1.],[0,2],cell_types=["ON","OFF"])
    with pytest.raises(ContractError):AggregationRule("extensive").aggregate([1.,1.],[0,2],backgrounds=["B1","B2"])


def test_matrix_and_ragged_type_history_are_not_flattened(context,pop):
    r=record(value=[[2.,-1.],[-1.,2.]],unit="S",quantity="axial_matrix",parameter_id="matrix")
    x,_=ParameterResolver(IntrinsicParameterRegistry([r]),context,{"RGC":pop}).resolve("RGC","axial_matrix","S")
    assert x.shape==(2,2,2)
    assert np.array_equal(x[0],[[2.,-1.],[-1.,2.]])
    # Different types/subjects keep separate kernel shapes; no forced global stack.
    q=replace(pop,population_id="OFF",ids=("o1","o2"),cell_types=("OFF","OFF"))
    h1=record(value=[1.,2.,3.],quantity="history",parameter_id="h1",unit="V")
    h2=record(value=[1.,2.],quantity="history",parameter_id="h2",unit="V",
              binding_subject="OFF",cell_types=("OFF",))
    resolver=ParameterResolver(IntrinsicParameterRegistry([h1,h2]),context,{"RGC":pop,"OFF":q})
    assert resolver.resolve("RGC","history","V")[0].shape==(2,3)
    assert resolver.resolve("OFF","history","V")[0].shape==(2,2)


def test_unit_domain_conflict_and_missing_fail_closed(context,pop):
    with pytest.raises(ContractError):record(unit="")
    with pytest.raises(ContractError):ParameterResolver(IntrinsicParameterRegistry([record()]),context,{"RGC":pop}).resolve("RGC","C","S")
    with pytest.raises(ContractError):ParameterResolver(IntrinsicParameterRegistry([record(applicability=Applicability(runtime_eccentricity_deg=(0.,.5)))]),context,{"RGC":pop}).resolve("RGC","C","F")
    with pytest.raises(ContractError):ParameterResolver(IntrinsicParameterRegistry([record(),record(parameter_id="other")]),context,{"RGC":pop}).resolve("RGC","C","F")
    with pytest.raises(ContractError):ParameterResolver(IntrinsicParameterRegistry(),context,{"RGC":pop}).resolve("RGC","C","F")


def test_safe_position_rule_no_extrapolation(context,pop):
    v=dict(rule="eccentricity_function",axis="eccentricity_deg",axis_unit="deg",
        anchors=[1.,2.],values=[10.,20.],interpolation="linear",coordinate_frame="functional_visual_field",
        layout="per_group")
    r=record(value_spec=v)
    x,_=ParameterResolver(IntrinsicParameterRegistry([r]),context,{"RGC":pop}).resolve("RGC","C","F")
    np.testing.assert_array_equal(x,[10.,20.])
    bad=replace(pop,eccentricity_deg=np.array([1.,3.]))
    with pytest.raises(ContractError):ParameterResolver(IntrinsicParameterRegistry([r]),context,{"RGC":bad}).resolve("RGC","C","F")


def test_derived_missing_and_cycle(context,pop):
    d=dict(rule="derived",operation="sum",dependencies=[dict(subject="RGC",quantity="C",unit="F")])
    with pytest.raises(ContractError):ParameterResolver(IntrinsicParameterRegistry([record(value_spec=d)]),context,{"RGC":pop}).resolve("RGC","C","F")
    d["dependencies"][0]["quantity"]="missing"
    with pytest.raises(ContractError):ParameterResolver(IntrinsicParameterRegistry([record(value_spec=d)]),context,{"RGC":pop}).resolve("RGC","C","F")


def test_conditional_scope_is_not_silent_fallback(context,pop):
    r=record(value_spec=dict(rule="conditional_override",value=1.),
        applicability=Applicability(conditional={"biological_cell_id":"Cell6"}))
    with pytest.raises(ContractError):ParameterResolver(IntrinsicParameterRegistry([r]),context,{"RGC":pop}).resolve("RGC","C","F")


def test_calibration_record_cannot_be_mutated():
    r=CalibrationRecord("id",("C",),{"groups":[0,1,2]},{"model":"fixed"},{"loss":"frozen"},
        {"C":1.},{"scope":"unknown"},("existing",),{"status":"inconclusive"},"conditional",
        ("not physiological",),"explicit user")
    before=r.identity
    with pytest.raises(TypeError):r.parameters["C"]=2.
    assert r.identity==before and r.qualification=="conditional"


def test_registry_and_calibration_file_identity(tmp_path):
    registry=IntrinsicParameterRegistry([record()])
    path=tmp_path/"registry.json";registry.write_json(path)
    assert IntrinsicParameterRegistry.read_json(path).identity==registry.identity
    with pytest.raises(FileExistsError):registry.write_json(path)
    r=CalibrationRecord("id",("C",),{"groups":[0,1,2]},{"model":"fixed"},{},
        {"C":1.},{},("existing",),{},"conditional",("not physiological",),"user")
    cp=tmp_path/"calibration.json";r.write_immutable(cp)
    assert CalibrationRecord.read_immutable(cp).identity==r.identity
    with pytest.raises(FileExistsError):r.write_immutable(cp)


def test_derived_unit_error_rejected_before_use(context,pop):
    g=record(value=2.,parameter_id="g",quantity="g",unit="S")
    bad=record(value_spec=dict(rule="derived",operation="ratio",dependencies=[
        dict(subject="RGC",quantity="g",unit="S"),dict(subject="RGC",quantity="g",unit="S")]))
    with pytest.raises(ContractError):
        ParameterResolver(IntrinsicParameterRegistry([g,bad]),context,{"RGC":pop}).resolve("RGC","C","F")
