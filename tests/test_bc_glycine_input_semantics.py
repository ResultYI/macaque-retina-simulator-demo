import pytest
from retina_sim.contracts import ContractError
from retina_sim.bc_glycine_input_semantics import bind_off_glycine_input


def test_generic_diffuse_is_not_identified_by_channel_name():
    with pytest.raises(ContractError,match='Channel name cannot identify'):
        bind_off_glycine_input('AII_DB1_glycine','diffuse_BC_OFF',['generic:1','generic:2'])


def test_explicit_borrow_preserves_other_subtype_and_unknown_identity():
    r=bind_off_glycine_input('AII_DB1_glycine','diffuse_BC_OFF',['db2:1','generic:2'],
        target_subtypes=['DB2','UNRESOLVED'],borrowed_template='DB1',borrowing_reason='Uncalibrated comparator only')
    assert [x['target_subtype'] for x in r['records']]==['DB2','UNRESOLVED']
    assert [x['target_group_id'] for x in r['records']]==['db2:1','generic:2']
    assert all(not x['other_subtypes_forced_zero'] for x in r['records'])


def test_rejects_misaligned_identity_and_implicit_cross_template():
    with pytest.raises(ContractError):
        bind_off_glycine_input('AII_DB1_glycine','diffuse_BC_OFF',['a','b'],target_subtypes=['DB1'])
    with pytest.raises(ContractError):
        bind_off_glycine_input('AII_DB1_glycine','diffuse_BC_OFF',['a','a'],target_subtypes=['DB1','DB1'])
    with pytest.raises(ContractError):
        bind_off_glycine_input('AII_DB1_glycine','diffuse_BC_OFF',['a'],target_subtypes=['DB2'])
    r=bind_off_glycine_input('AII_DB1_glycine','diffuse_BC_OFF',['a'],target_subtypes=['DB1'])
    assert r['records'][0]['classification']=='declared_target_family'
    assert r['records'][0]['quantitative_conductance_calibrated'] is False
