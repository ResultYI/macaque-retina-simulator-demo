"""Empty AII conductance is explicit finite-graph accounting, not a driver."""
from types import SimpleNamespace as NS
import hashlib,json
import numpy as np
import pytest
from scipy import sparse
from retina_sim.contracts import ContractError,RunMode
from retina_sim.disk_circuit import GlobalDiskCircuit
from retina_sim.finite_input_support import EMPTY_SUM_POLICY,OPTION,declared_empty_rows


def setup_declaration(tmp_path):
    route={'path':'route.json','sha256':'route-identity'}
    graph={'schema':'frozen_global_disk_circuit_v1','routes':[{'route_id':'RBC_AII_AMPA',
        'kind':'primary_chemical','pre':'rod_BC','post':'AII_AC','binding':route}]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph),encoding='utf-8')
    binding={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    declaration={'policy':EMPTY_SUM_POLICY,'evidence_class':'project_approximation',
        'source_graph':binding,'source_route':route,'cell_ids':['b','d']}
    spec=NS(component_id='AII_AC',mechanism='ReceptorConductanceMembrane',
        options={'cell_kind':'AII',OPTION:{'excitation':declaration}})
    component=NS(inputs={'excitation':NS(unit='S',cell_ids=('a','b','c','d'))})
    incoming=[NS(specification=NS(source_component='RBC_AII_AMPA',source_port='conductance'),
        matrix=sparse.csr_matrix(([1.,1.],([0,2],[0,0])),shape=(4,2)))]
    return spec,component,np.array([1.,0.,1.,0.]),incoming,declaration


def test_only_explicit_actual_empty_rows_are_allowed_and_values_unchanged(tmp_path):
    spec,c,totals,incoming,_=setup_declaration(tmp_path);before=totals.copy()
    result=declared_empty_rows(spec,c,'excitation',totals,incoming,tmp_path,RunMode.REFERENCE)
    np.testing.assert_array_equal(result,[False,True,False,True]);np.testing.assert_array_equal(totals,before)
    spec.options={}
    assert not declared_empty_rows(spec,c,'excitation',totals,incoming,tmp_path,RunMode.REFERENCE).any()


@pytest.mark.parametrize('ids',[['b'],['b','c','d'],['d','b'],['b','b']])
def test_stale_or_overbroad_cell_declaration_rejected(tmp_path,ids):
    spec,c,totals,incoming,d=setup_declaration(tmp_path);d['cell_ids']=ids
    with pytest.raises(ContractError,match='actual zero input rows'):
        declared_empty_rows(spec,c,'excitation',totals,incoming,tmp_path,RunMode.REFERENCE)


@pytest.mark.parametrize('change',['full_physiology','wrong_unit','wrong_mechanism','other_route','changed_graph'])
def test_policy_cannot_be_used_to_hide_other_inputs_or_source_changes(tmp_path,change):
    spec,c,totals,incoming,d=setup_declaration(tmp_path);mode=RunMode.REFERENCE
    if change=='full_physiology':mode=RunMode.FULL_PHYSIOLOGY
    if change=='wrong_unit':c.inputs['excitation'].unit='V'
    if change=='wrong_mechanism':spec.mechanism='HistoryConductanceRGC'
    if change=='other_route':incoming[0].specification.source_component='unrelated'
    if change=='changed_graph':(tmp_path/'graph.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ContractError):declared_empty_rows(spec,c,'excitation',totals,incoming,tmp_path,mode)


def test_negative_sum_is_never_allowed_by_an_empty_row_declaration(tmp_path):
    spec,c,totals,incoming,_=setup_declaration(tmp_path);totals[0]=-1
    allowed=declared_empty_rows(spec,c,'excitation',totals,incoming,tmp_path,RunMode.REFERENCE)
    assert not allowed[0]


def test_cancelling_contacts_cannot_be_declared_empty(tmp_path):
    spec,c,totals,incoming,_=setup_declaration(tmp_path)
    incoming[0].matrix=sparse.csr_matrix(([1.,1.,-1.,1.],([0,1,1,2],[0,0,1,0])),shape=(4,2))
    with pytest.raises(ContractError,match='structurally empty'):
        declared_empty_rows(spec,c,'excitation',totals,incoming,tmp_path,RunMode.REFERENCE)


def test_primary_preflight_reports_missing_row_before_assembly_and_records_named_policy():
    circuit=object.__new__(GlobalDiskCircuit);circuit.binding={'path':'synthetic','sha256':'synthetic'}
    descriptor={'route_id':'RBC_AII_AMPA','kind':'primary_chemical','pre':'rod_BC','post':'AII_AC',
        'binding':{'sha256':'synthetic-route'}}
    circuit.record={'reference_id':'synthetic','output_populations':['AII_AC'],'routes':[descriptor]}
    circuit.table=lambda name:NS(query_indices=lambda *a,**kw:np.array([0]))
    circuit._expand_components=lambda *args:False
    circuit.route=lambda desc:NS(dependency_rows=lambda rows:np.empty(0,dtype=np.int64),
        selected_rows=lambda rows:(np.array([0]),sparse.csr_matrix((1,1)),[{'missing_degree':3}],['INCOMPLETE_NO_INPUT_WITHIN_SUPPORT']))
    strict=circuit.owner_view((-.01,.01,-.01,.01),outputs=('AII_AC',))
    project=circuit.owner_view((-.01,.01,-.01,.01),outputs=('AII_AC',),empty_chemical_input_policy=EMPTY_SUM_POLICY)
    assert not strict['runtime_ready'] and len(strict['unsupported_rows'])==1
    assert project['runtime_ready'] and len(project['declared_empty_chemical_sums'])==1
    np.testing.assert_array_equal(strict['selected_rows']['AII_AC'],project['selected_rows']['AII_AC'])
