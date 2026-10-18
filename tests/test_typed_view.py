"""Build/identity tests only; never prepare background or advance neural time."""
from copy import copy
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from retina_sim.atlas import AtlasROI
from retina_sim.atlas_population_network import atlas_population_network
from retina_sim.contracts import ContractError,Quantity
from retina_sim.geometry_identity import compiled_geometry_identity,require_geometry_identity
from retina_sim.typed_view import CompiledTypedView,TypedBoundaryError,compiled_typed_view

ROOT=Path(__file__).resolve().parents[1]
OUTPUTS=('midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON')

@pytest.fixture(scope='module')
def network():return atlas_population_network(ROOT,execution_profile='atlas_isetbio_geometry_v1')[0]

@pytest.fixture(scope='module')
def view(network):return compiled_typed_view(network)

def roi(view,bounds):return AtlasROI(bounds,view.query_domain,view.reference_id)

def test_overlapping_owners_share_exact_cells_coordinates_and_edges(network,view):
    a=view.owner_view(roi(view,(.34,.357,-.01,.01)),OUTPUTS)
    b=view.owner_view(roi(view,(.343,.36,-.01,.01)),OUTPUTS)
    shared=set(a['owners']['midget_RGC_ON'])&set(b['owners']['midget_RGC_ON'])
    assert shared and a['geometry_identity']==b['geometry_identity']==compiled_geometry_identity(network)
    common=set(a['closure']['nodes'])&set(b['closure']['nodes'])
    for population,roles in [('midget_RGC_ON',('rf','ipl_post_arbor')),
            ('midget_BC_ON',('rf','ipl_preterminal')),('cones',('rf','opl_pedicle'))]:
        cid='outer_native' if population=='cones' else population
        ids=sorted(gid for component,gid in common if component==cid and gid in view.populations[population][0])
        assert ids
        for role in roles:
            left=view.coordinate(population,role,ids);right=view.coordinate(population,role,ids)
            assert left.values.tobytes()==right.values.tobytes()
            with pytest.raises(ValueError):left.values.setflags(write=True)
    ea={e[2]:e for e in a['closure']['edges'] if e[0] in common and e[1] in common}
    eb={e[2]:e for e in b['closure']['edges'] if e[0] in common and e[1] in common}
    assert ea and ea==eb
    assert a['canonical_model_unchanged'] and b['canonical_model_unchanged']

def test_full_owner_closure_includes_native_recurrence_rod_chemistry_and_gap(view):
    result=view.owner_view(roi(view,view.owner_domain),OUTPUTS);closure=result['closure']
    assert result['owner_status']['small_bistratified_S_ON']=='EMPTY_SAMPLE'
    assert all(result['owners'][cid] for cid in OUTPUTS[:-1])
    assert {x[0] for x in closure['external_cells']}=={'native_light','rod_light'}
    assert {'A','B'}<={e[2][2] for e in closure['edges'] if e[2][0]=='native'}
    assert any(e[2][0]=='gap' for e in closure['edges'])
    components={n[0] for n in closure['nodes']}
    assert {'rod','rod_BC','AII_AC','RBC_AII_AMPA','AII_FMB_glycine'}<=components
    assert closure['exterior_recurrent_closure']=='NOT_CERTIFIED'

def test_unavailable_typed_roles_targets_and_exterior_are_explicit(view):
    with pytest.raises(TypedBoundaryError,match='COORDINATE_ROLE_UNAVAILABLE'):
        view.coordinate('midget_RGC_ON','soma')
    with pytest.raises(TypedBoundaryError,match='COORDINATE_ROLE_UNAVAILABLE'):
        view.coordinate('midget_RGC_ON','ipl_preterminal')
    with pytest.raises(TypedBoundaryError,match='TARGET_UNAVAILABLE'):
        view.owner_view(roi(view,view.owner_domain),('invented_cell_type',))
    with pytest.raises(TypedBoundaryError,match='EXTERIOR_CLOSURE_UNCERTIFIED'):
        view.dependency_closure((),require_exterior_closure=True)
    for bounds,query in [((.339,.36,-.01,.01),view.query_domain),(view.owner_domain,(.299,.4,-.05,.05))]:
        with pytest.raises(TypedBoundaryError,match='MODEL_DOMAIN_UNAVAILABLE'):
            view.owner_view(AtlasROI(bounds,query,view.reference_id),OUTPUTS)
    with pytest.raises(TypedBoundaryError,match='REFERENCE'):
        view.owner_view(AtlasROI(view.owner_domain,view.query_domain,view.reference_id,'opl_pedicle'),OUTPUTS)

def test_missing_nonzero_input_row_is_rejected(network):
    c=copy(network);links=list(c.links);target='midget_RGC_ON_excitatory_midget_BC_ON'
    for i,link in enumerate(links):
        if link.specification.target_component==target:
            matrix=link.matrix.copy();matrix.data[matrix.indptr[0]:matrix.indptr[1]]=0
            links[i]=replace(link,matrix=matrix)
    c.links=tuple(links);v=compiled_typed_view(c)
    gid=c.components[target].inputs['activation'].cell_ids[0]
    with pytest.raises(TypedBoundaryError,match='MISSING_INPUT_ROW'):
        v.dependency_closure(((target,gid),))

def test_geometry_identity_catches_opl_only_change_and_stale_view(network):
    c=copy(network);c.populations=dict(c.populations);old=c.populations['cones'];q=old.pedicle_positions
    moved=Quantity(q.values+np.array([.000001,0.]),q.unit,q.meaning,q.evidence_tag,q.coordinate_frame)
    v=compiled_typed_view(c);expected=compiled_geometry_identity(c)
    c.populations['cones']=replace(old,pedicle_positions=moved)
    np.testing.assert_array_equal(old.receptive_field_positions.values,c.populations['cones'].receptive_field_positions.values)
    with pytest.raises(ContractError,match='identity mismatch'):require_geometry_identity(expected,c)
    with pytest.raises(TypedBoundaryError,match='STALE_MODEL'):v.dependency_closure(())

def test_geometry_identity_catches_actual_weights_even_with_same_assembly_hash(network):
    c=copy(network);original=compiled_geometry_identity(c);links=list(c.links)
    m=links[0].matrix.copy();m.data[0]*=.99;links[0]=replace(links[0],matrix=m);c.links=tuple(links)
    assert c.assembly_hash==network.assembly_hash and original!=compiled_geometry_identity(c)
    with pytest.raises(ContractError,match='identity mismatch'):require_geometry_identity(original,c)

def test_unknown_component_needs_an_explicit_dependency_adapter(network):
    c=copy(network);c.components=dict(c.components);component=copy(c.components['midget_BC_ON'])
    component.mechanism='unknown_spatial_mixer';c.components['midget_BC_ON']=component
    with pytest.raises(TypedBoundaryError,match='INTERNAL_DEPENDENCY_UNSUPPORTED'):compiled_typed_view(c)
