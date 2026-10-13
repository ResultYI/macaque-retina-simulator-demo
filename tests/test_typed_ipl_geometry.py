"""Approved effective geometry contracts, and a native short integration."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from retina_sim.ipl_geometry import (TYPED,RESEARCH,LEGACY,read_ipl_geometry,freeze_centroid_anchors,
    local_typed_sites,widefield_grid,typed_ipl_points)
from retina_sim.spatial_profiles import bounded_matrix
from retina_sim.contracts import ContractError,RunMode
from retina_sim.core_sources import load_core_registry
from retina_sim.assembly import RetinaBuilder,ModelSpec,ComponentSpec,ExternalInput
from examples.full_network_fixture import population,fixture_profile,q
from retina_sim.spatial import InputChannel
from retina_platform.config import RunConfig
ROOT=Path(__file__).resolve().parents[1]

def test_named_profile_and_old_config_migration():
    p=read_ipl_geometry(ROOT,TYPED)
    assert p['hard_support_mm']==.04 and p['widefield_AC_sampling']['spacing_mm']==.04/3
    assert p['boundary']['new_external_background']=='NOT_ENABLED'
    assert RunConfig().ipl_geometry_profile==RESEARCH
    assert RunConfig.from_dict({'execution_profile':'source_constrained_approximation'}).ipl_geometry_profile==LEGACY
    assert RunConfig.from_dict(RunConfig().to_dict()).ipl_geometry_profile==RESEARCH
    research=RunConfig.from_dict({'execution_profile':RESEARCH})
    assert research.ipl_geometry_profile==RESEARCH and research.owner_x_mm==.35
    assert research.hc_arbor_profile=='W2000_equal_area_bounded_edges_047_104'

def test_centroid_pool_is_frozen_and_final_query_is_independent():
    pre=np.array([[-.03,0],[.01,0],[.045,0]])
    anchors,records=freeze_centroid_anchors([[0,0]],pre,('a','b','c'))
    np.testing.assert_allclose(anchors,[[-.01,0]])
    assert records[0]['candidate_bc_ids']==['a','b']
    # c is never pulled into the seed pool to repair a deficit. Final query
    # uses the entire real presynaptic pool at the frozen final centre.
    W,rows=bounded_matrix(pre,anchors,8,support_mm=.04)
    assert rows[0]['missing_degree']==6 and W.nnz==2
    with pytest.raises(ContractError,match='ANCHOR_UNRESOLVED'):
        freeze_centroid_anchors([[.2,0]],pre,('a','b','c'))

def test_typed_local_units_do_not_borrow_rbc_count_and_keep_site_identity():
    sites=[('same',[.3,0],[.7,0]),('same',[.3,0],[.7,0]),('other',[.31,0],[.71,0])]
    a=local_typed_sites(sites,'local_feedforward_AC',(.69,.72,-.01,.01),profile_hash='x'*64)
    b=local_typed_sites(sites,'local_crossover_AC',(.69,.72,-.01,.01),profile_hash='x'*64)
    crop=local_typed_sites(sites,'local_feedforward_AC',(.705,.72,-.01,.01),profile_hash='x'*64)
    assert len(a[0])==2 and not set(a[0])&set(b[0])
    assert crop[0][0] in a[0]
    np.testing.assert_array_equal(a[2],b[2])

def test_global_mesh_crop_identity_and_eight_candidates_inside_hard_support():
    a,points,_=widefield_grid((-.05,.05,-.05,.05))
    b,small,_=widefield_grid((-.02,.02,-.02,.02))
    lookup=dict(zip(a,points))
    for cell,p in zip(b,small):np.testing.assert_array_equal(p,lookup[cell])
    targets=np.array([[0,0],[.006,.004],[-.009,-.006]])
    W,rows=bounded_matrix(points,targets,8,support_mm=.04)
    assert all(r['actual_degree']==8 and r['maximum_distance_mm']<=.04 for r in rows)
    np.testing.assert_allclose(np.asarray(W.sum(1)).ravel(),1)
    with pytest.raises(ContractError,match='no thinning'):widefield_grid((0,1,0,1),max_cells=10)

def test_role_accessor_rejects_soma_as_ipl_dendrite():
    p=population('r','RGC',('RGC',),np.array([[.3,0]]),('center',),'p','Macaque_software_fixture',np.array([[.8,0]]))
    p=replace(p,dendritic_positions=None)
    with pytest.raises(ContractError,match='soma/RF fallback forbidden'):typed_ipl_points(p,'postsynaptic')

def test_native_rgc_short_step_with_new_mesh_weights_preserves_shared_rng():
    # This is an explicit engineering integration fixture, not source anatomy.
    registry=load_core_registry(ROOT);ids,points,_=widefield_grid((-.05,.05,-.05,.05))
    W,rows=bounded_matrix(points,np.array([[0,0]]),8,support_mm=.04)
    ge=np.asarray(W@np.full(len(points),1e-9)).ravel()
    entries={
        'C':q([60e-12],'F','membrane_capacitance'),'gL':q([2e-9],'S','leak_conductance'),
        'EL':q([-.065],'V','leak_reversal'),'EE':q([0.],'V','excitatory_reversal'),'EI':q([-.08],'V','inhibitory_reversal'),
        'V0':q([-.065],'V','initial_membrane_voltage'),'alpha':q([150.],'Hz','spike_intensity_scale'),
        'mu':q([-.07],'V','spike_voltage_offset'),'beta':q([.005],'V','spike_voltage_scale'),
        'spike_history_kernel':q([-.012,-.008,-.004],'V','strict_past_spike_history_kernel'),
        'initial_spike_history':q(np.zeros((3,1)),'dimensionless','spike_count_history'),
        'spike_interval':q(.001,'s','spike_sampling_interval')}
    pid='typed_native_fixture';fixture_profile(registry,pid,1,
        {key:(value.values,value.unit,value.meaning) for key,value in entries.items()})
    pop=population('r','RGC',('RGC',),np.array([[.3,0]]),('center',),pid,'Macaque_software_fixture')
    spec=ModelSpec('typed_native_short',(pop,),(ComponentSpec('r','HistoryConductanceRGC',{'cells':'r'},pid,{'rng_seed':42}),),
        external_inputs=(ExternalInput('e','r','excitation'),ExternalInput('i','r','inhibition')),
        required_mechanisms=('RGC_spike_history',))
    c=RetinaBuilder(registry).build(spec,RunMode.NUMERIC_FIXTURE);runner=c.runner();s=runner.initialize()
    import copy
    original_rng=copy.deepcopy(s.spike_states['r']['rng_state'])
    channels={name:InputChannel(0,.0012,lambda t,q0=value:q0,value.unit,value.meaning,
        'analytic','synthetic_fixture',{'fixture':True})
        for name,value in [('e',q(ge,'S','excitatory_conductance')),('i',q([0.],'S','inhibitory_conductance'))]}
    end,result=runner.advance(s,channels,.0012)
    assert end.time_seconds==.0012 and np.all(np.isfinite(end.ordered_state_arrays[0]))
    assert end.ordered_state_arrays[0][0]>-.065
    assert 'rng_state' in end.spike_states['r'] and rows[0]['actual_degree']==8
    assert end.spike_states['r']['rng_state']!=original_rng


def test_view_exports_owner_and_real_dendrite_without_pruning_population():
    from types import SimpleNamespace
    from retina_platform.backend import _port_cells
    p=population('r','RGC',('ON','OFF'),np.array([[.3,0],[.31,0]]),('center','center'),
        'fixture','Macaque_software_fixture',np.array([[.8,0],[.81,0]]))
    p=replace(p,dendritic_positions=q([[.7,0],[.71,0]],'mm','dendritic_position','IPL'))
    c=SimpleNamespace(populations={'r':p})
    meta=_port_cells(c,tuple(reversed(p.cells.cell_id)),{'r':[True,False]})
    assert meta['is_owner']==[False,True] and len(p.cells.cell_id)==2
    assert meta['geometry']['IPL']['values']==[[.71,0],[.7,0]]
    assert meta['geometry']['Soma']['values']==[[.81,0],[.8,0]]
