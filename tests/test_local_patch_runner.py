"""Pure construction-IVP row restrictions; no neural initialization or forward run."""
from copy import deepcopy
import numpy as np
import pytest

from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_inner_dynamics import CONTACTS
from retina_sim.local_patch_runner import restrict_construction_payload, make_protocol
from retina_sim.spatial_group_model import POPULATIONS, RGC


@pytest.fixture
def source():
    states={(pop,'fixture','value'):np.arange(4,dtype=np.float64)+ordinal
            for ordinal,pop in enumerate(POPULATIONS)}
    states['cones','selected_original_z1_auxiliary','z1_aux']=np.array([8.,9.])
    pre={'individual_rod_RBC_contacts':'rods','RBC_AII_AMPA':'rod_BC',
         'AII_FMB_glycine':'AII_AC','AII_DB1_glycine':'AII_AC'}
    for component in CONTACTS:states[pre[component],component,'m']=np.arange(5,dtype=np.float64)
    parameters=dict(populations={pop:dict(V0=np.arange(4,dtype=np.float64),kernel=np.array([1.,2.,3.]))
        for pop in POPULATIONS}, contacts={component:{} for component in CONTACTS},
        route_gains=dict(unchanged_gain=1.2),rod_phototransduction=dict(timeStep=.0001))
    for component in ('AII_FMB_glycine','AII_DB1_glycine'):
        parameters['contacts'][component]['tau_decay']=np.arange(5,dtype=np.float64)+10.
    spikes={pop:dict(history=np.zeros((3,4)),conditional_rate_Hz=np.arange(4,dtype=np.float64),
        pending_events=[],last_bin_counts=None,rng_state=deepcopy(np.random.PCG64(1).state)) for pop in RGC}
    parent=dict(states=states,inner_parameters=parameters,spike_states=spikes,time_seconds=4.993,
        native_inputs={748:np.arange(4,dtype=np.float64)},rod_inputs={748:np.arange(4,dtype=np.float64)},
        observation_prehistory=dict(history_times=np.array([4.993]),history_values=np.arange(4,dtype=np.float64)[None,:],
            prehistory=np.arange(4,dtype=np.float64),delay_seconds=.00282),
        input_reference=dict(native_background_td=np.arange(4,dtype=np.float64),rod_background_rstar_s=.75))
    rows={pop:np.array([1,3],dtype=np.int64) for pop in POPULATIONS}
    contacts={component:np.array([0,2,4],dtype=np.int64) for component in CONTACTS}
    return parent,rows,contacts,dict(cones=np.array([1],dtype=np.int64))


def test_restricts_original_cells_contacts_inputs_and_histories_without_mutating_source(source):
    parent,rows,contacts,aux=source; before=canonical_hash(parent)
    result=restrict_construction_payload(parent,rows,contacts,aux,rng_seed=34)
    assert canonical_hash(parent)==before
    for key,value in parent['states'].items():
        selected=contacts[key[1]] if key[1] in CONTACTS else aux['cones'] if key[1]=='selected_original_z1_auxiliary' else rows[key[0]]
        np.testing.assert_array_equal(result['states'][key],value[selected])
        assert not np.shares_memory(result['states'][key],value)
    for pop in POPULATIONS:
        np.testing.assert_array_equal(result['inner_parameters']['populations'][pop]['V0'],[1.,3.])
        np.testing.assert_array_equal(result['inner_parameters']['populations'][pop]['kernel'],[1.,2.,3.])
    for component in ('AII_FMB_glycine','AII_DB1_glycine'):
        np.testing.assert_array_equal(result['inner_parameters']['contacts'][component]['tau_decay'],[10.,12.,14.])
    for category in ('native_inputs','rod_inputs'):
        np.testing.assert_array_equal(result[category][748],[1.,3.])
    np.testing.assert_array_equal(result['observation_prehistory']['history_values'],[[1.,3.]])
    for pop in RGC:
        assert result['spike_states'][pop]['history'].shape==(3,2)
        np.testing.assert_array_equal(result['spike_states'][pop]['conditional_rate_Hz'],[1.,3.])
    assert result['patch_initialization']['parent_rng_equivalence_claimed'] is False
    assert result['patch_initialization']['classification']=='AGGREGATE_SUBMODEL_IVP_RESTRICTION'


def test_patch_rng_stream_is_seeded_reproducibly_with_original_pcg64_population_order(source):
    parent,rows,contacts,aux=source
    first=restrict_construction_payload(parent,rows,contacts,aux,rng_seed=34)
    second=restrict_construction_payload(parent,rows,contacts,aux,rng_seed=34)
    assert canonical_hash(first)==canonical_hash(second)
    for ordinal,pop in enumerate(RGC):
        expected=np.random.PCG64(34+100+ordinal).state
        assert canonical_hash(first['spike_states'][pop]['rng_state'])==canonical_hash(expected)
    changed=restrict_construction_payload(parent,rows,contacts,aux,rng_seed=35)
    assert canonical_hash(first['spike_states'])!=canonical_hash(changed['spike_states'])
    assert canonical_hash(first['states'])==canonical_hash(changed['states'])


@pytest.mark.parametrize('rows',[np.array([3,1],np.uint64),np.array([1,1]),np.array([4]),np.array([],np.int64),np.array([True])])
def test_invalid_population_row_identity_is_rejected(source,rows):
    parent,populations,contacts,aux=source; populations['rods']=rows
    with pytest.raises(ContractError):restrict_construction_payload(parent,populations,contacts,aux,rng_seed=2)


def test_unknown_array_parameter_is_not_guessed_from_shape(source):
    parent,rows,contacts,aux=source
    parent['inner_parameters']['populations']['rods']['unregistered']=np.arange(4.)
    with pytest.raises(ContractError,match='Unregistered array parameter'):
        restrict_construction_payload(parent,rows,contacts,aux,rng_seed=2)


def test_advanced_sampler_cannot_be_mislabelled_as_construction_ivp(source):
    parent,rows,contacts,aux=source
    parent['spike_states'][RGC[0]]['pending_events']=[dict(time_seconds=4.994)]
    with pytest.raises(ContractError,match='unadvanced construction sampler'):
        restrict_construction_payload(parent,rows,contacts,aux,rng_seed=2)


def test_protocol_preserves_source_clocks_tolerances_and_cached_input_identity(source):
    protocol=make_protocol(source[0])
    assert protocol['start_seconds']==4.993 and protocol['stop_seconds']==4.993+.0011
    assert protocol['checkpoint_seconds']==4.993+.0005
    assert protocol['input_frame_indices']==[748]
    assert protocol['input_fps']==150. and protocol['rod_step_seconds']==.0001
    assert protocol['BC_RGC_period_seconds']==.001 and protocol['native_observation_delay_seconds']==.00282
    assert protocol['reference']==dict(method='joint_DOP853',rtol=1e-6,atol=1e-9,max_step=.0001)
    assert protocol['qualification']['fit_eligible'] is False
    for duration in (True,0.,-.001,float('nan'),.2):
        with pytest.raises(ContractError):make_protocol(source[0],duration_seconds=duration)


def test_selected_cone_auxiliary_port_uses_auxiliary_rows_not_large_cone_group_indices():
    from types import SimpleNamespace
    from retina_sim.local_patch_runner import _selected_snapshot
    sizes={pop:50000 if pop=='cones' else 1 for pop in POPULATIONS}
    populations={pop:SimpleNamespace(group_ids=np.arange(count),counts=np.ones(count,dtype=np.int64))
        for pop,count in sizes.items()}
    populations['cones'].recording_auxiliary_group_rows=np.array([46504,3],dtype=np.int64)
    core={pop:np.array([46504 if pop=='cones' else 0],dtype=np.int64) for pop in POPULATIONS}
    assembly=SimpleNamespace(core_population_rows=core,populations=populations,routes={},
        parent_population_rows={pop:np.arange(count) for pop,count in sizes.items()})
    state_key=('cones','selected_original_z1_auxiliary','z1_aux')
    snapshot=dict(time_seconds=4.993,side='right',event_generation=1,
        states={state_key:np.array([11.,22.])},rhs={state_key:np.array([.1,.2])},
        ports={('cones','z1_aux'):np.array([11.,22.]),('cones','Vis'):np.arange(50000,dtype=np.float64)})
    selected=_selected_snapshot(assembly,snapshot)
    np.testing.assert_array_equal(selected['states'][state_key],[11.])
    np.testing.assert_array_equal(selected['ports']['cones','z1_aux'],[11.])
    np.testing.assert_array_equal(selected['ports']['cones','Vis'],[46504.])
    np.testing.assert_array_equal(selected['port_rows']['cones','z1_aux']['local_auxiliary_rows'],[0])
    np.testing.assert_array_equal(selected['port_rows']['cones','z1_aux']['local_cone_group_rows'],[46504])
    np.testing.assert_array_equal(selected['port_rows']['cones','z1_aux']['parent_cone_group_rows'],[46504])
