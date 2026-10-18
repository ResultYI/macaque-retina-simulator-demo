"""Execution-only CUDA transport changes retain the complete joint model."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import json
import numpy as np
import pytest
from scipy import sparse
from retina_sim.assembly import CompiledLink,RetinaBuilder
from retina_sim.contracts import ContractError,Quantity,RunMode,SolverConfig
from retina_sim.gpu_compact import compact_release_layout,CompactCudaHybridRunner
from retina_sim.gpu_hybrid import CudaHybridRunner
from retina_sim.network import InputBundle

ROOT=Path(__file__).resolve().parents[1]


def test_all_consumer_kinds_private_remap_and_row_arithmetic():
    # Unsorted columns deliberately test preservation of each reduction order.
    a=sparse.csr_matrix(([1e16,1.,-1e16],[7,2,4],[0,3]),shape=(1,9))
    b=sparse.csr_matrix(([.5,.25],[1,7],[0,2]),shape=(1,9))
    links=tuple(CompiledLink(NS(source_component='native',source_port='release',target_component=cid,
        target_port=port),matrix) for cid,port,matrix in (('history','drive',a),('other','activation',b)))
    components={cid:NS(mechanism=kind,time_contract=NS(kind=clock)) for cid,kind,clock in
        (('history','HistoryBipolarTerminal','sampled_history'),('other','SynapticConductance','continuous'))}
    incoming={(link.specification.target_component,link.specification.target_port):[link] for link in links}
    network=NS(compiled=NS(links=links,components=components),_incoming=incoming)
    used,private,audit=compact_release_layout(network,'native',9)
    np.testing.assert_array_equal(used,[1,2,4,7]);release=np.arange(9,dtype=float)+.125
    assert {item['component'] for item in audit}=={'history','other'}
    for link in links:
        key=(link.specification.target_component,link.specification.target_port);new=private[key][0]
        assert new is not link and incoming[key][0] is link
        np.testing.assert_array_equal(new.matrix.data,link.matrix.data)
        np.testing.assert_array_equal(new.matrix.indptr,link.matrix.indptr)
        np.testing.assert_array_equal(used[new.matrix.indices],link.matrix.indices)
        np.testing.assert_array_equal(new.matrix@release[used],link.matrix@release)


@pytest.fixture(scope='module')
def gpu_fixture():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('Existing CUDA device required')
    from examples.full_network_fixture import full_fixture
    torch.set_num_threads(1);_,channels,audit=full_fixture(ROOT);spec=audit['spec']
    # QA graph for both arms: all nine native cones/HC retained, one release
    # column unused outside OPL. No scientific profile or frozen graph edited.
    links=tuple(replace(link,connections=replace(link.connections,
        pre_cell_id=tuple('cones:5' if cell=='cones:8' else cell for cell in link.connections.pre_cell_id)))
        if link.source_component=='outer' else link for link in spec.links)
    compiled=RetinaBuilder(audit['registry']).build(replace(spec,links=links),RunMode.NUMERIC_FIXTURE)
    old=CudaHybridRunner(compiled);new=CompactCudaHybridRunner(compiled)
    return torch,compiled,channels,old,new


def test_full_rhs_all_drives_and_source_matrices(gpu_fixture):
    torch,compiled,channels,old,new=gpu_fixture
    assert len(new.partition.used)==8 and new.partition.native.nc==9
    assert old.model.graph.graph_hash==new.model.graph.graph_hash
    assert old.model.parameters.parameter_hash==new.model.parameters.parameter_hash
    assert old.model.model_hash!=new.model.model_hash
    bundle=InputBundle({name:channel.at(.00037) for name,channel in channels.items()})
    initial=old._flat(old.initialize());perturbed=initial.copy()
    perturbed[:old.partition.offset]*=np.random.default_rng(72).uniform(.995,1.005,old.partition.offset)
    for values in (initial,perturbed,initial):
        device=torch.as_tensor(values.copy(),device='cuda')
        a=old.partition(.00037,device,bundle);b=new.partition(.00037,device,bundle)
        np.testing.assert_array_equal(a.cpu().numpy(),b.cpu().numpy())
        _,signals=old.model.evaluate(.00037,values,bundle)
        release=signals[('outer','release')].values
        _,private,_=compact_release_layout(old.model,'outer',9)
        for link in compiled.links:
            s=link.specification
            if s.source_component!='outer':continue
            mapped=next(value for value in private[(s.target_component,s.target_port)]
                        if value.specification is s)
            np.testing.assert_array_equal(link.matrix@release,mapped.matrix@release[new.partition.used])
    assert new.partition.layout['compact_release_bytes_per_RHS']==64


def test_actual_light_values_reuse_mutation_and_contracts(gpu_fixture):
    torch,_,channels,_,runner=gpu_fixture;p=runner.partition
    source=next(iter(channels.values())).at(0.)
    def q(values,unit=source.unit):return Quantity(values,unit,source.meaning,source.evidence_tag)
    first=q(source.values);device=p._light(first);uploads=p.statistics['light_uploads']
    assert p._light(q(first.values)) is device and p.statistics['light_uploads']==uploads
    # Read-only storage can be changed by a hostile/incorrect caller; do not
    # rely on identity or assume that a Quantity or time label remains constant.
    first.values.setflags(write=True);first.values[0]+=1.;first.values.setflags(write=False)
    assert p._light(first) is not device
    np.testing.assert_array_equal(p._light(first).cpu().numpy(),first.values)
    zero=q(np.zeros(9));negative_zero=q(np.full(9,-0.))
    device=p._light(zero);assert p._light(negative_zero) is not device
    values=runner._flat(runner.initialize());device=torch.as_tensor(values,device='cuda')
    with pytest.raises(ContractError):p(0.,device,InputBundle({'native_light':q(source.values,unit='Hz')}))
    with pytest.raises(ContractError):p(0.,device,InputBundle({'native_light':q(source.values[:8])}))
    with pytest.raises(ContractError):p(0.,device,InputBundle({}))


def test_joint_states_events_full_recording_rng_and_strict_restore(gpu_fixture):
    _,_,channels,old,new=gpu_fixture
    config=SolverConfig(rtol=1e-8,atol=1e-11,max_step=.0001,input_interpolation='analytic')
    left=old.initialize(config);right=new.initialize(config)
    with pytest.raises(ContractError):new.restore(old.checkpoint(left))
    with pytest.raises(ContractError):old.restore(new.checkpoint(right))
    times=np.array([0.,.00031,.00075,.00123,.0021])
    left,a=old.advance(left,channels,.0021,sample_times=times)
    right,b=new.advance(right,channels,.0021,sample_times=times)
    np.testing.assert_array_equal(old._flat(left),new._flat(right))
    for name in ('time_seconds','grid_origin_seconds','sample_index','component_tick_indices',
                 'spike_states','delay_states','link_delay_states','input_position'):
        assert json.dumps(getattr(left,name),sort_keys=True)==json.dumps(getattr(right,name),sort_keys=True)
    assert old.integration_metrics['nfev']==new.integration_metrics['nfev']
    assert a.outputs_with_units.keys()==b.outputs_with_units.keys()
    for key,value in a.outputs_with_units.items():
        np.testing.assert_array_equal(value.values,b.outputs_with_units[key].values)
        assert (value.unit,value.meaning,value.evidence_tag)==(b.outputs_with_units[key].unit,
            b.outputs_with_units[key].meaning,b.outputs_with_units[key].evidence_tag)
    payload=new.checkpoint(right);restored=new.restore(payload)
    assert new.checkpoint(restored)==payload
    first,_=new.advance(right,channels,.0026);second,_=new.advance(restored,channels,.0026)
    assert new.checkpoint(first)==new.checkpoint(second)


def test_named_opt_in_factory(gpu_fixture):
    from retina_sim.execution_backends import BACKENDS,CUDA_BACKENDS,CUDA_COMPACT,make_runner
    _,compiled,_,_,_=gpu_fixture
    assert CUDA_COMPACT in BACKENDS and CUDA_COMPACT in CUDA_BACKENDS
    assert isinstance(make_runner(compiled,CUDA_COMPACT),CompactCudaHybridRunner)


def test_complete_target_inner_rod_gap_and_history_mechanisms():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('Existing CUDA device required')
    from examples.full_target_network import full_target_network
    compiled,channels,_=full_target_network(ROOT,effective_cones=64,effective_rods=96,duration=.003,
        cone_route='native_shared_H1_approximation',bounds_mm=(.004,4.,-.04,.04))
    old=CudaHybridRunner(compiled);new=CompactCudaHybridRunner(compiled)
    config=SolverConfig(rtol=1e-8,atol=1e-11,max_step=.0001,
        input_interpolation=next(iter(channels.values())).interpolation)
    a=old.initialize(config);b=new.initialize(config);initial=old._flat(a)
    bundle=InputBundle({name:channel.at(.0001) for name,channel in channels.items()})
    device=torch.as_tensor(initial.copy(),device='cuda')
    np.testing.assert_array_equal(old.partition(.0001,device,bundle).cpu().numpy(),
                                  new.partition(.0001,device,bundle).cpu().numpy())
    a,x=old.advance(a,channels,.00035);b,y=new.advance(b,channels,.00035)
    np.testing.assert_array_equal(old._flat(a),new._flat(b))
    assert a.component_tick_indices==b.component_tick_indices
    assert json.dumps(a.spike_states,sort_keys=True)==json.dumps(b.spike_states,sort_keys=True)
    for key,value in x.outputs_with_units.items():
        np.testing.assert_array_equal(value.values,y.outputs_with_units[key].values)
    assert any(c.mechanism=='HistoryBipolarTerminal' for c in compiled.components.values())
    assert any(c.mechanism=='RodContactTRPM1' for c in compiled.components.values())
