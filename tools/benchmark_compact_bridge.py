"""Bounded compact-transport evidence: small full execution or frozen topology.

No fresh background, scientific graph change or whole-grid speed extrapolation.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def frozen():
    from retina_sim.disk_circuit import GlobalDiskCircuit
    from tools.audit_frozen_closure_support import SparseReader
    configuration='configurations/atlas_human_borrowed_disk3mm_floor_input_v2.json'
    cfg=json.loads((ROOT/configuration).read_bytes());circuit=GlobalDiskCircuit(ROOT,cfg['global_graph'])
    view=circuit.owner_view((.96,1.04,-.04,.04));selected=view['selected_rows'];reader=SparseReader(circuit)
    cones=selected['cones'];parts=[];consumers=[];checks=[]
    for descriptor in circuit.record['routes']:
        if descriptor['pre']!='cones' or descriptor['kind']=='native':continue
        posts=selected.get(descriptor['post'],np.empty(0,dtype=np.int64))
        if not len(posts):continue
        matrix=reader.rows(descriptor,posts)
        if np.any(matrix.data==0):raise AssertionError('Frozen outward route contains explicit zero edges')
        used=np.unique(matrix.indices);parts.append(used);checks.append(matrix)
        consumers.append({'population':descriptor['post'],'route':descriptor['route_id'],
            'route_binding':descriptor['binding'],'post_rows':len(posts),'edges':matrix.nnz,
            'unique_native_columns':len(used)})
    used=np.unique(np.concatenate(parts));local=np.searchsorted(cones,used)
    np.testing.assert_array_equal(cones[local],used)
    from scipy import sparse
    # Arithmetic/order proof on all actual frozen outward rows; this is a
    # topology test vector, not a physiological neural derivative.
    release=np.linspace(.125,1.25,circuit.record['tables']['cones']['count'])
    for m in checks:
        remap=sparse.csr_matrix((m.data.copy(),np.searchsorted(used,m.indices),m.indptr.copy()),
            shape=(m.shape[0],len(used)))
        np.testing.assert_array_equal(m@release,remap@release[used])
    return {'status':'ACTUAL_FROZEN_OUTWARD_TRANSPORT_LAYOUT_PASS','scope':'Frozen graph/CSR only; no neural step or whole-grid wall-time measurement',
        'configuration':configuration,'graph':circuit.binding,'selected_nodes':view['global_node_count'],
        'full_native_cones':len(cones),'compact_release_cells':len(used),
        'legacy_release_D2H_bytes_per_RHS':len(cones)*8,'compact_release_D2H_bytes_per_RHS':len(used)*8,
        'transport_bytes_reduction_factor':len(cones)/len(used),'all_selected_outward_source_routes':consumers,
        'all_rows_coefficients_reduction_order_equal':True,'entire_native_feedback_and_states_retained':True,
        'whole_grid_runtime_speedup':'NOT_MEASURED; bytes ratio is not wall-time speedup',
        'background_reuse_or_preparation':'NONE'}


def fixture():
    import torch
    from dataclasses import replace
    from examples.full_network_fixture import full_fixture
    from retina_sim.assembly import RetinaBuilder
    from retina_sim.contracts import RunMode,SolverConfig
    from retina_sim.gpu_compact import CompactCudaHybridRunner
    from retina_sim.gpu_hybrid import CudaHybridRunner
    from retina_sim.network import InputBundle
    if not torch.cuda.is_available():raise RuntimeError('Existing CUDA device required')
    torch.set_num_threads(1);_,channels,audit=full_fixture(ROOT);spec=audit['spec']
    links=tuple(replace(link,connections=replace(link.connections,
        pre_cell_id=tuple('cones:5' if cell=='cones:8' else cell for cell in link.connections.pre_cell_id)))
        if link.source_component=='outer' else link for link in spec.links)
    compiled=RetinaBuilder(audit['registry']).build(replace(spec,links=links),RunMode.NUMERIC_FIXTURE)
    old=CudaHybridRunner(compiled);new=CompactCudaHybridRunner(compiled)
    solver=SolverConfig(rtol=1e-6,atol=1e-9,max_step=.0001,input_interpolation='analytic')
    a=old.initialize(solver);b=new.initialize(solver);values=old._flat(a)
    device=torch.as_tensor(values.copy(),device='cuda');bundle=InputBundle({name:channel.at(.00037) for name,channel in channels.items()})
    # Warm both kernels and Numba dispatcher before the one measured pair.
    x=old.partition(.00037,device,bundle).cpu().numpy();y=new.partition(.00037,device,bundle).cpu().numpy()
    np.testing.assert_array_equal(x,y)
    rhs={}
    for label,runner in (('legacy',old),('compact',new)):
        torch.cuda.synchronize();tick=time.perf_counter()
        for _ in range(32):runner.partition(.00037,device,bundle)
        torch.cuda.synchronize();rhs[label]=time.perf_counter()-tick
    sample=np.array([0.,.00031,.00075,.00123,.0021]);arms={};end=[];outputs=[]
    # Identical independent initial state/input, original event/history readout.
    for label,runner,state in (('legacy',old,a),('compact',new,b)):
        before=dict(runner.integration_metrics);tick=time.perf_counter()
        state,result=runner.advance(state,channels,.0021,sample_times=sample)
        arms[label]={'wall_seconds':time.perf_counter()-tick,
            **{key:runner.integration_metrics[key]-before[key] for key in ('solver_calls','nfev','integration_wall_seconds')}}
        end.append(state);outputs.append(result)
    np.testing.assert_array_equal(old._flat(end[0]),new._flat(end[1]))
    for name in ('time_seconds','grid_origin_seconds','sample_index','component_tick_indices','spike_states',
                 'delay_states','link_delay_states','input_position'):
        assert json.dumps(getattr(end[0],name),sort_keys=True)==json.dumps(getattr(end[1],name),sort_keys=True)
    for key,value in outputs[0].outputs_with_units.items():
        other=outputs[1].outputs_with_units[key];np.testing.assert_array_equal(value.values,other.values)
        assert (value.unit,value.meaning,value.evidence_tag)==(other.unit,other.meaning,other.evidence_tag)
    assert arms['legacy']['nfev']==arms['compact']['nfev']
    assert arms['legacy']['solver_calls']==arms['compact']['solver_calls']
    return {'status':'SMALL_FULL_JOINT_COMPACT_EXECUTION_PAIR_PASS','scope':'Existing software QA model with one unused outward release column; not macaque physiology or whole-grid timing',
        'cells':sum(len(p.cells.cell_id) for p in compiled.populations.values()),'states':len(values),
        'recorded_window_seconds':.0021,'arms':arms,'short_window_speedup':arms['legacy']['wall_seconds']/arms['compact']['wall_seconds'],
        'RHS_32_repetitions_seconds':rhs,'full_RHS_states_outputs_units_clocks_rng_history_exact':True,
        'full_joint_layout':new.partition.layout,'transport_statistics':new.partition.statistics,
        'light':'analytic changing light; no false ZOH reuse',
        'new_explicit_backend_identity':new.model.model_hash,'legacy_backend_identity':old.model.model_hash,
        'checkpoint_migration':False,'fresh_background_preparation':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scope',choices=('fixture','frozen'),required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args();output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():raise ValueError('Use a fresh project output directory')
    output.mkdir(parents=True);tick=time.perf_counter();result=(fixture if args.scope=='fixture' else frozen)()
    result['total_wall_seconds']=time.perf_counter()-tick
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
