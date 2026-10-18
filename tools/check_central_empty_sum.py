"""One central frozen-graph assembly and100us CPU/CUDA check, without ISET/prep."""
from pathlib import Path
import hashlib,json,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_platform.config import RunConfig
from retina_sim.atlas_population_network import atlas_population_network
from retina_sim.contracts import SolverConfig
from retina_sim.execution_backends import CUDA,make_runner,verify_cuda_execution
from retina_sim.finite_input_support import OPTION


def main():
    output=ROOT/'reports/central_empty_sum_short_v1'
    if output.exists():raise ValueError('Retain prior checks and choose a fresh explicit output')
    output.mkdir();begin=time.perf_counter()
    profile='configurations/atlas_human_borrowed_disk3mm_functional_hc_empty_sum_v2.json'
    result={'schema':'central_empty_chemical_sum_short_check_v1','configuration':profile,
        'review_snapshot':'3333811d2dd737cacbb30bf746b19ad0ebdfb0e0',
        'post_review_work':True,'status':'RUNNING','background_preparation':'NOT_PERFORMED',
        'steady_state_claim':False,'physiological_validation':False,'default_adopted':False}
    def save():
        result['wall_seconds']=time.perf_counter()-begin
        (output/'check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    save()
    try:
        config=RunConfig.from_dict({'execution_profile':'atlas_isetbio_geometry_v1',
            'circuit_configuration_path':profile,'owner_x_mm':0.,'owner_y_mm':0.,
            'owner_width_mm':.02,'owner_height_mm':.02,'max_spatial_cells':2000000,
            'duration_seconds':.0001,'window_seconds':.0001,'sample_seconds':.0001,'warmup_seconds':0.})
        print('CENTRAL_EMPTY_SUM_ASSEMBLY_BEGIN',flush=True)
        compiled,_,audit=atlas_population_network(ROOT,duration=.0001,light_context=config.light_context(),
            owner_bounds_mm=config.owner_bounds(),halo_mm=.04,max_spatial_cells=config.max_spatial_cells,
            seed=config.seed,execution_profile=config.execution_profile,configuration_path=profile)
        view=compiled.global_disk_view
        previous=json.loads((ROOT/'reports/central_AII_actual_support_20261018.json').read_bytes())
        counts={name:len(rows) for name,rows in view['selected_rows'].items()}
        if compiled.global_disk_circuit.binding!=previous['global_graph'] or counts!=previous['selected_global_populations']:
            raise AssertionError('Explicit empty sum changed selected frozen graph/cells')
        spec=next(s for s in compiled.spec.components if s.component_id=='AII_AC')
        declaration=spec.options[OPTION]['excitation']
        link=next(x for x in compiled.links if x.specification.source_component=='RBC_AII_AMPA'
                  and x.specification.target_component=='AII_AC')
        if (link.matrix.nnz!=previous['selected_RBC_AII_edges'] or len(declaration['cell_ids'])!=previous['AII_without_any_RBC_chemical_edge']
            or np.count_nonzero(np.diff(link.matrix.indptr)==0)!=len(declaration['cell_ids'])):
            raise AssertionError('AII empty row/actual chemical edge count changed')
        result.update(status='ASSEMBLY_PASS_SHORT_EXECUTION_PENDING',selected_global_populations=counts,
            selected_global_nodes=view['global_node_count'],components=len(compiled.components),
            frozen_graph=compiled.global_disk_circuit.binding,
            frozen_geometry_proof=audit['atlas_runtime']['frozen_typed_geometry'],
            actual_AII_empty_chemical_rows=len(declaration['cell_ids']),actual_RBC_AII_edges=link.matrix.nnz,
            empty_AII_with_actual_gap=previous['empty_AII_with_actual_gap'],
            empty_AII_without_actual_gap=previous['empty_AII_without_actual_gap'],
            finite_empty_chemical_input=declaration,empty_owner_outputs=view['empty_output_samples'],
            selected_cells_edges_and_weights_unchanged=True,
            isolated_AII_scope='Retained original passive membrane and effective parameters; spontaneous drive not supplied')
        save();print('CENTRAL_EMPTY_SUM_ASSEMBLY_PASS',view['global_node_count'],len(declaration['cell_ids']),flush=True)
        solver=SolverConfig(**audit['resolved_execution_profile']['solver'])
        runner=make_runner(compiled,CUDA)
        result['execution_backend_check']=verify_cuda_execution(runner,solver,config.light_context())
        result['status']='CENTRAL_EXPLICIT_EMPTY_SUM_PROJECT_100US_EXECUTION_PASS'
        result['numerical_model_hash']=runner.model.model_hash
        result['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('retina_sim/finite_input_support.py','retina_sim/assembly.py','retina_sim/disk_circuit.py',
             'retina_sim/disk_runtime.py','retina_sim/atlas_primary_rod.py','retina_sim/engine.py',profile)}
        save();print('CENTRAL_EMPTY_SUM_SHORT_EXECUTION_PASS',result['execution_backend_check']['maximum_absolute_state_difference'],flush=True)
    except Exception as exc:
        result.update(status='BLOCKED',error=type(exc).__name__+': '+str(exc));save();traceback.print_exc();raise

if __name__=='__main__':main()
