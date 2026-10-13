"""One lossless conversion and actual production matrix/checkpoint equivalence check."""
from pathlib import Path
import json,sys,hashlib,gc,time

def main():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from retina_sim.atlas_graph_storage import convert_graph,FrozenGraphReader,materialize
    from retina_sim.contracts import canonical_hash,SolverConfig
    from retina_sim.background_checkpoint import BackgroundCheckpointStore,MemoryGuard,atomic_json
    from retina_platform.config import RunConfig
    from retina_sim.atlas_population_network import atlas_population_network
    from retina_sim.hybrid import HybridRunner
    configpath=root/'configurations/retina_platform_v1.json';config=json.loads(configpath.read_bytes());original_config=configpath.read_bytes();binding=config['atlas_target']['population_revision']
    directory=root/'data/atlas/graph_blocks_v1/population_revised_r035_v1';reportpath=root/'reports/graph_storage_integration_20261012.json'
    if reportpath.exists():raise FileExistsError('The single integration check has already completed')
    if directory.exists():
        m=json.loads((directory/'manifest.json').read_bytes());result={'path':str(directory),'source_json_sha256':m['source_json_sha256'],'logical_record_hash':m['logical_record_hash'],'storage_manifest_hash':m['storage_manifest_hash'],'files':len(m['files']),'compressed_payload_bytes':sum(x['bytes'] for x in m['files'].values())}
    else:result=convert_graph(root/binding['artifact'],directory,expected_source_sha256=binding['sha256'])
    reader=FrozenGraphReader(directory,expected_source_sha256=binding['sha256'],expected_manifest_hash=result['storage_manifest_hash'])
    raw=(root/binding['artifact']).read_bytes();original=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==binding['sha256']
    assert canonical_hash(original)==reader.semantic_hash==canonical_hash(materialize(reader)),'Logical graph rows/edges/weights/order/metadata differ'
    full_validation_io=dict(reader.io);del raw,original,reader;gc.collect()
    config['atlas_target']['population_revision_storage']=dict(result,path=str(directory.relative_to(root)),kind='frozen_retina_graph_blocks_v1')
    configpath.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    try:
        cfg=RunConfig.project_default(root);light=cfg.light_context();guard=MemoryGuard();guard.check();wall=time.perf_counter()
        compiled,_,audit=atlas_population_network(root,duration=.0012,light_context=light)
        runner=HybridRunner(compiled);reference=runner.initialize(SolverConfig(**audit['resolved_execution_profile']['solver']))
        store=BackgroundCheckpointStore(root,runner,reference,light,cfg.background_policy())
        state,value=store.load(root/cfg.background_checkpoint_path,require_converged=True)
        original_checkpoint=json.dumps(value['checkpoint'],sort_keys=True,allow_nan=False)
        assert runner.checkpoint(state)==original_checkpoint,'Stored complete state restored differently'
        assert store.binding==value['binding'],'Actual IDs/geometry/CSR matrices/gap/parameters/clock/solver differ from the existing prepared graph'
        assert len(compiled.components)==34 and store.coverage()['state_blocks']==77
        guard.check();report={'status':'PASS','storage':result,'original_graph_sha256_unchanged':binding['sha256'],'full_logical_graph_equal':True,
            'actual_compiled_matrix_and_parameter_binding_equal':True,'complete_prepared_checkpoint_bitwise_equal':True,
            'checkpoint_sha256':hashlib.sha256(runner.checkpoint(state).encode()).hexdigest(),'components':34,'state_blocks':77,
            'new_integration_or_preparation':False,'full_validation_io':full_validation_io,'production_storage':audit['artifact_storage'],
            'build_and_restore_wall_seconds':time.perf_counter()-wall,'process_memory':guard.last,
            'limits':'Storage read selection is lazy; actual selected circuit, matrices, parameters and ODE state remain fully resident. No full RF0-3 graph or streaming ODE execution.'}
        atomic_json(reportpath,report);print(json.dumps(report,ensure_ascii=False),flush=True)
    except BaseException:
        configpath.write_bytes(original_config);raise

if __name__=='__main__':main()
