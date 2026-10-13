"""One small rod-source parsing check; no model replay, fitting or background."""
from pathlib import Path
import sys,json,hashlib
import numpy as np

def main():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from retina_sim.chen2024_data import ChenPublishedData
    from retina_sim.contracts import ContractError
    out=root/'reports/chen2024_readonly_loader_check_20261012.json'
    if out.exists():raise FileExistsError('The one parsing validation is already complete')
    protect=['configurations/retina_platform_v1.json','configurations/effective_dynamics_v1.json','retina_sim/rod_pathway.py','retina_sim/chen2024.py']
    before={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in protect}
    data=ChenPublishedData(root);catalog=data.catalog('PrimateRodResponses');record=data.read_record('PrimateRodResponses',1,start_sample=0,stop_sample=256)
    original=data._group('PrimateRodResponses').flat[0].flat[0]
    assert np.array_equal(record.stimulus_Rstar_per_s,original.AdaptingStepStim.reshape(-1)[:256])
    assert np.array_equal(record.response_normalized,original.AdaptingStepResponse.reshape(-1)[:256])
    assert np.array_equal(record.relative_index_seconds,np.arange(256)*.001)
    assert all(not x.flags.writeable for x in [record.sample_index_zero_based,record.relative_index_seconds,record.stimulus_Rstar_per_s,record.response_normalized])
    rejected=False
    try:record.require_model_comparison_ready()
    except ContractError:rejected=True
    assert rejected and len(catalog['records'])==6 and all(r['samples']==123060 for r in catalog['records'])
    after={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in protect};assert before==after
    report={'status':'PASS','validation':'one rod record, first256 unaltered paired samples','metadata':record.metadata,
        'source_catalog':catalog,'selected_stimulus_range':[float(record.stimulus_Rstar_per_s.min()),float(record.stimulus_Rstar_per_s.max())],
        'selected_response_range':[float(record.response_normalized.min()),float(record.response_normalized.max())],
        'published_shape_and_bytes_reused_from_existing_inventory':True,'protected_files_unchanged':after,
        'model_comparison_rejected':True,'model_or_background_runs':0,'fit_or_resampling':False}
    out.write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':'PASS','output':str(out),'rod_records':6,'samples_per_record':123060,'dt_seconds':.001,
        'validated_samples':256,'blockers':record.metadata['relative_model_comparison_blockers']}),flush=True)

if __name__=='__main__':main()
