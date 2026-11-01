"""Identity regression on the exact failed manifest, with no model integration."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools import checkpoint_json_hash_runtime_v1 as candidate
from retina_sim.group_time_storage import load_bundle


def main():
    began=time.perf_counter();paths=[
        ROOT/'reports/WP1_LATIMER_CELL2_4_V1/cache/g4/CHECKPOINT_0065000',
        ROOT/'reports/WP1_LATIMER_CELL2_4_V1/cache/g1/CHECKPOINT_0071000',
        ROOT/'reports/CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1/halo_model_v2/seed']
    pins=[]
    for directory in paths:
        manifest=json.loads((directory/'manifest.json').read_text());expected=manifest.pop('manifest_hash')
        actual=candidate.canonical_hash(manifest)
        if actual!=expected:raise RuntimeError('Original saved manifest SHA differs')
        pins.append(dict(path=str(directory),original_saved_manifest_hash=expected,fast_path_hash=actual,bytes=(directory/'manifest.json').stat().st_size))
    values=[{'unicode':'视网膜','values':[None,True,-0.,1.25]}, {True:1,'True':2},(1,2),np.arange(12,dtype=np.float64).reshape(3,4)]
    for value in values:
        if candidate.canonical_hash(value)!=candidate.ORIGINAL(value):raise RuntimeError('Canonical contract mismatch')
    candidate.install()
    restored=load_bundle(paths[0],expected_kind='validation_evidence')
    if restored['payload']['controller']['time_seconds']!=6.5:raise RuntimeError('Restore prefix time differs')
    result=dict(passed=True,manifests=pins,non_plain_JSON_fallback_unchanged=True,
        exact_failed_checkpoint_read_with_all_array_SHA_checks=True,source_seconds=6.5,
        no_model_constructed_or_advanced=True,previous_normal_process_checkpoint=True,
        candidate=candidate.evidence(),wall_seconds=time.perf_counter()-began,root_cause_proven=False)
    out=ROOT/'reports/WP1_LATIMER_CELL2_4_V1/CHECKPOINT_HASH_REPAIR_REGRESSION.json'
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
