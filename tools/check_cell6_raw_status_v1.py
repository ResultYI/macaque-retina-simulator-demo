"""Read-only status; never resumes or launches computation."""
from pathlib import Path
import hashlib
import json

ROOT=Path(r'D:\macaque retinal simulator')
BASE=ROOT/'reports/CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1'


def main():
    source=json.loads((BASE/'SOURCE_IDENTITY.json').read_text(encoding='utf8'))
    cache=json.loads((BASE/'NEW_CELL6_CACHE_MANIFEST.json').read_text(encoding='utf8'))
    model=json.loads((BASE/'halo_model_v2/RESULT.json').read_text(encoding='utf8'))
    fit=json.loads((BASE/'READOUT_FIT.json').read_text(encoding='utf8'))
    prepared=hashlib.sha256((BASE/'SOURCE_PREPARATION.npz').read_bytes()).hexdigest()==source['preparation_npz_sha256']
    print(json.dumps(dict(status=cache['status'],prepared_data_hash_valid=prepared,prepared_groups=len(source['cases']),
        target=model['target'],target_background_constructed=model['status'],completed_neural_groups=cache['completed_groups'],
        readout_status=fit['status'],native_failure='NATIVE_FAILURE_EVIDENCE.json',automatic_restart=False,
        report=str(BASE/'FINAL_REPORT.md')),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
