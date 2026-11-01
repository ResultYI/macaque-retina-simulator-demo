"""Record the existing exact full-input check without altering its result or gates."""
from pathlib import Path
import hashlib,json


def install(directory):
    import numpy as np
    from retina_sim import contracts
    root=Path(r'D:\macaque retinal simulator\reports\WP1_LATIMER_CELL2_4_V1')
    source=json.loads((root/'preparation/SOURCE_IDENTITY.json').read_text(encoding='utf-8-sig'))
    expected={c['full_forward_binding']['input_payload_sha256']:c['group'] for c in source['cases'][1:]}
    original=contracts.canonical_hash;counter=0
    def witnessed(value):
        nonlocal counter
        result=original(value)
        if isinstance(value,dict) and len(value)==1065 and set(value)==set(range(1065)) and all(isinstance(v,np.ndarray) for v in value.values()):
            counter+=1
            record=dict(sequence=counter,actual_sha256=result,known_group=expected.get(result),exact_known_input=result in expected,
                gate_or_hash_result_changed=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            if result not in expected:
                np.savez_compressed(directory/f'UNMATCHED_INPUT_{counter}.npz',**{f'frame_{k}':v for k,v in value.items()})
                record['unmatched_actual_program_preserved']=True
            with (directory/'FULL_INPUT_HASH_WITNESS.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(record)+'\n')
        return result
    contracts.canonical_hash=witnessed
