"""Same canonical bytes for plain JSON bundle manifests, without nested generators."""
import hashlib,json
from pathlib import Path
import retina_sim.group_time_storage as storage

ORIGINAL=storage.canonical_hash
COUNTS={'plain_JSON':0,'original_non_JSON':0}


def plain_json(value):
    stack=[value];seen=set()
    while stack:
        item=stack.pop();kind=type(item)
        if kind in (type(None),bool,int,float,str):continue
        if kind not in (dict,list):return False
        identity=id(item)
        if identity in seen:return False
        seen.add(identity)
        if kind is dict:
            if any(type(key) is not str for key in item):return False
            stack.extend(item.values())
        else:stack.extend(item)
    return True


def canonical_hash(value):
    if plain_json(value):
        COUNTS['plain_JSON']+=1
        return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
    COUNTS['original_non_JSON']+=1
    return ORIGINAL(value)


def install():
    prior=getattr(storage,'_checkpoint_JSON_runtime',None)
    if prior is not None and prior is not canonical_hash:raise RuntimeError('Another checkpoint hash adapter already installed')
    storage.canonical_hash=canonical_hash;storage._checkpoint_JSON_runtime=canonical_hash


def evidence():
    return dict(schema='plain-json-canonical-bundle-io-v1',counts=dict(COUNTS),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='group_time_storage I/O hash binding only; contracts and original source files unchanged',
        JSON_options=dict(sort_keys=True,allow_nan=False,separators=[',',':'],ensure_ascii=True),
        all_array_file_SHA_and_layout_checks_retained=True,hash_schema_changed=False,
        no_identity_or_checksum_check_bypassed=True,native_corruption_root_cause_proven=False)
