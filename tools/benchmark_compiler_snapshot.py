"""Run retained compiler bytes for an old identity, without changing live files."""
from pathlib import Path
import hashlib,importlib.util,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
COMMIT='3333811d2dd737cacbb30bf746b19ad0ebdfb0e0'
SHA='4fed0da9eee54bb91b8e0d7823b9e1c9e34204d95bfb2a754e3b72589ebbd7c7'


def use_v7_compiler():
    name='retina_sim.assembly';path=ROOT/'reports/current_wip_review/frozen_v7_compiler/assembly.py'
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve()!=path:raise RuntimeError('Compiler snapshot must be selected before assembly imports')
        return {'path':str(path.relative_to(ROOT)),'sha256':SHA,'source_commit':COMMIT}
    content=subprocess.check_output(['git','show',COMMIT+':retina_sim/assembly.py'],cwd=ROOT)
    if hashlib.sha256(content).hexdigest()!=SHA:raise RuntimeError('Retained compiler source identity differs')
    path.parent.mkdir(exist_ok=True,parents=True)
    if path.exists():
        if path.read_bytes()!=content:raise RuntimeError('Existing retained compiler snapshot changed')
    else:path.write_bytes(content)
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    setattr(sys.modules['retina_sim'],'assembly',module)
    # The compiler itself hashes its actual __file__ bytes, as before. No
    # fabricated hashes, patched checkpoint identities, or live source edits.
    identity={'path':str(path.relative_to(ROOT)),'sha256':SHA,'source_commit':COMMIT,
        'execution':'Actual archived Python source loaded before all factories; original compiler file and current source left unchanged'}
    (path.parent/'provenance.json').write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
    return identity
