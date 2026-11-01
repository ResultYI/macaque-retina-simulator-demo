"""Passive native attachment for the already frozen independent readout phases."""
import faulthandler,sys
from pathlib import Path
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.run_cell6_matched_baselines_v1 import attachment_gate

if __name__=='__main__':
    nonce,folder,cell=sys.argv[1:];out=Path(folder);mode=out.name.rsplit('_',1)[-1]
    with (out/'NATIVE_FAULTHANDLER.log').open('x',encoding='utf8') as stream:
        faulthandler.enable(file=stream,all_threads=True);attachment_gate(nonce,out)
        from tools.wp1_fit_input_hash_witness_v1 import install
        install(out)
        from tools.wp1_latimer234_fit_v1 import execute
        execute(int(cell),mode,runtime_identity_file=out/'PYTHON_RUNTIME_IDENTITY.json')
