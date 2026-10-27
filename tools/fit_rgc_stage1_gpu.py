"""Explicit CUDA stage1 entry; separate execution identities, no CPU fallback."""
from pathlib import Path
from copy import deepcopy
from contextlib import contextmanager
import argparse,hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.rgc_stage1_gpu_cache import capture,read_cache,write
from tools import fit_rgc_stage1 as readout

OUT=ROOT/'reports/rgc_stage1_gpu_v1'
CONFIG=ROOT/'configurations/calibration/rgc_stage1_gpu_v1.json'
CODE=['retina_sim/ac_vs_compiled.py','retina_sim/ac_vs_gpu.py','retina_sim/rgc_stage1_gpu_cache.py',
      'tools/fit_rgc_stage1_gpu.py','tools/validate_rgc_stage1_gpu.py','retina_sim/group_gpu_rhs.py',
      'retina_sim/group_gpu_runtime.py','retina_sim/group_gpu_dop853.py','retina_sim/resident_dop853.py',
      'retina_sim/resident_cusparse.py',str(CONFIG.relative_to(ROOT)).replace('\\','/')]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare():
    config=read(CONFIG)
    if config['execution_backend']!='ac_vs_cuda_dop853_fp64_v1' or config['CPU_fallback'] or config['CPU_cache_import']:
        raise ContractError('Explicit GPU configuration without fallback/import required')
    baseline=readout.verify();gate=read(ROOT/config['required_validation'])
    if not gate['passed']:raise ContractError('Original-budget new-model CPU/GPU and restore checks must pass')
    OUT.mkdir(exist_ok=False);(OUT/'consistency').mkdir()
    spec=deepcopy(baseline);spec['execution_backend']=config['execution_backend'];spec['numerical']['mode']='gpu_dop853'
    spec['execution_configuration']=config;spec['source_CPU_contract_sha256']=sha(ROOT/config['source_CPU_contract'])
    spec['CPU_seal_receipt_sha256']=sha(ROOT/'reports/rgc_stage1_gpu_adaptation_v1/CPU_SEAL_RECEIPT.json')
    spec['GPU_validation_sha256']=sha(ROOT/config['required_validation'])
    spec['CPU_cached_trajectories_imported']=False;spec['cross_backend_checkpoint_restore_allowed']=False
    spec['code_pins'].update({n:sha(ROOT/n) for n in CODE})
    source=ROOT/'reports/rgc_stage1_complete_v1/actual_observations.npz';shutil.copy2(source,OUT/'actual_observations.npz')
    if sha(OUT/'actual_observations.npz')!=spec['counts_sha256']:raise ContractError('Actual train/validation observations changed')
    write(OUT/'CONTRACT.json',spec)
    write(OUT/'consistency/RESULT.json',dict(passed=True,source=config['required_validation'],sha256=spec['GPU_validation_sha256'],
        same_equations_parameters_inputs=True,CPU_cache_import=False,GPU_restore_gate=gate['GPU_new_process_final_causal_exact']))
    write(OUT/'CURRENT_STATUS.json',dict(status='GPU_PREPARED_NOT_RUNNING',execution_backend=spec['execution_backend'],GPU_only=True,optimizer_iterations=0))
    print(json.dumps(dict(status='GPU_PREPARED',contract_identity=canonical_hash(spec),cache_groups=[c['case_id'] for c in spec['cases']],CPU_cache_import=False)))

def verify():
    spec=read(OUT/'CONTRACT.json')
    if spec.get('execution_backend')!='ac_vs_cuda_dop853_fp64_v1' or spec['numerical']['mode']!='gpu_dop853':raise ContractError('GPU contract required')
    for n,digest in spec['code_pins'].items():
        if sha(ROOT/n)!=digest:raise ContractError('Frozen GPU source changed: '+n)
    if sha(OUT/'actual_observations.npz')!=spec['counts_sha256']:raise ContractError('Actual observed spike identity changed')
    if sha(ROOT/spec['execution_configuration']['required_validation'])!=spec['GPU_validation_sha256']:raise ContractError('GPU validation evidence changed')
    return spec

@contextmanager
def output_context():
    # Reuse the unchanged six-candidate objective/selection/validation implementation.
    # Only its application output directory and identity-checking cache reader change.
    oldout,oldreader=readout.OUT,readout.read_cache
    readout.OUT,readout.read_cache=OUT,read_cache
    try:yield
    finally:readout.OUT,readout.read_cache=oldout,oldreader

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','dry-run','capture','fit','evaluate','run']);p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.mode=='prepare':prepare();return
    spec=verify()
    if args.mode=='dry-run':print(json.dumps(dict(execution_backend=spec['execution_backend'],contract_identity=canonical_hash(spec),groups=len(spec['cases']),CPU_fallback=False,model_started=False)));return
    if args.mode in ('capture','run'):
        for case in spec['cases']:
            print(json.dumps(dict(stage='GPU_FIXED_ENDOGENOUS_CAPTURE',case_id=case['case_id'],split=case['split'])),flush=True)
            capture(ROOT,OUT,spec,case,resume=args.resume)
    with output_context():
        if args.mode in ('fit','run'):print(json.dumps(readout.optimize(spec)),flush=True)
        if args.mode in ('evaluate','run'):print(json.dumps(readout.evaluate(spec)),flush=True)

if __name__=='__main__':main()
