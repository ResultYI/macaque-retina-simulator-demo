"""Numerical compatibility probe for the separately identified laptop backend."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from retina_sim.group_gpu_runtime import CudaModule
    from retina_sim.resident_cusparse import CusparseAPI, NativeCSRPlan
    out = ROOT/'reports/million_gpu_resident_20261025_v1/compatibility.json'
    if out.exists():
        raise RuntimeError('Compatibility evidence already exists; do not overwrite')
    result = dict(status='IN_PROGRESS', strict_source_environment='BLOCKED_TORCH_VERSION_UNCHANGED',
                  python=sys.executable, neural_advance_started=False, drivers_or_packages_changed=False)
    module = plan = None
    try:
        torch.cuda.init()
        source = '''extern "C" __global__ void probe(const double* x, double* y) {
            int i=blockIdx.x*blockDim.x+threadIdx.x;
            if(i<8) y[i]=exp(log(x[i])) + pow(x[i], 2.5) / (1.0+x[i]);
        }'''
        module = CudaModule(torch, source, ['probe'])
        x = torch.arange(1, 9, dtype=torch.float64, device='cuda')/4
        y = torch.empty_like(x)
        module.launch('probe', 8, [x,y])
        torch.cuda.synchronize()
        cpu = x.cpu().numpy()
        expected = np.exp(np.log(cpu))+cpu**2.5/(1+cpu)
        error = np.max(np.abs(y.cpu().numpy()-expected))
        np.testing.assert_allclose(y.cpu().numpy(), expected, rtol=8192*np.finfo(float).eps, atol=0)
        crow = torch.tensor([0,2,3,5], dtype=torch.int64, device='cuda')
        col = torch.tensor([0,2,1,0,2], dtype=torch.int64, device='cuda')
        values = torch.tensor([2.,-1.,3.,.5,4.], dtype=torch.float64, device='cuda')
        matrix = torch.sparse_csr_tensor(crow,col,values,size=(3,3))
        api = CusparseAPI(torch)
        plan = NativeCSRPlan(matrix,api,torch)
        vector = torch.tensor([1.,2.,3.],dtype=torch.float64,device='cuda')
        outputs = [plan.mv(vector).cpu().numpy() for _ in range(3)]
        for actual in outputs:
            np.testing.assert_array_equal(actual,[-1.,6.,12.5])
        result.update(status='PASS_SELECTED_CUDA126_EXECUTION_ONLY', runtime=module.identity,
            nvrtc_log=module.compile_log, transcendental_max_abs_error=float(error),
            cusparse=dict(dll=str(api.path),dll_sha256=api.sha256,version=plan.version,
                algorithm='CSR_ALG2',index_dtype='int64',value_dtype='float64',
                preprocess=plan.preprocessed,buffer_bytes=plan.buffer_bytes,exact_repeats=3),
            nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free','--format=csv,noheader'],text=True).strip(),
            cuda_free_total_bytes=list(torch.cuda.mem_get_info()),checked_unix=time.time())
    except Exception as exc:
        result.update(status='FAIL_SELECTED_BACKEND_COMPATIBILITY',error=type(exc).__name__,message=str(exc))
        raise
    finally:
        if plan is not None: plan.close()
        if module is not None: module.close()
        out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
