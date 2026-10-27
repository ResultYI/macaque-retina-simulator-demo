"""Small repeat-slice ownership diagnostic using the real _integrate method.

The test-only DOP853 subclass records weakrefs; it changes no numerical method.
Automatic GC is disabled only inside each isolated probe to expose ownership.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import weakref
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'reports/solver_lifetime_memory_20261021_v1'


def load_baseline():
    path=BASE/'baseline_hybrid.py';identity=json.loads((BASE/'baseline_identity.json').read_bytes())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=identity['sha256']:raise ValueError('Baseline source changed')
    spec=importlib.util.spec_from_file_location('retina_sim._solver_lifetime_baseline',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def resources():
    from ctypes import wintypes
    from tools.supervise_regional_movie_cache import _resources
    kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wintypes.HANDLE
    return _resources(int(kernel.GetCurrentProcess()))


def worker(mode,branch,n,slices):
    import numpy as np
    from scipy.integrate import DOP853
    from scipy.integrate._ivp import ivp
    from retina_sim.hybrid import HybridRunner
    from retina_sim.network import InputBundle
    from retina_sim.contracts import SolverConfig
    recorded=[];dense_refs=[]
    class TrackedDOP853(DOP853):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            row=dict(solver=weakref.ref(self),K=weakref.ref(self.K_extended),accepted=[])
            row['fun_cycle_captures_self']=any(cell.cell_contents is self for cell in self.fun.__closure__ or ())
            recorded.append(row);self.probe_row=row
        def step(self):
            result=super().step();self.probe_row['accepted'].append(float(self.t));return result
        def _dense_output_impl(self):
            result=super()._dense_output_impl();dense_refs.append(weakref.ref(result.F));return result
    method=load_baseline().HybridRunner._integrate if mode=='baseline' else HybridRunner._integrate
    rates=np.linspace(1.,2.,n);initial=np.linspace(.1,1.,n)
    model=SimpleNamespace(parameters=object(),graph=object())
    model.rhs_flat=lambda t,y,bundle,p,g:-rates*y
    metrics=dict(solver_calls=0,nfev=0,njev=0,nlu=0,integration_wall_seconds=0.,minimum_interval_seconds=None,maximum_nfev_per_interval=0)
    runner=SimpleNamespace(model=model,integration_metrics=metrics,_flat=lambda state:state.y.copy())
    def update(state,y,t):state.y=y;state.time_seconds=float(t)
    runner._replace=update;state=SimpleNamespace(y=initial.copy(),time_seconds=0.)
    stimulus=SimpleNamespace(interpolation='linear',at=lambda t:InputBundle({}))
    config=SolverConfig(method='DOP853',rtol=1e-6,atol=1e-9,max_step=.0001,input_interpolation='linear')
    gc.collect();enabled=gc.isenabled();gc.disable()
    before=resources();started=time.perf_counter();cpu=time.process_time();live=[];hashes=[];row_counts=[]
    try:
        with patch.dict(ivp.METHODS,{'DOP853':TrackedDOP853}):
            for index in range(slices):
                stop=(index+1)*.0001
                samples=np.array([stop]) if branch=='endpoint' else np.array([index*.0001+.00005,stop])
                rows=method(runner,state,stimulus,stop,config,samples)
                hashes.append(hashlib.sha256(state.y.tobytes()).hexdigest());row_counts.append(len(rows))
                live.append(sum(row['solver']() is not None for row in recorded))
                del rows
        owner=state.y
        while isinstance(owner.base,np.ndarray):owner=owner.base
        retained=resources()
        report=dict(schema='small-real-integrate-lifetime-probe-1',mode=mode,branch=branch,n=n,slices=slices,
            GC_auto_disabled_for_repro_only=True,method='DOP853',rtol=config.rtol,atol=config.atol,
            max_step=config.max_step,t_eval='None' if branch=='endpoint' else 'Original two requested times',dense_output=False,
            state_hash_per_slice=hashes,accepted_times=[row['accepted'] for row in recorded],returned_rows_per_slice=row_counts,
            solver_fun_self_cycle_verified=all(row['fun_cycle_captures_self'] for row in recorded),
            live_solvers_per_slice=live,live_solvers_before_explicit_final_collection=live[-1],
            live_K_arrays_before_final_collection=sum(row['K']() is not None for row in recorded),
            bytes_per_K=16*n*8,retained_K_bytes=sum(row['K']() is not None for row in recorded)*16*n*8,
            live_dense_F_arrays=sum(ref() is not None for ref in dense_refs),dense_F_arrays_created=len(dense_refs),
            final_vector_bytes=state.y.nbytes,final_owner_bytes=owner.nbytes,final_vector_owns_data=bool(state.y.flags.owndata),
            resources_before=before,resources_before_final_collection=retained,
            metrics=metrics,wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,
            maximum_closed_form_error=float(np.max(abs(state.y-initial*np.exp(-rates*slices*.0001)))))
        tick=time.perf_counter();collected=gc.collect()
        report.update(final_collection_seconds=time.perf_counter()-tick,final_collection_objects=collected,
            live_solvers_after_final_collection=sum(row['solver']() is not None for row in recorded),
            live_K_arrays_after_final_collection=sum(row['K']() is not None for row in recorded),
            resources_after_final_collection=resources())
    finally:
        if enabled:gc.enable()
    path=BASE/(mode+'_'+branch+'.json');path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({key:report[key] for key in ('mode','branch','live_solvers_per_slice','retained_K_bytes',
        'live_solvers_after_final_collection','final_vector_bytes','final_owner_bytes','live_dense_F_arrays',
        'wall_seconds','CPU_seconds')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('baseline','current'),required=True)
    parser.add_argument('--branch',choices=('endpoint','sampled'),required=True)
    parser.add_argument('--n',type=int,default=65536);parser.add_argument('--slices',type=int,default=12)
    args=parser.parse_args()
    if not 1<=args.n<=65536 or not 1<=args.slices<=12:raise ValueError('Small bounded probes only')
    worker(args.mode,args.branch,args.n,args.slices)


if __name__=='__main__':main()
