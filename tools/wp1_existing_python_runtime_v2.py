"""Isolated existing CPython build, exact existing project scientific packages."""
from pathlib import Path
import hashlib,json,os,runpy,sys

ROOT=Path(r'D:\macaque retinal simulator')
PACKAGES=ROOT/'.venv/Lib/site-packages'


def main():
    target=Path(sys.argv[1]).resolve();args=sys.argv[2:]
    if not target.is_relative_to((ROOT/'tools').resolve()) or not target.is_file():raise ValueError('Owned tool script required')
    if len(args)<2:raise ValueError('Native nonce/output gate required')
    output=Path(args[1]).resolve()
    if not output.is_relative_to((ROOT/'reports').resolve()) or not output.is_dir():raise ValueError('Owned existing capture directory required')
    if sys.version_info[:3]!=(3,11,15) or not sys.flags.isolated or not sys.flags.no_site:raise RuntimeError('Exact isolated existing3.11.15 runtime required')
    sys.path[:0]=[str(ROOT),str(PACKAGES)]
    if target.name=='wp1_latimer234_fit_worker_v1.py':
        # The original readout entry declares these before scientific imports.
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    os.environ['NPY_DISABLE_CPU_FEATURES']='X86_V3'
    import numpy,scipy,numba,llvmlite
    versions={name:module.__version__ for name,module in [('numpy',numpy),('scipy',scipy),('numba',numba),('llvmlite',llvmlite)]}
    if versions!=dict(numpy='2.4.6',scipy='1.17.1',numba='0.67.0',llvmlite='0.49.0'):raise RuntimeError('Scientific package versions changed')
    modules={name:str(Path(module.__file__).resolve()) for name,module in [('numpy',numpy),('scipy',scipy),('numba',numba),('llvmlite',llvmlite)]}
    if not all(Path(p).is_relative_to(PACKAGES.resolve()) for p in modules.values()):raise RuntimeError('Scientific libraries escaped original project environment')
    dll=Path(sys.base_prefix)/'python311.dll'
    identity=dict(schema='existing-cpython-numpy-baseline-v2',numpy_disabled_features=os.environ['NPY_DISABLE_CPU_FEATURES'],numpy_minimum_dispatch=numpy.lib.introspect.opt_func_info(func_name='minimum',signature='float64'),version=sys.version,executable=sys.executable,
        interpreter_dll_sha256=hashlib.sha256(dll.read_bytes()).hexdigest(),scientific_versions=versions,
        scientific_module_paths=modules,isolated=True,no_site=True,development_mode=sys.flags.dev_mode,
        thread_environment_before_scientific_import={name:os.environ.get(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        bootstrap_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        package_installation=False,OS_or_default_environment_changed=False,root_cause_proven=False)
    path=output/'PYTHON_RUNTIME_IDENTITY.json'
    with path.open('x',encoding='utf8') as stream:json.dump(identity,stream,indent=2)
    sys.argv=[str(target),*args];runpy.run_path(str(target),run_name='__main__')


if __name__=='__main__':main()
