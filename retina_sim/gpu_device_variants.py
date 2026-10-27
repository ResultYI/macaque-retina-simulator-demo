"""Independent, opt-in single-RHS candidates; no v1 source is modified.

Graph replay changes dispatch only, SpMM changes only native A propagation, and
validation batching preserves every original checked region and domain. They
are deliberately separate variants, with private identities and no accepted
cross-device numerical error budget. No adaptive solver/event loop is captured.
"""
from copy import copy
from pathlib import Path
import hashlib
import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import json

from .contracts import ContractError, canonical_hash
from .gpu_device_joint import DeviceJointPartition, DeviceJointCudaRunner
from .gpu_native import CudaNative

VARIANTS = ('graph_inner', 'spmm_native', 'validation_batch', 'fused_native', 'graph_validation_batch')
SOURCES = {
    'CUDA_graphs': 'https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs',
    'CSR_SpMM': 'https://docs.pytorch.org/docs/2.14/generated/torch.sparse.mm.html',
    'NVRTC_Windows': 'https://docs.nvidia.com/cuda/nvrtc/index.html',
    'jiterator': 'https://docs.pytorch.org/docs/2.14/generated/torch.cuda.jiterator._create_jit_fn.html',
    'jiterator_NVRTC_options': 'https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/cuda/jit_utils.cpp#L1499-L1522',
}


class UnsupportedDeviceVariant(ContractError):
    """Requested device operation cannot run; never permits a CPU fallback."""


def compiler_support_facts():
    """Read-only support facts; no device initialization, compilation or kernel."""
    import torch
    root = Path(torch.__file__).parent
    packages = {}
    for name in ('torch', 'triton', 'triton-windows'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    vswhere = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'))/'Microsoft Visual Studio/Installer/vswhere.exe'
    vs = []
    if vswhere.exists():
        result = subprocess.run([str(vswhere), '-products', '*', '-requires',
            'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-format', 'json', '-utf8'],
            capture_output=True, text=True, encoding='utf8', timeout=15, check=True)
        vs = [dict(version=row['installationVersion'], product=row.get('productId')) for row in json.loads(result.stdout)]
    return dict(packages=packages, nvrtc_torch_libraries=[p.name for p in (root/'lib').glob('*nvrtc*.dll')],
        Visual_Studio_with_VC_tools=vs, compiler_executables_in_PATH={name: shutil.which(name) is not None for name in ('cl', 'nvcc', 'clang')},
        JIT_can_fuse_on_GPU=bool(torch._C._jit_can_fuse_on_gpu()),
        TensorExpr_fuser_enabled=bool(torch._C._jit_texpr_fuser_enabled()),
        jiterator_module_present=importlib.util.find_spec('torch.cuda.jiterator') is not None,
        NVRTC_options_review=dict(torch_source_version='v2.14.0',
            options=['--std=c++20', '--gpu-architecture=<actual device>', '-default-device', '-DNDEBUG or -lineinfo'],
            use_fast_math_requested=False, fmad_option_passed=False,
            caveat='NVRTC permits FMA contraction by default. Fused native source uses explicit double round intrinsics for multiply-add chains; pow/exp and other library/compiler rounding still require measured numerical acceptance.'),
        CUDA_initialized=bool(torch.cuda.is_initialized()), sources=SOURCES,
        conclusion='Native NVRTC/Jiterator and TensorExpr are candidates, not execution evidence. Missing Triton limits that backend only. No tools installed; no whole-program optimized JIT enabled.')


def make_release_fusion_probe(parameters):
    """Build an optional genuine NVRTC pointwise probe; first CUDA call compiles.

    This standalone probe is not silently installed into a production RHS. The
    caller compares it with the literal FP64 release formula and reports errors.
    """
    from torch.cuda.jiterator import _create_jit_fn
    code = '''template <typename T>
    T release_probe(T V, T feedback, T Vbar, T VI, T mu, T Vk, T Vn) {
        return (T(1) / (T(1) + exp(-((V-feedback-Vk)/Vn)))) / pow(Vbar/VI,mu);
    }'''
    return _create_jit_fn(code, VI=float(parameters['V_I']), mu=float(parameters['mu']),
                          Vk=float(parameters['V_k']), Vn=float(parameters['V_n']))


def validation_batch_source(source):
    """Replace scalar reductions by three batched checks, preserving snapshots."""
    finite = re.compile(r'    invalid = (?:invalid \| )?torch\.logical_not\(torch\.isfinite\((.+)\)\)\.any\(\)')
    domain = re.compile(r'    invalid = invalid \| \((.+) (<|<=) 0\.0\)\.any\(\)')
    counts = dict(finite=0, nonnegative=0, positive=0)
    result = []
    returned = False
    for line in source.splitlines():
        if line.startswith('def execute('):
            result.extend([line, '    finite_regions: List[Tensor] = []',
                '    nonnegative_regions: List[Tensor] = []', '    positive_regions: List[Tensor] = []'])
        elif match := finite.fullmatch(line):
            counts['finite'] += 1
            result.append('    finite_regions.append('+match[1]+')')
        elif match := domain.fullmatch(line):
            name = 'nonnegative' if match[2] == '<' else 'positive'
            counts[name] += 1
            result.append('    '+name+'_regions.append('+match[1]+')')
        elif line == '    return result, invalid':
            if not counts['finite']:
                raise ContractError('No original finite regions found for batching')
            result.append('    invalid = torch.logical_not(torch.isfinite(torch.cat(finite_regions, 0))).any()')
            for name, comparison in (('nonnegative', '<'), ('positive', '<=')):
                if counts[name]:
                    result.append('    invalid = invalid | (torch.cat('+name+'_regions, 0) '+comparison+' 0.0).any()')
            result.append(line)
            returned = True
        elif 'invalid' in line:
            raise ContractError('Unreviewed validation expression; cannot batch: '+line)
        else:
            result.append(line)
    if not returned:
        raise ContractError('Original device function return contract differs')
    return '\n'.join(result)+'\n', counts


def _validate_inner(plan, y, external):
    torch = plan.torch
    if y.dtype != torch.float64 or y.device.type != 'cuda' or y.shape != (plan.nstate,):
        raise ContractError('Variant inner state must retain complete CUDA FP64 layout')
    if len(external) != len(plan.external_sizes):
        raise ContractError('Variant inner external count differs')
    if any(value.dtype != torch.float64 or value.device != y.device or value.shape != (size,)
           for value, size in zip(external, plan.external_sizes)):
        raise ContractError('Variant inner external layout differs')


class BatchedValidationInner:
    def __init__(self, reference):
        self.__dict__.update(reference.__dict__)
        self.source, self.validation_regions = validation_batch_source(reference.source)
        self.source_sha256 = hashlib.sha256(self.source.encode()).hexdigest()
        # CompilationUnit resolves the generated ``torch`` name in this frame.
        torch = self.torch
        self.compilation = torch.jit.CompilationUnit(self.source)
        self.statistics = dict(calls=0, dynamic_cross_stage_cache=False)

    def __call__(self, y, external):
        _validate_inner(self, y, external)
        with self.torch.jit.optimized_execution(False):
            result, invalid = self.compilation.execute(y, external, self.constants, self.matrices, self.indices)
        if invalid.item():
            raise ContractError('Batched original finite/nonnegative/positive domain violated; no clipping')
        self.statistics['calls'] += 1
        return result


class CudaGraphInner:
    def __init__(self, reference):
        self.reference = reference
        self.torch = reference.torch
        self.nstate, self.external_sizes = reference.nstate, reference.external_sizes
        self.source_sha256 = reference.source_sha256
        self.graph = None
        self.unsupported = None
        self.statistics = dict(calls=0, captures=0, replays=0, D2D_input_copy_bytes=0,
                               dynamic_cross_stage_cache=False)

    def _execute(self):
        with self.torch.jit.optimized_execution(False):
            return self.reference.compilation.execute(self.state_buffer, self.external_buffers,
                self.reference.constants, self.reference.matrices, self.reference.indices)

    def _capture(self, y, external):
        torch = self.torch
        try:
            self.state_buffer = y.clone()
            self.external_buffers = [value.clone() for value in external]
            stream = torch.cuda.Stream(device=y.device)
            stream.wait_stream(torch.cuda.current_stream(y.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._execute()
            torch.cuda.current_stream(y.device).wait_stream(stream)
            # No native checks, .item(), event loop or adaptive solver in capture.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.result_buffer, self.invalid_buffer = self._execute()
            torch.cuda.current_stream(y.device).wait_stream(stream)
            self.statistics['captures'] += 1
        except (RuntimeError, NotImplementedError) as exc:
            self.graph = None
            self.unsupported = dict(status='UNSUPPORTED', phase='pure_inner_CUDA_graph_capture',
                error_type=type(exc).__name__, reason=str(exc), CPU_fallback=False)
            raise UnsupportedDeviceVariant(str(self.unsupported)) from exc

    def __call__(self, y, external):
        _validate_inner(self, y, external)
        if self.unsupported is not None:
            raise UnsupportedDeviceVariant(str(self.unsupported))
        if self.graph is None:
            self._capture(y, external)
        if y.device != self.state_buffer.device:
            raise ContractError('Captured device identity differs')
        self.state_buffer.copy_(y)
        for target, value in zip(self.external_buffers, external):
            target.copy_(value)
        try:
            self.graph.replay()
        except (RuntimeError, NotImplementedError) as exc:
            self.unsupported = dict(status='UNSUPPORTED', phase='pure_inner_CUDA_graph_replay',
                error_type=type(exc).__name__, reason=str(exc), CPU_fallback=False)
            raise UnsupportedDeviceVariant(str(self.unsupported)) from exc
        # Exact original validity predicate; CPU synchronization stays outside capture.
        if self.invalid_buffer.item():
            raise ContractError('Captured original finite/nonnegative/positive domain violated; no clipping')
        self.statistics['calls'] += 1
        self.statistics['replays'] += 1
        self.statistics['D2D_input_copy_bytes'] += y.numel()*y.element_size()+sum(v.numel()*v.element_size() for v in external)
        # A later replay must never overwrite a derivative retained by the solver.
        return self.result_buffer.clone()


class SpMMNative(CudaNative):
    def __init__(self, reference):
        if type(reference) is not CudaNative:
            raise ContractError('SpMM starts from the original reviewed native operators')
        self.__dict__.update(reference.__dict__)

    def __call__(self, y, light):
        torch, p = self.torch, self.p
        if y.dtype != torch.float64 or light.dtype != torch.float64 or y.shape != (sum(self.sizes),) or light.shape != (self.nc,):
            raise ContractError('CUDA native state/input layout differs')
        R, E, X, C, V, gi, Vbar, z1, z2, h = y.split(self.sizes)
        invalid = (~torch.isfinite(y)).any() | (~torch.isfinite(light)).any() | (light < 0).any()
        invalid = invalid | (gi <= 0).any() | (Vbar <= 0).any() | (V < 0).any() | (X < 0).any() | (C < 0).any()
        if invalid.item():
            raise ContractError('Native finite/positive/nonnegative domain violated; no CUDA clipping')
        current = X**p['n_X']; adaptation = (Vbar/p['V_I'])**p['mu']
        q = 1./(1.+torch.exp(-((V-torch.mv(self.B, h)-p['V_k'])/p['V_n'])))
        release = q/adaptation
        # Only the two independent A products change dispatch/reduction algorithm.
        paired = torch.sparse.mm(self.A, torch.stack((Vbar, z1), dim=1))
        pooled = (paired[:, 0]/p['V_I'])**p['mu']
        alpha = 1./(1.+(p['a_C']*C)**p['n_C']); beta = p['c_beta']+p['k_beta']*E
        rows = ((light-R)/p['tau_R'], (R-E)/p['tau_E'], alpha-beta*X,
            (current-C)/p['tau_C'], (current/gi-V)/p['tau_m'],
            (p['a_is']*V**p['gamma']-gi)/p['tau_is'], (V-Vbar)/p['tau_a'],
            (p['g_t']*release-z1)/p['tau_1'],
            (paired[:, 1]-z2)/(p['tau_2']*pooled), (z2-h)/(p['tau_h']*pooled))
        derivative = 1000.*torch.cat(rows)
        if not (torch.isfinite(derivative).all() & torch.isfinite(release).all()).item():
            raise ContractError('Nonfinite native CUDA derivative/release')
        return derivative, release


class FusedNative(CudaNative):
    """True pointwise fusion via existing PyTorch/NVRTC; three original SpMVs."""
    def __init__(self, reference):
        if type(reference) is not CudaNative:
            raise ContractError('Fusion requires the original reviewed native owner')
        self.__dict__.update(reference.__dict__)
        from torch.cuda.jiterator import _create_multi_output_jit_fn
        p = {key: 'T('+repr(float(value))+')' for key, value in self.p.items()}
        def build(name, inputs, outputs, body):
            signature = ', '.join(['T '+v for v in inputs]+['T& '+v for v in outputs])
            source = 'template <typename T> void '+name+'('+signature+') { '+body+' }'
            return _create_multi_output_jit_fn(source, len(outputs)), source
        self.signals, s1 = build('native_signals', ['X','V','Vbar','feedback'], ['current','adaptation','release'],
            f"current=pow(X,{p['n_X']}); adaptation=pow(Vbar/{p['V_I']},{p['mu']}); "
            f"release=(T(1)/(T(1)+exp(-((V-feedback-{p['V_k']})/{p['V_n']}))))/adaptation;")
        self.os, s2 = build('native_os', ['light','R','E','X','C','current'], ['dR','dE','dX','dC'],
            f"T alpha=T(1)/(T(1)+pow({p['a_C']}*C,{p['n_C']})); T beta=__dadd_rn({p['c_beta']},__dmul_rn({p['k_beta']},E)); "
            f"dR=(light-R)/{p['tau_R']}; dE=(R-E)/{p['tau_E']}; "
            f"dX=__dsub_rn(alpha,__dmul_rn(beta,X)); dC=(current-C)/{p['tau_C']};")
        self.is_, s3 = build('native_is', ['current','gi','V','Vbar','z1','release'], ['dV','dgi','dVbar','dz1'],
            f"dV=(current/gi-V)/{p['tau_m']}; dgi=__dsub_rn(__dmul_rn({p['a_is']},pow(V,{p['gamma']})),gi)/{p['tau_is']}; "
            f"dVbar=(V-Vbar)/{p['tau_a']}; dz1=__dsub_rn(__dmul_rn({p['g_t']},release),z1)/{p['tau_1']};")
        self.hc, s4 = build('native_hc', ['AVbar','Az1','z2','h'], ['dz2','dh'],
            f"T pooled=pow(AVbar/{p['V_I']},{p['mu']}); dz2=(Az1-z2)/({p['tau_2']}*pooled); "
            f"dh=(z2-h)/({p['tau_h']}*pooled);")
        self.fusion_sources = (s1, s2, s3, s4)
        self.fusion_source_sha256 = hashlib.sha256('\n'.join(self.fusion_sources).encode()).hexdigest()

    def __call__(self, y, light):
        torch = self.torch
        if y.dtype != torch.float64 or light.dtype != torch.float64 or y.shape != (sum(self.sizes),) or light.shape != (self.nc,):
            raise ContractError('CUDA native state/input layout differs')
        R, E, X, C, V, gi, Vbar, z1, z2, h = y.split(self.sizes)
        invalid = (~torch.isfinite(y)).any() | (~torch.isfinite(light)).any() | (light < 0).any()
        invalid = invalid | (gi <= 0).any() | (Vbar <= 0).any() | (V < 0).any() | (X < 0).any() | (C < 0).any()
        if invalid.item():
            raise ContractError('Native finite/positive/nonnegative domain violated; no CUDA clipping')
        try:
            current, adaptation, release = self.signals(X, V, Vbar, torch.mv(self.B, h))
            os_rows = self.os(light, R, E, X, C, current)
            is_rows = self.is_(current, gi, V, Vbar, z1, release)
            hc_rows = self.hc(torch.mv(self.A, Vbar), torch.mv(self.A, z1), z2, h)
        except (RuntimeError, NotImplementedError, AssertionError) as exc:
            raise UnsupportedDeviceVariant('NVRTC pointwise fusion unsupported; no fallback: '+str(exc)) from exc
        # Retain the original ms->s conversion after complete row concatenation.
        derivative = 1000.*torch.cat((*os_rows, *is_rows, *hc_rows))
        if not (torch.isfinite(derivative).all() & torch.isfinite(release).all()).item():
            raise ContractError('Nonfinite native CUDA derivative/release')
        return derivative, release


class VariantJointPartition(DeviceJointPartition):
    def __init__(self, reference, variant):
        if type(reference) is not DeviceJointPartition or variant not in VARIANTS:
            raise ContractError('An explicit independent device-v1 variant is required')
        self.__dict__.update(reference.__dict__)
        self.statistics = {name: 0 for name in reference.statistics}
        self.variant = variant
        if variant == 'graph_inner':
            self.inner = CudaGraphInner(reference.inner)
        elif variant == 'graph_validation_batch':
            self.inner = CudaGraphInner(BatchedValidationInner(reference.inner))
        elif variant == 'validation_batch':
            self.inner = BatchedValidationInner(reference.inner)
        elif variant == 'spmm_native':
            self.native = SpMMNative(reference.native)
        elif variant == 'fused_native':
            self.native = FusedNative(reference.native)
        self.variant_identity = dict(variant=variant, implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            base_generated_source_sha256=reference.inner.source_sha256,
            generated_source_sha256=self.inner.source_sha256, dtype='float64',
            native_fusion_source_sha256=getattr(self.native, 'fusion_source_sha256', None),
            native_fusion_rounding='Explicit double round intrinsics for multiply-add chains; NVRTC library pow/exp differences pending numerical acceptance; no fast_math requested; original 1000.*cat(rows) retained' if variant == 'fused_native' else None,
            original_native_A_B_reused=self.native.A is reference.native.A and self.native.B is reference.native.B,
            captured_scope='pure inner RHS only' if variant in ('graph_inner', 'graph_validation_batch') else None,
            composition=['validation_batch', 'graph_inner'] if variant == 'graph_validation_batch' else [variant],
            numerical_budget_accepted=False, default_changed=False, CPU_fallback=False,
            validation_regions=getattr(self.inner, 'validation_regions',
                getattr(getattr(self.inner, 'reference', None), 'validation_regions', None)))


class DeviceVariantCudaRunner(DeviceJointCudaRunner):
    @classmethod
    def from_runner(cls, reference, variant):
        if type(reference) is not DeviceJointCudaRunner:
            raise ContractError('Variant runner requires the unchanged device-v1 runner')
        self = cls.__new__(cls)
        self.__dict__.update(reference.__dict__)
        self.model = copy(reference.model)
        self.integration_metrics = dict(reference.integration_metrics)
        self.partition = VariantJointPartition(reference.partition, variant)
        identity = dict(reference.model.compute_metadata['integration_backend'],
            backend='cuda_device_joint_'+variant+'_experimental_v1', variant=self.partition.variant_identity)
        self.model.model_hash = canonical_hash(dict(base_model_hash=reference.model.model_hash, variant=identity))
        self.model.compute_metadata = dict(reference.model.compute_metadata, integration_backend=identity)
        return self
