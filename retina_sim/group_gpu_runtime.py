"""Explicit CUDA 12.6/NVRTC execution, without installations or driver changes.

Torch owns buffers and the CUDA context. NVRTC produces device-specific cubin;
no PTX upgrade requirement, fast math, Graph capture or legacy backend fallback.
"""
import ctypes as C
import hashlib
import os
from pathlib import Path

from .contracts import ContractError


class CudaModule:
    def __init__(self, torch, source, names):
        self.torch = torch
        self.device = torch.cuda.current_device()
        self.stream = int(torch.cuda.current_stream(self.device).cuda_stream)
        self.module = C.c_void_p()
        self.closed = False
        directory = Path(torch.__file__).parent / 'lib'
        paths = [p for p in directory.glob('nvrtc64_*.dll') if '.alt.' not in p.name]
        if len(paths) != 1 or os.name != 'nt':
            raise ContractError('Exactly one installed Windows Torch NVRTC DLL required')
        self.dll_directory = os.add_dll_directory(str(directory))
        self.nvrtc = C.CDLL(str(paths[0]))
        self.driver = C.WinDLL('nvcuda.dll')
        P, I, U, S = C.c_void_p, C.c_int, C.c_uint, C.c_size_t
        self._signatures(self.nvrtc, {
            'nvrtcVersion': [C.POINTER(I), C.POINTER(I)],
            'nvrtcCreateProgram': [C.POINTER(P), C.c_char_p, C.c_char_p, I, P, P],
            'nvrtcCompileProgram': [P, I, C.POINTER(C.c_char_p)],
            'nvrtcGetProgramLogSize': [P, C.POINTER(S)],
            'nvrtcGetProgramLog': [P, P],
            'nvrtcGetCUBINSize': [P, C.POINTER(S)], 'nvrtcGetCUBIN': [P, P],
            'nvrtcDestroyProgram': [C.POINTER(P)],
        })
        self._signatures(self.driver, {
            'cuModuleLoadData': [C.POINTER(P), P],
            'cuModuleGetFunction': [C.POINTER(P), P, C.c_char_p],
            'cuModuleUnload': [P],
            'cuLaunchKernel': [P, U, U, U, U, U, U, U, P, C.POINTER(P), P],
        })
        major, minor = I(), I()
        self._call(self.nvrtc, 'nvrtcVersion', C.byref(major), C.byref(minor))
        if (str(torch.__version__), torch.version.cuda, major.value, minor.value) != ('2.10.0+cu126', '12.6', 12, 6):
            raise ContractError('This independently identified backend requires installed Torch 2.10.0+cu126 / NVRTC 12.6')
        arch = ''.join(map(str, torch.cuda.get_device_capability(self.device)))
        options = ['--std=c++17', '--gpu-architecture=sm_'+arch, '--fmad=false',
                   '--ftz=false', '--prec-div=true', '--prec-sqrt=true']
        program = P()
        self._call(self.nvrtc, 'nvrtcCreateProgram', C.byref(program), source.encode(), b'retipath_million.cu', 0, None, None)
        try:
            argv = (C.c_char_p * len(options))(*(x.encode() for x in options))
            status = self.nvrtc.nvrtcCompileProgram(program, len(options), argv)
            size = S()
            self._call(self.nvrtc, 'nvrtcGetProgramLogSize', program, C.byref(size))
            log = C.create_string_buffer(max(1, size.value))
            self._call(self.nvrtc, 'nvrtcGetProgramLog', program, log)
            self.compile_log = log.value.decode(errors='replace')
            if status:
                raise ContractError('NVRTC compilation failed: '+self.compile_log)
            self._call(self.nvrtc, 'nvrtcGetCUBINSize', program, C.byref(size))
            cubin = C.create_string_buffer(size.value)
            self._call(self.nvrtc, 'nvrtcGetCUBIN', program, cubin)
            self._call(self.driver, 'cuModuleLoadData', C.byref(self.module), cubin)
            self.functions = {}
            for name in names:
                function = P()
                self._call(self.driver, 'cuModuleGetFunction', C.byref(function), self.module, name.encode())
                self.functions[name] = function
            self.identity = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                nvrtc=[major.value, minor.value], dll=str(paths[0]),
                dll_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                cubin_sha256=hashlib.sha256(cubin.raw).hexdigest(), options=options,
                device=torch.cuda.get_device_name(self.device), architecture='sm_'+arch)
        except Exception:
            self.close()
            raise
        finally:
            self._call(self.nvrtc, 'nvrtcDestroyProgram', C.byref(program))

    @staticmethod
    def _signatures(lib, declarations):
        for name, args in declarations.items():
            function = getattr(lib, name)
            function.argtypes, function.restype = args, C.c_int

    @staticmethod
    def _call(lib, name, *args):
        status = getattr(lib, name)(*args)
        if status:
            raise ContractError(f'{name} failed with CUDA status {status}')

    def launch(self, name, count, tensors):
        if self.closed or self.torch.cuda.current_device() != self.device or int(self.torch.cuda.current_stream(self.device).cuda_stream) != self.stream:
            raise ContractError('CUDA module closed or device/stream changed')
        if any(t.device.type != 'cuda' or t.device.index != self.device or not t.is_contiguous() or t.requires_grad for t in tensors):
            raise ContractError('Owned contiguous tensors on the bound CUDA device required')
        pointers = [C.c_void_p(t.data_ptr()) for t in tensors]
        params = (C.c_void_p * len(pointers))(*(C.cast(C.byref(p), C.c_void_p) for p in pointers))
        self._call(self.driver, 'cuLaunchKernel', self.functions[name], (int(count)+127)//128,
                   1, 1, 128, 1, 1, 0, C.c_void_p(self.stream), params, None)

    def close(self):
        if self.closed:
            return
        if self.module.value:
            self.torch.cuda.synchronize(self.device)
            self._call(self.driver, 'cuModuleUnload', self.module)
            self.module.value = None
        self.closed = True
        self.dll_directory.close()
