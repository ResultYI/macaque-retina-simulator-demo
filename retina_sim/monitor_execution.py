"""Explicit reversible CPU native-row scheduling for background monitoring.

Execution option only: no changed RHS, checks, state, clocks or solver. The
original full RHS keeps its work lock and native-row reduction order. Configure
the process Numba pool before import; this helper never changes environment/OS.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import os
from .contracts import ContractError


@contextmanager
def background_monitor_threads(runner, threads=1):
    """Temporarily select threads only during the original CPU full RHS call.

    The caller retains this context around background preparation, not around a
    different solver. Unsupported explicit candidates fail before preparation;
    use threads=1 to retain the reference path. Settings restore on exceptions.
    """
    if type(threads) is not int or threads not in (1, 2, 4):
        raise ContractError('Monitor native threads must be explicitly 1, 2 or 4')
    import numba
    original = runner.model.rhs_flat
    previous = numba.get_num_threads()
    receipt = dict(procedure='background_native_row_threads_v1',requested_threads=threads,
        previous_process_thread_mask=previous,pool_limit=int(numba.config.NUMBA_NUM_THREADS),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Original CPU full RHS only; CUDA inner, monitor checks and all clocks unchanged',
        applied=False,restored=False)
    if threads > numba.config.NUMBA_NUM_THREADS:
        raise ContractError('Requested monitor threads exceed process Numba pool; set NUMBA_NUM_THREADS before import, or explicitly use1')
    kernel = getattr(runner.model, '_compiled_rhs', None)
    if threads > 1:
        if kernel is None or not kernel.native_fused:
            raise ContractError('Parallel monitor requires the existing fused independent native-row RHS; use1 for the reference path')
        if threads > kernel.native_threads:
            raise ContractError('Requested monitor threads exceed existing native-row implementation bound')
        if any(os.environ.get(key) != '1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
            raise ContractError('Explicit parallel monitor requires OMP/BLAS process thread limits1 before import')
    if threads == previous == 1:
        try:
            yield receipt
        finally:
            receipt['restored'] = True
        return
    def rhs(*args, **kwargs):
        saved = numba.get_num_threads()
        try:
            numba.set_num_threads(threads)
            return original(*args, **kwargs)
        finally:
            numba.set_num_threads(saved)
    runner.model.rhs_flat = rhs
    receipt['applied'] = True
    try:
        yield receipt
    finally:
        runner.model.rhs_flat = original
        # The RHS wrapper restores after every invocation, including failures.
        receipt['restored'] = numba.get_num_threads() == previous
        if not receipt['restored']:
            numba.set_num_threads(previous)
            receipt['restored'] = True
