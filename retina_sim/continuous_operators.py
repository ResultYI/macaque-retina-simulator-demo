"""Linear spatial projection of a shared continuous time basis, not a solve."""
import numpy as np
from scipy import sparse

from .contracts import ContractError
from .continuous_state import ExponentialForced


def project_common_exponential(matrix, waveform, *, output_state_ids):
    """Precompute A*y0 and A*q_j, then evaluate A*y(t) without stage SpMV.

    Exact for the represented waveform only. A common fixed decay is required;
    nonlinear powers/sigmoids remain outside and after this projection.
    """
    if (not isinstance(waveform, ExponentialForced) or not waveform.accepted
            or not sparse.isspmatrix_csr(matrix) or matrix.dtype != np.float64
            or matrix.shape[1] != len(waveform.y0) or not np.isfinite(matrix.data).all()
            or np.any(matrix.data < 0)):
        raise ContractError('Accepted forced waveform and matching original nonnegative FP64 CSR required')
    if not np.all(waveform.decay == waveform.decay[0]):
        raise ContractError('Shared decay basis absent: split by exact time-constant class or retain stage SpMV')
    return ExponentialForced(t0=waveform.t0, t1=waveform.t1, epoch=waveform.epoch,
        state_ids=output_state_ids, y0=np.asarray(matrix@waveform.y0),
        coefficients=np.asarray(matrix@waveform.coefficients.T).T,
        decay=np.asarray(waveform.decay[0]))
