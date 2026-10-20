"""Fixed-weight, contact-first receptor reduction; no per-contact ODE states.

e_i = sum_j W_ij m_ij is exact only for fixed W, a common fixed tau_i,
and the declared initial mapping. Currents must share a postsynaptic driving
force. Nonlinearity precedes W; row sums are never silently normalized.
"""
import numpy as np
from scipy import sparse
from scipy.special import expit
from .contracts import ContractError

ROUTE = 'bc_ac_contact_first_candidate_v2'
POOLING = 'fixed_W_after_per_input_Hill_then_shared_tau_aggregate_gate'
MEANING = 'weighted_receptor_occupancy'
AGGREGATION = dict(fixed_weights=True, nonnegative_weights=True, fixed_tau=True,
    shared_contact_tau_per_cell=True, shared_postsynaptic_driving_force=True,
    initialization='explicit_weighted_aggregate',
    on_target='row_sum_W_minus_W_occupied', independent_contact_states=False)


def occupancy(u, K, hill):
    u = np.asarray(u, dtype=float)
    if not np.all(np.isfinite(u)) or np.any(u < 0):
        raise ContractError('Contact-first receptor requires finite nonnegative effective input')
    K = np.asarray(K, dtype=float); hill = np.asarray(hill, dtype=float)
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(hill)) or np.any(K <= 0) or np.any(hill <= 0):
        raise ContractError('Positive finite receptor K and Hill exponent required')
    return np.where(u == 0, 0., expit(hill*np.log(np.maximum(u, np.finfo(float).tiny)/K)))


def validate_reduction(W, tau, e0, *, contract, contact_tau=None, contact_initial=None):
    if contract != AGGREGATION:
        raise ContractError('Fixed W/shared tau/initial mapping/driving force aggregation contract required')
    W = sparse.csr_matrix(W, dtype=float, copy=True)
    tau = np.asarray(tau, dtype=float); e0 = np.asarray(e0, dtype=float)
    if (W.ndim != 2 or np.any(W.data < 0) or not np.all(np.isfinite(W.data)) or
            tau.shape != (W.shape[0],) or e0.shape != tau.shape or
            not np.all(np.isfinite(tau)) or np.any(tau <= 0)):
        raise ContractError('Finite fixed nonnegative W and per-cell positive fixed tau required')
    totals = np.asarray(W.sum(axis=1)).ravel()
    if np.any(totals <= 0) or not np.all(np.isfinite(e0)) or np.any(e0 < 0) or np.any(e0 > totals):
        raise ContractError('Initial weighted aggregate must lie in [0, row_sum(W)] with nonempty support')
    rows = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
    if contact_tau is not None:
        ct = np.asarray(contact_tau, dtype=float)
        if ct.shape != W.data.shape or not np.array_equal(ct, tau[rows]):
            raise ContractError('Contact time constants must equal the target cell fixed shared tau')
    if contact_initial is not None:
        m = np.asarray(contact_initial, dtype=float)
        if m.shape != W.data.shape or not np.all(np.isfinite(m)) or np.any((m < 0) | (m > 1)):
            raise ContractError('Contact initial gates require CSR-edge order and [0,1] values')
        mapped = np.bincount(rows, weights=W.data*m, minlength=W.shape[0])
        if not np.allclose(e0, mapped, rtol=0, atol=2e-15):
            raise ContractError('Initial aggregate differs from sum(W * contact_initial)')
    for array in (W.data, W.indices, W.indptr):
        array.setflags(write=False)
    totals.setflags(write=False)
    return W, totals


def aggregate_occupancy(W, u, K, hill):
    """Explicit module preprocessing, not f(Wu) and not a concentration map."""
    return np.asarray(W @ occupancy(u, K, hill)).ravel()
