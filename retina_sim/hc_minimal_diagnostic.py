"""Explicit opt-in for registered zero-feedback rows in a conditioned real graph."""
import numpy as np
from scipy import sparse
from .contracts import ContractError, canonical_hash
from .h1_h2_effective import matrix_id


def bind_diagnostic(A, B, *, cone_parent_rows, registered_unsupported_rows,
                    A_held_mass, B_held_mass, exterior_release, exterior_Vbar,
                    exterior_h, source_identity, feedback_off=False):
    """Keep fine dynamic weights and sum known held columns at a uniform background.

    B is always the original on-condition operator, even for an ablation.
    The only zero feedback rows admitted are exactly the explicit registry.
    """
    A, B = sparse.csr_matrix(A), sparse.csr_matrix(B)
    nh, nc = A.shape
    ids = np.asarray(cone_parent_rows)
    registry = np.asarray(registered_unsupported_rows)
    if (B.shape != (nc, nh) or ids.shape != (nc,) or ids.dtype.kind not in 'iu'
            or len(np.unique(ids)) != nc or registry.ndim != 1 or registry.dtype.kind not in 'iu'
            or len(np.unique(registry)) != len(registry) or not np.isin(registry, ids).all()
            or not source_identity or type(feedback_off) is not bool):
        raise ContractError('Explicit unique fine parent IDs and unsupported registry required')
    ah, bh = np.asarray(A_held_mass, float), np.asarray(B_held_mass, float)
    if ah.shape != (nh,) or bh.shape != (nc,):
        raise ContractError('Held-parent mass layout differs')
    if (not all(np.isfinite(v).all() and np.all(v >= 0) for v in (A.data, B.data, ah, bh))
            or not np.isfinite([exterior_release, exterior_Vbar, exterior_h]).all()
            or exterior_release < 0 or exterior_Vbar <= 0 or exterior_h < 0):
        raise ContractError('Invalid diagnostic weights or exterior background')
    am, bm = np.asarray(A.sum(1)).ravel(), np.asarray(B.sum(1)).ravel()
    unsupported = np.isin(ids, registry)
    if (not np.allclose(am+ah, 1., rtol=0, atol=2e-14)
            or not np.allclose(bm+bh, (~unsupported).astype(float), rtol=0, atol=2e-14)
            or np.any(bm[unsupported] != 0) or np.any(bh[unsupported] != 0)):
        raise ContractError('Unregistered zero B row or incomplete known parent weights')
    boundary = dict(A_release=ah*exterior_release, A_Vbar=ah*exterior_Vbar,
                    B_h=np.zeros(nc) if feedback_off else bh*exterior_h)
    used_B = sparse.csr_matrix(B.shape, dtype=float) if feedback_off else B.copy()
    identity = canonical_hash(dict(A=matrix_id(A), B_on=matrix_id(B), used_B=matrix_id(used_B),
        cone_parent_rows=ids, unsupported_feedback=registry, A_held_mass=ah, B_held_mass=bh,
        boundary=boundary, source_identity=source_identity, feedback_off=feedback_off))
    return dict(A=A.copy(), B=used_B, boundary=boundary, identity=identity,
        unsupported_feedback=unsupported, registered_unsupported_rows=registry.copy(),
        mass_ledger=dict(A_dynamic_mass=am, A_held_parent_mass=ah,
            A_missing_source_mass=np.zeros(nh), B_dynamic_mass=bm, B_held_parent_mass=bh,
            B_unsupported_deficit=unsupported.astype(float),
            B_ablation_removed_mass=(bm+bh) if feedback_off else np.zeros(nc)),
        crop_renormalized=False, diagnostic_only=True, production_adopted=False)


def require_diagnostic_native(binding, *, input_available, feedback_geometry_complete,
                              background_max_rhs, background_tolerance=1e-8):
    """Registered absence is allowed; unknown geometry/input is not."""
    nc = binding['B'].shape[0]
    for value in (input_available, feedback_geometry_complete):
        a = np.asarray(value)
        if a.shape != (nc,) or a.dtype != np.bool_ or not a.all():
            raise ContractError('Diagnostic input or local contact inventory incomplete')
    if (not binding.get('diagnostic_only') or not np.isfinite(background_max_rhs)
            or not np.isfinite(background_tolerance) or background_tolerance <= 0
            or not 0 <= background_max_rhs <= background_tolerance):
        raise ContractError('Diagnostic background admission failed')
    return dict(identity=binding['identity'], admitted=True,
                permission='registered_zero_B_minimal_real_graph_diagnostic',
                production_adopted=False, boundary_convergence_qualified=False)
