"""Opt-in effective spatial pooling; raw local contacts remain independent of A.

This is a finite-parent research operator, not a calibrated central-retina RF.
Uncertified exterior support is reported, never repaired by crop normalization.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from .contracts import ContractError, canonical_hash
from .hc_group_projection import _normalize_complete_rows, _member_mean, _validate_mapping
from .h1_h2_effective import matrix_id


def exponential_tail(radius_mm, lambda_mm):
    """Outside-disk mass of exp(-r/lambda)/(2*pi*lambda**2) in 2D."""
    if not np.isfinite(lambda_mm) or lambda_mm <= 0:
        raise ContractError('Positive finite spatial constant required')
    r = np.asarray(radius_mm, dtype=float)
    if not np.isfinite(r).all() or np.any(r < 0):
        raise ContractError('Finite nonnegative radius required')
    x = r / lambda_mm
    return (1 + x) * np.exp(-x)


def support_radius(lambda_mm, tail_mass):
    if not np.isfinite(tail_mass) or not 0 < tail_mass < 1:
        raise ContractError('Explicit tail mass budget must lie strictly between zero and one')
    exponential_tail(0., lambda_mm)
    hi = 1.
    while (1 + hi) * np.exp(-hi) > tail_mass:
        hi *= 2
    return lambda_mm * brentq(lambda x: (1+x)*np.exp(-x)-tail_mass, 0., hi)


def _exponential_weights(distance, length, coefficient=None):
    kernel = np.exp(-(distance-distance.min())/length)
    if coefficient is not None:
        kernel *= coefficient
    return kernel/kernel.sum()


def precompute_functional_crop(source_xy, source_types, hc_xy, hc_types, dynamic_source_rows,
                               uniform_A, uniform_held_mass, *, configuration,
                               source_envelope_center, source_envelope_radius_mm):
    """Stream full functional rows into a sparse dynamic crop and held-mass ledger."""
    if (configuration['schema'] != 'HC_biexponential_functional'
            or configuration['coordinate_frame'] != 'OPL_pedicle_mm' or configuration['H2'] is not None):
        raise ContractError('Functional H1 configuration with unchanged uniform H2 required')
    xy, hx = np.asarray(source_xy, float), np.asarray(hc_xy, float)
    ct, ht = np.asarray(source_types), np.asarray(hc_types)
    dyn = np.asarray(dynamic_source_rows)
    uniform = sparse.csr_matrix(uniform_A)
    held = np.asarray(uniform_held_mass, float).copy()
    if (xy.shape != (len(ct), 2) or hx.shape != (len(ht), 2) or not len(ct)
            or dyn.ndim != 1 or dyn.dtype.kind not in 'iu' or len(np.unique(dyn)) != len(dyn)
            or np.any(dyn < 0) or np.any(dyn >= len(ct)) or not set(ct) <= set('LMS')
            or not set(ht) <= {'H1','H2'} or not np.isfinite(xy).all() or not np.isfinite(hx).all()
            or uniform.shape != (len(ht), len(dyn)) or held.shape != (len(ht),)
            or not np.isfinite(held).all() or np.any(held < 0)):
        raise ContractError('Complete typed source envelope and fine dynamic mapping required')
    p = configuration['H1']; ls, ll, ws = p['lambda_s_um']/1000, p['lambda_l_um']/1000, p['w_s']
    if (ls, ll, ws) != (.020, .300, .15):
        raise ContractError('Final functional H1 parameters are fixed at 20/300/0.15')
    radius = support_radius(ll, configuration['tail_mass_budget'])
    center = np.asarray(source_envelope_center, float)
    if (center.shape != (2,) or not np.isfinite(center).all()
            or not np.isfinite(source_envelope_radius_mm)
            or np.any(np.linalg.norm(hx[ht == 'H1']-center, axis=1)+radius > source_envelope_radius_mm+1e-12)):
        raise ContractError('Source envelope does not contain complete finite functional support')
    tree = cKDTree(xy)
    local = np.full(len(ct), -1, dtype=np.int64); local[dyn] = np.arange(len(dyn))
    rr, cc, ss, llvalues = [], [], [], []
    count = np.zeros(len(ht), dtype=np.int64)
    component_held = np.zeros((2, len(ht)))
    for j in np.flatnonzero(ht == 'H1'):
        ids = np.asarray(tree.query_ball_point(hx[j], radius, return_sorted=True), dtype=np.int64)
        ids = ids[ct[ids] != 'S']
        if not len(ids):
            raise ContractError('Empty finite L/M functional source set')
        distance = np.linalg.norm(xy[ids]-hx[j], axis=1)
        short = _exponential_weights(distance, ls)
        long = _exponential_weights(distance, ll)
        inside = local[ids] >= 0
        rr.extend([j]*int(inside.sum())); cc.extend(local[ids[inside]])
        ss.extend(short[inside]); llvalues.extend(long[inside])
        component_held[:,j] = short[~inside].sum(), long[~inside].sum()
        held[j] = ws*component_held[0,j]+(1-ws)*component_held[1,j]
        count[j] = len(ids)
    shape = uniform.shape
    short = sparse.csr_matrix((ss, (rr,cc)), shape=shape)
    long = sparse.csr_matrix((llvalues, (rr,cc)), shape=shape)
    A = ws*short+(1-ws)*long+sparse.diags((ht == 'H2').astype(float))@uniform
    A = A.tocsr(); A.sort_indices()
    if (not np.isfinite(A.data).all() or np.any(A.data < 0)
            or not np.allclose(np.asarray(A.sum(1)).ravel()+held, 1., atol=2e-14, rtol=0)):
        raise ContractError('Functional crop/held mass conservation failed')
    return dict(A=A, short=short, long=long, held_mass=held, component_held_mass=component_held,
                full_LM_count=count, support_radius_mm=radius, crop_renormalized=False)


def build_spatial_operators(raw_contacts, cone_pedicle_mm, hc_anchor_mm,
                            cone_types, hc_types, *, configuration,
                            parent_disk_radius_mm, explicit_hc_coupling=False):
    """Build complete fine-cell A and typed local B before grouping or cropping.

    All coordinates use the same OPL origin in mm. The parent disk declaration
    must describe the supplied complete tissue, not the requested output ROI.
    Long-range coefficients are equal within the original type eligibility;
    they are effective inputs and are never inserted into the contact matrix.
    """
    if explicit_hc_coupling:
        raise ContractError('Effective wide kernel and explicit HC coupling are mutually exclusive')
    if set(configuration) != {'schema', 'H1', 'H2', 'coordinate_frame', 'tail_mass_budget'}:
        raise ContractError('Explicit independent H1/H2 spatial configuration required')
    if (configuration['schema'] not in ('HC_biexponential_candidate_v1', 'HC_biexponential_functional')
            or configuration['coordinate_frame'] != 'OPL_pedicle_mm'):
        raise ContractError('Unsupported spatial schema or coordinate frame')
    functional = configuration['schema'] == 'HC_biexponential_functional'
    if functional and configuration['H2'] is not None:
        raise ContractError('Final functional mode keeps H2 uniform')
    xy, hx = np.asarray(cone_pedicle_mm, float), np.asarray(hc_anchor_mm, float)
    ct, ht = np.asarray(cone_types), np.asarray(hc_types)
    raw = sparse.csr_matrix(raw_contacts, dtype=float, copy=True)
    raw.sum_duplicates(); raw.eliminate_zeros(); raw.sort_indices()
    if (xy.shape != (len(ct), 2) or hx.shape != (len(ht), 2)
            or raw.shape != (len(ht), len(ct)) or not len(ct) or not len(ht)
            or not np.isfinite(xy).all() or not np.isfinite(hx).all()
            or not set(ct) <= set('LMS') or not set(ht) <= {'H1', 'H2'}
            or not np.isfinite(parent_disk_radius_mm) or parent_disk_radius_mm <= 0
            or np.any(np.linalg.norm(xy, axis=1) > parent_disk_radius_mm + 1e-12)
            or np.any(np.linalg.norm(hx, axis=1) > parent_disk_radius_mm + 1e-12)):
        raise ContractError('Complete finite typed parent geometry in mm required')
    uniform, active = _normalize_complete_rows(raw)
    uniform.sort_indices()
    edges = raw.tocoo()
    if np.any((ht[edges.row] == 'H1') & (ct[edges.col] == 'S')):
        raise ContractError('Raw local contacts violate H1 S exclusion')
    allowed = ((ht[edges.row] == 'H1') & (ct[edges.col] != 'S')
               | (ht[edges.row] == 'H2') & (ct[edges.col] == 'S'))
    B, cone_active = _normalize_complete_rows(sparse.csr_matrix(
        (edges.data[allowed], (edges.col[allowed], edges.row[allowed])), shape=raw.shape[::-1]))
    tree = cKDTree(xy)
    rows, cols, values = [], [], []
    support = np.zeros(len(ht))
    certified = np.ones(len(ht), dtype=bool)
    short_rows, long_rows = [], []
    for j, kind in enumerate(ht):
        p = configuration[kind]
        start, stop = raw.indptr[j:j+2]
        local = raw.indices[start:stop]
        if p is None:
            rows.extend([j]*len(local)); cols.extend(local)
            values.extend(uniform.data[uniform.indptr[j]:uniform.indptr[j+1]])
            continue
        if set(p) != {'lambda_s_um', 'lambda_l_um', 'w_s', 'source', 'evidence_class'}:
            raise ContractError('Kernel parameters must carry independent source and evidence class')
        ls, ll, w = p['lambda_s_um']/1000., p['lambda_l_um']/1000., p['w_s']
        if (not np.isfinite([ls, ll, w]).all() or not 0 < ls < ll
                or not 0 <= w <= 1 or not p['source'] or not p['evidence_class']):
            raise ContractError('Invalid sourced biexponential parameters')
        radius = support_radius(ll, configuration['tail_mass_budget'])
        support[j] = radius
        certified[j] = np.linalg.norm(hx[j]) + radius <= parent_disk_radius_mm + 1e-12
        distant = np.asarray(sorted(tree.query_ball_point(hx[j], radius)), dtype=int)
        if kind == 'H1':
            distant = distant[ct[distant] != 'S']
        if (w > 0 and not len(distant if functional else local)) or (w < 1 and not len(distant)):
            raise ContractError(f'HC row {j} has an empty active mixture component; no weight redistribution')
        for ids, coefficient, length, weight, ledger in (
                (distant if functional else local, None if functional else raw.data[start:stop], ls, w, short_rows),
                (distant, np.ones(len(distant)), ll, 1-w, long_rows)):
            if weight == 0:
                continue
            distance = np.linalg.norm(xy[ids]-hx[j], axis=1)
            normalized = _exponential_weights(distance, length, coefficient)
            rows.extend([j]*len(ids)); cols.extend(ids); values.extend(weight*normalized)
            ledger.append((j, ids, normalized))
    A = sparse.csr_matrix((values, (rows, cols)), shape=raw.shape)
    expected = active.copy()
    for kind in ('H1', 'H2'):
        if configuration[kind] is not None:
            expected[ht == kind] = True
    if (not np.isfinite(A.data).all() or np.any(A.data < 0)
            or not np.allclose(np.asarray(A.sum(1)).ravel(), expected, atol=2e-14, rtol=0)):
        raise ContractError('Complete fine spatial row normalization failed')
    def component(records):
        rr, cc, vv = [], [], []
        for j, ids, weights in records:
            rr.extend([j]*len(ids)); cc.extend(ids); vv.extend(weights)
        return sparse.csr_matrix((vv, (rr, cc)), shape=raw.shape)
    identity = dict(configuration=configuration, raw_contacts=matrix_id(raw), A=matrix_id(A), B=matrix_id(B),
                    cone_pedicle_mm=xy, hc_anchor_mm=hx, cone_types=ct, hc_types=ht,
                    parent_disk_radius_mm=parent_disk_radius_mm,
                    implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return dict(A=A, B=B, short=component(short_rows), long=component(long_rows),
                uniform_A=uniform, raw_contacts=raw, support_radius_mm=support,
                support_certified=certified, HC_active=expected, cone_active=cone_active,
                identity=canonical_hash(identity), identity_record=identity,
                evidence=('functional exponentials on shared finite eligible sources' if configuration['schema']=='HC_biexponential_functional'
                          else 'legacy local narrow mask + eligible wide inputs; no measured synaptic strengths'),
                tail_budget_scope='continuum 2D tail only; not a native response or recursive closure bound',
                production_adopted=False, background_qualified=False)


def project_spatial_operators(result, cone_map, hc_map, cone_counts, hc_counts):
    """Aggregate fine weights by original members, retaining unsupported mass."""
    A, B = result['A'], result['B']
    cm, cc, cv = _validate_mapping(cone_map, cone_counts,
        (np.asarray(A.sum(0)).ravel() + np.asarray(B.sum(1)).ravel()) > 0, 'cones')
    hm, hc, hv = _validate_mapping(hc_map, hc_counts,
        (np.asarray(A.sum(1)).ravel() + np.asarray(B.sum(0)).ravel()) > 0, 'HC')
    for mapping, valid, labels in (
            (cm, cv, np.asarray(result['identity_record']['cone_types']) == 'S'),
            (hm, hv, np.asarray(result['identity_record']['hc_types']) == 'H2')):
        for group in np.unique(mapping[valid]):
            if len(np.unique(labels[mapping == group])) != 1:
                raise ContractError('Grouping cannot mix distinct typed feedback paths')
    outA = _member_mean(A, hm, cm, hc, (len(hc), len(cc)))
    outB = _member_mean(B, cm, hm, cc, (len(cc), len(hc)))
    af = np.bincount(hm[hv], weights=np.asarray(A.sum(1)).ravel()[hv], minlength=len(hc))/hc
    bf = np.bincount(cm[cv], weights=np.asarray(B.sum(1)).ravel()[cv], minlength=len(cc))/cc
    if not (np.allclose(np.asarray(outA.sum(1)).ravel(), af, atol=2e-14, rtol=0)
            and np.allclose(np.asarray(outB.sum(1)).ravel(), bf, atol=2e-14, rtol=0)):
        raise ContractError('All-member projection lost fine row mass')
    return dict(A=outA, B=outB, HC_supported_member_fraction=af, cone_supported_member_fraction=bf,
                identity=canonical_hash(dict(parent=result['identity'],cone_map=cm,hc_map=hm,
                    cone_counts=cc,hc_counts=hc,A=matrix_id(outA),B=matrix_id(outB))),
                cone_map=cm.copy(),hc_map=hm.copy(),production_adopted=False)


def condition_spatial_background(A, B, cone_rows, hc_rows, *, release0, Vbar0, Vh0,
                                 exterior, background_identity):
    """Crop a declared parent without renormalizing, retaining its DC products.

    ``exterior`` explicitly accounts for mass absent even from that parent.
    Its products are zero only when their missing mass is explicitly zero.
    This algebraic binding is not a background or closed-loop qualification.
    """
    A, B = (sparse.csr_matrix(m, dtype=float, copy=True) for m in (A, B))
    if B.shape != A.shape[::-1] or not all(A.shape):
        raise ContractError('Background parent A/B dimensions differ or are empty')
    for m in (A, B):
        m.sum_duplicates(); m.eliminate_zeros(); m.sort_indices()
        if not np.isfinite(m.data).all() or np.any(m.data < 0):
            raise ContractError('Background parent weights must be finite nonnegative')
    nh, nc = A.shape

    def vector(value, n, name):
        a = np.asarray(value, dtype=float)
        if a.shape != (n,) or not np.isfinite(a).all() or np.any(a < 0):
            raise ContractError('Invalid background vector: '+name)
        return a.copy()

    def rows(value, n):
        a = np.asarray(value)
        if (a.ndim != 1 or a.dtype.kind not in 'iu' or not len(a)
                or np.any(a < 0) or np.any(a >= n) or len(np.unique(a)) != len(a)):
            raise ContractError('Distinct nonempty parent row IDs required')
        return a.copy()

    ci, hi = rows(cone_rows, nc), rows(hc_rows, nh)
    r0 = vector(release0, nc, 'release0')
    v0 = vector(Vbar0, nc, 'Vbar0')
    h0 = vector(Vh0, nh, 'Vh0')
    if np.any(v0 <= 0) or not isinstance(background_identity, str) or not background_identity:
        raise ContractError('Positive Vbar and explicit background identity required')
    required = {'A_missing_mass', 'B_missing_mass', 'A_release', 'A_Vbar', 'B_h', 'evidence'}
    if not isinstance(exterior, dict) or set(exterior) != required or not exterior['evidence']:
        raise ContractError('Explicit exterior mass, background products and evidence required')
    ext = {key: vector(exterior[key], nc if key.startswith('B_') else nh, key)
           for key in required - {'evidence'}}
    am, bm = np.asarray(A.sum(1)).ravel(), np.asarray(B.sum(1)).ravel()
    for mass, missing in ((am, ext['A_missing_mass']), (bm, ext['B_missing_mass'])):
        if not np.allclose(mass+missing, 1., rtol=0, atol=2e-14):
            raise ContractError('Parent plus declared missing mass must equal one; no inferred repair')
    for key, mass in (('A_release', 'A_missing_mass'), ('A_Vbar', 'A_missing_mass'),
                      ('B_h', 'B_missing_mass')):
        if np.any(ext[key][ext[mass] == 0] != 0):
            raise ContractError('Nonzero exterior product without missing mass')
    if np.any((ext['A_missing_mass'] > 0) & (ext['A_Vbar'] <= 0)):
        raise ContractError('Missing A mass requires explicit positive Vbar background')
    ca, cb = A[hi][:, ci].tocsr(), B[ci][:, hi].tocsr()
    # Form exterior-only products directly, avoiding cancellation of large DC terms.
    outside_c = np.ones(nc, bool); outside_c[ci] = False
    outside_h = np.ones(nh, bool); outside_h[hi] = False
    boundary = dict(A_release=np.asarray(A[hi][:, outside_c]@r0[outside_c])+ext['A_release'][hi],
                    A_Vbar=np.asarray(A[hi][:, outside_c]@v0[outside_c])+ext['A_Vbar'][hi],
                    B_h=np.asarray(B[ci][:, outside_h]@h0[outside_h])+ext['B_h'][ci])
    ledger = dict(A_dynamic_mass=np.asarray(ca.sum(1)).ravel(),
                  B_dynamic_mass=np.asarray(cb.sum(1)).ravel(),
                  A_held_parent_mass=np.asarray(A[hi][:, outside_c].sum(1)).ravel(),
                  B_held_parent_mass=np.asarray(B[ci][:, outside_h].sum(1)).ravel(),
                  A_missing_source_mass=ext['A_missing_mass'][hi],
                  B_missing_source_mass=ext['B_missing_mass'][ci])
    identity = canonical_hash(dict(parent_A=matrix_id(A), parent_B=matrix_id(B),
        cone_rows=ci, hc_rows=hi, release0=r0, Vbar0=v0, Vh0=h0, exterior=exterior,
        background_identity=background_identity,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return dict(A=ca, B=cb, boundary=boundary, mass_ledger=ledger, identity=identity,
                cone_rows=ci, hc_rows=hi, crop_renormalized=False,
                background_qualified=False, closed_loop_qualified=False,
                external_background_evidence=exterior['evidence'])
