"""Fine-member eccentric H1 operators over existing OPL geometry.

The reference-domain extension is explicit and isolated from cell generation.
No spatial work belongs in the native RHS or the time runner.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .contracts import ContractError, canonical_hash
from .hc_biexponential import _exponential_weights, exponential_tail
from .hc_group_projection import _member_mean
from .hc_joint_v2 import JointV2HCDensityProfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_path(root, relative, sha256=None):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ContractError('Missing or out-of-workspace HC asset: '+str(relative))
    if sha256 is not None and digest(path) != sha256:
        raise ContractError('HC asset identity changed: '+str(relative))
    return path


def temporal_decision(root, binding):
    path = checked_path(root, binding['path'], binding['sha256'])
    result = json.loads(path.read_bytes())
    if (binding.get('parameter') != 'tau_h_ms' or binding.get('refit') is not False
            or binding.get('bounds_ms') != [10., 40.] or binding.get('fallback_ms') != 20.
            or result.get('status') != 'COMPLETE' or result['baseline']['tau_h_ms'] != 20.
            or result.get('total_parameter_points', 17) > 16):
        raise ContractError('Completed bounded single-parameter calibration required')
    baseline, candidate = result['baseline'], result['candidate']
    checked_path(root,'data/source_data/h1_temporal_targets.csv',result['protocol']['target_sha256'])
    expected_conditions={(100.,10.),(10.,10.),(1.,10.),(100.,100.),(10.,100.),(1.,100.)}
    for evaluation in (baseline,candidate):
        rows=evaluation['conditions']
        if (len(rows)!=6 or {(r['background_td'],r['pulse_ms']) for r in rows}!=expected_conditions
                or any(r['split']!=('heldout' if r['background_td']==1. else 'train') for r in rows)):
            raise ContractError('Calibration conditions or held-out split changed')
    if any(r['background_td'] == 1 for e in result['evaluations'] for r in e['conditions']):
        raise ContractError('Held-out condition entered the optimizer')
    accepted = (candidate['losses']['train'] < baseline['losses']['train']
                and candidate['losses']['heldout'] <= baseline['losses']['heldout']
                and all(r['polarity_hyperpolarizing'] for r in baseline['conditions']+candidate['conditions']))
    decision = 'ACCEPTED' if accepted else 'TEMPORAL_CALIBRATION_NOT_ACCEPTED'
    tau = float(candidate['tau_h_ms']) if accepted else 20.
    if (result['decision'] != decision or result['decision_tau_h_ms'] != tau
            or not np.isfinite(tau) or not 10 <= tau <= 40):
        raise ContractError('Calibration decision does not follow its recorded acceptance rule')
    return dict(tau_h_ms=tau, decision=decision, metrics_sha256=digest(path),
                protocol_identity=result['protocol_identity'])


def spatial_parameters(root, anchor_xy_mm, hc_types, configuration):
    cfg = configuration
    h1 = cfg.get('H1', {})
    reference = cfg.get('reference_evaluation', {})
    expected = dict(lambda_short='R(e)/ln(10)', lambda_long='0.300*R(e)/R(4.0)',
                    weight_short=.15, weight_long=.85, reference_lambda_long_mm=.3, support_factor=11.75637)
    if (cfg.get('schema') != 'HC_eccentric_biexponential' or cfg.get('H2') is not None
            or cfg.get('coordinate_frame') != 'OPL_pedicle_mm' or cfg.get('tail_mass_budget') != 1e-4
            or any(h1.get(k) != v for k, v in expected.items())
            or not h1.get('source') or not h1.get('evidence_class')
            or reference.get('eccentricity_mm') != 4. or not reference.get('authorization')
            or reference.get('policy') != 'same profile and contact formula; reference evaluation only; no geometry regeneration'):
        raise ContractError('Explicit authorized eccentric H1 spatial contract required')
    xy = np.asarray(anchor_xy_mm, float); types = np.asarray(hc_types)
    fovea = np.asarray(cfg['fovea_xy_mm'], float)
    if (xy.shape != (len(types), 2) or fovea.shape != (2,) or not np.isfinite(xy).all()
            or not np.isfinite(fovea).all() or not set(types) <= {'H1', 'H2'}):
        raise ContractError('Typed HC OPL anchors and explicit foveal origin required')
    profile = JointV2HCDensityProfile(root)
    if profile.profile_path.resolve() != checked_path(root, cfg['geometry_profile']):
        raise ContractError('Eccentric reference and actual contacts must use the same selected profile')
    # The user revoked the old reference-domain freeze. This view changes only
    # evaluation reach; the profile, coefficients, join and generated geometry stay intact.
    reference_profile = object.__new__(type(profile))
    reference_profile.__dict__.update(profile.__dict__)
    reference_profile.support_radius_mm = max(4., profile.support_radius_mm)
    r4 = float(reference_profile.contact_radius_mm(4., 'H1'))
    if not np.isfinite(r4) or r4 <= 0:
        raise ContractError('Existing contact formula is not positive/finite at 4 mm')
    eccentricity = np.linalg.norm(xy-fovea, axis=1)
    active = types == 'H1'
    radius = np.zeros(len(types)); short = np.zeros(len(types)); long = np.zeros(len(types))
    radius[active] = profile.contact_radius_mm(eccentricity[active], 'H1')
    short[active] = radius[active]/np.log(10.)
    long[active] = .3*radius[active]/r4
    support = 11.75637*long
    arrays = dict(eccentricity_mm=eccentricity, contact_radius_mm=radius,
                  lambda_short_mm=short, lambda_long_mm=long, support_radius_mm=support,
                  weight_short=np.where(active, .15, 0.), weight_long=np.where(active, .85, 0.))
    record = dict(configuration=cfg, geometry_profile_identity=profile.profile_hash,
        geometry_profile_sha256=profile.source_asset_hash, reference_contact_radius_mm=r4,
        original_application_limit_mm=profile.support_radius_mm,
        reference_evaluation_authorization=reference['authorization'],
        reference_is_new_geometry=False, anatomical_connections_changed=False,
        tail_budget=1e-4, tail_at_rounded_support_factor=float(exponential_tail(11.75637, 1.)),
        geometry_implementation_sha256=digest(Path(root)/'retina_sim/hc_joint_v2.py'),
        implementation_sha256=digest(Path(__file__)), arrays=arrays)
    return arrays, record, canonical_hash(record)


def precompute_crop(source_xy, source_types, hc_xy, hc_types, dynamic_source_rows,
                    uniform_A, uniform_held, *, parameters, envelope_center, envelope_radius_mm):
    """Normalize each complete eligible fine row, then retain dynamic columns."""
    xy, hx = np.asarray(source_xy, float), np.asarray(hc_xy, float)
    ct, ht = np.asarray(source_types), np.asarray(hc_types)
    dyn = np.asarray(dynamic_source_rows)
    uniform = sparse.csr_matrix(uniform_A)
    held = np.asarray(uniform_held, float).copy()
    center = np.asarray(envelope_center, float)
    if (xy.shape != (len(ct), 2) or hx.shape != (len(ht), 2) or center.shape != (2,)
            or not set(ct) <= set('LMS') or not set(ht) <= {'H1', 'H2'}
            or not np.isfinite(xy).all() or not np.isfinite(hx).all()
            or not np.isfinite(center).all() or not np.isfinite(envelope_radius_mm) or envelope_radius_mm <= 0
            or dyn.ndim != 1 or dyn.dtype.kind not in 'iu' or len(np.unique(dyn)) != len(dyn)
            or np.any(dyn < 0) or np.any(dyn >= len(ct))
            or uniform.shape != (len(ht), len(dyn)) or held.shape != (len(ht),)):
        raise ContractError('Complete typed source envelope and exact dynamic members required')
    h1 = ht == 'H1'
    ls, ll, support = [np.asarray(parameters[k]) for k in ('lambda_short_mm','lambda_long_mm','support_radius_mm')]
    if (any(a.shape != (len(ht),) or not np.isfinite(a).all() for a in (ls,ll,support))
            or np.any(ls[h1] <= 0) or np.any(ll[h1] <= ls[h1])
            or not np.array_equal(support[h1],11.75637*ll[h1])
            or np.any(np.linalg.norm(hx[h1]-center,axis=1)+support[h1] > envelope_radius_mm+1e-12)):
        raise ContractError('Source envelope cannot cover every complete eccentric H1 support')
    tree = cKDTree(xy)
    local = np.full(len(ct), -1, dtype=np.int64); local[dyn] = np.arange(len(dyn))
    indptr = [0]; indices = []; values = []
    component_held = np.zeros((2, len(ht))); counts = np.zeros(len(ht), dtype=np.int64)
    witnesses = {}
    h1_rows = np.flatnonzero(h1)
    selected = set(h1_rows[[np.argmin(parameters['eccentricity_mm'][h1]),
                             np.argmax(parameters['eccentricity_mm'][h1])]]) if len(h1_rows) else set()
    for j in range(len(ht)):
        if not h1[j]:
            begin,end = uniform.indptr[j:j+2]
            cols, weights = uniform.indices[begin:end].copy(), uniform.data[begin:end].copy()
        else:
            ids = np.asarray(tree.query_ball_point(hx[j],support[j],return_sorted=True),dtype=np.int64)
            ids = ids[ct[ids] != 'S']
            if not len(ids):
                raise ContractError('Empty eligible H1 support; no redistribution')
            distance = np.linalg.norm(xy[ids]-hx[j],axis=1)
            short, long = _exponential_weights(distance,ls[j]), _exponential_weights(distance,ll[j])
            inside = local[ids] >= 0
            cols = local[ids[inside]]
            weights = .15*short[inside]+.85*long[inside]
            component_held[:,j] = short[~inside].sum(), long[~inside].sum()
            held[j] = .15*component_held[0,j]+.85*component_held[1,j]
            counts[j] = len(ids)
            if j in selected:
                witnesses[j] = dict(source_rows=ids, distance_mm=distance,
                    short_unitmass=short, long_unitmass=long, A=.15*short+.85*long)
        indices.append(cols); values.append(weights); indptr.append(indptr[-1]+len(cols))
    A = sparse.csr_matrix((np.concatenate(values),np.concatenate(indices),np.array(indptr)),shape=uniform.shape)
    A.sort_indices()
    if (not np.isfinite(A.data).all() or np.any(A.data < 0)
            or A[h1][:,ct[dyn]=='S'].nnz
            or not np.allclose(np.asarray(A.sum(1)).ravel()+held,1.,atol=2e-14,rtol=0)):
        raise ContractError('Fine eccentric A type/mass conservation failed')
    return dict(A=A, held_mass=held, component_held_mass=component_held,
                full_LM_count=counts, witnesses=witnesses, crop_renormalized=False)


def project_members(A, held, cone_map, hc_map, cone_counts, hc_counts):
    """Use original members, never group centroids or cropped row normalization."""
    cm, hm = np.asarray(cone_map), np.asarray(hc_map)
    cc, hc = np.asarray(cone_counts), np.asarray(hc_counts)
    if (cm.shape != (A.shape[1],) or hm.shape != (A.shape[0],)
            or cm.dtype.kind not in 'iu' or hm.dtype.kind not in 'iu'
            or np.any(cm < 0) or np.any(hm < 0)
            or np.any(cm >= len(cc)) or np.any(hm >= len(hc))
            or np.any(cc <= 0) or np.any(hc <= 0)
            or not np.array_equal(np.bincount(cm,minlength=len(cc)),cc)
            or not np.array_equal(np.bincount(hm,minlength=len(hc)),hc)):
        raise ContractError('Existing complete member counts and maps required')
    grouped = _member_mean(A,hm,cm,hc,(len(hc),len(cc)))
    exterior = np.bincount(hm,weights=held,minlength=len(hc))/hc
    if not np.allclose(np.asarray(grouped.sum(1)).ravel()+exterior,1.,atol=2e-14,rtol=0):
        raise ContractError('Member projection lost dynamic/held mass')
    return grouped, exterior
