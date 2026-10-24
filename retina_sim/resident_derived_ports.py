"""Strict source-derived ribbon output check; original state budgets unchanged."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math

ADDENDUM_FILENAME = 'resident_ribbon_event_rate_addendum_v1.json'
ADDENDUM_SHA256 = '21e2cc8ad4a00ea3b9d4f331109a4c11c88f793334653cd4008addaa3026f735'
BUDGET_KIND = 'rod_ribbon_event_rate_bound_v1'
ROD_SOURCE_SHA256 = '5601dd757e22177fb02ebd72a4305ea145f8ddd05aa130822fe43f95b27c862c'
PARAMETERS = {
    'Npool': ('dimensionless', 'release_pool_size'),
    'U': ('dimensionless', 'release_probability'),
    'rmax': ('events/s', 'release_attempt_rate'),
    'r_half': ('V', 'release_half_voltage'),
    'r_slope': ('V', 'release_voltage_slope'),
}


def load_addendum():
    root = Path(__file__).resolve().parents[1]
    raw = (root/'configurations/numerical_equivalence'/ADDENDUM_FILENAME).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ADDENDUM_SHA256:
        raise ValueError('Frozen ribbon output addendum changed')
    if hashlib.sha256(Path(__file__).with_name('rod_pathway.py').read_bytes()).hexdigest() != ROD_SOURCE_SHA256:
        raise ValueError('Unreviewed ribbon source formula change')
    return json.loads(raw)


def ribbon_descriptors(compiled):
    """Bind only the two original typed ribbon ports and actual parameters."""
    from .contracts import canonical_hash
    load_addendum()
    specifications = {s.component_id:s for s in compiled.spec.components}
    result = {}
    for cid, component in compiled.components.items():
        declaration = component.outputs.get('event_rate')
        if declaration is None or declaration.unit != 'events/s':
            continue
        spec = specifications[cid]
        kind = spec.options.get('cell_kind')
        if (component.mechanism != 'ReceptorConductanceMembrane' or kind not in ('rod_BC_ON','AII') or
                declaration.meaning != 'expected_release_event_rate'):
            raise ValueError('Unsupported events/s source: '+cid)
        states = {s.state_name:s for s in component.state_specs}
        if (states['V'].unit != 'V' or states['V'].meaning != 'physical_membrane_voltage' or
                states['x'].unit != 'dimensionless' or states['x'].meaning != 'adaptation_gate' or
                tuple(states['V'].cell_ids) != tuple(declaration.cell_ids) or
                tuple(states['x'].cell_ids) != tuple(declaration.cell_ids)):
            raise ValueError('Ribbon dependency state layout differs: '+cid)
        params = {}
        for name,(unit,meaning) in PARAMETERS.items():
            values = component.parameters.quantity(name,unit,meaning,(len(declaration.cell_ids),)).values
            params[name] = [float(value).hex() for value in values]
        row = dict(budget_kind=BUDGET_KIND, component=cid, phenotype=kind,
            cell_ids=list(declaration.cell_ids), unit=declaration.unit, meaning=declaration.meaning,
            voltage_state=cid+'.V', resource_state=cid+'.x', parameters_hex=params,
            parameter_hash=component.parameters.parameter_hash,
            component_identity_hash=canonical_hash(component.identity),
            source_sha256=ROD_SOURCE_SHA256, addendum_sha256=ADDENDUM_SHA256)
        validate_descriptor(row)
        result[cid+'.event_rate'] = row
    return result


def validate_descriptor(row):
    if (row.get('budget_kind') != BUDGET_KIND or row.get('unit') != 'events/s' or
            row.get('meaning') != 'expected_release_event_rate' or
            row.get('phenotype') not in ('rod_BC_ON','AII') or
            row.get('addendum_sha256') != ADDENDUM_SHA256 or
            row.get('source_sha256') != ROD_SOURCE_SHA256 or
            row.get('voltage_state') != row['component']+'.V' or
            row.get('resource_state') != row['component']+'.x' or
            set(row.get('parameters_hex',{})) != set(PARAMETERS)):
        raise ValueError('Unsupported ribbon derived-port descriptor')
    n = len(row['cell_ids'])
    params = {}
    for name,values in row['parameters_hex'].items():
        if len(values) != n:
            raise ValueError('Ribbon parameter/cell order length mismatch')
        decoded = [float.fromhex(v) for v in values]
        if any(not math.isfinite(v) for v in decoded):
            raise ValueError('Nonfinite ribbon parameter')
        params[name] = decoded
    for i in range(n):
        if not (params['Npool'][i] > 0 and 0 < params['U'][i] <= 1 and
                params['rmax'][i] > 0 and params['r_slope'][i] > 0):
            raise ValueError('Original ribbon positive-parameter/U domain differs')
    return params


def compare_cell(*, pool, U, rmax, r_half, r_slope, V_reference, V_candidate,
                 x_reference, x_candidate, rate_reference, rate_candidate, B_V, B_x):
    values = locals().copy()
    if any(type(v) is not float or not math.isfinite(v) for v in values.values()):
        raise ValueError('Finite explicitly converted binary64 scalar witnesses required')
    q = {k:Fraction.from_float(v) for k,v in values.items()}
    if not (q['pool'] > 0 and 0 < q['U'] <= 1 and q['rmax'] > 0 and q['r_slope'] > 0):
        raise ValueError('Original ribbon parameter domain')
    if q['B_V'] < 0 or q['B_x'] < 0:
        raise ValueError('Negative original state budget')
    lo,hi = Fraction.from_float(-1e-10),Fraction.from_float(1.+1e-10)
    if not (lo <= q['x_reference'] <= hi and lo <= q['x_candidate'] <= hi):
        raise ValueError('Original adaptation_gate domain; no clipping')
    state_ok = (abs(q['V_candidate']-q['V_reference']) <= q['B_V'] and
                abs(q['x_candidate']-q['x_reference']) <= q['B_x'])
    bound = q['pool']*q['U']*q['rmax']*(q['B_x']+hi*min(Fraction(1),q['B_V']/(4*q['r_slope'])))
    difference = abs(q['rate_candidate']-q['rate_reference'])
    def exact(value):
        return [str(value.numerator),str(value.denominator)]
    return dict(passed=state_ok and difference <= bound, state_prerequisite=state_ok,
                allowance_exact=exact(bound), difference_exact=exact(difference),
                relative_allowance_exact=None if q['rate_reference']==0 else exact(bound/abs(q['rate_reference'])),
                added_roundoff_allowance=0)


def compare_vectors(descriptor, reference, candidate, V_reference, V_candidate,
                    x_reference, x_candidate, B_V, B_x, *, exact=False):
    import numpy as np
    params = validate_descriptor(descriptor)
    vectors = (reference,candidate,V_reference,V_candidate,x_reference,x_candidate,B_V,B_x)
    n = len(descriptor['cell_ids'])
    if any(not isinstance(v,np.ndarray) or v.dtype!=np.float64 or v.shape!=(n,) or
           not np.isfinite(v).all() for v in vectors):
        raise ValueError('Complete finite FP64 ribbon witnesses required')
    if exact:
        return dict(passed=bool(np.array_equal(reference,candidate)), cells=[], exact=True)
    rows = []
    for i in range(n):
        rows.append(compare_cell(pool=params['Npool'][i],U=params['U'][i],rmax=params['rmax'][i],
            r_half=params['r_half'][i],r_slope=params['r_slope'][i],
            V_reference=float(V_reference[i]),V_candidate=float(V_candidate[i]),
            x_reference=float(x_reference[i]),x_candidate=float(x_candidate[i]),
            rate_reference=float(reference[i]),rate_candidate=float(candidate[i]),
            B_V=float(B_V[i]),B_x=float(B_x[i])))
    return dict(passed=all(r['passed'] for r in rows),cells=rows,exact=False)
