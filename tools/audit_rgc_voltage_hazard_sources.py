"""Read-only source/units audit and exact algebra for the retained RGC sampler.

No assembly, ODE advance, random sampling, fitting, MATLAB or default changes.
The eight-state chain represents the existing three-bin binary history only.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import hashlib
import itertools
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.inner import HistoryRGC

BASE = ROOT / 'reports/modeling_scope_20261020'
DATA = ROOT / 'data/source_data/latimer2019_figshare_v1'
SNAPSHOT = BASE / 'standard_stimulus_rate_audit_v1/rate_audit/actual_RGC_parameter_snapshot.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def exact_chain(V, alpha, mu, beta, kernel, dt):
    states = np.asarray(list(itertools.product((0, 1), repeat=3)), int)
    lookup = {tuple(s): i for i, s in enumerate(states)}
    # Exercise the actual production intensity function, without a network run.
    model = SimpleNamespace(alpha=alpha, mu=mu, beta=beta, history_kernel=kernel)
    lam = HistoryRGC.intensity(model, np.full(8, V), states.T)
    independent = alpha * np.logaddexp(0., (V + states @ kernel - mu) / beta)
    np.testing.assert_allclose(lam, independent, rtol=1e-14, atol=1e-13)
    p = -np.expm1(-lam * dt)
    P = np.zeros((8, 8))
    for i, h in enumerate(states):
        P[i, lookup[(0, h[0], h[1])]] += 1 - p[i]
        P[i, lookup[(1, h[0], h[1])]] += p[i]
    A = P.T - np.eye(8)
    A[-1] = 1.
    b = np.zeros(8)
    b[-1] = 1.
    pi = np.linalg.solve(A, b)
    residual = float(np.max(np.abs(pi @ P - pi)))
    assert residual < 1e-14 and np.min(pi) >= 0 and abs(pi.sum() - 1) < 1e-14
    # Converting *all* voltage quantities to mV or to a leak-relative origin
    # must preserve the argument and rate. Neither operation is a new fit.
    mv = alpha * np.logaddexp(0., (1000*V + states@(1000*kernel) - 1000*mu)/(1000*beta))
    rel = alpha * np.logaddexp(0., (states@kernel - (mu-V))/beta)
    np.testing.assert_allclose(lam, mv, rtol=1e-14, atol=1e-13)
    np.testing.assert_allclose(lam, rel, rtol=1e-14, atol=1e-13)
    rows = []
    for i, h in enumerate(states):
        rows.append(dict(history_1ms=int(h[0]), history_2ms=int(h[1]), history_3ms=int(h[2]),
                         history_mV=float(1000*h@kernel), stationary_probability=float(pi[i]),
                         conditional_hazard_Hz=float(lam[i]), bin_probability=float(p[i]),
                         weighted_count_rate_Hz=float(pi[i]*p[i]/dt)))
    return dict(zero_history_hazard_Hz=float(lam[0]), zero_history_count_rate_Hz=float(p[0]/dt),
                mean_conditional_hazard_Hz=float(pi@lam), stationary_count_rate_Hz=float(pi@p/dt),
                stationary_residual=residual, voltage_unit_and_origin_invariance=True), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Fail rather than silently audit a changed scientific baseline.
    frozen = json.loads((BASE/'standard_stimulus_rate_audit_v1/baseline_frozen_identity.json').read_bytes())
    checked = {}
    for group in ('files', 'fixed_parameters_and_equations'):
        checked[group] = len(frozen[group])
        for rel, expected in frozen[group].items():
            assert sha(ROOT/rel) == expected, ('Frozen baseline changed', rel)
    registry = json.loads((ROOT/'configurations/effective_dynamics_v1.json').read_bytes())
    source = registry['source_assets']['PRIMARY_PATHWAY_project_profile']
    assert sha(ROOT/source['path']) == source['sha256']
    snapshot = json.loads(SNAPSHOT.read_bytes())
    ledger, cases = [], {}
    names = {'history_kernel': 'spike_history_kernel', 'dt': 'spike_interval'}
    for cid, item in snapshot.items():
        profile = registry['profiles']['effective_'+cid]
        values = {}
        for key, entry in item['parameters'].items():
            rec = profile[names.get(key, key)]
            actual = np.asarray(entry['values'])
            assert entry['unit'] == rec['unit']
            np.testing.assert_array_equal(actual, np.broadcast_to(rec['value'], actual.shape))
            assert rec['source_id'] == 'PRIMARY_PATHWAY_project_profile'
            assert rec['evidence_class'] == 'project_approximation'
            values[key] = actual if key == 'history_kernel' else float(actual.flat[0])
            ledger.append(dict(component_id=cid, parameter=key, unit=entry['unit'],
                               value=json.dumps(actual.tolist()), source_id=rec['source_id'],
                               evidence_class=rec['evidence_class'], uncertainty=rec['uncertainty']))
        assert values['history_kernel'].shape == (3,) and values['dt'] == .001
        case, rows = exact_chain(values['EL'], values['alpha'], values['mu'], values['beta'],
                                 values['history_kernel'], values['dt'])
        case['parameters'] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in values.items()}
        case['leak_only_time_constant_seconds'] = values['C']/values['gL']
        cases[cid] = case
    for filename, rows_ in [('eight_state_stationary.csv', rows), ('parameter_provenance.csv', ledger)]:
        with (args.output/filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_[0]))
            writer.writeheader()
            writer.writerows(rows_)
    # The frozen XML uses internal equ10/equ26 for displayed equations 12/28.
    paper = DATA/'latimer2019_PMC6989090.xml'
    assert sha(paper) == '09b5ce244f7abbdcc110046d0fc424f51c39055016e47c951098d49e2ec9bf9f'
    root = ET.parse(paper).getroot()
    mns = '{http://www.w3.org/1998/Math/MathML}'
    equations = {}
    for el in root.iter('disp-formula'):
        if el.findtext('label') in ('(12)', '(28)'):
            fractions = [x for x in el.iter(mns+'mfrac') if ''.join(x[-1].itertext()).strip() == 'β']
            assert len(fractions) == 1
            equations[el.findtext('label')] = dict(xml_id=el.attrib['id'], voltage_argument_divided_by_beta=True)
    assert len(equations) == 2
    identities = {}
    tree = json.loads((DATA/'author_repository_tree.json').read_bytes())
    assert tree['sha'] == '7beb7e72b0678192f0bc98392a678cdff9601791'
    blobs = {r['path']: r['sha'] for r in tree['tree']}
    for local, remote in [('author_pyCBEM__RGC_CBEM.py','pyCBEM/RGC_CBEM.py'),
                          ('author_pyCBEM__utils.py','pyCBEM/utils.py')]:
        path = DATA/local
        raw = path.read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert blob == blobs[remote]
        identities[local] = dict(sha256=sha(path), git_blob_sha1=blob, identity_matches_frozen_tree=True)
    result = dict(schema='rgc_voltage_hazard_source_audit_v1', scope='Read-only source and exact algebra',
                  diagnosis='No proven implementation/unit error; uncalibrated project output and membrane parameters',
                  zero_synaptic_conductance_is_not_dark_stimulus=True, cases=cases,
                  source_identity=dict(project_source_sha256=source['sha256'],
                      project_verification_level=source['verification_level'],
                      identity_check_is_not_physiological_validation=True,
                      paper_sha256=sha(paper), paper_equations=equations, author_python=identities),
                  parameter_snapshot_sha256=sha(SNAPSHOT),
                  protected_baseline_checks=checked, production_parameter_changes=False,
                  production_config_changes=False, new_neural_seconds=0, monte_carlo_trials=0,
                  fitted_parameters=0, matlab_launches=0,
                  checks=['production intensity equals independent softplus algebra',
                          'exact eight-state stationary distribution and normalization',
                          'SI/mV and leak-relative voltage-origin invariance',
                          'all five saved RGC parameter groups equal declared project records',
                          'frozen paper MathML division by beta',
                          'author Python git-blob identity', 'frozen video and parameter hashes'])
    write_json(args.output/'result.json', result)
    print(json.dumps(dict(diagnosis=result['diagnosis'], first_case=next(iter(cases.values())),
                          baseline_checked=checked, checks=len(result['checks'])), indent=2))


if __name__ == '__main__':
    main()
