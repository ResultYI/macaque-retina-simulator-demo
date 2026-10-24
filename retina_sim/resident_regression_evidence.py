"""Read-only verification of the actual reference-repair regression evidence.

No simulations, project imports, source edits, baseline regeneration or test
reruns. A successful inspection verifies existing reference-repair regressions
only. The missing event-rate numerical clause remains a separate blocker.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET


RECEIPT = 'reports/resident_runtime_validation_20261022_v1/reference_regression_receipt.json'
SHA256 = '4b90feac70d1186449854ad29c1b49bf461649944dc417682ae72ac119d3a9ef'


def load_reference_regression_receipt(root=None, receipt_path=None):
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    if receipt_path is None or Path(receipt_path).resolve() != (root/RECEIPT).resolve():
        raise ValueError("Explicit exact reviewed receipt path required")
    def path(name):
        candidate = (root/name).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError('Receipt path leaves the selected project')
        return candidate
    def digest(name):
        return hashlib.sha256(path(name).read_bytes()).hexdigest()
    if digest(RECEIPT) != SHA256:
        raise ValueError('Actual reviewed regression receipt changed')
    receipt = json.loads(path(RECEIPT).read_bytes())
    if (receipt['schema'] != 'resident_reference_regression_receipt_v1' or
            receipt['status'] != 'FOCUSED_REGRESSIONS_PASSED' or
            receipt['unique_cases_passed'] != 42 or
            any(receipt[key] is not False for key in
                ('full_graph_numerical_pass','performance_validated','fit_eligible'))):
        raise ValueError('Regression evidence scope differs')
    for group in ('source_sha256','evidence_sha256'):
        for name, expected in receipt[group].items():
            if digest(name) != expected:
                raise ValueError('Changed '+group+': '+name)
    base = 'reports/resident_runtime_validation_20261022_v1/'
    first = json.loads(path(base+'focused_r1/execution.json').read_bytes())
    second = json.loads(path(base+'focused_r2/execution.json').read_bytes())
    if first['returncode'] != 1 or first['timeout'] or second['returncode'] != 0 or second['timeout']:
        raise ValueError('Recorded regression process outcomes differ')
    tree = ET.parse(path(base+'focused_r2/pytest.xml'))
    cases = tree.findall('.//testcase')
    expected = {
        ('tests.test_boundary_replay','test_boundary_content_roundtrip_ownership_and_manifest_guard'),
        ('tests.test_resident_input_sides','test_pending_regressions_cannot_become_a_numerical_pass'),
    }
    if ({(c.get('classname'),c.get('name')) for c in cases} != expected or
            any(list(c.findall('failure'))+list(c.findall('error'))+list(c.findall('skipped')) for c in cases)):
        raise ValueError('The two initial setup errors lack the matching passing rerun')
    return dict(receipt_sha256=SHA256, reference_repairs_verified=True, unique_cases_passed=42,
                runtime_gate_opened=False, numerical_validation=False,
                next_blocker='Independent numerical and derived-port gates still required',
                required_test_rerun=False)
