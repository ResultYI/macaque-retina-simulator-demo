"""Authorized focused HC/BC software checks, before any new forward/fitting."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sys
import time

ROOT = Path(r'D:\macaque retinal simulator')

def main():
    # Reuse the installed pure-Python test runner, after locked scientific imports.
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    sys.path.append(r'D:\anaconda\Lib\site-packages')
    import pytest
    output = Path(sys.argv[2])
    start = time.perf_counter()
    targets = ['tests/test_hc_density_current_v2.py', 'tests/test_bc_pedicle_current_v1.py',
        'tests/test_hc_eccentric_projection.py::test_uses_HC_radius_and_keeps_empty_cones_without_nearest_fill',
        'tests/test_hc_eccentric_projection.py::test_radius_boundary_is_included_and_shape_errors_are_rejected',
        'tests/test_hc_arbor_static_pending.py', 'tests/test_hc_masked_construction_domain.py']
    code = pytest.main(['-q', '-p', 'no:cacheprovider', '--basetemp=' + str(output / 'pytest_tmp'), *targets])
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'test_exit_code': int(code),
        'wall_seconds': time.perf_counter() - start, 'targets': targets, 'pytest_source': pytest.__file__,
        'new_geometry_built': False, 'neural_run': False, 'fit_run': False}
    (output / 'SOFTWARE_CHECK.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))
    raise SystemExit(code)

if __name__ == '__main__':
    main()
