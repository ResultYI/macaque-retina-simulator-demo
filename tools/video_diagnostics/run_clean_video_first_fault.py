"""Unchanged video entry point with an original-result restore failure witness."""
from pathlib import Path
import faulthandler
import hashlib
import json
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if os.environ.get('OPENBLAS_NUM_THREADS') != '1':
    raise RuntimeError('Original BLAS environment must precede NumPy import')
OUT = (ROOT / sys.argv[sys.argv.index('--output') + 1]).resolve()
if not OUT.is_relative_to(ROOT):
    raise RuntimeError('Diagnostic output must stay inside project')
faulthandler.enable()
# Engine imports only contracts/NumPy/SciPy, not assembly or any factories.
# The original driver still selects the archived compiler before factories.
import retina_sim.engine as engine
from first_json_fault_witness import install
from first_geometry_fault_witness import install as install_geometry_witness

capture_directory = Path(os.environ['RETINA_FIRST_ERROR_DIR'])
restore_json_reference, capture_terminal_error = install(capture_directory)
restore_geometry_reference = install_geometry_witness(capture_directory / 'geometry_gates')

original_restore = engine.restore_payload
original_hash = engine.canonical_hash
active = {}


def save(name, value):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n', encoding='utf8')
    os.replace(temporary, path)


def witnessed_hash(value):
    actual = original_hash(value)
    payload = active.get('payload')
    if payload is not None and value is payload['state']:
        active['original_first_hash'] = actual
        save('restore_exact_first_hash.json', dict(
            checkpoint_version=payload['checkpoint_version'], expected=payload['payload_hash'],
            actual_original_first_result=actual, PYTHONMALLOC=os.environ.get('PYTHONMALLOC'),
            repeats_on_success=False, original_gate_result_unchanged=True))
    return actual


def witnessed_restore(payload, expected_model_hashes):
    active['payload'] = payload
    try:
        return original_restore(payload, expected_model_hashes)
    except BaseException as error:
        state = payload['state']
        options = dict(sort_keys=True, allow_nan=False, separators=(',', ':'))
        independent = hashlib.sha256(json.dumps(state, **options).encode()).hexdigest()
        blocks = [dict(index=i, length=len(values),
                       sha256=hashlib.sha256(json.dumps(values, **options).encode()).hexdigest())
                  for i, values in enumerate(state['ordered_state_arrays'])]
        save('restore_failure_witness.json', dict(
            error_type=type(error).__name__, error=str(error),
            original_first_hash=active.get('original_first_hash'), expected=payload['payload_hash'],
            repeated_original_hash=original_hash(state), independent_stdlib_hash=independent,
            checkpoint_version=payload['checkpoint_version'], blocks=blocks,
            payload_object_id=id(payload), state_object_id=id(state),
            checks_relaxed=False, scientific_state_changed=False))
        with (OUT / 'restore_failed_dynamic_DIAGNOSTIC_ONLY.json').open('w', encoding='utf8') as stream:
            json.dump(payload, stream, **options)
        raise
    finally:
        active.clear()


engine.restore_payload = witnessed_restore
engine.canonical_hash = witnessed_hash
try:
    sys.argv[0] = str(ROOT / 'tools/run_video_05s.py')
    runpy.run_module('tools.run_video_05s', run_name='__main__')
except BaseException as error:
    capture_terminal_error(error)
    raise
finally:
    engine.restore_payload = original_restore
    engine.canonical_hash = original_hash
    restore_json_reference()
    restore_geometry_reference()
