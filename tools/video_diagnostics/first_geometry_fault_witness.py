"""Observe original geometry gate results without replacing any decision."""
from pathlib import Path
import os
import pickle
import pprint
import traceback


def install(directory):
    import retina_sim.geometry_identity as geometry
    from retina_sim.disk_graph_storage import array_identity
    folder = Path(directory)
    folder.mkdir(exist_ok=True)
    original = geometry.require_geometry_identity
    sequence = 0

    def save(value):
        nonlocal sequence
        sequence += 1
        path = folder / ('geometry_gate_%03d.pickle' % sequence)
        temporary = path.with_suffix('.writing')
        with temporary.open('wb') as stream:
            pickle._Pickler(stream, protocol=4).dump(value)
        os.replace(temporary, path)
        path.with_suffix('.txt').write_text(pprint.pformat(value, sort_dicts=True) + '\n',
                                            encoding='utf8', errors='backslashreplace')

    def snapshot(declared, expected, compiled, error=None):
        matrices = {}
        for cid, component in compiled.components.items():
            for name in ('A', 'B'):
                matrix = getattr(component.native_model, name, None)
                if hasattr(matrix, 'indptr'):
                    matrices[cid + '.' + name] = {key: array_identity(getattr(matrix, key))
                                                  for key in ('data', 'indices', 'indptr')}
        differences = None
        if isinstance(declared, dict) and isinstance(expected, dict):
            differences = {key: {'declared': declared.get('fields', {}).get(key),
                                 'original_expected': expected.get('fields', {}).get(key)}
                           for key in set(declared.get('fields', {})) | set(expected.get('fields', {}))
                           if declared.get('fields', {}).get(key) != expected.get('fields', {}).get(key)}
        save({'original_gate_passed': error is None, 'declared': declared,
              'expected_original_first_result': expected, 'field_differences': differences,
              'native_matrix_bytes': matrices, 'assembly_hash': compiled.assembly_hash,
              'error': str(error) if error else None, 'stack': traceback.format_stack(),
              'original_decision_unchanged': True, 'repeated_computation_used_for_gate': False})

    def witnessed(declared, compiled):
        try:
            result = original(declared, compiled)
        except BaseException as error:
            expected = None
            frame = error.__traceback__
            while frame is not None:
                if frame.tb_frame.f_code is original.__code__:
                    expected = frame.tb_frame.f_locals.get('expected')
                    break
                frame = frame.tb_next
            try:
                snapshot(declared, expected, compiled, error)
            except BaseException:
                (folder / 'geometry_capture_secondary_failure.txt').write_text(traceback.format_exc(), encoding='utf8')
            raise
        try:
            snapshot(declared, result, compiled)
        except BaseException:
            (folder / 'geometry_capture_secondary_failure.txt').write_text(traceback.format_exc(), encoding='utf8')
        return result

    geometry.require_geometry_identity = witnessed

    def cleanup():
        geometry.require_geometry_identity = original

    return cleanup
