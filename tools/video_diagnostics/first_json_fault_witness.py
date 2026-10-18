"""Transparent first-error observer; saves evidence without JSON serialization."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import os
import pickle
import pprint
import sys
import traceback


def install(directory):
    import retina_sim.canonical_stream as canonical
    folder = Path(directory)
    folder.mkdir(exist_ok=True)
    original_module = canonical.json
    original_dumps = original_module.dumps
    captured = False

    def persist(name, value):
        target = folder/(name+'.pickle')
        temporary = target.with_suffix('.writing')
        with temporary.open('wb') as stream:
            pickle._Pickler(stream, protocol=4).dump(value)
        os.replace(temporary, target)
        (folder/(name+'.txt')).write_text(pprint.pformat(value, sort_dicts=True)+'\n',
                                         encoding='utf8', errors='backslashreplace')

    def terminal(error):
        persist('terminal_exception', dict(error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), first_canonical_JSON_error_captured=captured,
            evidence_serializer='pure Python pickle._Pickler and pprint; no JSON'))

    def observe(value, *args, **options):
        nonlocal captured
        try:
            return original_dumps(value, *args, **options)
        except BaseException as error:
            if not captured:
                captured = True
                try:
                    # Exact failing input first; no retry, normalization, or alternate encoder.
                    persist('first_JSON_input', dict(value=value, args=args, options=options))
                    frames = []
                    frame = sys._getframe(1)
                    while frame is not None:
                        item = dict(file=frame.f_code.co_filename, function=frame.f_code.co_name,
                                    line=frame.f_lineno)
                        if Path(frame.f_code.co_filename).name == 'canonical_stream.py':
                            for key in ('key', 'start', 'index'):
                                obj = frame.f_locals.get(key)
                                if type(obj) in (int, str):
                                    item[key] = obj
                            for key in ('value', 'values', 'block', 'array', 'row'):
                                obj = frame.f_locals.get(key)
                                if obj is not None:
                                    item[key] = dict(object_type=type(obj).__module__+'.'+type(obj).__qualname__,
                                                     object_id=id(obj))
                                    if isinstance(obj, (list, tuple, dict, str)):
                                        item[key]['length'] = len(obj)
                        frames.append(item)
                        frame = frame.f_back
                    modules = {name: vars(module).get('__file__')
                               for name, module in list(sys.modules.items()) if module is not None}
                    persist('first_JSON_metadata', dict(error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc(), object_type=type(value).__module__+'.'+type(value).__qualname__,
                        object_id=id(value), encoding_options=options,
                        element_types=dict(Counter(type(v).__name__ for v in value))
                                      if isinstance(value, (list, tuple)) else None,
                        container_frames_inner_to_outer=frames, loaded_Python_modules=modules,
                        original_call_result_or_exception_retained=True, retry_count=0,
                        evidence_serializer='pure Python pickle._Pickler and pprint; no JSON'))
                except BaseException:
                    (folder/'capture_secondary_failure.txt').write_text(traceback.format_exc(),
                        encoding='utf8', errors='backslashreplace')
            raise

    canonical.json = SimpleNamespace(dumps=observe)

    def cleanup():
        canonical.json = original_module

    return cleanup, terminal
