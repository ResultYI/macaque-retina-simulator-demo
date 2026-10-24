"""Explicit, bounded two-arm QA orchestration. NEVER invoked on import.

UNVALIDATED implementation. This file is not a permission to run it. The caller
must first obtain explicit release of the gaming restriction. No CLI, scheduler,
graph assembly, hidden warmup, threshold tuning or performance run is installed.
Reference evidence is spooled to bounded files, candidate evidence is compared
one callback at a time. QA readback/tracing/I/O cannot be used as speed evidence.
"""
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import ast
import hashlib
import json
import math
import shutil
import sys
import time


class QABudgetStop(RuntimeError):
    pass


# The reference repair and resident dispatch are written and rebound, but no
# regression has run. This is an execution hold, not a claim the old bugs remain
# unfixed. Remove it only after separately authorized, reviewed regression
# evidence for these exact sources; there is deliberately no bypass argument.
REFERENCE_REPAIR_STATUS = 'PATCH_WRITTEN_STATIC_ONLY_REGRESSION_PENDING'
REFERENCE_REGRESSION_PENDING = (
    'tests/test_movie_zoh_boundaries.py: repaired frame and CPU/CUDA adapter sides',
    'tests/test_resident_input_sides.py: resident sides, pins and fail-closed gate',
)
EXCLUDED_CACHE_PATHS = (
    'BoundaryReplay schema 1.2/content-bound repair remains outside resident QA; '
    'no old 1.1 cache or replay checkpoint migration is authorized here',
)


def _authorize(execution_authorized, restriction_release_evidence):
    if execution_authorized is not True or not isinstance(restriction_release_evidence, str) or not restriction_release_evidence.strip():
        raise PermissionError('Explicit later user authorization releasing the gaming restriction is required')


def _host(value):
    import numpy as np
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value, dtype=np.float64)


def hash_prepared_sources(sources):
    """Hash actual supplied channel source files/arrays, never filenames alone."""
    import numpy as np
    result = {}
    for name, entries in sources.items():
        digest = hashlib.sha256()
        if not entries:
            raise ValueError('Every channel requires its actual prepared content')
        for entry in entries:
            if isinstance(entry, (str, Path)):
                digest.update(b'file\0')
                with Path(entry).open('rb') as stream:
                    while True:
                        part = stream.read(1024*1024)
                        if not part:
                            break
                        digest.update(part)
            elif isinstance(entry, np.ndarray) and not entry.dtype.hasobject:
                digest.update((str(entry.dtype)+str(entry.shape)).encode())
                for part in np.nditer(entry, flags=['external_loop', 'buffered', 'zerosize_ok'],
                                      order='C', buffersize=65536):
                    digest.update(part.tobytes())
            else:
                raise ValueError('Prepared sources must be actual numeric arrays or readable files')
            digest.update(b'\0end\0')
        result[name] = digest.hexdigest()
    return result


class WitnessSpool:
    """No pickle, no silent overwrite, hard total disk and elapsed-time caps."""
    def __init__(self, directory, *, max_bytes, max_seconds, guard=None):
        if max_bytes <= 0 or not 0 < max_seconds <= 840:
            raise ValueError('Positive spool budget and at most 840 s of supervised QA execution required')
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.limit, self.seconds, self.used = int(max_bytes), float(max_seconds), 0
        self.start, self.guard, self.serial = time.perf_counter(), guard, 0
        self.sequence_serial,self.sequence_cache,self.sequence_files=0,{},{}
        self.array_files={}
        self.readonly=False

    @classmethod
    def open_existing(cls,directory,*,guard=None,max_seconds=840):
        result=cls.__new__(cls)
        result.root=Path(directory)
        if not result.root.is_dir():raise ValueError('Missing immutable reference evidence')
        result.limit,result.seconds,result.used=64*1024**3,max_seconds,0
        result.start,result.guard=time.perf_counter(),guard
        result.readonly=True
        return result

    def check(self, additional=0):
        if time.perf_counter()-self.start > self.seconds:
            raise QABudgetStop('QA wall budget reached')
        if self.used + additional > self.limit:
            raise QABudgetStop('QA spool byte budget reached')
        if shutil.disk_usage(self.root).free < additional + 1024**3:
            raise QABudgetStop('QA disk reserve reached')
        if self.guard is not None and self.guard() is False:
            raise QABudgetStop('Cooperative STOP/memory boundary guard requested stop')

    def write(self, key, witness):
        import numpy as np
        if self.readonly:raise ValueError('Archived reference evidence is read only')
        self.check()
        metadata_path = self.root / (key + '.json')
        if metadata_path.exists():
            raise ValueError('A QA evidence identifier cannot be overwritten')
        def encode(value):
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise ValueError('Object arrays are not numeric QA witnesses')
                # Content-address values only. Every occurrence is freshly
                # hashed, so a reused mutable buffer can never return old data.
                contiguous=np.ascontiguousarray(value)
                digest=hashlib.sha256(json.dumps([repr(value.dtype),value.shape]).encode())
                if contiguous.nbytes:digest.update(memoryview(contiguous).cast('B'))
                content=digest.hexdigest()
                if content in self.array_files:
                    return self.array_files[content]
                self.check(value.nbytes+4096)
                name = 'array_%09d.npy' % self.serial
                self.serial += 1
                path = self.root / name
                with path.open('xb') as stream:
                    np.save(stream, value, allow_pickle=False)
                size = path.stat().st_size
                self.used += size
                result={'__npy__': name, 'size': size, 'sha256': _file_hash(path)}
                self.array_files[content]=result
                return result
            if isinstance(value, dict):
                return {str(k): encode(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                if len(value)>1024 and all(type(v) is str for v in value):
                    # Lossless shared full cell orders, not sampled IDs. Only
                    # immutable tuples may reuse an identity-cached encoding.
                    cached=self.sequence_cache.get(id(value)) if isinstance(value,tuple) else None
                    if cached is not None and cached[0] is value:
                        return cached[1]
                    raw=json.dumps(value,separators=(',',':'),allow_nan=False).encode('utf8')
                    digest=hashlib.sha256(raw).hexdigest()
                    found=self.sequence_files.get(digest)
                    if found is None:
                        self.check(len(raw)+4096)
                        name='strings_%09d.json'%self.sequence_serial;self.sequence_serial+=1
                        with (self.root/name).open('xb') as stream:stream.write(raw)
                        self.used+=len(raw)
                        found=dict(__strings__=name,size=len(raw),sha256=digest)
                        self.sequence_files[digest]=found
                    result=dict(found,kind='tuple' if isinstance(value,tuple) else 'list')
                    if isinstance(value,tuple):self.sequence_cache[id(value)]=(value,result)
                    return result
                return [encode(v) for v in value]
            if isinstance(value, np.generic):
                return value.item()
            return value
        raw = json.dumps(encode(witness), sort_keys=True, allow_nan=False).encode('utf8')
        self.check(len(raw))
        with metadata_path.open('xb') as stream:
            stream.write(raw)
        self.used += len(raw)
        return key

    def save_checkpoint_bytes(self,key,serialized):
        """Keep the actual codec output for recovery; never rebuild from points."""
        if self.readonly or Path(key).name!=key:raise ValueError('Invalid checkpoint export target')
        self.check(len(serialized)*4+4096)
        path=self.root/(key+'.checkpoint.json')
        with path.open('xb') as stream:
            for start in range(0,len(serialized),1024**2):
                stream.write(serialized[start:start+1024**2].encode('utf8'))
        self.used+=path.stat().st_size
        self.write(key+'_receipt',dict(file=path.name,size=path.stat().st_size,sha256=_file_hash(path),
                                      actual_complete_codec_output=True,fit_eligible=False))
        return path

    @contextmanager
    def read(self, key):
        import numpy as np
        self.check()
        arrays = []
        array_cache={}
        sequences={}
        def decode(value):
            if isinstance(value,dict) and set(value)=={'__strings__','size','sha256','kind'}:
                name=value['__strings__'];kind=value['kind'];key=(name,kind,value['sha256'],value['size'])
                if Path(name).name!=name or kind not in ('tuple','list'):
                    raise ValueError('Invalid full-string-order reference')
                if key not in sequences:
                    path=self.root/name
                    if path.stat().st_size!=value['size'] or _file_hash(path)!=value['sha256']:
                        raise ValueError('QA full-string-order evidence changed')
                    items=json.loads(path.read_bytes())
                    if not isinstance(items,list) or any(type(v) is not str for v in items):
                        raise ValueError('Invalid full-string-order evidence')
                    sequences[key]=tuple(items) if kind=='tuple' else items
                return sequences[key]
            if isinstance(value, dict) and set(value) == {'__npy__', 'size', 'sha256'}:
                name = value['__npy__']
                if Path(name).name != name:
                    raise ValueError('Invalid spool reference')
                key=(name,value['sha256'],value['size'])
                if key in array_cache:return array_cache[key]
                path = self.root / name
                if path.stat().st_size != value['size'] or _file_hash(path) != value['sha256']:
                    raise ValueError('QA evidence file changed')
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                arrays.append(array)
                array_cache[key]=array
                return array
            if isinstance(value, dict):
                return {k: decode(v) for k, v in value.items()}
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value
        try:
            yield decode(json.loads((self.root/(key+'.json')).read_bytes()))
        finally:
            for array in arrays:
                if getattr(array, '_mmap', None) is not None:
                    array._mmap.close()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


@contextmanager
def load_checkpoint_point(spool,value,archived=None):
    if set(value)!={'__point_record__','sha256'}:
        yield value
        return
    key=value['__point_record__']
    reader=spool
    if not (spool.root/(key+'.json')).is_file() and archived is not None and key in archived.inventory:
        reader=archived
    root=reader.directory if reader is archived else reader.root
    if Path(key).name!=key or _file_hash(root/(key+'.json'))!=value['sha256']:
        raise ValueError('Checkpoint point metadata identity changed')
    with reader.read(key) as point:yield point


def _schedule(runner, snapshot, stimulus, samples, stop):
    """Forecast only original clock/breakpoint/dense callbacks, no model RHS."""
    import numpy as np
    from .contracts import canonical_hash
    from .resident_input_evidence import InputEvidenceHash
    input_hash=InputEvidenceHash()
    ticks = dict(snapshot.component_tick_indices)
    current, origin = float(snapshot.time_seconds), snapshot.grid_origin_seconds
    points, intervals, due_ticks = [], [], []
    breaks = iter(stimulus.breakpoints_seconds)
    next_break = next(breaks, stop)
    def point(t, phase):
        points.append(dict(id='point_%05d' % len(points), time_seconds=float(t), phase=phase,
                           context_hash=input_hash(stimulus.at(float(t)))))
    while current < stop-1e-14:
        grid = min(origin+ticks[cid]*p for cid, p in runner.periods.items())
        if abs(grid-current) < 1e-10:
            due = [cid for cid, p in runner.periods.items() if abs(origin+ticks[cid]*p-current) < 1e-12]
            due_ticks.append((current, due))
            for cid in due:
                ticks[cid] += 1
        point(current, 'before_interval')
        while next_break <= current+1e-14:
            next_break = next(breaks, stop)
        bounds = [float(stop), float(next_break), min(origin+ticks[cid]*p for cid, p in runner.periods.items())]
        for p in (runner.observation_dt, runner.delayed_step):
            if p:
                bounds.append(origin+(int(np.floor((current-origin)/p+1e-8))+1)*p)
        target = min(bounds)
        if target <= current:
            raise ValueError('Prospective clock schedule does not advance')
        requested = sorted(set(float(t) for t in samples if current < t <= target))
        dense = sorted(set(requested+[float(target)])) if any(t < target for t in requested) else []
        intervals.append(dict(id='interval_%05d' % len(intervals), start=current, stop=float(target), dense_times=dense))
        for t in dense or [target]:
            point(t, 'interval_sample')
        current = float(target)
    final = min(origin+ticks[cid]*p for cid, p in runner.periods.items())
    if abs(final-stop) < 1e-12:
        due_ticks.append((float(stop), [cid for cid, p in runner.periods.items()
                                       if abs(origin+ticks[cid]*p-stop) < 1e-12]))
    point(stop, 'final')
    return points, intervals, due_ticks


def _expression_identity(path):
    """Bind actual arithmetic ASTs, not a caller-supplied equality assertion."""
    names = {'h0', 'h1', 'h_abs', 'd0', 'd1', 'd2', 'y_new', 'f_new', 'scale',
             'err5', 'err3', 'norm5', 'norm3', 'error', 'factor', 'delta'}
    nodes = []
    for node in ast.walk(ast.parse(Path(path).read_text(encoding='utf8'))):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append((node.lineno, node.col_offset, ast.dump(node, include_attributes=False)))
    return hashlib.sha256(json.dumps([value for _, _, value in sorted(nodes)]).encode()).hexdigest()


class OperationProbe:
    """Read actual locals at original solver decisions/monitor/RNG returns.

    There is no RNG replay and no replacement equation. Only this thread is
    traced, and a pre-existing tracer is rejected instead of overwritten.
    """
    def __init__(self, runner, monitor, on_solver, on_draw):
        from . import gpu_dop853, resident_dop853
        from .background import _WindowMonitor
        from .resident_monitor import ResidentWindowMonitor
        from .gpu_compact import CompactNativeInnerPartition
        from .resident_input_evidence import InputEvidenceHash
        self.input_hash=InputEvidenceHash()
        self.runner, self.monitor = runner, monitor
        self.on_solver, self.on_draw = on_solver, on_draw
        self.frames, self.monitor_locals = {}, None
        self.partition_code = CompactNativeInnerPartition.__call__.__code__
        self.draw_count, self.draw_order = 0, []
        self.delay_row_cache={}
        self.solvers = {}
        for fun in (gpu_dop853.integrate, resident_dop853.integrate_device):
            path = Path(fun.__code__.co_filename)
            choices = [n.lineno for n in ast.walk(ast.parse(path.read_text(encoding='utf8')))
                       if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and
                       isinstance(n.test.left, ast.Name) and n.test.left.id == 'error' and
                       len(n.test.ops) == 1 and isinstance(n.test.ops[0], ast.Lt) and
                       len(n.test.comparators) == 1 and isinstance(n.test.comparators[0], ast.Constant) and
                       n.test.comparators[0].value == 1]
            if len(choices) != 1:
                raise ValueError('Unreviewed DOP853 decision instrumentation site')
            self.solvers[fun.__code__] = (choices[0], _expression_identity(path))
        self.monitors = {_WindowMonitor.observe.__code__, ResidentWindowMonitor.observe.__code__}
        self.rgc = {id(c.native_model): cid for cid, c in runner.compiled.components.items()
                    if hasattr(c.native_model, 'sample_spikes')}
        self.spike_codes = {c.native_model.sample_spikes.__func__.__code__
                           for c in runner.compiled.components.values() if hasattr(c.native_model, 'sample_spikes')}

    def trace(self, frame, event, arg):
        from .contracts import canonical_hash
        code, local = frame.f_code, frame.f_locals
        if code in self.solvers:
            if event == 'call':
                self.frames[id(frame)] = dict(attempts=[], rhs_inputs=[], initial_rhs_calls=None)
            if event == 'line' and frame.f_lineno == self.solvers[code][0]:
                record = self.frames[id(frame)]
                if not record['attempts']:
                    record['initial_rhs_calls'] = len(record['rhs_inputs'])-12
                record['attempts'].append(dict(t=float(local['t']), h=float(local['h']),
                    t_new=float(local['t_new']), accepted=bool(local['error'] < 1)))
            if event == 'return':
                record = self.frames.pop(id(frame), {})
                attempts = record.get('attempts', [])
                if arg is not None:
                    targets = [float(t) for t in local['sample_times']]
                    coefficients = {k: _host(local[k]) for k in ('A', 'B', 'E3', 'E5', 'AE', 'DD', 'at')}
                    nfev = int(arg['nfev'])
                    self.on_solver(dict(start=float(local['t0']), stop=float(local['t_bound']),
                        dense_times=targets, initial_rhs_calls=record['initial_rhs_calls'],
                        rhs_inputs=record['rhs_inputs'], full_state_dimension=int(local['n']),
                        attempts=attempts, accepted_steps=int(arg['steps']),
                        rejected_attempts=sum(not a['accepted'] for a in attempts), nfev=nfev,
                        dense_polynomials=(nfev-2-12*len(attempts))//3,
                        endpoint_choice='dense_sample' if targets else 'accepted_endpoint',
                        controller_restart=True, coefficient_hash=canonical_hash(coefficients),
                        expression_hash=self.solvers[code][1]))
            return self.trace
        if code is self.partition_code:
            if event == 'call' and self.frames:
                if len(self.frames) != 1:
                    raise RuntimeError('Nested solvers are outside this QA probe scope')
                record = next(iter(self.frames.values()))
                record['rhs_inputs'].append(dict(time_seconds=float(local['t']),
                    input_hash=self.input_hash(local['bundle']),
                    delay_left_limit=bool(local['bundle'].delay_left_limit)))
            return None
        if code in self.spike_codes:
            if event == 'return' and arg is not None:
                native = local['self']
                cid = self.rgc[id(native)]
                updated, events = arg
                event_map = {e['cell_index']: e['time_seconds'] for e in events}
                for i, cell in enumerate(native.state_specs[0].cell_ids):
                    self.on_draw(dict(component=cid, cell_id=cell, bin_start_seconds=float(local['t']),
                        dt=float(native.dt), voltage_state=cid+'.'+native.state_specs[0].state_name,
                        cell_index=i, alpha=float(native.alpha[i]), beta=float(native.beta[i]),
                        u=float(local['u'][i]), rate_Hz=float(local['rate'][i]),
                        probability=float(local['p'][i]), event_decision=bool(local['counts'][i]),
                        event_time_seconds=event_map.get(i), history_hash=canonical_hash(local['history']),
                        rng_before_hash=canonical_hash(local['aux']['rng_state']),
                        rng_after_hash=canonical_hash(updated['rng_state']), draw_index=self.draw_count))
                    self.draw_order.append([cid, cell])
                    self.draw_count += 1
            return self.trace
        if code in self.monitors:
            if event == 'return' and arg is None and local.get('self') is self.monitor and 'transitions' in local:
                self.monitor_locals = dict(local)
            return self.trace
        return None

    def __enter__(self):
        if sys.gettrace() is not None:
            raise RuntimeError('QA requires an exclusive Python trace context')
        sys.settrace(self.trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(None)


def _monitor_for(runner, reference_state, policy):
    from .background import _WindowMonitor
    from .resident_runner import ResidentNativeRunner
    return (runner.make_monitor(reference_state, policy) if isinstance(runner, ResidentNativeRunner)
            else _WindowMonitor(runner, reference_state, policy))


def _point_witness(runner, current, monitor, probe, expected):
    import numpy as np
    from .background import _history_consistency
    from .contracts import canonical_hash
    from .resident_monitor import resident_history_consistency
    from .resident_state import ResidentStateHandle
    local = probe.monitor_locals
    if not local or float(local['t']) != expected['time_seconds']:
        raise ValueError('Actual full monitor return was not captured at the expected point')
    resident = isinstance(current, ResidentStateHandle)
    y = _host(local['y'])
    if resident:
        rhs = np.concatenate((_host(local['native_rhs']), local['inner_rhs']))
        outputs = {cid+'.'+port: q.values for (cid, port), q in local['outputs'].items()}
        outputs.update({runner.partition.cid+'.'+name: _host(value)
                        for name, value in local['evidence'].native_outputs.items()})
    else:
        rhs = local['rhs']
        outputs = {cid+'.'+port: q.values for (cid, port), q in local['outputs'].items()}
    states, derivatives, transitions = {}, {}, {}
    for spec, sl in zip(runner.model.state_specs, runner.model._state_slices):
        states[spec.state_name], derivatives[spec.state_name] = y[sl], rhs[sl]
        cid = spec.state_name.split('.', 1)[0]
        if cid in local['transitions']:
            start = runner.model.slices[cid].start
            transitions[spec.state_name] = local['transitions'][cid][sl.start-start:sl.stop-start]
    declarations = {cid+'.'+port for cid, c in runner.compiled.components.items() for port in c.outputs}
    if set(outputs) != declarations:
        raise ValueError('Full original declared port coverage differs')
    structures, delays,normalized_queues = {}, {}, {}
    for cid, queue in current.delay_states.items():
        if resident:
            structure = dict(times=list(queue.history_times), delay_seconds=queue.delay_seconds,
                             interpolation=queue.interpolation, prehistory_stop_seconds=queue.prehistory_stop_seconds,
                             unit=queue.unit, meaning=queue.meaning, evidence_tag=queue.evidence_tag)
            rows = [_host(queue._prehistory)] + [_host(row) for row in queue._values]
        else:
            pre = queue['prehistory']
            structure = dict(times=list(queue['history_times']), delay_seconds=queue['delay_seconds'],
                             interpolation=queue['interpolation'], prehistory_stop_seconds=queue['prehistory_stop_seconds'],
                             unit=pre['unit'], meaning=pre['meaning'], evidence_tag=pre['evidence_tag'])
            rows = _history_rows(queue,probe.delay_row_cache.setdefault(cid,{}))
        structures[cid] = structure
        delays[cid] = dict(port=cid+'.Vh', values=np.stack(rows))
        if not resident:
            normalized_queues[cid]=dict(queue,history_values=delays[cid]['values'][1:],
                prehistory=dict(queue['prehistory'],values=delays[cid]['values'][0]))
    pending = {cid: [{k: e[k] for k in ('cell_index', 'bin_start_seconds', 'bin_stop_seconds')}
                     for e in aux['pending_events']] for cid, aux in current.spike_states.items()}
    # Execution-only backend/binding IDs are checked separately, not made equal
    # by discarding numerical/RNG/clock payloads. These generation witnesses use
    # physical phase/clock/queue identity shared by both execution architectures.
    phase = {k: v for k, v in current.input_position.items() if not k.startswith('resident_')}
    aux = dict(grid_origin_seconds=current.grid_origin_seconds,
        component_tick_indices=dict(current.component_tick_indices), sample_index=current.sample_index,
        history={cid: value['history'] for cid, value in current.spike_states.items()},
        last_bin_counts={cid: value['last_bin_counts'] for cid, value in current.spike_states.items()},
        rng_state={cid: value['rng_state'] for cid, value in current.spike_states.items()},
        draw_order=list(probe.draw_order), draw_count=probe.draw_count, delay_structure=structures,
        pending_event_identity=pending, input_phase=phase,
        event_order='snapshot;all_transitions;commit;tick;RGC_sample;sample_index',
        generations=dict(state_time=current.time_seconds, ticks=dict(current.component_tick_indices),
                         queues={cid: item['times'] for cid, item in structures.items()}))
    if resident:
        histories=resident_history_consistency(current,runner,monitor.policy)
    else:
        # Run the unchanged original predicates over exactly the already
        # materialized full rows, rather than decode every tuple twice.
        view=copy(current);view.delay_states=normalized_queues
        histories=_history_consistency(view,runner,monitor.policy)
    checks = dict(state_domain=sorted(set(monitor.domain_failures)),
        port_validation=sorted(declarations),
        gap_decisions=[x for x in monitor.domain_failures if x == 'reciprocal_gap_current_conservation'],
        history_decisions=histories, failure_labels=sorted(set(monitor.domain_failures)),
        coverage=[row['spec'].state_name for row in monitor.rows],
        failure_barrier=dict(kind='original_monitor_return', time_seconds=float(local['t'])))
    return dict(expected, context_hash=probe.input_hash(local['bundle']), state=states, rhs=derivatives,
                ports=outputs, transitions=transitions, delays=delays, aux=aux, checks=checks)


def _history_rows(queue,cache):
    """Reuse only immutable numeric tuples, pruning all departed rows each call."""
    import numpy as np
    live={};rows=[np.asarray(queue['prehistory']['values'],dtype=np.float64)]
    for row in queue['history_values']:
        old=cache.get(id(row))
        if old is not None and old[0] is row:
            value=old[1];live[id(row)]=old
        else:
            value=np.asarray(row,dtype=np.float64)
            if isinstance(row,tuple) and all(type(x) in (int,float,np.float64) for x in row):
                value.setflags(write=False);live[id(row)]=(row,value)
        rows.append(value)
    cache.clear();cache.update(live)
    return rows


def _monitor_witness(monitor, expected):
    return dict(expected, raw_report=monitor.report(expected['width']), rows=[dict(
        state=row['spec'].state_name, **{key: _host(row[key]) for key in ('den', 'lo', 'hi', 'rhs', 'defect')})
        for row in monitor.rows])


def _build_plan(reference, candidate, snapshot, reference_state, channels, policy, samples, stop,
                background_stop, frame_boundaries):
    from .background import NUMERICAL_UNIT_SCALES
    from .network import NetworkStimulus
    from .resident_qa import seal_plan, CONTRACT_SHA256
    from .resident_derived_ports import ribbon_descriptors, ADDENDUM_SHA256, BUDGET_KIND
    derived = ribbon_descriptors(reference.compiled)
    stimulus = NetworkStimulus(reference.compiled, channels, snapshot, stop,
                               snapshot.solver_config.input_interpolation, reference.model.evidence_tag)
    points, intervals, tick_schedule = _schedule(reference, snapshot, stimulus, samples, stop)
    flat = reference._flat(snapshot)
    _, signals = reference.model.evaluate(snapshot.time_seconds, flat, stimulus.at(snapshot.time_seconds))
    states = []
    for spec, initial, monitor_initial in zip(reference.model.state_specs,
                                              snapshot.ordered_state_arrays, reference_state.ordered_state_arrays):
        cid = spec.state_name.split('.', 1)[0]
        component = reference.compiled.components[cid]
        states.append(dict(name=spec.state_name, unit=spec.unit, meaning=spec.meaning,
            cell_ids=tuple(spec.cell_ids), initial=_host(initial), monitor_initial=_host(monitor_initial),
            period=reference.periods[cid] if component.time_contract.kind in ('causal_history', 'author_discrete') else None))
    ports = []
    for cid, c in reference.compiled.components.items():
        for name, port in c.outputs.items():
            row=dict(name=cid+'.'+name, unit=port.unit, meaning=port.meaning,
                     cell_ids=tuple(port.cell_ids), initial=signals[(cid, name)].values)
            if row['name'] in derived:
                row.update(budget_kind=BUDGET_KIND,derived_descriptor=derived[row['name']])
            ports.append(row)
    draws, cells = [], []
    for cid in snapshot.spike_states:
        cells.extend((cid, cell) for cell in reference.compiled.components[cid].state_specs[0].cell_ids)
    for t, due in tick_schedule:
        for cid in snapshot.spike_states:
            if cid not in due:
                continue
            native = reference.compiled.components[cid].native_model
            for i, cell in enumerate(native.state_specs[0].cell_ids):
                draws.append(dict(id='draw_%05d' % len(draws), component=cid, cell_id=cell,
                    bin_start_seconds=float(t), dt=float(native.dt), voltage_state=cid+'.'+native.state_specs[0].state_name,
                    cell_index=i, alpha=float(native.alpha[i]), beta=float(native.beta[i])))
    return seal_plan(dict(contract_hash=CONTRACT_SHA256, binding=candidate.binding,
        derived_port_addendum_sha256=ADDENDUM_SHA256,
        qa_implementation_sha256={name: _file_hash(Path(__file__).with_name(name))
                                  for name in ('resident_qa.py', 'resident_qa_harness.py',
                                               'resident_derived_ports.py','resident_regression_evidence.py','resident_input_evidence.py')},
        arm_identity_hashes=dict(reference=reference.model.model_hash, candidate=candidate.model.model_hash),
        unit_scales=deepcopy(NUMERICAL_UNIT_SCALES), monitor_relative_tolerance=policy.relative_tolerance,
        states=states, ports=ports, points=points, intervals=intervals, draws=draws, rgc_cells=cells,
        monitors=[dict(id='full_window', width=stop-snapshot.time_seconds, point_ids=[p['id'] for p in points])],
        checkpoint_ids=['runtime_roundtrip'], ownership_ids=['dense_and_input'],
        failure_cases=[
            dict(id='nan_state', label='nonfinite_state', barrier='before_original_rhs'),
            dict(id='infinite_state', label='nonfinite_state', barrier='before_original_rhs'),
            dict(id='negative_light', label='negative_light', barrier='original_rhs_input_validation'),
            dict(id='missing_channel', label='channel_coverage', barrier='original_rhs_input_validation'),
            dict(id='wrong_unit', label='input_unit', barrier='original_rhs_input_validation')],
        scenario=dict(start=snapshot.time_seconds, stop=stop, background_stop=background_stop,
                      frame_boundaries=list(frame_boundaries), rod_period=.0001, rgc_period=.001,
                      fit_eligible=False)))


@contextmanager
def _track_state(runner, initial, spool, ownership):
    """Instance wrappers track the original functional fork, not a replacement loop."""
    from .contracts import canonical_hash
    current = [initial]
    originals = {name: getattr(runner, name) for name in ('_sample', '_integrate')}
    instance_values = {name: runner.__dict__.get(name) for name in originals}
    present = {name: name in runner.__dict__ for name in originals}
    def sample(state, stimulus):
        spool.check()
        current[0] = state
        return originals['_sample'](state, stimulus)
    def integrate(state, stimulus, stop, config, samples):
        spool.check()
        current[0] = state
        rows = originals['_integrate'](state, stimulus, stop, config, samples)
        if 'sample' not in ownership:
            ownership['sample'] = next(iter(rows.values()))
            ownership['hash'] = canonical_hash(_host(ownership['sample']))
        if len(rows) > 1 and 'dense_sample' not in ownership:
            ownership['dense_sample'] = next(iter(rows.values()))
            ownership['dense_hash'] = canonical_hash(_host(ownership['dense_sample']))
        return rows
    runner._sample, runner._integrate = sample, integrate
    try:
        yield current
    finally:
        for name in originals:
            if present[name]:
                setattr(runner, name, instance_values[name])
            else:
                delattr(runner, name)


def _one_point(runner, state, reference_state, policy, channels, stop, identifier):
    from .contracts import canonical_hash
    from .network import NetworkStimulus
    from .resident_state import ResidentStateHandle
    stimulus = NetworkStimulus(runner.compiled, channels, state, stop,
                               state.solver_config.input_interpolation, runner.model.evidence_tag)
    monitor = _monitor_for(runner, reference_state, policy)
    probe = OperationProbe(runner, monitor, lambda _: None, lambda _: None)
    bundle = stimulus.at(state.time_seconds)
    expected = dict(id=identifier, time_seconds=state.time_seconds, phase='checkpoint', context_hash=probe.input_hash(bundle))
    values = state.y if isinstance(state, ResidentStateHandle) else runner._flat(state)
    with probe:
        monitor.observe(state.time_seconds, values, bundle)
    return _point_witness(runner, state, monitor, probe, expected)


def _checkpoint_witness(runner, state, reference_state, policy, channels, continuation_stop, spool):
    from .contracts import canonical_hash
    from .recording import RecordingPlan
    before = runner.checkpoint(state)
    spool.save_checkpoint_bytes('runtime_export_'+runner.model.model_hash[:20],before)
    payload = json.loads(before)
    metadata=dict(payload_hash=payload['payload_hash'],verified_payload_hash=canonical_hash(payload['payload']),
                  payload_fields=list(payload['payload']),supplied_state_hash_before=canonical_hash(payload))
    del payload
    restored = runner.restore(before)
    del before
    prefix='checkpoint_point_'+runner.model.model_hash[:20]+'_'
    def store(value, label):
        key=prefix+label
        spool.write(key,value)
        return dict(__point_record__=key,sha256=_file_hash(spool.root/(key+'.json')))
    exported = store(_one_point(runner, state, reference_state, policy, channels, continuation_stop, 'checkpoint_export'),'export')
    restored_point = store(_one_point(runner, restored, reference_state, policy, channels, continuation_stop, 'checkpoint_export'),'restore')
    restored_payload_hash=canonical_hash(json.loads(runner.checkpoint(restored))['payload'])
    endings, schedules = [], []
    for initial in (state, restored):
        monitor = _monitor_for(runner, reference_state, policy)
        schedule, draws = [], []
        probe = OperationProbe(runner, monitor, schedule.append, draws.append)
        with _track_state(runner, initial, spool, {}), probe:
            ending, _ = runner.advance(initial, channels, continuation_stop,
                recording_plan=RecordingPlan(ports=(), states=(), cell_ids=()), state_observer=monitor.observe)
        schedules.append(dict(solver=schedule, draws=draws))
        endings.append(store(_one_point(runner, ending, reference_state, policy, channels,
                                  continuation_stop, 'checkpoint_continuation'),'ending_'+str(len(endings))))
        del ending, monitor, probe
    return dict(id='runtime_roundtrip', runtime_identity_hash=runner.model.model_hash,
        restored_identity_hash=restored.model_hash, **metadata,
        restored_payload_hash=restored_payload_hash,
        layout_hash=canonical_hash(runner.model.state_specs),
        fit_eligible=False,
        supplied_state_hash_after=canonical_hash(json.loads(runner.checkpoint(state))),
        export=exported, restored=restored_point, uninterrupted=endings[0], continued=endings[1],
        uninterrupted_schedule=schedules[0], continued_schedule=schedules[1])


def _failure_witness(runner, state, reference_state, policy, channels, stop, case):
    import numpy as np
    from .contracts import canonical_hash, Quantity
    from .network import NetworkStimulus
    from .resident_state import ResidentStateHandle
    monitor = _monitor_for(runner, reference_state, policy)
    stimulus = NetworkStimulus(runner.compiled, channels, state, stop,
                               state.solver_config.input_interpolation, runner.model.evidence_tag)
    values = state.y.clone() if isinstance(state, ResidentStateHandle) else runner._flat(state).copy()
    bundle = stimulus.at(state.time_seconds)
    nonfinite = case['id'] in ('nan_state', 'infinite_state')
    if nonfinite:
        values[0] = float('nan') if case['id'] == 'nan_state' else float('inf')
    else:
        altered = dict(bundle.channels)
        key = runner.partition.native_external.channel
        original = altered[key]
        if case['id'] == 'missing_channel':
            del altered[key]
        elif case['id'] == 'wrong_unit':
            altered[key] = Quantity(original.values.copy(), 'invalid_QA_unit', original.meaning, original.evidence_tag)
        elif case['id'] == 'negative_light':
            bad = original.values.copy()
            bad[0] = -1.
            altered[key] = Quantity(bad, original.unit, original.meaning, original.evidence_tag)
        else:
            raise ValueError('Unregistered prospective failure probe')
        bundle = replace(bundle, channels=altered)
    aux_before = canonical_hash(dict(spikes=state.spike_states, ticks=state.component_tick_indices,
                                     sample_index=state.sample_index, time=state.time_seconds))
    kind, rhs_calls = None, [0]
    watched = {runner.model.rhs_flat.__func__.__code__}
    if hasattr(runner, 'evaluator'):
        watched.add(runner.evaluator.__call__.__func__.__code__)
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code in watched:
            rhs_calls[0] += 1
    if sys.getprofile() is not None:
        raise RuntimeError('Failure-barrier probe needs an exclusive profile context')
    try:
        sys.setprofile(profile)
        monitor.observe(state.time_seconds, values, bundle)
    except Exception as exc:
        kind = type(exc).__name__
    finally:
        sys.setprofile(None)
    if kind is None:
        raise ValueError('Invalid QA input was not rejected at the original barrier')
    if rhs_calls[0] != (0 if nonfinite else 1) or monitor.count != 1:
        raise ValueError('Input rejection moved across the original RHS/observation barrier')
    aux_after = canonical_hash(dict(spikes=state.spike_states, ticks=state.component_tick_indices,
                                    sample_index=state.sample_index, time=state.time_seconds))
    return dict(case,
                time_seconds=state.time_seconds, stage='monitor', exception_type=kind,
                checks_before=dict(observation_counter=monitor.count, rhs_calls=rhs_calls[0]),
                post_failure_updates=0 if aux_before == aux_after else 1)


def _recording_witness(result):
    if result is None:
        raise ValueError('Ordinary recording QA requires nonempty explicitly selected outputs')
    return dict(outputs={key: dict(values=q.values, unit=q.unit, meaning=q.meaning, evidence_tag=q.evidence_tag)
                         for key, q in result.outputs_with_units.items()}, metadata=result.metadata)


def _compare_recording(comparator, reference, candidate):
    """Use already frozen typed budgets for selected trace columns and rates."""
    import numpy as np
    from .resident_qa import QAIncomplete
    a, b = reference['outputs'], candidate['outputs']
    if set(a) != set(b):
        comparator._issue('FAIL', 'recording', 'Recorded output keys differ')
        return
    ids_a, ids_b = reference['metadata']['recorded_cell_ids'], candidate['metadata']['recorded_cell_ids']
    comparator._exact(ids_a, ids_b, 'recording.cell_order')
    for key in a:
        x, y = a[key], b[key]
        for label in ('unit', 'meaning', 'evidence_tag'):
            comparator._exact(x[label], y[label], 'recording.'+key+'.'+label)
        if x['values'].shape != y['values'].shape:
            comparator._issue('FAIL', 'recording.'+key, 'Recorded array shape differs')
            continue
        if key == 'time' or key.endswith(('.spike_count', '.window_rate')):
            if not np.array_equal(x['values'], y['values']):
                comparator._issue('FAIL', 'recording.'+key, 'Original timestamps/count-derived values differ')
            continue
        name = key.removeprefix('state:').replace('.observed_Vh', '.Vh')
        spec = comparator.states.get(name) if key.startswith('state:') else comparator.ports.get(name)
        conditional = key.endswith('.conditional_rate')
        if conditional:
            cid = key.rsplit('.', 1)[0]
            records = [item for item in comparator.plan['draws'] if item['component'] == cid]
            if not records:
                raise QAIncomplete('Missing frozen RGC parameters for recorded conditional rates')
            spec = comparator.states[records[0]['voltage_state']]
        if spec is None:
            raise QAIncomplete('No frozen typed propagation for recorded output: '+key)
        if spec.get('budget_kind'):
            if ids_a[key]!=list(spec['cell_ids']):
                raise QAIncomplete('Derived recording requires complete ordered dependency cells')
            names=(spec['derived_descriptor']['voltage_state'],spec['derived_descriptor']['resource_state'])
            for name in names:
                if 'state:'+name not in a or 'state:'+name not in b:
                    raise QAIncomplete('Missing same-time recorded derived dependency: '+name)
                if ids_a['state:'+name]!=ids_a[key] or ids_b['state:'+name]!=ids_b[key]:
                    raise QAIncomplete('Recorded derived dependency cell order differs')
            for column in range(x['values'].shape[1]):
                dep_a={name:a['state:'+name]['values'][:,column] for name in names}
                dep_b={name:b['state:'+name]['values'][:,column] for name in names}
                comparator.compare_derived_port(x['values'][:,column],y['values'][:,column],spec,
                    'recording.'+key+'.'+str(column),(dep_a,dep_b))
            continue
        index = {cell: i for i, cell in enumerate(spec['cell_ids'])}
        indices = [index[cell] for cell in ids_a[key]]
        selected = dict(spec, initial=spec['initial'][indices], cell_ids=ids_a[key])
        if not conditional:
            for column in range(x['values'].shape[1]):
                comparator._numeric(x['values'][:, column], y['values'][:, column], selected,
                                    'recording.'+key+'.'+str(column))
        else:
            _, voltage_budget = comparator._budget(selected, slice(None))
            parameters = {r['cell_id']: r for r in records}
            alpha_beta = np.array([parameters[cell]['alpha']/parameters[cell]['beta'] for cell in ids_a[key]])
            eps = comparator.contract['roundoff_epsilon_multiplier']*sys.float_info.epsilon
            allowed = alpha_beta[:, None]*voltage_budget[:, None] + eps*np.maximum(np.abs(x['values']), 1.)
            if (not np.isfinite(x['values']).all() or not np.isfinite(y['values']).all() or
                    np.any(np.abs(y['values']-x['values']) > allowed)):
                comparator._issue('FAIL', 'recording.'+key, 'Conditional rate exceeds original propagated budget')
    # Execution identity is intentionally different; physical/clock metadata is
    # compared, with each recorded event tied to its actually observed draw.
    for key in ('event_count', 'event_window_seconds', 'clock_contract', 'history_interval_seconds',
                'observation_interval_seconds', 'sampling_index', 'global_state_layout'):
        comparator._exact(reference['metadata'][key], candidate['metadata'][key], 'recording.metadata.'+key)
    for arm in (reference, candidate):
        for cid, events in arm['metadata']['spike_events'].items():
            for event in events:
                if not any(d['component'] == cid and d['cell_id'] == event['cell_id'] and
                           d['bin_start_seconds'] == event['bin_start_seconds'] for d in comparator.plan['draws']):
                    comparator._issue('INCONCLUSIVE', 'recording.events', 'Recorded inherited event lacks a prospective original draw witness')


def run_resident_qa(reference, snapshot, reference_state, channels, policy, *, prepared_sources,
                    sample_times, stop, background_stop, frame_boundaries, recording_plan,
                    output_directory, max_spool_bytes, continuation_stop, on_boundary,
                    execution_authorized=False, restriction_release_evidence=None, max_qa_seconds=180.,
                    reference_regression_receipt=None, context_sink=None, reuse_reference_main=None,
                    reuse_reference_complete=False,candidate_native_sparse_plan=False):
    """Run exactly one prospectively sealed serial correctness comparison.

    The reference is already assembled; this function never assembles, installs,
    warms up, fits, changes STOP, retries or benchmarks. `on_boundary` must check
    the caller's existing STOP/memory guard. `prepared_sources` contains actual
    source files/arrays for each channel. No result grants production acceptance.
    Pending repaired-reference regressions block this entry before numerical
    execution, even after a user releases the separate gaming restriction.
    """
    _authorize(execution_authorized, restriction_release_evidence)
    if reference_regression_receipt is None:
        return dict(status='BLOCKED_REFERENCE_REGRESSION_PENDING',
                    reference_repair_status=REFERENCE_REPAIR_STATUS,
                    blockers=list(REFERENCE_REGRESSION_PENDING),
                    excluded_cache_paths=list(EXCLUDED_CACHE_PATHS),
                    numerical_execution=False, performance_validated=False, fit_eligible=False,
                    action='Later user authorization and reviewed corrected-source regression evidence are required; old receipts do not qualify')
    from .resident_regression_evidence import load_reference_regression_receipt
    regression_evidence=load_reference_regression_receipt(receipt_path=reference_regression_receipt)
    import numpy as np
    from .contracts import canonical_hash
    from .engine import validate_state
    from .resident_runner import ResidentNativeRunner
    from .resident_qa import ResidentQAComparator, QAIncomplete, require_registered_observable_scales
    from .background import NUMERICAL_UNIT_SCALES
    from .resident_derived_ports import ribbon_descriptors
    derived=ribbon_descriptors(reference.compiled)
    require_registered_observable_scales(
        reference.model.state_specs, reference.compiled.components, NUMERICAL_UNIT_SCALES,derived)
    if not callable(on_boundary):
        raise ValueError('An explicit original STOP/memory boundary guard is required')
    if (not np.isfinite(continuation_stop) or not 0 < continuation_stop-stop <= .0001+1e-12 or
            any(channel.stop_seconds < continuation_stop for channel in channels.values())):
        raise ValueError('Prospectively covered checkpoint continuation of at most 100 us is required')
    if set(prepared_sources) != set(channels):
        raise ValueError('Every actual channel must have a prepared-content source binding')
    spool = WitnessSpool(output_directory, max_bytes=max_spool_bytes,
                         max_seconds=max_qa_seconds, guard=on_boundary)
    comparison = None
    try:
        validate_state(snapshot.dynamic, reference.model)
        validate_state(reference_state.dynamic, reference.model)
        sources_hash = hash_prepared_sources(prepared_sources)
        # The original runner is shallow-copied only to isolate instance hooks
        # and metrics. Source equations/compiled constants remain unchanged.
        ref = copy(reference)
        ref.integration_metrics = dict(reference.integration_metrics)
        if candidate_native_sparse_plan:
            from .resident_cusparse import make_candidate
            candidate=make_candidate(reference)
        else:
            candidate = ResidentNativeRunner.from_runner(reference, purpose='numerical_qa')
        initial = candidate.import_reference_state(snapshot, reference_state=reference_state,
            channels=channels, channel_data_hashes=sources_hash, monitor_policy=policy)
        samples = np.asarray(sample_times, dtype=float)
        if (samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all() or
                np.any(np.diff(samples) <= 0) or samples[0] < snapshot.time_seconds or samples[-1] > stop):
            raise ValueError('Original increasing sample times must be prospectively declared')
        plan = _build_plan(ref, candidate, snapshot, reference_state, channels, policy, samples,
                           stop, background_stop, frame_boundaries)
        plan['checkpoint_continuation_stop'] = continuation_stop
        plan['reference_regression_receipt_sha256']=regression_evidence['receipt_sha256']
        from .resident_recording_selection import selection_identity
        plan['recording_selection']=selection_identity(recording_plan)
        plan['qa_implementation_sha256']['resident_recording_selection.py']=_file_hash(
            Path(__file__).with_name('resident_recording_selection.py'))
        if candidate_native_sparse_plan:
            plan['candidate_native_sparse_plan']=candidate.native_sparse_plan_policy
            plan['qa_implementation_sha256']['resident_cusparse.py']=_file_hash(Path(__file__).with_name('resident_cusparse.py'))
        archived=None
        if reuse_reference_main is not None:
            from .resident_reference_reuse import ReferenceMainEvidence
            archived=ReferenceMainEvidence(reuse_reference_main,plan,on_boundary,complete_arm=reuse_reference_complete,
                                           candidate_revision=candidate_native_sparse_plan)
            plan['reference_main_reuse']=archived.provenance
            plan['qa_implementation_sha256']['resident_reference_reuse.py']=_file_hash(
                Path(__file__).with_name('resident_reference_reuse.py'))
        plan['qa_implementation_sha256']['canonical_stream.py']=_file_hash(Path(__file__).with_name('canonical_stream.py'))
        # Re-seal the complete final prospective metadata, before either advance.
        from .resident_qa import seal_plan
        plan = seal_plan(plan)
        spool.write('prospective_plan', plan)
        if candidate_native_sparse_plan:
            from .resident_cusparse import check_repeatability
            spool.write('native_sparse_repeatability',check_repeatability(candidate,initial,channels))
        evidence_number = [0]
        def sink(record):
            key = 'comparison_%07d' % evidence_number[0]
            evidence_number[0] += 1
            spool.write(key, record)
        comparison = ResidentQAComparator(plan, evidence_sink=sink)
        comparison.point_loader=lambda value:load_checkpoint_point(spool,value,archived)
        for arm, runner, start_state in (('reference', ref, snapshot), ('candidate', candidate, initial)):
            print('QA arm '+arm+(' (immutable main reference refresh)' if archived and arm=='reference' else ''),flush=True)
            if archived and archived.complete_arm and arm=='reference':
                recovered=archived.restore_reference(runner)
                exported=_one_point(runner,recovered,reference_state,policy,channels,continuation_stop,'checkpoint_export')
                archived.assert_exact(archived.export_key,exported)
                del recovered,exported
                print('Authentic complete reference arm restored and exact export point verified',flush=True)
                continue
            monitor = _monitor_for(runner, reference_state, policy)
            position = dict(points=0, intervals=0, draws=0)
            ownership = {}
            supplied_before = canonical_hash(json.loads(runner.checkpoint(start_state)))
            def pair(category, value, method):
                spool.check()
                key = category+'_'+value['id']
                if arm == 'reference':
                    if archived and key in archived.keys:
                        if category=='point':spool.write('refreshed_reference_final',value)
                        archived.assert_exact(key,value)
                    else:spool.write(key, value)
                else:
                    reader=archived if archived and key in archived.keys else spool
                    with reader.read(key) as old:
                        method(old, value)
                    if comparison.worst_status != 'PASS':
                        spool.write('failed_candidate_'+key,value)
                        raise QAIncomplete('First numerical/coverage/uncertainty gate stopped evaluation')
            def solver(value):
                i = position['intervals']
                if i >= len(plan['intervals']):
                    raise QAIncomplete('Unexpected additional solver interval')
                value['id'] = plan['intervals'][i]['id']
                position['intervals'] += 1
                pair('solver', value, comparison.compare_solver)
            def draw(value):
                i = position['draws']
                if i >= len(plan['draws']):
                    raise QAIncomplete('Unexpected additional RNG draw')
                value['id'] = plan['draws'][i]['id']
                position['draws'] += 1
                pair('draw', value, comparison.compare_draw)
            probe = OperationProbe(runner, monitor, solver, draw)
            if archived and arm=='reference':
                # Main solver trace is the preserved genuine original trace.
                # Refresh only materializes a genuine endpoint for new codec
                # tests; every real RNG draw and the final point must be exact.
                probe.solvers={}
                position['intervals']=len(plan['intervals'])
            with _track_state(runner, start_state, spool, ownership) as current, probe:
                def observe(t, y, bundle):
                    spool.check()
                    i = position['points']
                    if i >= len(plan['points']):
                        raise QAIncomplete('Unexpected additional original callback')
                    expected = plan['points'][i]
                    if float(t) != expected['time_seconds']:
                        raise QAIncomplete('Original callback timestamp differs from prospective clock schedule')
                    monitor.observe(t, y, bundle)
                    position['points'] += 1
                    if not (archived and arm=='reference') or position['points']==len(plan['points']):
                        witness = _point_witness(runner, current[0], monitor, probe, expected)
                        pair('point', witness, comparison.compare_point)
                end_state, result = runner.advance(start_state, channels, stop,
                    sample_times=samples, recording_plan=recording_plan, state_observer=observe)
            for category in position:
                if position[category] != len(plan[category]):
                    raise QAIncomplete('Original witness coverage missing: '+category)
            pair('monitor', _monitor_witness(monitor, plan['monitors'][0]), comparison.compare_monitor)
            recorded = _recording_witness(result)
            if arm == 'reference':
                if archived:archived.assert_exact('ordinary_recording',recorded)
                else:spool.write('ordinary_recording', recorded)
            else:
                with (archived if archived else spool).read('ordinary_recording') as old:
                    _compare_recording(comparison, old, recorded)
            if 'dense_sample' not in ownership:
                raise QAIncomplete('A real internal dense sample was not retained across workspace reuse')
            ownership_witness = dict(id='dense_and_input',
                snapshot_hash_at_return=canonical_hash([ownership['hash'], ownership['dense_hash']]),
                snapshot_hash_after_reuse=canonical_hash([canonical_hash(_host(ownership['sample'])),
                                                        canonical_hash(_host(ownership['dense_sample']))]),
                supplied_state_hash_before=supplied_before,
                supplied_state_hash_after=canonical_hash(json.loads(runner.checkpoint(start_state))))
            pair('ownership', ownership_witness, comparison.compare_ownership)
            print('QA '+arm+' main complete; checkpoint',flush=True)
            # Drop the callback's last full point before constructing codec
            # witnesses. Nothing numerical is omitted from its saved evidence.
            if 'witness' in locals():del witness
            probe.monitor_locals=None
            pair('checkpoint', _checkpoint_witness(runner, end_state, reference_state, policy, channels,
                                                   continuation_stop, spool), comparison.compare_checkpoint)
            for case in plan['failure_cases']:
                pair('failure', _failure_witness(runner, end_state, reference_state, policy, channels,
                                                 continuation_stop, case), comparison.compare_failure)
            del monitor,probe,result,recorded,end_state,ownership
        if hash_prepared_sources(prepared_sources) != sources_hash:
            comparison._issue('FAIL', 'prepared_inputs', 'Prepared source content changed during QA')
        if any(_file_hash(Path(__file__).with_name(name)) != digest
               for name, digest in plan['qa_implementation_sha256'].items()):
            comparison._issue('FAIL', 'qa_identity', 'QA implementation changed after prospective sealing')
        outcome = asdict(comparison.finish())
    except Exception as exc:
        if comparison is not None and not comparison.closed:
            comparison._issue('INCONCLUSIVE' if isinstance(exc, (QABudgetStop, QAIncomplete, TimeoutError)) else 'FAIL',
                              'harness_stop', type(exc).__name__+': '+str(exc))
            outcome = asdict(comparison.finish())
        else:
            outcome = dict(status='INCONCLUSIVE', error_type=type(exc).__name__, detail=str(exc),
                           numerical_only=True, fit_eligible=False, performance_validated=False)
    outcome.update(execution_authorization=restriction_release_evidence,
                   wall_seconds=time.perf_counter()-spool.start, spool_bytes=spool.used,
                   QA_trace_readback_and_IO_are_not_performance_evidence=True)
    # A budget stop must still leave a small readable terminal record. This
    # reserved report is never a replacement for missing numerical witnesses.
    target = spool.root/'outcome.json'
    with target.open('x', encoding='utf8') as stream:
        json.dump(outcome, stream, sort_keys=True, allow_nan=False, indent=2)
    # Optional same-process owner for subsequent bounded engineering work.
    # The numerical outcome is already immutable on disk; no graph is copied,
    # callback is inserted into either trajectory, or verdict changed here.
    if context_sink is not None and comparison is not None:
        if not callable(context_sink):
            raise ValueError('QA context sink must be callable')
        context_sink(dict(reference=ref,candidate=candidate,initial=initial,plan=plan,
                          comparator_template=comparison))
    return outcome

