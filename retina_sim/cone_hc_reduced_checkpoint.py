"""Atomic binary reduced checkpoints, deliberately incompatible with old states."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import uuid

import numpy as np

from .contracts import ContractError, SolverConfig, canonical_hash
from .cone_hc_split_checkpoint import file_sha, _flush_bytes
from .cone_hc_reduced_native import AnalyticREAnchor, ReducedNativeState, QuadraticNativeSegment
from .cone_hc_reduced_runner import ReducedPoint, ReducedState, ReducedCursor, CONSUMERS
from .resident_delay import ResidentTimestampedDelay


SCHEMA = 'cone-hc-reduced-complete-binary-checkpoint-1'


def _native_payload(state):
    state.check_owned()
    anchor = state.anchor
    return dict(time_seconds=state.time_seconds,y=state.y,source_version=state.source_version,
                anchor=dict(time_seconds=anchor.time_seconds,R=anchor.R,E=anchor.E,light=anchor.light,
                            input_stop_seconds=anchor.input_stop_seconds,input_epoch=anchor.input_epoch))


def _payload(runner,state):
    state.validate(runner)
    c = state.cursor
    if any(t!=state.time_seconds for t in c.consumed.values()):
        raise ContractError('Reduced checkpoint requires synchronous consumers at public time')
    return dict(point=dict(native=_native_payload(state.point.native),inner=state.point.inner,
                           compact_release=state.point.compact_release),
        cursor=dict(frontier=_native_payload(c.frontier),consumed=dict(c.consumed),next_id=c.next_id,
            segments=[dict(start=s.start,stop=s.stop,y0=s.y0,f0=s.f0,y1=s.y1,
                input_epoch=s.input_epoch,source_version=s.source_version,
                anchor=_native_payload(ReducedNativeState(s.start,s.y0,s.anchor,s.source_version,False))['anchor'])
                for s in c.segments],packets=[p.payload() for p in c.packets]),
        solver=asdict(state.solver_config),grid_origin_seconds=state.grid_origin_seconds,
        sample_index=state.sample_index,spike_states=state.spike_states,
        delay_states={k:v.payload() for k,v in state.delay_states.items()},
        input_position=state.input_position,component_tick_indices=state.component_tick_indices,
        link_delay_states=state.link_delay_states,generation=state.generation)


def save_complete(runner,state,directory):
    target = Path(directory).resolve()
    if target.exists() or not target.parent.is_dir():
        raise ContractError('Reduced checkpoint target must be absent with an existing parent')
    staging = target.with_name(target.name+'.writing-'+uuid.uuid4().hex)
    files = {}
    with runner._lock,runner.torch.no_grad():
        runner._check_runtime()
        payload = _payload(runner,state)  # Waits and privately copies every pending D2H packet.
        staging.mkdir()
        def encode(value):
            if hasattr(value,'detach'):
                value = value.detach().cpu().numpy()
            if isinstance(value,np.ndarray):
                if value.dtype.kind not in 'fiub' or value.dtype.hasobject:
                    raise ContractError('Checkpoint forbids object/pickle arrays')
                if not np.isfinite(value).all():
                    raise ContractError('Checkpoint forbids nonfinite arrays')
                name = 'array_%05d.npy'%len(files)
                path = staging/name
                with path.open('xb') as stream:
                    np.save(stream,value,allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                files[name] = dict(sha256=file_sha(path),bytes=path.stat().st_size,
                                  dtype=value.dtype.str,shape=list(value.shape))
                return {'array_file':name}
            if isinstance(value,np.generic):
                return value.item()
            if isinstance(value,dict):
                return {k:encode(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):
                return [encode(v) for v in value]
            return value
        encoded = encode(payload)
        manifest = dict(schema=SCHEMA,identity=runner.identity,input_binding=runner.binding_hash,
            native_source=runner.native.source_version,selected_ids=runner.selection.sha256,
            time_seconds=state.time_seconds,classification='B',fit_eligible=False,
            files=files,payload=encoded)
        digest = canonical_hash(manifest)
        _flush_bytes(staging/'manifest.json',json.dumps(dict(manifest,manifest_hash=digest),
            sort_keys=True,allow_nan=False).encode('utf8'))
        if target.exists():
            raise ContractError('Checkpoint target appeared; refusing replacement')
        os.replace(staging,target)
    return dict(schema=SCHEMA,directory=str(target),manifest_hash=digest,
                bytes=sum(v['bytes'] for v in files.values()),fit_eligible=False)


def load_complete(runner,directory,*,expected_manifest_hash):
    runner._check_runtime()
    directory = Path(directory).resolve()
    if '.writing-' in directory.name:
        raise ContractError('Uncommitted reduced checkpoint directory')
    raw = json.loads((directory/'manifest.json').read_bytes())
    digest = raw.pop('manifest_hash',None)
    if (raw.get('schema')!=SCHEMA or digest!=expected_manifest_hash or canonical_hash(raw)!=digest
            or raw.get('identity')!=runner.identity or raw.get('input_binding')!=runner.binding_hash
            or raw.get('native_source')!=runner.native.source_version
            or raw.get('selected_ids')!=runner.selection.sha256 or raw.get('fit_eligible') is not False):
        raise ContractError('Reduced checkpoint schema/content/model/input/selection differs')
    arrays = {}
    for name,item in raw['files'].items():
        path = directory/name
        if (Path(name).name!=name or path.resolve().parent!=directory or not path.is_file()
                or path.stat().st_size!=item['bytes'] or file_sha(path)!=item['sha256']):
            raise ContractError('Reduced checkpoint array receipt differs')
        a = np.load(path,allow_pickle=False)
        if a.dtype.str!=item['dtype'] or list(a.shape)!=item['shape'] or not np.isfinite(a).all():
            raise ContractError('Reduced checkpoint array layout/domain differs')
        arrays[name] = a
    def decode(v):
        if isinstance(v,dict) and set(v)=={'array_file'}:
            if v['array_file'] not in arrays:
                raise ContractError('Unverified checkpoint array reference')
            return arrays[v['array_file']].copy()
        if isinstance(v,dict):
            return {k:decode(x) for k,x in v.items()}
        if isinstance(v,list):
            return [decode(x) for x in v]
        return v
    p = decode(raw['payload'])
    def gpu(a):
        if not isinstance(a,np.ndarray) or a.dtype!=np.float64:
            raise ContractError('Reduced native checkpoint must be FP64')
        return runner.torch.as_tensor(a,dtype=runner.torch.float64,device=runner.native.device).clone()
    def anchor(a):
        return AnalyticREAnchor(a['time_seconds'],gpu(a['R']),gpu(a['E']),gpu(a['light']),
                                a['input_stop_seconds'],a['input_epoch'])
    def native(n):
        return ReducedNativeState(n['time_seconds'],gpu(n['y']),anchor(n['anchor']),n['source_version'],False)
    with runner._lock,runner.torch.no_grad():
        point = ReducedPoint(native(p['point']['native']),p['point']['inner'],p['point']['compact_release'])
        c = p['cursor']
        cursor = ReducedCursor(native(c['frontier']),consumed=c['consumed'],next_id=c['next_id'])
        previous = None
        for s,q in zip(c['segments'],c['packets'],strict=True):
            segment = QuadraticNativeSegment(s['start'],s['stop'],gpu(s['y0']),gpu(s['f0']),gpu(s['y1']),
                s['input_epoch'],anchor(s['anchor']),s['source_version'],False)
            packet = runner.selection.from_payload(q)
            if (segment.source_version!=runner.native.source_version or segment.stop<=point.native.time_seconds
                    or (previous is not None and segment.start!=previous)
                    or (packet.start,packet.stop,packet.input_epoch,packet.source_version)!=(
                        segment.start,segment.stop,segment.input_epoch,segment.source_version)
                    or packet.segment_id>=cursor.next_id):
                raise ContractError('Reduced checkpoint unconsumed segment ownership differs')
            projected = runner.torch.stack([v.index_select(0,runner.bridge.rows)
                for v in (segment.y0,segment.f0,segment.y1)]).cpu().numpy()
            if not np.array_equal(projected.view(np.uint64),packet.values.view(np.uint64)):
                raise ContractError('Reduced checkpoint CPU packet differs from accepted GPU segment')
            cursor.segments.append(segment)
            cursor.packets.append(packet)
            previous = segment.stop
        state = ReducedState(point,cursor,runner.identity,runner.binding_hash,SolverConfig(**p['solver']),
            p['grid_origin_seconds'],p['sample_index'],p['spike_states'],
            {k:ResidentTimestampedDelay.from_payload(v,runner.torch,runner.native.device)
             for k,v in p['delay_states'].items()},p['input_position'],p['component_tick_indices'],
            p['link_delay_states'],p['generation'])
        state.validate(runner)
        runner.native._validate(state.point.native.y,state.point.native.anchor,state.time_seconds)
        if state.time_seconds!=raw['time_seconds']:
            raise ContractError('Reduced checkpoint public time differs')
        return state
