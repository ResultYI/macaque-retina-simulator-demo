"""Lossless group replay sink using the existing bounded desktop container.

Group predictions are explicitly labelled. Recording never computes a missing
cell, reconstructs an unrecorded history or changes the computational mesh.
"""
from copy import deepcopy
from pathlib import Path
import json
import os

import numpy as np

from .contracts import ContractError, canonical_hash
from .spatial_group_graph import digest, verified_path
from tools.cone_hc_foveal_recording import ReplayWriter


SCHEMA = 'retipath-group-recording-cursor-1'
REPLAY_SCHEMA = 'retipath-spatial-group-replay-1'
ROOT = Path(__file__).resolve().parents[1]


def _layer(pop):
    return 'cone' if pop == 'cones' else 'HC' if pop == 'HC' else 'RGC' if 'RGC' in pop or pop == 'small_bistratified_S_ON' else 'BC' if 'BC' in pop else 'AC'


def _json_safe(value):
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    if isinstance(value, dict):return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [_json_safe(v) for v in value]
    return value


class GroupRecording:
    def __init__(self, assembly, out, times, artifactidentity=None, *, buffer_bytes=64*1024**2):
        self._configure(assembly, artifactidentity)
        self.writer = ReplayWriter(Path(out), self.components, times, buffer_bytes=buffer_bytes)
        self.path = self.writer.path
        (self.path/'observations').mkdir()
        (self.path/'events').mkdir()
        self.pending_delayed, self.pending_events = [], []
        self.delayed_rows = None
        self.native_readout_seconds = 0.

    def _configure(self, assembly, artifactidentity):
        self.assembly = assembly
        self.artifactidentity = assembly.artifact_sha256 if artifactidentity is None else deepcopy(artifactidentity)
        self.fields = tuple(assembly.recording_fields)
        self.field_by_key = {f.key:f for f in self.fields}
        if len(self.field_by_key) != 45:
            raise ContractError('All 45 existing group observation fields are required')
        components, self.rows = {}, {}
        subtype_cache={}
        def subtypes(pop):
            if pop in subtype_cache:return subtype_cache[pop]
            n=assembly.populations[pop].group_count
            meta=assembly.manifest.get('populations',{}).get(pop,{})
            relative='populations/'+pop+'/keys.npy'
            if relative in assembly.manifest.get('files',{}):
                keys=assembly.array(relative)
                labels=meta.get('subtype_labels',[])
                parent=meta.get('parent_population')
                if labels:
                    value=np.asarray(labels)[keys[:,1]]
                elif parent:
                    value=subtypes(parent)[keys[:,2]]
                else:value=np.full(n,pop)
            else:value=np.full(n,pop)
            subtype_cache[pop]=value
            return value
        for field in self.fields:
            pop = field.population; layout = assembly.populations[pop]
            rows = np.asarray(field.rows, dtype=np.int64)
            group_ids = [str(x) for x in layout.group_ids[rows]]
            original = field.original_fine_rows
            if original is None:
                ids = group_ids
                xy = layout.centroid_mm[rows]
            else:
                original = np.asarray(original,dtype=np.int64)
                if (original.shape != rows.shape or not np.array_equal(layout.fine_to_group[original],rows)):
                    raise ContractError('Original auxiliary row/group mapping differs')
                ids = [g+'|original_row:'+str(int(r)) for g,r in zip(group_ids,original)]
                if hasattr(layout,'source_positions_mm'):
                    xy = np.asarray(layout.source_positions_mm)[original]
                else:
                    coordinate = assembly.manifest['populations'][pop]['coordinate']
                    xy = np.load(verified_path(ROOT,coordinate),allow_pickle=False,mmap_mode='r')[original]
            role = layout.coordinate_role
            coordinate_key = 'RF' if role == 'rf' else 'soma' if role == 'soma' else 'IPL'
            cid = 'cone' if pop == 'cones' else pop
            component = components.setdefault(cid,dict(id=cid,layers=[_layer(pop)],
                mechanism='saved_group_state',ports={}))
            port = dict(key=field.key,unit=field.unit,meaning=field.meaning,layer=_layer(pop),
                cell_ids=ids,cell_types=[pop]*len(rows),subtypes=subtypes(pop)[rows].tolist(),
                group_ids=group_ids,group_rows=rows.tolist(),
                member_counts=np.asarray(layout.counts)[rows].tolist(),
                geometry={coordinate_key:dict(values=np.asarray(xy).tolist(),unit='mm',
                    frame=assembly.manifest.get('populations',{}).get(pop,{}).get('coordinate',{}).get('frame','original_source_coordinate_frame'),meaning=role)},
                temporal_kind='sampled',cell_identity_status=field.identity,
                recorded_count=len(rows),population_count=layout.group_count,
                selection='all computed representative groups' if field.primary else 'unchanged original-cell position selection mapped to group predictions',
                primary_field=field.primary,original_fine_global_rows=None if original is None else original.tolist(),
                original_identity_binding=dict(source_table=layout.source_table,source_keys=layout.source_keys),
                member_mapping=dict(artifact_sha256=assembly.artifact_sha256,population=pop,
                    member_offsets='populations/'+pop+'/member_offsets.npy',
                    member_global_rows='populations/'+pop+'/member_global_rows.npy',
                    counts='populations/'+pop+'/counts.npy'),
                coordinate_role=role)
            component['ports'][field.name] = port
            self.rows[field.key] = rows
        self.components = list(components.values())
        self.layout_identity = canonical_hash(dict(schema=REPLAY_SCHEMA,artifact=self.artifactidentity,
                                                  components=self.components))

    def _select(self, field, value):
        if isinstance(value, dict):
            if set(value) != {'values','group_rows'} or not np.array_equal(value['group_rows'],field.rows):
                raise ContractError('Preselected native observation must identify the exact ordered group rows')
            a = np.asarray(value['values'])
            expected = (len(field.rows),)
        else:
            a = np.asarray(value)
            expected = (self.assembly.populations[field.population].group_count,)
        if a.dtype != np.float64 or a.shape != expected or not np.isfinite(a).all():
            raise ContractError('Finite FP64 actual group observations required: '+field.key)
        return a.copy() if isinstance(value,dict) else a[field.rows].copy()

    def _next_file_free(self):
        path=self.path/'windows'/('window_%04d.npz'%self.writer.window_index)
        if path.exists():
            raise ContractError('Refusing to overwrite a later or unknown recording window')

    def record(self,t,native_values,inner_outputs,delayedHC=None,eventlogs=None):
        values={}
        for field in self.fields:
            if field.population in ('cones','HC'):
                name='gi' if field.name=='g_i' and 'g_i' not in native_values else field.name
                if name not in native_values:
                    raise ContractError('Missing actual native recording field: '+field.name)
                value=native_values[name]
            else:
                if (field.population,field.name) not in inner_outputs:
                    raise ContractError('Missing actual inner recording field: '+field.key)
                value=inner_outputs[field.population,field.name]
            values[field.key]=self._select(field,value)
        delayed=None
        if delayedHC is not None:
            n=self.assembly.populations['HC'].group_count
            if isinstance(delayedHC,dict):
                if set(delayedHC)!={'values','group_rows','query_time'} or delayedHC['query_time']!=t-.00282:
                    raise ContractError('Designated HC delay must identify its exact query time')
                rows=np.asarray(delayedHC['group_rows'])
                a=np.asarray(delayedHC['values'])
                if rows.dtype.kind not in 'iu' or rows.ndim!=1 or np.any(rows<0) or np.any(rows>=n) or len(np.unique(rows))!=len(rows):
                    raise ContractError('Designated HC delay has invalid group identities')
            else:
                rows=np.arange(n,dtype=np.int64);a=np.asarray(delayedHC)
            if a.dtype!=np.float64 or a.shape!=(len(rows),) or not np.isfinite(a).all():
                raise ContractError('Designated delayed HC values must be actual finite FP64 readouts')
            if self.delayed_rows is not None and not np.array_equal(rows,self.delayed_rows):
                raise ContractError('Designated delayed HC observation identities changed')
            delayed=dict(time=float(t),query_time=float(t-.00282),group_rows=rows.copy(),values=a.copy())
        events=None
        if eventlogs is not None:
            events=dict(time=float(t),events=_json_safe(eventlogs))
            json.dumps(events,allow_nan=False)
        if len(self.writer.pending_times)+1>=self.writer.window_samples:self._next_file_free()
        before=self.writer.window_index
        self.writer.append(t,values)
        if delayed is not None:
            self.delayed_rows=delayed['group_rows'].copy()
            self.pending_delayed.append(delayed)
        if events is not None:self.pending_events.append(events)
        if self.writer.window_index!=before:self._flush_auxiliary(before)

    def _flush_auxiliary(self,window_index):
        if self.pending_delayed:
            first=self.pending_delayed[0]['group_rows']
            if any(not np.array_equal(first,d['group_rows']) for d in self.pending_delayed):
                raise ContractError('Designated HC observation rows changed within one recording window')
            relative='observations/delayed_hc_%04d.npz'%window_index
            path=self.path/relative
            if path.exists():raise ContractError('Refusing to replace delayed HC evidence')
            np.savez_compressed(path,time=np.array([d['time'] for d in self.pending_delayed]),
                query_time=np.array([d['query_time'] for d in self.pending_delayed]),group_rows=first,
                values=np.column_stack([d['values'] for d in self.pending_delayed]),
                delay_seconds=np.array(.00282),feedback_uses_current_h=np.array(True))
            self.writer.files[relative]=digest(path)
            self.pending_delayed.clear()
        if self.pending_events:
            relative='events/events_%04d.json'%window_index;path=self.path/relative
            if path.exists():raise ContractError('Refusing to replace original event evidence')
            path.write_text(json.dumps(dict(schema='retipath-original-event-observations-1',
                records=self.pending_events),ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
            self.writer.files[relative]=digest(path)
            self.pending_events.clear()

    def _flush(self):
        if self.writer.pending_times:
            self._next_file_free()
            index=self.writer.window_index;self.writer.flush();self._flush_auxiliary(index)

    def checkpoint_payload(self):
        if self.writer.closed:raise ContractError('Finished replay has no active recording cursor')
        self._flush()
        payload=dict(schema=SCHEMA,layout_identity=self.layout_identity,artifactidentity=deepcopy(self.artifactidentity),
            times=self.writer.times.copy(),index=self.writer.index,window_index=self.writer.window_index,
            window_samples=self.writer.window_samples,files=dict(self.writer.files),
            delayed_rows=None if self.delayed_rows is None else self.delayed_rows.copy(),
            baseline={k:v.copy() for k,v in self.writer.baseline.items()},
            extrema=deepcopy(self.writer.extrema),relative_peak=dict(self.writer.relative_peak),
            write_seconds=self.writer.write_seconds,estimated_value_bytes=self.writer.estimated_value_bytes)
        return dict(payload,cursor_hash=canonical_hash(payload))

    @classmethod
    def restore(cls,assembly,out,payload,artifactidentity=None):
        result=cls.__new__(cls);result._configure(assembly,artifactidentity)
        keys={'schema','layout_identity','artifactidentity','times','index','window_index','window_samples','files',
              'baseline','extrema','relative_peak','write_seconds','estimated_value_bytes','delayed_rows','cursor_hash'}
        if (set(payload)!=keys or payload['schema']!=SCHEMA or payload['layout_identity']!=result.layout_identity
                or canonical_hash(payload['artifactidentity'])!=canonical_hash(result.artifactidentity)):
            raise ContractError('Recording checkpoint schema/layout/artifact differs')
        if payload['cursor_hash']!=canonical_hash({k:v for k,v in payload.items() if k!='cursor_hash'}):
            raise ContractError('Recording checkpoint cursor identity differs')
        path=Path(out).resolve();times=np.asarray(payload['times'])
        if (not path.is_dir() or times.dtype!=np.float64 or times.ndim!=1 or len(times)<2
                or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
                or type(payload['index']) is not int or not 0<=payload['index']<=len(times)
                or type(payload['window_index']) is not int or payload['window_index']<0
                or type(payload['window_samples']) is not int or not 1<=payload['window_samples']<=len(times)
                or not np.isfinite(payload['write_seconds']) or payload['write_seconds']<0):
            raise ContractError('Recording checkpoint time/cursor differs')
        if any((path/name).exists() for name in ('manifest.json','display_baseline.npz','stimulus.npz')):
            raise ContractError('Refusing to restore over a finalized or later replay result')
        # A rewind over a later run would overwrite unknown evidence. Require
        # exactly the committed files, keeping failures and later files intact.
        owned={p.relative_to(path).as_posix() for folder in ('windows','observations','events') for p in (path/folder).glob('*') if p.is_file()}
        if owned!=set(payload['files']):raise ContractError('Recording directory contains later or unknown files')
        seen=[]
        for relative,expected in payload['files'].items():
            source=(path/relative).resolve()
            if not source.is_relative_to(path) or not source.is_file() or digest(source)!=expected:
                raise ContractError('Recording checkpoint window identity differs')
        for i in range(payload['window_index']):
            with np.load(path/'windows'/('window_%04d.npz'%i),allow_pickle=False) as z:seen.extend(z['time'].tolist())
        if not np.array_equal(np.array(seen),times[:payload['index']]):
            raise ContractError('Recording checkpoint lost exact committed sample positions')
        w=ReplayWriter.__new__(ReplayWriter)
        w.path,w.times,w.components=path,times.copy(),result.components
        w.ports={p['key']:p for c in result.components for p in c['ports'].values()}
        if payload['estimated_value_bytes']!=sum(len(p['cell_ids']) for p in w.ports.values())*8*len(times):
            raise ContractError('Recording checkpoint declared byte ledger differs')
        for name in ('index','window_index','window_samples','write_seconds','estimated_value_bytes'):
            setattr(w,name,payload[name])
        w.buffer={k:np.empty((len(p['cell_ids']),w.window_samples),dtype=np.float64) for k,p in w.ports.items()}
        expected=set(w.ports) if w.index else set()
        if set(payload['baseline'])!=expected or set(payload['extrema'])!=expected or set(payload['relative_peak'])!=expected:
            raise ContractError('Recording checkpoint display baseline is incomplete')
        for k,a in payload['baseline'].items():
            if np.asarray(a).dtype!=np.float64 or np.shape(a)!=(len(w.ports[k]['cell_ids']),) or not np.isfinite(a).all():
                raise ContractError('Recording checkpoint baseline shape/domain differs')
        w.baseline={k:v.copy() for k,v in payload['baseline'].items()}
        w.extrema,w.relative_peak,w.files=deepcopy(payload['extrema']),dict(payload['relative_peak']),dict(payload['files'])
        w.pending_times,w.closed=[],False
        result.writer,result.path=w,path
        result.pending_delayed,result.pending_events=[],[]
        result.delayed_rows=None if payload['delayed_rows'] is None else np.array(payload['delayed_rows'],dtype=np.int64,copy=True)
        result.native_readout_seconds=0.
        return result

    def finish(self,frames,frame_times,source_indices,metadata=None):
        self._flush()
        metadata={} if metadata is None else deepcopy(metadata)
        replay=dict(status='computed_candidate_not_default',physiological_calibration=False,
            source='actual_saved_group_model',neural_data_generated=True)
        replay.update(metadata.get('replay',{}))
        replay['physiological_calibration']=False
        manifest=dict(metadata,group_replay_schema=REPLAY_SCHEMA,group_artifact_identity=self.artifactidentity,
            recording_layout_identity=self.layout_identity,model_scope=dict(output_radius_deg=6.,
                continuous_output_disk=True,complete_upstream_dependencies=True),
            stimulus_geometry=dict(width_deg=12.,height_deg=12.,center_deg=[0.,0.],
                model_output_disk_radius_deg=6.,full_frame_modelled=False,
                interpretation='The 0–6 degree output disk is inside the centered12degree square; upstream support extends beyond it'),
            observation_delay=dict(seconds=.00282,path='observations/delayed_hc_*.npz',
                designated_readout_only=True,feedback_uses_current_h=True),
            identity_semantics='Representative group states and original-position group predictions; no independent fine-cell truth',
            replay=replay)
        receipt=self.writer.finish(manifest,frames,frame_times,source_indices)
        target=self.path/'manifest.json';data=json.loads(target.read_text(encoding='utf8'))
        data['demo_schema']=REPLAY_SCHEMA
        pending=target.with_name('manifest.json.group-writing')
        pending.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n',encoding='utf8')
        os.replace(pending,target)
        return dict(receipt,sha256=digest(target),schema=REPLAY_SCHEMA,recorded_fields=len(self.fields))
