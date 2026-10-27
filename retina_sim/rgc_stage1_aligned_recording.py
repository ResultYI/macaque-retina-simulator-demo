"""Read-only snapshot observer; separate aligned NPY product, never a solver."""
from pathlib import Path
from time import perf_counter
import hashlib,json,math,shutil
import numpy as np

SCHEMA='cell6-aligned-replay-1'

def canonical_points(start,stop,dt,n,events):
    """Mirror original capture's author-tick filter, including exact float keys."""
    points={float(start):0};recorded=1
    for t,flags in events:
        if t<=start:continue
        j=int(round((t-start)/dt))
        if (flags['rod'] or t==stop) and j==recorded:
            points[float(t)]=j;recorded+=1
    if recorded!=n or len(points)!=n:raise ValueError('Original canonical calendar coverage differs')
    return points


def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(path)

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def copy_checked_prefix(source,target,receipt):
    """Copy only the accepted checkpoint prefix into a fresh product file."""
    with Path(source).open('rb') as f:
        version=np.lib.format.read_magic(f)
        if version!=(1,0):raise ValueError('Unexpected aligned NPY header version')
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(f);offset=f.tell();f.seek(0);header=f.read(offset)
        if tuple(shape)!=target.shape or fortran or np.dtype(dtype)!=target.dtype:raise ValueError('Recovery NPY layout differs')
        h=hashlib.sha256(header);remaining=int(receipt['rows_written']);tail=int(np.prod(shape[1:],dtype=np.int64))
        step=max(1,8388608//max(tail*target.dtype.itemsize,1))
        while remaining:
            take=min(remaining,step);raw=f.read(take*tail*target.dtype.itemsize)
            if len(raw)!=take*tail*target.dtype.itemsize:raise ValueError('Recovery prefix truncated')
            h.update(raw);target.append(np.frombuffer(raw,dtype=dtype).reshape((take,*shape[1:])));remaining-=take
        if h.hexdigest()!=receipt['sha256']:raise ValueError('Recovery checkpoint prefix SHA differs')

class NPYRows:
    """Append bounded row blocks under a fixed header; publish only when full."""
    def __init__(self,path,shape,dtype):
        self.path=Path(path);self.shape=tuple(shape);self.dtype=np.dtype(dtype);self.rows=0;self.hash=hashlib.sha256()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('xb') as f:np.lib.format.write_array_header_1_0(f,dict(descr=self.dtype.str,fortran_order=False,shape=self.shape))
        self.hash.update(self.path.read_bytes())
    def append(self,value):
        value=np.ascontiguousarray(value,dtype=self.dtype)
        if value.shape[1:]!=self.shape[1:] or self.rows+len(value)>self.shape[0]:raise ValueError('Aligned NPY row extent differs')
        data=memoryview(value).cast('B')
        with self.path.open('ab') as f:f.write(data)
        self.hash.update(data);self.rows+=len(value)
    def receipt(self,root):
        return dict(path=self.path.relative_to(root).as_posix(),sha256=self.hash.hexdigest(),shape=list(self.shape),dtype=self.dtype.str,rows_written=self.rows)

class PhaseStream:
    def __init__(self,root,name,fields,n,buffer_rows):
        self.root=root;self.name=name;self.fields=fields;self.n=n;self.row=0;self.used=0;self.capacity=buffer_rows
        self.values={f['id']:NPYRows(root/name/'values'/(f['id']+'.npy'),(n,f['count']),'<f8') for f in fields}
        self.buffers={f['id']:np.empty((buffer_rows,f['count']),dtype=np.float64) for f in fields}
        self.index_types=dict(time_source_s='<f8',model_time_s='<f8',side='|S5',phase='|S40',event_generation='<i8',clock_rod='<i8',clock_bc='<i8',clock_rgc='<i8',input_frame='<i8')
        self.indices={k:NPYRows(root/name/(k+'.npy'),(n,),d) for k,d in self.index_types.items()}
        self.index_buffers={k:np.empty(buffer_rows,dtype=d) for k,d in self.index_types.items()}
    def append(self,j,snapshot,phase,dt):
        if j!=self.row:raise ValueError('Aligned source row skipped/repeated; no partial prefix invented')
        i=self.used
        for f in self.fields:
            value=np.asarray(snapshot[f['kind']][tuple(f['key'])])
            if value.dtype!=np.float64 or value.shape!=(f['count'],) or not np.isfinite(value).all():raise ValueError('Aligned field shape/dtype/domain differs: '+str(f['key']))
            self.buffers[f['id']][i]=value
        aux=snapshot['aux'];clocks=aux['clocks']['indices'];frame=aux['input_position']['public_native_frame']
        values=dict(time_source_s=j*dt,model_time_s=snapshot['time_seconds'],side=snapshot['side'],phase=phase,event_generation=snapshot['event_generation'],clock_rod=clocks['rod'],clock_bc=clocks['bc'],clock_rgc=clocks['rgc'],input_frame=frame)
        for k,v in values.items():self.index_buffers[k][i]=v
        self.row+=1;self.used+=1
        if self.used==self.capacity:self.flush()
    def flush(self):
        if not self.used:return
        for k,v in self.values.items():v.append(self.buffers[k][:self.used])
        for k,v in self.indices.items():v.append(self.index_buffers[k][:self.used])
        self.used=0
    def manifest(self):
        signals=[]
        for f in self.fields:
            r=self.values[f['id']].receipt(self.root)
            signals.append(dict(f,values=r['path'],values_sha256=r['sha256'],shape=r['shape'],dtype=r['dtype'],point_index=self.name))
        return dict(rows=self.row,shared_point_index={k:v.receipt(self.root) for k,v in self.indices.items()},signals=signals)

class AlignedObserver:
    def __init__(self,directory,contract,assembly):
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=False);self.contract=contract;self.assembly=assembly
        self.n=contract['source_nodes'];self.dt=contract['source_dt_s'];self.start=contract['source_model_origin_s'];self.stop=self.start+(self.n-1)*self.dt
        self.streams={};self.bins={};self.bin_rows={};self.fields=[];self.error=None;self.closed=False;self.last_t=None;self.ordinal=0
        self.copy_wall=0.;self.flush_wall=0.;self.observations=0;self.extra_event_points=0
        self.canonical_points=None
        self.state_specs={tuple(x['key']):x for x in contract['state_fields']}
        self.node_populations={x['key'][0] for x in contract['state_fields']}
        self.rgc=contract['RGC_populations'];self._geometry()
        write(self.root/'recording_contract.json',contract)
    def set_canonical_points(self,points):
        if len(points)!=self.n or list(points.values())!=list(range(self.n)) or points.get(self.start)!=0:
            raise ValueError('Exact original capture point map differs')
        self.canonical_points=points
    def _geometry(self):
        self.geometry={};self.geometry_arrays={}
        for pop,p in self.assembly.populations.items():
            data={k:np.asarray(getattr(p,k)) for k in ['group_ids','counts','centroid_mm']}
            for k in ['member_offsets','member_global_rows','recording_auxiliary_group_rows']:
                if hasattr(p,k):data[k]=np.asarray(getattr(p,k))
            paths={}
            for k,v in data.items():
                path=self.root/'geometry'/pop/(k+'.npy');path.parent.mkdir(parents=True,exist_ok=True);np.save(path,v,allow_pickle=False)
                paths[k]=dict(path=path.relative_to(self.root).as_posix(),sha256=digest(path),shape=list(v.shape))
            functional='functional' in p.coordinate_role
            self.geometry[pop]=dict(groups=int(p.group_count),coordinate_role=p.coordinate_role,cell_count_not_applicable=functional,dynamic_states_recorded=pop in self.node_populations,arrays=paths,
                source_table=getattr(p,'source_table',None),source_keys=getattr(p,'source_keys',None),state_semantics='effective functional pool' if functional else 'representative group; not independent fine-cell states')
            self.geometry_arrays[pop]=data
        write(self.root/'geometry.json',self.geometry)
    def _axis(self,pop,mechanism,count):
        p=self.assembly.populations.get(pop)
        if p is not None and count==p.group_count and mechanism not in ['individual_rod_RBC_contacts','RBC_AII_AMPA','AII_FMB_glycine','AII_DB1_glycine']:
            return dict(axis_kind='node_group',population_registry='geometry.json',population=pop,axis_group_rows=list(range(count)),coordinate_role=p.coordinate_role)
        if p is not None and mechanism=='selected_original_z1_auxiliary' and hasattr(p,'recording_auxiliary_group_rows'):
            rows=np.asarray(p.recording_auxiliary_group_rows)
            if rows.shape==(count,):return dict(axis_kind='selected_auxiliary_group',population_registry='geometry.json',population=pop,axis_group_rows=rows.tolist(),coordinate_role=p.coordinate_role)
        contacts=getattr(getattr(self.assembly,'patch',None),'parent_contact_rows',{})
        axis=contacts.get(mechanism)
        return dict(axis_kind='contact_or_component_vector',population=pop,mechanism=mechanism,
            parent_contact_group_rows=None if axis is None or len(axis)!=count else np.asarray(axis).tolist(),coordinate_role='component_axis_not_cell_coordinates',spatial_node_plot_permitted=False)
    def _initialize(self,snapshot):
        if set(snapshot['states'])!=set(self.state_specs):raise ValueError('All frozen120 state fields must be present')
        for key,spec in self.state_specs.items():
            self.fields.append(dict(id='f'+str(len(self.fields)).zfill(4),kind='states',key=list(key),population=key[0],mechanism=key[1],field=key[2],unit=spec['unit'],count=spec['count'],role=spec['role'],axis=self._axis(key[0],key[1],spec['count'])))
        self.excluded_ports=[]
        for key,value in sorted(snapshot['ports'].items()):
            a=np.asarray(value)
            if key[0] in ['local_crossover_AC','widefield_AC'] or key[0] in self.rgc and any(w in key[1].lower() for w in ['rate','intensity','probability','count','spike','event']):
                self.excluded_ports.append(dict(key=list(key),reason='archived population or RGC non-state output kept off100us rate axis'));continue
            if a.dtype!=np.float64 or a.ndim!=1:
                self.excluded_ports.append(dict(key=list(key),reason='not declared one-dimensionalFP64 port; no coerced representation'));continue
            self.fields.append(dict(id='f'+str(len(self.fields)).zfill(4),kind='ports',key=list(key),population=key[0],mechanism='actual_snapshot_port',field=key[1],unit=snapshot['port_units'][key],count=len(a),role='derived_actual_model_output',axis=self._axis(key[0],'actual_snapshot_port',len(a))))
        for f in self.fields:
            axis=f['axis'];pop=f['population']
            if axis['axis_kind'] in ['node_group','selected_auxiliary_group']:
                data=self.geometry_arrays[pop];rows=axis['axis_group_rows'];functional=self.geometry[pop]['cell_count_not_applicable']
                f.update(group_ids=data['group_ids'][rows].tolist(),members=None if functional else data['counts'][rows].astype(int).tolist(),
                    centroid_mm=data['centroid_mm'][rows].tolist(),coordinate_role=axis['coordinate_role'],cell_count_not_applicable=functional,
                    state_semantics=self.geometry[pop]['state_semantics'])
        self._create_streams()
    def _create_streams(self):
        width=sum(f['count']*8 for f in self.fields);buffer_rows=max(1,min(64,self.contract['buffer_limit_bytes']//max(width*2,1)))
        expected=width*self.n*2+sum(self.assembly.populations[p].group_count*16*self.contract['rate_bins'] for p in self.rgc)
        free=shutil.disk_usage(self.root).free
        if free<expected+self.contract['disk_remaining_reserve_bytes']:raise ValueError('Insufficient real disk for actual aligned schema; no fields dropped')
        self.layout=dict(bytes_per_point=width,primary_and_post_data_bytes=width*self.n*2,expected_total_bytes=expected,buffer_rows_per_phase=buffer_rows,maximum_value_buffer_bytes=width*buffer_rows*2,free_disk_at_first_snapshot=free)
        self.streams={name:PhaseStream(self.root,name,self.fields,self.n,buffer_rows) for name in ['primary','post']}
        for pop in self.rgc:
            n=self.assembly.populations[pop].group_count
            self.bins[pop]={name:NPYRows(self.root/'rate_bins'/pop/(name+'.npy'),(self.contract['rate_bins'],n),'<f8' if name=='expected_rate_base_Hz' else '<i8') for name in ['expected_rate_base_Hz','sampled_group_counts_base']}
            self.bin_rows[pop]=dict(pre=0,post=0)
        write(self.root/'LAYOUT.json',dict(fields=self.fields,excluded_ports=self.excluded_ports,**self.layout))
    def resume_from(self,model_time):
        """Original arm already restored; observer alone imports verified prefix."""
        if self.error:return
        try:
            tick=int(round((model_time-self.start)/self.dt));name=str(tick).zfill(7)+'.json'
            candidates=[]
            for other in sorted(self.root.parent.iterdir()):
                marker=other/'checkpoints'/name
                if other==self.root or not marker.exists():continue
                record=json.loads(marker.read_text())
                if record['contract_sha256']==digest(self.root/'recording_contract.json'):candidates.append((other,record))
            if not candidates:raise ValueError('No same-contract aligned checkpoint prefix; legitimate one-case replay required')
            old,record=candidates[-1]
            if digest(old/'LAYOUT.json')!=record['layout_sha256']:raise ValueError('Recovery layout SHA differs')
            layout=json.loads((old/'LAYOUT.json').read_text());self.fields=layout['fields'];self.excluded_ports=layout['excluded_ports']
            if any(f['id']!='f'+str(i).zfill(4) for i,f in enumerate(self.fields)):raise ValueError('Recovery field IDs differ')
            frozen={tuple(f['key']):(f['count'],f['unit']) for f in self.fields if f['kind']=='states'}
            if frozen!={k:(s['count'],s['unit']) for k,s in self.state_specs.items()}:raise ValueError('Recovery state schema differs')
            if record['source_tick']!=tick or abs(record['model_time_s']-model_time)>1e-12:raise ValueError('Recovery watermark differs')
            self._create_streams()
            for stream in self.streams.values():
                for writer in list(stream.values.values())+list(stream.indices.values()):
                    rel=writer.path.relative_to(self.root).as_posix();receipt=record['files'][rel]
                    if receipt['rows_written']!=tick+1:raise ValueError('Accepted prefix does not end at complete pre/post row')
                    copy_checked_prefix(old/rel,writer,receipt)
                stream.row=tick+1
            bins=min(tick//10+1,self.contract['rate_bins'])
            for pop,rows in self.bins.items():
                for writer in rows.values():
                    rel=writer.path.relative_to(self.root).as_posix();receipt=record['files'][rel]
                    if receipt['rows_written']!=bins:raise ValueError('Accepted1ms bin prefix differs')
                    copy_checked_prefix(old/rel,writer,receipt)
                self.bin_rows[pop]=dict(pre=bins,post=bins)
            extra=record.get('extra_events')
            if extra:
                data=(old/'extra_event_index.jsonl').read_bytes()[:extra['bytes']]
                if hashlib.sha256(data).hexdigest()!=extra['sha256']:raise ValueError('Extra event prefix differs')
                (self.root/'extra_event_index.jsonl').write_bytes(data);self.extra_event_points=extra['points']
            write(self.root/'PREFIX_IMPORT.json',dict(source=str(old),source_tick=tick,source_model_time_s=model_time,all_prefix_SHA_verified=True,
                old_attempt_files_unchanged=True,rows_copied=tick+1,next_source_tick=tick+1,no_repeated_or_invented_rows=True))
        except Exception as e:self.fail(e)
    def observe(self,snapshot):
        if self.error:return
        started=perf_counter()
        try:
            t=float(snapshot['time_seconds'])
            if self.last_t is None or t!=self.last_t:self.ordinal=0;self.last_t=t
            else:self.ordinal+=1
            if self.ordinal>1:raise ValueError('Original pre/post observation pairing changed')
            if self.canonical_points is None:raise ValueError('Original capture canonical calendar is required')
            j=self.canonical_points.get(t);on_grid=j is not None
            phase=('initial-precommit' if self.ordinal==0 else 'initial-postcommit') if j==0 and on_grid else ('pre-commit' if self.ordinal==0 else 'post-commit' if on_grid else 'post-observation')
            # Fractional150Hz boundaries are actual events, not extra uniform rows.
            if not on_grid:
                with (self.root/'extra_event_index.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(model_time_s=t,time_source_s=t-self.start,side=snapshot['side'],phase=phase,event_generation=snapshot['event_generation'],state_values_not_regridded=True))+'\n')
                self.extra_event_points+=1;return
            if not 0<=j<self.n:raise ValueError('Observation escaped frozen source grid')
            if not self.streams:
                if j!=0 or self.ordinal!=0:raise ValueError('Missing aligned prefix: legitimate complete one-case replay required; no sparse interpolation')
                self._initialize(snapshot)
            which='primary' if self.ordinal==0 else 'post';self.streams[which].append(j,snapshot,phase,self.dt)
            if j%10==0 and j<self.n-1:
                bin_index=j//10;when='pre' if self.ordinal==0 else 'post'
                for pop in self.rgc:
                    if self.bin_rows[pop][when]!=bin_index:raise ValueError('Native1ms bin skipped or repeated')
                    value=snapshot['ports'][pop,'expected_rate_Hz'] if when=='pre' else snapshot['aux']['spike_states'][pop]['last_bin_counts']
                    key='expected_rate_base_Hz' if when=='pre' else 'sampled_group_counts_base'
                    self.bins[pop][key].append(np.asarray(value)[None,:]);self.bin_rows[pop][when]+=1
            self.observations+=1
        except Exception as e:self.fail(e)
        finally:self.copy_wall+=perf_counter()-started
    def fail(self,error):
        if self.error:return
        self.error=dict(type=type(error).__name__,reason=str(error),source_rows={k:v.row for k,v in self.streams.items()})
        write(self.root/'OBSERVER_FAILURE.json',self.error)
    def checkpoint(self,model_time=None):
        started=perf_counter()
        try:
            if not self.error:
                for stream in self.streams.values():stream.flush()
            write(self.root/'PROGRESS.json',dict(status='FAILED' if self.error else 'WRITING_NOT_UI_READY',rows={k:v.row for k,v in self.streams.items()},error=self.error))
            if model_time is not None and self.streams and not self.error:
                tick=int(round((model_time-self.start)/self.dt));writers=[w for s in self.streams.values() for w in list(s.values.values())+list(s.indices.values())]+[w for row in self.bins.values() for w in row.values()]
                if any(s.row!=tick+1 for s in self.streams.values()):raise ValueError('Checkpoint observer watermark is not complete pre/post')
                extra=self.root/'extra_event_index.jsonl';extra_info=None if not extra.exists() else dict(bytes=extra.stat().st_size,sha256=digest(extra),points=self.extra_event_points)
                folder=self.root/'checkpoints';folder.mkdir(exist_ok=True)
                write(folder/(str(tick).zfill(7)+'.json'),dict(source_tick=tick,model_time_s=model_time,contract_sha256=digest(self.root/'recording_contract.json'),
                    layout_sha256=digest(self.root/'LAYOUT.json'),files={w.path.relative_to(self.root).as_posix():w.receipt(self.root) for w in writers},extra_events=extra_info))
        except Exception as e:self.fail(e)
        self.flush_wall+=perf_counter()-started
    def close(self):
        if self.closed:return
        self.checkpoint();self.closed=True
    def publish(self,cache_directory):
        self.close()
        if self.error or len(self.streams)!=2 or any(s.row!=self.n for s in self.streams.values()):
            write(self.root/'STATUS.json',dict(status='INCOMPLETE_REQUIRES_LEGITIMATE_REPLAY',error=self.error,rows={k:v.row for k,v in self.streams.items()}));return False
        cache=Path(cache_directory);manifest=json.loads((cache/'manifest.json').read_text());reference=[];synthetic=self.contract.get('synthetic_test_only',False)
        if not synthetic and (manifest.get('schema')!='rgc-stage1-fixed-endogenous-cache-1' or manifest.get('status')!='COMPLETE_FIXED_ENDOGENOUS_CACHE'
                or manifest.get('cache_identity')!=self.contract['provenance']['cache_identity'] or manifest.get('units')!=dict(V='V',gE='S',gI='S') or not manifest.get('initial_match',{}).get('passed')):
            raise ValueError('Real original CPU cache identity/unit/initial-state receipt required')
        if digest(cache/'endogenous.npz')!=manifest['arrays_sha256']:raise ValueError('Original cache payload SHA differs')
        with np.load(cache/'endogenous.npz',allow_pickle=False) as data:
            if not np.array_equal(data['time_source_s'],np.arange(self.n)*self.dt):raise ValueError('Original source clock differs')
            for pop,selection in self.contract['canonical_RGC_recording'].items():
                for field,legacy in [('voltage','V'),('gE','gE'),('gI','gI')]:
                    f=next(x for x in self.fields if x['kind']=='ports' and x['key']==[pop,field]);actual=np.load(self.root/'primary/values'/(f['id']+'.npy'),mmap_mode='r')
                    ref=data[pop+':'+legacy]
                    same=all(np.array_equal(actual[i:i+512,selection['rows']],ref[i:i+512]) for i in range(0,self.n,512));del ref
                    reference.append(dict(population=pop,field=field,bitwise_equal=same));del actual
            baseline=np.load(self.root/'rate_bins/parasol_RGC_ON/expected_rate_base_Hz.npy',mmap_mode='r')
            if not np.array_equal(baseline[:,self.contract['canonical_RGC_recording']['parasol_RGC_ON']['rows']],data['expected_rate_base_Hz']):raise ValueError('Original1ms expected output differs')
        if not all(x['bitwise_equal'] for x in reference):raise ValueError('Original canonical RGC values differ; aligned product cannot be qualified')
        result=dict(schema=SCHEMA,status='SYNTHETIC_MICROCHECK_COMPLETE' if synthetic else 'COMPLETE_ALIGNED',synthetic_test_only=synthetic,provenance=dict(self.contract['provenance'],source_endogenous_sha256=manifest['arrays_sha256']),
            shared_point_index=self.streams['primary'].manifest()['shared_point_index'],signals=self.streams['primary'].manifest()['signals'],
            post_commit_product=self.streams['post'].manifest(),geometry=dict(path='geometry.json',sha256=digest(self.root/'geometry.json')),
            rate_bin_product_separate=dict(clock_s=.001,bins=self.contract['rate_bins'],time_source_s=(np.arange(self.contract['rate_bins'])*.001).tolist(),
                semantics='pre-event expectation for[start,start+1ms); sampledcounts are baseline sampled group objects, not fitted or independent fine-cell spikes',
                populations={p:{k:v.receipt(self.root) for k,v in rows.items()} for p,rows in self.bins.items()},fitted_output_not_yet_exported=True),
            legacy_reference_check=dict(passed=True,RGC_ports=reference,baseline1ms_expected_exact=True),layout=self.layout,extra_event_points=self.extra_event_points,
            recording_contract_sha256=digest(self.root/'recording_contract.json'),observer_seconds=dict(copy_and_binned_IO=self.copy_wall,checkpoint_flush=self.flush_wall),
            time_sampling='actual original snapshot at solver/event nodes; native values may use existing accepted dense representation; no checkpoint interpolation',
            model_or_solver_changed=False,group_state_not_fine_cell_state=True,validation_status='read actual fit result separately; recording is not fitting or physiological validation')
        write(self.root/'manifest.json',result);write(self.root/'STATUS.json',dict(status=result['status'],manifest_sha256=digest(self.root/'manifest.json')));return True

def observing_arm(base,factory):
    """Delegate every original operation unchanged; copy only returned witnesses."""
    class ObservedArm(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.aligned_observer=factory(self)
        def snapshot(self,*args,**kwargs):
            result=super().snapshot(*args,**kwargs)
            # initial_match/initial_snapshot are diagnostic witnesses with
            # queue_node=False, not canonical capture rows. Never infer phase
            # from every snapshot invocation at the same model time.
            if kwargs.get('queue_node',False):self.aligned_observer.observe(result)
            return result
        def checkpoint_payload(self,*args,**kwargs):
            self.aligned_observer.checkpoint(self.time_seconds);return super().checkpoint_payload(*args,**kwargs)
        @classmethod
        def from_checkpoint(cls,*args,**kwargs):
            result=super().from_checkpoint(*args,**kwargs);result.aligned_observer.resume_from(result.time_seconds);return result
        def close(self):
            try:self.aligned_observer.close()
            finally:super().close()
    return ObservedArm
