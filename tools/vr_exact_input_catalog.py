"""Frozen raw/optical definition with sealed, lazily loaded exact frame captures.

The definition binds real source bytes and the unchanged producer/reference.
It does NOT certify uncomputed outputs. Each forward block separately requires
and records the actual prepared-output receipts; missing frames are an error.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
    return h.hexdigest()


class ExactInputCatalog:
    def __init__(self,directory,*,raw_origin):
        from retina_sim.contracts import ContractError,canonical_hash
        from tools.run_vr_useful_split import require_frontend
        from tools.vr_teacher_handoff_v1.fit_integration_v1.capture_preparation_v2.handoff import approved_policy
        from tools.vr_teacher_handoff_v1.fit_integration_v1.pilot import Gray8Chunks
        self.directory=Path(directory).resolve();self.raw_origin=float(raw_origin)
        self.budget,self.frontend=require_frontend(self.directory)
        self.approval=json.loads((self.directory/'input_policy_approval.json').read_bytes())
        self.handoff,self.objects=approved_policy(ROOT,self.approval)
        self.received=self.objects['received_assets'];intake=Path(self.received['local_intake_root'])
        index_path=intake/self.received['stimulus_index']['path']
        if sha(index_path)!=self.received['stimulus_index']['sha256']:raise ContractError('Frozen raw input index changed')
        index=json.loads(index_path.read_bytes());raw_proofs=[]
        for item in index['chunks']:
            if item['start']>=9751:break
            path=(intake/item['path']).resolve()
            if not path.is_relative_to(intake) or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
                raise ContractError('Actual raw gray8 source content changed')
            raw_proofs.append({k:item[k] for k in ('start','stop','sha256','bytes')})
        self.reader=Gray8Chunks(intake,index)
        self.edges=self.raw_origin+np.arange(9752,dtype=np.float64)/150.
        ref=json.loads((self.directory/'actual_global_reference.json').read_bytes())
        table=json.loads((ROOT/'data/atlas/human_disk3mm_typed_tables_v1/cones/manifest.json').read_bytes())
        spec=table['extras']['subtype'];type_path=ROOT/spec['path']
        if sha(type_path)!=spec['sha256']:raise ContractError('Frozen global cone types changed')
        types=np.load(type_path,allow_pickle=False,mmap_mode='r')
        self.alpha=np.asarray(ref['q_ref_LMS_rstar_s'])[[{'L':0,'M':1,'S':2}[t.decode('ascii')] for t in types]]/100.
        self.background=np.load(self.directory/'linear_inputs/cone_background_native_td.npy',allow_pickle=False,mmap_mode='r')
        sources=['tools/export_vr_linear_disk_capture.m','tools/export_vr_linear_disk_batch.m',
            'tools/prepare_vr_optical_extension.py','tools/run_vr_optical_extension.py',
            'tools/vr_teacher_handoff_v1/fit_integration_v1/capture_preparation_v2/make_gray_reference_scene.m',
            'tools/vr_teacher_handoff_v1/fit_integration_v1/capture_preparation_v2/linear_gray.py']
        self.definition=dict(schema='exact-input-definition-and-sealed-evaluations-1',
            prepared_content_scope='Verified actual raw bytes plus frozen pure optical/conversion definition; evaluated array receipts separately mandatory for every forward block',
            raw_frame_range=[0,9751],raw_origin_seconds=self.raw_origin,frame_rate_Hz=150,
            raw_index_sha256=self.received['stimulus_index']['sha256'],verified_raw_files=raw_proofs,
            policy_sha256=self.approval['policy_sha256'],prefix_frontend_sha256=canonical_hash(self.frontend),
            numeric_global_reference_sha256=self.frontend['global_reference_numeric_sha256'],cone_types_sha256=spec['sha256'],
            optical_sources_sha256={p:sha(ROOT/p) for p in sources},provider_sha256=sha(__file__),
            native_conversion='counts / (1/150) / (q_ref_type/100), FP64 original expression order',
            unknown_future_arrays_are_not_certified=True,no_missing_input_extrapolation=True,
            optical_noise='none',eye_movements='none',physical_calibration_claim=False)
        self.definition_hash=canonical_hash(self.definition)
        self.records=[];self.locator={};self._arrays={};self._last_native=None
        self.refresh()

    def _arrays_for(self,record):
        key=record['path']
        if key not in self._arrays:
            # Two mapped bases, never all movie arrays or global state history.
            if len(self._arrays)>=2:self._arrays.clear()
            folder=Path(key).parent
            basis=np.load(folder/'unique_cone_expected_counts.npy',allow_pickle=False,mmap_mode='r')
            mapping=np.load(folder/'frame_to_unique.npy',allow_pickle=False,mmap_mode='r')
            if (basis.dtype!=np.float64 or basis.ndim!=2 or basis.shape[1]!=len(self.alpha)
                    or mapping.dtype!=np.int32 or len(mapping)!=record['stop']-record['start']
                    or np.any(mapping<0) or np.any(mapping>=len(basis))):
                from retina_sim.contracts import ContractError
                raise ContractError('Prepared exact frame basis layout changed')
            self._arrays[key]=(basis,mapping)
        return self._arrays[key]

    def vector(self,index):
        from retina_sim.contracts import ContractError
        if index not in self.locator:raise ContractError('ACTUAL_PREPARED_OPTICAL_FRAME_MISSING: '+str(index))
        record=self.locator[index];basis,mapping=self._arrays_for(record)
        return basis[int(mapping[index-record['start']])]

    def refresh(self):
        from retina_sim.contracts import ContractError,canonical_hash
        known={r['path']:r for r in self.records}
        for path in sorted((self.directory/'optical_extensions').glob('frames-*/frontend_receipt.json')):
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();key=str(path.resolve())
            if key in known:
                if known[key]['receipt_sha256']!=digest:raise ContractError('Committed input receipt changed')
                continue
            receipt=json.loads(raw)
            if (receipt['schema']!='actual-vr-optical-exact-frame-basis-1'
                    or receipt['input_policy_sha256']!=self.approval['policy_sha256']
                    or receipt['raw_index_sha256']!=self.received['stimulus_index']['sha256']
                    or receipt['prefix_frontend_sha256']!=canonical_hash(self.frontend)
                    or receipt['reference_global_sha256']!=self.frontend['global_reference_numeric_sha256']):
                raise ContractError('Prepared packet differs from frozen actual input definition')
            for name,expected in receipt['files'].items():
                file=path.parent/name
                if file.name!=name or sha(file)!=expected:raise ContractError('Prepared packet array content changed')
            start,stop=receipt['raw_frame_range']
            if not 0<=start<stop<=9751 or len(receipt['frame_sha256'])!=stop-start:
                raise ContractError('Prepared frame source range is invalid')
            record=dict(path=key,receipt_sha256=digest,start=start,stop=stop,receipt=receipt)
            self._arrays_for(record)
            basis,mapping=self._arrays_for(record)
            for i in range(start,stop):
                if i in self.locator:
                    if not np.array_equal(self.vector(i),basis[int(mapping[i-start])]):
                        raise ContractError('Overlapping exact captures differ; do not replace past input')
                else:self.locator[i]=record
            self.records.append(record)
        end=0
        while end in self.locator:end+=1
        self.available_stop_frame=end

    def native(self,t,port,evidence):
        from retina_sim.contracts import Quantity,ContractError
        if t<self.raw_origin:return Quantity(self.background,'td',port.meaning,evidence)
        index=int(np.searchsorted(self.edges,t,side='right')-1)
        if not 0<=index<9751:raise ContractError('Native raw input exhausted; no artificial guard')
        if self._last_native is None or self._last_native[0]!=index:
            self._last_native=(index,Quantity(np.asarray(self.vector(index))/(1/150.)/self.alpha,
                'td',port.meaning,evidence))
        return self._last_native[1]

    def require_block(self,start,stop):
        from retina_sim.contracts import ContractError,canonical_hash
        self.refresh()
        first=max(0,int(np.searchsorted(self.edges,start,side='right')-1))
        last=min(9750,int(np.searchsorted(self.edges,stop,side='right')-1))
        records={}
        for i in range(first,last+1):
            if i not in self.locator:raise ContractError('ACTUAL_PREPARED_OPTICAL_FRAME_MISSING: '+str(i))
            r=self.locator[i];records[r['path']]=r
        proofs=[]
        for r in records.values():
            path=Path(r['path'])
            if sha(path)!=r['receipt_sha256']:raise ContractError('Prepared receipt changed before forward')
            for name,digest in r['receipt']['files'].items():
                if sha(path.parent/name)!=digest:raise ContractError('Prepared data changed before forward')
            proofs.append(dict(path=str(path),receipt_sha256=r['receipt_sha256'],files=r['receipt']['files']))
        proof=dict(definition_sha256=self.definition_hash,required_frames=[first,last+1],prepared_output_receipts=proofs)
        return dict(proof,proof_sha256=canonical_hash(proof))
