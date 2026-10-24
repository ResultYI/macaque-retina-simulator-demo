"""Lossless reuse of a completed reference main trajectory, never a QA verdict.

All scientific prospective fields must match. Main-only reuse requires a fresh
trajectory to reproduce it exactly. Complete-arm reuse instead restores the
authenticated original codec export and reproduces its full export point
exactly. Every original reference witness remains fully compared to a freshly
executed candidate; a previous incomplete overall verdict is never reused.
"""
from pathlib import Path
import ast
import hashlib
import json


def archived_input_loop(source,expected_sha,namespace):
    """Execute the verified original input constructor, with its true identity."""
    path=Path(source);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('Archived input constructor changed')
    tree=ast.parse(raw)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loops=[n for n in ast.walk(main) if isinstance(n,ast.For) and
           any(isinstance(x,ast.FunctionDef) and x.name=='evaluate' for x in n.body)]
    if len(loops)!=1:raise ValueError('Unreviewed archived input constructor shape')
    namespace=dict(namespace,source_sha=expected_sha)
    exec(compile(ast.Module(body=loops,type_ignores=[]),str(path),'exec'),namespace)


class ReferenceMainEvidence:
    def __init__(self,directory,plan,guard,*,complete_arm=False,candidate_revision=False):
        from .resident_qa_harness import WitnessSpool,_file_hash
        from .resident_qa import plan_digest
        self.directory=Path(directory)
        self.complete_arm=complete_arm
        self.reader=WitnessSpool.open_existing(directory,guard=guard)
        producer=json.loads((self.directory.parent.parent/(self.directory.parent.name+'_supervisor')/'supervisor.json').read_bytes())
        source_dir=self.directory.parent.parent/(self.directory.parent.name+'_supervisor')/'sources'
        for name,digest in producer['source_sha256'].items():
            if _file_hash(source_dir/Path(name).name)!=digest:
                raise ValueError('Archived reference producer snapshot changed: '+name)
        with self.reader.read('prospective_plan') as old:
            if plan_digest(old)!=old['plan_sha256']:raise ValueError('Archived prospective plan changed')
            def scientific(value):
                result={k:v for k,v in value.items() if k not in ('qa_implementation_sha256','plan_sha256')}
                if candidate_revision:
                    # A stored reference does not depend on the implementation
                    # of a later candidate. Its physical binding and identity,
                    # all inputs, initial vectors and every budget remain exact.
                    result=dict(result,arm_identity_hashes=dict(reference=result['arm_identity_hashes']['reference']))
                    result.pop('candidate_native_sparse_plan',None)
                return result
            if plan_digest(scientific(old))!=plan_digest(scientific(plan)):
                raise ValueError('Scientific plan differs; reference evidence cannot be reused')
            self.old_plan_sha=old['plan_sha256']
        self.keys=set()
        for category,prefix in [('points','point'),('intervals','solver'),('draws','draw')]:
            self.keys.update(prefix+'_'+row['id'] for row in plan[category])
        self.keys.update(('monitor_full_window','ordinary_recording','ownership_dense_and_input'))
        if complete_arm:
            self.keys.add('checkpoint_runtime_roundtrip')
            self.keys.update('failure_'+row['id'] for row in plan['failure_cases'])
            with self.reader.read('checkpoint_runtime_roundtrip') as checkpoint:
                self.export_key=checkpoint['export']['__point_record__']
                for label in ('export','restored','uninterrupted','continued'):
                    ref=checkpoint[label];key=ref['__point_record__']
                    if Path(key).name!=key or _file_hash(self.directory/(key+'.json'))!=ref['sha256']:
                        raise ValueError('Archived complete checkpoint point identity differs')
                    self.keys.add(key)
            raw_key='runtime_export_'+plan['arm_identity_hashes']['reference'][:20]+'_receipt'
            self.keys.add(raw_key)
            with self.reader.read(raw_key) as receipt:
                name=receipt['file'];self.raw_path=self.directory/name
                if (Path(name).name!=name or receipt['actual_complete_codec_output'] is not True or
                        receipt['fit_eligible'] is not False or self.raw_path.stat().st_size!=receipt['size'] or
                        _file_hash(self.raw_path)!=receipt['sha256']):
                    raise ValueError('Authentic complete reference codec export is missing or changed')
                self.raw_sha=receipt['sha256']
        self.inventory={key:_file_hash(self.directory/(key+'.json')) for key in sorted(self.keys)}
        self.provenance=dict(producer_plan_sha256=self.old_plan_sha,
            producer_source_sha256=producer['source_sha256'],
            metadata_inventory_sha256=hashlib.sha256(json.dumps(self.inventory,sort_keys=True).encode()).hexdigest(),
            directory=str(self.directory.resolve()),
            reuse='Main evidence only; fresh exact endpoint/full-report/recording/RNG/ownership proof, fresh checkpoint and failure tests')
        if complete_arm:
            self.provenance.update(raw_checkpoint_sha256=self.raw_sha,
                reuse='Complete genuine reference arm including original checkpoint/continuations/failures; restore original raw export and require exact export-point reproduction. Candidate is executed fresh and all original gates recomputed.')
        if candidate_revision:
            self.provenance['new_candidate_identity_prospectively_bound']=plan['arm_identity_hashes']['candidate']

    def restore_reference(self,runner):
        from .resident_qa_harness import _file_hash
        if not self.complete_arm or _file_hash(self.raw_path)!=self.raw_sha:
            raise ValueError('Authentic reference checkpoint changed before restore')
        # Original loader verifies payload, full state layout, parameter/graph/
        # backend identities and all its original validation. No reconstruction
        # from point witnesses, relabelling or checkpoint migration.
        return runner.restore(self.raw_path.read_text(encoding='utf8'))

    def read(self,key):
        from .resident_qa_harness import _file_hash
        if key not in self.inventory or _file_hash(self.directory/(key+'.json'))!=self.inventory[key]:
            raise ValueError('Immutable reference metadata changed or missing: '+key)
        return self.reader.read(key)

    def assert_exact(self,key,value):
        import numpy as np
        def same(a,b,path):
            if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
                a,b=np.asarray(a),np.asarray(b)
                if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():
                    raise ValueError('Refreshed reference array differs exactly: '+path)
            elif isinstance(a,dict) and isinstance(b,dict):
                if set(a)!=set(b):raise ValueError('Refreshed reference schema differs: '+path)
                for k in a:same(a[k],b[k],path+'.'+str(k))
            elif isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
                if len(a)!=len(b):raise ValueError('Refreshed reference sequence differs: '+path)
                for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'.'+str(i))
            elif json.dumps(a,sort_keys=True,allow_nan=False)!=json.dumps(b,sort_keys=True,allow_nan=False):
                raise ValueError('Refreshed reference exact field differs: '+path)
        with self.read(key) as old:same(old,value,key)
