"""Prepare eccentric H1 assets from existing geometry; no neuron or optics run.

Execute explicitly after code review. Existing output is never overwritten.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import time
import traceback

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from retina_sim.contracts import ContractError
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.hc_eccentric import (checked_path,digest,spatial_parameters,
                                     temporal_decision,precompute_crop,project_members)
from retina_sim.hc_group_projection import _normalize_complete_rows,_member_mean
from retina_sim.h1_h2_effective import matrix_id
from tools.run_hc_minimal_real_diagnostic_v1 import peak_memory_bytes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = 'configurations/research/cone_hc_eccentric.json'


def write(path, value):
    def serial(v):
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,np.generic): return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=serial)+'\n',encoding='utf-8')


def prepare(root, configuration=DEFAULT):
    from tools.gin_hc2_bc1_context_support_v2 import load
    root = Path(root).resolve()
    config_path = checked_path(root,configuration)
    cfg = json.loads(config_path.read_bytes())
    if cfg.get('schema') != 'cone_hc_eccentric_research' or cfg.get('source_context') != 'GIN_69_6_HC2_BC1_SUPPORT_V2':
        raise ContractError('Explicit eccentric HC source context required')
    out = (root/cfg['assets_directory']).resolve()
    if not out.is_relative_to(root): raise ContractError('HC output must stay in workspace')
    out.mkdir(parents=True,exist_ok=False)
    clock=time.perf_counter()
    receipt=dict(status='PREPARING',started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'SPATIAL_RECEIPT.json',receipt)
    try:
        calibration=temporal_decision(root,cfg['calibration'])
        spatial_path=checked_path(root,cfg['spatial_configuration'])
        spatial=json.loads(spatial_path.read_bytes())
        assembly,seed=load()
        if assembly.repo_root.resolve()!=root: raise ContractError('Source context root differs')
        cp,hp=assembly.populations['cones'],assembly.populations['HC']
        ci,hi=cp.member_global_rows,hp.member_global_rows
        cm,hm=cp.fine_to_group[ci],hp.fine_to_group[hi]
        native_dir=root/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/executions/assemble_native_01'
        native_receipt_path=native_dir/'NATIVE_ASSEMBLY_RESULT.json'
        original=json.loads(native_receipt_path.read_bytes())
        cones=FrozenDiskTable(root,original['cone_table']);hcs=FrozenDiskTable(root,original['hc_table'])
        ct=cones.extras['subtype'].astype('U1');ht=hcs.extras['subtype'].astype('U2')
        anchor=hcs.coordinates['soma'][hi]
        parameters,parameter_record,spatial_identity=spatial_parameters(root,anchor,ht[hi],spatial)
        raw_path=native_dir/'RAW_HC_BY_CONE.npz'
        raw=sparse.load_npz(raw_path).tocsr()
        if raw.shape!=(len(ht),len(ct)): raise ContractError('Raw anatomical matrix/table layout differs')
        edges=raw.tocoo()
        if np.any((ht[edges.row]=='H1')&(ct[edges.col]=='S')):
            raise ContractError('Anatomical H1 input contains S cones')
        uniform,_=_normalize_complete_rows(raw)
        allowed=((ht[edges.row]=='H1')&(ct[edges.col]!='S'))|((ht[edges.row]=='H2')&(ct[edges.col]=='S'))
        fullB,_=_normalize_complete_rows(sparse.csr_matrix(
            (edges.data[allowed],(edges.col[allowed],edges.row[allowed])),shape=raw.shape[::-1]))
        fine_uniform=uniform[hi][:,ci].tocsr()
        outside_cones=np.ones(len(ct),bool);outside_cones[ci]=False
        uniform_held=np.asarray(uniform[hi][:,outside_cones].sum(1)).ravel()
        outside_hc=np.ones(len(ht),bool);outside_hc[hi]=False
        fineB=fullB[ci][:,hi].tocsr()
        fineBheld=np.asarray(fullB[ci][:,outside_hc].sum(1)).ravel()
        fineBsupported=np.asarray(fullB[ci].sum(1)).ravel()
        B=_member_mean(fineB,cm,hm,cp.counts,(cp.group_count,hp.group_count))
        Bheld=np.bincount(cm,weights=fineBheld,minlength=cp.group_count)/cp.counts
        Bsupported=np.bincount(cm,weights=fineBsupported,minlength=cp.group_count)/cp.counts
        originalB=assembly.matrix('routes/HC_to_cones/mean_weights.npz')
        if np.max(abs((B-originalB).data),initial=0.)>2e-14:
            raise ContractError('Reused member-projected local B differs from source')
        B=originalB.copy().tocsr()

        # The previously verified typed source envelope already covers the short
        # central scaled kernels. Missing coverage stops preparation; no new labels.
        legacy_path=checked_path(root,cfg['diagnostic_input_configuration'])
        legacy=json.loads(legacy_path.read_bytes())
        old_receipt_path=checked_path(root,**{'relative':legacy['assets']['spatial_receipt']['path'],
                                           'sha256':legacy['assets']['spatial_receipt']['sha256']})
        old_receipt=json.loads(old_receipt_path.read_bytes())
        source_inputs={name:expected for name,expected in old_receipt['source_files'].items()
                       if name.endswith(('SOURCE_LABELS.npz','cone_pedicle_mm.npy','official_source_rows.npy','RESULT.json'))}
        for name,expected in source_inputs.items(): checked_path(root,name,expected)
        labels_path=next(checked_path(root,p) for p in source_inputs if p.endswith('SOURCE_LABELS.npz'))
        labels=np.load(labels_path,allow_pickle=False)
        source=root/'reports/CONE_HC_OUTER_SOURCE_EXTENSION_20261102_v3/cones_01'
        source_record=json.loads((source/'RESULT.json').read_bytes())
        is_path=source/'cone_inner_segment_mm.npy'
        checked_path(root,is_path,source_record['files'][is_path.name])
        source_inputs[is_path.relative_to(root).as_posix()]=source_record['files'][is_path.name]
        isxy=np.load(is_path,mmap_mode='r',allow_pickle=False)
        xy=np.load(source/'cone_pedicle_mm.npy',mmap_mode='r',allow_pickle=False)
        official=np.load(source/'official_source_rows.npy',mmap_mode='r',allow_pickle=False)
        selected=labels['cone_candidate_rows'];types=labels['cone_types']
        center=np.asarray(old_receipt['source_envelope_center_mm'])
        envelope=old_receipt['source_envelope_radius_mm']
        known=np.asarray(cKDTree(xy).query_ball_point(center,envelope,return_sorted=True),dtype=np.int64)
        if not np.array_equal(np.sort(selected),known):
            raise ContractError('Existing typed source envelope is incomplete')
        official_dynamic=np.array([int(k.decode('ascii').split(':')[-1]) for k in cones.keys[ci]])
        if np.any(np.diff(official)<=0): raise ContractError('Official source lookup is not unique/sorted')
        candidate=np.searchsorted(official,official_dynamic)
        if np.any(candidate>=len(official)) or not np.array_equal(official[candidate],official_dynamic):
            raise ContractError('Actual dynamic fine members absent from source')
        lookup=np.full(len(xy),-1,dtype=np.int64);lookup[selected]=np.arange(len(selected))
        dynamic=lookup[candidate]
        if (np.any(dynamic<0) or not np.array_equal(types[dynamic],ct[ci])
                or not np.array_equal(xy[candidate],cones.coordinates['opl_pedicle'][ci])):
            raise ContractError('Source type/pedicle/member identity differs')
        result=precompute_crop(xy[selected],types,anchor,ht[hi],dynamic,fine_uniform,uniform_held,
            parameters=parameters,envelope_center=center,envelope_radius_mm=envelope)
        A,Aheld=project_members(result['A'],result['held_mass'],cm,hm,cp.counts,hp.counts)
        oldA=assembly.matrix('routes/cones_to_HC/mean_weights.npz')
        h2=assembly.h2_rows
        if np.max(abs((A[h2]-oldA[h2]).data),initial=0.)>2e-14:
            raise ContractError('H2 input changed during H1 preparation')
        h1_mask=assembly.types['HC']=='H1'
        A=(sparse.diags(h1_mask.astype(float))@A+sparse.diags((~h1_mask).astype(float))@oldA).tocsr()
        parent_native=assembly.runtime_factories['native'](assembly,bound=False)
        if parent_native.S_efficacy_ratio!=1.:
            raise ContractError('Unexpected H2 efficacy; cannot silently replace it')
        Q2=parent_native.Q['H2']
        for name,value in [('A_fine_dynamic',result['A']),('B_fine_dynamic',fineB),('A_group',A),('B_group',B),('H2_Q',Q2)]:
            sparse.save_npz(out/(name+'.npz'),value)
        np.savez_compressed(out/'MEMBERS.npz',cone_parent_rows=ci,hc_parent_rows=hi,
            cone_group=cm,hc_group=hm,cone_counts=cp.counts,hc_counts=hp.counts,
            cone_keys=cones.keys[ci],hc_keys=hcs.keys[hi],cone_types=ct[ci],hc_types=ht[hi],
            cone_pedicle_mm=cones.coordinates['opl_pedicle'][ci],cone_IS_mm=isxy[candidate],
            hc_anchor_mm=anchor,**parameters)
        np.savez_compressed(out/'MASS_LEDGER.npz',A_fine_held=result['held_mass'],
            A_component_held=result['component_held_mass'],full_LM_count=result['full_LM_count'],
            A_group_held=Aheld,B_group_held=Bheld,B_group_supported=Bsupported)
        witnesses={}
        for row,value in result['witnesses'].items():
            prefix='row_'+str(row)+'__'
            witnesses.update({prefix+k:v for k,v in value.items()})
            witnesses[prefix+'official_source_rows']=official[selected[value['source_rows']]]
            witnesses[prefix+'cone_pedicle_mm']=xy[selected[value['source_rows']]]
            witnesses[prefix+'cone_types']=types[value['source_rows']]
        witnesses['fine_rows']=np.array(sorted(result['witnesses']),dtype=np.int64)
        np.savez_compressed(out/'H1_COMPLETE_ROW_WITNESSES.npz',**witnesses)
        inputs=[config_path,spatial_path,legacy_path,old_receipt_path,native_receipt_path,raw_path,
            root/cfg['calibration']['path'],root/cfg['bc_configuration'],root/spatial['geometry_profile'],
            Path(__file__),root/'retina_sim/hc_eccentric.py',root/'retina_sim/cone_hc_eccentric.py']
        receipt.update(status='PRECOMPUTED',source_assembly=assembly.artifact_sha256,
            geometry_regenerated=False,optical_recomputed=False,crop_renormalized=False,
            anatomical_contacts_added=0,source_envelope_center_mm=center,source_envelope_radius_mm=envelope,
            fine_cones=len(ci),fine_HC=len(hi),group_cones=cp.group_count,group_HC=hp.group_count,
            A_identity=matrix_id(A),B_identity=matrix_id(B),H2_A_identity=matrix_id(A[h2]),
            H2_Q_identity=matrix_id(Q2),H2_coupling_rate_per_second=parent_native.coupling_rate,
            H2_parameters=dict(parent_native.parameters),H1_explicit_lateral_coupling=False,
            spatial_identity=spatial_identity,spatial_parameters=parameter_record,calibration=calibration,
            source_files={**source_inputs,**{p.relative_to(root).as_posix():digest(p) for p in inputs}},
            output_files={p.name:digest(p) for p in out.glob('*.npz')},
            scientific_qualification='conditional finite held-background model; central spatial calibration absent')
    except Exception as error:
        receipt.update(status='FAILED_RETAINED',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(ended_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-clock,
                       peak_working_set_bytes=peak_memory_bytes())
        write(out/'SPATIAL_RECEIPT.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',default=DEFAULT)
    args=parser.parse_args()
    result=prepare(ROOT,args.configuration)
    print(json.dumps({k:result[k] for k in ('status','wall_seconds','peak_working_set_bytes')}),flush=True)
