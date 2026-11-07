"""Precompute final functional weights from existing typed sources; no dynamics."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import time
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from retina_sim.contracts import ContractError
from retina_sim.hc_biexponential import precompute_functional_crop,support_radius
from retina_sim.h1_h2_effective import matrix_id
from retina_sim.cone_hc_spatial import FUNCTIONAL_KERNEL


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    clock=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    legacy_path=root/'configurations/research/cone_hc_uniform.json'
    old=json.loads(legacy_path.read_bytes())
    config_path=root/'configurations/research/HC_biexponential_functional.json'
    cfg=json.loads(config_path.read_bytes())
    inputs=[legacy_path,config_path,Path(__file__),root/'retina_sim/hc_biexponential.py']
    assets={}
    for name,binding in old['assets'].items():
        p=root/binding['path']
        if sha(p)!=binding['sha256']:raise ContractError('Frozen input changed: '+name)
        assets[name]=p;inputs.append(p)
    g=np.load(assets['geometry'],allow_pickle=False)
    labels_path=assets['geometry'].parent/'SOURCE_LABELS.npz'
    labels=np.load(labels_path,allow_pickle=False)
    source=root/'reports/CONE_HC_OUTER_SOURCE_EXTENSION_20261102_v3/cones_01'
    source_record=json.loads((source/'RESULT.json').read_bytes());inputs += [source/'RESULT.json',labels_path]
    for name in ('cone_pedicle_mm.npy','official_source_rows.npy'):
        if sha(source/name)!=source_record['files'][name]:raise ContractError('Original Cone source changed')
        inputs.append(source/name)
    xy=np.load(source/'cone_pedicle_mm.npy',mmap_mode='r')
    official=np.load(source/'official_source_rows.npy',mmap_mode='r')
    selected=labels['cone_candidate_rows'];types=labels['cone_types']
    target=int(g['target_hc_local']);center=g['hc_anchor_mm'][target]
    radius=support_radius(.300,cfg['tail_mass_budget'])
    envelope=radius+np.linalg.norm(g['hc_anchor_mm']-center,axis=1).max()
    known=np.asarray(cKDTree(xy).query_ball_point(center,envelope,return_sorted=True),dtype=int)
    np.testing.assert_array_equal(np.sort(selected),known)
    if np.linalg.norm(center)+envelope>source_record['IS_radius_mm']:
        raise ContractError('Existing source range cannot certify this local asset')
    dynamic_official=np.array([int(k.decode().split(':')[-1]) for k in g['cone_keys']])
    dynamic_candidate=np.searchsorted(official,dynamic_official)
    np.testing.assert_array_equal(official[dynamic_candidate],dynamic_official)
    lookup=np.full(len(xy),-1,dtype=int);lookup[selected]=np.arange(len(selected))
    dynamic=lookup[dynamic_candidate]
    if np.any(dynamic<0):raise ContractError('Dynamic member outside typed source envelope')
    np.testing.assert_array_equal(xy[dynamic_candidate],g['cone_pedicle_mm'])
    np.testing.assert_array_equal(types[dynamic],g['cone_types'])
    source_receipt=json.loads(assets['source_receipt'].read_bytes())
    hm=json.loads((root/source_receipt['hc_table']['path']).read_bytes())
    hctypes=np.load(root/hm['extras']['subtype']['path'],mmap_mode='r')[g['hc_parent_rows']].astype('U2')
    uniform=sparse.load_npz(assets['A_uniform']).tocsr()
    with np.load(assets['held_mass']) as z:uniform_held=z['uniform']
    operator=precompute_functional_crop(xy[selected],types,g['hc_anchor_mm'],hctypes,dynamic,uniform,uniform_held,
        configuration=cfg,source_envelope_center=center,source_envelope_radius_mm=envelope)
    for name in ('A','short','long'):sparse.save_npz(out/(name+'_dynamic.npz'),operator[name])
    np.savez_compressed(out/'HELD_MASS.npz',uniform=uniform_held,biexponential=operator['held_mass'])
    np.savez_compressed(out/'MASS_LEDGER.npz',component_held_mass=operator['component_held_mass'],
        A_dynamic_mass=np.asarray(operator['A'].sum(1)).ravel(),A_held_mass=operator['held_mass'],
        full_LM_count=operator['full_LM_count'],HC_parent_rows=g['hc_parent_rows'],HC_types=hctypes)
    # Store the complete target row as an independently readable formula witness.
    coords=xy[selected];distance=np.linalg.norm(coords-center,axis=1)
    eligible=(types!='S')&(distance<=radius)
    short=np.zeros(len(selected));long=np.zeros(len(selected))
    short[eligible]=np.exp(-distance[eligible]/.020);short/=short.sum()
    long[eligible]=np.exp(-distance[eligible]/.300);long/=long.sum()
    final=.15*short+.85*long
    np.testing.assert_allclose(operator['A'][target].toarray()[0],final[dynamic],rtol=2e-14,atol=1e-17)
    np.savez_compressed(out/'TARGET_COMPLETE_ROW.npz',A=final,short_unitmass=short,long_unitmass=long,
        source_candidate_rows=selected,official_source_rows=official[selected],cone_pedicle_mm=coords,
        cone_types=types,dynamic_source_rows=dynamic,target_hc_anchor_mm=center,
        target_hc_parent_row=g['hc_parent_rows'][target],distance_mm=distance)
    receipt=dict(status='PRECOMPUTED',started_utc=started,kernel_definition=FUNCTIONAL_KERNEL,
        parameters=old['H1_parameters'],H2_mode='uniform',A_identity=matrix_id(operator['A']),
        B_identity=matrix_id(sparse.load_npz(assets['B_local'])),support_radius_mm=radius,
        source_envelope_radius_mm=envelope,source_envelope_center_mm=center.tolist(),
        support_definition='both components on same existing finite OPL L/M support; no contact mask',
        anatomical_contacts_added=0,crop_renormalized=False,source_geometry_regenerated=False,
        fine_cones=len(dynamic),HC_rows=len(hctypes),configuration=cfg,
        source_files={p.relative_to(root).as_posix():sha(p) for p in inputs},
        wall_seconds=time.perf_counter()-clock,physiological_prediction_qualified=False)
    for name in ('A_dynamic.npz','short_dynamic.npz','long_dynamic.npz','HELD_MASS.npz','MASS_LEDGER.npz','TARGET_COMPLETE_ROW.npz'):
        receipt.setdefault('output_files',{})[name]=sha(out/name)
    (out/'SPATIAL_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    new=dict(old,kernel_definition=FUNCTIONAL_KERNEL,execution_mode='local_cpu',spatial_mode='biexponential')
    new['parameter_source']='User-approved final functional kernel, 2026-11-02; central physiology uncalibrated'
    new['assets']=dict(old['assets'])
    for key,name in [('A_biexponential','A_dynamic.npz'),('held_mass','HELD_MASS.npz'),('spatial_receipt','SPATIAL_RECEIPT.json')]:
        p=out/name;new['assets'][key]=dict(path=p.relative_to(root).as_posix(),sha256=sha(p))
    new['limitations']=dict(old['limitations'],supported_asset='existing complete typed-source envelope; pinned 321-Cone /249-H1 local CPU research asset',
        prior_spatial_failure='Historical contact-masked kernel; limitation retained; final functional kernel not spatially qualified')
    dest=root/'configurations/research/cone_hc_biexponential_functional.json'
    with dest.open('x',encoding='utf-8') as f:json.dump(new,f,indent=2)
    print(json.dumps(dict(status=receipt['status'],wall_seconds=receipt['wall_seconds'],A_nnz=operator['A'].nnz,
                          support_radius_mm=radius,config=dest.relative_to(root).as_posix()),indent=2),flush=True)


if __name__=='__main__':main()
