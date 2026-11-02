"""Install the authorized owner/support separation without changing coefficients."""
from pathlib import Path
from copy import deepcopy
import shutil
import time
import numpy as np
from scipy.optimize import brentq
from retina_sim.spatial_group_graph import digest
from retina_sim.contracts import canonical_hash
from tools.durable_research_runtime_v1 import atomic_json

ROOT=Path(r'D:\macaque retinal simulator')
OUT=ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2'
STAGE=Path(r'C:\Users\win11-pc\Documents\Codex\2026-10-29\task\bc_revision\owner_support_v2')
def read(path):
    import json
    return json.loads(path.read_text(encoding='utf-8-sig'))

def build(output,guard):
    start=time.perf_counter();hc=read(ROOT/'configurations/hc_density_rf_joint_v2.json')
    bc=read(ROOT/'configurations/bc_cone_pedicle_anchored_v1.json')
    source=ROOT/'data/atlas/human_disk3mm_typed_tables_v1/cones/manifest.json';table=read(source)
    xy=np.load(ROOT/table['coordinates']['rf']['path'],mmap_mode='r',allow_pickle=False)
    ped=np.load(ROOT/table['coordinates']['opl_pedicle']['path'],mmap_mode='r',allow_pickle=False)
    own=np.linalg.norm(xy,axis=1)<=1.338;rmax=float(np.linalg.norm(ped[own],axis=1).max())
    c=np.array([v['c'] for v in hc['density']['coefficients']]);lam=np.array([v['lambda_per_mm'] for v in hc['density']['coefficients']])
    old=read(ROOT/'configurations/hc_joint_synthetic_v1.json')['source_records']['EglenWong2008']
    reference=np.array([old['H1_count'],old['H2_count']])/np.prod(old['area_mm'])
    lengths=np.array(old['dmin_mean_um'])*.001
    def envelope(r):
        rho=float(np.exp(r*lam)@c);d1=3.7+3.8*(r-.6)/4.9
        contact=max(np.sqrt(d1/(np.pi*.8*rho)),np.sqrt(2.2/(np.pi*.2*rho)))
        pair=float(np.max(lengths*np.sqrt(np.r_[reference,reference.sum()]/np.array([.8*rho,.2*rho,rho]))))
        return float(contact),pair
    def residual(r):return r-rmax-sum(envelope(r))
    upper=rmax+.01
    while residual(upper)<0:
        upper=rmax+2*(upper-rmax)
        if upper>=table['source_bindings']['fixed_RF_query_radius_mm']:raise ValueError('Finite source support insufficient')
    support=float(np.nextafter(brentq(residual,rmax,upper,xtol=1e-14),np.inf))
    while residual(support)<0:support=float(np.nextafter(support,np.inf))
    contact,pair=envelope(support)
    domain=dict(schema='IS-RF-owner-to-physical-support-1',authorization='Sentinel_5a05022065f081919c81f4f9db2d6697',
        owner_RF_radius_deg=6.,owner_RF_radius_mm=1.338,mm_per_degree=.223,
        cone_source_table=dict(path=source.relative_to(ROOT).as_posix(),sha256=digest(source)),owner_cones=int(own.sum()),
        owner_pedicles_outside_prior_physical_domain=int(np.sum(own&(np.linalg.norm(ped,axis=1)>1.338))),
        BC_anchor_radius_mm=rmax,HC_support_radius_mm=support,HC_outer_contact_margin_mm=contact,HC_outer_pair_margin_mm=pair,
        support_equation='R=max(owner pedicle radius)+max(H1,H2 contact radius atR)+inherited outer pair context atR',
        support_residual_mm=residual(support),original_parameter_application_radius_mm=1.338,
        outer_status='USER_AUTHORIZED_ENGINEERING_EXTRAPOLATION_NOT_NEW_PHYSIOLOGY',
        external_cones='Only existing finite registered source; outside0..6deg are held context, not new optical/dynamic owners',
        infinite_retina_boundary_complete=False,FIT_READY=False,background_qualified=False,production_adopted=False)
    hcnew=deepcopy(hc);hcnew['split']['application_scope_mm'][1]=support
    hcnew['geometry']['supported_r_mm'][1]=support;hcnew['owner_support']=domain
    bcnew=deepcopy(bc);bcnew['application_anchor_radius_mm'][1]=rmax;bcnew['owner_support']=domain
    bcnew['application_status']='RF0..6deg owner ancestry; actual physical pedicle support with explicitly approved outer engineering extension'
    bcnew['outside_application']='reject outside the source-derived finite support; never substitute a6deg pedicle mask for the RF owner mask'
    destinations={'configurations/hc_density_rf_joint_v2_owner_support_v1.json':hcnew,
        'configurations/bc_cone_pedicle_anchored_owner_support_v1.json':bcnew,
        'configurations/research/GIN69_OWNER_SUPPORT_V2.json':domain}
    for relative,value in destinations.items():
        path=ROOT/relative
        if path.exists():raise FileExistsError('Existing domain configuration retained: '+relative)
    backup=output/'previous';backup.mkdir()
    names=('hc_joint_v2.py','hc_joint_v2_geometry.py','bc_pedicle_density.py','bc_pedicle_graph.py')
    for name in names:
        shutil.copy2(ROOT/'retina_sim'/name,backup/name)
        shutil.copy2(STAGE/name,ROOT/'retina_sim'/name)
    for relative,value in destinations.items():atomic_json(ROOT/relative,value)
    master=read(ROOT/'configurations/research/HC69_HC2_BC1_master_v1.json')
    master.update(profile_id='HC69_owner_support_v2',asset_path='data/atlas/GIN_69_6_HC2_BC1_SUPPORT_V2/HC_master.json',
        expected_asset_sha256=None,density_profile_path='configurations/hc_density_rf_joint_v2_owner_support_v1.json',
        construction_domain_pending=True,canonical_disk_radius_mm=support,owner_support=domain,
        scope='New finite local HC master; ownerRF disk separate from source-derived physical support, outer engineering extrapolation')
    masterpath=ROOT/'configurations/research/HC69_owner_support_v2_master.json'
    if masterpath.exists():raise FileExistsError('Existing new master configuration retained')
    atomic_json(masterpath,master)
    for kind,param in [('hc','configurations/hc_density_rf_joint_v2_owner_support_v1.json'),('bc','configurations/bc_cone_pedicle_anchored_owner_support_v1.json')]:
        path=ROOT/f'configurations/{kind}_density_geometry_current.json';shutil.copy2(path,backup/path.name);selected=read(path)
        selected['bindings']['parameters']=dict(path=param,sha256=digest(ROOT/param))
        for binding in selected['bindings'].values():binding['sha256']=digest(ROOT/binding['path'])
        selected['bindings']['owner_support']=dict(path='configurations/research/GIN69_OWNER_SUPPORT_V2.json',sha256=digest(ROOT/'configurations/research/GIN69_OWNER_SUPPORT_V2.json'))
        selected['owner_support_identity']=canonical_hash(domain);selected['asset_status_report']='reports/GIN_69_6_HC2_BC1_SUPPORT_V2/CURRENT_ASSEMBLY_STAGE.json'
        if kind=='hc':selected['master_configuration']=masterpath.relative_to(ROOT).as_posix()
        atomic_json(path,selected)
    previous=ROOT/'reports/GIN_69_6_HC2_BC1_FIT_V1/SPLIT_CONTRACT.json'
    shutil.copy2(previous,OUT/'SPLIT_CONTRACT.json')
    atomic_json(output/'RESULT.json',dict(status='CURRENT_HC_BC_OWNER_SUPPORT_INSTALLED',domain=domain,
        original_split_bytes_preserved=True,coefficients_and_ratios_changed=False,wall_seconds=time.perf_counter()-start,
        changed_source_files=list(names),new_configuration_paths=list(destinations),new_geometry_built=False))
    print(str(domain),flush=True);guard()
