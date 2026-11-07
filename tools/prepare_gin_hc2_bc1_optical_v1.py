"""Bind the already approved source scene to new actual Cone membership."""
from pathlib import Path
from copy import deepcopy
import json
import time
import numpy as np
from scipy.io import savemat
from retina_sim.contracts import canonical_hash
from retina_sim.spatial_group_model import GroupModelAssembly
from retina_sim.spatial_group_graph import digest
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.cone_aperture_eccentricity import resolve_apertures
from tools.durable_research_runtime_v1 import atomic_json

ROOT=Path(r'D:\macaque retinal simulator')
OUT=ROOT/'reports/GIN_69_6_HC2_BC1_FIT_V1'
OLD=ROOT/'reports/GIN_69_6_RADIAL_CONDITIONAL_V1/candidate_v2'

def build(output,guard,*,source_out=OUT):
    OUT=Path(source_out)
    start=time.perf_counter();folder=OUT/'optical'
    if folder.exists():raise RuntimeError('Existing optical request retained')
    base=GroupModelAssembly(ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2')
    with np.load(OUT/'executions/assemble_native_01/MEMBERSHIP.npz',allow_pickle=False) as z:
        groups=z['global_cone_parent_group_rows'];local=z['local_cone_rows'];counts=z['cone_counts']
    pop=base.populations['cones']
    fine=np.concatenate([pop.member_global_rows[pop.member_offsets[g]:pop.member_offsets[g+1]] for g in groups])
    cone=FrozenDiskTable(ROOT,pop.source_table);xy=np.asarray(cone.coordinates['rf'][fine])
    types=cone.extras['subtype'][fine].astype('U1')
    profile=json.loads((ROOT/'configurations/isetbio_native_cone_aperture_v1.json').read_bytes())
    apertures=resolve_apertures(tuple('actual_fine_cone_row_'+str(i) for i in fine),types,xy,profile,support_positions_mm=cone.coordinates['rf'])
    cfg=deepcopy(json.loads((OLD/'optical/CONFIG.json').read_bytes()))
    cfg['aperture_profile_sha256']=canonical_hash(profile)
    cfg['aperture_model']=profile['profile_id']
    center=(xy.min(0)+xy.max(0))/2
    samples=int(np.ceil(np.max(xy.max(0)-xy.min(0))*1000+100));samples+=samples%2==0
    if samples>1024:raise RuntimeError('Existing optical tile limit exceeded; needs existing tiling adapter, no reduced resolution')
    folder.mkdir()
    savemat(folder/'geometry.mat',dict(is_mm=xy,types=np.array([{'L':1,'M':2,'S':3}[t] for t in types]),
        reference_diameter_um=apertures['reference_diameter_um'],zone_diameter_um=apertures['zone_diameter_um']),do_compression=True)
    np.savez_compressed(folder/'APERTURE_GEOMETRY.npz',**apertures,fine_rows=fine,
        global_group_counts=counts,local_cone_rows=local,global_cone_parent_group_rows=groups)
    f=.223e-3/np.tan(np.pi/180)
    cfg.update(geometry_mat=str(folder/'geometry.mat'),center_mm=center.tolist(),samples=samples,
        patch_fov_deg=float(np.degrees(2*np.arctan(samples*.001e-3/(2*f)))),
        optics_cache=str(folder/'cache'),cache_directory=str(folder/'cache'),
        new_network_membership_source='executions/assemble_native_01/MEMBERSHIP.npz',
        source_scene_reused=True,old_capture_reused=False,
        geometry_scope='All native parent groups needed for independent exterior background and selected local Cone states')
    cfg['tile_signature']=canonical_hash(dict(geometry=digest(folder/'geometry.mat'),center=center,aperture=cfg['aperture_profile_sha256']))
    cfg.pop('request_signature',None);cfg['request_signature']=canonical_hash(cfg)
    atomic_json(folder/'CONFIG.json',cfg)
    atomic_json(folder/'SOURCE_IDENTITY.json',dict(prior_approved_input_manifest_sha256=digest(OLD/'INPUT_APPROXIMATION_MANIFEST.json'),
        original_source_bank_sha256=digest(Path(cfg['source_bank'])),source_scale_sha256=digest(OLD/'SOURCE_SCALE.json'),
        actual_cone_table=cone.binding,geometry_sha256=digest(folder/'geometry.mat'),aperture_geometry_sha256=digest(folder/'APERTURE_GEOMETRY.npz'),
        native_membership_sha256=digest(OUT/'executions/assemble_native_01/MEMBERSHIP.npz'),
        preserved=['author source clock751/150Hz','approved+x meridian','4mm pupil','central weighted minute222.2td approximation',
            'same source gamma/SPD','same RF pixel to original selected RGC mapping','same QE/aperture/physical units'],
        frames='source frame0 independent background plus751..758 only; no long forward or spike data read',
        FIT_READY=False,background_qualified=False,production_adopted=False))
    result=dict(status='NEW_OPTICAL_REQUEST_READY',fine_cones=len(fine),global_groups=len(groups),local_groups=len(local),
        samples=samples,frame_count=len(cfg['frame_indices']),wall_seconds=time.perf_counter()-start)
    atomic_json(output/'RESULT.json',result);print(json.dumps(result),flush=True)
