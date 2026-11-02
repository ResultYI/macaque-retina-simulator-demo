"""One approved source-bound radial instance; no production defaults changed."""
from pathlib import Path
from copy import deepcopy
import numpy as np
from scipy import sparse
from scipy.io import loadmat,savemat
from retina_sim.contracts import canonical_hash
from retina_sim.group_time_storage import save_bundle,load_bundle
from retina_sim.cone_aperture_eccentricity import resolve_apertures
from retina_sim.h1_h2_effective import matrix_id
from tools.cross_position_context_v1 import ROOT,read,write,patch,typed
from tools.prepare_gin_radial_candidate_v1 import sha

BASE=ROOT/'reports/GIN_69_6_RADIAL_CONDITIONAL_V1/candidate_v2'
HC=ROOT/'reports/retinotopic_rf_revision/native_validation_v1'

def optical_setup():
    out=BASE/'optical';out.mkdir();base,local=patch(BASE/'prepared')
    group=base.root/'populations/cones';offset=np.load(group/'member_offsets.npy');members=np.load(group/'member_global_rows.npy')
    fine=np.concatenate([members[offset[r]:offset[r+1]] for r in local.parent_population_rows['cones']])
    table=ROOT/base.manifest['populations']['cones']['source_table']['path'];xy=np.load(table.parent/'rf.npy')[fine]
    types=np.load(table.parent/'subtype.npy')[fine].astype('U1');profile=read(ROOT/'configurations/research/macaque_cone_aperture_eccentricity_v1.json')
    aperture=resolve_apertures(tuple('actual_fine_cone_row_'+str(i) for i in fine),types,xy,profile)
    savemat(out/'geometry.mat',dict(is_mm=xy,types=np.array([{'L':1,'M':2,'S':3}[t] for t in types]),reference_diameter_um=aperture['reference_diameter_um'],zone_diameter_um=aperture['zone_diameter_um']),do_compression=True)
    np.savez_compressed(out/'APERTURE_GEOMETRY.npz',**aperture,fine_rows=fine,local_group_counts=local.populations['cones'].counts)
    oldref=ROOT/'reports/iset_disk_input_cache/79384a1477b0b3c11fdce71169c4cda04ed23c0340d5576ef3469039de34f6d7/reference/capture.mat'
    ref=json.loads(loadmat(oldref,variable_names=['manifest_json'])['manifest_json'].item())
    p=deepcopy(read(ROOT/'configurations/iset_lcd_apple_mw_v1.json'));p.update(profile_id='gin_source_CRT_4mm_literature_scale_v1',scope='approved_radial_conditional_candidate',pupil_diameter_mm=4.,display='Source_GunSpectra_relative_CRT')
    p.pop('reference_retinal_illuminance_td');p.pop('reference_linear_value')
    p.update(isetcam_root=str(ROOT/'vendor_pinned/isetcam'),isetbio_root=str(ROOT/'vendor_pinned/isetbio'))
    center=(xy.min(0)+xy.max(0))/2;samples=int(np.ceil(np.max(xy.max(0)-xy.min(0))*1000+100));samples+=samples%2==0
    if samples>1024:raise ValueError('Existing optical sampling limit exceeded')
    scale=read(BASE/'SOURCE_SCALE.json');f=.223e-3/np.tan(np.pi/180)
    binding=read(BASE/'INPUT_APPROXIMATION_MANIFEST.json');g=read(BASE/'prepared/GEOMETRY_SELECTION.json')
    target=np.asarray(g['target_centroid_deg'])*.223;pitch=.223*4.6/256
    imagecenter=target-np.array([binding['source_RF_fractional_column']-127.5,-(binding['source_RF_fractional_row']-127.5)])*pitch
    half=127.5*pitch
    c=dict(profile=p,geometry_mat=str(out/'geometry.mat'),center_mm=center,source_bank=str(BASE/'SOURCE_BANK.mat'),
        source_gun_spectra=str(ROOT/'data/source_packages/WP1_20261031/Schottdorf_Lee_xage77/stimuli/GunSpectra.txt'),
        mean_gamma_RGB=scale['mean_gamma_RGB'],reference_rgb=scale['background_rgb'],samples=samples,
        patch_fov_deg=float(np.degrees(2*np.arctan(samples*.001e-3/(2*f)))),exposure_seconds=1/150,
        foveal_reference_diameter_um=ref['reference_diameter_um'],reference_geometry_source_sha256=sha(oldref),
        optics_cache=str(out/'cache'),cache_directory=str(out/'cache'),frame_indices=list(range(1,10)),
        window_edges_seconds=(np.arange(10)/150).tolist(),plot_bounds_mm=[imagecenter[0]-half,imagecenter[0]+half,imagecenter[1]-half,imagecenter[1]+half],
        native_control=True,aperture_profile_sha256=canonical_hash(profile),
        source_RF_mapping='Author fitted RF pixel maps to selected OFF-parasol group RF centroid; image top maps +y; nominal4.6/256deg per pixel',
        old_3mm_or_Latimer_cache_reused=False)
    c['request_signature']=canonical_hash(c);c['tile_signature']=canonical_hash(dict(geometry=sha(out/'geometry.mat'),center=center,aperture=c['aperture_profile_sha256']))
    write(out/'CONFIG.json',c)

def corrected_typed(guard=lambda:None):
    h,old,labels=typed(BASE/'prepared',guard=guard);record=read(HC/'CANDIDATE_MODEL.json');operators={}
    for n in ('A','typed_B'):
        b=record['group_native_operators'][n];p=ROOT/b['path']
        if sha(p)!=b['sha256']:raise ValueError('Corrected HC identity changed')
        operators[n]=sparse.load_npz(p)
    h.global_A=operators['A'];h.global_B=operators['typed_B'];c,rows=h.native_rows['cones'],h.native_rows['HC']
    A=h.global_A[rows][:,c].tocsr();B=h.global_B[c][:,rows].tocsr()
    h.operators=dict(A1=A[h.h1_rows],A2=A[h.h2_rows],B1=B[:,h.h1_rows],B2=B[:,h.h2_rows])
    h._new_matrices.update({'routes/cones_to_HC/mean_weights.npz':A,'routes/HC_to_cones/mean_weights.npz':B})
    h._new_matrices.update({'h1_h2/'+n+'.npz':v for n,v in h.operators.items()})
    h.connection_identity=dict(parent=h.connection_identity,corrected_HC_descriptor_sha256=sha(HC/'CANDIDATE_MODEL.json'),global_A=matrix_id(h.global_A),global_B=matrix_id(h.global_B),local_A=matrix_id(A),local_B=matrix_id(B))
    h.artifact_sha256=canonical_hash(dict(connection=h.connection_identity,parameters=h.h2_parameters))
    original=h.runtime_factories['inner'];matrices={};bindings={}
    for route in read(BASE/'POOLING_RESULT.json')['routes']:
        if not route['dynamic_local_closure']:raise ValueError('Unclosed pooling support')
        p=BASE/'pooling'/route['route']/'group_candidate_local.npz';matrices[route['route']]=sparse.load_npz(p);bindings[route['route']]=sha(p)
    def inner_factory(a,parameters):
        model=original(a,parameters)
        for n,matrix in matrices.items():
            if model.mean[n].shape!=matrix.shape:raise ValueError('Pooling dimensions changed')
            model.mean[n]=matrix
        model.identity=canonical_hash(dict(parent_inner=model.identity,pooling_candidate='RETINOTOPIC_GEOMETRY_ANCHORED_POOLING_V1',matrices=bindings,frozen_parameters=canonical_hash(parameters),new_HC=h.artifact_sha256))
        return model
    h.runtime_factories=dict(h.runtime_factories,inner=inner_factory);h.pool_bindings=bindings
    return h,old,labels

def protocol():
    from tools.integrated_far_context_v1 import protocol as original
    return original()

def load():
    from retina_sim.hc_lateral_interface import context
    h,_,_=corrected_typed();seed=load_bundle(BASE/'background/seed',expected_kind='source_construction_seed')['payload']
    h.parent.patch.boundary_runtime=load_bundle(BASE/'background/inner_boundary',expected_kind='validation_evidence')['payload']
    h.native_boundary=seed['h1_h2_model']['boundary'];a=context(h,seed,read(BASE/'background/HC_POSITION_BINDING.json'),ROOT)
    a.parameter_binding_id=seed['lateral_hc_interface']['binding_id'];return a,seed

import json
