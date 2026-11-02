"""Compatible local equilibrium using frozen equations and held-boundary rule."""
from copy import deepcopy
import time
import numpy as np
from scipy.io import loadmat
from retina_sim.contracts import canonical_hash
from retina_sim.isetbio_input import REFERENCE_SCHEMA,reference_equivalent_drive
from retina_sim.h1_h2_background import initialize_joint_background
from retina_sim.hc_lateral_candidate import LateralHCCandidate
from retina_sim.cone_aperture_hc_bridge import prepare_aperture_background
from retina_sim.h1_h2_effective import save_artifact
from retina_sim.group_time_storage import save_bundle
from retina_sim.group_time_monitor import frozen_budget
from tools.gin_radial_context_v1 import ROOT,BASE,read,write,corrected_typed,protocol
from tools.prepare_gin_radial_candidate_v1 import sha

def initialize(guard):
    start=time.perf_counter();out=BASE/'background';out.mkdir();opt=BASE/'optical';config=read(opt/'CONFIG.json');meta=read(opt/'cache/metadata.json')
    capture=loadmat(opt/'CAPTURE.mat',variable_names=['expected_counts','manifest_json']);q=np.asarray(capture['expected_counts'],dtype=float)/config['exposure_seconds']
    with np.load(opt/'APERTURE_GEOMETRY.npz',allow_pickle=False) as z:types=z['cone_types'].astype('U1');counts=z['local_group_counts']
    scope=canonical_hash(dict(config_sha256=sha(opt/'CONFIG.json'),cache_metadata_sha256=sha(opt/'cache/metadata.json'),capture_sha256=sha(opt/'CAPTURE.mat')))
    bridge=dict(schema=REFERENCE_SCHEMA,profile_id='gin_CRT4mm_fixed_source_reference_v1',source_id='69#6_radial_literature_brightness_approximation',
        reference_conditions='Fixed virtual foveal reference geometry inherited; same source pre-live grey spectrum and4mm optics, one reference triplet across all local cones',
        scope='engineering_fixture',evidence_tag='project_approximation',capture_scope_sha256=scope,
        reference_scene_sha256=sha(opt/'CONFIG.json'),reference_capture_sha256=sha(opt/'cache/model.mat'),
        reference_luminance_cd_m2=meta['reference_luminance_cd_m2'],pupil_diameter_mm=4.,
        reference_rates_rstar_s_LMS=dict(zip(('L','M','S'),np.asarray(meta['q_ref_LMS']).ravel().tolist())))
    drives,result=reference_equivalent_drive(q,types,bridge,capture_scope_sha256=scope)
    cuts=np.r_[0,np.cumsum(counts)];group=np.stack([np.mean(drives[:,cuts[i]:cuts[i+1]],axis=1) for i in range(len(counts))],axis=1)
    bg=group[0].copy();write(out/'FIXED_LMS_BRIDGE.json',dict(profile=bridge,result=result,capture_scope=scope,background_range=[float(bg.min()),float(bg.max())],
        input_mapping='Rstar/s -> fixed global LMS reference-equivalent drive -> one arithmetic mean over actual fine members',
        physical_photopic_td_claim=False,reference_area_contract=meta['reference_area_contract']))
    np.savez_compressed(out/'MOVIE_NATIVE_INPUTS.npz',native_background=bg,live_native_drive=group[1:],fine_rates_rstar_s=q,time_live_frame_start_s=np.arange(8)/150)
    guard();t=time.perf_counter();h,old,labels=corrected_typed(guard);outer=h.seed(old,guard=guard)
    write(out/'INITIALIZATION_PROGRESS.json',dict(stage='corrected_full_parent_reference_and_held_boundary_ready',wall_seconds=time.perf_counter()-start))
    joint,binding=initialize_joint_background(h,outer,ROOT,protocol(),guard=guard);source_seconds=time.perf_counter()-t
    native=LateralHCCandidate(h,coupling_rate_per_second=50.,S_efficacy_ratio=1.)
    cfg=deepcopy(read(ROOT/'configurations/research/HC_homotypic_lateral_only_v2.json'));cfg.pop('candidate_sha256')
    cfg.update(parent_initialized_model=joint['joint_background_initialization']['initialized_model_id'],parent_binding=binding.identity,selected_native_execution=native.source_version,
        binding_scope='69#6 real radial eccentricity3.88deg with approved+x meridian and source4mm brightness approximation; frozen latest HC/pooling',
        HC_projection_source_sha256=sha(ROOT/'reports/retinotopic_rf_revision/native_validation_v1/CANDIDATE_MODEL.json'),pooling_matrices=h.pool_bindings)
    cfg['candidate_sha256']=canonical_hash(cfg);t=time.perf_counter()
    a,seed,interface,y,equilibrium=prepare_aperture_background(h,joint,cfg,bg,ROOT,protocol(),input_identity=dict(mode='new',optical_capture_id=sha(opt/'CAPTURE.mat'),aperture_profile=config['aperture_profile_sha256'],bridge_profile=canonical_hash(bridge)))
    equilibrium_seconds=time.perf_counter()-t;guard();model=a.runtime_factories['inner'](a,seed['inner_parameters']);state={k:np.array(seed['states'][k],copy=True) for k in model.state_keys}
    provider=a.lateral_native_factory(a)
    from types import SimpleNamespace
    yy,anchor=provider.construction_point(SimpleNamespace(states=seed['states'],time_seconds=0.),bg,1.,0);release=provider.evaluate(yy,0.,anchor).outputs['release']
    rhs=model.rhs(0.,state,release);after,_,_=model.commit_ticks(0.,state,cone_release=release,bc_tick=True)
    history={':'.join(k):float(abs(after[k]-v).max(initial=0)) for k,v in state.items() if k[1]=='HistoryBipolarTerminal' and k[2].startswith('history_')}
    norms=[float((abs(rhs[k])*.0001/frozen_budget(state[k],block.unit)[2]).max(initial=0)) for k,block in zip(model.state_keys,model.blocks)]
    if any(history.values()) or not all(np.isfinite(v).all() for v in rhs.values()) or max(norms)>1:raise ValueError('Original initialization history/finite/one-step state budget failed')
    write(out/'HC_POSITION_BINDING.json',cfg);save_artifact(h,labels,out/'typed_artifact')
    save_bundle(out/'inner_boundary',kind='validation_evidence',identity=dict(artifact=h.artifact_sha256),payload=h.boundary_runtime,qualification=dict(default_adopted=False,fit_eligible=False,background_qualified=False,FIT_READY=False))
    save_bundle(out/'seed',kind='source_construction_seed',identity=seed['joint_background_initialization'],payload=seed,qualification=dict(default_adopted=False,fit_eligible=False,background_qualified=False,FIT_READY=False,production_adopted=False))
    np.savez_compressed(out/'BACKGROUND.npz',native_background=bg,cone_release=release,native_packed=y)
    write(out/'RESULT.json',dict(status='PASS',initialization_wall_seconds=time.perf_counter()-start,source_context_and_joint_seconds=source_seconds,optical_local_equilibrium_seconds=equilibrium_seconds,
        initial_inner100us_over_original_budget=max(norms),initial_native_max_rhs=float(abs(provider.rhs(yy,0.,anchor)).max(initial=0)),
        first_BC_history_changes=history,native_equilibrium=equilibrium,inner_states=model.nstate,independent_new_optical_background=True,
        native_boundary_rule='Corrected complete-fine HC global-reference equilibrium then local held products; no stimulus-synchronous exterior',
        inner_boundary_rule='Existing source construction t0 held products; no new boundary parameter',
        source_acquisition_preroll_replayed=False,short_forward_starts_from_independent_steady_source_frame0=True,
        physical_whole_retina_closure=False,parameters_sha256=canonical_hash(seed['inner_parameters']),initialized_model=seed['joint_background_initialization'],
        FIT_READY=False,background_qualified=False,production_adopted=False))
