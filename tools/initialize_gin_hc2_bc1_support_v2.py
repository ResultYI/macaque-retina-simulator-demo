"""Fresh actual-input background for the new HC/BC member-projected network."""
from copy import deepcopy
from types import SimpleNamespace
import time
import numpy as np
from scipy.io import loadmat
from retina_sim.contracts import canonical_hash
from retina_sim.isetbio_input import REFERENCE_SCHEMA,reference_equivalent_drive
from retina_sim.group_parameter_binding import build_group_construction_seed
from retina_sim.h1_h2_effective import H1H2Assembly,verify_typed_source,joint_equilibrium,save_artifact
from retina_sim.h1_h2_background import initialize_joint_background
from retina_sim.hc_lateral_candidate import LateralHCCandidate
from retina_sim.cone_aperture_hc_bridge import prepare_aperture_background
from retina_sim.ac_vs_model import ACVSAssembly,load_config,redesign_seed
from retina_sim.group_time_storage import save_bundle
from retina_sim.group_time_monitor import frozen_budget
from retina_sim.vh2005 import SOURCE_VALUES
from retina_sim.spatial_group_graph import digest
from tools.durable_research_runtime_v1 import atomic_json
from tools.gin_hc2_bc1_context_support_v2 import ROOT,OUT,OLD,GROUP,NATIVE,NewLocalAssembly,NativeParent,read,bundle,protocol

def build(output,guard):
    start=time.perf_counter();opt=OUT/'optical';cfg=read(opt/'CONFIG.json');meta=read(opt/'cache/metadata.json')
    if read(opt/'MATLAB_RESOURCE.json')['status']!='PASS':raise ValueError('Completed optical capture required')
    capture=loadmat(opt/'CAPTURE.mat',variable_names=['expected_counts'])
    q=np.asarray(capture['expected_counts'],dtype=np.float64)/cfg['exposure_seconds']
    ap=bundle(opt/'APERTURE_GEOMETRY.npz');types=ap['cone_types'].astype('U1');counts=ap['global_group_counts'];keep=ap['local_cone_rows']
    scope=canonical_hash(dict(config=digest(opt/'CONFIG.json'),metadata=digest(opt/'cache/metadata.json'),capture=digest(opt/'CAPTURE.mat')))
    bridge=dict(schema=REFERENCE_SCHEMA,profile_id='gin_CRT4mm_fixed_source_reference_v1',source_id='69#6_radial_literature_brightness_approximation',
        reference_conditions='Same fixed foveal reference geometry, source pre-live grey spectrum and4mm optics; one reference triplet',
        scope='engineering_fixture',evidence_tag='project_approximation',capture_scope_sha256=scope,
        reference_scene_sha256=digest(opt/'CONFIG.json'),reference_capture_sha256=digest(opt/'cache/model.mat'),
        reference_luminance_cd_m2=meta['reference_luminance_cd_m2'],pupil_diameter_mm=4.,
        reference_rates_rstar_s_LMS=dict(zip(('L','M','S'),np.asarray(meta['q_ref_LMS']).ravel().tolist())))
    drive,conversion=reference_equivalent_drive(q,types,bridge,capture_scope_sha256=scope)
    cuts=np.r_[0,np.cumsum(counts)];group=np.stack([np.mean(drive[:,cuts[i]:cuts[i+1]],axis=1) for i in range(len(counts))],axis=1)
    fullbg=group[0];bg=fullbg[keep];guard()
    np.savez_compressed(output/'MOVIE_NATIVE_INPUTS.npz',native_background=bg,full_native_background=fullbg,
        live_native_drive=group[1:,keep],full_live_native_drive=group[1:],fine_rates_rstar_s=q,time_live_frame_start_s=np.arange(8)/150)
    atomic_json(output/'FIXED_LMS_BRIDGE.json',dict(profile=bridge,result=conversion,capture_scope=scope,
        physical_photopic_td_claim=False,reference_area_contract=meta['reference_area_contract']))
    stage=time.perf_counter();local=NewLocalAssembly(bg);full=NativeParent(local,fullbg)
    labels,evidence=verify_typed_source(full,ROOT)
    ac=ACVSAssembly(local,load_config(ROOT/'configurations/ac_vs_functional_v1.json'))
    h=H1H2Assembly(ac,full,labels,evidence,engineering=True)
    eq=joint_equilibrium(h.global_A,h.global_B,h.global_h1,h.global_h2,fullbg,h.h2_parameters,guard=guard)
    c,rows=h.native_rows['cones'],h.native_rows['HC'];p=SOURCE_VALUES
    carrier={k:v[c] for k,v in eq['cone'].items()}
    w=p['g_t']*(h.global_A[rows]@eq['release'])
    carrier.update(z1=p['g_t']*eq['release'][c],w=w,z2=w,Vh=eq['h'][rows])
    reference=dict(schema='joint-H1H2-complete-parent-construction-reference-1',artifact_sha256=local.artifact_sha256,
        background_sha256=canonical_hash(bg),equilibrium_residual=eq['residual'],native=carrier,
        release=eq['release'][c],total_feedback=(eq['feedback']@eq['h'])[c],
        complete_parent=full.artifact_sha256,source_sha256=digest(ROOT/'retina_sim/h1_h2_effective.py'))
    old_policy_path=ROOT/local.original_geometry.manifest['configuration']['source_run']/'input_policy.json'
    old_policy=read(old_policy_path)
    policy=dict(approval_status='USER_AUTHORIZED_GIN69_HC2_BC1_20261101',artifact_sha256=local.artifact_sha256,
        rod=old_policy['rod'],rod_source_policy_sha256=digest(old_policy_path),
        input_approximation_manifest=dict(path=(OLD/'INPUT_APPROXIMATION_MANIFEST.json').relative_to(ROOT).as_posix(),sha256=digest(OLD/'INPUT_APPROXIMATION_MANIFEST.json')),
        optical_capture_sha256=digest(opt/'CAPTURE.mat'),new_HC_BC=True,old_neural_seed_reused=False,
        train_dev_test_split=dict(path=(OUT/'SPLIT_CONTRACT.json').relative_to(ROOT).as_posix(),sha256=digest(OUT/'SPLIT_CONTRACT.json')),FIT_READY=False,background_qualified=False,production_adopted=False)
    policy_path=OUT/'input_policy.json'
    if policy_path.exists():
        if read(policy_path)!=policy:raise ValueError('Existing run input policy differs')
    else:atomic_json(policy_path,policy)
    initial=build_group_construction_seed(local,ROOT,input_policy_binding=dict(path=policy_path.relative_to(ROOT).as_posix(),sha256=digest(policy_path)),
        rng_seed=local.config['seed'],guard=guard,native_reference=reference)
    seed=dict(inner_parameters=deepcopy(initial.inner_parameters),states=deepcopy(initial.states),spike_states=deepcopy(initial.spike_states),
        observation_prehistory=deepcopy(initial.observation_prehistory),input_reference=deepcopy(initial.input_reference),
        time_seconds=initial.time_seconds,rng_seed=local.config['seed'],
        native_inputs={i:bg.copy() for i in range(8)},
        rod_inputs={i:np.full(local.populations['rods'].group_count,float(initial.input_reference['rod_background_rstar_s'])) for i in range(8)},
        patch_initialization=dict(source_construction=initial.identity,old_seed_reused=False),
        patch_boundary=dict(identity=local.boundary_runtime['identity'],policy=local.manifest['boundary']))
    seed['inner_parameters']['identity']=canonical_hash(dict(source_construction=initial.identity,
        resolved_parameters=seed['inner_parameters'],local_artifact=local.artifact_sha256))
    seed=redesign_seed(ac,seed);outer=h.seed(seed,guard=guard)
    atomic_json(output/'INITIALIZATION_PROGRESS.json',dict(stage='NEW_SOURCE_AND_COMPLETE_PARENT_NATIVE_READY',wall_seconds=time.perf_counter()-start))
    joint,binding=initialize_joint_background(h,outer,ROOT,protocol(),guard=guard)
    source_seconds=time.perf_counter()-stage
    native=LateralHCCandidate(h,coupling_rate_per_second=50.,S_efficacy_ratio=1.)
    configuration=deepcopy(read(ROOT/'configurations/research/HC_homotypic_lateral_only_v2.json'));configuration.pop('candidate_sha256')
    configuration.update(parent_initialized_model=joint['joint_background_initialization']['initialized_model_id'],
        parent_binding=binding.identity,selected_native_execution=native.source_version,
        binding_scope='69#6 true3.88deg, approved+x/brightness approximation, newHCv2BCv1; actual local member projection',
        HC_projection_source_sha256=digest(NATIVE/'NATIVE_ASSEMBLY_RESULT.json'),new_group_artifact=local.artifact_sha256)
    configuration['candidate_sha256']=canonical_hash(configuration);stage=time.perf_counter()
    assembly,final,interface,y,equilibrium=prepare_aperture_background(h,joint,configuration,bg,ROOT,protocol(),
        input_identity=dict(mode='new',optical_capture_id=digest(opt/'CAPTURE.mat'),aperture_profile=cfg['aperture_profile_sha256'],bridge_profile=canonical_hash(bridge)))
    lateral_seconds=time.perf_counter()-stage;guard()
    model=assembly.runtime_factories['inner'](assembly,final['inner_parameters']);state={k:np.array(final['states'][k],copy=True) for k in model.state_keys}
    provider=assembly.lateral_native_factory(assembly)
    yy,anchor=provider.construction_point(SimpleNamespace(states=final['states'],time_seconds=0.),bg,1.,0)
    release=provider.evaluate(yy,0.,anchor).outputs['release'];rhs=model.rhs(0.,state,release)
    after,_,_=model.commit_ticks(0.,state,cone_release=release,bc_tick=True)
    history={':'.join(k):float(abs(after[k]-v).max(initial=0)) for k,v in state.items() if k[1]=='HistoryBipolarTerminal' and k[2].startswith('history_')}
    norms=[float((abs(rhs[k])*.0001/frozen_budget(state[k],block.unit)[2]).max(initial=0)) for k,block in zip(model.state_keys,model.blocks)]
    if any(history.values()) or not all(np.isfinite(v).all() for v in rhs.values()) or max(norms)>1:
        raise ValueError('Original initialization history/finite/one-step budget failed')
    atomic_json(output/'HC_POSITION_BINDING.json',configuration);save_artifact(h,labels,output/'typed_artifact')
    qualification=dict(default_adopted=False,fit_eligible=False,background_qualified=False,FIT_READY=False,production_adopted=False)
    save_bundle(output/'inner_boundary',kind='validation_evidence',identity=dict(artifact=h.artifact_sha256),payload=h.boundary_runtime,qualification=qualification)
    save_bundle(output/'seed',kind='source_construction_seed',identity=final['joint_background_initialization'],payload=final,qualification=qualification)
    np.savez_compressed(output/'BACKGROUND.npz',native_background=bg,cone_release=release,native_packed=y)
    result=dict(status='PASS_NEW_HC_BC_COMPATIBLE_BACKGROUND',initialization_wall_seconds=time.perf_counter()-start,
        source_context_and_joint_seconds=source_seconds,lateral_equilibrium_seconds=lateral_seconds,
        initial_inner100us_over_original_budget=max(norms),initial_native_max_rhs=float(abs(provider.rhs(yy,0.,anchor)).max(initial=0)),
        first_BC_history_changes=history,native_equilibrium=equilibrium,inner_state_scalars=model.nstate,
        source_parent_equilibrium_residual=eq['residual'],independent_new_seed=True,old_seed_reused=False,
        input_approximations='approved+x meridian and222.2td central weighted minute anchor,4mm pupil; no recovered absolute CRT claim',
        native_boundary='Complete new typed parent equilibrium, then held exterior; unchanged sealed homotypic lateral law',
        inner_boundary='Actual full fine incident gap degree; original per-type t0 V0 held exterior',
        neural_forward_started=False,split_approved=True,FIT_READY=False,background_qualified=False,production_adopted=False,
        group_artifact=local.artifact_sha256,initialized_model=final['joint_background_initialization'])
    atomic_json(output/'RESULT.json',result)
    atomic_json(OUT/'BACKGROUND_LOCATION.json',dict(path=output.relative_to(ROOT).as_posix(),result_sha256=digest(output/'RESULT.json'),status=result['status']))
    print(str({k:v for k,v in result.items() if k not in ('initialized_model','first_BC_history_changes')}),flush=True)
