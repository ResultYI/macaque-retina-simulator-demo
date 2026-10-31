"""Exact8166 halo owner over the frozen far-point upstream; no core relabel."""
from pathlib import Path
from copy import deepcopy
import sys
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.integrated_far_context_v1 import ProjectionAssembly,PREPARED,POOL,BASE as INT,HC,read,write
from tools.cross_position_context_v1 import patch
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.local_patch_graph import LocalPatchAssembly
from retina_sim.local_patch_runner import prepare_seed,restrict_construction_payload
from retina_sim.local_patch_boundary import BoundedInnerDynamics,BoundedNativePhysics
from retina_sim.group_time_storage import load_bundle,save_bundle
from retina_sim.ac_vs_model import ACVSAssembly,load_config,redesign_seed
from retina_sim.h1_h2_effective import H1H2Assembly,verify_typed_source,matrix_id
from retina_sim.hc_lateral_candidate import LateralHCCandidate
from retina_sim.hc_lateral_interface import initialize_from_native,context
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.disk_graph_storage import FrozenSparseRoute,file_sha
from retina_sim.geometry_anchored_pooling import anchored_radius,select_row

BASE=ROOT/'reports/CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1'
MODEL=BASE/'halo_model_v2';TARGET='sg2:61468d3761f46cda:parasol_RGC_ON:8166'


class ActiveACParent:
    """Apply the existing ACVS route choice before local closure, not after it."""
    def __init__(self,base):
        from retina_sim.spatial_group_model import AC
        from retina_sim.ac_vs_model import LEGACY_S
        self.original=base
        self.routes={n:r for n,r in base.routes.items() if (r.pre not in AC and r.post not in AC)
            or r.post==LEGACY_S or (r.pre==LEGACY_S and r.post=='small_bistratified_S_ON')}
        self.manifest=deepcopy(base.manifest)
        self.manifest['routes']=[r for r in self.manifest['routes'] if r['name'] in self.routes]
    def __getattr__(self,name):return getattr(self.original,name)


def protocol():
    from tools.integrated_far_context_v1 import protocol as original
    p=original();p.update(stop_seconds=7.1,checkpoint_seconds=7.1,input_frame_indices=list(range(1065)),
        cell6_named_halo_target=TARGET,scope='new raw Cell6 conditional experiment, unchanged original clocks')
    return p


def local_model():
    base,old=patch(PREPARED)
    path=MODEL/'local_static';descriptor=read(path/'selection_identity.json')
    with np.load(path/'selection.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    if canonical_hash(arrays)!=descriptor['selection_sha256']:raise ContractError('Named halo selection changed')
    rows={k[12:]:v for k,v in arrays.items() if k.startswith('population__')}
    contacts={k[9:]:v for k,v in arrays.items() if k.startswith('contact__')}
    core={k[6:]:v for k,v in arrays.items() if k.startswith('core__')}
    result=object.__new__(LocalPatchAssembly)
    result._assemble(ROOT,path,ActiveACParent(base),old.config,rows,contacts,core,old.target_parent_row,canonical_hash(descriptor),descriptor)
    result.runtime_factories=dict(inner=BoundedInnerDynamics,native=BoundedNativePhysics)
    result.boundary_runtime=load_bundle(INT.parent.parent/'position_S0_far_originaltarget_20261029_resume_v2/background/inner_boundary',expected_kind='validation_evidence')['payload']
    return base,old,result


class HaloProjection(ProjectionAssembly):
    def __init__(self,parent):
        super().__init__(parent)
        matrices={};bindings={}
        for route in read(POOL/'STRUCTURE_RESULT.json')['routes']:
            if route['location']!='far':continue
            name=route['route'];path=(MODEL/'pooling_row'/name/'group_candidate_local.npz' if name=='diffuse_BC_ON_to_parasol_RGC_ON'
                else POOL/'structure/far'/name/'group_candidate_local.npz')
            matrices[name]=sparse.load_npz(path);bindings[name]=file_sha(path)
        original=parent.runtime_factories['inner']
        self.connection_identity.update(named_halo_target=TARGET,named_selection_sha256=file_sha(MODEL/'local_static/selection_identity.json'))
        self.artifact_sha256=canonical_hash(dict(connection=self.connection_identity,parameters=parent.h2_parameters))
        def factory(a,parameters):
            model=original(a,parameters)
            for name,matrix in matrices.items():
                if model.mean[name].shape!=matrix.shape:raise ContractError('Named halo pooling dimensions differ')
                model.mean[name]=matrix
            model.identity=canonical_hash(dict(parent=model.identity,pooling=bindings,HC=self.artifact_sha256,named_halo=TARGET))
            return model
        self.runtime_factories=dict(parent.runtime_factories,inner=factory);self.pool_bindings=bindings


def projection():
    base,old,local=local_model()
    ac=ACVSAssembly(local,load_config(ROOT/'configurations/ac_vs_functional_v1.json'))
    labels,evidence=verify_typed_source(base,ROOT)
    h=H1H2Assembly(ac,base,labels,evidence,engineering=True)
    return HaloProjection(h),local,old


def build_pooling_row(base,local):
    name='diffuse_BC_ON_to_parasol_RGC_ON';folder=POOL/'structure/far'/name;meta=read(folder/'SOURCE.json')
    pre=FrozenDiskTable(ROOT,meta['pre_table']);post=FrozenDiskTable(ROOT,meta['post_table'])
    route=FrozenSparseRoute(ROOT,meta['source']);group=base.populations['parasol_RGC_ON']
    fine=group.member_global_rows[group.member_offsets[8166]:group.member_offsets[8167]]
    fine,old,_,statuses=route.selected_rows(fine)
    if not all(s.startswith('READY') for s in statuses):raise ContractError('Original8166 fine source status unsupported')
    eligible=np.flatnonzero(pre.extras['eligible_drive']);tree=cKDTree(pre.coordinates['ipl_preterminal'][eligible])
    anchors=meta['anchors'];e=np.linalg.norm(post.coordinates['rf'][fine],axis=1)/.223
    radii=anchored_radius(e,anchors['near']['patch_center_eccentricity_deg'],anchors['far']['patch_center_eccentricity_deg'],anchors['near']['radius_mm'],anchors['far']['radius_mm'])
    pgroup=base.populations['diffuse_BC_ON'];source_groups=local.parent_population_rows['diffuse_BC_ON']
    row=np.zeros(len(source_groups));detail=[]
    for j,f in enumerate(fine):
        mass=float(old[j].sum());found,weights,status=select_row(tree,post.coordinates['ipl_post_arbor'][f],radii[j],mass,meta['complete_source_support_radius_mm'])
        if found is None:raise ContractError('8166 candidate row unsupported: '+status)
        source_fine=eligible[found];groups=pgroup.fine_to_group[source_fine];indices=np.searchsorted(source_groups,groups)
        if np.any(indices>=len(source_groups)) or not np.array_equal(source_groups[indices],groups):raise ContractError('8166 active candidate upstream missing')
        np.add.at(row,indices,weights/len(fine))
        detail.append(dict(fine_row=int(f),radius_mm=float(radii[j]),inputs=len(found),old_mass=mass,new_mass=float(weights.sum()),status=status))
    with np.load(folder/'IDENTITIES.npz',allow_pickle=False) as z:previous=z['post_parent_groups']
    matrix=sparse.load_npz(folder/'group_candidate_local.npz');wanted=local.parent_population_rows['parasol_RGC_ON']
    result=sparse.vstack([sparse.csr_matrix(row[None,:]) if g==8166 else matrix[int(np.flatnonzero(previous==g)[0])] for g in wanted],format='csr')
    dest=MODEL/'pooling_row'/name;dest.mkdir(parents=True,exist_ok=False);sparse.save_npz(dest/'group_candidate_local.npz',result)
    write(dest/'IDENTITY.json',dict(target=TARGET,member_details=detail,old_anchors=anchors,
        same_formula=True,new_anchor_fit=False,local_renormalization=False,all_member_average=True,
        source=meta['source'],matrix_identity=matrix_id(result),existing_rows_preserved=True,qualification='ENGINEERING_SPATIAL_POOLING_CANDIDATE'))


def prepare(guard):
    MODEL.mkdir(exist_ok=False);base,old=patch(PREPARED);guard()
    old_seed=prepare_seed(ROOT,old,old.config,guard=guard)  # Original source/code-pin checks unchanged.
    folder=MODEL/'local_static';folder.mkdir()
    with np.load(PREPARED/'local_static/selection.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    arrays['population__parasol_RGC_ON']=np.union1d(arrays['population__parasol_RGC_ON'],[8166]).astype(np.int64)
    descriptor=deepcopy(read(PREPARED/'local_static/selection_identity.json'))
    descriptor.update(schema='explicit-named-halo-owner-cell6-v1',selection_sha256=canonical_hash(arrays),
        named_halo_output=dict(group_id=TARGET,parent_row=8166,members=2,core_promoted=False,authorization='User approved raw rebuild plus exact8166 inclusion'),
        active_route_selection='Exact existing ACVSAssembly predicate applied before local chemical closure; inactive ordinary AC routes absent, active closure guard unchanged',
        active_route_source_sha256=file_sha(ROOT/'retina_sim/ac_vs_model.py'))
    np.savez_compressed(folder/'selection.npz',**arrays);write(folder/'selection_identity.json',descriptor)
    _,_,local=local_model()
    for pop in ('cones','HC','AII_AC','rods','rod_BC',*('midget_BC_ON','midget_BC_OFF','S_ON_BC','diffuse_BC_ON','diffuse_BC_OFF')):
        if not np.array_equal(local.parent_population_rows[pop],old.parent_population_rows[pop]):raise ContractError('Upstream selection changed: '+pop)
    build_pooling_row(base,local);guard()
    parent=load_bundle((ROOT/old.config['input_identity']['path']).parent,expected_kind='source_construction_seed')['payload']
    restricted=restrict_construction_payload(parent,local.parent_population_rows,local.parent_contact_rows,local.parent_auxiliary_rows,rng_seed=old.config['seed'],allow_empty=True)
    restricted['patch_boundary']=deepcopy(old_seed['patch_boundary'])
    ac=ACVSAssembly(local,load_config(ROOT/'configurations/ac_vs_functional_v1.json'));prepared=redesign_seed(ac,restricted)
    labels,evidence=verify_typed_source(base,ROOT);h=H1H2Assembly(ac,base,labels,evidence,engineering=True);a=HaloProjection(h)
    old_joint=load_bundle(INT/'background/seed',expected_kind='source_construction_seed')['payload']
    prepared['states']={k:v for k,v in prepared['states'].items() if k[0] not in ('cones','HC')}
    prepared['states'].update({k:v.copy() for k,v in old_joint['states'].items() if k[0] in ('cones','HC')})
    for name in ('h1_h2_model','observation_prehistory','input_reference'):prepared[name]=deepcopy(old_joint[name])
    prepared['native_inputs']={i:prepared['input_reference']['native_background_td'].copy() for i in range(1065)}
    rod=float(prepared['input_reference']['rod_background_rstar_s'])
    prepared['rod_inputs']={i:np.full(a.populations['rods'].group_count,rod,dtype=np.float64) for i in range(1065)}
    prepared['time_seconds']=0.;prepared['patch_initialization'].update(named_halo=TARGET,core_unchanged=True)
    # All physical values are inherited; only ownership-indexed arrays may grow.
    for pop,fields in old_joint['inner_parameters']['populations'].items():
        for key,value in fields.items():
            other=prepared['inner_parameters']['populations'][pop][key]
            if isinstance(value,np.ndarray) and key in ('V0','baseline','reference_input'):
                indices=np.searchsorted(a.populations[pop].group_ids,old.populations[pop].group_ids) if pop=='parasol_RGC_ON' else np.arange(len(value))
                if not np.array_equal(value,np.asarray(other)[indices]):raise ContractError('Frozen per-group parameter changed')
            elif not np.array_equal(value,other):raise ContractError('Frozen scientific parameter changed: '+pop+'.'+key)
    cfg=deepcopy(read(INT/'background/HC_BINDING.json'));cfg.pop('candidate_sha256')
    native=LateralHCCandidate(a,coupling_rate_per_second=50.,S_efficacy_ratio=1.)
    cfg.update(selected_native_execution=native.source_version,pooling_matrices=a.pool_bindings,named_halo_target=TARGET,
        binding_scope='Same upstream/coupling and physical parameters; explicit additional8166 halo output and own existing crossAC pool')
    cfg['candidate_sha256']=canonical_hash(cfg)
    with np.load(INT/'background/BACKGROUND.npz',allow_pickle=False) as z:y=z['native_packed'].copy()
    final,seed,binding=initialize_from_native(a,prepared,cfg,y,ROOT,protocol(),guard=guard)
    final.parameter_binding_id=seed['lateral_hc_interface']['binding_id'];guard()
    write(MODEL/'HC_BINDING.json',cfg)
    save_bundle(MODEL/'seed',kind='source_construction_seed',identity=seed['joint_background_initialization'],payload=seed,
        qualification=dict(default_adopted=False,fit_eligible=False,FIT_READY=False,background_qualified=False,production_adopted=False))
    write(MODEL/'RESULT.json',dict(status='NAMED_HALO_BACKGROUND_CONSTRUCTED',target=TARGET,target_local_row=int(np.flatnonzero(final.populations['parasol_RGC_ON'].group_ids==TARGET)[0]),
        members=2,core_unchanged=True,upstream_selection_unchanged=True,old_inner_boundary_retained=True,
        old_source_code_pins_verified=True,parameters_value_check_passed=True,
        new_model_identity=final.artifact_sha256,new_binding=seed['lateral_hc_interface']['binding_id'],initialization=seed['joint_background_initialization'],
        pooling=a.pool_bindings,FIT_READY=False,background_qualified=False,production_adopted=False))
    return final,seed


def load():
    a,_,_=projection();seed=load_bundle(MODEL/'seed',expected_kind='source_construction_seed')['payload']
    cfg=read(MODEL/'HC_BINDING.json');final=context(a,seed,cfg,ROOT);final.parameter_binding_id=seed['lateral_hc_interface']['binding_id']
    return final,seed
