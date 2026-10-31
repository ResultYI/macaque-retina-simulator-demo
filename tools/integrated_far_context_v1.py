"""Single far-patch composition of the saved HC correction and pooling candidate."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys

ROOT=Path(r'D:\macaque retinal simulator')
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.disk_graph_storage import file_sha
from retina_sim.group_time_storage import load_bundle,save_bundle
from retina_sim.h1_h2_effective import MECHANISM,matrix_id
from retina_sim.hc_lateral_candidate import LateralHCCandidate
from retina_sim.hc_lateral_interface import context,initialize_from_native
from tools.cross_position_context_v1 import load_background

BASE=ROOT/'reports/retinotopic_rf_revision/INTEGRATED_FAR_HC_POOLING_V1'
HC=ROOT/'reports/retinotopic_rf_revision/native_validation_v1'
POOL=ROOT/'reports/retinotopic_rf_revision/SPATIAL_POOLING_CANDIDATE_V1'
PREPARED=ROOT/'reports/position_S0_far_originaltarget_20261029_resume_v2/prepared'
OLD_BG=PREPARED.parent/'background'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    from retina_sim.parameter_system import data
    with Path(path).open('x',encoding='utf-8') as f:json.dump(data(value),f,indent=2,ensure_ascii=False,allow_nan=False)


class ProjectionAssembly:
    def __init__(self,parent):
        self.base=parent
        descriptor=read(HC/'CANDIDATE_MODEL.json')
        operators={}
        for name in ('A','typed_B'):
            source=descriptor['group_native_operators'][name]
            if file_sha(ROOT/source['path'])!=source['sha256']:raise ContractError('Saved corrected HC operator differs')
            operators[name]=sparse.load_npz(ROOT/source['path'])
        c,h=parent.native_rows['cones'],parent.native_rows['HC']
        self.A=operators['A'][h][:,c].tocsr();self.B=operators['typed_B'][c][:,h].tocsr()
        self.operators=dict(A1=self.A[parent.h1_rows],A2=self.A[parent.h2_rows],
                            B1=self.B[:,parent.h1_rows],B2=self.B[:,parent.h2_rows])
        self.connection_identity=dict(parent=parent.connection_identity,
            corrected_HC_descriptor_sha256=file_sha(HC/'CANDIDATE_MODEL.json'),
            local_A=matrix_id(self.A),local_B=matrix_id(self.B),
            rule='Saved complete-fine normalization then all-member projection; no new geometry rule')
        self.artifact_sha256=canonical_hash(dict(connection=self.connection_identity,parameters=parent.h2_parameters))
        original=parent.runtime_factories['inner']
        bindings={}
        matrices={}
        for route in read(POOL/'STRUCTURE_RESULT.json')['routes']:
            if route['location']!='far':continue
            if not route['dynamic_local_closure']:raise ContractError('Original pooling support not closed')
            path=POOL/'structure/far'/route['route']/'group_candidate_local.npz'
            matrices[route['route']]=sparse.load_npz(path);bindings[route['route']]=file_sha(path)
        def inner_factory(a,parameters):
            model=original(a,parameters)
            for name,matrix in matrices.items():
                if model.mean[name].shape!=matrix.shape:raise ContractError('Original pooling dimensions differ')
                model.mean[name]=matrix
            model.identity=canonical_hash(dict(parent_inner=model.identity,
                pooling_candidate='RETINOTOPIC_GEOMETRY_ANCHORED_POOLING_V1',matrices=bindings,
                frozen_parameters=canonical_hash(parameters),new_HC=self.artifact_sha256))
            return model
        self.runtime_factories=dict(parent.runtime_factories,inner=inner_factory)
        self.pool_bindings=bindings
    def __getattr__(self,name):return getattr(self.base,name)
    def matrix(self,name):
        if name=='routes/cones_to_HC/mean_weights.npz':return self.A
        if name=='routes/HC_to_cones/mean_weights.npz':return self.B
        if name.startswith('h1_h2/') and name.endswith('.npz'):return self.operators[Path(name).stem]
        return self.base.matrix(name)


def source_context():
    old,seed,identity=load_background(PREPARED,OLD_BG)
    projection=ProjectionAssembly(old.parent)
    return projection,seed,identity


def protocol():
    original=read(ROOT/'reports/external_video_two_positions_4s_20261029_v1/PREPARATION.json')['protocol']
    p=deepcopy(original)
    p.update(start_seconds=0.,stop_seconds=.05,clock_origin_seconds=0.,input_origin_seconds=0.,
             input_frame_indices=list(range(8)),checkpoint_seconds=.05)
    return p


def prepare(guard):
    out=BASE/'background';out.mkdir(exist_ok=False)
    projection,parent,parent_identity=source_context();guard()
    original_hash=canonical_hash(parent);prepared=deepcopy(parent)
    with np.load(HC/'fresh_native_initial_state.npz',allow_pickle=False) as z:
        packed=z['packed'].copy();light=z['background_light'].copy()
        if not np.array_equal(z['cone_parent_rows'],projection.native_rows['cones']) or not np.array_equal(z['HC_parent_rows'],projection.native_rows['HC']):
            raise ContractError('Corrected HC source rows differ from the original far patch')
        boundary={k:z[v].copy() for k,v in [('A_release','A_release_boundary'),('A_Vbar','A_Vbar_boundary'),('B_h','B_h_boundary')]}
    if not np.array_equal(light,parent['input_reference']['native_background_td']):raise ContractError('Saved optical S0 changed')
    prepared['h1_h2_model']['boundary']=boundary
    native=LateralHCCandidate(projection,coupling_rate_per_second=50.,S_efficacy_ratio=1.)
    native.boundary=boundary
    count=5*native.nc+2*len(native.h1_rows)+native.nh
    if packed.shape!=(count,):raise ContractError('Saved HC probe packed layout unexpected')
    y=np.zeros(native.nstate,dtype=np.float64);y[:count]=packed
    state=prepared['states']
    for name in ('X','C','Vis','g_i','Vbar'):state['cones','reduced_native',name]=y[native.slices[name]].copy()
    for name in ('w','z2'):state['HC',MECHANISM,name]=y[native.slices[name]].copy()
    carrier=y[native.slices['Vh']]
    for name,selection in [('h1',native.h1_rows),('h2',native.h2_rows)]:state['HC',MECHANISM,name]=carrier[selection].copy()
    _,anchor=native.construction_point(SimpleNamespace(states=state,time_seconds=0.),light,1.,0)
    release=native.evaluate(y,0.,anchor).outputs['release']
    y[native.slices['z1_aux']]=native.p['g_t']*release[native._aux]
    state['cones','selected_original_z1_auxiliary','z1_aux']=y[native.slices['z1_aux']].copy()
    native_residual=float(abs(native.rhs(y,0.,anchor)).max(initial=0))
    if native_residual>1e-7:raise ContractError('Saved latest HC equilibrium not stationary in full native provider')
    cfg=deepcopy(read(OLD_BG/'HC_POSITION_BINDING.json'));cfg.pop('candidate_sha256')
    cfg.update(selected_native_execution=native.source_version,
        binding_scope='Single far patch: corrected saved HC projection plus frozen engineering pooling; unchanged dynamics',
        HC_projection_source_sha256=file_sha(HC/'CANDIDATE_MODEL.json'),pooling_matrices=projection.pool_bindings)
    cfg['candidate_sha256']=canonical_hash(cfg)
    a,seed,binding=initialize_from_native(projection,prepared,cfg,y,ROOT,protocol(),guard=guard)
    model=a.runtime_factories['inner'](a,seed['inner_parameters']);inner={k:seed['states'][k] for k in model.state_keys}
    from retina_sim.group_time_monitor import frozen_budget
    rhs=model.rhs(0.,inner,release)
    norms=[float((abs(rhs[k])*.0001/frozen_budget(inner[k],b.unit)[2]).max(initial=0)) for k,b in zip(model.state_keys,model.blocks)]
    if max(norms)>1:raise ContractError('New coupled inner background exceeds original budget')
    if canonical_hash(seed['inner_parameters'])!=canonical_hash(parent['inner_parameters']):raise ContractError('Parameters changed')
    if canonical_hash(parent)!=original_hash:raise ContractError('Old S0 seed mutated')
    write(out/'HC_BINDING.json',cfg)
    save_bundle(out/'seed',kind='source_construction_seed',identity=seed['joint_background_initialization'],payload=seed,
        qualification=dict(default_adopted=False,fit_eligible=False,FIT_READY=False,background_qualified=False,production_adopted=False))
    np.savez_compressed(out/'BACKGROUND.npz',native_background=light,cone_release=release,native_packed=y)
    write(out/'RESULT.json',dict(status='PASS',native_max_RHS=native_residual,inner_one100us_step_over_budget=max(norms),
        parameters_sha256=canonical_hash(seed['inner_parameters']),source_parent_identity=parent_identity,
        initialized_model=seed['joint_background_initialization'],HC_artifact=a.artifact_sha256,pooling_matrices=projection.pool_bindings,
        inner_held_boundary_hash=canonical_hash(a.boundary_runtime),old_inner_held_boundary_retained=True,
        native_held_boundary='Saved new corrected-HC global-direct background; local homotypic sealed as previously validated',
        independent_inner_reconstruction=True,old_S0_not_overwritten=True,FIT_READY=False,background_qualified=False,production_adopted=False))
    return a,seed


def load():
    projection,parent,_=source_context()
    seed=load_bundle(BASE/'background/seed',expected_kind='source_construction_seed')['payload']
    cfg=read(BASE/'background/HC_BINDING.json')
    a=context(projection,seed,cfg,ROOT)
    a.parameter_binding_id=seed['lateral_hc_interface']['binding_id']
    return a,seed
