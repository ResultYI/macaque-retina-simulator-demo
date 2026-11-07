"""New explicit joint equilibrium from approved static gray exterior, no movie solve."""
from pathlib import Path
import sys,json,time
from copy import deepcopy
import numpy as np
from scipy import sparse
from scipy.io import loadmat
from scipy.spatial import cKDTree
from scipy.special import expit
from scipy.sparse.linalg import LinearOperator,gmres

ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from retina_sim.contracts import canonical_hash
from retina_sim.vh2005 import SOURCE_VALUES as P
from retina_sim.h1_h2_effective import cone_fixed_point,MECHANISM,matrix_id
from retina_sim.isetbio_input import reference_equivalent_drive
from retina_sim.group_parameter_binding import bind_bc_physiology
from retina_sim.group_time_storage import save_bundle
from tools.prepare_gin69_joint_hc import guard
from tools.gin_hc2_bc1_context_support_v2 import load as old_load
from tools.gin69_joint_context import JointAssembly,initialize_inner


def scalar_source_background(light):
    beta=P['c_beta']+P['k_beta']*light
    lo=np.zeros_like(light);hi=1/beta
    for _ in range(64):
        mid=(lo+hi)/2;res=beta*mid*(1+(P['a_C']*mid)**P['n_C'])-1
        hi=np.where(res>0,mid,hi);lo=np.where(res>0,lo,mid)
    current=(lo+hi)/2;V=(current/P['a_is'])**(1/(1+P['gamma']));scale=(V/P['V_I'])**P['mu']
    lo=np.zeros_like(V);hi=P['g_t']/scale
    for _ in range(64):
        mid=(lo+hi)/2;target=P['g_t']*expit((V-mid-P['V_k'])/P['V_n'])/scale
        hi=np.where(mid>target,mid,hi);lo=np.where(mid>target,lo,mid)
    h=(lo+hi)/2;release=expit((V-h-P['V_k'])/P['V_n'])/scale
    if np.max(abs(h-P['g_t']*release))>1e-10:raise ValueError('Static homogeneous native fixed point failed')
    return V,release


def main():
    began=time.perf_counter();out=ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2';dest=out/'joint_gray_background'
    if (dest/'seed').exists():raise FileExistsError('Complete joint seed retained')
    rates=np.asarray(loadmat(dest/'CAPTURE.mat',variable_names=['rates'])['rates']).ravel()
    z=np.load(dest/'SOURCE_GEOMETRY.npz',allow_pickle=False)
    ids=z['candidate_rows'];types=z['cone_types'].astype('U1');ped=z['pedicle_mm']
    bridge=json.loads((out/'executions/initialize_01/FIXED_LMS_BRIDGE.json').read_bytes())
    light,conversion=reference_equivalent_drive(rates[None,:],types,bridge['profile'],capture_scope_sha256=bridge['capture_scope'])
    Vext,rext=scalar_source_background(light[0]);guard()
    np.savez_compressed(dest/'STATIC_NATIVE_SOURCES.npz',candidate_rows=ids,native_td=light[0],Vbar=Vext,release=rext)
    parent,seed=old_load();native=parent.runtime_factories['native'](parent,bound=False)
    mem=np.load(out/'executions/assemble_native_01/MEMBERSHIP.npz',allow_pickle=False)
    rec=json.loads((out/'executions/assemble_native_01/NATIVE_ASSEMBLY_RESULT.json').read_bytes())
    hm=json.loads((ROOT/rec['hc_table']['path']).read_bytes())
    hx=np.load(ROOT/hm['coordinates']['soma']['path']);ht=np.load(ROOT/hm['extras']['subtype']['path']).astype('U2')
    cm=json.loads((ROOT/rec['cone_table']['path']).read_bytes());keys=np.load(ROOT/cm['source_keys']['path']).astype('U')
    official=np.load(ROOT/'reports/CONE_HC_OUTER_SOURCE_EXTENSION_20261102_v3/cones_01/official_source_rows.npy')
    maps=np.full(len(official),-1,dtype=int)
    source_rows=np.searchsorted(official,np.array([int(k.split(':')[-1]) for k in keys]))
    maps[source_rows]=mem['global_cone_fine_to_group'];mapping=maps[ids]
    nh,nc=len(mem['counts']),len(mem['cone_counts']);ar=np.zeros(nh);av=np.zeros(nh)
    tree=cKDTree(ped)
    for j in np.flatnonzero(ht=='H1'):
        guard();k=np.asarray(tree.query_ball_point(hx[j],3.526911366748626,return_sorted=True),int)
        d=np.linalg.norm(ped[k]-hx[j],axis=1);s=np.exp(-(d-d.min())/.020);s/=s.sum();l=np.exp(-(d-d.min())/.300);l/=l.sum()
        w=.15*s+.85*l;exterior=mapping[k]<0;g=mem['fine_to_group'][j];count=mem['counts'][g]
        ar[g]+=np.dot(w[exterior],rext[k[exterior]])/count
        av[g]+=np.dot(w[exterior],Vext[k[exterior]])/count
        if j%200==0:print(json.dumps(dict(stage='exterior_products',fine_HC_row=int(j),wall_seconds=time.perf_counter()-began)),flush=True)
    A=sparse.load_npz(out/'joint_functional_assets/A_parent.npz');B=parent.global_B
    hrows=parent.native_rows['HC'];crows=parent.native_rows['cones']
    Q=native.Qfull.copy().tolil();Q[parent.h1_rows,:]=0;Q[:,parent.h1_rows]=0;Q=Q.tocsr()
    coo=Q.tocoo();Qg=sparse.csr_matrix((coo.data,(hrows[coo.row],hrows[coo.col])),shape=(nh,nh))
    linear=sparse.eye(nh)+50.*.020*Qg
    bg=parent.global_light;cone=cone_fixed_point(bg);scale=(cone['Vbar']/P['V_I'])**P['mu'];gain=np.full(nh,P['g_t'])
    def signals(h):return expit((cone['Vis']-B@h-P['V_k'])/P['V_n'])/scale
    def residual(h):return linear@h-gain*(A@signals(h)+ar)
    h=gain*(A@signals(np.zeros(nh))+ar)*.5;history=[]
    for iteration in range(200):
        guard();f=residual(h);error=float(np.max(abs(f)));history.append(error)
        if error<=1e-10:break
        q=expit((cone['Vis']-B@h-P['V_k'])/P['V_n']);d=q*(1-q)/(P['V_n']*scale)
        J=LinearOperator((nh,nh),matvec=lambda v:linear@v+gain*(A@(d*(B@v))),dtype=float)
        delta,status=gmres(J,-f,rtol=1e-10,atol=1e-14,restart=32,maxiter=200)
        if status:raise ValueError('Original Newton GMRES budget failed')
        for power in range(32):
            proposed=h+delta*.5**power
            if np.all(proposed>=0) and np.max(abs(residual(proposed)))<error:h=proposed;break
        else:raise ValueError('Original Newton line search failed')
    else:raise ValueError('Joint outer equilibrium failed')
    release=signals(h);localA=A[hrows][:,crows].tocsr();localB=parent.matrix('routes/HC_to_cones/mean_weights.npz')
    # Calculate exterior-only products directly; never subtract nearly equal DC sums.
    outside_c=np.ones(nc,bool);outside_c[crows]=False;outside_h=np.ones(nh,bool);outside_h[hrows]=False
    boundary=dict(A_release=np.asarray(A[hrows][:,outside_c]@release[outside_c])+ar[hrows],
        A_Vbar=np.asarray(A[hrows][:,outside_c]@cone['Vbar'][outside_c])+av[hrows],
        B_h=np.asarray(B[crows][:,outside_h]@h[outside_h]))
    sparse.save_npz(dest/'A_local.npz',localA);sparse.save_npz(dest/'H2_Q.npz',Q)
    np.savez_compressed(dest/'LOCAL_OPERATORS.npz',**boundary)
    configuration=dict(schema='GIN69_joint_functional_H1_constrained_BC',kernel_identity=matrix_id(A),
        frozen_optical_profile=bridge['profile'],exterior='approved static source grey extension; optical-axis flat reservoir',
        homogeneous_exterior_native_DC=True,conversion=conversion,H1_explicit_lateral=False,H2_original_Q=matrix_id(Q),
        outer_equilibrium_residual=error,precision='FP64',FIT_READY=False,background_qualified=False,production_adopted=False)
    (dest/'MODEL_CONFIGURATION.json').write_text(json.dumps(configuration,indent=2),encoding='utf-8')
    a=JointAssembly(parent,localA,boundary,Q,configuration)
    t=seed['time_seconds']
    for name,value in cone.items():seed['states']['cones','analytic_RE' if name in ('R','E') else 'reduced_native',name]=value[crows].copy()
    w=P['g_t']*(A[hrows[a.h1_rows]]@release+ar[hrows[a.h1_rows]])
    for name,value in [('w',w),('z2',w),('h1',h[hrows[a.h1_rows]]),('h2',h[hrows[a.h2_rows]])]:seed['states']['HC',MECHANISM,name]=value.copy()
    aux=a.populations['cones'].recording_auxiliary_group_rows
    seed['states']['cones','selected_original_z1_auxiliary','z1_aux']=P['g_t']*release[crows][aux]
    seed['observation_prehistory'].update(history_times=np.array([t]),history_values=h[hrows][None,:].copy(),prehistory=h[hrows].copy(),origin_seconds=t)
    seed['h1_h2_model'].update(artifact=a.artifact_sha256,boundary=boundary,actual_joint_configuration=configuration)
    seed['lateral_hc_interface'].update(configuration_sha256=a.lateral_configuration['candidate_sha256'],native_execution=a.lateral_configuration['selected_native_execution'])
    seed=initialize_inner(a,seed)
    a,seed=bind_bc_physiology(a,seed,ROOT,ROOT/'configurations/research/BC_PHYSIOLOGY_2026-11-02.json')
    seed['lateral_hc_interface']['binding_id']=a.parameter_binding_id
    identity=canonical_hash(dict(artifact=a.artifact_sha256,parameters=seed['inner_parameters'],configuration=configuration))
    save_bundle(dest/'seed',kind='source_construction_seed',identity=identity,payload=seed,
        qualification=dict(default_adopted=False,fit_eligible=False,explicit_research_authorization=True))
    result=dict(status='JOINT_BACKGROUND_CREATED',identity=identity,artifact=a.artifact_sha256,
        outer_residual=error,newton_residuals=history,wall_seconds=time.perf_counter()-began,
        H1_biexponential=True,BC_constraint=seed['bc_physiology']['configuration_sha256'],sealed_accessed=False)
    (dest/'RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
