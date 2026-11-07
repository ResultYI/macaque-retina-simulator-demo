"""Bounded manufactured finite-parent experiment; never opens fitted recordings."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import argparse
import json
import hashlib
import time
import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.special import expit

from retina_sim.contracts import canonical_hash
from retina_sim.hc_biexponential import build_spatial_operators, project_spatial_operators
from retina_sim.h1_h2_effective import H1H2Native, joint_equilibrium, matrix_id
from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
from retina_sim.vh2005 import SOURCE_VALUES


def native(A, B, identity, light):
    nc, nh = B.shape
    pars=dict(tau_seconds=SOURCE_VALUES['tau_h']*.001,input_scale=SOURCE_VALUES['g_t'],feedback_scale=1.)
    pop=lambda n:SimpleNamespace(group_count=n,group_ids=np.arange(n))
    cones=pop(nc);cones.recording_auxiliary_group_rows=np.array([],dtype=int)
    cones.recording_auxiliary_fine_rows=np.array([],dtype=int)
    assembly=SimpleNamespace(populations=dict(cones=cones,HC=pop(nh)),
        matrix=lambda path:A if 'cones_to_HC' in path else B,
        artifact_sha256=identity,h1_rows=np.array([0]),h2_rows=np.array([1]),
        h2_parameters=pars,layout=dict(manufactured=True),connection_identity=dict(identity=identity))
    n=H1H2Native(assembly)
    n.boundary=dict(A_release=np.zeros(nh),A_Vbar=np.zeros(nh),B_h=np.zeros(nc))
    eq=joint_equilibrium(A,B,assembly.h1_rows,assembly.h2_rows,light,pars)
    y=np.concatenate([eq['cone'][k] for k in ('X','C','Vis','g_i','Vbar')]+[eq['w'],eq['z2'],eq['h'],np.array([])])
    anchor=AnalyticREAnchor(0.,light,light,light,.16,0)
    error=float(np.max(np.abs(n.rhs(y,0.,anchor))))
    if error>1e-8:raise RuntimeError('Incompatible native background')
    return n,y,error


def integrate(n,y,light,stimulus,start=0.,end=.12,anchor=None):
    anchor=anchor or AnalyticREAnchor(0.,light,light,stimulus,.16,0)
    times=np.linspace(start,end,61)
    solution=solve_ivp(lambda t,v:n.rhs(v,t,anchor),(start,end),y,method='DOP853',
        rtol=1e-7,atol=1e-10,max_step=.0002,t_eval=times)
    if not solution.success or not np.isfinite(solution.y).all():raise RuntimeError(solution.message)
    p=SOURCE_VALUES
    release=expit((solution.y[n.slices['Vis']]-n.B@solution.y[n.slices['Vh']]-p['V_k'])/p['V_n'])
    release/=(solution.y[n.slices['Vbar']]/p['V_I'])**p['mu']
    return solution,release,anchor


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];began=datetime.now(timezone.utc).isoformat();clock=time.perf_counter()
    cfg=json.loads((root/'configurations/research/HC_biexponential_vh2007_reference_v1.json').read_bytes())
    # Dense, regular manufactured disk; only two central HCs exist in this fixture.
    axis=np.arange(-180,181)*.02;xx,yy=np.meshgrid(axis,axis)
    xy=np.c_[xx.ravel(),yy.ravel()];xy=xy[np.linalg.norm(xy,axis=1)<=3.6+1e-12]
    r=np.linalg.norm(xy,axis=1)
    index=np.rint(xy/.02).astype(int);label=(index[:,0]+2*index[:,1])%10
    ct=np.where(label==1,'S',np.where(label%2==0,'L','M'))
    ht=np.array(['H1','H2']);hx=np.zeros((2,2))
    local=r<=.05
    raw=sparse.csr_matrix(np.vstack([local&(ct!='S'),local]).astype(float))
    result=build_spatial_operators(raw,xy,hx,ct,ht,configuration=cfg,parent_disk_radius_mm=3.6)
    assert result['support_certified'].all()
    uniform_cfg=dict(cfg,H1=None)
    old=build_spatial_operators(raw,xy,hx,ct,ht,configuration=uniform_cfg,parent_disk_radius_mm=3.6)
    # Group by annulus, subtype and local feedback eligibility; preserve every member.
    ring=np.floor(r/.04+1e-10).astype(int)
    keys,cm=np.unique(np.c_[ring,label==1,label%2,local],axis=0,return_inverse=True)
    counts=np.bincount(cm);hm=np.arange(2)
    projected={name:project_spatial_operators(value,cm,hm,counts,np.ones(2,dtype=int))
               for name,value in [('uniform',old),('biexponential',result)]}
    np.savez_compressed(out/'GEOMETRY_MEMBERS.npz',cone_inner_segment_mm=xy,cone_pedicle_mm=xy,
        hc_anchor_mm=hx,cone_types=ct,hc_types=ht,cone_to_group=cm,cone_counts=counts,
        hc_to_group=hm,group_keys=keys)
    for name,value in [('reference',result),('uniform',old)]:
        for key in ('A','B','short','long'):sparse.save_npz(out/(name+'_'+key+'.npz'),value[key])
    # Spatial transfer of A in the stimulus (IS) coordinate system, before feedback.
    frequencies=np.r_[0.,np.geomspace(.03,20,100)]
    spatial={name:np.array([value['A']@np.cos(2*np.pi*f*xy[:,0]) for f in frequencies])
             for name,value in [('uniform',old),('biexponential',result)]}
    bins=np.arange(91)*.04
    radial={name:np.array([value['A']@(ring==k).astype(float) for k in range(91)])
            for name,value in [('uniform',old),('biexponential',result)]}
    np.savez_compressed(out/'SPATIAL.npz',frequency_cycles_per_mm=frequencies,radius_mm=bins,
        **{'frequency_'+k:v for k,v in spatial.items()},**{'annular_mass_'+k:v for k,v in radial.items()})
    light=np.full(len(counts),1000.)
    masks=dict(center=keys[:,0]==0,annulus=(keys[:,0]>=3)&(keys[:,0]<8),
               fullfield=np.ones(len(counts),bool),outer_spot=(keys[:,0]>=8)&(keys[:,0]<9))
    center=int(cm[np.argmin(r)])
    summary={};identities={};checkpoint={}
    for condition in ('uniform','biexponential','feedback_off'):
        matrices=projected['uniform' if condition=='uniform' else 'biexponential']
        A=matrices['A'];B=matrices['B'] if condition!='feedback_off' else sparse.csr_matrix(matrices['B'].shape)
        identity=canonical_hash(dict(parent=matrices['identity'],B=matrix_id(B),parameters=SOURCE_VALUES))
        n,y,error=native(A,B,identity,light);identities[condition]=dict(artifact=identity,native=n.source_version)
        np.savez_compressed(out/(condition+'_background.npz'),y=y,light=light)
        base=expit((y[n.slices['Vis']]-B@y[n.slices['Vh']]-SOURCE_VALUES['V_k'])/SOURCE_VALUES['V_n'])
        base/=(y[n.slices['Vbar']]/SOURCE_VALUES['V_I'])**SOURCE_VALUES['mu']
        summary[condition]=dict(background_max_rhs=error,center_release_baseline=float(base[center]),experiments={})
        for name,mask in masks.items():
            stimulus=light*(1+.01*mask)
            sol,release,anchor=integrate(n,y,light,stimulus)
            dh=sol.y[n.slices['Vh']]-y[n.slices['Vh'],None]
            dr=release[center]-base[center]
            np.savez_compressed(out/(condition+'_'+name+'.npz'),t=sol.t,y=sol.y,
                release=release,hc_delta=dh,center_release_delta=dr,light=stimulus)
            summary[condition]['experiments'][name]=dict(center_release_delta_final=float(dr[-1]),
                center_release_delta_peak=float(dr[np.argmax(np.abs(dr))]),hc_delta_final=dh[:,-1].tolist(),nfev=sol.nfev)
            if condition=='biexponential' and name=='annulus':
                mid=30;t=float(sol.t[mid]);R,E=anchor.at(t)
                np.savez_compressed(out/'native_checkpoint.npz',y=sol.y[:,mid],R=R,E=E,
                    light=stimulus,time=t,stop=.16,epoch=0,identity=n.source_version)
                with np.load(out/'native_checkpoint.npz',allow_pickle=False) as z:
                    if str(z['identity'])!=n.source_version:raise RuntimeError('Checkpoint identity differs')
                    restart=AnalyticREAnchor(float(z['time']),z['R'],z['E'],z['light'],float(z['stop']),int(z['epoch']))
                    resumed,_,_=integrate(n,z['y'],light,stimulus,start=t,anchor=restart)
                difference=float(np.max(abs(resumed.y[:,-1]-sol.y[:,-1])))
                if difference>1e-7:raise RuntimeError('Checkpoint numerical restart differs')
                checkpoint=dict(maximum_final_state_difference=difference,
                    scope='native state + analytic R/E restart only; no grouped scheduler checkpoint qualification')
        print(condition+' complete',flush=True)
    on=summary['biexponential']['experiments']['annulus']['center_release_delta_final']
    off=summary['feedback_off']['experiments']['annulus']['center_release_delta_final']
    mechanism_pass=bool(on>0 and abs(off)<=1e-10)
    payload=dict(started_utc=began,ended_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-clock,
        status='MANUFACTURED_NATIVE_MECHANISM_PASS' if mechanism_pass else 'MANUFACTURED_SURROUND_SIGN_FAIL',
        mechanism_pass=mechanism_pass,real_tissue_qualification='BLOCKED',
        scientific_scope='Regular synthetic cone disk, two central HCs; remote cones have no local HC feedback. Not a retina mosaic or physiological fit.',
        IS_pedicle_relation='identical manufactured coordinates; not a Schein physiological validation',
        configuration=cfg,identities=identities,counts=dict(fine_cones=len(xy),cone_groups=len(counts),HC=2),
        parameters=SOURCE_VALUES,contrast=.01,background_td=1000.,duration_seconds=.12,
        solver=dict(method='DOP853',rtol=1e-7,atol=1e-10,max_step=.0002),
        support_radius_mm=result['support_radius_mm'].tolist(),experiments=summary,checkpoint=checkpoint,
        experiment_A='Static fine A impulse/radial/spatial-frequency transfer plus native annular stimuli; matched physiological HC RF not completed',
        experiment_B='Manufactured finite-parent native fixed-point/on-off mechanism only',
        experiment_C='NOT_RUN: current tissue upstream support/closure and background not qualified',
        production_adopted=False,physiologically_calibrated=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'RESULT.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    if not mechanism_pass:raise SystemExit('Surround sign gate failed; full failure receipt saved, no physiological qualification')


if __name__=='__main__':main()
