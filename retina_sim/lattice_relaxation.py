"""Numerical repairs for the pinned retinalattice spring model.

The source initialization, nine-neighbor spacing lookup, normalized repulsive
spring law/gain, ellipse, density and point count are retained. Repairs concern
current-topology cache validity, Euler step inversion, projection, diagnostics
and double-precision coordinate/quality pairing; no biological hard core exists.
"""
import time
import numpy as np
from scipy.spatial import Delaunay,cKDTree
from .contracts import ContractError
from .lattice_quality import triangle_quality

ALGORITHM='pinned_retinalattice_springs_numerical_repair_v4_source_stop'
LEGACY_ALGORITHM='pinned_retinalattice_springs_numerical_repair_v3'
STOPPING_CONTRACT='pinned_source_OR_quality_width_or_internal_p30_v1'

def ellipse_distance(x,radius):
    return np.sqrt(x[:,0]**2+(x[:,1]/1.2247)**2)-radius

def project_ellipse(x,radius,*,max_steps=12):
    """Newton's implicit-surface projection with the missing |gradient|^2."""
    x=np.asarray(x,dtype=float).copy();count=0
    tol=64*np.finfo(float).eps*max(1.,float(radius))
    for _ in range(max_steps):
        d=ellipse_distance(x,radius);outside=d>tol
        if not np.any(outside):return x,count
        y=x[outside];r=d[outside]+radius
        gradient=np.c_[y[:,0]/r,y[:,1]/(1.2247**2*r)]
        x[outside]-=d[outside,None]*gradient/np.sum(gradient**2,axis=1)[:,None]
        count+=int(outside.sum())
    if np.max(ellipse_distance(x,radius))>tol:raise ContractError('Numerical ellipse projection failed')
    return x,count

def signed_double_area(x,triangles):
    p=x[triangles];a=p[:,1]-p[:,0];b=p[:,2]-p[:,0]
    return a[:,0]*b[:,1]-a[:,1]*b[:,0]

def finite_area_faces(x,triangles):
    """Reject floating-point zero-area Qhull faces, never points.

    The bound scales only with coordinate/edge rounding, not cone diameter,
    target spacing or a biological exclusion radius.
    """
    p=x[triangles];edges=p[:,1:]-p[:,:1]
    coordinate_scale=np.maximum(np.max(np.abs(p),axis=(1,2)),1.)
    edge_scale=np.maximum(np.max(np.abs(edges),axis=(1,2)),1.)
    roundoff=64*np.finfo(float).eps*coordinate_scale*edge_scale
    return np.abs(signed_double_area(x,triangles))>roundoff

def minimum_path_area(before,after,triangles):
    """Signed triangle area throughout the straight accepted point motion."""
    p=before[triangles];v=(after-before)[triangles]
    a=p[:,1]-p[:,0];b=p[:,2]-p[:,0];u=v[:,1]-v[:,0];w=v[:,2]-v[:,0]
    cross=lambda x,y:x[:,0]*y[:,1]-x[:,1]*y[:,0]
    q0=cross(a,b);q1=cross(u,b)+cross(a,w);q2=cross(u,w)
    sign=np.sign(q0);q0*=sign;q1*=sign;q2*=sign
    minimum=np.minimum(q0,q0+q1+q2)
    at=np.zeros(len(q0));positive=q2>0;at[positive]=-q1[positive]/(2*q2[positive])
    mid=positive&(at>0)&(at<1)
    minimum[mid]=np.minimum(minimum[mid],q0[mid]+q1[mid]*at[mid]+q2[mid]*at[mid]**2)
    return minimum

def swept_point_collision(before,after,tree):
    """Detect a machine-coincident encounter anywhere in a proposed motion.

    Search radius comes from the actual proposed movement, not anatomy. Hull
    membership changes are legal and need this point test rather than a fixed
    boundary-triangle orientation constraint.
    """
    movement=after-before;maximum=float(np.linalg.norm(movement,axis=1).max())
    tolerance=64*np.finfo(float).eps*max(1.,float(np.abs(before).max()),float(np.abs(after).max()))
    pairs=tree.query_pairs(2*maximum+tolerance,output_type='ndarray')
    if not len(pairs):return False,0
    a,b=pairs.T;r=before[a]-before[b];v=movement[a]-movement[b]
    vv=np.sum(v*v,axis=1);at=np.zeros(len(pairs));moving=vv>0
    at[moving]=np.clip(-np.sum(r[moving]*v[moving],axis=1)/vv[moving],0.,1.)
    distance=np.linalg.norm(r+at[:,None]*v,axis=1)
    return bool(np.any(distance<=tolerance)),len(pairs)

class SourceSpringField:
    def __init__(self,lookup_xy_um,lookup_spacing_um,radius_um,border_tolerance_um=.00215):
        self.lookup=np.asarray(lookup_xy_um,dtype=float);self.spacing=np.asarray(lookup_spacing_um,dtype=float).reshape(-1)
        self.tree=cKDTree(self.lookup);self.radius=float(radius_um);self.border_tolerance=float(border_tolerance_um)
    def targets(self,centers):
        d,idx=self.tree.query(centers,k=9);total=d.sum(1)
        if np.any(total<=0):raise ContractError('Invalid source spacing lookup')
        if not np.all(np.isfinite(self.spacing[idx])):
            raise ContractError('Source spring k9 lookup selected an unsupported density node; no extrapolation')
        # Literal source weights sum to8; global spring normalization cancels
        # that constant. Preserve it for source-force parity.
        w=(total[:,None]-d)/total[:,None]
        return np.sum(w*self.spacing[idx],axis=1)
    def topology(self,x):
        delaunay=Delaunay(x);tri=delaunay.simplices
        tri=tri[ellipse_distance(x[tri].mean(1),self.radius)<self.border_tolerance]
        valid=finite_area_faces(x,tri);self.last_roundoff_faces=int(np.sum(~valid));tri=tri[valid]
        hull=np.zeros(len(x),bool);hull[np.unique(delaunay.convex_hull)]=True
        self.interior_guard_triangles=tri[~np.any(hull[tri],axis=1)]
        self.hull_event_triangles=tri[np.any(hull[tri],axis=1)]
        springs=unique_spring_edges(tri,len(x))
        if len(np.unique(springs))!=len(x):raise ContractError('Numerical triangulation lost point coverage; no point deletion/jitter allowed')
        return tri,springs
    def force(self,x,springs):
        vector=x[springs[:,0]]-x[springs[:,1]];length=np.linalg.norm(vector,axis=1)
        if np.any(length<=0):raise ContractError('Numerically coincident spring endpoints; no jitter/delete fallback')
        desired=self.targets((x[springs[:,0]]+x[springs[:,1]])/2)
        desired*=np.sqrt(np.sum(length**2)/np.sum(desired**2))
        magnitude=np.maximum(1.1*desired-length,0.)
        pair=magnitude[:,None]/length[:,None]*vector;force=np.zeros_like(x)
        np.add.at(force,springs[:,0],pair);np.add.at(force,springs[:,1],-pair)
        return force,desired,length

def unique_spring_edges(triangles,point_count):
    """Source lexicographic unique pairs, with the same accumulation order.

    Int64 is necessary: point_row*N+neighbor_row exceeds int32 here. Sorting
    these keys is exactly equivalent to np.unique(unordered_pairs,axis=0).
    """
    pairs=np.sort(np.concatenate((triangles[:,[0,1]],triangles[:,[0,2]],triangles[:,[1,2]])),axis=1).astype(np.int64,copy=False)
    codes=np.unique(pairs[:,0]*point_count+pairs[:,1])
    return np.c_[codes//point_count,codes%point_count]


def quality_state(x,triangles):
    q=triangle_quality(x,triangles)
    p=np.percentile(q,[.8,3,7,15,99.8],method='hazen')
    return {'q_0_8':float(p[0]),'q_min':float(q.min()),'quantiles':p.tolist(),'triangles':len(q)}

def quality_widths(q):
    """The pinned source's histogram-edge width statistic, not a new score."""
    edges=np.arange(.5,1.00001,.01)
    bins=np.array([np.argmin(np.abs(edges-v)) for v in q['quantiles'][1:]])
    return edges[bins[-1]]-edges[bins[:3]]


def width_condition(widths):
    if len(widths)<2:return False
    # The source uses ALL width-history differences divided by the current
    # widths. Preserve its inf/NaN semantics rather than adding a new gate.
    with np.errstate(divide='ignore',invalid='ignore'):
        relative=np.diff(np.asarray(widths),axis=0)/widths[-1]
    return bool(np.any(relative>.08) and not np.any(np.isnan(relative)))


def source_motion_statistic(before,force,field):
    """Literal pinned-source statistic in microns, with nominal deltaT=0.2.

    smoothGrid returns d evaluated BEFORE boundary projection, at the proposed
    position. Its internal mask is d < -borderTolerance. MATLAB prctile's
    default exact percentile corresponds to NumPy's Hazen method. This is
    sqrt(deltaT)*|F|, not either actual or maximum displacement. Numerical
    backtracking must not reduce this nominal statistic into a false stop.
    """
    interior=ellipse_distance(before+.2*force,field.radius)<-field.border_tolerance
    values=np.sqrt(np.sum(.2*force[interior]**2,axis=1))
    return (float(np.percentile(values,30,method='hazen')) if len(values) else None),interior


def relax(initial_raw_um,field,*,max_iterations=300,max_wall_seconds=900.,d_tolerance_um=.000215,
          quality_threshold=.84,on_snapshot=None,on_progress=None,resume=None,on_checkpoint=None,on_step=None,
          allow_legacy_stop_contract_migration=False,checkpoint_interval_iterations=25):
    """Bound additional accepted steps; original maximum Euler step remains .2.

    A Delaunay graph is valid only for its current coordinates. Step rejection
    enforces orientation over the entire segment, without imposing a physical
    distance. Either source quality-plus-width OR source internal-p30 motion
    can stop. True maximum displacement remains a diagnostic, not a gate.
    """
    started=time.monotonic();prior={} if resume is None else resume['generation']
    if resume is not None:
        if prior['algorithm']!=ALGORITHM and not (allow_legacy_stop_contract_migration and prior['algorithm']==LEGACY_ALGORITHM):
            raise ContractError('Incompatible continuation algorithm or unapproved stop-contract migration')
        expected={'quality':quality_threshold,'d_tolerance_um':d_tolerance_um,'max_Euler_step':.2}
        if prior['thresholds']!=expected:raise ContractError('Continuation thresholds changed')
        x=np.array(resume['current_raw_um'],dtype=float,copy=True)
        if x.shape!=np.asarray(initial_raw_um).shape:raise ContractError('Continuation point count changed')
        projected_x,_=project_ellipse(x,field.radius)
        if not np.array_equal(projected_x,x):raise ContractError('Continuation would alter the saved state')
        projected=prior['projection_corrections_total']
    else:x,projected=project_ellipse(initial_raw_um,field.radius)
    tri,springs=field.topology(x);initial_roundoff_faces=field.last_roundoff_faces
    q=quality_state(x,tri);best=x.copy();best_q=q;best_iteration=0
    trace=[];widths=[];rejections=0;reason='iteration_budget';last_movement=None
    if resume is not None:
        if q!=prior['final_quality']:raise ContractError('Saved current coordinates and quality differ')
        best=np.array(resume['best_raw_um'],dtype=float,copy=True)
        best_q=quality_state(best,field.topology(best)[0])
        if best_q!=prior['selected_quality']:raise ContractError('Saved best coordinates and quality differ')
        tri,springs=field.topology(x)  # Restore the current state's guard topology.
        best_iteration=prior['selected_iteration'];trace=list(prior['trace'])
        if len(trace)!=prior['accepted_iterations']:raise ContractError('Incomplete continuation history')
        widths=[quality_widths(row) for row in trace]
        rejections=prior['backtracks_total'];last_movement=prior['actual_final_max_movement_um']
        initial_roundoff_faces=prior['initial_roundoff_zero_area_faces_excluded']
    offset=len(trace);previous_elapsed=prior.get('elapsed_seconds',0.)
    def result_record(stop_reason):
        return {'algorithm':ALGORITHM,'continuation_schema':2,'stopping_contract':STOPPING_CONTRACT,'stop_reason':stop_reason,
            'resume_algorithm':prior.get('algorithm'),'stop_contract_changed_at_iteration':offset if prior.get('algorithm')==LEGACY_ALGORITHM else prior.get('stop_contract_changed_at_iteration'),
            'source_motion_definition':'Hazen p30(sqrt(sum(0.2*F**2))) on ellipse_distance(before+0.2*F)<-0.00215 um; nominal dt irrespective of guard backtracking',
            'quality_history_definition':'All accepted fresh-topology quality samples, including retained v3 history; source width formula with no rolling window',
            'quality_sampling_cadence':'Each fresh topology under approved cache repair, rather than original delayed topology cache',
            'maximum_true_displacement_is_stop_gate':False,
            'accepted_iterations':len(trace),'accepted_iterations_this_segment':len(trace)-offset,
            'resumed_from_iteration':offset,'selected_iteration':best_iteration,
            'selected_quality':best_q,'final_quality':q,'actual_final_max_movement_um':last_movement,
            'backtracks_total':rejections,'projection_corrections_total':projected,
            'initial_roundoff_zero_area_faces_excluded':initial_roundoff_faces,
            'thresholds':{'quality':quality_threshold,'d_tolerance_um':d_tolerance_um,'max_Euler_step':.2},
            'point_count':len(x),'point_count_changed':len(x)!=len(initial_raw_um),
            'elapsed_seconds':previous_elapsed+time.monotonic()-started,
            'segment_elapsed_seconds':time.monotonic()-started,'physiological_validation':False,
            'best_coordinates_and_quality_evaluated_together':True,'trace':trace}
    # A crash after atomic checkpointing the triggering step must not cause
    # another accepted step when that already-terminal checkpoint is resumed.
    if trace and prior.get('algorithm')==ALGORITHM:
        terminal=('original_quality_plus_width_condition' if trace[-1]['quality_width_stop'] else
            'original_internal_p30_motion_condition' if trace[-1].get('source_motion_stop') else None)
        if terminal:
            result=result_record(terminal)
            if on_checkpoint:on_checkpoint(x,best,result)
            return best,x,result
    for iteration in range(offset+1,offset+max_iterations+1):
        if time.monotonic()-started>=max_wall_seconds:reason='wall_budget';break
        force,desired,length=field.force(x,springs)
        step=.2;before=x;guard_triangles=field.interior_guard_triangles;hull_triangles=field.hull_event_triangles
        tree=cKDTree(before)
        reference,_=project_ellipse(before+.2*force,field.radius)
        nominal_motion=float(np.linalg.norm(reference-before,axis=1).max())
        source_motion,source_interior=source_motion_statistic(before,force,field)
        source_motion_stop=source_motion is not None and source_motion<d_tolerance_um
        for backtrack in range(21):
            trial,projection_count=project_ellipse(before+step*force,field.radius)
            area=minimum_path_area(before,trial,guard_triangles)
            # Floating-point degeneracy only; no cone-diameter/min-spacing rule.
            area_scale=np.maximum(np.abs(signed_double_area(before,guard_triangles)),1.)
            collision,pair_checks=swept_point_collision(before,trial,tree)
            if np.all(area>64*np.finfo(float).eps*area_scale) and not collision:break
            step*=.5
        else:reason='numerical_step_stalled';break
        x=trial;rejections+=backtrack;projected+=projection_count
        movement=np.linalg.norm(x-before,axis=1);last_movement=float(movement.max())
        previous_springs=springs
        # Refresh before observing quality or using a changed Delaunay graph.
        tri,springs=field.topology(x);q=quality_state(x,tri)
        topology_changed=not np.array_equal(previous_springs,springs)
        # Literal source histogram-width test (histcounts returns bin edges).
        widths.append(quality_widths(q));condition_width=width_condition(widths)
        quality_stop=q['q_0_8']>quality_threshold and condition_width
        row={'iteration':iteration,**q,'actual_max_movement_um':last_movement,
            'nominal_source_step_max_movement_um':nominal_motion,
            'actual_p30_movement_um':float(np.percentile(movement,30,method='hazen')),
            'source_interior_p30_motion_um':source_motion,'source_interior_point_count':int(source_interior.sum()),
            'source_motion_nominal_dt':.2,'source_motion_stop':source_motion_stop,
            'true_interior_p30_nominal_displacement_um':float(np.percentile(np.linalg.norm(reference[source_interior]-before[source_interior],axis=1),30,method='hazen')) if np.any(source_interior) else None,
            'accepted_step':step,'backtracks':backtrack,'topology_changed':topology_changed,
            'projection_corrections':projection_count,'minimum_path_signed_area_um2':float(area.min()) if len(area) else None,
            'roundoff_zero_area_faces_excluded':field.last_roundoff_faces,
            'legal_hull_orientation_events':int(np.sum(minimum_path_area(before,trial,hull_triangles)<=0)),
            'swept_point_collision_pairs_checked':pair_checks,
            'quality_width_stop':quality_stop,'histogram_widths':widths[-1].tolist(),
            'width_condition':condition_width,'maximum_movement_source_row':int(movement.argmax()),
            'maximum_movement_radius_um':float(np.linalg.norm(x[movement.argmax()])),
            'elapsed_seconds':previous_elapsed+time.monotonic()-started}
        if on_step:on_step(before,x,previous_springs,springs,force,desired,row)
        trace.append(row)
        if q['q_0_8']>=best_q['q_0_8']:
            best=x.copy();best_q=q;best_iteration=iteration
        if on_progress and (iteration==1 or iteration%10==0):on_progress(row)
        if on_snapshot and (iteration%25==0 or quality_stop):on_snapshot(best,best_iteration,best_q,trace)
        if on_checkpoint and (iteration%checkpoint_interval_iterations==0 or quality_stop or source_motion_stop):on_checkpoint(x,best,result_record('checkpoint'))
        if quality_stop:reason='original_quality_plus_width_condition';break
        if source_motion_stop:reason='original_internal_p30_motion_condition';break
    result=result_record(reason)
    if on_checkpoint:on_checkpoint(x,best,result)
    return best,x,result
