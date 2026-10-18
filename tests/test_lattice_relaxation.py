import numpy as np
from retina_sim.lattice_relaxation import project_ellipse,ellipse_distance,minimum_path_area,SourceSpringField,quality_state,finite_area_faces,swept_point_collision

def test_projection_uses_implicit_gradient_normalization():
    r=800.;x=np.array([[0.,1.2247*(r+10.)],[r+10.,0.],[600.,900.]])
    projected,n=project_ellipse(x,r)
    assert n and np.max(ellipse_distance(projected,r))<1e-10
    np.testing.assert_allclose(projected[0],[0.,1.2247*r],rtol=0,atol=1e-10)

def test_motion_guard_catches_inversion_and_midstep_collapse():
    x=np.array([[0.,0.],[1.,0.],[0.,1.]])
    swap=np.array([[0.,0.],[0.,1.],[1.,0.]])
    assert minimum_path_area(x,swap,[[0,1,2]])[0]<=0
    reverse=-x
    # Both endpoints have positive orientation but the midpoint collapses.
    assert minimum_path_area(x,reverse,[[0,1,2]])[0]<=0
    assert minimum_path_area(x,x+.01,[[0,1,2]])[0]>0

def test_constant_lookup_preserves_source_normalized_spring_law():
    grid=np.array([(x,y) for x in (-2.,0.,2.) for y in (-2.,0.,2.)]);spacing=np.ones(9)*4
    field=SourceSpringField(grid,spacing,800.)
    x=np.array([[0.,0.],[1.,0.]]);force,desired,length=field.force(x,np.array([[0,1]]))
    np.testing.assert_allclose(desired,[1.],rtol=1e-15,atol=0)
    np.testing.assert_allclose(force,[[-.1,0],[.1,0]],rtol=0,atol=2e-16)
    assert length[0]==1.

def test_quality_is_from_the_supplied_double_precision_snapshot():
    x=np.array([[0.,0.],[1.,0.],[.5,np.sqrt(3)/2]])
    record=quality_state(x,np.array([[0,1,2]]))
    assert abs(record['q_0_8']-1.)<1e-15 and abs(record['q_min']-1.)<1e-15

def test_roundoff_face_filter_preserves_all_point_identities():
    x=np.array([[800.,800.],[801.,800.],[802.,800.],[801.,801.]])
    tri=np.array([[0,1,2],[0,1,3],[1,2,3]])
    valid=finite_area_faces(x,tri)
    assert valid.tolist()==[False,True,True]
    assert np.unique(tri[valid]).tolist()==[0,1,2,3]

def test_swept_collision_distinguishes_legal_hull_change_from_point_collision():
    from scipy.spatial import cKDTree
    x=np.array([[0.,0.],[2.,0.],[1.,1.]])
    hull_change=np.array([[0.,0.],[2.,0.],[1.,-1.]])
    assert not swept_point_collision(x,hull_change,cKDTree(x))[0]
    swapped=x.copy();swapped[[0,1]]=x[[1,0]]
    assert swept_point_collision(x,swapped,cKDTree(x))[0]


def test_continuation_matches_uninterrupted_coordinates_and_stopping_history():
    from retina_sim.lattice_relaxation import relax
    rng=np.random.default_rng(41)
    x=np.array([(a,b) for a in range(7) for b in range(7)],dtype=float)+rng.uniform(-.1,.1,(49,2))
    lookup=np.array([(a,b) for a in (-10.,0.,10.) for b in (-10.,0.,10.)])
    def field():return SourceSpringField(lookup,np.ones(9),800.)
    kw=dict(quality_threshold=2.,d_tolerance_um=0.)
    full_best,full_end,full=relax(x,field(),max_iterations=10,**kw)
    first_best,first_end,first=relax(x,field(),max_iterations=4,**kw)
    resumed_best,resumed_end,resumed=relax(x,field(),max_iterations=6,**kw,
        resume={'current_raw_um':first_end,'best_raw_um':first_best,'generation':first})
    np.testing.assert_array_equal(resumed_end,full_end)
    np.testing.assert_array_equal(resumed_best,full_best)
    assert resumed['selected_iteration']==full['selected_iteration']
    for a,b in zip(full['trace'],resumed['trace']):
        assert {k:v for k,v in a.items() if k!='elapsed_seconds'}=={k:v for k,v in b.items() if k!='elapsed_seconds'}


def test_source_motion_uses_nominal_dt_preprojection_mask_and_hazen_p30():
    from retina_sim.lattice_relaxation import source_motion_statistic
    class Domain:
        radius=10.;border_tolerance=.00215
    x=np.c_[np.array([0.,1.,2.,3.,4.,9.999]),np.zeros(6)]
    f=np.c_[np.array([1.,2.,3.,4.,5.,1000.]),np.zeros(6)]
    value,mask=source_motion_statistic(x,f,Domain())
    assert mask.tolist()==[True,True,True,True,True,False]
    # Hazen p30 of [1,2,3,4,5] is 2, whereas linear p30 is 2.2.
    np.testing.assert_allclose(value,2*np.sqrt(.2),rtol=1e-15)
    assert not np.isclose(value,.2*2)  # The source statistic is not displacement.


def test_source_motion_branch_stops_even_with_large_maximum_and_failed_quality():
    from retina_sim.lattice_relaxation import relax
    grid=np.array([(a,b) for a in range(3) for b in range(3)],dtype=float)
    lookup=np.array([(a,b) for a in (-10.,0.,10.) for b in (-10.,0.,10.)])
    class OneMovingPoint(SourceSpringField):
        def force(self,x,springs):
            f=np.zeros_like(x);f[4,0]=.01
            return f,np.ones(len(springs)),np.linalg.norm(x[springs[:,0]]-x[springs[:,1]],axis=1)
    best,current,result=relax(grid,OneMovingPoint(lookup,np.ones(9),800.),max_iterations=4,quality_threshold=2.)
    assert result['stop_reason']=='original_internal_p30_motion_condition'
    assert result['accepted_iterations']==1
    row=result['trace'][-1]
    assert row['source_motion_stop'] and not row['quality_width_stop']
    assert row['actual_max_movement_um']>result['thresholds']['d_tolerance_um']
    # Simulate interruption after the triggering atomic checkpoint, before
    # the worker can write its final record. Resume must not take another step.
    checkpoint=dict(result,stop_reason='checkpoint')
    _,again,resumed=relax(grid,OneMovingPoint(lookup,np.ones(9),800.),max_iterations=4,quality_threshold=2.,
        resume={'current_raw_um':current,'best_raw_um':best,'generation':checkpoint})
    assert resumed['accepted_iterations']==1 and resumed['accepted_iterations_this_segment']==0
    np.testing.assert_array_equal(again,current)


def test_quality_branch_does_not_require_movement_branch(monkeypatch):
    import retina_sim.lattice_relaxation as module
    monkeypatch.setattr(module,'width_condition',lambda widths:True)
    rng=np.random.default_rng(41)
    x=np.array([(a,b) for a in range(5) for b in range(5)],dtype=float)+rng.uniform(-.1,.1,(25,2))
    lookup=np.array([(a,b) for a in (-10.,0.,10.) for b in (-10.,0.,10.)])
    _,_,result=module.relax(x,SourceSpringField(lookup,np.ones(9),800.),max_iterations=4,quality_threshold=0.,d_tolerance_um=0.)
    assert result['stop_reason']=='original_quality_plus_width_condition'
    assert result['accepted_iterations']==1
    assert result['trace'][-1]['quality_width_stop'] and not result['trace'][-1]['source_motion_stop']


def test_width_condition_retains_whole_history_and_source_nan_inf_semantics():
    from retina_sim.lattice_relaxation import width_condition
    assert width_condition([np.array(v) for v in [[.3,.2,.1],[.2,.3,.1],[.1,.1,.1],[.1,.1,.1]]])
    assert not width_condition([np.zeros(3),np.zeros(3)])
    assert width_condition([np.array(v) for v in [[.3,.3,.1],[.2,.2,.2],[.1,.1,0.]]])


def test_encoded_edges_preserve_full_master_pair_identity_and_order():
    from retina_sim.lattice_relaxation import unique_spring_edges
    tri=np.array([[0,70000,120000],[70000,120000,121810],[120000,0,70000]],dtype=np.int32)
    expected=np.unique(np.sort(np.concatenate((tri[:,[0,1]],tri[:,[0,2]],tri[:,[1,2]])),axis=1),axis=0)
    np.testing.assert_array_equal(unique_spring_edges(tri,121811),expected)
