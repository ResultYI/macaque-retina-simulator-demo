"""CPU-only static graph algebra, identity, provenance and contact regressions."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from retina_sim.contracts import ContractError
from retina_sim.spatial_group_graph import (SCHEMA, polar_cells, group_members,
    aggregate_edges, mean_matrix, moments, gap_laplacian, verified_path, load_static_graph)


def test_polar_boundaries_adjacent_ULP_and_angle_seam():
    bands=[dict(outer_deg=.5,radial=2,angular=4),dict(outer_deg=2.,radial=2,angular=8)]
    radius=np.array([0.,.5,np.nextafter(.5,0),np.nextafter(.5,np.inf),2.])
    xy=np.column_stack([radius,np.zeros_like(radius)])
    np.testing.assert_array_equal(polar_cells(xy,bands,1.),[0,4,4,8,16])
    np.testing.assert_array_equal(polar_cells([[.1,-0.],[.1,0.],[0.,.1],[-.1,0.],[0.,-.1]],bands,1.),[0,0,1,2,3])
    with pytest.raises(ContractError,match='outside'):
        polar_cells([[np.nextafter(2.,np.inf),0]],bands,1.)
    with pytest.raises(ContractError):
        polar_cells([[np.nan,0]],bands,1.)


def test_membership_keeps_subtypes_parent_classes_and_empty_slots_absent():
    rows=np.array([2,4,7,9,11,12])
    q=group_members(rows,np.array([3]*6),subtype=np.array([0,0,1,1,0,0]),
        parent=np.array([1,1,2,2,3,1]),eligibility=np.array([1,1,1,0,1,1]))
    assert len(q['counts'])==4 and sorted(q['counts'])==[1,1,1,3]
    assert set(q['member_global_rows'])==set(rows)
    np.testing.assert_array_equal(np.diff(q['member_offsets']),q['counts'])
    for i in range(len(q['counts'])):
        expected=rows[q['fine_to_group']==i]
        np.testing.assert_array_equal(q['member_global_rows'][q['member_offsets'][i]:q['member_offsets'][i+1]],expected)


def test_aggregation_matches_RWP_and_preserves_empty_and_partial_DC_rows():
    w=sparse.csr_matrix([[.25,.75,0],[0,0,0],[0,.4,.6],[0,0,0]])
    post=np.array([0,0,1,2]);pre=np.array([0,0,1])
    c=w.tocoo()
    total=aggregate_edges(post[c.row],pre[c.col],c.data,(3,2))
    p=sparse.csr_matrix((np.ones(3),(np.arange(3),pre)),shape=(3,2))
    r=sparse.csr_matrix((1/np.bincount(post)[post],(post,np.arange(4))),shape=(3,4))
    np.testing.assert_allclose(mean_matrix(total,np.bincount(post)).toarray(),(r@w@p).toarray(),rtol=0,atol=1e-16)
    np.testing.assert_allclose(np.asarray(mean_matrix(total,np.bincount(post)).sum(1)).ravel(),[.5,1,0])
    with pytest.raises(ContractError):
        aggregate_edges(np.array([0]),np.array([0]),[-1.],(1,1))


def test_gap_projection_cancels_internal_edges_and_preserves_total_current():
    fine=sparse.csr_matrix([[0,2,1],[2,0,4],[1,4,0]],dtype=float)
    groups=np.array([0,0,1]);p=sparse.csr_matrix((np.ones(3),(np.arange(3),groups)),shape=(3,2))
    c=fine.tocoo();adj=aggregate_edges(groups[c.row],groups[c.col],c.data,(2,2))
    lap=gap_laplacian(adj)
    expected=p.T@(sparse.diags(np.asarray(fine.sum(1)).ravel())-fine)@p
    np.testing.assert_array_equal(lap.toarray(),expected.toarray())
    np.testing.assert_array_equal(lap.toarray(),[[5,-5],[-5,5]])
    assert np.sum(lap@np.array([-.06,-.04]))==pytest.approx(0,abs=1e-15)
    with pytest.raises(ContractError,match='reciprocity'):
        gap_laplacian(sparse.csr_matrix([[0,1],[2,0]]))


def test_moments_preserve_mean_and_within_group_reconstruction_error():
    v=np.array([1e9+1,1e9-1,7.]);g=np.array([0,0,1]);n=np.array([2,1])
    mean,var,lo,hi=moments(v,g,n)
    np.testing.assert_array_equal(mean,[1e9,7.])
    np.testing.assert_array_equal(var,[1.,0.])
    assert np.sum(var*n)==np.sum((v-mean[g])**2)
    np.testing.assert_array_equal(hi-lo,[2.,0.])


def test_verified_source_and_static_loader_reject_tampering(tmp_path):
    p=tmp_path/'a.npy';np.save(p,np.array([1.,2.]))
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert verified_path(tmp_path,dict(path='a.npy',sha256=digest))==p
    raw=dict(schema=SCHEMA,runtime_ready=False,model_forward_started=False,
        files={'a.npy':dict(bytes=p.stat().st_size,sha256=digest)})
    expected=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(raw,manifest_sha256=expected)))
    assert load_static_graph(tmp_path)['schema']==SCHEMA
    p.write_bytes(p.read_bytes()+b'x')
    with pytest.raises(ContractError):
        load_static_graph(tmp_path)
    with pytest.raises(ContractError):
        verified_path(tmp_path,dict(path='../a.npy',sha256=digest))


@pytest.mark.parametrize('component,zero_weights',[
    ('RBC_AII_AMPA',False),('individual_rod_RBC_contacts',False),('individual_rod_RBC_contacts',True)])
def test_actual_route_builder_uses_actual_profile_and_contact_coefficients(tmp_path,component,zero_weights):
    from tools.build_spatial_group_graph import Builder
    source=tmp_path/'block.npz'
    weights=np.array([0.,0.]) if zero_weights else np.array([.99,.01])
    np.savez(source,data=weights,indices=np.array([0,1]),indptr=np.array([0,2]),
        shape=np.array([1,2]),row_status=np.array(['READY_FIXTURE']))
    overlap=tmp_path/'overlap.npy';np.save(overlap,np.array([.4,.8]))
    binding=lambda p:dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    b=Builder.__new__(Builder);b.root=tmp_path;b.work=tmp_path/'out';b.work.mkdir();b.routes=[]
    b.check=lambda **kw:None
    b.groups={'pre':dict(global_map=np.array([0,0]),counts=np.array([2])),
              'post':dict(global_map=np.array([0]),counts=np.array([1]))}
    route=dict(path='route.json',sha256='fixture')
    block=dict(binding(source),start=0,stop=1)
    graph={'contact_geometry':{component:dict(route=route,binding={'path':'contact.json'})}}
    b.read=lambda row:dict(blocks=[block]) if row==route else dict(overlap=binding(overlap),count=2)
    b.array=lambda row:np.load(tmp_path/row['path'],mmap_mode='r')
    profile='effective_individual_TRPM1' if component=='individual_rod_RBC_contacts' else 'effective_'+component
    b.route(dict(pre='pre',post='post',kind='primary_chemical',binding=route),graph,
        {profile:{'g_peak':{'value':2e-10}}})
    result=b.routes[0]
    expected=2e-9*weights.sum() if component=='individual_rod_RBC_contacts' else 2e-10*1.2
    assert result['contact']['coefficient_sum']==pytest.approx(expected,rel=1e-15)
    assert result['contact']['groups']==1 and result['contact']['fine_contacts']==2
    np.testing.assert_array_equal(np.load(b.work/'routes/pre_to_post/fine_contact_to_group.npy'),[0,0])
    assert not (b.work/'manifest.json').exists()


def test_static_module_import_does_not_import_torch():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-B','-c',
        "import tools.build_spatial_group_graph,sys; assert 'torch' not in sys.modules"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
