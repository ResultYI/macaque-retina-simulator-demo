"""Domain/identity and unchanged source arithmetic for the approved disk build."""
from pathlib import Path
import numpy as np
import pytest
from retina_sim.atlas import CanonicalCellTable,AtlasROI
from retina_sim.disk_geometry import DiskAtlasROI
from retina_sim.contracts import Quantity,ContractError
from retina_sim.source_atlas import SourceDensityAtlas
from retina_sim.lattice_relaxation import SourceSpringField

ROOT=Path(__file__).resolve().parents[1]

def test_disk_queries_keep_global_ids_and_exclude_square_corners():
    xy=np.array([[-3.,0.],[0.,0.],[3.,0.],[3.1,0.],[2.9,2.9],[-1.,1.]])
    table=CanonicalCellTable('atlas','1','fixed','cones',tuple(map(str,range(len(xy)))),
        {'rf':Quantity(xy,'mm','functional_position','synthetic_fixture','functional_visual_field')},'reference')
    roi=DiskAtlasROI((-3.,3.,-3.,3.),(-3.3,3.3,-3.3,3.3),'reference',query_radius_mm=3.3)
    result=table.query(roi);again=table.query(roi)
    assert result.source_keys==('0','1','2','3','5') and result.source_keys==again.source_keys
    np.testing.assert_array_equal(result.owner,[True,True,True,False,True])
    assert roi.maximum_owner_radius()==3.
    left=DiskAtlasROI((-3.,0.,-3.,3.),roi.query_bounds_mm,'reference',query_radius_mm=3.3)
    right=DiskAtlasROI((0.,3.,-3.,3.),roi.query_bounds_mm,'reference',query_radius_mm=3.3)
    assert not np.any(left.owner_mask(xy)&right.owner_mask(xy))
    np.testing.assert_array_equal(table.query(left).coordinate('rf').values,table.query(right).coordinate('rf').values)

def test_large_density_batch_preserves_existing_cardinal_and_central_rules():
    field=SourceDensityAtlas(ROOT,profile_id='retina_atlas_Schein_bridge_research_v1').cones
    xy=np.array([[0.,0.],[.01,.02],[.04,-.02],[.35,0.],[0.,3.],[3.,0.],[-3.,0.],[2.,2.],[-2.,-2.]])
    expected=field(Quantity(xy,'mm','functional_position','project_approximation','functional_visual_field'))
    np.testing.assert_allclose(field.array_values(xy),expected,rtol=4e-15,atol=0)
    with pytest.raises(ContractError,match='no extrapolation'):field.array_values([[7.,0.]])

def test_source_k9_math_preserved_and_undefined_selected_nodes_rejected():
    xy=np.array(np.meshgrid(np.arange(-2.,3.),np.arange(-2.,3.))).reshape(2,-1).T
    field=SourceSpringField(xy,np.ones(len(xy)),100.)
    np.testing.assert_allclose(field.targets([[0.,0.]]),[8.],rtol=0,atol=1e-15)
    invalid=np.ones(len(xy));invalid[np.argmin(np.linalg.norm(xy,axis=1))]=np.nan
    with pytest.raises(ContractError,match='unsupported density node'):
        SourceSpringField(xy,invalid,100.).targets([[0.,0.]])

def test_exact_hc_rule_cache_keeps_values_and_original_backend():
    from retina_sim import hc_mosaic
    from retina_sim.geometry import RetinalDomain
    from retina_sim.disk_quadrature import memoized_hc_rules
    original=hc_mosaic._Quadrature;profile=hc_mosaic.HCDensityProfile(ROOT)
    bounds=(.3,.4,-.05,.05);domain=RetinalDomain(bounds,bounds,lambda xy:np.ones(len(xy),bool),'fixture','fixture')
    q=original(profile,domain,8,.001,.00001,12,100000)
    expected=q.rule(bounds,8)
    with memoized_hc_rules() as instances:
        cached=hc_mosaic._Quadrature(profile,domain,8,.001,.00001,12,100000)
        first=cached.rule(bounds,8);count=cached.evaluated_points;second=cached.rule(bounds,8)
        np.testing.assert_array_equal(first,expected);np.testing.assert_array_equal(second,expected)
        first[0]=0
        np.testing.assert_array_equal(cached.rule(bounds,8),expected)
        assert cached.evaluated_points==count and cached.rule_cache_hits==2
    assert hc_mosaic._Quadrature is original

def test_fixed_cone_tile_index_preserves_original_boolean_selection():
    import json
    from retina_sim.isetbio_geometry import FixedISETConeMaster
    profile=json.loads((ROOT/'configurations/retina_atlas_isetbio_source_stop_v1.json').read_bytes())
    master=FixedISETConeMaster(ROOT,profile['cone_master']);size=.032
    for tx,ty in [(0,0),(-1,-1),(10,-1),(-16,15)]:
        xy=master.positions;x0,x1,y0,y1=tx*size,(tx+1)*size,ty*size,(ty+1)*size
        keep=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
        ids,actual=master.tile(tx,ty,size)
        assert ids==tuple(k for k,v in zip(master.source_keys,keep) if v)
        np.testing.assert_array_equal(actual,xy[keep])

def test_disk_source_integrals_and_field_identity():
    from scipy.integrate import quad
    from retina_sim.population_revision import ApprovedPackerRodDensity,ApprovedParasolRFDensity
    from retina_sim.disk_population import IntegratedDiskDomain
    from retina_sim.geometry import integrate_density
    radius=3.17552;field=lambda xy:np.full(len(xy),5000.)
    mass=np.pi*radius**2*5000
    domain=IntegratedDiskDomain((-radius,radius,-radius,radius),lambda xy:np.sum(xy**2,axis=1)<=radius**2,
        'test',field,mass,'constant_times_area')
    assert integrate_density(field,domain)['rounding_target']==158398
    with pytest.raises(ContractError,match='different density'):integrate_density(lambda xy:np.ones(len(xy)),domain)
    rods=ApprovedPackerRodDensity(ROOT);r=3.25552
    radial=0.
    for k,m in enumerate(rods.rows):
        angle=k*np.pi/2
        radial+=quad(lambda x:x*rods([[x*np.cos(angle),x*np.sin(angle)]])[0],0,r,
            points=[x['eccentricity_mm'] for x in rods.rows[m] if x['eccentricity_mm']<r],epsabs=1e-6)[0]
    np.testing.assert_allclose(rods.disk_mass(r),np.pi/2*radial,rtol=1e-12)
    assert round(ApprovedParasolRFDensity('ON').disk_mass(3))==30726

def test_frozen_global_csr_chunks_keep_weights_and_complete_dependencies(tmp_path):
    from retina_sim.disk_graph_storage import materialize_bounded_route,FrozenSparseRoute,dependency_closure
    from retina_sim.spatial_profiles import bounded_matrix
    pre=np.array([[0.,0.],[.01,0.],[.02,0.],[.1,0.]])
    post=np.array([[.005,0.],[.015,0.],[.3,0.]])
    recipe={'pre_population':'a','post_population':'b','pre_count':len(pre),'post_count':len(post),'source':'fixture'}
    binding=materialize_bounded_route(tmp_path,'route',recipe,pre,post,2,block_rows=1,gaussian_scale=.03)
    route=FrozenSparseRoute(tmp_path,binding);expected,rows=bounded_matrix(pre,post,2,support_mm=.04,gaussian_scale=.03)
    selected,actual,audits,statuses=route.selected_rows([2,0,1])
    np.testing.assert_array_equal(actual.toarray(),expected.toarray())
    assert [row['post_index'] for row in audits]==[0,1,2]
    assert route.record['missing_degree_rows']==1 and route.record['empty_rows']==1
    with pytest.raises(ContractError,match='omits frozen dependencies'):route.view([0],[0])
    _,_,view,_,_=route.view([0],[0,1])
    np.testing.assert_array_equal(view.toarray(),expected[[0]][:,[0,1]].toarray())
    closure=dependency_closure([('a','b',route)],{'b':[0]})
    np.testing.assert_array_equal(closure['a'],[0,1])
    # Reopening construction reuses the exact fixed blocks instead of rerunning queries.
    assert materialize_bounded_route(tmp_path,'route',recipe,pre,post,2,block_rows=2,gaussian_scale=.03)==binding
    with pytest.raises(ContractError,match='recipe differs'):
        materialize_bounded_route(tmp_path,'route',recipe,pre,post,2,gaussian_scale=.06)

def test_frozen_typed_arrays_share_ids_across_overlapping_disk_views(tmp_path):
    from retina_sim.disk_tables import write_table,FrozenDiskTable
    xy=np.array([[0.,0.],[2.9,0.],[2.9,2.9]])
    metadata={'atlas_id':'fixed','atlas_version':'1','source_version':'source','population':'cones','reference_id':'ref'}
    q=Quantity(xy,'mm','functional_position','synthetic_fixture','functional_visual_field')
    keys=('row0','row1','row2')
    table=write_table(tmp_path,'table',metadata,3,lambda start,stop:keys[start:stop],{'rf':q})
    again=FrozenDiskTable(tmp_path,table.binding)
    roi=DiskAtlasROI((-3,3,-3,3),(-3,3,-3,3),'ref')
    indices=table.query_indices(roi,owner=True)
    np.testing.assert_array_equal(indices,[0,1])
    assert table.rows(indices).global_cell_ids==again.rows(indices).global_cell_ids
    with pytest.raises(ContractError,match='coordinates differ'):
        write_table(tmp_path,'table',metadata,3,lambda start,stop:keys[start:stop],{'rf':Quantity(xy+.01,'mm','functional_position','synthetic_fixture','functional_visual_field')})
    assert write_table(tmp_path,'table',metadata,3,lambda start,stop:keys[start:stop],{'rf':q}).binding==table.binding
    with pytest.raises(ContractError,match='source keys differ'):
        write_table(tmp_path,'table',metadata,3,lambda start,stop:tuple(reversed(keys))[start:stop],{'rf':q})
    with pytest.raises(ContractError,match='declaration differs'):
        write_table(tmp_path,'table',metadata,3,lambda start,stop:keys[start:stop],{'rf':Quantity(xy,'mm','functional_position','changed_scope','functional_visual_field')})
    with pytest.raises(ContractError,match='auxiliary arrays differ'):
        write_table(tmp_path,'table',metadata,3,lambda start,stop:keys[start:stop],{'rf':q},extras={'subtype':np.array(['L','M','S'])})

def test_global_array_source_adapters_preserve_scalar_source_equations():
    from scipy.optimize import brentq
    from retina_sim.disk_source_geometry import schein_array,wool_temporal_array
    from retina_sim.geometry import PackerDensityField
    from retina_sim.runtime_profiles import frozen_packer_table
    atlas=SourceDensityAtlas(ROOT,profile_id='retina_atlas_Schein_bridge_research_v1')
    theta=np.r_[np.arange(80)*2*np.pi/80,np.pi/2+1e-13]
    radius=np.linspace(.00001,3.199,len(theta));xy=radius[:,None]*np.c_[np.cos(theta),np.sin(theta)]
    expected=atlas.henle.apply(Quantity(xy,'mm','functional_position','project_approximation','functional_visual_field')).values
    np.testing.assert_allclose(schein_array(atlas.henle,xy),expected,rtol=3e-15,atol=1e-15)
    with pytest.raises(ContractError,match='DIRECTION_UNDEFINED'):schein_array(atlas.henle,[[0.,0.]])
    field=PackerDensityField(frozen_packer_table(ROOT),central_policy='central_constant_cap',angular_policy='log_NT_cosine')
    reference=[]
    for point in xy:
        target=field(point[None,:])[0]
        try:reference.append(brentq(lambda r:field.radial(np.array([r]),np.array([np.pi]))[0]-target,field.minimum,10.))
        except ValueError:reference.append(np.nan)
    np.testing.assert_allclose(wool_temporal_array(field,xy),reference,rtol=0,atol=2e-12,equal_nan=True)

def test_global_optical_tiles_and_shared_reference_are_roi_independent():
    from retina_sim.disk_input import optical_tiles,join_captures
    xy=np.array([[-.512,0.],[-1e-9,0.],[0.,0.],[.512,0.],[2.99,-.001]])
    tiles=optical_tiles(xy);rows=[tile['indices'] for tile in tiles]
    np.testing.assert_array_equal(np.sort(np.concatenate(rows)),np.arange(len(xy)))
    assert all(tile['scene_samples']==593 for tile in tiles)
    captures=[]
    for indices in rows:
        raw={'q_ref_LMS':np.array([3.,4.,5.]),'window_edges_seconds':np.array([0.,.0001]),
            'expected_counts':(np.array(indices)+1.)[None,:],'background_counts':np.ones(len(indices)),
            'diameters_um':np.ones(len(indices))*2}
        captures.append((raw,{'global_reference_definition_sha256':'reference','fixed_radiance_gain':2.}))
    joined=join_captures(rows,captures,len(xy),reference_identity='reference')
    np.testing.assert_array_equal(joined['expected_counts'],np.arange(1.,6.)[None,:])
    captures[-1][0]['q_ref_LMS']=np.array([3.,4.,6.])
    with pytest.raises(ContractError,match='altered fixed LMS'):join_captures(rows,captures,len(xy),reference_identity='reference')

def test_global_midget_freeze_uses_private_parent_and_actual_centroid_disk(tmp_path,monkeypatch):
    from retina_sim.disk_tables import write_table,FrozenDiskTable
    from retina_sim.disk_inner import freeze_midget_rgcs
    from retina_sim import runtime_profiles
    source=runtime_profiles.frozen_packer_table(ROOT)
    monkeypatch.setattr(runtime_profiles,'frozen_packer_table',lambda root:source)
    rf=np.c_[[.01,.015,2.990,2.991,3.001,2.992,2.993],np.zeros(7)]
    ipl=np.c_[[.15,.151,3.,3.001,3.002,3.003,3.004],np.zeros(7)]
    catalog={'tables':{}}
    for polarity in ('ON','OFF'):
        name='midget_BC_'+polarity
        meta={'atlas_id':'fixture','atlas_version':'1','source_version':'synthetic','population':name,'reference_id':'ref'}
        table=write_table(tmp_path,name,meta,7,lambda a,b:tuple('parent'+str(i) for i in range(a,b)),
            {'rf':Quantity(rf,'mm','functional_position','synthetic_fixture','functional_visual_field'),
             'ipl_preterminal':Quantity(ipl,'mm','terminal_position','synthetic_fixture','IPL')})
        catalog['tables'][name]=table.binding
    result=freeze_midget_rgcs(tmp_path,catalog,'output')
    table=FrozenDiskTable(tmp_path,result['tables']['midget_RGC_ON'])
    parents=table.extras['parent_bc_row'];centers=table.coordinates['rf']
    assert 4 in parents  # Its actual connected RF is inside even though seed RF>3.
    np.testing.assert_array_equal(centers[np.flatnonzero(parents==0)[0]],rf[0])
    assert np.all(np.linalg.norm(centers,axis=1)<=3)

def test_floor_minimum_keeps_actual_inputs_and_continuous_deficit(tmp_path,monkeypatch):
    from retina_sim.disk_tables import write_table,FrozenDiskTable
    from retina_sim.disk_graph_storage import FrozenSparseRoute
    from retina_sim.disk_inner import freeze_midget_rgcs
    from retina_sim import runtime_profiles,disk_inner
    source=runtime_profiles.frozen_packer_table(ROOT)
    monkeypatch.setattr(runtime_profiles,'frozen_packer_table',lambda root:source)
    monkeypatch.setattr(disk_inner,'wool_midget_profile',lambda x:{
        'effective_contact_target':np.full(len(x),1.003),'private_line_branch':np.zeros(len(x),dtype=bool),
        'center_radius_mm':np.full(len(x),.00315)})
    xy=np.c_[1.+np.arange(7)*.02,np.zeros(7)];catalog={'tables':{}}
    for polarity in ('ON','OFF'):
        name='midget_BC_'+polarity;meta={'atlas_id':'fixture','atlas_version':'1','source_version':'same','population':name,'reference_id':'ref'}
        table=write_table(tmp_path,name,meta,7,lambda a,b:tuple(str(i) for i in range(a,b)),{
            'rf':Quantity(xy,'mm','functional_position','synthetic_fixture','functional_visual_field'),
            'ipl_preterminal':Quantity(xy,'mm','terminal_position','synthetic_fixture','IPL')})
        catalog['tables'][name]=table.binding
    old=freeze_midget_rgcs(tmp_path,catalog,'old')
    new=freeze_midget_rgcs(tmp_path,catalog,'floor',minimum_degree_rule='floor_minimum_keep_continuous_deficit_v1')
    assert old['blocked_candidates']['midget_RGC_ON']['count']==7
    assert new['blocked_candidates']['midget_RGC_ON']['count']==0
    table=FrozenDiskTable(tmp_path,new['tables']['midget_RGC_ON'])
    np.testing.assert_array_equal(table.coordinates['rf'],xy)
    route=FrozenSparseRoute(tmp_path,new['candidate_routes']['midget_RGC_ON'])
    _,matrix,rows,statuses=route.selected_rows(np.arange(7))
    np.testing.assert_array_equal(matrix.toarray(),np.eye(7))
    assert all(row['target_degree']==1.003 and row['actual_degree']==1 and row['missing_degree']==1 for row in rows)
    assert all(row['minimum_required_degree']==1 and row['minimum_degree_deficit']==0 for row in rows)
    assert all(np.isclose(row['unmet_continuous_target_count'],.003) for row in rows)
    assert all(row['population_average_target_satisfied']=='NOT_ESTABLISHED' for row in rows)
    assert all(s.startswith('READY_PROJECT_WOOL_FLOOR') for s in statuses)


def test_global_recording_view_retains_whole_native_feedback_component(tmp_path):
    from scipy import sparse
    from retina_sim.disk_tables import write_table
    from retina_sim.disk_graph_storage import SparseRouteWriter,file_sha,save_json
    from retina_sim.disk_circuit import GlobalDiskCircuit,SCHEMA
    tables={}
    for name,xy,role,meaning,frame in (
        ('cones',[[.35,0.],[.37,0.]],'rf','functional_position','functional_visual_field'),
        ('HC',[[.36,0.]],'soma','anatomical_position','anatomical_retina'),
        ('parasol_RGC_ON',[[.35,0.]],'rf','functional_position','functional_visual_field')):
        meta={'atlas_id':'fixture','atlas_version':'1','source_version':'fixed','population':name,'reference_id':'ref'}
        table=write_table(tmp_path,name,meta,len(xy),lambda a,b:tuple(str(i) for i in range(a,b)),
            {role:Quantity(xy,'mm',meaning,'synthetic_fixture',frame)})
        tables[name]={'kind':'typed','binding':table.binding,'count':table.count}
    routes=[]
    for name,pre,post,matrix,kind in (
        ('exc','cones','parasol_RGC_ON',sparse.csr_matrix([[1.,0.]]),'inner_chemical'),
        ('A','cones','HC',sparse.csr_matrix([[.5,.5]]),'native'),
        ('B','HC','cones',sparse.csr_matrix([[1.],[1.]]),'native')):
        recipe={'route_id':name,'pre_population':pre,'post_population':post,'pre_count':matrix.shape[1],'post_count':matrix.shape[0]}
        writer=SparseRouteWriter(tmp_path,'routes/'+name,recipe)
        writer.append(0,matrix,[{'actual_degree':int(n),'missing_degree':0} for n in np.diff(matrix.indptr)])
        routes.append({'pre':pre,'post':post,'kind':kind,'route_id':name,'binding':writer.freeze()})
    np.save(tmp_path/'native.npy',np.zeros(3,dtype=np.int64),allow_pickle=False)
    np.save(tmp_path/'electrical.npy',np.zeros(0,dtype=np.int64),allow_pickle=False)
    record={'schema':SCHEMA,'reference_id':'ref','tables':tables,'routes':routes,'output_populations':['parasol_RGC_ON'],
        'regenerate_on_ROI':False,'crop_normalization':False,'exterior_feedback':False,
        'native_components':{'path':'native.npy','sha256':file_sha(tmp_path/'native.npy')},
        'electrical_components':{'path':'electrical.npy','sha256':file_sha(tmp_path/'electrical.npy'),'offsets':{'AII_AC':0},'counts':{'AII_AC':0}}}
    save_json(tmp_path/'graph.json',record);circuit=GlobalDiskCircuit(tmp_path,{'path':'graph.json','sha256':file_sha(tmp_path/'graph.json')})
    view=circuit.owner_view((.34,.36,-.01,.01),outputs=('parasol_RGC_ON',))
    np.testing.assert_array_equal(view['selected_rows']['cones'],[0,1])
    assert view['runtime_ready'] and view['global_node_count']==4
    with pytest.raises(ContractError,match='exceeds execution node budget'):
        circuit.owner_view((.34,.36,-.01,.01),outputs=('parasol_RGC_ON',),max_nodes=2)

def test_streamed_canonical_hash_keeps_original_json_identity():
    import hashlib,json
    from enum import Enum
    from retina_sim.contracts import canonical_hash
    def original(value):
        def convert(v):
            if isinstance(v,np.ndarray):return {'dtype':str(v.dtype),'shape':list(v.shape),'data':v.tolist()}
            if isinstance(v,Enum):return v.value
            if hasattr(v,'__dataclass_fields__'):return {k:convert(getattr(v,k)) for k in v.__dataclass_fields__}
            if isinstance(v,dict):return {str(k):convert(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return [convert(x) for x in v]
            return v
        return hashlib.sha256(json.dumps(convert(value),sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
    values=[None,{},[],np.empty((2,0)),np.array(1.25),np.arange(17000,dtype=float)/10,
        ['中文','\\"\n',-0.,True,None],{1:'first','1':'last','x':np.array([[1.,-0.],[1e-300,1e200]])},
        Quantity([[.1,.2]],'mm','functional_position','synthetic_fixture','functional_visual_field'),
        tuple(str(i) for i in range(17001)),[np.array([1,2],dtype=np.int16),{'x':np.array(['日','本'])}]]
    assert all(canonical_hash(value)==original(value) for value in values)
    with pytest.raises(ValueError):canonical_hash(np.array([np.nan]))

def test_fixed_global_contact_geometry_entry_order_and_overlap(tmp_path):
    from retina_sim.disk_graph_storage import materialize_bounded_route,FrozenSparseRoute
    from retina_sim.disk_contact_geometry import materialize_contacts,FixedContactGeometry,global_contact_rows
    pre=np.array([[0.,0.],[.01,0.],[.02,0.]]);post=np.array([[.005,0.],[.015,0.]])
    recipe={'route_id':'fixture','pre_population':'a','post_population':'b','pre_count':3,'post_count':2}
    binding=materialize_bounded_route(tmp_path,'route',recipe,pre,post,2,block_rows=1)
    route=FrozenSparseRoute(tmp_path,binding)
    contact_binding=materialize_contacts(tmp_path,'contacts',binding,pre,post,frame='IPL',overlap_radius_mm=.03)
    geometry=FixedContactGeometry(tmp_path,contact_binding);_,matrix,_,_=route.selected_rows([0,1]);coo=matrix.tocoo()
    np.testing.assert_array_equal(geometry.positions,(pre[coo.col]+post[coo.row])/2)
    np.testing.assert_array_equal(geometry.overlap,np.exp(-(np.linalg.norm(pre[coo.col]-post[coo.row],axis=1)/.03)**2))
    np.testing.assert_array_equal(global_contact_rows(route,[1]),[2,3])
    from retina_sim.disk_runtime import DiskSpatialAsset
    from retina_sim.atlas_runtime import _population
    for frame in ('IPL','OPL'):
        frozen=FixedContactGeometry(tmp_path,materialize_contacts(tmp_path,'contacts_'+frame,binding,pre,post,frame=frame))
        adapter=DiskSpatialAsset.__new__(DiskSpatialAsset)
        adapter._ids={'a':('a0','a1','a2'),'b':('b0','b1')}
        descriptor={'pre':'a','post':'b','pre_ID_prefix':'','binding':frozen.binding}
        adapter._contact=lambda cid:(descriptor,matrix,[],frozen,np.arange(matrix.nnz))
        record=adapter.contact_record('actual_contact');anchor=record['physical_contact_anchor']
        pop=_population(record,'BC','fixture','synthetic_fixture',opl_post=anchor['values'] if frame=='OPL' else None)
        assert pop.dendritic_positions.coordinate_frame==frame
        np.testing.assert_array_equal(pop.dendritic_positions.values,geometry.positions)
        if frame=='OPL':
            assert pop.terminal_positions is None and not record['coordinates']
        else:assert pop.terminal_positions.coordinate_frame=='IPL'

def test_compact_global_input_checks_rows_units_and_actual_background(tmp_path):
    from types import SimpleNamespace as NS
    from retina_sim.contracts import CellTable
    from retina_sim.spatial import Population
    from retina_sim.assembly import ExternalInput
    from retina_sim.runtime_profiles import LightContext
    from retina_sim.isetbio_input import REFERENCE_SCHEMA
    from retina_sim.disk_input_packet import write_global_input,load_global_input
    ids=('fixture_cone_a','fixture_cone_b');xy=Quantity([[.35,0.],[.351,0.]],'mm','functional_position','synthetic_fixture','functional_visual_field')
    ecc=Quantity([.35,.351],'mm','global_eccentricity','synthetic_fixture','functional_visual_field')
    cells=CellTable(ids,('cone',)*2,('L','S'),('fixture',)*2,('fixture',)*2,('fixture',)*2,None,xy,'mm','functional_visual_field',ecc,('fixture',)*2,('synthetic_fixture',)*2)
    pop=Population('cones',cells,('center',)*2,receptive_field_positions=xy)
    port=NS(cell_ids=ids,unit='td',meaning='native_retinal_illuminance',evidence_tag='synthetic_fixture')
    component=NS(mechanism='VanHateren2005NativeConeH1',inputs={'light':port},outputs={},native_model=None,
        parameters=NS(parameter_hash='a'*64),binding_options_hash='b'*64,implementation_hash='c'*64,
        time_contract=NS(kind='continuous',fixed_step_seconds=None))
    graph={'path':'synthetic_graph','sha256':'d'*64};table={'path':'synthetic_cones','sha256':'e'*64};view={'sha256':'f'*64}
    compiled=NS(populations={'cones':pop},components={'outer_native':component},links=(),clock={'base_step_seconds':.0001},
        spec=NS(gap_junctions=(),external_inputs=(ExternalInput('native_light','outer_native','light'),)),assembly_hash='0'*64,
        geometry_source_binding={'synthetic_fixture':True},input_geometry_binding_required=True,
        global_disk_circuit=NS(binding=graph,record={'tables':{'cones':{'binding':table}}}),
        global_spatial_asset=NS(binding=view),global_disk_view={'selected_rows':{'cones':np.array([3,5],dtype=np.int64)}})
    light=LightContext();scope='1'*64
    reference={'schema':REFERENCE_SCHEMA,'profile_id':'synthetic_fixed_LMS','source_id':'fixture','scope':'engineering_fixture',
        'evidence_tag':'synthetic_fixture','reference_conditions':'synthetic contract fixture only','capture_scope_sha256':scope,
        'reference_scene_sha256':'2'*64,'reference_capture_sha256':'3'*64,'reference_luminance_cd_m2':100/(np.pi*1.5**2),
        'pupil_diameter_mm':3.,'reference_rates_rstar_s_LMS':{'L':1000.,'M':1000.,'S':1000.}}
    declared={'population_id':'cones','global_population':'cones','population_binding':table,'global_rows_key':'rows',
        'data_axes':['time','cell'],'coordinate_role':'rf','coordinate_unit':'mm','coordinate_frame':'functional_visual_field','reference_id':'ref',
        'coordinate_registration':'synthetic_fixture','positions_key':'xy','values_key':'counts','unit':'R*/window',
        'input_stage':'reference_equivalent_cone_drive','evidence_tag':'synthetic_fixture','value_semantics':'expected',
        'photon_noise':{'owner':'none','model':'none'},'mechanism_ownership':{'photopigment_capture':'upstream','phototransduction':'consumer','neural_noise':'consumer'},
        'background':{'includes_background':True,'unit':'td','value':100.,'source_id':light.background_source},
        'prehistory':{'kind':'declared_uniform_background','stop_seconds':0.},
        'time':{'representation':'integration_windows','reference':'stimulus_relative_seconds','source_origin_seconds':0.,'start_seconds':0.,
            'stop_seconds':.0001,'axis_key':'edges','interpolation':'zero_order_hold','source_grid_step_seconds':.0001},
        'reference_profile':reference,'capture_scope_sha256':scope,'capture_types_key':'types','background_counts_key':'bg',
        'capture_pupil_diameter_mm':3.,'capture_scope':'engineering_fixture'}
    arrays={'xy':xy.values,'rows':np.array([3,5]),'types':np.array(['L','S'],dtype='S1'),
        'counts':np.array([[.1,.2]]),'bg':np.array([[.1,.1]]),'edges':np.array([0.,.0001])}
    path=tmp_path/'valid.npz'
    write_global_input(path,compiled,light,reference_id='ref',arrays=arrays,declarations={'native_light':declared},producer={'name':'synthetic fixture'})
    bundle=load_global_input(path,compiled,light,reference_id='ref')
    np.testing.assert_allclose(bundle.channels['native_light'].at(0).values,[100.,200.],rtol=1e-15)
    for name,changed,message in [('rows',dict(arrays,rows=np.array([3,6])),'row identity'),('background',dict(arrays,bg=np.array([[.1,.2]])),'background differs')]:
        bad=tmp_path/(name+'.npz')
        write_global_input(bad,compiled,light,reference_id='ref',arrays=changed,declarations={'native_light':declared},producer={'name':'synthetic fixture'})
        with pytest.raises(ContractError,match=message):load_global_input(bad,compiled,light,reference_id='ref')
