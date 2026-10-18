"""Source-bound geometry/identity tests; no MATLAB or neural integration."""
from pathlib import Path
import hashlib,json
import numpy as np
import pytest
from scipy import sparse
from retina_sim.contracts import Quantity,ContractError
from retina_sim.atlas import AtlasROI
from retina_sim.source_atlas import SourceDensityAtlas
from retina_sim.isetbio_geometry import connected_cone_rf_centers,cone_apertures,FixedISETConeMaster,input_geometry_support
from retina_sim.geometry import RetinalDomain,capacity_constrained_mosaic
from test_atlas_inner import cones,build

ROOT=Path(__file__).resolve().parents[1]

def test_connected_center_orientation_private_line_and_no_soma():
    q=Quantity([[.1,-0.],[1.,0.],[2.,1.]],'mm','functional_position','synthetic_fixture','functional_visual_field')
    w=sparse.csr_matrix([[.7,0,0],[0,.25,.75]])
    a=connected_cone_rf_centers(q,w);b=connected_cone_rf_centers(q,w.T,orientation='iset_cone_by_post')
    np.testing.assert_array_equal(a.values,b.values)
    assert a.values[0].tobytes()==q.values[0].tobytes()
    np.testing.assert_array_equal(a.values[1],[1.75,.75])
    assert a.coordinate_frame=='functional_visual_field' and a.meaning=='functional_position'

@pytest.mark.parametrize('weights',[[[0,0]],[[1,-1]],[[float('nan'),1]]])
def test_unresolved_or_invalid_center_rejected(weights):
    q=Quantity([[0,0],[1,0]],'mm','functional_position','synthetic_fixture','functional_visual_field')
    with pytest.raises(ContractError):connected_cone_rf_centers(q,weights)

def test_midget_adapter_keeps_private_parent_and_changes_only_rf_role():
    master=cones();layout=build(master,cone_subtypes=('L','M','L'),midget_rf_rule='connected_cone_centroid_isetbio_v1')
    for polarity in ('ON','OFF'):
        table=layout.tables['midget_RGC_'+polarity]
        np.testing.assert_array_equal(table.coordinate('rf').values,master.coordinate('rf').values)
        np.testing.assert_array_equal(table.coordinate('ipl_post_arbor').values,master.coordinate('opl_pedicle').values)
        assert 'soma' not in table.coordinates
    assert layout.metadata['midget_RF_geometry_source']['commit']=='d2924c2fe1d41c7dc5af1006d8081735b08f1b35'

def test_peripheral_centers_use_enabled_wool_pool_instead_of_seed():
    x=np.arange(12)*.001;master=cones(rf=np.column_stack((.03+x,x*0)),pedicle=np.column_stack((.4+x,x*0)))
    layout=build(master,cone_subtypes=('L','M')*6,temporal_equivalent_mm=np.full(12,3.),midget_rf_rule='connected_cone_centroid_isetbio_v1')
    for polarity in ('ON','OFF'):
        bc='midget_BC_'+polarity;rgc='midget_RGC_'+polarity
        w=layout.contacts[bc+'->'+rgc]['matrix']@layout.contacts['cones->'+bc]['matrix']
        expected=connected_cone_rf_centers(master.coordinate('rf'),w)
        np.testing.assert_array_equal(layout.tables[rgc].coordinate('rf').values,expected.values)
        assert not np.array_equal(expected.values,master.coordinate('rf').values)
    with pytest.raises(ContractError,match='Unresolved RF center'):
        build(cones(),cone_subtypes=('L','M','L'),temporal_equivalent_mm=np.full(3,3.),midget_rf_rule='connected_cone_centroid_isetbio_v1')

@pytest.fixture(scope='module')
def atlas():return SourceDensityAtlas(ROOT,profile_id='retina_atlas_isetbio_macaque_v1')

def test_fixed_master_overlap_identity_and_query_order(atlas):
    ref=atlas.config['reference_id'];a=AtlasROI((.33,.35,-.01,.01),(.32,.36,-.02,.02),ref)
    b=AtlasROI((.34,.36,-.01,.01),(.33,.37,-.02,.02),ref)
    x=atlas.query(a);y=atlas.query(b);z=atlas.query(a)
    def rows(p):return dict(zip(p.selection.global_cell_ids,map(tuple,p.selection.coordinate('rf').values)))
    xx,yy,zz=rows(x),rows(y),rows(z);common=xx.keys()&yy.keys()
    assert common and xx==zz and all(xx[k]==yy[k] for k in common)
    assert x.manifest['density_scale']==1 and x.manifest['mosaic_rule']=='isetbio_fixed_macaque_cone_master_v1'
    assert atlas.cone_master.manifest['human_density_used'] is False

def test_master_boundary_budget_and_hash_checks(atlas):
    with pytest.raises(ContractError,match='geometry master'):atlas.cone_master.require_bounds((.5,.6,-.01,.01))
    with pytest.raises(ContractError):atlas.query(AtlasROI((.34,.36,-.01,.01),(.30,.40,-.05,.05),atlas.config['reference_id']),max_cells=1)
    binding=dict(atlas.config['cone_master']);binding['sha256']='0'*64
    with pytest.raises(ContractError,match='hash'):FixedISETConeMaster(ROOT,binding)

def test_apertures_recomputed_and_selected_by_identity():
    xy=np.column_stack((np.arange(10)*.002,np.zeros(10)));ids=tuple('c'+str(i) for i in range(10))
    a=cone_apertures(xy,support_ids=ids,selected_ids=('c8','c1'))
    expected=cone_apertures(xy)[[8,1]];np.testing.assert_array_equal(a,expected)
    np.testing.assert_allclose(cone_apertures(xy*2),cone_apertures(xy)*2,rtol=0,atol=0)


def test_human_borrowing_requires_named_species_asset_and_transform(tmp_path):
    from retina_sim.isetbio_geometry import COMMIT,SCHEMA,HUMAN_SCHEMA,HUMAN_ASSET_SHA256,HUMAN_TRANSFORM
    xy=np.array([[.1,.1],[.2,-.1],[-.2,.1]])
    payload=tmp_path/'positions.npz';np.savez(payload,positions_mm=xy,source_keys=np.array(['row0','row1','row2']))
    manifest={'schema':HUMAN_SCHEMA,'isetbio_commit':COMMIT,'human_density_used':True,
        'density_species':'Homo sapiens','geometry_route':'official_human_borrowed_approximation_v1',
        'population':'cones','position_role':'rf_input_inner_segment_project_transfer',
        'coordinate_transform':dict(HUMAN_TRANSFORM),'complete_query_bounds_mm':[-1,1,-1,1],
        'official_asset':{'sha256':HUMAN_ASSET_SHA256,'eye':'right eye','nominal_FOV_degrees':58},
        'payload':{'path':payload.name,'sha256':hashlib.sha256(payload.read_bytes()).hexdigest()}}
    def load(value):
        path=tmp_path/'manifest.json';path.write_text(json.dumps(value),encoding='utf-8')
        return FixedISETConeMaster(tmp_path,{'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    np.testing.assert_array_equal(load(manifest).positions,xy)
    for changed in (dict(manifest,schema=SCHEMA),dict(manifest,human_density_used=False),
                    dict(manifest,official_asset=dict(manifest['official_asset'],eye='left eye')),
                    dict(manifest,official_asset=dict(manifest['official_asset'],sha256='0'*64)),
                    dict(manifest,coordinate_transform=dict(HUMAN_TRANSFORM,spatial_warp=True))):
        with pytest.raises(ContractError):load(changed)

def test_capacity_tail_keeps_original_count_and_audit():
    bounds=(0,.053,0,.071);domain=RetinalDomain(bounds,bounds,lambda xy:np.ones(len(xy),bool),'synthetic_fixture','tail_roundoff_regression')
    field=lambda xy:np.full(len(xy),5000.)
    m=capacity_constrained_mosaic(field,domain,20261013,quadrature_shape=(64,64),iterations=1)
    assert len(m.cell_ids)==int(np.floor(.053*.071*5000+.5))
    assert m.diagnostics['maximum_capacity_relative_residual']<=1e-8

def test_historical_iset_profile_requires_own_background_and_preserves_binding():
    from retina_platform.config import RunConfig
    cfg=RunConfig.from_dict({'execution_profile':'atlas_isetbio_geometry_v1'})
    assert cfg.background_initialization_mode=='bounded_convergence' and cfg.background_require_converged
    assert cfg.background_checkpoint_path=='' and cfg.warmup_seconds==0
    from retina_sim.atlas_population_network import circuit_configuration
    assert circuit_configuration(ROOT)['frozen_graph']['artifact']=='data/atlas/isetbio_macaque_cones_v1/local_population_graph.json'

def test_actual_new_graph_support_and_empty_owner_declaration():
    from retina_sim.atlas_population_network import load_population_record
    record,binding=load_population_record(ROOT,execution_profile='atlas_isetbio_geometry_v1')
    ids,xy=input_geometry_support(ROOT,record);lookup={key:i for i,key in enumerate(ids)}
    cone_ids=record['cones']['global_cell_ids'];rows=[lookup[key] for key in cone_ids]
    np.testing.assert_array_equal(xy[rows],record['cones']['coordinates']['rf']['values'])
    assert len(ids)>len(cone_ids) and record['isetbio_geometry']['rod_RBC_AII_retained']
    assert len(record['inner_tables']['S_ON_BC']['global_cell_ids'])>0
    verification=json.loads((ROOT/'reports/isetbio_model_20261015/verification.json').read_bytes())
    assert verification['status']=='ASSEMBLY_PASS' and verification['populations']['rod']>0
    assert 'small_bistratified_S_ON' in verification['empty_owner_outputs']

def test_gui_mode_switch_clears_old_geometry_state_without_running(monkeypatch):
    from types import SimpleNamespace
    from retina_platform.app import RetinaApp
    from retina_platform.config import RunConfig
    class Value:
        def __init__(self,x):self.x=str(x)
        def get(self):return self.x
        def set(self,x):self.x=str(x)
    app=object.__new__(RetinaApp);app.project_root=ROOT;app.session=None;app.media=None
    app.config=RunConfig.from_dict({'execution_profile':'atlas_population_revision_v1'});app.config.frontend_signature='old_geometry'
    app.config.native_background_values=(1.,);app.config.background_checkpoint_path='old_checkpoint.json'
    app.params={k:Value('' if getattr(app.config,k) is None else getattr(app.config,k)) for k,_,_ in RetinaApp.PARAMS}
    app.params.update(media_path=Value('preserved.png'),gaze_csv=Value(''),hc_geometry_path=Value(app.config.hc_geometry_path))
    app.params['execution_profile'].set('ISET几何');app.status_var=Value('')
    app.roi_labels={'custom':'Custom'};app.regional_support={'roi_presets':{}}
    states={};app.entries={'execution_profile':SimpleNamespace(configure=lambda **kw:states.update(kw))}
    app.buttons={'hc_geometry_json':SimpleNamespace(configure=lambda **kw:None)}
    monkeypatch.setattr(RunConfig,'validate',lambda self,media=None:None)
    app._choose_execution_profile();cfg=app._read_config()
    assert cfg.execution_profile=='atlas_isetbio_geometry_v1' and cfg.background_checkpoint_path==''
    assert cfg.native_background_values is None and cfg.frontend_signature=='' and cfg.media_path=='preserved.png'
    assert states['state']=='readonly'
