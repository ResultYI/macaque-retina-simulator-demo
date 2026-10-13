"""Only approved density rules and actual frozen-master identity/count checks."""
from pathlib import Path
import json
import numpy as np
import pytest
from retina_sim.population_revision import ApprovedPackerRodDensity,ApprovedParasolRFDensity
from retina_sim.equal_mass_master import FixedEqualMassMaster
from retina_sim.atlas import AtlasROI
from retina_sim.contracts import ContractError

ROOT=Path(__file__).resolve().parents[1]


def test_rod_nodes_quadratic_join_origin_and_no_hard_hole():
    field=ApprovedPackerRodDensity(ROOT)
    for theta,m in zip((0,np.pi/2,np.pi,3*np.pi/2),field.rows):
        direction=np.array([np.cos(theta),np.sin(theta)]);rows=field.rows[m]
        xy=np.array([r['eccentricity_mm']*direction for r in rows])
        np.testing.assert_allclose(field(xy),[r['density_per_mm2'] for r in rows],rtol=1e-12,atol=1e-8)
        r1=rows[0]['eccentricity_mm'];rho1=rows[0]['density_per_mm2']
        np.testing.assert_allclose(field(np.array([direction*r1/2])),[rho1/4],rtol=1e-12)
    assert field(np.array([[0.,0.]]))[0]==0
    assert field(np.array([[.00001,0],[.02,0]])).min()>0
    with pytest.raises(ContractError,match='no extrapolation'):field(np.array([[7.,0]]))


def test_rod_angular_convex_blend_and_source_uncertainty_kept():
    field=ApprovedPackerRodDensity(ROOT);r=.3;phi=np.pi/6
    expected=field(np.array([[r,0]]))[0]*np.cos(phi)**2+field(np.array([[0,r]]))[0]*np.sin(phi)**2
    np.testing.assert_allclose(field(np.array([[r*np.cos(phi),r*np.sin(phi)]])),[expected],rtol=1e-12)
    assert all(row['eccentricity_digitization_error_mm']>0 for row in field.metadata['first_nodes'].values())


def test_parasol_fixed_coefficients_equal_polarity_and_domain():
    on,off=ApprovedParasolRFDensity('ON'),ApprovedParasolRFDensity('OFF');xy=np.array([[0,0],[.35,0],[0,.35],[3,0]])
    r=np.linalg.norm(xy,axis=1);expected=.04*(580000*np.exp(-4.2*r)+56000*np.exp(-.56*r))
    np.testing.assert_array_equal(on(xy),expected);np.testing.assert_array_equal(on(xy),off(xy))
    assert on.profile_hash!=off.profile_hash
    with pytest.raises(ContractError,match='limited'):on(np.array([[3.01,0]]))


def test_actual_fixed_master_quota_crop_overlap_and_independent_namespaces():
    graph=json.loads((ROOT/'data/atlas/population_revised_r035_v1.json').read_bytes());seen=set();positions={}
    for name,meta in graph['population_revision']['masters'].items():
        path=Path(meta['path']);payload=json.loads(path.read_bytes());master=FixedEqualMassMaster(payload,path,meta['sha256'])
        assert len(master.table.source_keys)==payload['diagnostics']['rounding_target']
        assert len(master.table.source_keys)==int(np.floor(payload['diagnostics']['mass']+.5))
        assert meta['namespace'] not in seen;seen.add(meta['namespace'])
        b=master.bounds;dx=(b[1]-b[0])/5;dy=(b[3]-b[2])/5
        a=(b[0]+dx,b[1]-dx,b[2]+dy,b[3]-dy);c=(b[0]+2*dx,b[1]-dx,b[2]+dy,b[3]-2*dy)
        first=master.crop(AtlasROI(a,a,master.table.reference_id));second=master.crop(AtlasROI(c,c,master.table.reference_id))
        lookup=dict(zip(first.global_cell_ids,first.coordinate('rf').values))
        for gid,p in zip(second.global_cell_ids,second.coordinate('rf').values):np.testing.assert_array_equal(p,lookup[gid])
        assert master.recipe['generator']=='greedy_capacitated_equal_rho_mass' and master.recipe['iterations']==1
        positions[name]=master.table.coordinate('rf').values
    assert not np.array_equal(positions['parasol_RGC_ON'],positions['parasol_RGC_OFF'])
