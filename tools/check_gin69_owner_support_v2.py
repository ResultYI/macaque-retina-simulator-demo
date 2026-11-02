"""Narrow regression of owner ancestry, support closure and stale asset admission."""
from pathlib import Path
import json
import time
import numpy as np
from retina_sim.bc_current import current_bc, require_current_bc_asset
from retina_sim.hc_current import current_hc, require_current_hc_asset
from retina_sim.bc_pedicle_density import RegisteredPedicleField, anchor_rows, RATIOS
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.hc_mosaic import HCDensityProfile
from retina_sim.contracts import ContractError
from tools.durable_research_runtime_v1 import atomic_json

ROOT=Path(r'D:\macaque retinal simulator')
def read(p):return json.loads(p.read_bytes())
def build(output,guard):
    start=time.perf_counter();domain=read(ROOT/'configurations/research/GIN69_OWNER_SUPPORT_V2.json')
    selected=current_bc(ROOT);cfg=read(ROOT/selected['bindings']['parameters']['path'])
    cone=FrozenDiskTable(ROOT,domain['cone_source_table']);field=RegisteredPedicleField(cone,cfg)
    groups=anchor_rows(tuple(cone.ids(0,cone.count)),field.labels,field.owner_rows)
    owner=field.owner_rows;outside=owner[np.linalg.norm(field.xy[owner],axis=1)>1.338]
    assert len(owner)==190498 and len(outside)==35518
    assert np.array_equal(np.sort(groups['midget_BC_OFF'][0]),owner)
    assert np.array_equal(np.sort(groups['midget_BC_ON'][0]),owner[field.labels[owner]!='S'])
    assert np.isin(outside,groups['midget_BC_OFF'][0]).all()
    for name,ratio in RATIOS.items():
        count=len(owner) if name!='S_ON_BC' else int(np.sum(field.labels[owner]=='S'))
        assert len(groups[name][0])==(2*count*ratio.numerator+ratio.denominator)//(2*ratio.denominator)
    profile=HCDensityProfile(ROOT);hc=current_hc(ROOT)
    old=read(ROOT/'configurations/hc_density_rf_joint_v2.json');new=read(ROOT/hc['bindings']['parameters']['path'])
    assert old['density']==new['density'] and old['geometry']['central_join']==new['geometry']['central_join']
    radius=domain['HC_support_radius_mm']
    assert radius>=domain['BC_anchor_radius_mm']+domain['HC_outer_contact_margin_mm']+domain['HC_outer_pair_margin_mm']-1e-14
    assert profile.support_radius_mm==radius
    for kind in ('H1','H2'):
        assert np.isfinite(profile.contact_radius_mm(np.array([0.,.58,.60,1.338,radius]),kind)).all()
    for require in (require_current_bc_asset,require_current_hc_asset):
        try:require(ROOT,{})
        except ContractError:pass
        else:raise AssertionError('Old unbound assets admitted')
    guard();result=dict(status='PASS',scope='software owner/support regression only; no neural or physiological validation',
        owner_cones=len(owner),recovered_displaced_pedicle_owners=len(outside),
        canonical_BC_counts={n:len(x[0]) for n,x in groups.items()},domain=domain,
        density_coefficients_unchanged=True,private_parent_exact=True,old_assets_rejected=True,
        wall_seconds=time.perf_counter()-start)
    atomic_json(output/'RESULT.json',result);print(json.dumps(result),flush=True)
