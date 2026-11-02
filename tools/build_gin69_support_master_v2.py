"""Construct one finite HC master from the actual new local Cone dependencies."""
from pathlib import Path
import json
import time
import numpy as np
from retina_sim.hc_current import current_hc
from retina_sim.hc_mosaic import HCDensityProfile, RadialMaskedHCDomain
from retina_sim.atlas_hc_master import materialize_hc_master
from tools.durable_research_runtime_v1 import atomic_json

ROOT=Path(r'D:\macaque retinal simulator')
OUT=ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2'

def build(output,guard):
    fact=json.loads((OUT/'executions/assemble_fine_01/FINE_ASSEMBLY_RESULT.json').read_bytes())
    selected=current_hc(ROOT);path=ROOT/selected['master_configuration']
    cfg=json.loads(path.read_bytes());profile=HCDensityProfile(ROOT)
    if (ROOT/cfg['asset_path']).exists():raise FileExistsError('Existing HC master retained')
    radius=cfg['owner_support']['HC_outer_contact_margin_mm']
    p=fact['native_cone_pedicle_bounds_mm']
    view=[p[0]-radius,p[1]+radius,p[2]-radius,p[3]+radius]
    domain=RadialMaskedHCDomain(view,view,profile.support_radius_mm,'gin69-owner-support-v2','new-HC-construction')
    bound=domain.density_bounds(profile,view);ref=profile.reference_density
    margin=float(np.max(profile.reference_lengths_mm*np.sqrt([
        ref[0]/bound['H1_min'],ref[1]/bound['H2_min'],ref.sum()/bound['total_min']])))
    required=fact['native_cone_radial_max_mm']+radius+margin
    if required>profile.support_radius_mm:raise ValueError('Actual local envelope exceeds authorized source-derived support')
    canonical=[view[0]-margin,view[1]+margin,view[2]-margin,view[3]+margin]
    cfg.update(canonical_bounds_mm=canonical,requested_native_view_bounds_mm=view,
        required_actual_radial_envelope_mm=required,construction_domain_pending=False,
        mask_source_id='HC69_owner_support_v2_actual_dependency_domain')
    atomic_json(path,cfg)
    atomic_json(output/'DOMAIN.json',dict(new_view_bounds_mm=view,new_canonical_bounds_mm=canonical,
        pair_margin_mm=margin,contact_margin_mm=radius,required_radial_envelope_mm=required,
        source_domain_radius_mm=profile.support_radius_mm,owner_support=cfg['owner_support'],
        boundary='Complete direct local contact envelope; finite HC context is not infinite-retina completeness'))
    guard();start=time.perf_counter()
    atlas=json.loads((ROOT/'configurations/retina_atlas_human_borrowed_disk3mm_v1.json').read_bytes())
    result=materialize_hc_master(ROOT,atlas,profile);guard()
    result.update(wall_seconds=time.perf_counter()-start,current_HC_identity=current_hc(ROOT)['identity'],neural_run=False)
    atomic_json(output/'MASTER_RESULT.json',result)
    print(json.dumps(result),flush=True)
