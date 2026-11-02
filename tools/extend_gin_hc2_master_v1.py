"""One source-bound HC master expansion required by actual new BC dependencies."""
from pathlib import Path
import json
import time
import numpy as np
from retina_sim.hc_current import current_hc
from retina_sim.hc_mosaic import HCDensityProfile, RadialMaskedHCDomain
from retina_sim.atlas_hc_master import materialize_hc_master
from tools.durable_research_runtime_v1 import atomic_json

ROOT=Path(r'D:\macaque retinal simulator')
OUT=ROOT/'reports/GIN_69_6_HC2_BC1_FIT_V1'

def build(output,guard):
    fact=json.loads((OUT/'executions/assemble_fine_01/FINE_ASSEMBLY_RESULT.json').read_bytes())
    selected=current_hc(ROOT);path=ROOT/selected['master_configuration']
    old=json.loads(path.read_bytes())
    if old['asset_path']!='data/atlas/GIN_69_6_HC2_BC1_V1/HC_master.json':
        raise RuntimeError('Expansion starts only from the explicitly diagnosed first master')
    atomic_json(output/'PREVIOUS_MASTER_CONFIGURATION.json',old)
    profile=HCDensityProfile(ROOT)
    radius=max(float(profile.contact_radius_mm(1.338,k)) for k in ('H1','H2'))
    p=fact['native_cone_pedicle_bounds_mm'];view=[p[0]-radius,p[1]+radius,p[2]-radius,p[3]+radius]
    domain=RadialMaskedHCDomain(view,view,1.338,'gin69-HC2-complete-BC-dependencies','new-HC-construction')
    bound=domain.density_bounds(profile,view);ref=profile.reference_density
    margin=float(np.max(profile.reference_lengths_mm*np.sqrt([
        ref[0]/bound['H1_min'],ref[1]/bound['H2_min'],ref.sum()/bound['total_min']])))
    required=fact['native_cone_radial_max_mm']+radius+margin
    if required>1.338:raise RuntimeError('Actual contact/pair envelope exceeds approved source disk')
    b=old['canonical_bounds_mm']
    canonical=[min(b[0],view[0]-margin),max(b[1],view[1]+margin),min(b[2],view[2]-margin),max(b[3],view[3]+margin)]
    asset='data/atlas/GIN_69_6_HC2_BC1_V1/HC_complete_BC_dependency_master.json'
    if (ROOT/asset).exists():raise RuntimeError('Completed expansion retained; do not regenerate')
    cfg=dict(old,profile_id='HC69_HC2_BC1_complete_BC_dependency_master_v1',asset_path=asset,
        expected_asset_sha256=None,canonical_bounds_mm=canonical,requested_native_view_bounds_mm=view,
        required_actual_radial_envelope_mm=required,
        mask_source_id='HC69_HC2_complete_actual_newBC_dependency_domain_v1',
        scope='Independent finite master for actual selected new BC dependencies; earlier5724-cell asset retained. Same density, contact, sampler and seed; changed canonical domain implies new cell identities.')
    atomic_json(path,cfg)
    atomic_json(output/'DOMAIN.json',dict(source_fine_assembly='executions/assemble_fine_01/FINE_ASSEMBLY_RESULT.json',
        previous_asset=old['asset_path'],previous_sha256=old['expected_asset_sha256'],
        previous_missing_contact_margin_cones=fact['HC_master_coverage']['cones_lacking_conservative_contact_margin'],
        new_view_bounds_mm=view,new_canonical_bounds_mm=canonical,pair_margin_mm=margin,
        required_radial_envelope_mm=required,source_domain_radius_mm=1.338,
        parameters_changed=False,old_cell_identity_reused=False))
    guard();start=time.perf_counter()
    atlas=json.loads((ROOT/'configurations/retina_atlas_human_borrowed_disk3mm_v1.json').read_bytes())
    result=materialize_hc_master(ROOT,atlas,profile);guard()
    result.update(wall_seconds=time.perf_counter()-start,current_HC_identity=current_hc(ROOT)['identity'],
        reuse='BC populations/fine routes and unchanged original cells retained; only insufficient HC source cohort replaced by new independently identified master',
        neural_run=False)
    atomic_json(output/'MASTER_RESULT.json',result)
    print(json.dumps(result),flush=True)
