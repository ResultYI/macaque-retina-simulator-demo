"""New bounded source-domain HC master for the approved real69#6 target."""
from pathlib import Path
import json
import time
import numpy as np
from retina_sim.hc_current import current_hc
from retina_sim.hc_mosaic import HCDensityProfile, RadialMaskedHCDomain
from retina_sim.atlas_hc_master import materialize_hc_master
from tools.durable_research_runtime_v1 import atomic_json

ROOT = Path(r'D:\macaque retinal simulator')
OUT = ROOT / 'reports/GIN_69_6_HC2_BC1_FIT_V1'

def build(output, guard):
    source = json.loads((OUT / 'geometry_preparation/run_02/NEW_GEOMETRY_DOMAIN.json').read_bytes())
    profile = HCDensityProfile(ROOT)
    radius = max(float(profile.contact_radius_mm(1.338, k)) for k in ('H1', 'H2'))
    facts = source['old_source_facts']['cones']['coordinates']['opl_pedicle']
    p = facts['bounds_mm']
    view = [p[0]-radius, p[1]+radius, p[2]-radius, p[3]+radius]
    masked = RadialMaskedHCDomain(view, view, 1.338, 'gin69-HC2-source-domain-mask-v1', 'new-HC-construction')
    # The permitted source interval provides a conservative interaction bound.
    bound = masked.density_bounds(profile, view)
    ref = profile.reference_density
    margin = float(np.max(profile.reference_lengths_mm*np.sqrt([
        ref[0]/bound['H1_min'], ref[1]/bound['H2_min'], ref.sum()/bound['total_min']])))
    required = facts['radial_extent_mm'][1]+radius+margin
    if required > 1.338:
        raise RuntimeError('Actual required pedicle/contact/pair envelope exceeds the approved HC domain: '+str(required))
    canonical = [view[0]-margin, view[1]+margin, view[2]-margin, view[3]+margin]
    selected = current_hc(ROOT)
    path = ROOT / selected['master_configuration']
    cfg = json.loads(path.read_bytes())
    if cfg.get('construction_domain_pending') is not True:
        asset = ROOT/cfg['asset_path']
        if asset.exists() or cfg.get('expected_asset_sha256') is not None or cfg.get('canonical_bounds_mm') != canonical:
            raise RuntimeError('Existing or different HC construction binding retained')
    atomic_json(output/'MASTER_CONFIGURATION_BEFORE.json', cfg)
    cfg.update(profile_id='HC69_HC2_BC1_bounded_master_v1',
        asset_path='data/atlas/GIN_69_6_HC2_BC1_V1/HC_master.json', expected_asset_sha256=None,
        canonical_bounds_mm=canonical, canonical_disk_radius_mm=1.338,
        construction_domain_pending=False, max_cells=20000,
        mask_source_id='HC69_6_v2_actual_pedicle_dependencies_inside_source_radius1p338_v1',
        scope='New finite69#6 real3.88deg/+x HC master. Source disk clips only unused rectangle corners; every actual required pedicle+contact+pair envelope lies inside. Not full0..6deg generation.',
        requested_native_view_bounds_mm=view, required_actual_radial_envelope_mm=required)
    atomic_json(path, cfg)
    domain = dict(source_cell='69#6', source_eccentricity_deg=3.88,
        actual_pedicle_extent_mm=facts['radial_extent_mm'], native_view_bounds_mm=view,
        canonical_bounds_mm=canonical, source_mask_radius_mm=1.338,
        conservative_joint_pair_context_mm=margin, actual_required_radius_mm=required,
        unused_bbox_corners_masked=True, actual_required_dependencies_clipped=False,
        old_HC_positions_reused=False, density_extended=False, seed=cfg['seed'],
        previous_rectangle_preflight='geometry_preparation/run_02/NEW_GEOMETRY_DOMAIN.json was a rectangle-only rejection, not evidence of actual source-domain overrun')
    atomic_json(output/'DOMAIN.json', domain)
    guard()
    started = time.perf_counter()
    atlas = json.loads((ROOT/'configurations/retina_atlas_human_borrowed_disk3mm_v1.json').read_bytes())
    result = materialize_hc_master(ROOT, atlas, profile)
    guard()
    result.update(wall_seconds=time.perf_counter()-started, domain=domain,
        current_HC_identity=current_hc(ROOT)['identity'], neural_run=False, background_reused=False)
    atomic_json(output/'MASTER_RESULT.json', result)
    print(json.dumps(result), flush=True)
