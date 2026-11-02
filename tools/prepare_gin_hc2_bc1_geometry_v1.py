"""Read unchanged source anatomy and choose a bounded new-HC construction domain."""
from pathlib import Path
import json
import sys
import time
import numpy as np
from retina_sim.spatial_group_model import GroupModelAssembly
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.hc_mosaic import HCDensityProfile
from retina_sim.hc_current import current_hc
from retina_sim.bc_current import current_bc
from retina_sim.contracts import canonical_hash

ROOT = Path(r'D:\macaque retinal simulator')
OLD = ROOT / 'reports/GIN_69_6_RADIAL_CONDITIONAL_V1/candidate_v2'

def read(path):
    return json.loads(path.read_bytes())

def main():
    output = Path(sys.argv[2])
    started = time.perf_counter()
    parent = GroupModelAssembly(ROOT / 'reports/spatial_group_graph_v02_20261025_v1/artifact_r2')
    with np.load(OLD / 'prepared/local_static/selection.npz', allow_pickle=False) as selected:
        groups = {name: selected['population__' + name] for name in ('cones', 'HC')}
    fields = {}
    for name in ('cones', 'HC'):
        pop = parent.populations[name]
        fine = np.concatenate([pop.member_global_rows[pop.member_offsets[r]:pop.member_offsets[r+1]] for r in groups[name]])
        table = FrozenDiskTable(ROOT, parent.manifest['populations'][name]['source_table'])
        values = {}
        for role, xy in table.coordinates.items():
            points = xy[fine]
            values[role] = {'bounds_mm': [float(points[:, 0].min()), float(points[:, 0].max()),
                float(points[:, 1].min()), float(points[:, 1].max())],
                'radial_extent_mm': [float(np.linalg.norm(points, axis=1).min()), float(np.linalg.norm(points, axis=1).max())]}
        fields[name] = {'fine_members': len(fine), 'old_group_count': len(groups[name]), 'coordinates': values,
            'table': table.binding, 'old_density_geometry_not_reused': name == 'HC'}
    profile = HCDensityProfile(ROOT)
    pedicle = fields['cones']['coordinates']['opl_pedicle']['bounds_mm']
    legacy = fields['HC']['coordinates']['soma']['bounds_mm']
    radius = max(float(profile.contact_radius_mm(1.338, name)) for name in ('H1', 'H2'))
    hull = [min(pedicle[0] - radius, legacy[0]), max(pedicle[1] + radius, legacy[1]),
            min(pedicle[2] - radius, legacy[2]), max(pedicle[3] + radius, legacy[3])]
    corners = np.array([[hull[i], hull[j]] for i in (0, 1) for j in (2, 3)])
    maximum_hull = float(np.linalg.norm(corners, axis=1).max())
    if maximum_hull > profile.support_radius_mm:
        result = {'schema': 'gin69-HC2-BC1-source-anatomy-preparation-v1', 'source_cell': '69#6',
            'source_eccentricity_deg': 3.88, 'model_reference_meridian_deg': 0.,
            'old_source_facts': fields, 'HC_contact_envelope_max_mm': radius,
            'new_HC_required_view_bounds_mm': hull, 'required_rectangle_corner_radius_mm': maximum_hull,
            'permitted_HC_soma_contact_anchor_radius_mm': profile.support_radius_mm,
            'status': 'BLOCKED_HC_APPLICATION_DOMAIN_NO_EXTRAPOLATION',
            'joint_pair_context_not_yet_added': True,
            'current_HC_identity': current_hc(ROOT)['identity'], 'current_BC_identity': current_bc(ROOT)['identity'],
            'neural_run': False, 'population_constructed': False, 'background_reused': False,
            'wall_seconds': time.perf_counter() - started}
        result['identity'] = canonical_hash(result)
        (output / 'NEW_GEOMETRY_DOMAIN.json').write_text(json.dumps(result, indent=2), encoding='utf8')
        print(json.dumps(result))
        return
    conservative = profile.bounds_for_rectangle(hull)
    ref = profile.reference_density
    context = float(np.max(profile.reference_lengths_mm * np.sqrt([
        ref[0] / conservative['H1_min'], ref[1] / conservative['H2_min'], ref.sum() / conservative['total_min']])))
    bounds = [hull[0] - context, hull[1] + context, hull[2] - context, hull[3] + context]
    corners = np.array([[bounds[i], bounds[j]] for i in (0, 1) for j in (2, 3)])
    maximum = float(np.linalg.norm(corners, axis=1).max())
    result = {'schema': 'gin69-HC2-BC1-source-anatomy-preparation-v1', 'source_cell': '69#6',
        'source_eccentricity_deg': 3.88, 'model_reference_meridian_deg': 0.,
        'old_source_facts': fields, 'HC_contact_envelope_max_mm': radius,
        'new_HC_view_bounds_mm': hull, 'joint_mosaic_pair_context_mm': context,
        'new_HC_canonical_bounds_mm': bounds, 'maximum_HC_canonical_radius_mm': maximum,
        'domain_within_v2': maximum <= 1.338,
        'current_HC_identity': current_hc(ROOT)['identity'], 'current_BC_identity': current_bc(ROOT)['identity'],
        'neural_run': False, 'background_reused': False, 'wall_seconds': time.perf_counter() - started}
    result['identity'] = canonical_hash(result)
    (output / 'NEW_GEOMETRY_DOMAIN.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
