"""Freeze BC populations/contacts using the selected registered-pedicle policy."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from .bc_current import current_bc
from .bc_pedicle_density import RATIOS, RegisteredPedicleField, anchor_rows, crop_anchor_rows
from .contracts import ContractError, Quantity
from .disk_tables import FrozenDiskTable, write_table
from .disk_graph_storage import SparseRouteWriter, save_json
from .ipl_geometry import read_ipl_geometry, RESEARCH

def freeze_pedicle_bc_tables(root, outer, support, directory):
    root = Path(root).resolve()
    directory = Path(directory)
    selected = current_bc(root)
    if selected is None:
        raise ContractError('Current BC policy not selected')
    config = json.loads((root / selected['bindings']['parameters']['path']).read_bytes())
    if config['ratios'] != {name: [v.numerator, v.denominator] for name, v in RATIOS.items()}:
        raise ContractError('Frozen BC density ratios changed; implementation/parameter identity must agree')
    cone = FrozenDiskTable(root, outer['cones'])
    field = RegisteredPedicleField(cone, config)
    radius = float(support['limits']['inner_OPL_support_radius_mm'])
    if radius <= 0 or radius > field.scope or (field.owner_support and radius != field.scope):
        raise ContractError('BC_V1_ANCHOR_DOMAIN_UNSUPPORTED: explicit complete owner/support domain required')
    ids = tuple(cone.ids(0, cone.count))
    eligible = field.owner_rows if field.owner_support else np.flatnonzero(np.linalg.norm(field.xy, axis=1) <= radius)
    groups = anchor_rows(ids, field.labels, eligible)
    canonical_counts = {name:len(rows[0]) for name,rows in groups.items()}
    query = support.get('bc_anchor_query')
    groups = crop_anchor_rows(groups,field.xy,query)
    cfg = read_ipl_geometry(root, RESEARCH)
    version = cone.metadata['source_version'] + '|BC_pedicle_v1|' + selected['identity']
    result = {'tables': {}, 'routes': {}, 'inner_source_version': version,
        'IPL_profile_sha256': cfg['profile_sha256'], 'bc_density_geometry_identity': selected['identity'],
        'density_semantics': 'functional input-anchor density on OPL pedicles; soma transform absent',
        'owner_support': field.owner_support,
        'RBC': 'unchanged independent rod population and contacts', 'subtype_density_claim': False,
        'status': 'CURRENT_BC_TABLES_AND_DIRECT_CONTACTS_CONSTRUCTING', 'population_counts': {},
        'canonical_population_counts': canonical_counts, 'anchor_query': query,
        'crop_policy': 'Complete canonical quota allocation before crop; same cells/IDs/slots across overlapping queries, no ROI resampling'}
    for name, (indices, slots) in groups.items():
        keys = tuple(ids[i] if name.startswith('midget') else ids[i] + ':bc_anchor_slot:' + str(slot)
                     for i, slot in zip(indices, slots))
        metadata = dict(cone.metadata, population=name, source_version=version)
        table = write_table(root, directory / name, metadata, len(indices), lambda a, b: keys[a:b],
            {'rf': Quantity(field.rf[indices], 'mm', 'functional_position', 'project_approximation', 'functional_visual_field'),
             'ipl_preterminal': Quantity(field.xy[indices], 'mm', 'terminal_position', 'project_approximation', 'IPL')},
            extras={'cone_row': indices, 'bc_anchor_slot': slots, 'eligible_drive': np.ones(len(indices), bool)},
            source_bindings={'cones': cone.binding, 'IPL_profile_sha256': cfg['profile_sha256'],
                'bc_density_geometry_identity': selected['identity'],
                'density_semantics': result['density_semantics'], 'soma': 'ABSENT; no invented displacement',
                'OPL_to_IPL': 'existing effective co-location; not measured morphology',
                'rf_semantics': 'source Cone ancestry coordinate, not a measured BC functional RF',
                'parentage': 'exact private Cone ID' if name.startswith('midget') else 'sampling anchor only, no private input claim',
                'finite_population_sampling': 'exact rational round-half-up quota on full canonical parents; deterministic hash ranking; never ROI resampled',
                'registration': cone.record['source_bindings']['Schein'],
                'physical_support_radius_mm': radius, 'global_row_gain': 1.,
                'owner_support': field.owner_support,
                'anchor_query': query, 'canonical_quota_count': canonical_counts[name]})
        route = 'cones->' + name
        recipe = {'route_id': route, 'pre_population': 'cones', 'post_population': name,
            'pre_table': cone.binding, 'post_table': table.binding, 'pre_count': cone.count, 'post_count': table.count,
            'bc_density_geometry_identity': selected['identity'], 'kernel': 'uniform_direct_contact_disk_or_private_ID',
            'weights': 'fine full-parent row sum1; no crop renormalization, no member multiplication',
            'boundary': 'explicit finite registered parent; incomplete rows held unsupported, no nearest fill',
            'R_is_functional_RF': False, 'density_estimator': config['local_density'],
            'eligibility': 'ON-midget LM; OFF-midget LMS; diffuse LM; S-ON S'}
        writer = SparseRouteWriter(root, directory / 'routes' / route.replace('->', '_to_'), recipe)
        if not writer.manifest.exists():
            for start in range(writer.next_row, table.count, 4096):
                stop = min(table.count, start + 4096)
                rows, columns, weights, audits, statuses = [], [], [], [], []
                for j in range(start, stop):
                    parent = int(indices[j])
                    if name.startswith('midget'):
                        found = np.array([parent], dtype=np.int64)
                        audit = {'status': 'READY_PROJECT_PRIVATE_PARENT', 'actual_degree': 1,
                            'parent_cone_id': ids[parent], 'distance_mm': 0., 'selection': 'exact Cone ID, no radius'}
                    else:
                        found, audit = field.contacts(field.xy[parent], name)
                    statuses.append(audit['status'])
                    audits.append(dict(audit, sampling_anchor_cone_id=ids[parent],
                        anchor_slot=int(slots[j]), input_owner='private Cone' if name.startswith('midget') else 'explicit contact edges'))
                    rows.extend([j - start] * len(found))
                    columns.extend(found.tolist())
                    if len(found):
                        weights.extend([1. / len(found)] * len(found))
                matrix = sparse.csr_matrix((weights, (rows, columns)), shape=(stop - start, cone.count))
                writer.append(start, matrix, audits, row_status=statuses)
        result['tables'][name] = table.binding
        result['routes'][route] = writer.freeze()
        result['population_counts'][name] = table.count
        save_json(root / directory / 'BC_catalog.json', result)
    result.update(status='CURRENT_BC_TABLES_AND_DIRECT_CONTACTS_FROZEN', physiological_validation=False,
        FIT_READY=False, background_qualified=False, production_adopted=False)
    save_json(root / directory / 'BC_catalog.json', result)
    return result
