"""Read only new BC contacts and existing HC coverage; no dynamics or generation."""
from pathlib import Path
import json
import sys
import time
import numpy as np
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.spatial_group_graph import verified_path
from tools.durable_research_runtime_v1 import atomic_json

ROOT = Path(r'D:\macaque retinal simulator')

def main(output):
    start = time.perf_counter()
    catalog = json.loads((ROOT/'data/atlas/GIN_69_6_HC2_BC1_V1/BC/BC_catalog.json').read_bytes())
    columns = []
    facts = {}
    for name, binding in catalog['routes'].items():
        record = json.loads(verified_path(ROOT, binding).read_bytes())
        ids = []
        for block in record['blocks']:
            with np.load(verified_path(ROOT, block), allow_pickle=False) as z:
                ids.append(z['indices'].copy())
        unique = np.unique(np.concatenate(ids))
        facts[name] = dict(unique_cone_columns=len(unique), edges=record['edges'])
        columns.append(unique)
    columns = np.unique(np.concatenate(columns))
    table = FrozenDiskTable(ROOT, record['recipe']['pre_table'])
    xy = np.asarray(table.coordinates['opl_pedicle'][columns])
    cfg = json.loads((ROOT/'configurations/research/HC69_HC2_BC1_master_v1.json').read_bytes())
    b = cfg['canonical_bounds_mm']
    distances = np.column_stack((xy[:,0]-b[0], b[1]-xy[:,0], xy[:,1]-b[2], b[3]-xy[:,1]))
    contained = np.min(distances, axis=1) >= 0
    result = dict(schema='GIN69-new-BC-direct-Cone-dependency-scope-1',
        scope='All saved BC rows, conservative superset of final selected runtime BC; not a final closure claim',
        cone_table=record['recipe']['pre_table'], unique_cones=len(columns), routes=facts,
        pedicle_bounds_mm=[float(xy[:,0].min()),float(xy[:,0].max()),float(xy[:,1].min()),float(xy[:,1].max())],
        pedicle_radial_extent_mm=[float(np.linalg.norm(xy,axis=1).min()),float(np.linalg.norm(xy,axis=1).max())],
        existing_HC_master_bounds_mm=b, outside_existing_HC_rectangle_count=int(np.count_nonzero(~contained)),
        minimum_rectangle_clearance_mm=float(distances.min()),
        source_domain_radius_mm=1.338, source_domain_contains_all_cones=bool(np.all(np.linalg.norm(xy,axis=1)<=1.338)),
        no_model_or_population_generated=True, wall_seconds=time.perf_counter()-start)
    np.save(output/'BC_REQUIRED_CONE_ROWS.npy', columns, allow_pickle=False)
    atomic_json(output/'DEPENDENCY_SCOPE.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    main(Path(sys.argv[2]))
