"""Freeze new BC anchors/contacts for the unchanged69#6 downstream population."""
from pathlib import Path
import json
import time
import numpy as np
from retina_sim.spatial_group_model import GroupModelAssembly, AC, RGC
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.disk_population import FixedDiskPopulation
from retina_sim.disk_inner import freeze_bc_tables
from tools.durable_research_runtime_v1 import atomic_json

ROOT=Path(r'D:\macaque retinal simulator')
OUT=ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2'

def build(output,guard):
    started=time.perf_counter()
    domain=json.loads((ROOT/'configurations/research/GIN69_OWNER_SUPPORT_V2.json').read_bytes())
    radius=domain['BC_anchor_radius_mm']
    base=GroupModelAssembly(ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2')
    old=ROOT/'reports/GIN_69_6_RADIAL_CONDITIONAL_V1/candidate_v2/prepared/local_static/selection.npz'
    facts={};all_points=[]
    with np.load(old,allow_pickle=False) as selected:
        for name in (*AC,*RGC,'AII_AC'):
            guard();pop=base.populations[name];rows=selected['population__'+name]
            fine=np.concatenate([pop.member_global_rows[pop.member_offsets[r]:pop.member_offsets[r+1]] for r in rows]) if len(rows) else np.empty(0,np.int64)
            binding=base.manifest['populations'][name]['source_table']
            if name=='AII_AC':
                points=FixedDiskPopulation(ROOT,binding).positions[fine]
                coordinate='existing AII physical IPL chart used by current gap/chemical rules'
            else:
                table=FrozenDiskTable(ROOT,binding)
                role='ipl_post_arbor' if 'ipl_post_arbor' in table.coordinates else 'ipl_preterminal'
                points=table.coordinates[role][fine];coordinate=role
            facts[name]={'unchanged_source_table':binding,'actual_fine_members':len(fine),'coordinate':coordinate,
                'radial_extent_mm':[float(np.linalg.norm(points,axis=1).min()),float(np.linalg.norm(points,axis=1).max())] if len(fine) else None}
            if len(points):all_points.append(points)
    points=np.concatenate(all_points);hard=.04
    bounds=[float(points[:,0].min()-hard),float(points[:,0].max()+hard),
        float(points[:,1].min()-hard),float(points[:,1].max()+hard)]
    required=float(np.linalg.norm(points,axis=1).max()+hard)
    scope={'source_cell':'69#6','source_eccentricity_deg':3.88,'unchanged_downstream_facts':facts,
        'BC_anchor_query':{'bounds_mm':bounds,'unit':'mm','frame':'OPL'},
        'required_actual_BC_anchor_support_radius_mm':required,'permitted_radius_mm':radius,
        'existing_downstream_hard_support_mm':hard,'old_BC_coordinates_or_connections_reused':False,
        'newBC_quota':'full RF owner0..1.338mm ancestry before local pedicle crop','owner_support':domain}
    atomic_json(output/'BC_SCOPE.json',scope)
    if required>radius:
        raise RuntimeError('Actual selected downstream BC dependency envelope exceeds BC candidate scope: '+str(required))
    support={'limits':{'inner_OPL_support_radius_mm':radius},'bc_anchor_query':scope['BC_anchor_query']}
    guard()
    result=freeze_bc_tables(ROOT,{'cones':base.manifest['populations']['cones']['source_table']},support,
        'data/atlas/GIN_69_6_HC2_BC1_SUPPORT_V2/BC')
    guard()
    result.update(wall_seconds=time.perf_counter()-started,scope=scope,neural_run=False)
    atomic_json(output/'BC_RESULT.json',result)
    atomic_json(OUT/'BC_catalog.json',result)
    print(json.dumps(result),flush=True)
