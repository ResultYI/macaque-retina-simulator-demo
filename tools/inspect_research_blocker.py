"""Read exact failed geometry; never rerun its mosaic or invent connections."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial import cKDTree

def inspect(out):
    run=next((out/'run').iterdir());partial=json.loads((run/'partial_build_audit.json').read_text(encoding='utf-8'))
    hc=partial['HC_geometry'];a=hc['contact_audit'];ipl=json.loads((out/'ipl_geometry.json').read_text(encoding='utf-8'))
    points=np.array([r['arbor_anchor_xy_OPL_mm'] for r in a['hc_geometry']]);types=np.array([r['type'] for r in a['hc_geometry']])
    radii={'H1':.00804,'H2':.01222};details=[]
    for cone in partial['uncovered_cones']:
        row=dict(cone,missing_feedback_contacts=0,missing_unit_DC_input=True,nearest_eligible_arbor={})
        for t in ('H1','H2'):
            if t=='H1' and cone['type']=='S':continue
            indices=np.flatnonzero(types==t);dist,i=cKDTree(points[indices]).query(cone['pedicle_OPL_mm']);j=int(indices[int(i)])
            row['nearest_eligible_arbor'][t]={'hc_id':a['hc_geometry'][j]['hc_id'],
                'anchor_OPL_mm':points[j].tolist(),'anchor_distance_mm':float(dist),'fixed_disk_radius_mm':radii[t],
                'passes_finite_disk_narrow_phase':bool(dist<=radii[t]+1e-12),'passes_final_40um':bool(dist<=.04+1e-12)}
        details.append(row)
    result={'status':'FULL_NATIVE_BUILD_BLOCKED','failed_run_directory':str(run.resolve()),
        'time_integration':'NOT_RUN','source_geometry_cells':len(a['paired_cones']),'HC_active_cells':len(points),
        'HC_edges':a['edge_count'],'uncovered_feedback_cones':details,'uncovered_owner_count':sum(c['is_owner'] for c in details),
        'IPL_static_geometry_status':'PASS_DECLARED_INPUT_CONDITIONS; not a native assembly bypass',
        'owner_counts':ipl['owner_counts'],'IPL_mandatory_deficits':len(ipl['degree_deficits']),
        'S_relay_legacy_exact_degree_mismatch':len(ipl['legacy_exact_degree_mismatch']),
        'native_blocker':'hc_contact_runtime_view and NativeConeH1 require nonzero unit-DC feedback rows; all 11 have zero legal finite contacts',
        'potential_change_requires_new_explicit_model_policy':'Finite zero-HC-feedback rows would change native spatial admissibility and background equilibrium construction; not one of the supplied three rules',
        'prohibited_workarounds_used':False,'seed_or_domain_search_for_pass':'NOT_RUN',
        'old_strict_failures':'PRESERVED; this failure is a separate new-profile attempt',
        'population_diagnostics':a['population_diagnostics'],'HC_active_generation_envelope_mm':hc['active_bounds_mm'],
        'actual_HC_anchor_extent':a['actual_HC_anchor_extent'],'actual_HC_footprint_extent':a['actual_HC_footprint_extent'],
        'pedicle_query':a['required_pedicle_query_domain'],'physiological_validation':'NOT_RUN'}
    from retina_sim.runtime_profiles import frozen_packer_table
    table=frozen_packer_table(ROOT);rf=np.array([c['input_xy_mm'] for c in a['paired_cones']])
    result['actual_provider_support']={'scope':'read-only derivation from exact saved RF/OPL instance and frozen provider descriptor',
        'RF_radial_extent_mm':[float(np.linalg.norm(rf,axis=1).min()),float(np.linalg.norm(rf,axis=1).max())],
        'density_asset_sha256':table.asset_sha256,
        'tabulated_radius_by_meridian_mm':{m:[min(float(r['eccentricity_mm']) for r in table.rows if r['meridian']==m),
            max(float(r['eccentricity_mm']) for r in table.rows if r['meridian']==m)] for m in ('nasal','temporal')},
        'RF_materialized_bounds_mm':ipl['generation_plan']['RF_generation_bounds_mm'],
        'OPL_materialized_bounds_mm':a['cone_pedicle_generation_bounds_mm'],
        'physical_arbor_query_guard_upper_bounds_mm':[.45778,1.05222],
        'query_guard_is_anchor_permission':False,'required_owner_dependency_truncated':False,
        'outer_halo_arbor_truncated_count':a['required_pedicle_query_domain']['halo_incomplete_count'],
        'source_to_OPL_mapping':'named McGregor numerical P-chart project adapter; no measured continuous full-2D map claim'}
    (out/'blocker_details.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out.parent/'research_prototype_latest.json').write_text(json.dumps({'acceptance_directory':str(out.resolve()),'status':'FULL_NATIVE_BUILD_BLOCKED'},indent=2)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    result=inspect(Path(sys.argv[1]));print(json.dumps({k:v for k,v in result.items() if k not in ('uncovered_feedback_cones','population_diagnostics','actual_HC_anchor_extent','actual_HC_footprint_extent','pedicle_query')},indent=2))
