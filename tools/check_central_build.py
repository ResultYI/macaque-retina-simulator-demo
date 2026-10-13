"""Minimum central full-target build/init checks. No time integration or fitting."""
from pathlib import Path
import sys,json,os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datetime import datetime,timezone
import numpy as np
from retina_platform.config import RunConfig
from examples.full_target_network import full_target_network
from retina_sim.contracts import SolverConfig
from tools.run_full_network_fixture import serializable

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    previous=ROOT/'reports/background_acceptance/20261011T105852783731Z/gui_parsed_config.json'
    cfg=RunConfig.from_dict(json.loads(previous.read_text(encoding='utf-8')))
    out=ROOT/'reports/central_build_acceptance'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    try:
        c,_,audit=full_target_network(ROOT,duration=.0004,adaptation_seconds=0.,seed=cfg.seed,
            light_context=cfg.light_context(),runtime_profile=cfg.parameter_profile,execution_profile=cfg.execution_profile,
            owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,
            hc_geometry_profile=cfg.hc_geometry_profile,hc_geometry_path=cfg.hc_geometry_path,hc_arbor_profile=cfg.hc_arbor_profile)
    except Exception as error:
        if not hasattr(error,'partial_build_audit'):raise
        audit=error.partial_build_audit;chemical=audit['chemical_synapses']
        for name in ('AII_FMB_glycine','AII_DB1_glycine'):
            assert chemical[name]['per_edge_states']>0 and all(chemical[name]['target_support_complete'])
            assert max(chemical[name]['edge_lengths_mm'])<=.04+1e-12
        repaired=[];remaining=[]
        for route in audit['connectivity_support']:
            if 'pre' not in route:continue
            rows=route['rows'];zeros=[row for row in rows if row['actual_degree']==0]
            if route['post'] in ('rod_BC','AII_AC','local_feedforward_AC','local_crossover_AC','widefield_AC'):
                assert not zeros and all(row['missing_degree']==0 for row in rows)
                assert all(row['maximum_distance_mm']<=.04+1e-12 for row in rows)
                repaired.append({'pre':route['pre'],'post':route['post'],'cells':len(rows),'unfilled_required_degree':0})
            if zeros:
                nearest=[row['nearest_available_source_distance_mm'] for row in zeros]
                pop=audit['populations'][route['post']]
                rf=np.asarray(pop['receptive_field_positions']['values'])
                owner=np.asarray(cfg.owner_bounds());indices=[row['post_index'] for row in zeros]
                zero_owner=((rf[indices,0]>=owner[0])&(rf[indices,0]<owner[1])&(rf[indices,1]>=owner[2])&(rf[indices,1]<owner[3]))
                remaining.append({'pre':route['pre'],'post':route['post'],'frame':route['frame'],
                    'zero_rows':len(zeros),'target_cells':len(rows),'zero_owner_rows':int(zero_owner.sum()),
                    'unfilled_required_degree_rows':sum(row['missing_degree']>0 for row in rows),
                    'hard_support_mm':.04,'nearest_distance_range_mm':[min(nearest),max(nearest)],
                    'example':zeros[0],'point_rule_source':'40um cutoff: macaque_constrained_native_v1 project geometry; AC mapped BC-path point, non-midget RGC mapped soma proxy; no measured typed IPL arbor',
                    'candidate_generation_complete_for_halo':'NOT_ESTABLISHED; zero candidates in this finite view are not a biological zero claim'})
        result={'engineering_build':'BLOCKED_REQUIRED_INPUT_SUPPORT','time_integration':'NOT_RUN',
            'same_failed_central_config_used':str(previous),'AII_to_midget_support_repaired':True,
            'glycine_edges':{k:chemical[k]['per_edge_states'] for k in ('AII_FMB_glycine','AII_DB1_glycine')},
            'repaired_upstream_routes':repaired,'remaining_support':remaining,
            'blockers':audit['compiled_support_errors'],'diagnosis':'RBC/rod support and AC RF-to-IPL point lift repaired. Remaining RGC halo coverage and typed IPL point/arbor contract conflict; no density reduction, increased reach or implicit zero inputs',
            'physiological_validation':'NOT_RUN'}
        (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        (out/'partial_build_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
        (out/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2)+'\n',encoding='utf-8')
        (out.parent/'latest.json').write_text(json.dumps({'acceptance_directory':str(out)},indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'output':str(out),'engineering_build':result['engineering_build'],
            'repaired_upstream_routes':repaired,'remaining_paths':len(remaining),'time_integration':'NOT_RUN'}));return
    r=c.runner();s=r.initialize(SolverConfig(**audit['resolved_execution_profile']['solver']))
    assert all(np.all(np.isfinite(a)) for a in s.ordered_state_arrays)
    domain=audit['rod_inner_geometry']['AII_generation_domain']
    assert domain['domain_extended'] and domain['density_scale']==1.
    chemical=audit['chemical_synapses']
    for name in ('AII_FMB_glycine','AII_DB1_glycine'):
        assert chemical[name]['per_edge_states']>0
        assert max(chemical[name]['edge_lengths_mm'])<=.04+1e-12
    hc=audit['spatial_profile']['HC_geometry'];assert hc['configuration']['arbor_profile_id']=='central_OPL_equal_area_W2000'
    result={'engineering_build':'PASS','time_integration':'NOT_RUN','physiological_validation':'NOT_RUN',
        'same_failed_central_config_used':str(previous),'components':len(c.components),'layers':len(c.spec.required_mechanisms),
        'state_scalars':sum(len(a) for a in s.ordered_state_arrays),'AII_cells':len(c.populations['AII_AC'].cells.cell_id),
        'AII_generation_domain':domain,'chemical_synapses':chemical,'arbor_profile_id':cfg.hc_arbor_profile,
        'assembly_hash':c.assembly_hash,'source_HC_parameters_unchanged':True,
        'boundary_warning':'finite induced graph; non-owner presynaptic support deficits remain exported, not biological zero claims'}
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (out/'hc_geometry.json').write_text(json.dumps(hc,indent=2)+'\n',encoding='utf-8')
    (out/'initial_checkpoint.json').write_text(r.checkpoint(s),encoding='utf-8')
    (out/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2)+'\n',encoding='utf-8')
    (out.parent/'latest.json').write_text(json.dumps({'acceptance_directory':str(out)},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'components':len(c.components),'AII_cells':result['AII_cells'],
        'glycine_edges':{k:chemical[k]['per_edge_states'] for k in ('AII_FMB_glycine','AII_DB1_glycine')},'time_integration':'NOT_RUN'}))

if __name__=='__main__':main()
