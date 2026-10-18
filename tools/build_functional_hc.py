"""Build the approved reversible HC functional alternative on fixed real anchors."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.disk_graph_storage import file_sha,save_json
from retina_sim.disk_hc_functional import freeze_functional_hc

def main():
    report=ROOT/'reports/human_disk3mm_functional_hc_20261017';report.mkdir(parents=True,exist_ok=True)
    previous=ROOT/'reports/human_disk3mm_build_20261017'
    original=json.loads((ROOT/'configurations/atlas_human_borrowed_disk3mm_floor_input_v2.json').read_bytes())
    path=ROOT/original['global_graph']['path']
    if file_sha(path)!=original['global_graph']['sha256']:raise ValueError('Floor graph identity changed')
    graph=json.loads(path.read_bytes())
    outer=json.loads((previous/'outer_tables.json').read_bytes());anatomical=json.loads((previous/'outer_connections.json').read_bytes())
    save_json(report/'authorization.json',{'UTC':'2026-10-17T15:01:00Z','user_evidence':'Sentinel_4eb30bb2cc1481919b6ca9e8eabe4093',
        'scope':'Necessary reversible mainline model adjustments with explicit source/derived/borrowed/project labels; address effective-area hard cutoff, no invented cells/source claims, real targeted acceptance',
        'recovery_snapshot':'reports/workspace_recovery_20261017T150406Z/verification_complete.json',
        'macaque5855_snapshot_modified':False})
    operator=freeze_functional_hc(ROOT,outer,anatomical,'data/atlas/human_disk3mm_functional_hc_v1')
    save_json(report/'outer_connections.json',operator)
    replacement={operator[key]['path']:key for key in ('native_A','native_B')}
    for descriptor in graph['routes']:
        if descriptor['kind']=='native':
            key='native_A' if descriptor['route_id']=='native_cone_to_HC_A' else 'native_B'
            descriptor['binding']=operator[key]
    graph['native_components']=operator['native_component_labels'];graph['native_owner_zero_feedback_cones']=operator['owner_zero_feedback_cones']
    graph['native_operator_representation']={key:operator[key] for key in ('representation','profile','operator_audit','anatomical_reference')}
    graph['source_profiles']['HC_functional_operator']=operator['profile']
    graph['status']='FUNCTIONAL_HC_AND_FLOOR_MIDGET_FROZEN_PENDING_REAL_ACCEPTANCE'
    target=ROOT/'data/atlas/human_disk3mm_functional_hc_v1/circuit/manifest.json';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and json.loads(target.read_bytes())!=graph:raise ValueError('Existing functional graph differs; retain it')
    save_json(target,graph);binding={'path':target.relative_to(ROOT).as_posix(),'sha256':file_sha(target)}
    cfg=dict(original,global_graph=binding,status=graph['status'],native_operator_representation=graph['native_operator_representation'])
    cfg['migration']=dict(original['migration'],global_native_owner_zero_feedback_cones=0,
        retained_anatomical_owner_no_feedback_cones=anatomical['owner_zero_feedback_cones'],ready_for_single_default_switch=False)
    config=ROOT/'configurations/atlas_human_borrowed_disk3mm_functional_hc_v1.json'
    if config.exists() and json.loads(config.read_bytes())!=cfg:raise ValueError('Existing functional configuration differs')
    save_json(config,cfg);save_json(report/'global_graph.json',binding)
    save_json(report/'geometry_pipeline.json',{'status':'FUNCTIONAL_ALTERNATIVE_FROZEN_REQUIRES_TARGETED_ACTUAL_ACCEPTANCE',
        'default_changed':False,'configuration':config.relative_to(ROOT).as_posix(),'anatomical_central_coverage':'UNRESOLVED_REFERENCE_RETAINED',
        'physiological_validation':False})
    print('FUNCTIONAL_HC_ALTERNATIVE_FROZEN',operator['edges'],operator['operator_audit']['maximum_owner_functional_distance_mm'],flush=True)

if __name__=='__main__':main()
