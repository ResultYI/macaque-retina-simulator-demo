"""Read-only real-parent first-hop witness; no graph rebuild or neural run."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np
from scipy import sparse
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.hc_current import current_hc
from retina_sim.hc_biexponential import support_radius


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    base=root/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/executions/assemble_native_01'
    record=json.loads((base/'NATIVE_ASSEMBLY_RESULT.json').read_bytes())
    current=current_hc(root)
    if record['current_HC_identity']!=current['identity']:raise RuntimeError('Saved HC graph is stale')
    cones=FrozenDiskTable(root,record['cone_table']);hc=FrozenDiskTable(root,record['hc_table'])
    raw=sparse.load_npz(base/'RAW_HC_BY_CONE.npz').tocsr()
    xy=cones.coordinates['opl_pedicle'];hx=hc.coordinates['soma']
    ct=cones.extras['subtype'].astype('U1');ht=hc.extras['subtype'].astype('U2')
    with np.load(base/'MEMBERSHIP.npz',allow_pickle=False) as z:
        local_h=np.isin(z['fine_to_group'],z['local_HC_rows'])
        represented=z['global_cone_fine_to_group']>=0
        native=np.isin(z['global_cone_fine_to_group'],z['local_cone_rows'])
    candidates=np.flatnonzero(local_h & (ht=='H1'))
    if not len(candidates):raise RuntimeError('No actual retained H1')
    j=int(candidates[np.argmin(np.linalg.norm(hx[candidates]-hx[candidates].mean(0),axis=1))])
    cfg=json.loads((root/'configurations/research/HC_biexponential_vh2007_reference_v1.json').read_bytes())
    radius=support_radius(cfg['H1']['lambda_l_um']/1000,cfg['tail_mass_budget'])
    distance=np.linalg.norm(xy-hx[j],axis=1)
    needed=np.flatnonzero((distance<=radius)&(ct!='S'))
    coo=raw.tocoo();allowed=((ht[coo.row]=='H1')&(ct[coo.col]!='S'))|((ht[coo.row]=='H2')&(ct[coo.col]=='S'))
    has_feedback=np.zeros(cones.count,bool);has_feedback[coo.col[allowed]]=True
    missing_feedback=needed[~has_feedback[needed]]
    missing_groups=needed[~represented[needed]]
    missing_dynamic=needed[~native[needed]]
    envelope=float(np.linalg.norm(xy,axis=1).max())
    necessary_outer=float(np.linalg.norm(hx[j])+radius)
    np.savez_compressed(out/'FIRST_HOP_WITNESS.npz',hc_parent_row=j,required_cone_parent_rows=needed,
        missing_local_feedback=missing_feedback,missing_group_members=missing_groups,
        outside_existing_native_input_rows=missing_dynamic)
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status='BLOCKED_FIRST_HOP',
        current_HC_identity=current['identity'],hc_parent_row=j,hc_id=hc.ids(j,j+1)[0],
        hc_anchor_mm=hx[j].tolist(),tail_mass_budget=cfg['tail_mass_budget'],support_radius_mm=radius,
        required_single_hc_outer_radius_mm=necessary_outer,existing_cone_outer_envelope_mm=envelope,
        even_outer_envelope_insufficient=necessary_outer>envelope,
        available_first_hop_cones=len(needed),missing_local_feedback=len(missing_feedback),
        absent_existing_group_map=len(missing_groups),outside_existing_native_input_rows=len(missing_dynamic),
        input_scope='Outside selected native rows is not certified stimulus-synchronous; not proof no optical file exists anywhere',
        finite_graph_closure_completed=False,missing_exterior_cone_count='unknown: no source cells outside available geometry',
        HC_experiment='NOT_RUN',cone_surround='NOT_RUN',RGC_experiment='NOT_RUN',production_changed=False)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    files=[base/'RAW_HC_BY_CONE.npz',base/'MEMBERSHIP.npz',base/'NATIVE_ASSEMBLY_RESULT.json',
        root/'configurations/research/HC_biexponential_vh2007_reference_v1.json',Path(__file__)]
    (out/'IDENTITIES.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
