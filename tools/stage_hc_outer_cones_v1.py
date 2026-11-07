"""Materialize a larger position-only candidate from the SAME official source.

No selection of this candidate, subtype assignment, HC extrapolation or neural
run is performed. Old source-row identities and all overlap positions survive.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib
import numpy as np
from scipy.io import loadmat
from retina_sim.isetbio_geometry import HUMAN_ASSET_SHA256
from retina_sim.henle_bridge import ScheinBoundedBridge
from retina_sim.hc_expansion_geometry import bounded_pedicles,native_td_step


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    source=root/'vendor_pinned/isetbio/data/sdr/isetbio-mosaics/cone_lattices/right/nominal-fov-58deg.mat'
    if sha(source)!=HUMAN_ASSET_SHA256:raise RuntimeError('Official source bytes changed')
    d=loadmat(source,variable_names=['rfPositions','whichEye','neuronType'],squeeze_me=True)
    if d['whichEye']!='right eye' or d['neuronType']!='cones':raise RuntimeError('Wrong source population')
    all_xy=-np.asarray(d['rfPositions'],dtype=np.float64)/1000.
    rows=np.flatnonzero(np.linalg.norm(all_xy,axis=1)<=6.)
    xy=all_xy[rows]
    print('source selection',len(rows),flush=True)
    old_dir=root/'data/atlas/isetbio_human_borrowed_cones_disk3mm_v1'
    manifest=json.loads((old_dir/'manifest.json').read_bytes())
    if sha(root/manifest['payload']['path'])!=manifest['payload']['sha256']:raise RuntimeError('Old asset identity changed')
    with np.load(root/manifest['payload']['path'],allow_pickle=False) as old:
        old_rows=old['official_source_rows'];indices=np.searchsorted(rows,old_rows)
        if np.any(indices>=len(rows)) or not np.array_equal(rows[indices],old_rows):raise RuntimeError('Old source rows omitted')
        if not np.array_equal(xy[indices],old['positions_mm']):raise RuntimeError('Overlap positions changed')
        expected=np.array(['sdr_human_54eb0a9ef905_row:'+str(i) for i in old_rows])
        if not np.array_equal(expected,old['source_keys']):raise RuntimeError('Source-key convention changed')
        overlap_count=len(indices)
    bridge=ScheinBoundedBridge(root)
    pedicles=bounded_pedicles(bridge,xy)
    np.save(out/'official_source_rows.npy',rows,allow_pickle=False)
    np.save(out/'cone_inner_segment_mm.npy',xy,allow_pickle=False)
    np.save(out/'cone_pedicle_mm.npy',pedicles,allow_pickle=False)
    stimulus=dict(center_mm=[1.0744117751749263,-.01152576578789823],inner_radius_mm=.12,
        outer_radius_mm=.32,contrast=.01,onset_seconds=.02,offset_seconds=.12)
    # Native illuminance controls are not reused from display/video captures.
    for label,t in [('background',0.),('annulus_on',.02),('annulus_off',.12)]:
        np.save(out/(label+'_td.npy'),native_td_step(xy,1000.,time_seconds=t,**stimulus),allow_pickle=False)
    r=dict(status='POSITION_AND_NATIVE_INPUT_CANDIDATE_MATERIALIZED_NOT_SELECTED',
        completed_utc=datetime.now(timezone.utc).isoformat(),official_source_sha256=HUMAN_ASSET_SHA256,
        official_source_count=len(all_xy),candidate_count=len(xy),old_overlap_count=overlap_count,
        added_positions=len(xy)-overlap_count,old_overlap_exact=True,
        source_key_rule='sdr_human_54eb0a9ef905_row:<official_source_row>',
        IS_radius_mm=6.,Henle_profile=bridge.provenance,
        biological_scope='Existing human-borrowed geometry; not macaque measured positions',
        subtype_assignment='NOT_ASSIGNED; official position source contains no LMS types',
        HC_connections='NOT_BUILT_PENDING_SCIENTIFIC_CHOICE',
        input=dict(kind='native Td equation-level diagnostic; not optical/display capture',background_td=1000.,
            stimulus=stimulus,all_candidate_cones_have_background_and_on_off_inputs=True),
        recurrent_closure_complete=False,production_selected=False)
    r['files']={p.name:sha(p) for p in out.glob('*.npy')}
    (out/'RESULT.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps({k:r[k] for k in ['status','candidate_count','old_overlap_count','added_positions','old_overlap_exact']}),flush=True)


if __name__=='__main__':main()
