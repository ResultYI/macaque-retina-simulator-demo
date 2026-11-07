"""Materialize only the static sources required by the local H1 and held B context."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.io import savemat
from scipy.spatial import cKDTree

ROOT=Path(r'D:\macaque retinal simulator')
sys.path.insert(0,str(ROOT))
from retina_sim.gin69_outer_gaussian import outer_apertures
from retina_sim.gin69_frozen_aperture import resolve_apertures
from tools.prepare_gin69_joint_hc import guard


def main():
    out=ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2'
    dest=out/'joint_gray_background';dest.mkdir(exist_ok=True)
    if any(dest.iterdir()):raise FileExistsError('Existing background preparation retained')
    source=ROOT/'reports/CONE_HC_OUTER_SOURCE_EXTENSION_20261102_v3/cones_01'
    with np.load(out/'joint_functional_assets/SOURCE_LABELS.npz',allow_pickle=False) as z:
        ids=z['cone_candidate_rows'];ct=z['cone_types'].astype('U1')
    ped=np.load(source/'cone_pedicle_mm.npy',mmap_mode='r',allow_pickle=False)[ids]
    xy=np.load(source/'cone_inner_segment_mm.npy',mmap_mode='r',allow_pickle=False)[ids]
    rec=json.loads((out/'executions/assemble_native_01/NATIVE_ASSEMBLY_RESULT.json').read_bytes())
    hm=json.loads((ROOT/rec['hc_table']['path']).read_bytes())
    hx=np.load(ROOT/hm['coordinates']['soma']['path'],allow_pickle=False)
    ht=np.load(ROOT/hm['extras']['subtype']['path'],allow_pickle=False).astype('U2')
    # Full parent H1 supports are also used to produce the original held B background.
    mask=(ct!='S')&(cKDTree(hx[ht=='H1']).query(ped)[0]<=3.526911366748626)
    ids=ids[mask];ct=ct[mask];xy=xy[mask];ped=ped[mask]
    profile=json.loads((ROOT/'configurations/research/macaque_cone_aperture_eccentricity_v1.json').read_bytes())
    with np.load(out/'optical/APERTURE_GEOMETRY.npz',allow_pickle=False) as z:
        known={k:z[k].copy() for k in z.files}
    check=outer_apertures(ROOT,known['positions_mm'],profile)
    for key in ('reference_diameter_um','effective_area_m2','zone_diameter_um'):
        np.testing.assert_array_equal(check[key],known[key])
    ap=outer_apertures(ROOT,xy,profile);guard()
    np.savez_compressed(dest/'SOURCE_GEOMETRY.npz',candidate_rows=ids,cone_types=ct,IS_mm=xy,pedicle_mm=ped,
        reference_diameter_um=ap['reference_diameter_um'],zone_diameter_um=ap['zone_diameter_um'],
        effective_area_m2=ap['effective_area_m2'])
    savemat(dest/'INPUT.mat',dict(eccentricity_deg=ap['eccentricity_deg'],
        types=np.where(ct=='L',1,2),diameters_um=ap['reference_diameter_um'],zones_um=ap['zone_diameter_um'],
        area_m2=ap['effective_area_m2']),do_compression=True)
    cfg=json.loads((out/'optical/CONFIG.json').read_bytes())
    cfg.update(extension_input=str(dest/'INPUT.mat'),extension_output=str(dest/'CAPTURE.mat'),
               extension_run=str(out/'executions/joint_gray_capture_01'))
    (dest/'CONFIG.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='STATIC_GRAY_INPUT_READY',sources=len(ids),above_6deg=int((ap['eccentricity_deg']>6).sum()),
        old_fields_exact=True,diameter_range_um=[float(ap['reference_diameter_um'].min()),float(ap['reference_diameter_um'].max())])))


if __name__=='__main__':main()
