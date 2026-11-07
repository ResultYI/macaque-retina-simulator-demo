"""Generate one native 0-6 degree optical exposure with explicit source pins."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys
import time
import numpy as np
from scipy.io import loadmat, savemat
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retina_sim.native_cone_apertures import PROFILE,native_apertures
from retina_sim.contracts import canonical_hash
from retina_platform.iset_frontend import digest
from retina_platform.disk_frontend import _run_batch

OLD=ROOT/'reports/cone_video_113_4s_eccentric_aperture_0_6deg_20261029_v1'
OUT=ROOT/'reports/ISETBIO_NATIVE_CONE_APERTURE_20261102_v1'


def main():
    started=time.time();OUT.mkdir(exist_ok=False)
    xy=np.load(ROOT/'data/atlas/human_disk3mm_typed_tables_v1/cones/rf.npy')
    diameters,areas=native_apertures(xy)
    distances,_=cKDTree(xy).query(xy,k=7)
    independent=.79*np.median(distances[:,1:],axis=1)*1000
    error=float(np.max(np.abs(diameters-independent)))
    assert error<1e-10,'Native block halo changes full-support apertures'
    with np.load(OLD/'geometry.npz',allow_pickle=False) as z:
        geometry={k:z[k].copy() for k in z.files if k!='sigma_um'}
    rows=geometry['source_table_rows']
    assert np.array_equal(geometry['xy_mm'],xy[rows])
    geometry.update(apertures_um=diameters[rows],effective_area_m2=areas[rows])
    np.savez_compressed(OUT/'geometry.npz',**geometry)
    plan=json.loads((OLD/'capture_plan.json').read_bytes())
    jobs=[];tasks=[];sources={str(p.relative_to(ROOT)):digest(p) for p in [
        ROOT/'retina_sim/native_cone_apertures.py',ROOT/'tools/generate_native_cone_apertures.m',
        ROOT/'tools/native_cone_apertures_from_positions.m',ROOT/'tools/native_cone_kernel_groups.m',
        ROOT/'tools/cone_video_gaussian_sampling.m',ROOT/'tools/cone_native_aperture.m',
        ROOT/'tools/cone_optics_mosaic.m',ROOT/'tools/export_cone_video_aperture_tile.m',
        ROOT/'tools/matlab/sample_cone_aperture_zones.m']}
    for task in plan['tasks']:
        if task['first']!=0:continue
        c=json.loads((Path(task['output']).parent/'export.json').read_bytes())
        selected=np.load(task['rows']);folder=OUT/('tile_%d_%d'%tuple(task['tile_index']));folder.mkdir()
        savemat(folder/'geometry.mat',dict(is_mm=geometry['xy_mm'][selected],
            types=np.array([{'L':1,'M':2,'S':3}[t] for t in geometry['types'][selected]]),
            reference_diameter_um=geometry['apertures_um'][selected],
            zone_diameter_um=geometry['apertures_um'][selected]),do_compression=True)
        c.update(geometry_mat=str(folder/'geometry.mat'),frame_indices=[1],window_edges_seconds=[0.,1/30],
                 aperture_model=PROFILE['profile_id'],aperture_profile_sha256=canonical_hash(PROFILE))
        c['tile_signature']=canonical_hash(dict(geometry=digest(folder/'geometry.mat'),sources=sources))
        c['request_signature']=canonical_hash(c)
        (folder/'export.json').write_text(json.dumps(c,indent=2),encoding='utf8')
        jobs.append(dict(configuration=str(folder/'export.json'),output=str(folder/'capture.mat')))
        tasks.append(dict(rows=selected,folder=folder,signature=c['request_signature']))
    (OUT/'REQUEST.json').write_text(json.dumps(dict(profile=PROFILE,sources=sources,
        complete_support_cones=len(xy),display_cones=len(rows),diameter_check_max_abs_um=error,
        reused_spectral_OI_cache=str(OLD/'optics_cache'),
        cache_scope='Aperture-independent optical PSF, spectrum and QE only; all Cone captures recalculated',
        source_rgb_sha256=digest(OLD/'source_rgb.mat')),indent=2),encoding='utf8')
    profile=c['profile']
    def worker(i):
        folder=OUT/f'worker_{i}';folder.mkdir()
        _run_batch(ROOT,profile,jobs[i::2],folder,lambda s:print(s,flush=True),None,
                   max_threads=4,batch_function='export_cone_video_aperture_batch')
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,range(2)))
    counts=np.zeros(len(rows));background=counts.copy();seen=np.zeros(len(rows),bool);controls=[]
    for task in tasks:
        raw=loadmat(task['folder']/'capture.mat');meta=json.loads(raw['manifest_json'].item())
        assert meta['request_signature']==task['signature']
        selected=task['rows'];assert not seen[selected].any();seen[selected]=True
        counts[selected]=raw['expected_counts'].ravel();background[selected]=raw['background_counts'].ravel()
        if meta['native_control']:controls.append(meta['native_control'])
    assert seen.all() and np.isfinite(counts).all() and np.all(background>0) and len(controls)==2
    folder=OUT/'first_frame';folder.mkdir()
    meta=dict(aperture_model=PROFILE['profile_id'],native_controls=controls,exposure_seconds=1/30,
              phototransduction_run=False,downstream_circuit_run=False,physiological_validation=False)
    savemat(folder/'capture.mat',dict(expected_counts=counts[None,:],background_counts=background[None,:],
        diameters_um=geometry['apertures_um'],window_edges_seconds=np.array([0.,1/30]),
        manifest_json=json.dumps(meta)),do_compression=True)
    rgb=loadmat(OLD/'source_rgb.mat',variable_names=['source_rgb'])['source_rgb'][:,140:500,:,0]
    np.savez_compressed(OUT/'preview_data.npz',xy_mm=geometry['xy_mm'],
        apertures_um=geometry['apertures_um'],types=geometry['types'],rgb=rgb,rates_Rstar_s=counts*30)
    result=dict(status='PASS',complete_support_cones=len(xy),captured_cones=len(rows),
        exact_six_neighbour_halo_max_abs_um=error,tiles=len(tasks),native_controls=controls,
        normal_worker_exit=True,wall_seconds=time.time()-started,
        full_video_run=False,HC_rerun=False,physiological_validation=False)
    (OUT/'VALIDATION.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
