"""Approved radial69#6 candidate: source scale, geometry and frozen pooling."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,sys,time
import cv2
import numpy as np
from scipy.io import savemat
from retina_sim.contracts import canonical_hash
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.local_patch_graph import LocalPatchAssembly
from retina_sim.spatial_group_model import GroupModelAssembly,RGC
from tools.prepare_cross_position_coverage_v1 import choose_geometry,selection
from tools.cross_position_context_v1 import write,read

ROOT=Path(r'D:\macaque retinal simulator')
PARENT=ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2'
OLD=ROOT/'reports/layerwise_physiology/cross_position_functional_coverage_v1_20261029/restored_resume_v1'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def source(out):
    binding=read(ROOT/'reports/GIN_REAL_TARGET_BINDING_V1/BINDING.json')
    definition=dict(schema='gin69-6-approved-radial-conditional-v1',source_cell='69#6',source_type='M off',mapped_type='parasol_RGC_OFF',
        source_eccentricity_deg=3.88,source_azimuth_deg=None,model_reference_meridian_deg=0.,meridian_status='ASSUMED_PROJECT_POSITIVE_X',
        approval='User2026-11-01T08:13Z: radial conditional candidate with reference meridian and literature brightness approximation',
        archive_sha256=binding['source_archive_sha256'],movie_sha256=binding['movie_sha256'],movie_path=binding['movie_path'],
        source_RF_fractional_column=binding['RF_fractional_column'],source_RF_fractional_row=binding['RF_fractional_row'],
        clock=dict(first_frame_zero_based=751,playback_Hz=150,container_fps_ignored=True,recording_bins_seconds=.001),
        brightness=dict(status='ASSUMED_LITERATURE_SCALE_NOT_RECOVERED_CRT_CALIBRATION',anchor_td=222.2,pupil_diameter_mm=4.,
            calibration_frames_zero_based=[751,9750],calibration_count=9000,diameter_deg=.25,
            pixel_center=[127.5,127.5],pixel_pitch_deg=4.6/256,
            weight='cos(pi*r/diameter) for r<=diameter/2; discrete weights divided by their sum; time mean of9000frames',
            global_gain='222.2/(pi*2^2) divided by photopic luminance of the weighted gamma-corrected source-gun spectrum',
            source='vanHateren2002 p9946 central15arcmin cosine weighted flower-show CTIS; original exact segment/weights unproved',
            not_full_field_or_10min_average=True),
        source_background='Actual decoded pre-live frame0, including compression nonuniformity, captured spatially; its meanRGB defines only the fixed reference spectrum; independent steady local initialization, not a replay of full acquisition pre-roll',
        short_forward=dict(duration_seconds=.05,first8_live_source_frames=list(range(751,759)),purpose='cost and end-to-end state/clock numerical check, not fitting'),
        rod='Inherited independent engineering rod background, no calibrated CRT-to-Rod mapping',
        FIT_READY=False,background_qualified=False,production_adopted=False)
    write(out/'INPUT_APPROXIMATION_MANIFEST.json',definition)
    movie=ROOT/binding['movie_path']
    if sha(movie)!=binding['movie_sha256']:raise ValueError('Movie changed')
    axis=(np.arange(256)-127.5)*4.6/256;xx,yy=np.meshgrid(axis,axis);r=np.hypot(xx,yy)
    w=np.where(r<=.125,np.cos(np.pi*r/.25),0.);w/=w.sum();take=w>0
    accumulator=np.zeros(3);bank=[];frame_hashes={};background=None
    exponents=np.array([2.3122,2.2752,2.2818]);offset=np.array([.01451,.005123,.02612]);scale=np.array([.9855,.9949,.9739])
    cap=cv2.VideoCapture(str(movie));start=time.perf_counter()
    try:
        for i in range(9751):
            ok,bgr=cap.read()
            if not ok:raise ValueError('Source minute truncated at '+str(i))
            rgb=bgr[:,:,::-1]
            if i==0:
                background=rgb.mean(axis=(0,1))/255;bank.append(rgb.copy())
            if 751<=i<759:
                bank.append(rgb.copy());frame_hashes[str(i)]=hashlib.sha256(rgb.tobytes()).hexdigest()
            if i>=751:
                values=offset+scale*(rgb[take].astype(float)/256)**exponents
                accumulator+=w[take]@values
    finally:cap.release()
    mean=accumulator/9000
    savemat(out/'SOURCE_BANK.mat',dict(source_rgb=np.stack(bank,axis=3)),do_compression=True)
    np.savez_compressed(out/'SOURCE_SCALE.npz',spatial_weights=w,mean_gamma_RGB=mean,background_rgb=background)
    write(out/'SOURCE_SCALE.json',dict(mean_gamma_RGB=mean,background_rgb=background,source_frames=9000,
        nonzero_weight_pixels=int(take.sum()),weight_sum=float(w.sum()),frame_hashes=frame_hashes,
        preparation_wall_seconds=time.perf_counter()-start,source_bank_sha256=sha(out/'SOURCE_BANK.mat'),
        approximation_manifest_sha256=sha(out/'INPUT_APPROXIMATION_MANIFEST.json')))

def geometry(out,guard):
    dest=out/'prepared';dest.mkdir()
    g=choose_geometry(PARENT,[3.88,0.]);pop='parasol_RGC_OFF';xy=np.load(PARENT/f'populations/{pop}/centroid_mm.npy')/.223
    row=int(np.argmin(np.linalg.norm(xy-[3.88,0.],axis=1)))
    if np.linalg.norm(xy[row]-[3.88,0.])>.15:raise ValueError('No matching existing OFF-parasol core group')
    g.update(selected_type=pop,selected_parent_group_row=row,selection_rule='Nearest existing OFF-parasol centroid to approved radial center[3.88,0]; no output selection',
        target_centroid_deg=xy[row],target_centroid_ecc_deg=float(np.linalg.norm(xy[row])),source_ecc_deg=3.88)
    write(dest/'GEOMETRY_SELECTION.json',g);base=GroupModelAssembly(PARENT)
    seed=OLD/'parent_construction_seed';meta=read(seed/'manifest.json')
    if meta['identity']['artifact_sha256']!=base.artifact_sha256 or meta['identity']['seed']['neural_model_advanced']:raise ValueError('Parent construction identity changed')
    cfg=deepcopy(read(OLD/'local_static/selection_identity.json')['configuration']);cfg.pop('azimuth_deg',None)
    cfg.update(eccentricity_deg=3.88,rf_center_deg=[3.88,0.],target_rgc_type=pop,target_rgc_id=None,
        input_identity=dict(path=str((seed/'manifest.json').relative_to(ROOT)),sha256=sha(seed/'manifest.json')),
        position_provenance='Real source radial eccentricity3.88deg; approved assumed+x meridian',
        target_provenance='Nearest existing OFF-parasol group; member/centroid offset reported, not source-cell anatomy identity')
    present=[n for n in RGC if g['geometric_presence'][n]['core_centroid_groups']]
    small=selection(base,cfg,present,guard);bigcfg=deepcopy(cfg);bigcfg['patch_geometry']['support_radius_deg']=.6
    large=selection(base,bigcfg,present,guard);rows=dict(small[0])
    for n in ('cones','HC'):
        if not np.isin(rows[n],large[0][n]).all():raise ValueError('Native support shrank')
        rows[n]=large[0][n]
    arrays={**{'population__'+n:v for n,v in rows.items()},**{'contact__'+n:v for n,v in small[1].items()},**{'core__'+n:v for n,v in small[2].items()}}
    folder=dest/'local_static';folder.mkdir();np.savez_compressed(folder/'selection.npz',**arrays)
    desc=dict(schema='gin-radial-native060-inner030-v1',configuration=cfg,parent_artifact=base.artifact_sha256,
        selection_sha256=canonical_hash(arrays),native_radius=.6,inner_radius=.3,present_core_types=present,no_new_geometry=True)
    write(folder/'selection_identity.json',desc)
    local=object.__new__(LocalPatchAssembly);local._assemble(ROOT,folder,base,cfg,rows,small[1],small[2],small[3],canonical_hash(desc),desc)
    target=local.populations[pop];selected=int(np.flatnonzero(local.parent_population_rows[pop]==row)[0])
    write(dest/'STATIC_COUNTS.json',dict(counts={n:p.group_count for n,p in local.populations.items()},members={n:int(p.counts.sum()) for n,p in local.populations.items()},
        target_group_id=str(target.group_ids[selected]),target_local_row=selected,target_members=int(target.counts[selected]),
        target_centroid_deg=xy[row],core_radius_deg=.15,inner_radius_deg=.3,native_radius_deg=.6,source_parent_construction=str(seed),old_equilibrium_reused=False))
    from tools.gin_radial_pooling_v1 import build
    build(out,arrays,guard)

def main():
    nonce,folder,stage=sys.argv[1:];execution=Path(folder);out=execution.parent if stage=='structure_finish' else execution
    memory=MemoryGuard(24*1024**3,8*1024**3);start=time.perf_counter()
    def guard():
        if (out/'REQUEST_STOP').exists():raise RuntimeError('Explicit stop')
        if not memory.check():raise MemoryError(str(memory.last))
    guard()
    try:
        if stage=='prepare':source(out);geometry(out,guard)
        elif stage=='structure_finish':
            from tools.cross_position_context_v1 import patch
            _,local=patch(out/'prepared');g=read(out/'prepared/GEOMETRY_SELECTION.json');pop=g['selected_type']
            selected=int(np.flatnonzero(local.parent_population_rows[pop]==g['selected_parent_group_row'])[0]);target=local.populations[pop]
            write(out/'prepared/STATIC_COUNTS.json',dict(counts={n:p.group_count for n,p in local.populations.items()},members={n:int(p.counts.sum()) for n,p in local.populations.items()},
                target_group_id=str(target.group_ids[selected]),target_local_row=selected,target_members=int(target.counts[selected]),target_centroid_deg=g['target_centroid_deg'],
                core_radius_deg=.15,inner_radius_deg=.3,native_radius_deg=.6,source_parent_construction=str(OLD/'parent_construction_seed'),old_equilibrium_reused=False))
            with np.load(out/'prepared/local_static/selection.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            from tools.gin_radial_pooling_v1 import build
            build(out,arrays,guard)
        else:raise ValueError(stage)
    finally:write(execution/(stage+'_RESOURCE.json'),dict(wall_seconds=time.perf_counter()-start,peak_rss_bytes=memory.peak_rss_bytes,memory=memory.last))

if __name__=='__main__':main()
