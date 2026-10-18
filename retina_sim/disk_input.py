"""Fixed optical tiles and one shared LMS reference for the global disk.

Tile ownership is tied to global coordinates, never to a requested neural ROI.
MW optics and finite optical halos retain their existing approximation status.
"""
from pathlib import Path
import json
import numpy as np
from .contracts import ContractError,canonical_hash
from .disk_graph_storage import save_json,file_sha,array_identity

TILE_MM=.512
HALO_MM=.04
PITCH_MM=.001

def optical_tiles(positions,*,tile_mm=TILE_MM,halo_mm=HALO_MM,pitch_mm=PITCH_MM):
    xy=np.asarray(positions,dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.all(np.isfinite(xy)):raise ContractError('Finite global Nx2 cone coordinates required')
    if (tile_mm,halo_mm,pitch_mm)!=(TILE_MM,HALO_MM,PITCH_MM):raise ContractError('Explicit fixed optical sampling contract required')
    if not len(xy):return []
    ij=np.floor(xy/tile_mm).astype(np.int64);pairs=np.unique(ij,axis=0);result=[]
    for tx,ty in pairs:
        indices=np.flatnonzero(np.all(ij==[tx,ty],axis=1));center=(np.array([tx,ty])+.5)*tile_mm
        extent=tile_mm+2*halo_mm;samples=int(round(extent/pitch_mm))+1
        if samples>1024:raise ContractError('Fixed optical tile exceeds original scene budget')
        result.append({'tile_index':[int(tx),int(ty)],'indices':indices,'center_mm':center,
            'owner_bounds_mm':[tx*tile_mm,(tx+1)*tile_mm,ty*tile_mm,(ty+1)*tile_mm],
            'scene_samples':samples,'pitch_mm':pitch_mm,'halo_mm':halo_mm,
            'finite_support_claim':'fixed canonical finite optical halo; no infinite-PSF closure claim'})
    return result

def freeze_input_plan(root,cone_table,directory):
    root=Path(root).resolve();directory=(root/directory).resolve()
    if not directory.is_relative_to(root):raise ContractError('Input plan outside project')
    directory.mkdir(parents=True,exist_ok=True);path=directory/'manifest.json'
    xy=cone_table.coordinates['rf'];diameter=cone_table.extras['aperture_diameter_um']
    if diameter.shape!=(cone_table.count,) or np.any(diameter<=0) or not np.all(np.isfinite(diameter)):
        raise ContractError('Frozen global cone apertures missing')
    reference={'mode':'one_global_foveal_LMS_reference_v1','reference_position_mm':[0.,0.],
        'reference_diameter_um':float(np.mean(diameter)),'blur_um':float(np.mean(diameter)),
        'reference_illuminance_td':100.,'cone_table_binding':cone_table.binding,
        'global_apertures':array_identity(diameter),'coefficients':'one fixed LMS triplet and one radiance gain for every tile/ROI',
        'physiological_calibration':False,
        'cone_geometry_route':cone_table.record.get('source_bindings',{}).get('cone_geometry_route','legacy_macaque_geometry'),
        'scope':('approved human-MW/LCD/official-human-borrowed-cone mixed reference'
            if cone_table.record.get('source_bindings',{}).get('density_species')=='Homo sapiens'
            else 'approved human-MW/LCD/macaque-geometry mixed reference')}
    reference['definition_sha256']=canonical_hash(reference)
    records=[]
    for tile in optical_tiles(xy):
        tx,ty=tile['tile_index'];target=directory/f'tile_{tx}_{ty}_global_rows.npy'
        if target.exists():
            if not np.array_equal(np.load(target,allow_pickle=False),tile['indices']):raise ContractError('Frozen optical tile rows differ')
        else:np.save(target,tile['indices'],allow_pickle=False)
        records.append({key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in tile.items() if key!='indices'}|
            {'global_rows':{'path':target.relative_to(root).as_posix(),'sha256':file_sha(target)}})
    value={'schema':'fixed_global_disk_input_plan_v1','cone_table':cone_table.binding,'reference':reference,'tiles':records,
        'optical_ownership':'global half-open tile indices; same cone always uses same optical tile',
        'phototransduction':'Python native mechanism','photon_noise':'none','input_stage':'expected R*/window then fixed LMS reference-equivalent drive',
        'density_or_gain_fit_per_ROI':False,'regenerate_on_ROI':False,'status':'FROZEN_PLAN_REQUIRES_ISET_CAPTURE',
        'exporter_sha256':file_sha(root/'tools/export_disk_mixed_reference.m')}
    if path.exists():
        if json.loads(path.read_bytes())!=value:raise ContractError('Frozen input plan changed; retained')
    else:save_json(path,value)
    return {'path':path.relative_to(root).as_posix(),'sha256':file_sha(path)}

def join_captures(tile_rows,tile_captures,count,*,reference_identity):
    """Validate and join a complete capture without any tile-wise rescaling."""
    if len(tile_rows)!=len(tile_captures):raise ContractError('Optical tile/capture count differs')
    seen=np.zeros(count,dtype=bool);expected=background=diameters=None;fixed_q=None;fixed_gain=None;edges=None
    for indices,capture in zip(tile_rows,tile_captures):
        rows=np.asarray(indices,dtype=np.int64);raw,meta=capture
        if np.any(rows<0) or np.any(rows>=count) or len(np.unique(rows))!=len(rows) or np.any(seen[rows]):
            raise ContractError('Optical tiles duplicate or exceed global cone rows')
        if meta.get('global_reference_definition_sha256')!=reference_identity:raise ContractError('Tile uses a different global reference')
        q=np.asarray(raw['q_ref_LMS']).reshape(-1);gain=float(meta['fixed_radiance_gain']);ticks=np.asarray(raw['window_edges_seconds']).reshape(-1)
        if fixed_q is None:
            fixed_q=q;fixed_gain=gain;edges=ticks
            expected=np.empty((len(ticks)-1,count));background=np.empty(count);diameters=np.empty(count)
        if not np.array_equal(q,fixed_q) or gain!=fixed_gain or not np.array_equal(edges,ticks):
            raise ContractError('Tile altered fixed LMS coefficients, radiance gain or clock')
        values=np.asarray(raw['expected_counts']);bg=np.asarray(raw['background_counts']).reshape(-1);apertures=np.asarray(raw['diameters_um']).reshape(-1)
        if values.shape!=(len(ticks)-1,len(rows)) or bg.shape!=(len(rows),) or apertures.shape!=(len(rows),):
            raise ContractError('Tile capture cell/time layout differs')
        if not np.all(np.isfinite(values)) or np.any(values<0) or not np.all(np.isfinite(bg)) or np.any(bg<=0):
            raise ContractError('Invalid global expected/background captures')
        expected[:,rows]=values;background[rows]=bg;diameters[rows]=apertures;seen[rows]=True
    if not np.all(seen):raise ContractError('Global optical capture has missing cone rows; no crop fill')
    return {'expected_counts':expected,'background_counts':background,'diameters_um':diameters,
        'q_ref_LMS':fixed_q,'fixed_radiance_gain':fixed_gain,'window_edges_seconds':edges}
