"""Shared GUI/CLI source-atlas geometry preview and fixed asset export."""
from pathlib import Path
import json
import numpy as np

from retina_sim.atlas import AtlasROI
from retina_sim.atlas_store import save_atlas_asset
from retina_sim.source_atlas import SourceDensityAtlas


def export_atlas_preview(root, destination, owner_bounds_mm, *, halo_mm=.04, max_cells=60000, phase='phase1',profile_id='retina_atlas_v1'):
    from PIL import Image,ImageDraw
    from .plotting import scatter_plot,font
    if not np.isfinite(halo_mm) or halo_mm < 0:
        raise ValueError('Atlas halo must be finite nonnegative mm')
    atlas = SourceDensityAtlas(root,profile_id=profile_id)
    x0,x1,y0,y1 = owner_bounds_mm
    roi = AtlasROI(tuple(owner_bounds_mm),(x0-halo_mm,x1+halo_mm,y0-halo_mm,y1+halo_mm),
                   atlas.config['reference_id'])
    patch = atlas.query(roi, max_cells=max_cells, phase=phase)
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    source = patch.manifest['source_bundle']
    boundary = {k:patch.manifest[k] for k in ('owner_domain','query_support','boundary_mode','recurrent_closure')}
    rf = patch.selection.coordinate('rf').values
    keys = patch.selection.source_keys
    table = atlas._table('cones', keys, rf)
    files = {}
    for name, current in [('rf_cells.json',table),('mapped_cells.json',patch.mapped_table)]:
        files[name] = save_atlas_asset(destination/name,current,asset_id=atlas.config['atlas_id']+':'+name,
            asset_kind='canonical_atlas',source=source,
            provenance={'profile_sha256':atlas.profile_hash,'identity':'fixed global tiles, finite view',
                        'not_a_complete_animal_retina':True,'geometry_only':True},boundary=boundary)
    manifest = dict(patch.manifest, assets_sha256=files,
        mapping_status_counts={s:patch.mapping_status.count(s) for s in sorted(set(patch.mapping_status))},
        mapped_count=len(patch.mapped_table.source_keys),
        mapped_global_ids=patch.mapped_table.global_cell_ids)
    (destination/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    owner = patch.selection.owner
    left,_=scatter_plot(rf,owner.astype(float),np.arange(len(rf)),(600,400),coordinate='RF',
        fixed_range=(0,1),value_unit='owner',title=f'RF: {int(owner.sum())} owner / {len(rf)} query cones')
    mapped = patch.mapped_table
    opl=np.empty((0,2));mapped_owner=np.zeros(0,dtype=bool)
    if len(mapped.source_keys):
        opl = mapped.coordinate('opl_pedicle').values
        ids = set(patch.selection.global_cell_ids[i] for i in np.flatnonzero(owner))
        mapped_owner = np.array([i in ids for i in mapped.global_cell_ids])
    right,_=scatter_plot(opl,mapped_owner.astype(float),np.arange(len(opl)),(600,400),coordinate='OPL',
        fixed_range=(0,1),value_unit='owner',title=f'OPL: {len(mapped.source_keys)} supported mappings')
    composite=Image.new('RGB',(1200,460),'#f8fafc')
    composite.paste(left,(0,32));composite.paste(right,(600,32))
    draw=ImageDraw.Draw(composite)
    draw.text((16,8),'来源约束复合几何 · Packer 密度 / Schein 位移 · 未标定网络',font=font(14),fill='#243447')
    draw.text((16,436),'轴：x 鼻侧正，y 上方正 · 0=halo，1=owner · HC arbor / S 完整分区 / IPL 待补',font=font(12),fill='#64748b')
    image = destination/'geometry.png'
    composite.save(image)
    return {'path':str(destination),'image':str(image),'owner_count':int(owner.sum()),
            'query_count':len(rf),'mapped_count':len(mapped.source_keys),'network_run_ready':False}
