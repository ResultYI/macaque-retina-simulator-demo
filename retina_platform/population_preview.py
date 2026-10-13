"""View the exact production population artifact without generating any cells."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
from retina_sim.atlas_population_network import load_population_record
from .atlas_connections_preview import _panel
from .plotting import scatter_plot,font


def export_population_preview(root,destination,owner_bounds_mm=None,*,halo_mm=.04,max_cells=60000):
    record,binding=load_population_record(root,owner_bounds_mm=owner_bounds_mm,halo_mm=halo_mm)
    tables=record['inner_tables'];rod=record['rod_inner']['tables']
    counts={k:len(v['global_cell_ids']) for k,v in {**tables,**rod}.items()}
    counts['cones']=len(record['cones']['global_cell_ids']);counts['HC']=len(record['HC']['table']['global_cell_ids'])
    if sum(counts.values())>max_cells:raise ValueError('Frozen artifact exceeds preview cell budget; no density reduction')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    # Derive view counts from actual tables. The frozen historical summary is not edited.
    summary={'population_counts':counts,'HC_candidate_edges':len(record['HC_candidate_edges']),
             'inner_weighted_edges':sum(len(c['edges']) for c in record['inner_contacts'].values())}
    manifest={'schema_version':'1.0','profile':'atlas_population_revision_v1','geometry_only':True,
              'artifact_binding':binding,'owner_domain':record['atlas']['owner_domain'],
              'query_support':record['atlas']['query_support'],'summary':summary,
              'runtime_connections':'rod/RBC/AII physical assembly recorded by Run manifest',
              'scientific_validation':False,'population_regenerated':False}
    (destination/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    panels=[_panel(record)]
    xy=np.concatenate([np.asarray(tables[k]['coordinates']['rf']['values']) for k in ('parasol_RGC_ON','parasol_RGC_OFF')])
    polarity=np.r_[np.ones(counts['parasol_RGC_ON']),np.zeros(counts['parasol_RGC_OFF'])]
    panels.append(scatter_plot(xy,polarity,np.arange(len(xy)),(600,400),coordinate='RF',value_unit='ON1 OFF0',
        fixed_range=(0,1),title='独立 parasol RF master 裁切 · ON / OFF')[0])
    xy=np.asarray(record['rod_inner']['typed_points']['rods']['OPL_spherule']['values'])
    panels.append(scatter_plot(xy,np.ones(len(xy)),np.arange(len(xy)),(600,400),coordinate='OPL',value_unit='rod',
        fixed_range=(0,1),title=f'rod OPL · {len(xy)} · 无锥 Henle 位移')[0])
    xy=np.asarray(rod['AII_AC']['coordinates']['ipl_post_arbor']['values'])
    panels.append(scatter_plot(xy,np.ones(len(xy)),np.arange(len(xy)),(600,400),coordinate='IPL',value_unit='AII',
        fixed_range=(0,1),title=f'AII IPL master · {len(xy)} · 有限开放边界')[0])
    image=Image.new('RGB',(1200,870),'#f8fafc');draw=ImageDraw.Draw(image)
    for i,p in enumerate(panels):image.paste(p,((i%2)*600,32+(i//2)*400))
    draw.text((16,7),'当前冻结人口 / 连接 · GUI / CLI / Run 共用同一 artifact',font=font(15),fill='#243447')
    draw.text((16,840),f"cone {counts['cones']} · HC {counts['HC']} · RBC {counts['rod_BC']} · HC 边 {summary['HC_candidate_edges']} · 内层边 {summary['inner_weighted_edges']}",font=font(13),fill='#64748b')
    image.save(destination/'population.png')
    return {'path':str(destination),'image':str(destination/'population.png'),'summary':summary,'artifact_binding':binding}
