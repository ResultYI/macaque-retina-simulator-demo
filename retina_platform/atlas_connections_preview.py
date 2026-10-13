"""One connection-preview export, shared by the GUI and bounded geometry suite."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw

from retina_sim.atlas import AtlasROI
from retina_sim.atlas_connections import ConnectionAtlas
from .plotting import scatter_plot,empty_plot,font


def _roi(atlas,center,size,halo):
    x,y=center;w,h=size
    owner=(x-w/2,x+w/2,y-h/2,y+h/2)
    query=(owner[0]-halo,owner[1]+halo,owner[2]-halo,owner[3]+halo)
    return AtlasROI(owner,query,atlas.atlas.config['reference_id'])


def _panel(result,size=(600,400)):
    cones=result['cones'];points=np.asarray(cones['coordinates']['opl_pedicle']['values']).reshape((-1,2))
    if not len(points):
        if result['HC'] is not None and result['HC_reference_query'] is not None:
            hc=result['HC'];xy=np.asarray(hc['OPL_anchor']['values']).reshape((-1,2))
            im,_=scatter_plot(xy,(np.array(hc['types'])=='H2').astype(float),np.arange(len(xy)),size,
                coordinate='OPL HC',fixed_range=(0,1),value_unit='H2',title='独立 HC 物理域候选；cone RF→OPL 仍缺')
            return im
        return empty_plot(size,'联合HC master越域；RF→OPL缺段另列' if result['summary']['HC_query_status']=='HC_FIXED_MASTER_UNSUPPORTED'
            else 'RF→OPL 未支持；不作坐标替代')
    cone_ids=cones['global_cell_ids']
    types=result['cone_subtypes'];unknown=np.array(types)=='UNKNOWN'
    values=unknown.astype(float) if np.any(unknown) else np.array([{'L':0,'M':1,'S':2}[t] for t in types])
    image,locations=scatter_plot(points,values,np.arange(len(points)),size,
        coordinate='OPL',fixed_range=(0,1) if np.any(unknown) else (0,2),value_unit='unknown' if np.any(unknown) else 'L0 M1 S2',title='cone pedicle；联合HC master越域' if result['HC'] is None
            else 'cone pedicle / joint HC finite candidates')
    draw=ImageDraw.Draw(image)
    # Use the exact transform of the reused scatter renderer for HC and edges.
    mins,maxs=points.min(0),points.max(0);span=np.maximum(maxs-mins,1e-6)
    w,h=image.size;left,top,right,bottom=64,43,w-107,h-54
    scale=min((right-left)/span[0],(bottom-top)/span[1]);mid=(mins+maxs)/2
    def pixel(xy):return (np.asarray(xy)-mid)*np.array([scale,-scale])+np.array([(left+right)/2,(top+bottom)/2])
    hc=result['HC']
    if hc is not None:
        hxy=hc['OPL_anchor']['values'];hids=hc['table']['global_cell_ids']
        lookup_h={g:pixel(p) for g,p in zip(hids,hxy)}
        lookup_c={g:pixel(p) for g,p in zip(cone_ids,points)}
        # Limit rendering only, with deterministic edge order; exported graph
        # includes every edge. No density, contact or weight is changed.
        edges=sorted(result['HC_candidate_edges'],key=lambda x:x['edge_id'])
        for e in edges[:600]:
            a,b=lookup_h[e['hc_id']],lookup_c[e['cone_id']]
            if all(left<=p[0]<=right and top<=p[1]<=bottom for p in (a,b)):
                draw.line([tuple(a),tuple(b)],fill='#d2dae3' if e['eligibility']=='conditional_subtype_unknown' else '#90b5bc',width=1)
        for xy,t in zip(hxy,hc['types']):
            p=pixel(xy)
            if left<=p[0]<=right and top<=p[1]<=bottom:
                color='#216fbd' if t=='H1' else '#e05a45'
                draw.rectangle((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill=color)
    return image


def export_connection_previews(root,destination,cases,*,max_cells=60000,enable_central_candidate=True,
                               atlas_profile_id='retina_atlas_v1',s_partition_profile='nemestrina_strict_pending_v1'):
    atlas=ConnectionAtlas(root,atlas_profile_id=atlas_profile_id,s_partition_profile=s_partition_profile)
    results=[]
    for case in cases:
        roi=_roi(atlas,case['center_mm'],case.get('size_mm',(.02,.02)),case.get('halo_mm',.04))
        phase='phase2' if max(np.linalg.norm([x,y]) for x in roi.owner_bounds_mm[:2] for y in roi.owner_bounds_mm[2:])>3 else 'phase1'
        result=atlas.preview(roi,phase=phase,max_cells=max_cells,
            enable_central_candidate=enable_central_candidate,hc_reference_bounds_mm=case.get('HC_reference_bounds_OPL_mm'))
        result['case']=case;results.append(result)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    (destination/'connections.json').write_text(json.dumps(results,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    manifest={'profile':'atlas_connection_preview_joint_v2','geometry_only':True,'dynamics':'NOT_RUN',
        'HC_mosaic':'approved joint soft-core fixed canonical master then crop; no Poisson fallback',
        'HC_master_scope':atlas.hc_master.config['scope'],
        'atlas_profile_id':atlas_profile_id,'S_partition_profile':s_partition_profile,
        'connections_sha256':hashlib.sha256((destination/'connections.json').read_bytes()).hexdigest(),
        'implementation_sha256':{name:hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() for name in
            ('retina_sim/atlas_connections.py','retina_sim/atlas_hc.py','retina_sim/atlas_hc_master.py','retina_sim/atlas_inner.py',
             'retina_sim/atlas_rod_inner.py','retina_platform/atlas_connections_preview.py')},
        'HC_gap':'off; unit-safe interfaces separately tested; no positive defaults',
        'central_candidate_explicit':enable_central_candidate,'cases':[{'case':r['case'],'summary':r['summary'],
            'closure_blockers':r['closure_blockers']} for r in results],
        'scientific_validation':False,'same_atlas_across_cases':True,'graph_rendering':'first600stable-sortededges/panel only; fullgraph export unchanged'}
    (destination/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    columns=2;rows=(len(results)+1)//2
    image=Image.new('RGB',(1200,rows*460),'#f8fafc');draw=ImageDraw.Draw(image)
    for i,r in enumerate(results):
        col,row=i%columns,i//columns;x,y=col*600,row*460
        image.paste(_panel(r),(x,y+32))
        s=r['summary']
        draw.text((x+16,y+8),r['case']['label'],font=font(14),fill='#243447')
        hc_text='HC master越域' if s['HC_query_status']=='HC_FIXED_MASTER_UNSUPPORTED' else f"联合HC候选边 {s['HC_candidate_edges']}"
        draw.text((x+16,y+430),f"owner {s['owner_cones']} · OPL {s['mapped_owner_cones']} · {hc_text} · inner权重边 {s['inner_weighted_edges']}",font=font(11),fill='#64748b')
        rod=s['rod_inner_population_counts']
        draw.text((x+16,y+445),f"rod {rod.get('rods','缺来源')} · RBC {rod.get('rod_BC','缺来源')} · AII {rod.get('AII_AC',0)} · rod内层边 {s['rod_inner_weighted_edges']}",font=font(11),fill='#64748b')
    image.save(destination/'connections.png')
    return {'path':str(destination),'image':str(destination/'connections.png'),
        'case_summaries':[r['summary'] for r in results],'network_run_ready':False}
