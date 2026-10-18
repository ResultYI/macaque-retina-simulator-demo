"""Read actual frozen central AII chemical and electrical support; no simulation."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.disk_circuit import GlobalDiskCircuit


def main():
    output=ROOT/'reports/central_AII_actual_support_20261018.json'
    if output.exists():raise ValueError('Retain the existing diagnostic')
    config=json.loads((ROOT/'configurations/atlas_human_borrowed_disk3mm_functional_hc_v1.json').read_bytes())
    circuit=GlobalDiskCircuit(ROOT,config['global_graph']);begin=time.perf_counter()
    view=circuit.owner_view((-.01,.01,-.01,.01));selected=view['selected_rows']
    desc=next(r for r in circuit.record['routes'] if r['route_id']=='RBC_AII_AMPA')
    post,matrix,audits,statuses=circuit.route(desc).selected_rows(selected['AII_AC'])
    empty=post[np.diff(matrix.indptr)==0];electrical=np.zeros(len(empty),dtype=np.int64);routes=[]
    for descriptor in circuit.record['routes']:
        if descriptor['kind']!='gap' or 'AII_AC' not in (descriptor['pre'],descriptor['post']):continue
        pre,post_name=descriptor['pre'],descriptor['post']
        a=selected.get(pre,np.empty(0,dtype=np.int64));b=selected.get(post_name,np.empty(0,dtype=np.int64))
        _,_,m,_,_=circuit.route(descriptor).view(b,a)
        incident=np.zeros(len(empty),dtype=np.int64)
        if post_name=='AII_AC':
            lookup={int(row):i for i,row in enumerate(b)}
            incident+=np.asarray([m.indptr[lookup[int(row)]+1]-m.indptr[lookup[int(row)]] for row in empty])
        if pre=='AII_AC':
            counts=np.bincount(m.indices,minlength=len(a));lookup={int(row):i for i,row in enumerate(a)}
            incident+=np.asarray([counts[lookup[int(row)]] for row in empty])
        electrical+=incident
        routes.append({'route':descriptor['route_id'],'selected_edges':m.nnz,
            'chemically_empty_AII_with_incident_gap':int(np.count_nonzero(incident)),
            'incidences_count_both_directions_if_present':int(incident.sum())})
    xy=circuit.table('AII_AC').positions[empty]
    result={'schema':'frozen_primary_support_diagnostic_v1','global_graph':config['global_graph'],
        'owner_RF_bounds_mm':view['owner_bounds_mm'],'runtime_ready_reported_by_old_route_filter':view['runtime_ready'],
        'old_unsupported_rows_count':len(view['unsupported_rows']),
        'selected_global_populations':{key:len(rows) for key,rows in selected.items()},
        'selected_AII_count':len(selected['AII_AC']),'selected_RBC_AII_edges':matrix.nnz,
        'AII_without_any_RBC_chemical_edge':len(empty),'empty_AII_global_rows':empty.tolist(),
        'empty_AII_positions_mm':xy.tolist(),'empty_AII_radii_mm':np.linalg.norm(xy,axis=1).tolist(),
        'empty_AII_with_actual_gap':int(np.count_nonzero(electrical)),
        'empty_AII_without_actual_gap':int(np.count_nonzero(electrical==0)),'gap_routes':routes,
        'weights_or_cells_changed':False,'chemical_zero_drive_policy_added':False,
        'physiological_validation':False,'wall_seconds':time.perf_counter()-begin,
        'interpretation':'Existing preflight omits primary_chemical routes; assembly still rejects empty chemical rows. Gap incidences alone do not establish calibration.'}
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('selected_AII_count','selected_RBC_AII_edges',
        'AII_without_any_RBC_chemical_edge','empty_AII_with_actual_gap','empty_AII_without_actual_gap','gap_routes','wall_seconds')},indent=2),flush=True)

if __name__=='__main__':main()
