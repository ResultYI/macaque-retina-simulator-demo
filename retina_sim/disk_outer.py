"""Freeze actual finite HC contacts and reciprocal native pooling globally."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from .contracts import ContractError,Quantity
from .atlas_hc import AtlasHCProvider
from .hc_arbor import validate_footprints,_candidate_contacts
from .disk_tables import FrozenDiskTable
from .disk_graph_storage import SparseRouteWriter,FrozenSparseRoute,save_json,file_sha

def _write_matrix(root,directory,recipe,matrix,*,unit,empty_status):
    writer=SparseRouteWriter(root,directory,recipe)
    if not writer.manifest.exists():
        for start in range(writer.next_row,matrix.shape[0],4096):
            stop=min(matrix.shape[0],start+4096);block=matrix[start:stop]
            sums=np.asarray(block.sum(1)).ravel();degree=np.diff(block.indptr)
            rows=[{'actual_degree':int(d),'missing_degree':0,'sum_weights_after_normalization':float(value),
                'unit':unit,'boundary_policy':'one finite global canonical network'} for d,value in zip(degree,sums)]
            writer.append(start,block,rows,row_status=['READY_GLOBAL_NATIVE_POOL' if d else empty_status for d in degree])
    return writer.freeze()

def freeze_outer_contacts(root,outer,directory,*,owner_radius=3.):
    root=Path(root).resolve();directory=Path(directory);cone=FrozenDiskTable(root,outer['cones']);hc=FrozenDiskTable(root,outer['HC'])
    provider=AtlasHCProvider(root);positions=cone.coordinates['opl_pedicle'];anchors=hc.coordinates['soma'];tree=cKDTree(positions)
    cone_ids=tuple(gid for start in range(0,cone.count,8192) for gid in cone.ids(start,min(cone.count,start+8192)))
    types=cone.extras['subtype'];eligible={'H1':np.asarray(types!=b'S'),'H2':np.ones(cone.count,dtype=bool)}
    recipe={'route_id':'cones_HC_raw_geometric_contacts','pre_population':'cones','post_population':'HC',
        'pre_table':cone.binding,'post_table':hc.binding,'pre_count':cone.count,'post_count':hc.count,
        'footprint_profile_sha256':provider.profile_sha256,'H1':'exclude S','H2':'all typed cones',
        'geometry':'unchanged per-ID candidate footprints with approved inward extrapolation',
        'weight':'one per actual eligible finite candidate; no nearest fill or expanded arbor',
        'boundary':'one finite canonical HC cohort; no exterior feedback'}
    writer=SparseRouteWriter(root,directory/'raw_cone_HC',recipe)
    if not writer.manifest.exists():
        for start in range(writer.next_row,hc.count,4096):
            stop=min(hc.count,start+4096);ids=hc.ids(start,stop)
            subtypes=tuple(value.decode('ascii') for value in hc.extras['subtype'][start:stop])
            points=Quantity(anchors[start:stop],'mm','arbor_anchor_position','HC_soma_to_OPL_colocation_project_v1','OPL')
            report=provider.footprints(ids,subtypes,points,mode='connection_preview',enable_central_candidate=True)
            if report['blocked_records']:raise ContractError('Approved HC footprint unsupported for actual fixed cohort: '+str(report['blocked_records'][:2]))
            configured=validate_footprints(report['footprints'],require_configured=False);columns=[];rows=[];audits=[]
            for j,(gid,subtype) in enumerate(zip(ids,subtypes)):
                footprint=configured['by_hc_id'].get(gid,configured[subtype])
                if footprint is None:raise ContractError('Actual HC has no approved finite footprint: '+gid)
                indices,distances,_,_=_candidate_contacts(tree,positions,anchors[start+j],footprint,cone_ids,None,eligible[subtype])
                columns.extend(indices.tolist());rows.extend([j]*len(indices))
                audits.append({'actual_degree':len(indices),'missing_degree':0,'HC_type':subtype,
                    'arbor_radius_mm':footprint.get('radius_mm'),'finite_envelope_mm':footprint['envelope_mm'],
                    'maximum_distance_mm':float(distances.max()) if len(distances) else None,
                    'empty_meaning':'complete finite selected model input; not biological absence'})
            matrix=sparse.csr_matrix((np.ones(len(columns)),(rows,columns)),shape=(stop-start,cone.count))
            writer.append(start,matrix,audits,row_status=['READY_FINITE_GEOMETRIC_CONTACTS' if row['actual_degree'] else
                'COMPLETE_EMPTY_DIRECT_INPUT' for row in audits])
            save_json(root/directory/'construction_status.json',{'stage':'HC_CONTACTS','completed_HC_rows':stop,'HC_count':hc.count,'edges':writer.state['edges']})
    raw_binding=writer.freeze();reader=FrozenSparseRoute(root,raw_binding)
    raw=sparse.vstack([reader.block(block)[0] for block in reader.record['blocks']],format='csr')
    hsum=np.asarray(raw.sum(1)).ravel();csum=np.asarray(raw.sum(0)).ravel()
    inv_h=np.zeros_like(hsum);inv_h[hsum>0]=1/hsum[hsum>0]
    inv_c=np.zeros_like(csum);inv_c[csum>0]=1/csum[csum>0]
    A=(sparse.diags(inv_h)@raw).tocsr();B=(sparse.diags(inv_c)@raw.T).tocsr()
    base={'raw_contacts':raw_binding,'normalization':'once on complete finite global eligible partners; no ROI renormalization',
        'physiological_validation':False,'native_kinetics':'existing shared H1 project route, not class-specific measured calibration'}
    a_recipe=dict(base,route_id='native_cone_to_HC_A',pre_population='cones',post_population='HC',
        pre_table=cone.binding,post_table=hc.binding,pre_count=cone.count,post_count=hc.count)
    b_recipe=dict(base,route_id='native_HC_to_cone_B',pre_population='HC',post_population='cones',
        pre_table=hc.binding,post_table=cone.binding,pre_count=hc.count,post_count=cone.count)
    a_binding=_write_matrix(root,directory/'native_A',a_recipe,A,unit='dimensionless',empty_status='NATIVE_DIRECT_HC_INPUT_UNDEFINED_IF_SELECTED')
    b_binding=_write_matrix(root,directory/'native_B',b_recipe,B,unit='dimensionless',empty_status='NATIVE_UNIT_DC_FEEDBACK_UNSUPPORTED_IF_SELECTED')
    # These labels accelerate exact closure; no additional recurrent edge is made.
    adjacency=sparse.bmat([[None,raw],[raw.T,None]],format='csr')
    number,labels=connected_components(adjacency,directed=False,return_labels=True)
    labels_path=root/directory/'native_component_labels.npy'
    if labels_path.exists():
        if not np.array_equal(np.load(labels_path,allow_pickle=False),labels):raise ContractError('Frozen native closure component labels differ')
    else:np.save(labels_path,labels,allow_pickle=False)
    owner=np.sum(cone.coordinates['rf']**2,axis=1)<=owner_radius**2;zero=np.flatnonzero(owner&(csum==0))
    zeros_path=root/directory/'owner_cones_without_HC.npy'
    if zeros_path.exists():
        if not np.array_equal(np.load(zeros_path,allow_pickle=False),zero):raise ContractError('Frozen zero-partner audit differs')
    else:np.save(zeros_path,zero,allow_pickle=False)
    result={'raw_contacts':raw_binding,'native_A':a_binding,'native_B':b_binding,'edges':raw.nnz,
        'native_component_labels':{'path':labels_path.relative_to(root).as_posix(),'sha256':file_sha(labels_path),
            'ordering':'all HC rows, then all cone rows','components':int(number),'largest_node_count':int(np.bincount(labels).max())},
        'owner_zero_feedback_cones':len(zero),'empty_HC_rows':int(np.count_nonzero(hsum==0)),
        'zero_partner_cone_rows':{'path':zeros_path.relative_to(root).as_posix(),'sha256':file_sha(zeros_path)},
        'zero_partner_policy':'retained explicit unsupported native unit-DC rows; no invented feedback, gain or partner',
        'status':'GLOBAL_CONTACTS_FROZEN_NATIVE_OWNER_GAPS_REQUIRE_SCIENTIFIC_DECISION' if len(zero) else
            'GLOBAL_OUTER_CONTACTS_FROZEN_PENDING_INNER_ASSEMBLY',
        'whole_exterior_recurrent_closure':'NOT_CERTIFIED; exact dependencies only inside one supplied finite global graph',
        'physiological_validation':False,'old_default_changed':False}
    final_path=root/directory/'manifest.json'
    if final_path.exists():
        if json.loads(final_path.read_bytes())!=result:raise ContractError('Frozen global outer contact summary differs; retained')
    else:save_json(final_path,result)
    return result
