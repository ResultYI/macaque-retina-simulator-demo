"""Named HC contact projections; retain the historical nearest-count alternative."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from .contracts import ContractError
from .disk_graph_storage import SparseRouteWriter,save_json,file_sha,array_identity
from .disk_tables import FrozenDiskTable
from .disk_outer import _write_matrix
from .atlas_hc import AtlasHCProvider
from .contracts import Quantity


def eccentric_projection(cone_opl, cone_types, hc_opl, hc_types, owner,
                         cohort_radius, hc_radii_mm):
    """Select actual pedicles inside each HC's finite OPL equal-area disk.

    Counts are outputs. Empty rows remain empty and no nearest partner is added.
    The supplied cohort certifies owner-cone neighborhoods independently of ROI.
    """
    xy = np.asarray(cone_opl, dtype=float)
    hxy = np.asarray(hc_opl, dtype=float)
    labels, ht = np.asarray(cone_types), np.asarray(hc_types)
    owner = np.asarray(owner)
    radii = np.asarray(hc_radii_mm, dtype=float)
    if (xy.ndim != 2 or xy.shape[1] != 2 or hxy.ndim != 2 or hxy.shape[1] != 2
            or not np.isfinite(xy).all() or not np.isfinite(hxy).all()):
        raise ContractError('Finite actual cone/HC Nx2 OPL anchors required')
    if (labels.shape != (len(xy),) or ht.shape != (len(hxy),)
            or owner.shape != (len(xy),) or owner.dtype.kind != 'b'
            or radii.shape != (len(hxy),) or not np.isfinite(radii).all()
            or np.any(radii <= 0) or not len(hxy)):
        raise ContractError('Eccentric HC radii, types and owner layout differ')
    if not set(labels) <= {'L', 'M', 'S'} or not set(ht) <= {'H1', 'H2'}:
        raise ContractError('Actual resolved cone and HC types required')
    if (not np.isfinite(cohort_radius) or cohort_radius <= 0
            or np.any(np.linalg.norm(hxy, axis=1) > cohort_radius + 1e-12)):
        raise ContractError('Explicit complete canonical HC cohort radius required')
    if np.any(np.linalg.norm(xy[owner], axis=1) + radii.max() > cohort_radius + 1e-12):
        raise ContractError('Eccentric owner neighborhood exceeds certified canonical HC support')
    tree = cKDTree(xy)
    rows, columns = [], []
    maximum = np.zeros(len(xy))
    max_ratio = 0.
    for start in range(0, len(hxy), 4096):
        stop = min(start + 4096, len(hxy))
        neighbors = tree.query_ball_point(hxy[start:stop], radii[start:stop] + 1e-12,
                                          workers=1, return_sorted=True)
        for j, candidates in enumerate(neighbors, start):
            indices = np.asarray(candidates, dtype=np.int64)
            if ht[j] == 'H1':
                indices = indices[labels[indices] != 'S']
            distances = np.linalg.norm(xy[indices] - hxy[j], axis=1)
            if len(indices):
                max_ratio = max(max_ratio, float(np.max(distances / radii[j])))
                np.maximum.at(maximum, indices, distances)
            rows.append(indices)
            columns.append(np.full(len(indices), j, dtype=np.int64))
    row, column = np.concatenate(rows), np.concatenate(columns)
    raw = sparse.csr_matrix((np.ones(len(row)), (row, column)), shape=(len(xy), len(hxy)))
    audit = {'subtypes': {}, 'owner_complete': True,
             'maximum_owner_functional_distance_mm': float(maximum[owner].max(initial=0)),
             '_per_cone_support_mm': maximum, '_hc_radii_mm': radii,
             'maximum_distance_over_HC_radius': max_ratio,
             'cone_count': len(xy), 'HC_count': len(hxy), 'edges': raw.nnz,
             'selection': 'all eligible actual pedicles inside each HC OPL disk',
             'integer_divergence_imposed': False, 'outside_radius_fallback': False,
             'empty_rows_retained': True, 'anatomical_contact_claim': False,
             'biological_zero_innervation_claim': False, 'physiological_validation': False}
    for kind in ('H1', 'H2'):
        selected = np.flatnonzero(ht == kind)
        degree = np.asarray(raw[:, selected].sum(axis=1)).ravel()
        convergence = np.asarray(raw[:, selected].sum(axis=0)).ravel()
        audit['subtypes'][kind] = {
            'HC_count': len(selected), 'actual_associations': int(degree.sum()),
            'mean_convergence_all_cohort': float(convergence.mean()) if len(selected) else None,
            'owner_mean_divergence_all_cones': float(degree[owner].mean()) if owner.any() else None,
            'radius_um_quantiles': np.percentile(radii[selected]*1000, [0, 50, 95, 100]).tolist() if len(selected) else []}
    return raw, audit


def functional_projection(cone_opl,cone_types,hc_opl,hc_types,owner,cohort_radius,anchors):
    """Finite canonical typed neighborhood, not a biological reach estimate.

    The support is the actual kth-anchor ball. Every owner ball must be wholly
    inside the supplied complete HC cohort; nearest points beyond missing tissue
    are never used as an exterior completion. Halo closure is explicitly finite.
    """
    xy=np.asarray(cone_opl,dtype=float);hxy=np.asarray(hc_opl,dtype=float)
    labels=np.asarray(cone_types);ht=np.asarray(hc_types);owner=np.asarray(owner,dtype=bool)
    if xy.ndim!=2 or xy.shape[1]!=2 or hxy.ndim!=2 or hxy.shape[1]!=2 or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(hxy)):
        raise ContractError('Finite actual cone/HC Nx2 OPL reference anchors required')
    if len(labels)!=len(xy) or len(ht)!=len(hxy) or owner.shape!=(len(xy),):raise ContractError('Functional projection identity layout differs')
    if not set(labels)<=set(('L','M','S')) or not set(ht)<=set(('H1','H2')):raise ContractError('Actual resolved cone and HC types required')
    if not np.isfinite(cohort_radius) or cohort_radius<=0 or np.any(np.linalg.norm(hxy,axis=1)>cohort_radius+1e-12):
        raise ContractError('Explicit complete canonical HC cohort radius required')
    low,high=anchors['project_representative_radii_mm'];r=np.linalg.norm(xy,axis=1)
    row=[];column=[];audit={};largest=np.zeros(len(xy));owner_incomplete=[]
    for kind in ('H1','H2'):
        eligible=np.flatnonzero(labels!='S') if kind=='H1' else np.arange(len(xy))
        source=np.flatnonzero(ht==kind);central,peripheral=anchors[kind]
        mean=central+(peripheral-central)*np.clip((r[eligible]-low)/(high-low),0,1)
        count=np.floor(mean).astype(int);extra=int(np.floor(mean.sum()+.5)-count.sum())
        count[np.argsort(-(mean-count),kind='stable')[:extra]]+=1
        maximum=int(count.max(initial=0))
        if maximum<1 or len(source)<maximum:raise ContractError('Insufficient actual typed HC anchors; no synthetic partners')
        tree=cKDTree(hxy[source]);distance,index=tree.query(xy[eligible],k=maximum,workers=1)
        distance=np.asarray(distance).reshape(len(eligible),maximum);index=np.asarray(index).reshape(len(eligible),maximum)
        local_support=distance[np.arange(len(eligible)),count-1]
        complete=local_support<=cohort_radius-r[eligible]+1e-12
        owner_incomplete.extend(eligible[owner[eligible]&~complete].tolist())
        keep=np.arange(maximum)[None,:]<count[:,None]
        row.append(np.broadcast_to(eligible[:,None],keep.shape)[keep]);column.append(source[index[keep]])
        largest[eligible]=np.maximum(largest[eligible],local_support)
        audit[kind]={'eligible_cones':len(eligible),'actual_associations':int(count.sum()),
            'continuous_mean_target_sum':float(mean.sum()),'integer_count_residual':float(count.sum()-mean.sum()),
            'mean_targets':array_identity(mean),'integer_counts':array_identity(count),
            'actual_support_mm':array_identity(local_support),
            'support_um_quantiles':np.percentile(local_support*1000,[0,50,95,100]).tolist(),
            'owner_ball_support_complete':bool(np.all(complete[owner[eligible]])),
            'halo_ball_support_uncertified_count':int(np.sum(~complete&~owner[eligible]))}
    if owner_incomplete:raise ContractError('Functional owner neighborhood exceeds certified canonical HC support: '+str(owner_incomplete[:10]))
    row=np.concatenate(row);column=np.concatenate(column)
    raw=sparse.csr_matrix((np.ones(len(row)),(row,column)),shape=(len(xy),len(hxy)))
    if np.any(np.diff(raw.indptr)==0):raise ContractError('Unresolved actual functional cone projection; no zero-row repair')
    return raw,{'subtypes':audit,'owner_complete':True,'maximum_owner_functional_distance_mm':float(largest[owner].max(initial=0)),
        '_per_cone_support_mm':largest,
        'cone_count':len(xy),'HC_count':len(hxy),'edges':raw.nnz,
        'anatomical_contact_claim':False,'biological_zero_innervation_claim':False,'physiological_validation':False}


def freeze_functional_hc(root,outer,anatomical_reference,directory,profile_path='configurations/hc_eccentric_size_w2000_v1.json'):
    root=Path(root).resolve();directory=Path(directory);profile_file=root/profile_path;profile=json.loads(profile_file.read_bytes())
    sized = profile['schema'] == 'HC_eccentric_equal_area_connection_profile_v1'
    if not sized and profile['schema']!='HC_effective_functional_projection_profile_v1':raise ContractError('Named HC connection profile required')
    source=profile['primary_source']
    if file_sha(root/source['path'])!=source['sha256']:raise ContractError('Frozen W2000 primary identity changed')
    cone=FrozenDiskTable(root,outer['cones']);hc=FrozenDiskTable(root,outer['HC'])
    owner=np.sum(cone.coordinates['rf']**2,axis=1)<=9
    args = (cone.coordinates['opl_pedicle'],cone.extras['subtype'].astype('U1'),
        hc.coordinates['soma'],hc.extras['subtype'].astype('U2'),owner,hc.record['source_bindings']['fixed_cohort_OPL_radius_mm'])
    representation = 'effective_functional_association_not_anatomical_synapse'
    size_binding = None
    if sized:
        size = profile['size_profile']
        if file_sha(root/size['path']) != size['sha256']:
            raise ContractError('Frozen HC size profile changed')
        provider = AtlasHCProvider(root)
        radii = np.empty(hc.count)
        for start in range(0, hc.count, 4096):
            stop = min(start+4096, hc.count)
            query = provider.footprints(hc.ids(start, stop), args[3][start:stop],
                Quantity(args[2][start:stop], 'mm', 'arbor_anchor_position',
                         'HC_soma_to_OPL_colocation_project_v1', 'OPL'),
                mode='connection_preview', enable_central_candidate=profile['enable_central_candidate'])
            if query['blocked_records']:
                raise ContractError('HC size unavailable: '+str(query['blocked_records'][:1]))
            radii[start:stop] = [row['radius_mm'] for row in query['per_hc_coverage']]
        raw_B, audit = eccentric_projection(*args, radii)
        representation = 'literature_constrained_equal_area_contacts'
        size_binding = size
        directory_path = root/directory
        directory_path.mkdir(parents=True, exist_ok=True)
        path = directory_path/'hc_radius_mm.npy'
        value = audit.pop('_hc_radii_mm')
        if path.exists():
            if not np.array_equal(np.load(path, allow_pickle=False), value):
                raise ContractError('Frozen eccentric HC radii changed')
        else:
            np.save(path, value, allow_pickle=False)
        audit['hc_radii'] = {'path':path.relative_to(root).as_posix(), 'sha256':file_sha(path),
                            'unit':'mm', 'coordinate_frame':'OPL', 'size_profile':size}
    else:
        raw_B,audit=functional_projection(*args,profile['divergence_anchors'])
    support_path=root/directory/'functional_cone_support_mm.npy';support_path.parent.mkdir(parents=True,exist_ok=True)
    support=audit.pop('_per_cone_support_mm')
    if support_path.exists():
        if not np.array_equal(np.load(support_path,allow_pickle=False),support):raise ContractError('Frozen functional support changed')
    else:np.save(support_path,support,allow_pickle=False)
    audit['per_cone_support']={'path':support_path.relative_to(root).as_posix(),'sha256':file_sha(support_path),
        'unit':'mm','meaning':'maximum distance to an actual selected HC partner' if sized else 'maximum actual selected typed-anchor distance, not measured dendrite radius'}
    raw_A=raw_B.T.tocsr();csum=np.asarray(raw_B.sum(1)).ravel();hsum=np.asarray(raw_A.sum(1)).ravel()
    inv_c=np.divide(1.,csum,out=np.zeros_like(csum),where=csum>0)
    B=(sparse.diags(inv_c)@raw_B).tocsr();inv_h=np.divide(1.,hsum,out=np.zeros_like(hsum),where=hsum>0)
    A=(sparse.diags(inv_h)@raw_A).tocsr()
    recipe={'route_id':'cones_HC_effective_functional_projection','pre_population':'cones','post_population':'HC',
        'pre_table':cone.binding,'post_table':hc.binding,'pre_count':cone.count,'post_count':hc.count,
        'profile':{'path':profile_path,'sha256':file_sha(profile_file)},'operator_audit':audit,
        'representation':representation,
        'original_anatomical_reference':anatomical_reference['raw_contacts'],
        'original_anatomical_owner_no_feedback_count':anatomical_reference['owner_zero_feedback_cones']}
    if sized: recipe['size_profile'] = size_binding
    writer=SparseRouteWriter(root,directory/'raw_functional_projection',recipe)
    if not writer.manifest.exists():
        for start in range(writer.next_row,hc.count,4096):
            stop=min(hc.count,start+4096);block=raw_A[start:stop];counts=np.diff(block.indptr)
            writer.append(start,block,[{'actual_degree':int(n),'missing_degree':0,'role':'equal-area disk contact' if sized else 'functional association, not measured dendritic contact'} for n in counts],
                row_status=['READY_SIZED_HC_CONTACTS' if sized and n else 'READY_FUNCTIONAL_PROJECT_POOL' if n else
                            'COMPLETE_EMPTY_DIRECT_INPUT' if sized else 'COMPLETE_EMPTY_FUNCTIONAL_DIRECT_INPUT' for n in counts])
    raw_binding=writer.freeze();base={'raw_contacts':raw_binding,'normalization':'unit DC once on complete canonical functional projection; no ROI normalization',
        'representation':representation,'profile':recipe['profile'],'physiological_validation':False}
    if sized: base['size_profile'] = size_binding
    a=dict(base,route_id='native_cone_to_HC_A',pre_population='cones',post_population='HC',pre_table=cone.binding,post_table=hc.binding,pre_count=cone.count,post_count=hc.count)
    b=dict(base,route_id='native_HC_to_cone_B',pre_population='HC',post_population='cones',pre_table=hc.binding,post_table=cone.binding,pre_count=hc.count,post_count=cone.count)
    ab=_write_matrix(root,directory/'native_A',a,A,unit='dimensionless',empty_status='NATIVE_DIRECT_HC_INPUT_UNDEFINED_IF_SELECTED')
    bb=_write_matrix(root,directory/'native_B',b,B,unit='dimensionless',empty_status='NATIVE_UNIT_DC_FEEDBACK_UNSUPPORTED_IF_SELECTED')
    graph=sparse.bmat([[None,raw_A],[raw_B,None]],format='csr');number,labels=connected_components(graph,directed=False,return_labels=True)
    path=root/directory/'native_component_labels.npy'
    if path.exists():
        if not np.array_equal(np.load(path,allow_pickle=False),labels):raise ContractError('Functional component labels changed')
    else:np.save(path,labels,allow_pickle=False)
    value={'raw_contacts':raw_binding,'native_A':ab,'native_B':bb,'edges':raw_B.nnz,
        'native_component_labels':{'path':path.relative_to(root).as_posix(),'sha256':file_sha(path),'ordering':'all HC rows, then all cone rows',
            'components':int(number),'largest_node_count':int(np.bincount(labels).max())},
        'owner_zero_feedback_cones':int(np.sum(owner&(csum==0))),'empty_HC_rows':int(np.sum(hsum==0)),
        'representation':representation,'operator_audit':audit,
        'anatomical_reference':anatomical_reference,'profile':recipe['profile'],
        'status':'ECCENTRIC_CONTACTS_FROZEN_EMPTY_ROWS_RETAINED' if sized else 'FUNCTIONAL_PROJECTION_FROZEN_REQUIRES_ACTUAL_RUNTIME_ACCEPTANCE',
        'whole_exterior_recurrent_closure':'NOT_CERTIFIED','physiological_validation':False,'default_changed':False}
    save_json(root/directory/'operator.json',value);return value
