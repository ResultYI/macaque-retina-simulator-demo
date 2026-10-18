"""Named functional projection; anatomical contact references remain independent."""
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


def freeze_functional_hc(root,outer,anatomical_reference,directory,profile_path='configurations/hc_functional_divergence_w2000_v1.json'):
    root=Path(root).resolve();directory=Path(directory);profile_file=root/profile_path;profile=json.loads(profile_file.read_bytes())
    if profile['schema']!='HC_effective_functional_projection_profile_v1':raise ContractError('Named functional HC profile required')
    source=profile['primary_source']
    if file_sha(root/source['path'])!=source['sha256']:raise ContractError('Frozen W2000 primary identity changed')
    cone=FrozenDiskTable(root,outer['cones']);hc=FrozenDiskTable(root,outer['HC'])
    owner=np.sum(cone.coordinates['rf']**2,axis=1)<=9
    raw_B,audit=functional_projection(cone.coordinates['opl_pedicle'],cone.extras['subtype'].astype('U1'),
        hc.coordinates['soma'],hc.extras['subtype'].astype('U2'),owner,hc.record['source_bindings']['fixed_cohort_OPL_radius_mm'],profile['divergence_anchors'])
    support_path=root/directory/'functional_cone_support_mm.npy';support_path.parent.mkdir(parents=True,exist_ok=True)
    support=audit.pop('_per_cone_support_mm')
    if support_path.exists():
        if not np.array_equal(np.load(support_path,allow_pickle=False),support):raise ContractError('Frozen functional support changed')
    else:np.save(support_path,support,allow_pickle=False)
    audit['per_cone_support']={'path':support_path.relative_to(root).as_posix(),'sha256':file_sha(support_path),
        'unit':'mm','meaning':'maximum actual selected typed-anchor distance, not measured dendrite radius'}
    raw_A=raw_B.T.tocsr();csum=np.asarray(raw_B.sum(1)).ravel();hsum=np.asarray(raw_A.sum(1)).ravel()
    B=(sparse.diags(1/csum)@raw_B).tocsr();inv_h=np.divide(1.,hsum,out=np.zeros_like(hsum),where=hsum>0)
    A=(sparse.diags(inv_h)@raw_A).tocsr()
    recipe={'route_id':'cones_HC_effective_functional_projection','pre_population':'cones','post_population':'HC',
        'pre_table':cone.binding,'post_table':hc.binding,'pre_count':cone.count,'post_count':hc.count,
        'profile':{'path':profile_path,'sha256':file_sha(profile_file)},'operator_audit':audit,
        'representation':'effective_functional_association_not_anatomical_synapse',
        'original_anatomical_reference':anatomical_reference['raw_contacts'],
        'original_anatomical_owner_no_feedback_count':anatomical_reference['owner_zero_feedback_cones']}
    writer=SparseRouteWriter(root,directory/'raw_functional_projection',recipe)
    if not writer.manifest.exists():
        for start in range(writer.next_row,hc.count,4096):
            stop=min(hc.count,start+4096);block=raw_A[start:stop];counts=np.diff(block.indptr)
            writer.append(start,block,[{'actual_degree':int(n),'missing_degree':0,'role':'functional association, not measured dendritic contact'} for n in counts],
                row_status=['READY_FUNCTIONAL_PROJECT_POOL' if n else 'COMPLETE_EMPTY_FUNCTIONAL_DIRECT_INPUT' for n in counts])
    raw_binding=writer.freeze();base={'raw_contacts':raw_binding,'normalization':'unit DC once on complete canonical functional projection; no ROI normalization',
        'representation':'effective_functional_association_not_anatomical_synapse','profile':recipe['profile'],'physiological_validation':False}
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
        'owner_zero_feedback_cones':0,'empty_HC_rows':int(np.sum(hsum==0)),
        'representation':'effective_functional_association_not_anatomical_synapse','operator_audit':audit,
        'anatomical_reference':anatomical_reference,'profile':recipe['profile'],
        'status':'FUNCTIONAL_PROJECTION_FROZEN_REQUIRES_ACTUAL_RUNTIME_ACCEPTANCE',
        'whole_exterior_recurrent_closure':'NOT_CERTIFIED','physiological_validation':False,'default_changed':False}
    save_json(root/directory/'operator.json',value);return value
