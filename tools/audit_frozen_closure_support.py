"""Read-only cell dependency BFS and finite-support audit of existing graphs.

No model assembly, ISET, integration, edge/cell edits or new scientific condition.
Large row-audit JSON is not decoded when only sparse topology is needed.
"""
from pathlib import Path
import argparse,gc,hashlib,json,sys,time
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order,connected_components
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.contracts import canonical_hash
from retina_sim.disk_circuit import GlobalDiskCircuit,OUTPUTS
from retina_sim.disk_geometry import DiskAtlasROI
from retina_sim.disk_graph_storage import array_identity
from retina_sim.background_checkpoint import MemoryGuard


class SparseReader:
    def __init__(self,circuit):self.circuit=circuit;self.verified=set()
    def rows(self,descriptor,posts):
        posts=np.asarray(posts,dtype=np.int64);route=self.circuit.route(descriptor);parts=[]
        for block in route.record['blocks']:
            local=posts[(posts>=block['start'])&(posts<block['stop'])]-block['start']
            if not len(local):continue
            path=(ROOT/block['path']).resolve()
            if not path.is_relative_to(ROOT):raise ValueError('Sparse block escaped project')
            if block['path'] not in self.verified:
                with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                if digest!=block['sha256']:raise ValueError('Frozen sparse block changed')
                self.verified.add(block['path'])
            with np.load(path,allow_pickle=False) as archive:
                matrix=sparse.csr_matrix((archive['data'],archive['indices'],archive['indptr']),shape=tuple(archive['shape']))
                parts.append(matrix[local])
        return sparse.vstack(parts,format='csr') if parts else sparse.csr_matrix((0,route.record['recipe']['pre_count']))


def audit_case(label,configuration,bounds,expected_count):
    begin=time.perf_counter();cfg=json.loads((ROOT/configuration).read_bytes())
    circuit=GlobalDiskCircuit(ROOT,cfg['global_graph']);reader=SparseReader(circuit)
    roi=DiskAtlasROI(tuple(bounds),tuple(bounds),circuit.record['reference_id'])
    selected={};owner_seed={};owner_rf={};owner_roles={};events=[];rounds=[];seen={}
    def counts():return {name:len(rows) for name,rows in selected.items() if rows}
    for name,descriptor in circuit.record['tables'].items():
        table=circuit.table(name)
        xy=table.positions if descriptor['kind']=='disk_population' else table.coordinates.get('rf')
        owner_rf[name]=None if xy is None else int(np.count_nonzero(roi.owner_mask(xy)))
        owner_roles[name]='RF role unavailable; no soma/RF substitution' if xy is None else 'declared global RF chart'
    for name in OUTPUTS:
        rows=circuit.table(name).query_indices(roi,owner=True)
        selected[name]=set(map(int,rows));owner_seed[name]=len(rows)
    def expansion(round_id,kind):
        before=counts();changed=circuit._expand_components(selected,kind);after=counts()
        delta={name:after.get(name,0)-before.get(name,0) for name in after if after.get(name,0)!=before.get(name,0)}
        if delta:events.append({'round':round_id,'edge_type':kind,'added_by_population':delta,'counts_after':after})
        return changed
    changed=True;round_id=0
    while changed:
        round_id+=1
        if round_id>64:raise ValueError('Unexpected dependency propagation depth')
        before=counts();changed=expansion(round_id,'native_components')|expansion(round_id,'electrical_components')
        for descriptor in circuit.record['routes']:
            if descriptor['kind'] in ('native','gap'):continue
            pending=selected.get(descriptor['post'],set())-seen.setdefault(descriptor['binding']['sha256'],set())
            if not pending:continue
            seen[descriptor['binding']['sha256']].update(pending)
            matrix=reader.rows(descriptor,sorted(pending))
            # Match the existing accelerator first; independent BFS below then
            # discards stored zero weights rather than trusting row membership.
            parents=np.unique(matrix.indices);target=selected.setdefault(descriptor['pre'],set())
            old=len(target);target.update(map(int,parents));added=len(target)-old;changed|=bool(added)
            events.append({'round':round_id,'edge_type':descriptor['kind'],'route':descriptor['route_id'],
                'pending_post_rows':len(pending),'stored_edges':matrix.nnz,'added_population':descriptor['pre'],
                'added_nodes':added,'counts_after':counts() if added else None})
        after=counts();rounds.append({'round':round_id,'before':before,'after':after,
            'added_nodes':sum(after.values())-sum(before.values()),
            'added_by_population':{name:after.get(name,0)-before.get(name,0) for name in after if after.get(name,0)!=before.get(name,0)}})
    chosen={name:np.array(sorted(rows),dtype=np.int64) for name,rows in selected.items() if rows}
    selected=None;names=sorted(chosen);offsets={};total=0
    for name in names:offsets[name]=total;total+=len(chosen[name])
    if total!=expected_count:raise AssertionError('Existing source closure count changed: '+str(total))
    trace_identity=canonical_hash({'graph':circuit.binding,'owner':tuple(bounds),
        'selected_rows':{name:array_identity(rows) for name,rows in chosen.items()}})
    if label=='functional_central':
        original=json.loads((ROOT/'reports/central_empty_sum_short_v1/check.json').read_bytes())
        if trace_identity!=original['frozen_geometry_proof']['view']['view_sha256']:
            raise AssertionError('Independent trace row identities differ from the executed central view')
    edge_rows=[];edge_cols=[];routes=[];zero_entries=0
    for descriptor in circuit.record['routes']:
        pre,post=descriptor['pre'],descriptor['post']
        if pre not in chosen or post not in chosen:continue
        matrix=reader.rows(descriptor,chosen[post]);keep=matrix.data!=0
        parent=matrix.indices[keep];local=np.searchsorted(chosen[pre],parent)
        if len(parent) and (np.any(local>=len(chosen[pre])) or np.any(chosen[pre][local]!=parent)):
            raise AssertionError('Selected closure lost a nonzero directed parent')
        rows=np.repeat(np.arange(len(chosen[post]),dtype=np.int32),np.diff(matrix.indptr))[keep]
        edge_rows.append(rows+offsets[post]);edge_cols.append(local.astype(np.int32)+offsets[pre])
        zeros=int(np.count_nonzero(~keep));zero_entries+=zeros
        routes.append({'route':descriptor['route_id'],'edge_type':descriptor['kind'],'pre':pre,'post':post,
            'selected_pre_nodes':len(chosen[pre]),'selected_post_nodes':len(chosen[post]),
            'directed_nonzero_dependency_edges':len(parent),'stored_zero_weight_entries':zeros})
    seeds=[]
    for name in OUTPUTS:
        rows=circuit.table(name).query_indices(roi,owner=True)
        if len(rows):seeds.extend((offsets[name]+np.searchsorted(chosen[name],rows)).tolist())
    edge_rows.append(np.full(len(seeds),total,dtype=np.int32));edge_cols.append(np.asarray(seeds,dtype=np.int32))
    rr=np.concatenate(edge_rows);cc=np.concatenate(edge_cols);edge_rows=edge_cols=None
    graph=sparse.csr_matrix((np.ones(len(rr),dtype=np.uint8),(rr,cc)),shape=(total+1,total+1));rr=cc=None
    reached=breadth_first_order(graph,total,directed=True,return_predecessors=False)
    mask=np.zeros(total+1,dtype=bool);mask[reached]=True
    direct=graph[seeds].indices;direct_mask=np.zeros(total,dtype=bool);direct_mask[direct]=True
    populations=[]
    for name in names:
        sl=slice(offsets[name],offsets[name]+len(chosen[name]));required=int(np.count_nonzero(mask[sl]))
        populations.append({'population':name,'owner_RF_nodes':owner_rf[name],'owner_role':owner_roles[name],
            'requested_output_owner_nodes':owner_seed.get(name,0),'direct_parents_of_output_owners':int(np.count_nonzero(direct_mask[sl])),
            'required_by_independent_nonzero_directed_BFS':required,'selected_by_current_component_accelerator':len(chosen[name]),
            'unnecessary_selected_nodes':len(chosen[name])-required})
    for name in sorted(set(circuit.record['tables'])-set(names)):
        populations.append({'population':name,'owner_RF_nodes':owner_rf[name],'owner_role':owner_roles[name],
            'requested_output_owner_nodes':owner_seed.get(name,0),'direct_parents_of_output_owners':0,
            'required_by_independent_nonzero_directed_BFS':0,'selected_by_current_component_accelerator':0,
            'unnecessary_selected_nodes':0})
    target='midget_RGC_ON';target_rows=circuit.table(target).query_indices(roi,owner=True)
    target_index=offsets[target]+int(np.searchsorted(chosen[target],target_rows[0]))
    single_reached=breadth_first_order(graph,target_index,directed=True,return_predecessors=False)
    single_mask=np.zeros(total,dtype=bool);single_mask[single_reached]=True
    single_counts={name:int(np.count_nonzero(single_mask[offsets[name]:offsets[name]+len(chosen[name])])) for name in names}
    single={'target_component':target,'target_global_row':int(target_rows[0]),
        'target_global_id':circuit.table_record(target,target_rows[:1])['global_cell_ids'][0],
        'scope':'First owner target used by the existing20ms matched response; other requested owner outputs removed only in this read-only dependency query',
        'required_nodes':len(single_reached),'required_by_population':single_counts,
        'native_nodes':single_counts['cones']+single_counts['HC'],
        'extra_nodes_for_all_ROI_owner_outputs':total-len(single_reached)}
    native=np.concatenate([np.arange(offsets[name],offsets[name]+len(chosen[name]),dtype=np.int32) for name in ('HC','cones')])
    number,labels=connected_components(graph[native][:,native],directed=True,connection='strong')
    sizes=np.bincount(labels);native_proof={'selected_native_nodes':len(native),'strong_component_count':int(number),
        'largest_strong_component_nodes':int(sizes.max()),'native_feedback_routes':[
            row for row in routes if row['edge_type']=='native'],
        'interpretation':'Nonzero cone release/z1 -> HC A pooling/state -> cone B feedback/release recursion in the frozen model; exact cell-level directed dependency, not evidence that physiology requires a whole disk for a small ROI'}
    result={'case':label,'graph':circuit.binding,'configuration':configuration,'owner_bounds_RF_mm':bounds,
        'selection_precedes_state_allocation_ISET_and_integration':True,
        'selection_call_chain':'check_disk3mm graph qualification -> disk_runtime.disk_network owner_view -> runtime_record -> factories/assembly -> ISET -> runner.initialize/integration',
        'requested_output_owner_nodes':sum(owner_seed.values()),'selected_global_nodes':total,
        'independent_directed_required_nodes':len(reached)-1,'unnecessary_selected_nodes':total-(len(reached)-1),
        'stored_zero_weight_entries_ignored_by_independent_BFS':zero_entries,'view_semantic_sha256':trace_identity,
        'populations':populations,'rounds':rounds,'expansion_events':events,'routes':routes,'native_feedback':native_proof,
        'single_recorded_owner_query':single,
        'scope':'Existing finite graph only; no exterior physiological closure or new regional boundary approximation',
        'wall_seconds':time.perf_counter()-begin}
    if label=='functional_central':result['central_AII_finite_support']=aii_audit(circuit,reader,chosen)
    print(json.dumps({key:result[key] for key in ('case','requested_output_owner_nodes','selected_global_nodes','independent_directed_required_nodes','unnecessary_selected_nodes','native_feedback','wall_seconds')},indent=2),flush=True)
    return result


def aii_audit(circuit,reader,chosen):
    descriptor=next(row for row in circuit.record['routes'] if row['route_id']=='RBC_AII_AMPA')
    route=circuit.route(descriptor);recipe=route.record['recipe'];support=float(recipe['hard_support_mm'])
    rbc=circuit.table('rod_BC');aii=circuit.table('AII_AC')
    if recipe['pre_master']!=rbc.binding or recipe['post_master']!=aii.binding:
        raise AssertionError('RBC/AII source population identity differs')
    for key,positions in (('pre_coordinates',rbc.positions),('post_coordinates',aii.positions)):
        if array_identity(positions)!=recipe['resolved_numerics'][key]:raise AssertionError('Route coordinates changed')
    matrix=reader.rows(descriptor,chosen['AII_AC']);empty_local=np.flatnonzero(np.diff(matrix.indptr)==0)
    empty=chosen['AII_AC'][empty_local];xy=aii.positions[empty]
    tree=cKDTree(rbc.positions);nearest,nearest_rows=tree.query(xy)
    candidates=tree.query_ball_point(xy,support);complete=np.linalg.norm(xy,axis=1)+support<=rbc.radius
    omitted=np.setdiff1d(np.unique(matrix.indices),chosen['rod_BC'])
    if len(omitted) or not np.all(complete) or any(candidates):
        raise AssertionError('An empty AII row has omitted candidates, incomplete support or a source-query mismatch')
    _,_,audits,statuses=route.selected_rows(empty)
    ids=circuit.table_record('AII_AC',empty)['global_cell_ids'];gap_count=np.zeros(len(empty),dtype=np.int64)
    out_count=np.zeros(len(empty),dtype=np.int64)
    for d in circuit.record['routes']:
        pre,post=d['pre'],d['post']
        if pre not in chosen or post not in chosen:continue
        if d['kind']=='gap' and 'AII_AC' in (pre,post):
            m=reader.rows(d,chosen[post]);nonzero=m.data!=0
            if post=='AII_AC':
                counts=np.bincount(np.repeat(np.arange(len(chosen[post])),np.diff(m.indptr))[nonzero],minlength=len(chosen[post]))
                gap_count+=counts[np.searchsorted(chosen[post],empty)]
            if pre=='AII_AC':
                columns=m.indices[nonzero];gap_count+=np.array([np.count_nonzero(columns==index) for index in empty])
        if d['kind']=='primary_chemical' and pre=='AII_AC':
            m=reader.rows(d,chosen[post]);columns=m.indices[m.data!=0]
            out_count+=np.array([np.count_nonzero(columns==index) for index in empty])
    rows=[]
    for i,index in enumerate(empty):
        rows.append({'AII_global_row':int(index),'AII_global_id':ids[i],'position_project_IPL_mm':xy[i].tolist(),
            'global_RBC_candidates_within_hard_support':len(candidates[i]),'global_CSR_degree':0,'runtime_CSR_degree':0,
            'hard_support_mm':support,'full_support_inside_fixed_source_disk':bool(complete[i]),
            'source_disk_radius_mm':rbc.radius,'source_boundary_margin_mm':float(rbc.radius-np.linalg.norm(xy[i])-support),
            'nearest_global_RBC_distance_mm':float(nearest[i]),'nearest_global_RBC_row':int(nearest_rows[i]),
            'raw_row_status':statuses[i],'raw_row_audit':audits[i],
            'actual_selected_gap_incidences_both_directions':int(gap_count[i]),
            'actual_selected_outgoing_glycine_contacts':int(out_count[i]),
            'engineering_empty_sum_permitted':True,'biological_absence_established':False})
    return {'graph':circuit.binding,'route':descriptor['binding'],'pre_source':rbc.binding,'post_source':aii.binding,
        'selected_AII':len(chosen['AII_AC']),'global_RBC_source_count':len(rbc.positions),'selected_RBC':len(chosen['rod_BC']),
        'empty_rows':len(rows),'empty_rows_with_actual_gap':int(np.count_nonzero(gap_count)),
        'empty_rows_also_without_gap':int(np.count_nonzero(gap_count==0)),
        'omitted_nonzero_RBC_parents':len(omitted),'all_empty_supports_complete_in_frozen_source':bool(np.all(complete)),
        'all_empty_global_candidate_queries_zero':not any(candidates),
        'filtering':'All180838 frozen RBC positions searched before nearest-k budget; no ROI/type/exterior candidate substitution. Empty global rows remain empty after exact selected-view filtering.',
        'density_source':rbc.record['recipe']['source'],
        'central_density_rule':'Existing Packer rod quadratic continuation below first digitized node, then rhoRod/18 for RBC; no finite hard rod-free hole is declared by this source rule.',
        'calibration_scope':'Complete support of the frozen project graph only. The40um support, RBC density transfer, AII map/gaps and isolated passive outputs are not demonstrated physiological anatomy.',
        'rows':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='reports/current_wip_review/closure_and_AII_finite_support_audit.json')
    args=parser.parse_args();output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Audit output outside project')
    if output.exists():raise ValueError('Retain the prior audit')
    guard=MemoryGuard(max_rss_bytes=2*1024**3,min_available_bytes=9*1024**3)
    if not guard.check():raise MemoryError('Preserve the active worker resource reserve')
    result={'schema':'existing_graph_directed_closure_and_finite_support_audit_v1','neural_simulation_performed':False,
        'cells_edges_weights_changed':False,'default_changed':False,'cases':[]}
    for case,config,bounds,count in (
        ('original_transition','configurations/atlas_human_borrowed_disk3mm_floor_input_v2.json',(.96,1.04,-.04,.04),879437),
        ('functional_central','configurations/atlas_human_borrowed_disk3mm_functional_hc_empty_sum_v2.json',(-.01,.01,-.01,.01),894425)):
        result['cases'].append(audit_case(case,config,bounds,count));gc.collect()
        if not guard.check():raise MemoryError('Preserve the active worker resource reserve')
    result['memory']=guard.last
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('READ_ONLY_CLOSURE_SUPPORT_AUDIT_COMPLETE',str(output),flush=True)

if __name__=='__main__':main()
