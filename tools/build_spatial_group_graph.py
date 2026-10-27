"""Build and audit a static group graph directly from frozen arrays, never a runtime."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from scipy import sparse

from retina_sim.contracts import ContractError
from retina_sim.spatial_group_graph import (SCHEMA, digest, verified_path, polar_cells,
    group_members, aggregate_edges, mean_matrix, moments, gap_laplacian, save_csr,
    load_static_graph)

RGC = ('midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON')
BC = ('midget_BC_ON','midget_BC_OFF','S_ON_BC','diffuse_BC_ON','diffuse_BC_OFF')
AC = ('local_feedforward_AC','local_crossover_AC','widefield_AC')
PARENT = {'midget_BC_ON':('cones','cone_row'), 'midget_BC_OFF':('cones','cone_row'),
          'S_ON_BC':('cones','cone_row'), 'midget_RGC_ON':('midget_BC_ON','parent_bc_row'),
          'midget_RGC_OFF':('midget_BC_OFF','parent_bc_row'),
          'small_bistratified_S_ON':('S_ON_BC','parent_bc_row')}


def json_write(path,value):
    path=Path(path)
    pending=path.with_name(path.name+'.writing')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    os.replace(pending,path)


class Builder:
    config_schema = 'retipath-static-group-build-1'
    artifact_schema = SCHEMA

    def validate_configuration(self):
        if (self.config.get('schema') != self.config_schema
                or self.config.get('scope') != 'static_only_no_solver_or_fit_approval'
                or self.config.get('local_fine_targets') != []):
            raise ContractError('This static build covers the explicit base grouping only')

    def __init__(self,root,config_path,output):
        self.root=Path(root).resolve()
        self.config_path=Path(config_path).resolve()
        self.config=json.loads(self.config_path.read_text(encoding='utf8'))
        self.validate_configuration()
        self.output=Path(output).resolve()
        self.work=self.output.with_name(self.output.name+'.building')
        if (not self.output.is_relative_to(self.root) or self.output==self.root
                or self.output.exists() or self.work.exists() or not self.output.parent.is_dir()):
            raise ContractError('New static output within the workspace required; preserve existing runs')
        self.work.mkdir()
        self.started=time.perf_counter()
        from retina_sim.background_checkpoint import MemoryGuard
        limits=self.config['static_resource_limits']
        self.guard=MemoryGuard(limits['max_rss_bytes'],limits['minimum_available_bytes'])
        self.groups={}
        self.routes=[]
        self.stage='PREFLIGHT'
        self.guard_tick=0.
        self.metadata=dict(schema=self.artifact_schema,runtime_ready=False,model_forward_started=False,
            fit_eligible=False,default_adopted=False,spatial_approximation_accepted=False,
            configuration=self.config,configuration_sha256=digest(self.config_path),
            implementation={p:digest(self.root/p) for p in
                ('tools/build_spatial_group_graph.py','retina_sim/spatial_group_graph.py')})
        self.check(force=True)

    def prepare(self, graph, selected, profiles):
        """Subclass hook for static policies, never a numerical runtime."""

    def complete_metadata(self):
        """Subclass hook before immutable file identities are finalized."""

    def verify_output(self):
        load_static_graph(self.work)

    def budget_audit(self, counts, contact, cell_states, contact_states, edges):
        budgets=self.config['proposal_budgets']
        return dict(core_state_scalars=cell_states+contact_states,
            cell_states_including_analytic_RE=cell_states,contact_states=contact_states,
            extra_RGC_history_probability_scalars=8*sum(counts[n] for n in RGC),
            runtime_edge_entries=edges,edge_scope='One aggregated route/contact pair term per source route plus gap Laplacian; not compiled-kernel instruction count',
            cell_slots=sum(counts.values()),cell_slot_proposal_met=sum(counts.values())<=budgets['base_cell_slots'],
            core_proposal_met=cell_states+contact_states<=budgets['core_state_scalars'],
            edge_proposal_met=edges<=budgets['runtime_edge_entries'],contact_groups=contact,
            contact_proposal_met={k:v<=budgets['contact_pairs'][k] for k,v in contact.items()},
            scope='Base grouping; no requested local fine targets; derived-parameter/runtime-stage costs not certified')

    def check(self,*,force=False):
        now=time.perf_counter()
        if now-self.started>self.config['static_resource_limits']['max_wall_seconds']:
            raise ContractError('Static build wall limit reached; partial evidence retained')
        if (self.work/'REQUEST_STOP').exists():
            raise ContractError('Explicit static build stop; partial evidence retained')
        if force or now-self.guard_tick>=2.:
            if not self.guard.check():
                raise ContractError('Static build memory stop: '+str(self.guard.stop_reason))
            self.guard_tick=now
            json_write(self.work/'status.json',dict(stage=self.stage,pid=os.getpid(),
                elapsed_seconds=now-self.started,resources=self.guard.last,
                model_forward_started=False))

    def read(self,binding):
        return json.loads(verified_path(self.root,binding).read_text(encoding='utf8'))

    def array(self,binding):
        return np.load(verified_path(self.root,binding),allow_pickle=False,mmap_mode='r')

    def arrays(self,directory,arrays):
        folder=self.work/directory
        folder.mkdir(parents=True,exist_ok=True)
        for name,value in arrays.items():
            with (folder/(name+'.npy')).open('xb') as stream:
                np.save(stream,value,allow_pickle=False)

    def populations(self,graph,selected,profiles):
        order=['cones','HC','rods','rod_BC','AII_AC',*BC,*AC,*RGC]
        for name in order:
            self.stage='GROUPING_'+name
            self.check(force=True)
            descriptor=graph['tables'][name]
            record=self.read(descriptor['binding'])
            rows=np.asarray(selected[name],dtype=np.int64)
            if np.any(rows<0) or np.any(rows>=record['count']):
                raise ContractError('Selected fine row outside source table')
            coordinates=record.get('coordinates',{'rf':dict(record.get('positions',{}),frame='functional_visual_field',unit='mm')})
            role='rf' if 'rf' in coordinates else 'soma' if name=='HC' else 'ipl_preterminal'
            if role not in coordinates or coordinates[role].get('unit')!='mm':
                raise ContractError('No approved source coordinate for '+name)
            xy=np.asarray(self.array(coordinates[role])[rows],dtype=np.float64)
            cells=polar_cells(xy,self.config['bands'],self.config['mm_per_degree'])
            if name in RGC and np.any(np.hypot(xy[:,0],xy[:,1])>self.config['output_radius_deg']*self.config['mm_per_degree']):
                raise ContractError('RGC owner outside frozen output disk')
            subtype=None
            labels=[]
            if name in ('cones','HC'):
                raw=self.array(record['extras']['subtype'])[rows]
                unique,subtype=np.unique(raw,return_inverse=True)
                labels=[v.decode('ascii') for v in unique]
                if set(labels)-set(('L','M','S') if name=='cones' else ('H1','H2')):
                    raise ContractError('Unknown spectral/HC subtype')
            parent=None
            if name in PARENT:
                parent_name,key=PARENT[name]
                source_parent=self.array(record['extras'][key])[rows]
                parent=self.groups[parent_name]['global_map'][source_parent]
                if np.any(parent<0):
                    raise ContractError('Missing original private-line parent in selected closure')
            eligibility=None
            if 'eligible_drive' in record.get('extras',{}):
                eligibility=self.array(record['extras']['eligible_drive'])[rows].astype(np.int64)
            group=group_members(rows,cells,subtype=subtype,parent=parent,eligibility=eligibility)
            n=len(group['counts'])
            mapping=np.full(record['count'],-1,dtype=np.int32)
            mapping[rows]=group['fine_to_group']
            group.update(global_map=mapping,rows=rows)
            self.groups[name]=group
            centroid=np.column_stack([np.bincount(group['fine_to_group'],weights=xy[:,k],minlength=n)/group['counts'] for k in range(2)])
            lo=np.full((n,2),np.inf);hi=np.full((n,2),-np.inf)
            for k in range(2):
                np.minimum.at(lo[:,k],group['fine_to_group'],xy[:,k])
                np.maximum.at(hi[:,k],group['fine_to_group'],xy[:,k])
            profile_name='effective_'+name
            definitions=profiles.get(profile_name,{})
            group['description']=dict(name=name,group_count=n,fine_neurons=len(rows),
                full_asset_count=record['count'],source_table=descriptor['binding'],
                source_keys=record['source_keys'],coordinate_role=role,coordinate=coordinates[role],
                subtype_labels=labels,parent_population=PARENT.get(name,(None,None))[0],
                key_columns=['polar_cell','subtype','parent_group','eligible_drive'],
                parameter_class_source=profile_name if definitions else 'original_native_or_author_profile',
                derived_parameter_fields=[k for k,v in definitions.items() if v.get('binding')=='derived'],
                parameter_runtime_binding='NOT_YET_CONSTRUCTED; no per-cell derived value silently averaged',
                state_semantics='representative_group_not_independent_fine_cell_state')
            values={k:group[k] for k in ('keys','counts','member_offsets','member_global_rows')}
            values.update(fine_to_group=mapping,centroid_mm=centroid,bounds_min_mm=lo,bounds_max_mm=hi,
                group_ids=np.asarray(['sg:'+self.metadata['configuration_sha256'][:16]+':'+name+':'+str(i) for i in range(n)]))
            totals={}
            for field in ('C','gL'):
                row=definitions.get(field,{})
                if row.get('binding')=='broadcast' and np.isscalar(row.get('value')):
                    totals[field+'_total']=group['counts']*float(row['value'])
            if totals:
                values.update(totals)
                group['description']['membrane_totals']='Sum of source broadcast C/gL, not means; derived initial states remain unbound'
            self.arrays('populations/'+name,values)

    def route(self,descriptor,graph,profiles):
        pre,post=descriptor['pre'],descriptor['post']
        self.stage='AGGREGATING_'+pre+'_TO_'+post
        source=self.read(descriptor['binding'])
        left,right=self.groups[pre],self.groups[post]
        shape=(len(right['counts']),len(left['counts']))
        total=sparse.csr_matrix(shape,dtype=np.float64)
        contact=next((k for k,v in graph['contact_geometry'].items() if v['route']==descriptor['binding']),None)
        coefficient=sparse.csr_matrix(shape,dtype=np.float64) if contact else None
        overlap=None
        contact_record=None
        if contact:
            contact_record=self.read(graph['contact_geometry'][contact]['binding'])
            if contact_record.get('overlap'):
                overlap=self.array(contact_record['overlap'])
            profile_name='effective_individual_TRPM1' if contact=='individual_rod_RBC_contacts' else 'effective_'+contact
            contact_profile=profiles[profile_name]
        expected_dc=np.zeros(shape[0]);zeros=np.zeros(shape[0],np.int64)
        fine_entries=0;offset=0;source_pairs=[];source_indices=[]
        selected_statuses={}
        for block in source['blocks']:
            self.check()
            path=verified_path(self.root,block)
            with np.load(path,allow_pickle=False) as arrays:
                pointer=arrays['indptr'];columns=arrays['indices'];values=arrays['data']
                if tuple(arrays['shape'])!=(block['stop']-block['start'],left['global_map'].size):
                    raise ContractError('Original route block shape differs')
                if not np.isfinite(values).all() or np.any(values<0):
                    raise ContractError('Original route weights must be finite nonnegative')
                fine_post=np.repeat(np.arange(block['start'],block['stop']),np.diff(pointer))
                post_group=right['global_map'][fine_post]
                mask=post_group>=0
                pg=post_group[mask]
                cg=left['global_map'][columns[mask]]
                if np.any(cg<0):
                    raise ContractError('Frozen selected closure omits a required presynaptic row')
                w=values[mask]
                total=total+aggregate_edges(pg,cg,w,shape)
                row_group=right['global_map'][block['start']:block['stop']]
                selected=row_group>=0
                row_sum=np.asarray(sparse.csr_matrix((values,columns,pointer),shape=tuple(arrays['shape'])).sum(axis=1)).ravel()
                expected_dc+=np.bincount(row_group[selected],weights=row_sum[selected],minlength=shape[0])
                zero_selected=selected & (np.diff(pointer)==0)
                zeros+=np.bincount(row_group[zero_selected],minlength=shape[0])
                labels,counts=np.unique(arrays['row_status'][selected],return_counts=True)
                for label,count in zip(labels,counts):
                    selected_statuses[str(label)]=selected_statuses.get(str(label),0)+int(count)
                fine_entries+=len(w)
                if contact:
                    idx=offset+np.flatnonzero(mask)
                    source_indices.append(idx)
                    source_pairs.append(pg.astype(np.int64)*shape[1]+cg)
                    if contact=='individual_rod_RBC_contacts':
                        gain=2e-9*w
                    else:
                        if overlap is None:
                            raise ContractError('Original chemical physical overlap missing')
                        gain=float(contact_profile['g_peak']['value'])*overlap[idx]
                    coefficient=coefficient+aggregate_edges(pg,cg,gain,shape)
                offset+=len(values)
        if contact_record is not None and offset!=contact_record['count']:
            raise ContractError('Original contact entry ordering/size differs')
        average=mean_matrix(total,right['counts'])
        original_dc=expected_dc/right['counts']
        residual=float(np.max(np.abs(np.asarray(average.sum(axis=1)).ravel()-original_dc),initial=0.))
        scale=max(1.,float(np.max(np.abs(original_dc),initial=0.)))
        if residual>1e-12*scale:
            raise ContractError('Group projection failed constant-signal identity')
        name=pre+'_to_'+post
        folder=self.work/'routes'/name
        folder.mkdir(parents=True)
        save_csr(folder/'sum_weights.npz',total)
        save_csr(folder/'mean_weights.npz',average)
        self.arrays('routes/'+name,dict(zero_input_member_counts=zeros,expected_mean_row_sum=original_dc))
        result=dict(name=name,pre=pre,post=post,kind=descriptor['kind'],source=descriptor['binding'],
            fine_edge_entries=fine_entries,group_pair_entries=total.nnz,source_row_status_counts=selected_statuses,
            fine_empty_rows=int(zeros.sum()),all_empty_groups=int(np.count_nonzero(zeros==right['counts'])),
            partially_empty_groups=int(np.count_nonzero((zeros>0)&(zeros<right['counts']))),
            constant_signal_max_abs_residual=residual,
            sum_semantics='P_post.T W P_pre; raw source weights, not automatically physical conductance',
            mean_semantics='diag(1/member_count_post) P_post.T W P_pre; no crop renormalization')
        if contact:
            save_csr(folder/'contact_coefficient_S.npz',coefficient)
            pairs=np.concatenate(source_pairs)
            indices=np.concatenate(source_indices)
            keys=np.unique(pairs)  # Keep even zero-gain original contact classes.
            ids=np.searchsorted(keys,pairs)
            if np.any(ids>=len(keys)) or not np.array_equal(keys[ids],pairs):
                raise ContractError('Contact group membership lost a positive original contact')
            counts=np.bincount(ids,minlength=len(keys))
            self.arrays('routes/'+name,dict(fine_contact_global_rows=indices,
                fine_contact_to_group=ids.astype(np.int32),contact_member_offsets=np.r_[0,np.cumsum(counts)],
                contact_member_global_rows=indices[np.argsort(ids,kind='stable')],
                contact_post_group=keys//shape[1],contact_pre_group=keys%shape[1]))
            result['contact']=dict(component=contact,groups=len(keys),fine_contacts=len(indices),
                coefficient_unit='S',coefficient_sum=float(coefficient.sum()),
                coefficient_formula='sum(2e-9 * original W)' if contact=='individual_rod_RBC_contacts' else 'sum(g_peak * frozen physical_overlap); original W values are not ncontact',
                normalization='TRPM1 gate direct; chemical double-exponential Z remains a separate runtime factor',
                source_geometry=graph['contact_geometry'][contact]['binding'],
                parameter_class={k:v for k,v in contact_profile.items() if k in ('K','hill','tau','tau_rise','tau_decay','decay_minus100','decay_0','g_peak')})
        result['grouping_runtime_ready']=False
        self.routes.append(result)
        return total

    def inputs(self):
        self.stage='PROJECTING_EXISTING_INPUT'
        folder=self.root/self.config['source_run']/'centered_inputs_v2'
        receipt=json.loads((folder/'frontend_receipt.json').read_text(encoding='utf8'))
        for name,expected in receipt['files'].items():
            if Path(name).name!=name or digest(folder/name)!=expected:
                raise ContractError('Existing optical cache identity differs')
        group=self.groups['cones'];rows=group['rows'];g=group['fine_to_group'];n=group['counts']
        values=np.load(folder/'cone_unique_counts.npy',allow_pickle=False,mmap_mode='r')
        if values.shape!=(51,group['global_map'].size):
            raise ContractError('Expected original distinct-frame and cone ordering')
        shape=(len(values),len(n))
        mean=np.empty(shape);variance=np.empty(shape);lo=np.empty(shape);hi=np.empty(shape)
        rms=[]
        relative=[]
        contrast_relative=[]
        background=np.load(folder/'cone_background_counts.npy',allow_pickle=False,mmap_mode='r')
        bg=moments(background[rows],g,n)
        for i,v in enumerate(values):
            self.check()
            mean[i],variance[i],lo[i],hi[i]=moments(v[rows],g,n)
            rms.append(float(np.sqrt(np.sum(variance[i]*n)/sum(n))))
            original=v[rows]
            energy=float(np.sum(original**2))
            relative.append(float(np.sqrt(np.sum(variance[i]*n)/energy)) if energy else None)
            change=original-background[rows]
            projected=mean[i][g]-bg[0][g]
            energy=float(np.sum(change**2))
            contrast_relative.append(float(np.sqrt(np.sum((change-projected)**2)/energy)) if energy else None)
        native_background=np.load(folder/'cone_background_native_td.npy',allow_pickle=False,mmap_mode='r')
        bgtd=moments(native_background[rows],g,n)[0]
        reference=json.loads((folder/'actual_global_reference.json').read_text(encoding='utf8'))
        labels=group['description']['subtype_labels']
        types=[{'L':0,'M':1,'S':2}[labels[k]] for k in group['keys'][:,1]]
        alpha=np.array(reference['q_ref_LMS_rstar_s'])[types]/100.
        native=mean/receipt['actual_capture_window_seconds']/alpha
        self.arrays('input',dict(group_unique_counts=mean,group_count_variance=variance,
            group_count_min=lo,group_count_max=hi,group_native_td=native,
            group_background_counts=bg[0],group_background_variance=bg[1],group_background_native_td=bgtd,
            source_frame_to_unique=np.load(folder/'source_frame_to_unique.npy',allow_pickle=False)))
        self.metadata['input']=dict(source_receipt_sha256=digest(folder/'frontend_receipt.json'),
            source_files=receipt['files'],source_directory=str(folder.relative_to(self.root)),
            capture_window_seconds=receipt['actual_capture_window_seconds'],unique_frames=51,
            cone_groups=len(n),units=dict(counts='expected R*/window',variance='(expected R*/window)^2',native='td reference-equivalent'),
            all_frame_reconstruction_rmse_counts=rms,max_within_group_count_span=float(np.max(hi-lo)),
            all_frame_relative_L2_counts=relative,all_frame_relative_L2_change_from_background=contrast_relative,
            variance_scope='Spatial heterogeneity of expected optical capture, not photon noise or neural-response error',
            group_mean_value_bytes=int(mean.nbytes),original_mean_value_bytes=int(values.nbytes),
            thresholds_applied=False,spatial_error_acceptance='NOT_EVALUATED',optics_recomputed=False)

    def build(self):
        graph=self.read(self.config['graph'])
        verified_path(self.root,dict(path=self.config['plan'],sha256=self.config['plan_sha256']))
        selection_path=verified_path(self.root,dict(path=self.config['selected_rows'],sha256=self.config['selected_rows_sha256']))
        profile_path=self.root/'configurations/effective_dynamics_v1.json'
        profiles=json.loads(profile_path.read_text(encoding='utf8'))['profiles']
        self.metadata['parameter_sources']=dict(effective_profile_sha256=digest(profile_path),
            main_configuration_sha256=digest(self.root/'configurations/retina_platform_v1.json'),
            parameter_classes='Exact phenotype/source-profile and categorical subtype/parent/eligibility. Derived runtime parameters unresolved.',
            derived_runtime_binding_pending=True)
        with np.load(selection_path,allow_pickle=False) as selected:
            self.prepare(graph,selected,profiles)
            self.populations(graph,selected,profiles)
        gap={}
        for route in graph['routes']:
            total=self.route(route,graph,profiles)
            if route['kind']=='gap':
                gap[(route['post'],route['pre'])]=total
        self.stage='VERIFYING_GAP_CONSERVATION'
        names=['AII_AC','midget_BC_ON','diffuse_BC_ON','S_ON_BC']
        adjacency=sparse.bmat([[gap.get((post,pre),sparse.csr_matrix((len(self.groups[post]['counts']),len(self.groups[pre]['counts'])))) for pre in names] for post in names],format='csr')
        lap=gap_laplacian(adjacency)
        (self.work/'gap').mkdir()
        save_csr(self.work/'gap/total_adjacency_S.npz',adjacency)
        save_csr(self.work/'gap/laplacian_S.npz',lap)
        self.metadata['gap']=dict(population_order=names,group_counts=[len(self.groups[n]['counts']) for n in names],
            laplacian_entries=lap.nnz,max_row_sum_abs=float(np.max(np.abs(np.asarray(lap.sum(axis=1)).ravel()),initial=0.)),
            symmetry_verified=True,interpretation='P.T L P; internal-group currents cancel, reciprocal routes counted once per directed matrix entry')
        self.inputs()
        counts={name:len(g['counts']) for name,g in self.groups.items()}
        contact={r['contact']['component']:r['contact']['groups'] for r in self.routes if 'contact' in r}
        cell_states=7*counts['cones']+3*counts['HC']+7*sum(counts[n] for n in BC)+sum(counts[n] for n in AC)+sum(counts[n] for n in RGC)+7*counts['rods']+3*counts['rod_BC']+3*counts['AII_AC']
        contact_states=contact['individual_rod_RBC_contacts']+2*sum(contact[n] for n in ('RBC_AII_AMPA','AII_FMB_glycine','AII_DB1_glycine'))
        edges=sum(r['group_pair_entries'] for r in self.routes if r['kind']!='gap')+lap.nnz
        self.metadata['budget_audit']=self.budget_audit(counts,contact,cell_states,contact_states,edges)
        self.metadata.update(populations={n:g['description'] for n,g in self.groups.items()},routes=self.routes,
            source_fine_neurons=sum(len(g['rows']) for g in self.groups.values()),
            status='STATIC_GRAPH_COMPLETE_RUNTIME_UNBOUND',wall_seconds=time.perf_counter()-self.started)
        self.check(force=True)
        self.metadata['resources']=self.guard.last
        self.complete_metadata()
        if 'torch' in sys.modules:
            raise ContractError('Static builder unexpectedly imported Torch')
        self.metadata['torch_imported']=False
        self.metadata['files']={p.relative_to(self.work).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
            for p in sorted(self.work.rglob('*')) if p.is_file() and p.name!='status.json'}
        payload=json.dumps(self.metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        json_write(self.work/'manifest.json',dict(self.metadata,manifest_sha256=hashlib.sha256(payload).hexdigest()))
        self.verify_output()
        self.stage='STATIC_GRAPH_COMPLETE_RUNTIME_UNBOUND'
        self.check(force=True)
        os.replace(self.work,self.output)
        print(json.dumps(dict(output=str(self.output),**self.metadata['budget_audit'],wall_seconds=self.metadata['wall_seconds']),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',default='configurations/spatial_group_graph_foveal_v1.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    builder=Builder(ROOT,ROOT/args.configuration,ROOT/args.output)
    try:
        builder.build()
    except Exception as exc:
        json_write(builder.work/'failure.json',dict(stage=builder.stage,type=type(exc).__name__,message=str(exc),
            model_forward_started=False,elapsed_seconds=time.perf_counter()-builder.started,resources=builder.guard.last))
        raise


if __name__=='__main__':
    main()
