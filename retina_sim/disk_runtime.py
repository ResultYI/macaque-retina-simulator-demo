"""Bind an exact frozen global view to the existing factories and solver."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
from .contracts import ContractError,Quantity,canonical_hash
from .disk_circuit import GlobalDiskCircuit,OUTPUTS
from .disk_graph_storage import FrozenSparseRoute,array_identity
from .disk_contact_geometry import FixedContactGeometry,global_contact_rows

class DiskRuntimeRecord(dict):
    def __init__(self,value,identity):
        super().__init__(value);self.semantic_hash=identity
        self.population_counts={'cones':len(value['cones']['global_cell_ids']),**{
            'inner_tables/'+name:len(table['global_cell_ids']) for name,table in value['inner_tables'].items()}}

def _quantity(values,meaning,frame):return {'values':np.asarray(values).tolist(),'unit':'mm','meaning':meaning,
    'frame':frame,'evidence_tag':'existing_effective_coordinate_rule_project_approximation'}

def runtime_record(root,circuit,view):
    from .source_atlas import SourceDensityAtlas
    from .atlas_s_partition import EffectiveMulattaSPartition
    root=Path(root).resolve();selected=view['selected_rows'];records={name:circuit.table_record(name,rows) for name,rows in selected.items()}
    if not view['runtime_ready']:
        raise ContractError('DISK_VIEW_SOURCE_OR_GEOMETRY_UNSUPPORTED: '+str(view['unsupported_rows'][:5]))
    for name in ('cones','HC','rods','rod_BC','AII_AC'):
        if name not in records:raise ContractError('DISK_VIEW_PRIMARY_PATH_UNSUPPORTED: no '+name+' states in exact source closure; no replacement driver')
    atlas_profile=circuit.record['source_profiles']['cone_atlas']
    atlas=SourceDensityAtlas(root,profile_id=atlas_profile);s=EffectiveMulattaSPartition(atlas)
    hconfig=json.loads((root/'configurations/atlas_hc_disk3mm_v1.json').read_bytes())
    cone=circuit.table('cones');hc=circuit.table('HC')
    if cone.metadata['source_version']!=atlas.mosaic_namespace or cone.record['source_bindings']['S_profile_sha256']!=s.profile_sha256:
        raise ContractError('Global cone/S namespace differs from selected frozen source profile')
    if hc.record['source_bindings']['master_sha256']!=hconfig['expected_asset_sha256']:
        raise ContractError('Global HC table belongs to a different joint master')
    bounds=tuple(view['owner_bounds_mm']);inner={};parents={}
    for name in circuit.record['output_populations']:
        if name not in records:
            # Empty owner populations retain a full source binding, without a
            # synthetic empty-coordinate substitute for actual source anatomy.
            records[name]=circuit.table_record(name,np.empty(0,dtype=np.int64))
    for name,descriptor in circuit.record['tables'].items():
        if descriptor['kind']!='typed' or name in ('cones','HC'):continue
        table=records.get(name)
        if table is None:table=circuit.table_record(name,np.empty(0,dtype=np.int64))
        if name=='widefield_AC':
            xy=np.asarray(table['coordinates']['ipl_preterminal']['values']).reshape(-1,2)
            table=dict(table,global_eccentricity={'values':np.linalg.norm(xy,axis=1).tolist(),'unit':'mm','frame':'anatomical_retina',
                'meaning':'global_eccentricity','registration':'existing registered physical IPL point chart; no RF or soma fabricated'})
        inner[name]=table
        parents[name]=list(table['source_keys']) if name.startswith('midget') or name=='S_ON_BC' or name=='small_bistratified_S_ON' else []
    contacts={}
    for descriptor in circuit.record['routes']:
        if descriptor['kind']!='inner_chemical':continue
        pre,post=descriptor['pre'],descriptor['post'];pre_record=records['cones'] if pre=='cones' else inner[pre];post_record=inner[post]
        pre_rows=selected.get(pre,np.empty(0,dtype=np.int64));post_rows=selected.get(post,np.empty(0,dtype=np.int64))
        _,_,matrix,rows,statuses=circuit.route(descriptor).view(post_rows,pre_rows)
        edge=matrix.tocoo();pre_ids=pre_record['global_cell_ids'];post_ids=post_record['global_cell_ids']
        pre_points=pre_record['coordinates']['opl_pedicle'] if pre=='cones' else pre_record['coordinates']['ipl_preterminal']
        frame='OPL' if pre=='cones' else 'IPL'
        post_values=(post_record['coordinates']['ipl_preterminal'] if frame=='OPL' else post_record['coordinates']['ipl_post_arbor'])['values']
        contacts[pre+'->'+post]={'pre_ids':pre_ids,'post_ids':post_ids,'frame':frame,'row_status':statuses,'rows':rows,
            'edges':[{'pre_id':pre_ids[i],'post_id':post_ids[j],'weight':float(weight),'unit':'dimensionless'} for j,i,weight in zip(edge.row,edge.col,edge.data)],
            'pre_positions':pre_points,'post_positions':_quantity(post_values,'dendritic_position',frame),
            'evidence_class':'project_approximation','boundary_mode':'frozen_global_finite_circuit','query_support_complete':'exact finite graph dependencies; no exterior closure claim'}
    masters={name:circuit.table(name).record for name in ('rods','rod_BC','AII_AC')}
    # Native support is the original global cone query, independent of owner box.
    radius=cone.record['source_bindings']['fixed_RF_query_radius_mm'];query=[-radius,radius,-radius,radius]
    global_ids=tuple(gid for name,table in records.items() if name not in OUTPUTS or name in selected for gid in table['global_cell_ids'])
    value={'atlas':{'atlas_id':atlas.config['atlas_id'],'atlas_version':atlas.config['atlas_version'],'reference_id':atlas.config['reference_id'],
            'profile_sha256':atlas.profile_hash,'mosaic_namespace':atlas.mosaic_namespace,'owner_domain':list(bounds),'query_support':query,
            'cone_master':dict(atlas.config['cone_master']),'global_graph':circuit.binding,'owner_disk_radius_mm':3.,'density_scale':1.,
            'boundary_mode':'one_finite_global_canonical_network','recurrent_exterior_closure':'NOT_CERTIFIED'},
        'cones':records['cones'],'cone_subtypes':[x.decode('ascii') for x in cone.extras['subtype'][selected['cones']]],
        'cone_type_provenance':{'profile_id':s.config['profile_id'],'profile_sha256':s.profile_sha256},
        'HC':{'table':records['HC'],'types':[x.decode('ascii') for x in hc.extras['subtype'][selected['HC']]],
            'OPL_anchor':_quantity(hc.coordinates['soma'][selected['HC']],'arbor_anchor_position','OPL'),
            'metadata':{'file_sha256':hconfig['expected_asset_sha256'],'global_table':hc.binding,'source_evidence':'approved effective geometry; no calibrated HC kinetics claim'}},
        'HC_candidate_edges':(), 'inner_tables':inner,'inner_parents':parents,'inner_contacts':contacts,
        'inner_metadata':{'global_graph':circuit.binding,'rule':'unchanged typed effective sites and weights frozen globally','soma':'ABSENT outside separately supplied source views'},
        'rod_inner':{'tables':{name:records[name] for name in ('rods','rod_BC','AII_AC')},'contacts':{},
            'metadata':{'source_status':'approved_source_plus_project_completion','rod_source_error':None,'masters':masters,
                'RBC_to_rod_density_ratio':1/18,'AII_density_cells_per_mm2':5000,'geometry_on_ROI':'read selected rows of fixed global masters'}},
        'population_revision':{'masters':masters,'global_graph':circuit.binding,'independent_parasol':True,'global_geometry_radius_mm':3.,'physiological_validation':False,
            'execution_profile_id':'atlas_disk3mm_dependency_v1','scope':'Exact finite dependency view of one approved global3mm geometry; existing effective dynamics, missing source rows retained'},
        'summary':{'S_partition_profile':s.config['profile_id'],'HC_query_status':'supported','zero_owner_candidate_rows':0,
            'scope':'selected exact dependencies of one global disk; unsupported rows checked before assembly'},
        'dependency_query':{'requested_RF_query_mm':query,'expanded_RF_query_mm':query,'owner_disk_radius_mm':3.,
            'global_dependency_view':True,'geometry_regenerated':False,'weights_renormalized':False,'recurrent_exterior_closure':'NOT_CERTIFIED'},
        'required_global_dependency_ids':global_ids,'isetbio_geometry':{'master':dict(atlas.config['cone_master']),'atlas_profile':atlas_profile,
            'global_graph':circuit.binding,'physiology_parameters_changed':False}}
    identity=canonical_hash({'graph':circuit.binding,'owner':bounds,'selected_rows':{name:array_identity(rows) for name,rows in selected.items()}})
    context=SimpleNamespace(atlas=atlas,hc_master=SimpleNamespace(config=hconfig),s_partition=s)
    return DiskRuntimeRecord(value,identity),context

class DiskSpatialAsset:
    allow_absent_contact_targets=True
    def __init__(self,circuit,view,record,*,max_compile_bytes,empty_chemical_input_policy='reject_empty_chemical_input_v1'):
        self.circuit=circuit;self.view=view;self.source=record;self.max_compile_bytes=max_compile_bytes
        self.empty_chemical_input_policy=empty_chemical_input_policy
        self.binding={'global_graph':circuit.binding,'view_sha256':record.semantic_hash};self._pools={}
        self._ids={name:tuple(table['global_cell_ids']) for name,table in dict(record['inner_tables'],cones=record['cones'],HC=record['HC']['table'],**record['rod_inner']['tables']).items()}
    def _descriptor(self,name):
        values=[value for value in self.circuit.record['routes'] if value['route_id']==name]
        if len(values)!=1:raise ContractError('Unknown/ambiguous frozen route: '+name)
        return values[0]
    def native_dependency_ids(self,cone_needed):
        if not set(cone_needed)<=set(self._ids['cones']):raise ContractError('Native view omitted required cone states')
        return set(self._ids['cones']),set(self._ids['HC'])
    def native_matrices(self,cones,hcs):
        if tuple(cones.cells.cell_id)!=self._ids['cones'] or tuple(hcs.cells.cell_id)!=self._ids['HC']:
            raise ContractError('Native fixed view identity/order differs')
        A,B=self._load_native()
        if np.any(np.asarray(A.sum(1)).ravel()==0) or np.any(np.asarray(B.sum(1)).ravel()==0):raise ContractError('Native global unit-DC rows unsupported; no replacement feedback')
        return A,B
    def _load_native(self):
        if not hasattr(self,'_native'):
            c=self.view['selected_rows']['cones'];h=self.view['selected_rows']['HC']
            A=self.circuit.route(self._descriptor('native_cone_to_HC_A')).view(h,c)[2]
            B=self.circuit.route(self._descriptor('native_HC_to_cone_B')).view(c,h)[2]
            self._native=(A,B)
        return self._native
    def native_contact_summary(self,cones,hcs):
        return {'storage':'frozen global CSR; exact selected closure','edge_count':self._load_native()[0].nnz,
            'global_graph':self.circuit.binding,'crop_renormalization':False}
    def _contact(self,cid):
        if cid not in self._pools:
            descriptor=self.circuit.record['contact_geometry'][cid];route=FrozenSparseRoute(self.circuit.root,descriptor['route'])
            pre=descriptor['pre'];post=descriptor['post'];a=self.view['selected_rows'].get(pre,np.empty(0,dtype=np.int64));b=self.view['selected_rows'].get(post,np.empty(0,dtype=np.int64))
            _,_,matrix,rows,_=route.view(b,a);geometry=FixedContactGeometry(self.circuit.root,descriptor['binding']);entries=global_contact_rows(route,b)
            if len(entries)!=matrix.nnz:raise ContractError('Selected frozen contact entry order differs')
            self._pools[cid]=(descriptor,matrix,rows,geometry,entries)
        return self._pools[cid]
    def primary_pool(self,name,pre_ids,post_ids):
        cid='individual_rod_RBC_contacts' if name=='rod_RBC' else name
        d,matrix,rows,geometry,entries=self._contact(cid)
        expected_pre=tuple(d['pre_ID_prefix']+gid for gid in self._ids[d['pre']])
        if tuple(pre_ids)!=expected_pre or tuple(post_ids)!=self._ids[d['post']]:raise ContractError('Frozen primary contact population ordering differs: '+name)
        rows=[dict(row,post_global_row=row['post_index'],post_index=j) for j,row in enumerate(rows)]
        return matrix,rows,None if geometry.overlap is None else np.asarray(geometry.overlap[entries])
    def empty_chemical_declaration(self,name,pre,post,matrix):
        from .finite_input_support import EMPTY_SUM_POLICY
        if self.empty_chemical_input_policy!=EMPTY_SUM_POLICY or name!='RBC_AII_AMPA':return None
        if (pre,post)!=('rod_BC','AII_AC') or matrix.shape!=(len(self._ids[post]),len(self._ids[pre])):
            raise ContractError('Frozen empty chemical declaration route/order differs')
        empty=np.flatnonzero(np.diff(matrix.indptr)==0)
        if not len(empty):return None
        return {'policy':EMPTY_SUM_POLICY,'evidence_class':'project_approximation','source_graph':self.circuit.binding,
            'source_route':self._descriptor(name)['binding'],'cell_ids':[self._ids[post][i] for i in empty]}
    def contact_record(self,cid):
        d,matrix,_,geometry,entries=self._contact(cid);coo=matrix.tocoo()
        pre=tuple(d['pre_ID_prefix']+gid for gid in self._ids[d['pre']]);post=self._ids[d['post']]
        keys=[pre[i]+'>'+post[j] for j,i in zip(coo.row,coo.col)];xy=geometry.positions[entries]
        frame=geometry.record['recipe']['coordinate_frame']
        coordinates=({'ipl_preterminal':_quantity(xy,'terminal_position','IPL'),
            'ipl_post_arbor':_quantity(xy,'dendritic_position','IPL')} if frame=='IPL' else {})
        return {'population':cid,'global_cell_ids':[cid+':'+hashlib.sha256(key.encode()).hexdigest()[:32] for key in keys],
            'source_keys':keys,'coordinates':coordinates,
            'physical_contact_anchor':_quantity(xy,'contact_position',frame),
            'global_eccentricity':{'values':np.linalg.norm(xy,axis=1).tolist(),'unit':'mm','frame':'anatomical_retina',
                'meaning':'global_eccentricity','registration':'registered physical contact chart; no neuronal soma or RF assigned'},
            'frozen_contact_geometry':d['binding']}
    def gaps(self):
        from .assembly import GapJunction
        from .mechanism_contracts import EvidenceValue
        from examples.full_target_network import PROJECT_SOURCE
        result=[]
        for descriptor in self.circuit.record['routes']:
            if descriptor['kind']!='gap' or descriptor['route_id'].endswith('_reverse'):continue
            pre,post=descriptor['pre'],descriptor['post'];a=self.view['selected_rows'].get(pre,np.empty(0,dtype=np.int64));b=self.view['selected_rows'].get(post,np.empty(0,dtype=np.int64))
            if not len(a) or not len(b):continue
            matrix=self.circuit.route(descriptor).view(b,a)[2].tocoo();keep=np.ones(matrix.nnz,dtype=bool)
            if pre==post:keep=matrix.row<matrix.col
            if not np.any(keep):continue
            left=tuple(self._ids[pre][i] for i in matrix.col[keep]);right=tuple(self._ids[post][j] for j in matrix.row[keep]);g=matrix.data[keep]
            cid='AII_AII_homologous' if pre==post else descriptor['route_id']
            evidence=EvidenceValue(cid,Quantity(g,'S','gap_junction_conductance','project_approximation'),'project_approximation',PROJECT_SOURCE,
                'Unchanged reciprocal global terminal gaps, frozen conductance/overlap',{'support_mm':.04,'radius_mm':.03},'Uncalibrated project conductance; no HC gap')
            result.append(GapJunction(pre,left,post,right,evidence))
        return result
    def validate_compiled(self,compiled):
        aliases={'rods':'rod'}
        for name,indices in self.view['selected_rows'].items():
            pid=aliases.get(name,name)
            if pid not in compiled.populations or tuple(compiled.populations[pid].cells.cell_id)!=self._ids[name]:
                raise ContractError('Compiled model lost selected global state identities: '+name)
        expected=self._load_native();native=compiled.components['outer_native'].native_model
        for name,matrix in zip(('A','B'),expected):
            actual=getattr(native,name)
            if actual.shape!=matrix.shape or any(not np.array_equal(getattr(actual,key),getattr(matrix,key)) for key in ('data','indices','indptr')):
                raise ContractError('Compiled native weights differ from frozen global pools: '+name)
        for cid,descriptor in self.circuit.record['contact_geometry'].items():
            if descriptor['post'] not in self.view['selected_rows']:continue
            record=self.contact_record(cid)
            if record['global_cell_ids'] and (cid not in compiled.populations or tuple(compiled.populations[cid].cells.cell_id)!=tuple(record['global_cell_ids'])):
                raise ContractError('Compiled contact identities differ from frozen global entries: '+cid)
            if record['global_cell_ids']:
                pop=compiled.populations[cid];anchor=record['physical_contact_anchor'];actual=pop.dendritic_positions
                if actual is None or actual.unit!=anchor['unit'] or actual.coordinate_frame!=anchor['frame'] or not np.array_equal(actual.values,np.asarray(anchor['values'])):
                    raise ContractError('Compiled contact anchor/frame differs from frozen global entries: '+cid)
                terminal=pop.terminal_positions
                if anchor['frame']=='OPL' and terminal is not None:
                    raise ContractError('OPL receptor contact must not acquire an invented IPL terminal: '+cid)
                if anchor['frame']=='IPL' and (terminal is None or terminal.coordinate_frame!='IPL' or not np.array_equal(terminal.values,np.asarray(anchor['values']))):
                    raise ContractError('Compiled IPL contact terminal differs from frozen global entries: '+cid)
        return {'global_graph':self.circuit.binding,'view':self.binding,'selected_nodes':self.view['global_node_count'],
            'crop_renormalization':False,'geometry_regenerated':False,'finite_global_dependencies_complete':True,
            'exterior_recurrent_closure':'NOT_CERTIFIED','physiological_validation':False}


def disk_network(root,configuration,*,duration,light_context,owner_bounds_mm,halo_mm,max_spatial_cells,seed,
                 rod_reference_policy=None):
    """The same mechanism factories/time ownership on a frozen global view."""
    from .atlas_runtime import atlas_midget_network,q
    from .atlas_primary_rod import attach_primary_rod
    from .geometry_identity import compiled_geometry_identity
    from .disk_geometry import DiskAtlasROI
    from .background_checkpoint import MemoryGuard
    if halo_mm!=.04 or configuration.get('parameter_profile')!='effective_dynamics_v1':
        raise ContractError('Global disk retains its named existing40um/effective-dynamics configuration')
    circuit=GlobalDiskCircuit(root,configuration['global_graph'])
    atlas_profile=circuit.record['source_profiles']['cone_atlas']
    if configuration.get('atlas_profile')!=atlas_profile:
        raise ContractError('Selected circuit cone atlas differs from frozen graph; no species-route fallback')
    policy=configuration.get('empty_chemical_input_policy','reject_empty_chemical_input_v1')
    owner_options=({'owner_global_rows':configuration['owner_global_rows']}
                   if 'owner_global_rows' in configuration else {})
    view=circuit.owner_view(owner_bounds_mm,max_nodes=configuration['budgets']['max_dependency_nodes'],
        empty_chemical_input_policy=policy,**owner_options)
    if max(len(rows) for rows in view['selected_rows'].values())>max_spatial_cells:
        raise ContractError('Exact global view exceeds requested per-population budget; no thinning or crop normalization')
    guard=MemoryGuard(min_available_bytes=configuration['budgets']['minimum_available_bytes'])
    if not guard.check():raise ContractError('Insufficient available RAM for the unchanged global dependency view')
    record,context=runtime_record(root,circuit,view)
    spatial=DiskSpatialAsset(circuit,view,record,max_compile_bytes=configuration['budgets']['max_compile_bytes'],empty_chemical_input_policy=policy)
    compiled,channels,audit=atlas_midget_network(root,duration=duration,light_context=light_context,owner_bounds_mm=owner_bounds_mm,
        halo_mm=halo_mm,max_spatial_cells=max_spatial_cells,seed=seed,connection_record=record,atlas_profile_id=atlas_profile,
        allow_empty_owner_outputs=True,spatial_asset=spatial,_atlas_context=context,max_compile_bytes=spatial.max_compile_bytes)
    compiled,channels,audit=attach_primary_rod(Path(root),compiled,channels,audit,record,light_context,
        spatial_asset=spatial,rod_reference_policy=rod_reference_policy)
    proof=spatial.validate_compiled(compiled);roi=DiskAtlasROI(tuple(owner_bounds_mm),tuple(owner_bounds_mm),circuit.record['reference_id'])
    for name,pop in compiled.populations.items():
        rf=pop.receptive_field_positions
        audit['population_owner_mask'][name]=([False]*len(pop.cells.cell_id) if rf is None else roi.owner_mask(rf.values).tolist())
    compiled.global_disk_circuit=circuit;compiled.global_disk_view=view;compiled.global_spatial_asset=spatial
    compiled.compile_resource_budget_bytes=spatial.max_compile_bytes
    compiled.frozen_typed_geometry_binding=spatial.binding
    compiled.input_geometry_binding_required=True
    compiled.geometry_source_binding={'execution_profile':'atlas_isetbio_geometry_v1','storage_kind':'global_disk_graph_v1',
        'global_graph':circuit.binding,'exact_view':spatial.binding,'input_plan':configuration['input_plan']}
    if owner_options:
        compiled.geometry_source_binding['owner_global_rows']=configuration['owner_global_rows']
    radius=circuit.table('rods').radius
    compiled.typed_view_domain={'reference_id':circuit.record['reference_id'],'owner_domain_mm':tuple(owner_bounds_mm),
        'query_domain_mm':(-radius,radius,-radius,radius),'empty_owner_outputs':tuple(view['empty_output_samples']),'owner_disk_radius_mm':3.}
    audit['atlas_runtime'].update(frozen_typed_geometry=proof,graph_binding=circuit.binding,
        compiled_geometry_identity=compiled_geometry_identity(compiled),global_geometry_radius_mm=3.,
        geometry_or_edge_resampled=False,full_20_layer_network_claim=False)
    if owner_options:audit['atlas_runtime']['owner_output_global_rows']=configuration['owner_global_rows']
    audit['execution_profile']=dict(audit['execution_profile'],profile_id='atlas_isetbio_geometry_v1',storage_kind='global_disk_graph_v1',
        scope='Exact finite dependency view of the approved global3mm disk; effective dynamics and declared unresolved source gaps retained',
        execution_mode='frozen_global_disk_dependency_view',physiological_calibration=False)
    audit['execution_profile'].update(
        frozen_default_graph=circuit.binding,
        BC_route='Actual HistoryBipolarTerminal activation=dynamic terminal_release; generator_activation diagnostic only; original effective generator/terminal parameters retained',
        unsupported_paths='Actual frozen missing degrees/empty rows retained; exact selected rod/RBC/AII and independent parasol routes instantiated where graph-qualified',
        initialization='Declared construction background outer equilibrium; actual physical per-cone input background is prepared separately under the unchanged required numerical policy',
        native_operator_representation=circuit.record.get('native_operator_representation',{}).get('representation','finite_anatomical_candidate_contacts_with_explicit_missing_rows'))
    from .spatial import InputChannel
    nr=len(compiled.populations['rod'].cells.cell_id)
    channels['rod_light']=InputChannel(0,duration,lambda t:q(np.full(nr,light_context.rod_background_rstar_s),'R*/s','photoisomerization_rate'),
        'R*/s','photoisomerization_rate','zero_order_hold','project_approximation',{'source':'declared uniform rod background'})
    audit['resolved_execution_profile']=audit['execution_profile'];return compiled,channels,audit
