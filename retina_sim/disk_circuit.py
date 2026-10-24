"""One frozen logical disk graph and exact recording-view dependencies."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from .contracts import ContractError,Quantity
from .disk_graph_storage import FrozenSparseRoute,file_sha,save_json
from .disk_tables import FrozenDiskTable
from .disk_population import FixedDiskPopulation
from .disk_geometry import DiskAtlasROI

SCHEMA='frozen_global_disk_circuit_v1'
OUTPUTS=('midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON')


def explicit_owner_indices(indices, positions, roi):
    """Validate already existing output rows; never synthesize or move cells."""
    raw=np.asarray(indices)
    if raw.ndim!=1 or (raw.size and raw.dtype.kind not in 'iu'):
        raise ContractError('Explicit owner rows must be a one-dimensional integer list')
    rows=raw.astype(np.int64)
    if len(np.unique(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=len(positions)):
        raise ContractError('Explicit owner rows duplicate or exceed the fixed population')
    if len(rows) and not np.all(roi.owner_mask(positions[rows])):
        raise ContractError('Explicit owner cell lies outside the declared recording ROI')
    return np.sort(rows)

def freeze_disk_circuit(root,report,directory,*,atlas_profile='retina_atlas_disk3mm_v1',
                        configuration_path='configurations/atlas_disk3mm_v1.json'):
    root=Path(root).resolve();report=root/report;directory=root/directory;directory.mkdir(exist_ok=True,parents=True)
    read=lambda name:json.loads((report/(name+'.json')).read_bytes())
    outer=read('outer_tables');native=read('outer_connections');inner=read('inner_tables');aii=read('aii_bc_connections');primary=read('primary_graph')
    populations=read('population_build')['fixed_populations'];tables={}
    contact_geometry=read('contact_geometry')['contacts']
    expected_contacts={'individual_rod_RBC_contacts','RBC_AII_AMPA','AII_FMB_glycine','AII_DB1_glycine'}
    if set(contact_geometry)!=expected_contacts:raise ContractError('Global physical contact geometry is incomplete; no assembly-ready binding')
    from .disk_contact_geometry import FixedContactGeometry
    for descriptor in contact_geometry.values():
        geometry=FixedContactGeometry(root,descriptor['binding']);route=FrozenSparseRoute(root,descriptor['route'])
        if geometry.record['recipe']['route']!=descriptor['route'] or geometry.record['count']!=route.record['edges']:
            raise ContractError('Global contact geometry belongs to a different frozen route')
    for name,binding in dict(inner['tables'],cones=outer['cones'],HC=outer['HC']).items():
        table=FrozenDiskTable(root,binding);tables[name]={'kind':'typed','binding':binding,'count':table.count}
    for name in ('rods','rod_BC','AII_AC'):
        table=FixedDiskPopulation(root,populations[name]);tables[name]={'kind':'disk_population','binding':table.binding,'count':len(table.positions)}
    routes=[]
    def put(binding,kind):
        route=FrozenSparseRoute(root,binding);r=route.record['recipe'];pre=r['pre_population'];post=r['post_population']
        if r['pre_count']!=tables[pre]['count'] or r['post_count']!=tables[post]['count']:raise ContractError('Global route population ordering/count differs')
        routes.append({'pre':pre,'post':post,'kind':kind,'binding':binding,'route_id':r['route_id']})
    for binding in inner['routes'].values():put(binding,'inner_chemical')
    for name in ('rod_RBC','RBC_AII_AMPA'):put(primary['routes'][name],'primary_chemical')
    for binding in aii['chemical_routes'].values():put(binding,'primary_chemical')
    put(native['native_A'],'native');put(native['native_B'],'native')
    put(primary['routes']['AII_AII_bidirectional_gap'],'gap')
    for record in aii['gap_routes'].values():
        put(record['forward'],'gap');put(record['reverse'],'gap')
    # Electrical connected components are exact dependency accelerators. The
    # sparse conductance matrices remain the source of every physical gap.
    names=('AII_AC','midget_BC_ON','diffuse_BC_ON','S_ON_BC');offsets={};next_row=0
    for name in names:offsets[name]=next_row;next_row+=tables[name]['count']
    row=[];column=[]
    for descriptor in routes:
        if descriptor['kind']!='gap':continue
        route=FrozenSparseRoute(root,descriptor['binding'])
        for block in route.record['blocks']:
            matrix,_,_=route.block(block);matrix=matrix.tocoo()
            row.extend((matrix.row+block['start']+offsets[descriptor['post']]).tolist())
            column.extend((matrix.col+offsets[descriptor['pre']]).tolist())
    gap_graph=sparse.csr_matrix((np.ones(len(row),dtype=np.uint8),(row,column)),shape=(next_row,next_row))
    number,labels=connected_components(gap_graph,directed=False,return_labels=True)
    path=directory/'electrical_component_labels.npy'
    if path.exists():
        if not np.array_equal(np.load(path,allow_pickle=False),labels):raise ContractError('Frozen electrical closure labels differ')
    else:np.save(path,labels,allow_pickle=False)
    value={'schema':SCHEMA,'owner_disk_radius_mm':3.,'reference_id':outer['reference_id'],'tables':tables,'routes':routes,
        'native_components':native['native_component_labels'],
        'electrical_components':{'path':path.relative_to(root).as_posix(),'sha256':file_sha(path),'offsets':offsets,
            'counts':{name:tables[name]['count'] for name in names},'component_count':int(number)},
        'source_profiles':{'scope':'configurations/retina_disk3mm_scope_v1.json','cone_atlas':atlas_profile,
            'dynamics':'effective_dynamics_v1','input_plan':read('input_plan')},
        'contact_geometry':contact_geometry,
        'output_populations':list(OUTPUTS),'native_owner_zero_feedback_cones':native['owner_zero_feedback_cones'],
        'blocked_midget_candidates':inner['midget_blocked_candidates'],
        'regenerate_on_ROI':False,'crop_normalization':False,'exterior_feedback':False,
        'physiological_validation':False,'status':'GLOBAL_GEOMETRY_FROZEN_PENDING_EXECUTABLE_ASSEMBLY_AND_CAPTURE',
        'default_adopted':False}
    manifest=directory/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_bytes())!=value:raise ContractError('Frozen global circuit differs; retained')
    else:save_json(manifest,value)
    binding={'path':manifest.relative_to(root).as_posix(),'sha256':file_sha(manifest)}
    configuration={'schema_version':'1.0','profile_id':'atlas_isetbio_geometry_v1','storage_kind':'global_disk_graph_v1',
        'atlas_profile':atlas_profile,'global_graph':binding,'input_plan':read('input_plan'),
        'parameter_profile':'effective_dynamics_v1','owner_disk_radius_mm':3.,'halo_mm':.04,
        'budgets':{'max_dependency_nodes':2000000,'max_compile_bytes':4*1024**3,'minimum_available_bytes':8*1024**3},
        'status':'GLOBAL_GEOMETRY_FROZEN_PENDING_ASSEMBLY_AND_CAPTURE','default_adopted':False,
        'migration':{'status':'CANDIDATE_NOT_DEFAULT','canonical_target_outputs':list(OUTPUTS),'current_empty_owner_outputs':[],
            'ready_for_single_default_switch':False,'global_native_owner_zero_feedback_cones':native['owner_zero_feedback_cones']}}
    config_path=root/configuration_path
    if config_path.exists():
        if json.loads(config_path.read_bytes())!=configuration:raise ContractError('Existing disk circuit configuration differs; retained')
    else:save_json(config_path,configuration)
    return binding

class GlobalDiskCircuit:
    def __init__(self,root,binding):
        self.root=Path(root).resolve();self.binding=dict(binding);path=(self.root/binding['path']).resolve()
        if not path.is_relative_to(self.root) or file_sha(path)!=binding['sha256']:raise ContractError('Global circuit identity mismatch')
        self.record=json.loads(path.read_bytes());r=self.record
        if r['schema']!=SCHEMA or r['regenerate_on_ROI'] is not False or r['crop_normalization'] is not False or r['exterior_feedback'] is not False:
            raise ContractError('Unsupported global disk circuit policy')
        self._tables={};self._routes={};self._components={}
    def table(self,name):
        if name not in self._tables:
            record=self.record['tables'][name];kind=record['kind']
            self._tables[name]=(FrozenDiskTable if kind=='typed' else FixedDiskPopulation)(self.root,record['binding'])
        return self._tables[name]
    def route(self,descriptor):
        key=descriptor['binding']['sha256']
        if key not in self._routes:self._routes[key]=FrozenSparseRoute(self.root,descriptor['binding'])
        return self._routes[key]
    def _component_labels(self,name):
        if name not in self._components:
            record=self.record[name];path=(self.root/record['path']).resolve()
            if not path.is_relative_to(self.root) or file_sha(path)!=record['sha256']:raise ContractError('Frozen dependency labels changed')
            self._components[name]=np.load(path,mmap_mode='r',allow_pickle=False)
        return self._components[name]
    def _expand_components(self,selected,kind):
        record=self.record[kind];labels=self._component_labels(kind)
        if kind=='native_components':
            names=('HC','cones');offsets={'HC':0,'cones':self.record['tables']['HC']['count']}
            counts={name:self.record['tables'][name]['count'] for name in names}
        else:names=tuple(record['offsets']);offsets=record['offsets'];counts=record['counts']
        query=[np.asarray(sorted(selected.get(name,set())),dtype=np.int64)+offsets[name] for name in names if selected.get(name)]
        if not query:return False
        component_ids=np.unique(labels[np.concatenate(query)]);keep=np.isin(labels,component_ids);changed=False
        for name in names:
            rows=np.flatnonzero(keep[offsets[name]:offsets[name]+counts[name]])
            target=selected.setdefault(name,set());before=len(target);target.update(map(int,rows));changed|=len(target)>before
        return changed
    def owner_view(self,bounds,*,outputs=OUTPUTS,max_nodes=2000000,empty_chemical_input_policy='reject_empty_chemical_input_v1',owner_global_rows=None):
        from .finite_input_support import EMPTY_SUM_POLICY,REJECT_EMPTY_POLICY
        if empty_chemical_input_policy not in (EMPTY_SUM_POLICY,REJECT_EMPTY_POLICY):
            raise ContractError('Unknown frozen empty chemical input policy')
        roi=DiskAtlasROI(tuple(bounds),tuple(bounds),self.record['reference_id'],owner_radius_mm=3.,query_radius_mm=3.)
        if owner_global_rows is not None and (not isinstance(owner_global_rows,dict) or set(owner_global_rows)-set(outputs)):
            raise ContractError('Explicit owner selection must name only requested output populations')
        selected={};owner={};seen={};empty=[];blocked_owner=[]
        for name in outputs:
            if name not in self.record['output_populations']:raise ContractError('Unknown output population')
            table=self.table(name)
            indices=(table.query_indices(roi,owner=True) if owner_global_rows is None else
                     explicit_owner_indices(owner_global_rows.get(name,[]),table.coordinates['rf'],roi))
            owner[name]=indices;selected[name]=set(map(int,indices))
            if not len(indices):empty.append(name)
            blocked=self.record.get('blocked_midget_candidates',{}).get(name,{}).get('parent_BC_rows',[])
            if blocked and owner_global_rows is None:
                pre_name=name.replace('RGC','BC');pre=self.table(pre_name);candidate=np.asarray(blocked,dtype=np.int64)
                rows=candidate[roi.owner_mask(pre.coordinates['rf'][candidate])]
                blocked_owner.extend({'route':'unresolved_'+name,'post_global_parent_BC_row':int(index),
                    'status':'UNRESOLVED_ACTUAL_RF_CENTER_FROM_BLOCKED_SOURCE_POOL'} for index in rows)
        if len(empty)==len(outputs):
            if blocked_owner:raise ContractError('UNRESOLVED_RGC_OUTPUT_COVERAGE: blocked candidates retained; no replacement RF cell')
            raise ContractError('EMPTY_OUTPUT_SAMPLE: no forced population or origin cell')
        changed=True
        while changed:
            changed=self._expand_components(selected,'native_components')|self._expand_components(selected,'electrical_components')
            for descriptor in self.record['routes']:
                if descriptor['kind'] in ('native','gap'):continue
                key=descriptor['binding']['sha256'];pending=selected.get(descriptor['post'],set())-seen.setdefault(key,set())
                if not pending:continue
                seen[key].update(pending);parents=self.route(descriptor).dependency_rows(sorted(pending))
                target=selected.setdefault(descriptor['pre'],set());before=len(target);target.update(map(int,parents));changed|=len(target)>before
            if sum(map(len,selected.values()))>max_nodes:raise ContractError('Exact global dependency closure exceeds execution node budget; no crop or thinning applied')
        selected={name:np.asarray(sorted(indices),dtype=np.int64) for name,indices in selected.items() if indices}
        unsupported=list(blocked_owner);empty_chemical=[]
        for descriptor in self.record['routes']:
            post=selected.get(descriptor['post'])
            if post is not None and descriptor['kind']=='primary_chemical':
                rows,matrix,_,_=self.route(descriptor).selected_rows(post)
                for index in rows[np.diff(matrix.indptr)==0]:
                    row={'route':descriptor['route_id'],'post_global_row':int(index),
                        'status':'EMPTY_ACTUAL_PRIMARY_CHEMICAL_ROW'}
                    if empty_chemical_input_policy==EMPTY_SUM_POLICY and descriptor['route_id']=='RBC_AII_AMPA':
                        empty_chemical.append(dict(row,policy=EMPTY_SUM_POLICY,evidence_class='project_approximation'))
                    else:unsupported.append(row)
            if post is None or descriptor['kind'] not in ('inner_chemical','native'):continue
            _,_,audits,statuses=self.route(descriptor).selected_rows(post)
            unsupported.extend({'route':descriptor['route_id'],'post_global_row':int(index),'status':status,
                'missing_degree':audit.get('missing_degree')} for index,audit,status in zip(post,audits,statuses) if not status.startswith('READY'))
        return {'schema':'exact_global_disk_dependency_view_v1','graph':self.binding,'owner_bounds_mm':tuple(bounds),'owner_disk_radius_mm':3.,
            'selected_rows':selected,'owner_output_rows':owner,'empty_output_samples':tuple(empty),'unsupported_rows':unsupported,
            'runtime_ready':not unsupported,'geometry_regenerated':False,'weights_renormalized':False,
            'empty_chemical_input_policy':empty_chemical_input_policy,'declared_empty_chemical_sums':empty_chemical,
            'owner_selection_mode':'all_actual_cells_in_rectangle' if owner_global_rows is None else 'explicit_existing_global_rows',
            'requested_owner_global_rows':None if owner_global_rows is None else {name:rows.tolist() for name,rows in owner.items()},
            'global_node_count':sum(map(len,selected.values())),'physical_calibration':False}
    def table_record(self,name,indices):
        table=self.table(name);canonical=table.rows(indices)
        if self.record['tables'][name]['kind']=='disk_population':
            xy=table.positions[indices];coordinates=dict(canonical.coordinates)
            if name in ('rod_BC','AII_AC'):
                coordinates.update(ipl_preterminal=Quantity(xy,'mm','terminal_position','existing_nominal_physical_chart_project_transfer','IPL'),
                    ipl_post_arbor=Quantity(xy,'mm','dendritic_position','existing_nominal_physical_chart_project_transfer','IPL'))
                from .atlas import CanonicalCellTable
                canonical=CanonicalCellTable(canonical.atlas_id,canonical.atlas_version,canonical.source_version,canonical.population,
                    canonical.source_keys,coordinates,canonical.reference_id)
        return canonical.record()
