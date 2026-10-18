"""One local revised-population graph; actual cones/HC and contact laws retained."""
from pathlib import Path
import json,hashlib
import numpy as np
from numpy.polynomial.legendre import leggauss
from .contracts import Quantity,canonical_hash,ContractError
from .atlas import AtlasROI,CanonicalCellTable
from .atlas_connections import ConnectionAtlas
from .atlas_typed_completion import complete_typed_contacts
from .ipl_geometry import freeze_centroid_anchors,local_typed_sites,bounds_of
from .spatial_profiles import bounded_matrix
from .population_revision import ApprovedPackerRodDensity,ApprovedParasolRFDensity,APPROVAL
from .equal_mass_master import materialize_equal_mass_master


def canonical(rec):
    return CanonicalCellTable(rec['atlas_id'],rec['atlas_version'],rec['source_version'],rec['population'],tuple(rec['source_keys']),
        {k:Quantity(v['values'],v['unit'],v['meaning'],v['evidence_tag'],v['frame']) for k,v in rec['coordinates'].items()},rec['reference_id'])


def coordinate(values,meaning,frame):return Quantity(values,'mm',meaning,'project_approximation',frame)


def density_budget(field,bounds):
    z,w=leggauss(64);x0,x1,y0,y1=bounds
    xy=np.array(np.meshgrid(x0+(z+1)*(x1-x0)/2,y0+(z+1)*(y1-y0)/2,indexing='ij')).reshape(2,-1).T
    mass=float((field(xy).reshape(64,64)*w[:,None]*w[None,:]).sum()*(x1-x0)*(y1-y0)/4)
    n=int(np.floor(mass+.5));side=max(64,int(np.ceil(np.sqrt(8*n))))
    return {'bounds_mm':list(bounds),'mass_Gauss64_estimate':mass,'rounded_estimate':n,'quadrature_side':side,
        'estimated_bytes':side*side*80+n*96,'estimate_only':True}


def revise_local_graph(root,max_cells=60000,*,connection_record=None,atlas_profile_id=None,population_masters=None,rebuild_uncovered_masters=False):
    root=Path(root);execution=json.loads((root/'configurations/atlas_midget_research_v1.json').read_bytes());frozen=execution['frozen_default_graph']
    if connection_record is None:
        raw=(root/frozen['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=frozen['sha256']:raise ContractError('Original actual atlas graph identity mismatch')
        connection_record=json.loads(raw)
    atlas=ConnectionAtlas(root,atlas_profile_id=atlas_profile_id or execution['atlas_profile'],s_partition_profile=execution['S_partition_profile'])
    record=complete_typed_contacts(connection_record,atlas);tables=record['inner_tables'];contacts=record['inner_contacts']
    cones=canonical(record['cones']);owner=tuple(record['atlas']['owner_domain']);reference=cones.reference_id
    seed_bounds=tuple(record['dependency_query']['expanded_RF_query_mm']);output_bounds=tuple(record['dependency_query']['requested_RF_query_mm'])
    masters={};budget={};parasol_sites=[];source_tables={k:canonical(v) for k,v in tables.items()};rebuilt={}
    def master_for(name,bounds,field,seed,source):
        if population_masters is None:return materialize_equal_mass_master(root,name,bounds,field,seed,source,max_cells)
        master=population_masters.master(name,expected_source=source)
        try:population_masters.require_domain(name,bounds)
        except ContractError as exc:
            if not rebuild_uncovered_masters or 'MASTER_DOMAIN_INSUFFICIENT' not in str(exc):raise
            # Explicit offline migration only. The unchanged density provider
            # checks its own source domain; runtime ROI queries never opt in.
            replacement=materialize_equal_mass_master(root,name,bounds,field,seed,source,max_cells)
            rebuilt[name]={'previous':master.manifest(),'replacement':replacement.manifest(),
                'reason':'new cone-dependent finite support exceeds old fixed population asset',
                'density_and_sampling_laws_changed':False}
            return replacement
        return master
    def put(pre,post,pre_positions,post_positions,k,*,sigma=None,exact=False):
        pre_table=cones if pre=='cones' else source_tables[pre];post_table=source_tables[post]
        eligible=np.arange(len(pre_table.source_keys))
        upstream=None
        if pre in ('diffuse_BC_ON','diffuse_BC_OFF'):upstream=contacts['cones->'+pre]['row_status']
        elif pre in ('local_feedforward_AC','local_crossover_AC','widefield_AC'):
            upstream=contacts[('diffuse_BC_OFF' if pre=='local_crossover_AC' else 'diffuse_BC_ON')+'->'+pre]['row_status']
        if upstream is not None:eligible=np.array([i for i,s in enumerate(upstream) if s.startswith('READY')],dtype=int)
        matrix,rows=bounded_matrix(pre_positions.values[eligible],post_positions.values,k,support_mm=.04,gaussian_scale=sigma)
        edges=[];statuses=[]
        for j,row in enumerate(rows):
            indices=matrix.getrow(j).indices;status='READY_PROJECT_FINITE_POOL' if len(indices) else 'INCOMPLETE_NO_INPUT_WITHIN_40UM'
            if exact and row['missing_degree']:status='INCOMPLETE_REQUIRED_DEGREE'
            statuses.append(status)
            if status.startswith('READY'):
                edges.extend({'pre_id':pre_table.global_cell_ids[eligible[i]],'post_id':post_table.global_cell_ids[j],
                    'weight':float(matrix[j,i]),'unit':'dimensionless'} for i in indices)
            row.update(selection_budget=k,exact_budget_enforcement=exact,post_id=post_table.global_cell_ids[j])
        def qr(q):return {'values':q.values.tolist(),'unit':q.unit,'meaning':q.meaning,'frame':q.coordinate_frame,'evidence_tag':q.evidence_tag}
        contacts[pre+'->'+post]={'pre_ids':list(pre_table.global_cell_ids),'post_ids':list(post_table.global_cell_ids),
            'frame':post_positions.coordinate_frame,'row_status':statuses,'rows':rows,'edges':edges,
            'pre_positions':qr(pre_positions),'post_positions':qr(post_positions),'evidence_class':'project_approximation',
            'boundary_mode':'finite_supplied_candidates','query_support_complete':'finite master/query scope; exterior recurrent closure uncertified'}
    for index,polarity in enumerate(('ON','OFF')):
        name='parasol_RGC_'+polarity;field=ApprovedParasolRFDensity(polarity);budget[name]=density_budget(field,seed_bounds)
        if budget[name]['rounded_estimate']>max_cells:raise ContractError('Local parasol master exceeds budget; no density reduction')
        master=master_for(name,seed_bounds,field,20261014+index,field.metadata)
        masters[name]=master
        seed=master.crop(AtlasROI(owner,output_bounds,reference),max_cells);rf=seed.coordinate('rf')
        p=atlas.atlas.henle.apply(rf);pre='diffuse_BC_'+polarity;bc=source_tables[pre]
        ready=np.array([i for i,s in enumerate(contacts['cones->'+pre]['row_status']) if s.startswith('READY')])
        anchors,anchor_records=freeze_centroid_anchors(p.values,bc.coordinate('ipl_preterminal').values[ready],
            tuple(bc.global_cell_ids[i] for i in ready))
        table=CanonicalCellTable(seed.atlas_id,seed.atlas_version,seed.source_version,name,seed.source_keys,
            {'rf':rf,'ipl_post_arbor':coordinate(anchors,'dendritic_position','IPL')},reference)
        source_tables[name]=table;tables[name]=table.record();record['inner_parents'][name]=[]
        parasol_sites.extend((gid,r,a) for gid,r,a in zip(table.global_cell_ids,rf.values,anchors))
        record.setdefault('parasol_anchors',{})[name]=anchor_records
    all_p=cones.coordinate('opl_pedicle').values;ac_bounds=bounds_of(all_p,.04)
    sites=[('cone_site:'+gid,r,p) for gid,r,p in zip(cones.global_cell_ids,cones.coordinate('rf').values,all_p)]+parasol_sites
    for name,pre in (('local_feedforward_AC','diffuse_BC_ON'),('local_crossover_AC','diffuse_BC_OFF')):
        old=source_tables[name];_,rf,points,keys=local_typed_sites(sites,name,ac_bounds,profile_hash=atlas.inner_config['profile_sha256'])
        table=CanonicalCellTable(old.atlas_id,old.atlas_version,old.source_version,name,tuple(keys),
            {'rf':coordinate(rf,'functional_position','functional_visual_field'),
             'ipl_preterminal':coordinate(points,'terminal_position','IPL'),'ipl_post_arbor':coordinate(points,'dendritic_position','IPL')},reference)
        source_tables[name]=table;tables[name]=table.record()
        put(pre,name,source_tables[pre].coordinate('ipl_preterminal'),table.coordinate('ipl_post_arbor'),4)
    for name,pre,inh in (('midget_RGC_ON','midget_BC_ON','local_crossover_AC'),('midget_RGC_OFF','midget_BC_OFF','local_feedforward_AC'),
        ('small_bistratified_S_ON','S_ON_BC','local_feedforward_AC'),('parasol_RGC_ON','diffuse_BC_ON','widefield_AC'),('parasol_RGC_OFF','diffuse_BC_OFF','widefield_AC')):
        table=source_tables[name];sigma=.02+.06*np.linalg.norm(table.coordinate('rf').values,axis=1)
        put(inh,name,source_tables[inh].coordinate('ipl_preterminal'),table.coordinate('ipl_post_arbor'),1 if name.startswith('midget') else 8,
            sigma=sigma,exact=True)
        if name.startswith('parasol'):put(pre,name,source_tables[pre].coordinate('ipl_preterminal'),table.coordinate('ipl_post_arbor'),8,sigma=sigma,exact=True)
        if name=='small_bistratified_S_ON':put('diffuse_BC_OFF',name,source_tables['diffuse_BC_OFF'].coordinate('ipl_preterminal'),
            table.coordinate('ipl_post_arbor'),8,sigma=sigma,exact=True)
    # Actual typed owner ancestors determine the physical AII/BC terminal support.
    needed=set()
    for name in ('midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON'):
        table=source_tables[name];selected=table.query(AtlasROI(owner,owner,reference));needed.update(selected.global_cell_ids)
    while True:
        before=len(needed)
        for c in contacts.values():needed.update(e['pre_id'] for e in c['edges'] if e['post_id'] in needed)
        if len(needed)==before:break
    bc_points=[]
    for name in ('midget_BC_ON','midget_BC_OFF','diffuse_BC_ON','diffuse_BC_OFF','S_ON_BC'):
        table=source_tables[name];bc_points.extend(p for g,p in zip(table.global_cell_ids,table.coordinate('ipl_preterminal').values) if g in needed)
    if not bc_points:raise ContractError('Revised owner circuit has no supported BC ancestors')
    bounds=bounds_of(np.asarray(bc_points),.04)
    aii_bounds=(min(bounds[0],output_bounds[0]),max(bounds[1],output_bounds[1]),min(bounds[2],output_bounds[2]),max(bounds[3],output_bounds[3]))
    expand=lambda b:(b[0]-.04,b[1]+.04,b[2]-.04,b[3]+.04)
    rbc_bounds=expand(aii_bounds);rod_bounds=expand(rbc_bounds);rod_field=ApprovedPackerRodDensity(root)
    fields={'rods':(rod_field,rod_field.metadata,rod_bounds,20261011),
        'rod_BC':(lambda xy:rod_field(xy)/18,{'provider_id':'existing_RBC_rhoRod_over18','density_ratio':1/18,'rod_provider':rod_field.metadata,
            'source_level':'existing effective density transfer','approval':APPROVAL},rbc_bounds,20261012),
        'AII_AC':(lambda xy:np.full(len(xy),5000.),{'provider_id':'existing_AII_constant5000','density_cells_mm2':5000,
            'source_level':'existing effective density transfer; independent of rod','approval':APPROVAL},aii_bounds,20261013)}
    rod_tables={}
    for name,(field,source,bounds,seed) in fields.items():
        budget[name]=density_budget(field,bounds)
        if budget[name]['rounded_estimate']>max_cells:raise ContractError('Local '+name+' master exceeds budget; no density reduction')
        master=master_for(name,bounds,field,seed,source)
        if len(master.table.source_keys)>max_cells:raise ContractError('Shared complete population exceeds budget; no thinning: '+name)
        masters[name]=master
        table=master.table;xy=table.coordinate('rf').values
        coords=dict(table.coordinates,ipl_preterminal=coordinate(xy,'terminal_position','IPL'),ipl_post_arbor=coordinate(xy,'dendritic_position','IPL'))
        rod_tables[name]=CanonicalCellTable(table.atlas_id,table.atlas_version,table.source_version,name,table.source_keys,coords,reference)
    record['rod_inner']={'tables':{k:t.record() for k,t in rod_tables.items()},'typed_points':{
        'rods':{'OPL_spherule':{'values':rod_tables['rods'].coordinate('rf').values.tolist(),'unit':'mm','meaning':'terminal_position','frame':'OPL','evidence_tag':'project_approximation'},'cone_Henle_applied':False},
        'rod_BC':{'OPL_dendrite':{'values':rod_tables['rod_BC'].coordinate('rf').values.tolist(),'unit':'mm','meaning':'dendritic_position','frame':'OPL','evidence_tag':'project_approximation'}}},
        'contacts':{},'metadata':{'source_status':'approved_source_plus_project_completion',
        'rod_source_error':None,'masters':{k:m.manifest() for k,m in masters.items()},'RBC_to_rod_density_ratio':1/18,'AII_density_cells_per_mm2':5000,
        'source_level':'Packer digitized author-map points + approved quadratic completion; effective RBC/AII transfers','approval':APPROVAL,
        'boundary':'fixed local rectangular masters, finite_open, no edge compensation, no crop quota'}}
    record['population_revision']={'approval':APPROVAL,'budgets':budget,'masters':{k:m.manifest() for k,m in masters.items()},
        'master_domain_derivation':'parasol RF master from existing explicit RF dependency bounds; AII from actual owner BC IPL ancestors plus40um, union original rod/RF patch; RBC and rod each add one unchanged40um support',
        'HC_changed':False,'new_hard_rod_hole':False,'independent_parasol':True,'global_0_to_3_coverage':False}
    if population_masters is not None:record['population_revision']['shared_catalog']=population_masters.binding
    if rebuilt:record['population_revision']['explicit_migration_rebuilt_masters']=rebuilt
    return record


def rebuild_roi_from_shared_masters(root,roi,catalog,*,atlas_profile_id='retina_atlas_isetbio_macaque_v1',max_cells=60000):
    """Reuse fixed populations for a covered ROI; never generate replacement sites.

    Cone/HC and contact-source coverage keep their existing independent checks.
    The catalog's bounds do not certify recurrent closure or new source domains.
    """
    atlas=ConnectionAtlas(root,atlas_profile_id=atlas_profile_id,s_partition_profile='deMonasterio1985_mulatta_effective_S_v1')
    if catalog.reference_id!=roi.reference_id or catalog.record['cone_master_binding']!=dict(atlas.atlas.config['cone_master']):
        raise ContractError('Shared populations and cone geometry reference differ')
    phase='phase2' if max(np.hypot(x,y) for x in roi.owner_bounds_mm[:2] for y in roi.owner_bounds_mm[2:])>3 else 'phase1'
    source=atlas.preview(roi,max_cells=max_cells,phase=phase)
    return revise_local_graph(root,max_cells,connection_record=source,atlas_profile_id=atlas_profile_id,population_masters=catalog)
