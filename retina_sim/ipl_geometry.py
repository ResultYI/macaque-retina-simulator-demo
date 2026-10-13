"""Approved typed effective IPL geometry, independent of native equations."""
from pathlib import Path
from dataclasses import dataclass
import hashlib,json
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import brentq
from .contracts import ContractError,Quantity

TYPED='p_chart_typed_effective_ipl_v1'
RESEARCH='retina_contact_geometry_effective_v1'
TYPED_PROFILES=(TYPED,RESEARCH)
LEGACY='legacy_point_ipl_v1'
IPL_ROUTES=(('midget_RGC_ON','midget_BC_ON','local_crossover_AC'),
    ('midget_RGC_OFF','midget_BC_OFF','local_feedforward_AC'),
    ('parasol_RGC_ON','diffuse_BC_ON','widefield_AC'),
    ('parasol_RGC_OFF','diffuse_BC_OFF','widefield_AC'),
    ('small_bistratified_S_ON','S_ON_BC','local_feedforward_AC'))


@dataclass(frozen=True)
class ParasolRFSeedInput:
    """Explicit supplied positions/site IDs; never an implicit density provider."""
    positions: Quantity
    site_ids: tuple
    provenance: dict

    def __post_init__(self):
        ids=tuple(self.site_ids)
        if not ids or len(set(ids))!=len(ids) or any(not isinstance(v,str) or not v for v in ids):
            raise ContractError('Parasol RF seed IDs must be explicit, unique and nonempty')
        self.positions.require('mm','functional_position',(len(ids),2),'functional_visual_field')
        if not isinstance(self.provenance,dict) or any(not isinstance(self.provenance.get(k),str) or
            not self.provenance[k].strip() for k in ('source_id','locator','origin','evidence_class')):
            raise ContractError('Parasol RF seeds require source_id, locator, origin and evidence_class')
        object.__setattr__(self,'site_ids',ids)
        object.__setattr__(self,'provenance',dict(self.provenance))

    @classmethod
    def from_legacy_scaffold(cls,scaffold,canonical_instance_id):
        return cls(scaffold.positions,tuple('parasol_scaffold:'+canonical_instance_id+':'+s for s in scaffold.cell_ids),
            {'source_id':'PRIMARY_PATHWAY_project_profile','locator':'examples/full_target_network.py: rbc_field / inner_rf_scaffold',
             'origin':'legacy rod-derived RBC RF scaffold; frozen geometry, not independent parasol physiological density',
             'evidence_class':'project_approximation'})

    def record(self):
        return {'positions':{'values':self.positions.values.tolist(),'unit':self.positions.unit,
            'meaning':self.positions.meaning,'frame':self.positions.coordinate_frame,'evidence_tag':self.positions.evidence_tag},
            'site_ids':list(self.site_ids),'provenance':dict(self.provenance),
            'physiological_validation':False,'default_adoption_for_new_atlas':False}

def read_ipl_geometry(root,profile_id):
    if profile_id==LEGACY:return {'profile_id':LEGACY,'evidence_class':'project_approximation','claim':'Legacy soma/point and RBC-scaffold geometry; retained for reproduction'}
    if profile_id not in TYPED_PROFILES:raise ContractError('Unknown named IPL geometry profile')
    path=Path(root)/'configurations'/f'{profile_id}.json';raw=path.read_bytes();p=json.loads(raw)
    if p.get('schema_version')!='1.0' or p.get('profile_id')!=profile_id or p['hard_support_mm']!=.04:
        raise ContractError('Invalid typed IPL profile identity/support')
    if not np.isclose(p['widefield_AC_sampling']['spacing_mm'],.04/3,rtol=0,atol=1e-16):
        raise ContractError('Fixed approved IPL grid spacing required')
    if (p.get('contact_frame')!='IPL' or p.get('contact_metric')!='point_proxy' or
        p['widefield_AC_sampling'].get('origin_mm')!=[0.,0.] or
        p['widefield_AC_sampling'].get('kind')!='fixed_global_square_grid' or
        p['RGC_degree'].get('midget_inhibition')!=1 or p['RGC_degree'].get('nonmidget')!=8 or
        p['boundary'].get('new_external_background')!='NOT_ENABLED' or
        p['boundary'].get('new_recurrent_truncation')!='NOT_ENABLED'):
        raise ContractError('Named IPL v1 frame/grid/degree/boundary contract differs from implementation')
    if profile_id==RESEARCH:
        relay=p.get('S_ON_effective_contract',{})
        if (relay.get('legacy_selection_budget')!=8 or relay.get('minimum_legal_selected_inputs')!=1 or
            relay.get('exact_target_enforcement') is not False or relay.get('pre')!='S_ON_BC' or
            relay.get('post')!='small_bistratified_S_ON'):
            raise ContractError('Invalid named S effective relay contract')
    return dict(p,profile_sha256=hashlib.sha256(raw).hexdigest())

def bounds_of(points,padding=0.):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=2 or not len(points) or not np.all(np.isfinite(points)):
        raise ContractError('Nonempty finite IPL points required')
    lo=points.min(0)-padding;hi=points.max(0)+padding
    return (float(lo[0]),float(hi[0]),float(lo[1]),float(hi[1]))

def in_bounds(points,bounds):
    p=np.asarray(points);x0,x1,y0,y1=bounds
    return (p[:,0]>=x0)&(p[:,0]<x1)&(p[:,1]>=y0)&(p[:,1]<y1)

def inverse_p_points(displacement,points):
    points=np.asarray(points,float);out=[]
    for point in points:
        radius=float(np.linalg.norm(point));theta=float(np.arctan2(point[1],point[0]))
        def f(r):return float(r+.85*(displacement.radial(np.array([r]),np.array([theta]))[0]-r))-radius
        if radius<=0 or f(1e-9)>=0 or f(10.)<=0:
            raise ContractError('P_CHART_INVERSE_UNSUPPORTED: input point outside existing invertible chart')
        r=brentq(f,1e-9,10.,xtol=1e-12)
        out.append(point*(r/radius))
    return np.asarray(out)

def plan_feedforward_guard(displacement,rf_core,opl_bounds,*,support_mm=.04):
    """Finite three-edge guard; this never changes a final contact cutoff."""
    h=support_mm/3;pad=3*support_mm+h
    x0,x1,y0,y1=opl_bounds;physical=(x0-pad,x1+pad,y0-pad,y1+pad)
    xs=np.linspace(physical[0],physical[1],129);ys=np.linspace(physical[2],physical[3],129)
    border=np.concatenate([np.c_[xs,np.full(len(xs),physical[2])],np.c_[xs,np.full(len(xs),physical[3])],
        np.c_[np.full(len(ys),physical[0]),ys],np.c_[np.full(len(ys),physical[1]),ys]])
    inverse=inverse_p_points(displacement,border);b=bounds_of(inverse,1e-6)
    resolved=(min(b[0],rf_core[0]),max(b[1],rf_core[1]),min(b[2],rf_core[2]),max(b[3],rf_core[3]))
    return resolved,{'RF_output_cohort_bounds_mm':list(rf_core),'required_BC_P_bounds_mm':list(physical),
        'RF_generation_bounds_mm':list(resolved),'single_edge_support_mm':support_mm,
        'potential_guard_mm':pad,'rule':'centroid seed + AC + BC input support and one grid spacing rounding guard',
        'inverse_rule':'existing P chart, sampled rectangle boundary 129/side + 1um numerical box guard; no physiological radius change',
        'recurrent_dependency_claim':'NOT_ESTABLISHED','new_boundary_operator':False}

def freeze_centroid_anchors(seed_points,pre_points,pre_ids,*,support_mm=.04):
    seed_points=np.asarray(seed_points);pre_points=np.asarray(pre_points);tree=cKDTree(pre_points);anchors=[];records=[]
    for j,seed in enumerate(seed_points):
        indices=sorted(tree.query_ball_point(seed,support_mm),key=lambda i:pre_ids[i])
        if not indices:raise ContractError(f'ANCHOR_UNRESOLVED: parasol seed {j}, no typed BC within {support_mm}mm')
        anchor=pre_points[indices].mean(0);anchors.append(anchor)
        records.append({'seed_xy_mm':seed.tolist(),'anchor_xy_mm':anchor.tolist(),'candidate_bc_ids':[pre_ids[i] for i in indices],
            'candidate_weights':[1/len(indices)]*len(indices),'weights_source':'approved equal-weight seed pool',
            'freeze_rule':'once before final contact queries; no deficit-driven relocation'})
    return np.asarray(anchors),records

def local_typed_sites(sites,ac_type,bounds,*,profile_hash):
    # sites: frozen site ID, RF seed, P-chart contact point. Typed IDs are
    # independent state identities even when two types share the same xy.
    unique={}
    for site_id,rf,p in sites:
        if site_id in unique:
            if not np.array_equal(unique[site_id][1],p):raise ContractError('Frozen IPL site ID has conflicting geometry')
        else:unique[site_id]=(np.asarray(rf),np.asarray(p))
    selected=[(key,*unique[key]) for key in sorted(unique) if in_bounds(np.asarray([unique[key][1]]),bounds)[0]]
    ids=tuple(f'ipl:{profile_hash[:16]}:{ac_type}:{hashlib.sha256(key.encode()).hexdigest()[:24]}' for key,_,_ in selected)
    return ids,np.asarray([rf for _,rf,_ in selected]),np.asarray([p for _,_,p in selected]),[key for key,_,_ in selected]

def widefield_grid(bounds,*,spacing_mm=.04/3,max_cells=8192):
    if not np.isclose(spacing_mm,.04/3,rtol=0,atol=1e-16):raise ContractError('Approved fixed widefield grid required')
    x0,x1,y0,y1=bounds
    ix=np.arange(int(np.floor(x0/spacing_mm)),int(np.ceil(x1/spacing_mm))+1)
    iy=np.arange(int(np.floor(y0/spacing_mm)),int(np.ceil(y1/spacing_mm))+1)
    count=len(ix)*len(iy)
    if count>max_cells:raise ContractError(f'Widefield fixed grid requires {count} nodes, budget {max_cells}; no thinning')
    pairs=[(int(x),int(y)) for x in ix for y in iy]
    ids=tuple(f'ipl:widefield_AC:grid40over3:{x}:{y}' for x,y in pairs)
    return ids,np.asarray(pairs,float)*spacing_mm,{'spacing_mm':spacing_mm,'origin_mm':[0.,0.],
        'index_bounds':[int(ix[0]),int(ix[-1]),int(iy[0]),int(iy[-1])],'generated_count':count,
        'is_biological_density':False,'tile_identity':'integer global lattice keys; crop preserves coordinates and IDs'}

def typed_ipl_points(pop,role):
    # A BC terminal compartment also receives AII inhibition/gap. Do not
    # replace that explicit compartment with the BC's OPL dendrite or soma.
    point=pop.terminal_positions if role=='presynaptic' or set(pop.cells.cell_type)=={'BC'} else pop.dendritic_positions
    if point is None or point.coordinate_frame!='IPL' or point.unit!='mm':
        raise ContractError(f'{pop.population_id}: explicit IPL {role} point missing; soma/RF fallback forbidden')
    return point.values


def typed_layouts(bc_populations,cone_ids,cone_rf,cone_p,scaffold,output_domain,displacement,
                  configuration,canonical_instance_id,max_cells,*,parasol_rf_seeds=None):
    """Pure geometry planner used by the real builder and source-only diagnostics.

    It builds no mechanism and invents no neuron density or replacement drive.
    Native assembly uses the returned nodes without pruning their halo states.
    """
    rgcs={};acs={};audit={'populations':{},'anchors':{}}
    all_rgcs=[];parasol_sites=[];support=configuration['hard_support_mm']
    if parasol_rf_seeds is None:
        if scaffold is None:raise ContractError('PARASOL_RF_SEEDS_UNBOUND: explicit seed source required; no rod provider fallback')
        seeds=ParasolRFSeedInput.from_legacy_scaffold(scaffold,canonical_instance_id)
    else:
        if not isinstance(parasol_rf_seeds,ParasolRFSeedInput):raise ContractError('Explicit ParasolRFSeedInput required')
        seeds=parasol_rf_seeds
    scaffold_rf=seeds.positions.values
    audit['parasol_RF_seed_binding']=seeds.record()
    scaffold_p=displacement.apply(Quantity(scaffold_rf,'mm','functional_position','project_approximation',
        'functional_visual_field'))['pedicle'].values
    for cid,source,_ in IPL_ROUTES:
        pre=bc_populations[source]
        if cid.startswith('parasol'):
            rf=scaffold_rf
            anchors,records=freeze_centroid_anchors(scaffold_p,typed_ipl_points(pre,'presynaptic'),pre.cells.cell_id)
            parents=();site_ids=seeds.site_ids
            parasol_sites.extend((site_id,point,anchor) for site_id,point,anchor in zip(site_ids,rf,anchors))
        else:
            indices=np.flatnonzero(in_bounds(pre.cells.rf_position.values,output_domain.bounds_mm))
            rf=pre.cells.rf_position.values[indices];anchors=typed_ipl_points(pre,'presynaptic')[indices]
            parents=tuple(pre.cells.cell_id[i] for i in indices);site_ids=parents
            records=[{'parent_bc_id':parent,'anchor_xy_mm':anchor.tolist(),
                'freeze_rule':'qualified parent P; not single-input convergence claim'} for parent,anchor in zip(parents,anchors)]
        if not len(rf):raise ContractError('EMPTY_RGC_OUTPUT_COHORT: '+cid)
        rgcs[cid]={'RF':rf,'P':anchors,'parents':parents,'site_ids':site_ids};all_rgcs.append(anchors)
        audit['anchors'][cid]={'records':records,'frame':'IPL','metric':'point_proxy',
            'soma':'legacy G visualization proxy stored separately; not used for non-midget contacts',
            'regions':configuration['strata'][cid]}
    ac_bounds=bounds_of(np.concatenate(all_rgcs),support)
    sites=[('cone_site:'+cone_ids[i],cone_rf[i],cone_p[i]) for i in range(len(cone_ids))]+parasol_sites
    for ac_type in ('local_feedforward_AC','local_crossover_AC'):
        ids,rf,points,site_ids=local_typed_sites(sites,ac_type,ac_bounds,profile_hash=configuration['profile_sha256'])
        if len(ids)>max_cells:raise ContractError(f'{ac_type}: effective unit budget exceeded; no thinning')
        acs[ac_type]=(ids,rf,points)
        audit['populations'][ac_type]={'entity_kind':'typed_effective_computational_unit','count':len(ids),
            'site_ids':site_ids,'cell_ids':ids,'sampling':'one independent type/site state; no RBC density provider',
            'regions':configuration['strata'][ac_type],'P_generation_bounds_mm':list(ac_bounds)}
    ids,points,grid=widefield_grid(ac_bounds,max_cells=max_cells)
    acs['widefield_AC']=(ids,inverse_p_points(displacement,points),points)
    audit['populations']['widefield_AC']=dict(grid,cell_ids=ids,entity_kind='typed_effective_computational_unit',
        regions=configuration['strata']['widefield_AC'],P_generation_bounds_mm=list(ac_bounds))
    audit['dynamics']='unchanged native factories, per-type shared PCG64 and global solver'
    return rgcs,acs,audit
