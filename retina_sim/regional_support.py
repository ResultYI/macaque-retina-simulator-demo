"""Source-specific regional support, independent of UI navigation labels."""
from pathlib import Path
import hashlib,json
import numpy as np
from .contracts import ContractError,Quantity

def load_regional_support(root):
    path=Path(root)/'configurations/regional_support_v1.json'
    raw=path.read_bytes();data=json.loads(raw)
    if data.get('schema_version')!='1.0' or data.get('profile_id')!='regional_support_v1':
        raise ContractError('Unknown regional support contract')
    return dict(data,path=str(path),sha256=hashlib.sha256(raw).hexdigest())

def navigation_regions(positions):
    r=np.linalg.norm(positions,axis=1)
    return tuple('center' if x<.5 else 'transition' if x<3 else 'periphery' for x in r)

def radial_rectangle_range(bounds):
    """Exact radial range, including interior/edge minimum rather than corners only."""
    x0,x1,y0,y1=bounds
    if x1<=x0 or y1<=y0:raise ContractError('Positive 2D ROI extent required')
    nearest=np.array([np.clip(0.,x0,x1),np.clip(0.,y0,y1)])
    farthest=max(np.hypot(x,y) for x in (x0,x1) for y in (y0,y1))
    return float(np.linalg.norm(nearest)),float(farthest)

def hc_interval_provider(source,bounds):
    lower,upper=radial_rectangle_range(bounds)
    for row in source['table2']:
        lo,hi=row['eccentricity_interval_mm']
        if hi>lo and lower>=lo and upper<=hi:
            return dict(row,profile_id='Wassle_animal_'+row['animal']+'_interval',
                evidence_class='project_approximation',reported_endpoint_class='measurement',primary_verification='pending',
                spatial_extension='constant within this one animal interval; project stationarity')
    raise ContractError(f'HC OPL halo radius [{lower:.4g},{upper:.4g}] mm is not covered by one animal interval; central author estimates, point anchors and mixed animals cannot silently fill a 2D field')

def opl_roi_domain(displacement,rf_domain,seed):
    """Named bounding-box adapter of mapped RF borders, not a density rescaling."""
    from .geometry import RetinalDomain
    def mapped_bounds(bounds):
        x0,x1,y0,y1=bounds;u=np.linspace(0.,1.,65)
        edge=np.r_[np.c_[x0+u*(x1-x0),np.full(len(u),y0)],np.c_[x0+u*(x1-x0),np.full(len(u),y1)],
            np.c_[np.full(len(u),x0),y0+u*(y1-y0)],np.c_[np.full(len(u),x1),y0+u*(y1-y0)]]
        points=displacement.apply(Quantity(edge,'mm','functional_position','synthetic_fixture','functional_visual_field'),origin_seed=seed)['pedicle'].values
        lo,hi=points.min(axis=0),points.max(axis=0)
        return (float(lo[0]),float(hi[0]),float(lo[1]),float(hi[1]))
    bounds=mapped_bounds(rf_domain.bounds_mm);owner=mapped_bounds(rf_domain.owner_bounds_mm)
    return RetinalDomain(bounds,owner,lambda xy:np.ones(len(xy),bool),'McGregor_RF_OPL_rectangular_box_adapter_v1',rf_domain.label),{
        'input_frame':'functional_visual_field','output_frame':'OPL','bounds_mm':list(bounds),'owner_bounds_mm':list(owner),
        'adapter':'McGregor_RF_OPL_rectangular_box_adapter_v1','boundary_samples_per_edge':65,
        'evidence_class':'project_approximation','density_rule':'integrate HC physical OPL field; never multiply RF cone density',
        'geometry_limit':'mapped border bounding rectangle, not exact image of region; actual cone pedicle containment checked'}

def regional_binding_audit(root,compiled,execution_profile,light_context):
    contract=load_regional_support(root);result=[]
    observations=Path(root)/'configurations/regional_observations_v1.json'
    observation_contract=json.loads(observations.read_bytes())
    for cid,c in compiled.components.items():
        # Component specs are the authoritative population role bindings.
        specification=next(s for s in compiled.spec.components if s.component_id==cid)
        populations=[compiled.populations[p] for p in specification.population_roles.values()]
        frames=sorted({q.coordinate_frame for pop in populations for q in (
            pop.receptive_field_positions,pop.anatomical_positions,pop.pedicle_positions,pop.terminal_positions,pop.dendritic_positions) if q is not None})
        pid=c.parameters.profile.profile_id
        if cid=='physical_OS':policy='Chen peripheral LM source only; omitted outside >20 deg'
        elif cid=='rod':policy='author primateRod source; independent regional calibration not claimed'
        elif cid=='outer_native':policy='native_LM source regional extension; S/H2 shared H1 approximation'
        elif pid.startswith('effective_'):policy='shared_effective_dynamics; regional physiology NOT_BOUND'
        else:policy='adapter/reference; no independent regional physiological calibration'
        result.append({'component_id':cid,'parameter_profile_id':pid,'ui_regions':sorted({r for pop in populations for r in pop.regions}),
            'source_ids':list(c.source_ids),'profile_conditions':c.parameters.profile.conditions,
            'typed_population_roles':{role:{'cell_types':sorted(set(compiled.populations[p].cells.cell_type)),
                'subtypes':sorted(set(compiled.populations[p].cells.subtype))} for role,p in specification.population_roles.items()},
            'input_ports':{name:{'unit':port.unit,'meaning':port.meaning} for name,port in c.inputs.items()},
            'output_ports':{name:{'unit':port.unit,'meaning':port.meaning} for name,port in c.outputs.items()},
            'background_context_hash':light_context.background_context_hash,
            'model_form':('conductance membrane ODE + softplus rate + strict-past spike-count history; no relative-threshold/absolute-refractory state'
                if specification.mechanism=='HistoryConductanceRGC' else specification.mechanism),
            'allowed_approximation_ids':execution_profile['allowed_approximation_ids'],
            'population_frames':frames,'regional_parameter_binding':policy,'assay_target_binding':'none; source observation target is not a model coefficient'})
    return {'contract':contract,'component_bindings':result,'observation_catalog':{
        'path':str(observations),'sha256':hashlib.sha256(observations.read_bytes()).hexdigest(),'usage':'reference_only; source ports/conditions/frames, never mechanism parameters'},'validation_status':'STATIC_REGIONAL_FOLLOWUP_NOT_RUN',
        'source_model_candidates':[{key:row[key] for key in ('candidate_id','source_id','status','runtime_parameter_binding')}
            for row in observation_contract.get('generative_candidates',[])],
        'paired_data_asset_status':{key:row['status'] for key,row in observation_contract.get('paired_data_assets',{}).items()},
        'region_definitions_are_source_specific':True,'automatic_STA_to_tau_or_kernel_conversion':False}
