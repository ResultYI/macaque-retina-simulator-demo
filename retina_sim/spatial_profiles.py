"""Source-limited spatial construction with explicit finite-domain approximations."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from .contracts import ContractError
from .geometry import RetinalDomain, MosaicBackend, integrate_density

def source_roi(root, density, seed, *, owner_bounds=(4.59,4.61,-.01,.01), halo_mm=.04, max_cells=8192,displacement=None,hc_configuration=None,generation_bounds_mm=None):
    """A selectable patch within one source HC interval, never a global animal."""
    owner=tuple(owner_bounds)
    if halo_mm<.04:raise ContractError('This profile requires at least 40 um halo for its 40 um one-edge support; recurrent boundary error remains declared')
    bounds=(owner[0]-halo_mm,owner[1]+halo_mm,owner[2]-halo_mm,owner[3]+halo_mm)
    output_bounds=bounds
    if generation_bounds_mm is not None:
        bounds=tuple(generation_bounds_mm)
        if not (bounds[0]<=output_bounds[0] and bounds[1]>=output_bounds[1] and bounds[2]<=output_bounds[2] and bounds[3]>=output_bounds[3]):
            raise ContractError('Feedforward generation guard must retain original output/halo cohort domain')
    domain=RetinalDomain(bounds,owner,lambda xy:np.ones(len(xy),bool),
        'ISETMacaque_PackerConeRc','source_limited_peripheral_patch')
    from .regional_support import hc_interval_provider,opl_roi_domain
    if displacement is None:raise ContractError('Source HC requires explicit RF-to-OPL ROI adapter; no implicit frame identity')
    hc_domain,hc_frame=opl_roi_domain(displacement,domain,seed)
    configuration=hc_configuration or {'profile_id':'source_patch_v1'}
    if configuration['profile_id']=='hc_joint_synthetic_v1':
        from .hc_geometry import joint_hc_geometry
        hc,hc_geometry=joint_hc_geometry(root,hc_domain,seed,max_cells,configuration)
        hc_frame=dict(hc_frame,mapped_input_bounds_mm=list(hc_domain.bounds_mm),
            bounds_mm=hc_geometry['active_bounds_mm'],canonical_bounds_mm=hc_geometry['canonical_bounds_mm'],
            soma_density_plane='physical anatomical_retina',anchor_policy='co_located_lumped_anchor; HC soma is not cone-displaced')
        profile_id=configuration['profile_id']
        density_source={'field':'continuous total density + H1 fraction; cross-animal synthetic profile',
            'evidence':'source observations and new project shape/mosaic choices remain separate',
            'source_records':configuration['source_records'],'assumptions':configuration['assumptions']}
    else:
        source=json.loads((Path(root)/'sources/spatial/wassle2000_HC_density_tables.json').read_text(encoding='utf-8'))
        row=hc_interval_provider(source,hc_domain.bounds_mm)
        hc=[]
        for ordinal,subtype in enumerate(('H1','H2')):
            hc.append((subtype,generate_density(lambda xy,v=row[subtype]:np.full(len(xy),v),hc_domain,seed+20+ordinal,max_cells)))
        profile_id=row['profile_id']
        density_source={'animal':row['animal'],'OPL_interval_mm':row['eccentricity_interval_mm'],
            'H1_cells_per_mm2':row['H1'],'H2_cells_per_mm2':row['H2'],
            'evidence':'reported measurement quotation; primary verification pending','spatial_extension':row['spatial_extension']}
        hc_geometry={'profile_id':'source_patch_v1','configuration':configuration,'validation_status':'original independent patch retained'}
    mass=integrate_density(density,domain)
    if mass['rounding_target']>max_cells:
        raise ContractError(f'Source density requires {mass["rounding_target"]} cones; budget {max_cells}; reduce ROI or select engineering_preview')
    mosaic=generate_density(density,domain,seed,max_cells)
    return domain,mosaic,hc,{'profile_id':profile_id,
        'owner_bounds_mm':list(owner),'halo_bounds_mm':list(bounds),'halo_mm':halo_mm,
        'RF_output_cohort_bounds_mm':list(output_bounds),'RF_generation_guard_bounds_mm':list(bounds),
        'cone_density_scale':1.,'cone_count_policy':'round integral rho dA; no density thinning',
        'HC_density_source':density_source,'HC_geometry':hc_geometry,
        'HC_frame_policy':hc_frame,
        'central_HC':('synthetic central log-smoothstep with explicit source/shape separation; NOT_RUN' if configuration['profile_id']=='hc_joint_synthetic_v1' else 'author scalar anchor retained as reference only; no 2D field extrapolation'),
        'boundary_policy':'finite induced graph; exterior couplings omitted, no reflected/duplicated cells',
        'halo_claim':'finite buffer only; HC and electrical loops have unbounded recurrent dependency; no boundary-error-free claim',
        'density_field':'Packer published-code meridians; log PCHIP and angular extension are project approximations',
        'unknown_subtype_density':'BC/AC/RGC geometry remains effective units; RBC/AII source-anchor formulas are project transfers, no measured full-field subtype density'}

def generate_density(field,domain,seed,max_cells):
    n=integrate_density(field,domain)['rounding_target']
    if n<1 or n>max_cells:raise ContractError(f'Density-derived cell count {n} outside declared budget; change ROI, no thinning')
    side=max(64,int(np.ceil(np.sqrt(8*n))))
    return MosaicBackend().generate('greedy_capacitated_equal_rho_mass',field,domain,seed,
        quadrature_shape=(side,side),iterations=1)

def bounded_matrix(pre,target,k,*,support_mm,gaussian_scale=None,policy='finite_induced_graph'):
    """Hard support first, then declared pooling normalization; deficits are exported."""
    pre,target=np.asarray(pre),np.asarray(target)
    radius=np.broadcast_to(support_mm,(len(target),))
    if np.any(radius<=0):raise ContractError('Positive explicit hard support required')
    tree=cKDTree(pre);rows=[];cols=[];weights=[];audit=[]
    scale=None if gaussian_scale is None else np.broadcast_to(gaussian_scale,(len(target),))
    desired=np.broadcast_to(k,(len(target),))
    for j,point in enumerate(target):
        candidates=np.array(tree.query_ball_point(point,radius[j]),dtype=int)
        available=len(candidates)
        distance=np.linalg.norm(pre[candidates]-point,axis=1)
        order=np.argsort(distance,kind='stable')[:int(np.ceil(desired[j]))]
        candidates,distance=candidates[order],distance[order]
        w=np.ones(len(candidates)) if scale is None else np.exp(-(distance/scale[j])**2)
        if len(w) and desired[j]%1 and len(w)==int(np.ceil(desired[j])):w[-1]*=desired[j]%1
        before=float(w.sum())
        if before:w/=before
        rows.extend([j]*len(candidates));cols.extend(candidates.tolist());weights.extend(w.tolist())
        audit.append({'post_index':j,'target_degree':float(desired[j]),'actual_degree':len(candidates),'available_candidate_degree':available,
            'edge_count':len(candidates),'sum_weights_before_normalization':before,'sum_weights_after_normalization':float(w.sum()),
            'N_eff':float(w.sum()**2/(w@w)) if len(w) and w@w>0 else 0.,
            'missing_degree':max(0,int(np.ceil(desired[j]))-len(candidates)),
            'hard_support_mm':float(radius[j]),'maximum_distance_mm':float(distance.max()) if len(distance) else None,
            'normalization_input_sum':before,'boundary_policy':policy})
    return sparse.csr_matrix((weights,(rows,cols)),shape=(len(target),len(pre))),audit

def aii_ipl_support_domain(input_domain,mapped_bc_owner_bounds,*,support_mm,bc_target_positions=None):
    """Cover actual BC IPL owner targets as well as the existing rod/RBC patch.

    The existing AII density is a physical-tissue project transfer. A cone RF
    rectangle cannot stand in for the displaced BC's IPL rectangle centrally.
    This changes the generated support mask, not density, reach or kinetics.
    """
    from .geometry import RetinalDomain
    original=tuple(input_domain.bounds_mm);owner=tuple(input_domain.owner_bounds_mm)
    x0,x1,y0,y1=map(float,mapped_bc_owner_bounds)
    owner_needed=(x0-support_mm,x1+support_mm,y0-support_mm,y1+support_mm)
    # Preserve an already covered source-patch domain. Halo targets alone must
    # not expand the old AII population and create unsupported RBC inputs.
    tol=1e-12
    covered=(owner_needed[0]>=original[0]-tol and owner_needed[1]<=original[1]+tol
        and owner_needed[2]>=original[2]-tol and owner_needed[3]<=original[3]+tol)
    needed=owner_needed
    if bc_target_positions is not None:
        points=np.asarray(bc_target_positions)
        lo,hi=points.min(axis=0),points.max(axis=0)
        needed=(float(lo[0]-support_mm),float(hi[0]+support_mm),float(lo[1]-support_mm),float(hi[1]+support_mm))
    rectangles=[original] if covered else [original,needed]
    bounds=(min(r[0] for r in rectangles),max(r[1] for r in rectangles),min(r[2] for r in rectangles),max(r[3] for r in rectangles))
    owner_union=(min(owner[0],x0),max(owner[1],x1),min(owner[2],y0),max(owner[3],y1))
    # Keep the existing rectangular equal-mass backend. This is actual tissue
    # between the source patches, not a disconnected mask with unaccounted
    # quadrature/centroid errors. Every cell in the enlarged physical rectangle
    # is generated at the unchanged declared density and consumes the budget.
    resolved=input_domain if covered else RetinalDomain(bounds,owner_union,lambda xy:np.ones(len(xy),bool),
        'AII_existing_density_actual_IPL_support_union_v1','AII_IPL_support_union')
    return resolved,{'frame':'IPL','unit':'mm','generation_rectangles_mm':[list(bounds)],
        'input_and_BC_required_rectangles_mm':[list(r) for r in rectangles],
        'prior_input_rectangle_mm':list(original),'mapped_BC_owner_rectangle_mm':[x0,x1,y0,y1],
        'required_BC_owner_support_rectangle_mm':list(owner_needed),'prior_domain_covers_owner_support':covered,
        'required_all_BC_support_rectangle_mm':list(needed),
        'support_target_scope':'all actual generated BC IPL targets' if bc_target_positions is not None else 'mapped owner rectangle',
        'domain_extended':not covered,'density_scale':1.,'density_value_changed':False,
        'source_scope':'existing 5000/mm2 Mills-anchor physical density transfer remains project_approximation',
        'BC_IPL_coordinate_rule':'existing BC terminals co-located with mapped cone pedicles; no new morphology displacement inferred',
        'mask_policy':'complete physical canonical bounding rectangle over both source patches; intermediate tissue consumes unchanged density/budget, no thinned or fictitious contacts',
        'owner_policy':'actual source owner patch union; intermediate tissue is halo, not an optical RF owner',
        'unbounded_recurrent_boundary_claim':False}

def finite_support_coverage(target,support_mm,rectangles):
    """Conservative complete-circle coverage; absent support is not a zero edge."""
    target=np.asarray(target);covered=np.zeros(len(target),bool)
    for x0,x1,y0,y1 in rectangles:
        covered|=(target[:,0]-support_mm>=x0-1e-12)&(target[:,0]+support_mm<=x1+1e-12)&(target[:,1]-support_mm>=y0-1e-12)&(target[:,1]+support_mm<=y1+1e-12)
    return covered


def presynaptic_support_domain(target_domain,support_mm,*,domain_name):
    """Generate upstream physical tissue, without changing a connection reach.

    This is the rectangular backend's complete support envelope. Density is
    still evaluated at each physical point by the caller's existing field.
    """
    from .geometry import RetinalDomain
    if not np.isfinite(support_mm) or support_mm<=0:
        raise ContractError('Positive finite presynaptic generation halo required')
    x0,x1,y0,y1=target_domain.bounds_mm
    return RetinalDomain((x0-support_mm,x1+support_mm,y0-support_mm,y1+support_mm),
        target_domain.owner_bounds_mm,lambda xy:np.ones(len(xy),bool),
        target_domain.mask_source_id,domain_name)

def hc_contact_matrices(cone_opl,cone_types,hc_opl,hc_types,owner,*,support_mm=.04):
    """Balanced source-anchored divergence; feedback returns over those same contacts."""
    cone_opl,hc_opl=np.asarray(cone_opl),np.asarray(hc_opl)
    types=np.asarray(cone_types);hct=np.asarray(hc_types);B=sparse.lil_matrix((len(types),len(hct)));audits=[]
    r=np.linalg.norm(cone_opl,axis=1)
    for subtype,central,peripheral in [('H1',3.7,7.5),('H2',2.2,3.2)]:
        eligible=np.flatnonzero(types!='S') if subtype=='H1' else np.arange(len(types))
        source=np.flatnonzero(hct==subtype)
        mean=central+(peripheral-central)*np.clip((r[eligible]-.75)/(5.5-.75),0,1)
        counts=np.floor(mean).astype(int)
        extra=int(np.floor(mean.sum()+.5)-counts.sum())
        counts[np.argsort(-(mean-counts),kind='stable')[:extra]]+=1
        matrix,rows=bounded_matrix(hc_opl[source],cone_opl[eligible],counts,support_mm=support_mm)
        for i,row in enumerate(rows):
            if owner[eligible[i]] and row['missing_degree']:
                raise ContractError(f'HC {subtype} owner contact quota deficit; enlarge halo/support explicitly, no unbounded nearest')
        for j,i in enumerate(eligible):B[i,source]=matrix.getrow(j).toarray()
        audits.append({'subtype':subtype,'requested_mean':mean.tolist(),'integer_counts':counts.tolist(),
            'rows':rows,'evidence':'source anchor interpolation is project geometry; H2 3.2 is midpoint of reported 2.7..3.7 interval',
            'feedback':'only existing feedforward contacts; no independently chosen feedback neighbors'})
    B=B.tocsr();total=np.asarray(B.sum(axis=1)).ravel()
    B=sparse.diags(np.divide(1.,total,out=np.zeros_like(total),where=total>0))@B
    A=B.T.tocsr();total=np.asarray(A.sum(axis=1)).ravel()
    A=sparse.diags(np.divide(1.,total,out=np.zeros_like(total),where=total>0))@A
    return A,B,audits
