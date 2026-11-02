"""Typed OPL circle supports for current user-selected HC v2."""
import numpy as np
from .hc_arbor import validate_footprints
from .contracts import ContractError

def footprints(profile,ids,types,xy,*,mode,enable_central_candidate,pedicle_density,morphology):
    if pedicle_density is not None or morphology:
        raise ContractError('HC v2 frozen circle rule cannot silently accept pedicle-density or morphology overrides')
    records=[];blocked=[];shapes={'H1':None,'H2':None,'by_hc_id':{}}
    for cid,kind,point in zip(ids,types,xy):
        r=float(np.linalg.norm(point[:2]));failure=None
        if kind not in ('H1','H2'):failure='HC_SUBTYPE_UNSUPPORTED'
        elif r>profile.support_radius_mm:failure='HC_V2_CONTACT_ANCHOR_OUTSIDE_BOUND_PHYSICAL_SUPPORT'
        elif r<.5 and not enable_central_candidate:failure='CENTRAL_INWARD_CANDIDATE_NOT_SELECTED'
        row=dict(hc_id=cid,cell_type=kind,coordinate_frame='OPL',unit='mm',anchor_position_mm=point.tolist(),radius_from_fovea_mm=r,
            soma_OPL_mapping='Existing HC_soma_to_OPL_colocation_project_v1; no Cone IS/RF substitution',
            source_domain='User v2 engineering application; not physiological validation')
        if failure:
            row.update(coverage_status='unsupported',reason=failure);blocked.append(dict(row))
        else:
            radius=float(profile.contact_radius_mm(r,kind));central=r<.6
            a,b=profile.config['geometry']['central_join']['interval_mm'];joining=a<r<b
            row.update(radius_mm=radius,coverage_status='project_approximation',evidence_status='EXTRAPOLATED_UNCALIBRATED' if central else 'USER_JOINT_V2_ENGINEERING_CONTACT_CIRCLE',
                low_evidence=True,candidate_id='HC_C1_inner_join_058_060_v1' if joining else 'retained_central_constant_footprint_v1' if central else profile.profile_id,
                representation='circular_equivalent_dendritic_cone_contact_not_functional_RF',
                application_domain=[a,b] if joining else [0.,a] if central else [b,profile.support_radius_mm],density_basis='HC subtype soma density under existing soma-OPL colocation',
                D1_denominator='Unresolved all-cone versus eligible-LM literature denominator; formula unchanged, no correction factor',
                span_is_functional_RF=False)
            if r>1.338:
                row.update(evidence_status='USER_AUTHORIZED_OUTER_ENGINEERING_EXTRAPOLATION',
                    original_application_radius_mm=1.338,authorization=profile.config['owner_support']['authorization'])
            shapes['by_hc_id'][cid]=dict(kind='disk',radius_mm=radius,unit='mm',coordinate_frame='OPL',source_profile_id=profile.profile_id,
                evidence_class='project_approximation',approximation_id=row['candidate_id'],kernel={'kind':'uniform'},
                application_anchor_band_mm=row['application_domain'],representation=row['representation'],
                source_locator='User supplied HC v2 with user-authorized fixed0.58..0.60mm inner C1 join; no functionalRF claim')
        records.append(row)
    return dict(footprints=validate_footprints(shapes,require_configured=False),blocked_records=blocked,per_hc_coverage=records,
        manifest=dict(profile_id=profile.profile_id,profile_sha256=profile.profile_hash,current_hc_identity=profile.metadata['current_hc_identity'],
            mode=mode,central_candidate_explicitly_enabled=enable_central_candidate,geometry_ready=not blocked,
            maximum_generated_envelope_mm=max((r.get('radius_mm',0.) for r in records),default=0.),
            finite_anchor_curves=[],physiological_validation=False,per_cone_integer_divergence_imposed=False,
            type_eligibility='Unchanged H1 excludes S; H2 inputs LMS; separate H2 feedback remains S-only',
            central_join='Fixed0.58..0.60mm cubic Hermite: central value/zero slope to v2 endpoint value/analytic slope; engineering width, no spike/contact-count tuning',
            normalization='Raw disk edges1; normalize once on complete fine graph then all-member projection; no local crop renormalization',
            boundary_claim='Rule domain applies to HC anchor; full pedicle query-envelope completeness remains separately required',
            inward_candidate_completes_coverage=False))
