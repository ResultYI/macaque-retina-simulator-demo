"""Integrated HC geometry profiles: original source patch or joint synthetic field.

This module resolves the same configuration used by GUI/CLI/network assembly.
Named optional arbor supports are explicit project approximations with source
domains; they are not measured maximum reach and do not modify HC dynamics.
"""
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import math
import numpy as np
from .contracts import ContractError, Quantity
from .geometry import RetinalDomain, Mosaic

HC_PROFILES = ('source_patch_v1', 'hc_joint_synthetic_v1')
HC_ARBOR_PROFILES = ('explicit_footprints', 'central_OPL_equal_area_W2000','W2000_equal_area_bounded_edges_047_104')


def read_hc_geometry(root, profile_id='source_patch_v1', *, config_path='',
                     h1_radius_mm=None, h2_radius_mm=None,
                     arbor_profile_id='explicit_footprints'):
    if profile_id not in HC_PROFILES:
        raise ContractError('Unknown named HC geometry profile')
    if arbor_profile_id not in HC_ARBOR_PROFILES:
        raise ContractError('Unknown named HC arbor profile')
    if profile_id == 'source_patch_v1':
        if arbor_profile_id != 'explicit_footprints':
            raise ContractError('source_patch_v1 is incompatible with the named central HC arbor profile; select the joint synthetic HC geometry explicitly')
        if config_path or h1_radius_mm is not None or h2_radius_mm is not None:
            raise ContractError('Explicit arbor overrides require the joint synthetic HC profile')
        return {'profile_id': profile_id, 'arbor_profile_id': arbor_profile_id,
                'validation_status': 'original_source_patch_route_retained'}
    path = Path(root) / 'configurations/hc_joint_synthetic_v1.json'
    raw = path.read_bytes()
    data = json.loads(raw)
    overrides = {}
    if config_path:
        override_path = Path(config_path)
        if not override_path.is_absolute():
            override_path = Path(root) / override_path
        override_raw = override_path.read_bytes()
        overrides = json.loads(override_raw)
        allowed = {'profile_id', 'canonical_bounds_mm', 'footprints', 'generator',
                   'divergence_profile', 'require_owner_quota', 'gap_candidate_max_distance_mm',
                   'arbor_profile_id'}
        if not isinstance(overrides, dict) or set(overrides) - allowed:
            raise ContractError('HC geometry JSON may configure only canonical domain, generator, finite footprints and divergence policy')
        if overrides.get('profile_id', profile_id) != profile_id:
            raise ContractError('HC geometry file identity does not match selected profile')
        external_arbor = overrides.get('arbor_profile_id', arbor_profile_id)
        if external_arbor not in HC_ARBOR_PROFILES:
            raise ContractError('Unknown named HC arbor profile in geometry JSON')
        if arbor_profile_id != 'explicit_footprints' and external_arbor != arbor_profile_id:
            raise ContractError('Selected HC arbor profile conflicts with geometry JSON arbor_profile_id')
        arbor_profile_id = external_arbor
        for key, value in overrides.items():
            if key == 'generator':
                data[key] = dict(data[key], **value)
            else:
                data[key] = value
        data['configuration_input'] = {'path': str(override_path.resolve()),
            'sha256': hashlib.sha256(override_raw).hexdigest(), 'scope': 'user supplied finite geometry; not primary verification'}
    if data.get('profile_id') != profile_id or data.get('schema_version') != '1.0':
        raise ContractError('Unsupported HC synthetic profile identity/schema')
    footprints = dict(data.get('footprints', {}))
    if arbor_profile_id in ('central_OPL_equal_area_W2000','W2000_equal_area_bounded_edges_047_104'):
        if h1_radius_mm is not None or h2_radius_mm is not None or 'footprints' in overrides:
            raise ContractError('central_OPL_equal_area_W2000 conflicts with manual radii or JSON footprints; select explicit_footprints to configure different geometry')
        if any(row is not None for key, row in footprints.items() if key != 'by_hc_id') or footprints.get('by_hc_id'):
            raise ContractError('Named central HC arbor profile cannot silently replace configured base footprints')
        arbor_path = Path(root) / ('configurations/hc_arbor_central_W2000_v1.json' if arbor_profile_id=='central_OPL_equal_area_W2000'
            else 'configurations/hc_arbor_W2000_equal_area_bounded_edges_047_104.json')
        arbor_raw = arbor_path.read_bytes()
        arbor_data = json.loads(arbor_raw)
        if (arbor_data.get('schema_version') != '1.0' or
                arbor_data.get('arbor_profile_id') != arbor_profile_id or
                profile_id not in arbor_data.get('compatible_geometry_profile_ids', ())):
            raise ContractError('Unsupported central HC arbor profile identity/schema')
        footprints = dict(arbor_data['footprints'])
        data['arbor_source_profile'] = dict(arbor_data,
            source_configuration={'path': str(arbor_path),
                'sha256': hashlib.sha256(arbor_raw).hexdigest(),
                'identity_scope': 'named calculation-support descriptor, not independent primary verification'})
        data['contact_source_radial_support_mm'] = arbor_data['contact_source_radial_support_mm']
        if arbor_profile_id=='W2000_equal_area_bounded_edges_047_104':
            if (arbor_data.get('source_anchor_band_mm')!=[.5,1.] or arbor_data.get('application_anchor_band_mm')!=[.47,1.04]
                or arbor_data.get('auto_extend') is not False or
                any(arbor_data['footprints'][t]['radius_mm']!=r for t,r in (('H1',.00804),('H2',.01222)))):
                raise ContractError('Bounded constant-footprint contract identity differs; no automatic extension')
            data.update(anchor_applicability_band_mm=arbor_data['application_anchor_band_mm'],
                source_anchor_band_mm=arbor_data['source_anchor_band_mm'],contact_policy='finite_geometry_first_v1',
                final_contact_distance_mm=.04,divergence_profile=None,require_owner_quota=False)
        data['arbor_configuration_status'] = ('NAMED_BOUNDED_ANCHOR_EXTRAPOLATION; fixed footprints, unmeasured edge anchors'
            if arbor_profile_id=='W2000_equal_area_bounded_edges_047_104' else
            'NAMED_EQUAL_AREA_CALCULATION_SUPPORT; no measured maximum reach or sub-0.5 mm extension')
        data['allowed_approximation_ids'] = list(dict.fromkeys(
            data.get('allowed_approximation_ids', []) + arbor_data['allowed_approximation_ids']))
    data['arbor_profile_id'] = arbor_profile_id
    for subtype, radius in (('H1', h1_radius_mm), ('H2', h2_radius_mm)):
        if radius is not None:
            if isinstance(radius, bool) or not isinstance(radius, (float, int)) or not math.isfinite(radius) or radius <= 0:
                raise ContractError('Explicit HC arbor radius must be finite and positive, in mm')
            footprints[subtype] = {'kind': 'disk', 'radius_mm': float(radius),
                'unit':'mm','coordinate_frame':'OPL',
                'source_profile_id': 'user_configured_finite_HC_footprint_v1',
                'evidence_class': 'project_approximation',
                'approximation_id': 'HC_user_finite_footprint_v1',
                'scope': 'explicit effective arbor footprint; no measured central morphology claim'}
    data['footprints'] = footprints
    # New user-supplied morphology cannot acquire a measurement label through
    # a source-name string. This identity gate is not biological validation.
    from .hc_arbor import validate_footprints
    validate_footprints(footprints, require_configured=False)
    by_id = footprints.get('by_hc_id', {})
    rows = [row for key, row in footprints.items() if key != 'by_hc_id'] + list(by_id.values())
    if data.get('divergence_profile') is not None:
        rows.append(data['divergence_profile'])
    verified_material = []
    for row in rows:
        if row is None or row.get('evidence_class') not in ('measurement', 'published_fit'):
            continue
        proof = row.get('primary_verification', {})
        if proof.get('verified') is not True or not all(proof.get(key) for key in (
                'material_path', 'material_sha256', 'locator', 'condition', 'geometry_sha256')):
            raise ContractError('Scientific HC geometry requires complete frozen primary verification')
        material = Path(proof['material_path'])
        if not material.is_absolute():material = Path(root) / material
        if not material.is_file() or hashlib.sha256(material.read_bytes()).hexdigest() != proof['material_sha256']:
            raise ContractError('HC primary geometry material identity mismatch')
        descriptor = {key: value for key, value in row.items() if key not in (
            'source_profile_id', 'evidence_class', 'approximation_id', 'source_locator',
            'primary_verification', 'cross_animal_transfer', 'cross_species_transfer', 'meridian_transfer')}
        geometry_hash = hashlib.sha256(json.dumps(descriptor, sort_keys=True,
            ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
        if geometry_hash != proof['geometry_sha256']:
            raise ContractError('HC primary verification does not bind this exact finite geometry')
        verified_material.append({'source_profile_id':row['source_profile_id'],
            'material_sha256':proof['material_sha256'],'geometry_sha256':geometry_hash,
            'locator':proof['locator'],'condition':proof['condition'],
            'scope':'file and exact geometry identity checked; physiological validation remains separate'})
    data['primary_geometry_identity_checks'] = verified_material
    data['base_configuration'] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    data['implementation_status'] = 'IMPLEMENTED_NOT_RUN'
    data['resolved_configuration_sha256'] = hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    return data


def joint_hc_geometry(root, mapped_domain, seed, max_cells, configuration):
    """Generate one canonical physical instance and crop a shared-ID active view."""
    from .hc_mosaic import HCDensityProfile, generate_joint_hc
    from .hc_arbor import footprint_extent_mm
    from .regional_support import radial_rectangle_range
    extent = footprint_extent_mm(configuration['footprints'], require_configured=True)
    profile = HCDensityProfile(root, profile_data=configuration)
    pair_halo = profile.max_interaction_support_mm()
    x0, x1, y0, y1 = mapped_domain.bounds_mm
    needed = (x0-extent, x1+extent, y0-extent, y1+extent)
    # The active contact view has HC somas capable of reaching the pedicle
    # patch. Its canonical context covers all pair supports and complete arbor
    # envelopes; this does not supply the ungenerated exterior cone inputs.
    context = max(pair_halo, extent)
    required = (needed[0]-context, needed[1]+context,
                needed[2]-context, needed[3]+context)
    bounds = configuration.get('canonical_bounds_mm')
    if bounds is None:
        bounds = required
        policy = 'resolved input patch canonical instance; changed domain creates another identity'
    else:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4 or not all(
                isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in bounds):
            raise ContractError('Canonical HC physical bounds require four finite mm values')
        bounds = tuple(bounds)
        if not (bounds[0] <= required[0] < required[1] <= bounds[1] and
                bounds[2] <= required[2] < required[3] <= bounds[3]):
            raise ContractError('Canonical HC instance does not cover mapped OPL/contact view plus finite arbor and pair-interaction context; no halo clipping')
        policy = 'explicit fixed canonical instance; tile views share its IDs and locations'
    lower, upper = radial_rectangle_range(bounds)
    if lower < 0 or upper > configuration['support_radius_mm']:
        raise ContractError('HC canonical domain/arbor buffer exceeds the 0..7.1 mm synthetic source support; no endpoint clamping or halo clipping')
    owner = mapped_domain.owner_bounds_mm
    canonical = RetinalDomain(tuple(bounds), owner, lambda xy: np.ones(len(xy), bool),
        'HC_physical_rectangular_tissue_mask_v1', 'hc_joint_canonical_physical')
    view = RetinalDomain(needed, owner, canonical.mask, canonical.mask_source_id, 'hc_joint_active_OPL_view')
    master = generate_joint_hc(profile, canonical, seed, max_cells, options=configuration['generator'])
    active = master.crop(view)
    result = []
    for subtype in ('H1', 'H2'):
        selected = np.asarray(active.types) == subtype
        ids = tuple(i for i, keep in zip(active.cell_ids, selected) if keep)
        points = active.positions.values[selected]
        diagnostics = dict(active.diagnostics, subtype=subtype, active_count=len(ids),
            owner_count=int(active.owner[selected].sum()), halo_count=int((~active.owner[selected]).sum()),
            mass_basis='canonical per-type partition masses; crop count is actual retained instance count')
        result.append((subtype, Mosaic(Quantity(points, 'mm', 'anatomical_position',
            'project_approximation', 'anatomical_retina'), ids, active.owner[selected], diagnostics)))
    audit = {'profile_id': configuration['profile_id'], 'configuration': configuration,
        'canonical_bounds_mm': list(bounds), 'active_bounds_mm': list(needed),
        'owner_bounds_mm': list(owner), 'domain_policy': policy,
        'soma_plane': 'anatomical_retina physical projection', 'arbor_anchor_plane': 'OPL',
        'anchor_policy': 'co_located_lumped_anchor', 'HC_is_displaced_by_cone_map': False,
        'master': master.diagnostics, 'active': active.diagnostics,
        'canonical_instance':{'cell_ids':list(master.cell_ids),'types':np.asarray(master.types).tolist(),
            'soma_xy_mm':master.positions.values.tolist(),'frame':'anatomical_retina',
            'active_cell_ids':list(active.cell_ids),'view_operation':'crop same complete instance; no resampling'},
        'arbor_buffer_mm': extent, 'mosaic_pair_support_halo_mm': pair_halo,
        'canonical_context_buffer_mm': context,
        'boundary_policy': 'finite induced graph; ungenerated cone input beyond active RF domain is audited',
        'density_claim': 'cross-animal source-constrained synthetic field, not one measured whole retina',
        'validation_status': 'IMPLEMENTED_NOT_RUN'}
    return result, audit


def hc_contact_runtime_view(population, feedforward, feedback,*,zero_contact_certificate=None):
    """Select contact participants without altering the generated tissue instance.

    The existing native dynamics require unit-DC input rows. An HC with no
    generated cone contact cannot affect this finite graph (HC gap candidates
    are not dynamical edges), so its anatomy remains in the exported instance
    while it has no state in this runtime view. No cone or anatomical HC is
    resampled, deleted from the instance, or given a fabricated contact.
    """
    from scipy import sparse
    A = sparse.csr_matrix(feedforward)
    B = sparse.csr_matrix(feedback)
    ids = population.cells.cell_id
    if A.shape[0] != len(ids) or B.shape != (A.shape[1], A.shape[0]):
        raise ContractError('HC runtime view matrix dimensions do not follow supplied cell IDs')
    if np.any(A.data < 0) or np.any(B.data < 0) or not (
            np.all(np.isfinite(A.data)) and np.all(np.isfinite(B.data))):
        raise ContractError('HC runtime contacts require finite nonnegative weights')
    sums = np.asarray(A.sum(axis=1)).ravel()
    selected = sums > 0
    feedback_sums = np.asarray(B.sum(axis=1)).ravel()
    if not np.any(selected):
        raise ContractError('HC_CONTACT_RUNTIME_UNCONFIGURED: no HC has cone input in the finite generated contact graph; generated anatomy is not HC-free')
    if not np.allclose(sums[selected], 1., rtol=0, atol=1e-12):
        raise ContractError('Native HC runtime requires unit pooling of actual finite contacts')
    missing = np.flatnonzero(feedback_sums == 0)
    if len(missing) and zero_contact_certificate is None:
        raise ContractError('HC_CONTACT_RUNTIME_UNCOVERED_CONES: '+str(len(missing))+
            ' generated cones have no finite HC feedback contact (indices '+
            ','.join(map(str, missing[:12]))+'); configure a supported arbor/input domain. No nearest fill or implicit background feedback is inserted')
    if zero_contact_certificate is not None:
        from .finite_feedback_native import POLICY
        if (zero_contact_certificate.get('policy_id')!=POLICY or zero_contact_certificate.get('support_complete') is not True
            or zero_contact_certificate.get('zero_cone_indices')!=missing.tolist()):
            raise ContractError('Exact complete-support zero-contact certificate required by runtime view')
    if not np.allclose(feedback_sums[feedback_sums>0], 1., rtol=0, atol=1e-12):
        raise ContractError('Native cone feedback requires unit pooling of actual finite contacts')
    # A zero input row must also have no reciprocal feedback edge.
    if B[:, ~selected].nnz:
        raise ContractError('HC forward/feedback topology disagrees on disconnected cells')
    if ((A.T!=0)!=(B!=0)).nnz:raise ContractError('HC forward/feedback topology must share all real partners')
    indices = np.flatnonzero(selected)
    def items(values):
        return tuple(values[i] for i in indices)
    def subset_quantity(value):
        return None if value is None else Quantity(value.values[indices], value.unit,
            value.meaning, value.evidence_tag, value.coordinate_frame)
    cells = population.cells
    cells = replace(cells, **{key: items(getattr(cells, key)) for key in (
        'cell_id', 'cell_type', 'subtype', 'species_profile_id', 'hemiretina',
        'meridian', 'parameter_profile_id', 'evidence_tags')},
        soma_position=subset_quantity(cells.soma_position),
        rf_position=subset_quantity(cells.rf_position),
        eccentricity=subset_quantity(cells.eccentricity))
    view = replace(population, cells=cells, regions=items(population.regions),
        **{key: subset_quantity(getattr(population, key)) for key in (
            'anatomical_positions', 'receptive_field_positions', 'pedicle_positions',
            'terminal_positions', 'dendritic_positions')})
    audit = {'policy': 'finite_contact_participant_runtime_view_v1',
        'generated_active_cell_ids': list(ids), 'runtime_cell_ids': list(view.cells.cell_id),
        'geometry_only_cell_ids': [cell_id for cell_id, keep in zip(ids, selected) if not keep],
        'generated_active_count': len(ids), 'runtime_count': len(indices),
        'geometry_only_reason': 'no generated cone input in this finite graph; no HC gap edges connected to dynamics',
        'density_instance_thinned': False, 'cone_population_changed': False,
        'native_kinetics_changed': False, 'implicit_background_feedback': False,
        'geometry_only_state': 'NOT_INSTANTIATED; no simulated HC response claimed',
        'implementation_status': 'IMPLEMENTED_NOT_RUN'}
    if zero_contact_certificate is not None:
        audit.update(zero_contact_execution_policy=zero_contact_certificate['policy_id'],
            zero_feedback_cone_ids=zero_contact_certificate['zero_cone_ids'],
            support_complete_zero_feedback=True,all_cones_and_dynamic_halo_retained=True)
    return view, A[selected, :].tocsr(), B[:, selected].tocsr(), selected, audit
