"""Local-patch requests and source preflight; no geometry or solver fallback.

The fine-source closure is currently absent from the delivered workspace. This
module deliberately does not turn an aggregate graph into a fine-cell model.
"""
from copy import deepcopy
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

from .contracts import ContractError, canonical_hash
from .spatial_group_graph import digest
from .spatial_group_model import POPULATIONS, RGC, _verify_artifact

SCHEMA = 'local-retinal-patch-request-1'
BOUNDARY = 'exact_dependency_closure_required_v1'
BOUNDED_BOUNDARY = 'bounded_background_halo_v2'
FULL_REFERENCE = 'full_reference_v2'


def _positive(value, name, *, zero=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (not zero and value == 0)):
        raise ContractError('Finite ' + ('nonnegative ' if zero else 'positive ') + name + ' required')


def _binding(value):
    if (not isinstance(value, dict) or set(value) != {'path', 'sha256'}
            or not isinstance(value['path'], str) or not value['path']
            or PureWindowsPath(value['path']).drive or PureWindowsPath(value['path']).root
            or PurePosixPath(value['path']).is_absolute()
            or not isinstance(value['sha256'], str)
            or re.fullmatch('[0-9a-f]{64}', value['sha256']) is None):
        raise ContractError('Explicit relative path and SHA256 binding required')


def validate_patch_config(raw):
    """Validate a request, without asserting source, boundary or runtime readiness."""
    required = {'schema', 'eccentricity_deg', 'target_rgc_type', 'target_rgc_id',
                'patch_geometry', 'seed', 'input_identity', 'observation_targets',
                'boundary_policy', 'solver_backend', 'source_artifact',
                'position_provenance', 'target_provenance'}
    if (not isinstance(raw, dict) or not required <= set(raw)
            or set(raw) - required - {'azimuth_deg', 'rf_center_deg', 'representation'}
            or raw['schema'] != SCHEMA):
        raise ContractError('Complete versioned local-patch request required; unknown fields rejected')
    if raw.get('representation', 'original_fine_required_v1') not in ('original_fine_required_v1', 'frozen_representative_groups_v1'):
        raise ContractError('Explicit supported source representation required')
    _positive(raw['eccentricity_deg'], 'functional RF eccentricity', zero=True)
    if raw['target_rgc_type'] not in RGC:
        raise ContractError('Unsupported RGC type')
    if raw['target_rgc_id'] is not None and (not isinstance(raw['target_rgc_id'], str) or not raw['target_rgc_id']):
        raise ContractError('Optional target ID must be an actual source-cell identity')
    if ('azimuth_deg' in raw) == ('rf_center_deg' in raw):
        raise ContractError('Supply exactly one of azimuth_deg or functional rf_center_deg')
    if 'azimuth_deg' in raw:
        angle = raw['azimuth_deg']
        if type(angle) not in (int, float) or not math.isfinite(angle):
            raise ContractError('Finite azimuth required')
    else:
        xy = raw['rf_center_deg']
        if (not isinstance(xy, list) or len(xy) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in xy)
                or not math.isclose(math.hypot(*xy), raw['eccentricity_deg'], rel_tol=0, abs_tol=1e-10)):
            raise ContractError('Functional RF coordinates and eccentricity disagree')
    geometry = raw['patch_geometry']
    if not isinstance(geometry, dict) or set(geometry) != {'shape', 'core_radius_deg', 'support_radius_deg'} or geometry['shape'] != 'disk':
        raise ContractError('Explicit disk request with core/support radii required')
    for key in ('core_radius_deg', 'support_radius_deg'):
        _positive(geometry[key], key)
    if geometry['support_radius_deg'] < geometry['core_radius_deg']:
        raise ContractError('Support must contain the core')
    if type(raw['seed']) is not int or raw['seed'] < 0:
        raise ContractError('Nonnegative integer seed required')
    if raw['solver_backend'] != 'scipy_cpu_v1' or raw['boundary_policy'] not in (BOUNDARY, BOUNDED_BOUNDARY, FULL_REFERENCE):
        raise ContractError('Original CPU backend and a supported explicit boundary policy required')
    _binding(raw['source_artifact'])
    _binding(raw['input_identity'])
    observations = raw['observation_targets']
    if (not isinstance(observations, list) or not observations
            or any(not isinstance(x, str) or x not in POPULATIONS for x in observations)
            or len(set(observations)) != len(observations)):
        raise ContractError('Unique known populations for full effective-state observation required')
    for key in ('position_provenance', 'target_provenance'):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ContractError('Explicit ' + key + ' required')
    return deepcopy(raw)


def load_patch_config(path):
    return validate_patch_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def requested_rf_center(config):
    config = validate_patch_config(config)
    if 'rf_center_deg' in config:
        return tuple(config['rf_center_deg'])
    angle = math.radians(config['azimuth_deg'])
    return (config['eccentricity_deg'] * math.cos(angle),
            config['eccentricity_deg'] * math.sin(angle))


def expand_patch_request(config, support_radius_deg):
    """Nested geometric REQUEST only; this does not construct or qualify a graph."""
    result = validate_patch_config(config)
    _positive(support_radius_deg, 'support radius')
    if support_radius_deg <= result['patch_geometry']['support_radius_deg']:
        raise ContractError('Expansion must strictly increase the support radius')
    result['patch_geometry']['support_radius_deg'] = support_radius_deg
    return result


def _inspect_binding(root, binding, role):
    _binding(binding)
    path = (root / binding['path']).resolve()
    if not path.is_relative_to(root):
        raise ContractError('Source binding escapes the project: ' + binding['path'])
    actual = digest(path) if path.is_file() else None
    return dict(role=role, **binding, actual_sha256=actual,
                status='MISSING' if actual is None else 'MATCH' if actual == binding['sha256'] else 'HASH_MISMATCH')


def audit_patch_sources(config, repo_root):
    """Verify the frozen reference and list exact missing fine-source bindings.

    Matching top-level manifests is a necessary precondition. It never certifies
    their transitive payloads, input remapping, model construction or a boundary.
    """
    config = validate_patch_config(config)
    root = Path(repo_root).resolve()
    reference = _inspect_binding(root, config['source_artifact'], 'frozen_reference_manifest')
    if reference['status'] != 'MATCH':
        raise ContractError('Frozen reference manifest identity differs')
    artifact_manifest = (root / config['source_artifact']['path']).resolve()
    if artifact_manifest.name != 'manifest.json':
        raise ContractError('source_artifact must bind the actual frozen manifest.json')
    artifact = artifact_manifest.parent
    _, manifest, identity = _verify_artifact(artifact)
    scale = manifest['configuration']['mm_per_degree']
    _positive(scale, 'source mm/degree conversion')
    if config['eccentricity_deg'] > manifest['configuration']['output_radius_deg']:
        raise ContractError('Requested functional eccentricity exceeds source output scope')
    rows = [_inspect_binding(root, manifest['configuration']['graph'], 'fine_global_graph'),
            _inspect_binding(root, config['input_identity'], 'input_bundle_manifest')]
    for name, population in manifest['populations'].items():
        for field in ('source_table', 'source_keys', 'coordinate'):
            binding = {k: population[field][k] for k in ('path', 'sha256')}
            rows.append(_inspect_binding(root, binding, name + ':' + field))
    for route in manifest['routes']:
        rows.append(_inspect_binding(root, route['source'], 'fine_route:' + route['name']))
    missing = [x for x in rows if x['status'] != 'MATCH']
    return dict(schema='local-retinal-patch-source-audit-1',
                status='BLOCKED_MISSING_OR_CHANGED_SOURCES' if missing else 'TOP_LEVEL_SOURCES_MATCH_BUILD_NOT_IMPLEMENTED',
                request_sha256=canonical_hash(config), source_artifact_sha256=identity,
                requested_rf_center_deg=requested_rf_center(config), mm_per_degree=scale,
                mm_per_degree_source='frozen reference configuration; no new anatomical registration',
                source_bindings=rows, missing_or_changed_count=len(missing),
                source_dependency_payload_closure_verified=False, runtime_ready=False,
                graph_constructed=False, model_forward_started=False,
                boundary_qualified=False, input_remapping_qualified=False,
                checkpoint_restore_verified=False, fit_eligible=False, default_adopted=False,
                representation=config.get('representation', 'original_fine_required_v1'),
                fine_reconstruction_blocked=bool(missing),
                aggregate_reference_available=True,
                blocking_conditions=['FINE_SOURCE_PROVENANCE_INCOMPLETE'] if missing else [],
                note='Source inspection alone never admits simulation; aggregate mode requires its independently verified dependency-closed graph and IVP.')
