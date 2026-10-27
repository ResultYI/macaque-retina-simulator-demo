"""Local-patch request/source guards only; no graph build or neural execution."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from retina_sim.contracts import ContractError, canonical_hash
from retina_sim import local_patch as patch


def request():
    return dict(schema=patch.SCHEMA, eccentricity_deg=4.59, azimuth_deg=0.,
        target_rgc_type='parasol_RGC_ON', target_rgc_id=None,
        patch_geometry=dict(shape='disk', core_radius_deg=.05, support_radius_deg=.3),
        seed=17, input_identity=dict(path='input/manifest.json', sha256='a'*64),
        observation_targets=['cones', 'HC', 'rods', 'rod_BC', 'AII_AC', 'parasol_RGC_ON'],
        boundary_policy=patch.BOUNDARY, solver_backend='scipy_cpu_v1',
        source_artifact=dict(path='reference/manifest.json', sha256='b'*64),
        position_provenance='representative engineering azimuth; no cell registration',
        target_provenance='explicit provisional type mapping; not teacher electrophysiology')


@pytest.mark.parametrize('ecc,azimuth,xy', [
    (0., 37., (0., 0.)), (2., 90., (0., 2.)), (4.59, 180., (-4.59, 0.))])
def test_functional_eccentricity_and_azimuth_are_not_fixed(ecc, azimuth, xy):
    raw=request(); raw.update(eccentricity_deg=ecc, azimuth_deg=azimuth)
    assert patch.requested_rf_center(raw)==pytest.approx(xy, abs=1e-12)
    valid=patch.validate_patch_config(raw)
    valid['observation_targets'].append('midget_RGC_ON')
    assert valid['observation_targets']!=raw['observation_targets']


def test_explicit_functional_rf_position_and_file_roundtrip(tmp_path):
    raw=request(); raw.pop('azimuth_deg')
    raw.update(eccentricity_deg=5., rf_center_deg=[3., 4.], target_rgc_id='actual-source-row:12')
    path=tmp_path/'request.json'; path.write_text(json.dumps(raw), encoding='utf-8-sig')
    assert patch.load_patch_config(path)==raw
    assert patch.requested_rf_center(raw)==(3., 4.)
    raw['rf_center_deg']=[3., 4.1]
    with pytest.raises(ContractError, match='eccentricity disagree'):
        patch.validate_patch_config(raw)


@pytest.mark.parametrize('field,value', [
    ('eccentricity_deg', True), ('eccentricity_deg', float('nan')),
    ('eccentricity_deg', float('inf')), ('eccentricity_deg', -1.),
    ('azimuth_deg', False), ('azimuth_deg', float('nan')),
    ('seed', True), ('seed', -1), ('seed', 2.),
    ('solver_backend', 'gpu_dop853'), ('boundary_policy', 'truncate_and_renormalize'),
    ('target_rgc_type', 'unknown'), ('target_rgc_id', ''),
    ('observation_targets', ['cones', 'cones']), ('observation_targets', ['missing']),
    ('position_provenance', ''), ('target_provenance', ' ')])
def test_unsafe_or_unsupported_requests_are_rejected(field, value):
    raw=request(); raw[field]=value
    with pytest.raises(ContractError):patch.validate_patch_config(raw)


@pytest.mark.parametrize('geometry', [
    dict(shape='disk', core_radius_deg=True, support_radius_deg=.3),
    dict(shape='disk', core_radius_deg=.1, support_radius_deg=float('nan')),
    dict(shape='disk', core_radius_deg=.2, support_radius_deg=.1),
    dict(shape='disk', core_radius_deg=0., support_radius_deg=.3)])
def test_invalid_core_or_support_is_rejected(geometry):
    raw=request(); raw['patch_geometry']=geometry
    with pytest.raises(ContractError):patch.validate_patch_config(raw)


def test_position_exclusivity_and_unknown_fields_are_rejected():
    for changed in ('both', 'neither', 'extra'):
        raw=request()
        if changed=='both':raw['rf_center_deg']=[4.59, 0.]
        elif changed=='neither':raw.pop('azimuth_deg')
        else:raw['silently_drop_HC']=True
        with pytest.raises(ContractError):patch.validate_patch_config(raw)


def test_support_expansion_keeps_same_core_target_sources_and_seed():
    raw=request(); before=deepcopy(raw)
    expanded=patch.expand_patch_request(raw, .6)
    assert raw==before
    assert expanded['patch_geometry']['support_radius_deg']==.6
    raw_core=deepcopy(raw); expanded_core=deepcopy(expanded)
    raw_core['patch_geometry'].pop('support_radius_deg')
    expanded_core['patch_geometry'].pop('support_radius_deg')
    assert canonical_hash(raw_core)==canonical_hash(expanded_core)
    for radius in (.3, .2, True, float('nan')):
        with pytest.raises(ContractError):patch.expand_patch_request(raw, radius)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    root=tmp_path/'project'; root.mkdir()
    def binding(relative, content):
        path=root/relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf8')
        return dict(path=relative, sha256=patch.digest(path))
    raw=request()
    raw['source_artifact']=binding('reference/manifest.json', 'frozen reference bytes')
    raw['input_identity']=binding('input/manifest.json', 'identity bound input')
    populations={name:{key:binding(f'fine/{name}_{key}.json', name+':'+key)
        for key in ('source_table', 'source_keys', 'coordinate')} for name in patch.POPULATIONS}
    manifest=dict(configuration=dict(mm_per_degree=.3, output_radius_deg=6.,
        graph=binding('fine/graph.json', 'original graph identity')),
        populations=populations, routes=[dict(name='cones_to_HC',
            source=binding('fine/cones_to_HC.json', 'original weighted route'))])
    verified=[]
    def verify(directory):
        verified.append(Path(directory))
        return Path(directory), manifest, 'immutable-reference-identity'
    monkeypatch.setattr(patch, '_verify_artifact', verify)
    return root, raw, manifest, verified


def test_all_source_roles_and_bindings_are_retained_without_claiming_runtime(sources):
    root, raw, manifest, verified=sources
    result=patch.audit_patch_sources(raw, root)
    assert verified==[root/'reference']
    assert result['status']=='TOP_LEVEL_SOURCES_MATCH_BUILD_NOT_IMPLEMENTED'
    assert len(result['source_bindings'])==2+3*len(patch.POPULATIONS)+len(manifest['routes'])
    expected={name+':'+key:binding for name,pop in manifest['populations'].items()
        for key,binding in pop.items()}
    expected['fine_global_graph']=manifest['configuration']['graph']
    expected['input_bundle_manifest']=raw['input_identity']
    expected['fine_route:cones_to_HC']=manifest['routes'][0]['source']
    for row in result['source_bindings']:
        assert row['path']==expected[row['role']]['path']
        assert row['sha256']==expected[row['role']]['sha256']
        assert row['actual_sha256']==row['sha256'] and row['status']=='MATCH'
    for key in ('runtime_ready', 'source_dependency_payload_closure_verified',
                'graph_constructed', 'model_forward_started', 'boundary_qualified',
                'input_remapping_qualified', 'checkpoint_restore_verified', 'fit_eligible', 'default_adopted'):
        assert result[key] is False
    assert result['missing_or_changed_count']==0


def test_missing_and_changed_sources_are_separately_evidenced(sources):
    root, raw, manifest, _=sources
    missing=manifest['populations']['cones']['source_keys']
    (root/missing['path']).unlink()
    changed=manifest['routes'][0]['source']
    (root/changed['path']).write_text('changed upstream source', encoding='utf8')
    result=patch.audit_patch_sources(raw, root)
    failed={x['role']:x for x in result['source_bindings'] if x['status']!='MATCH'}
    assert result['missing_or_changed_count']==2 and not result['runtime_ready']
    assert failed['cones:source_keys']['status']=='MISSING'
    assert failed['cones:source_keys']['actual_sha256'] is None
    assert failed['fine_route:cones_to_HC']['status']=='HASH_MISMATCH'
    assert failed['fine_route:cones_to_HC']['sha256']==changed['sha256']
    assert result['status']=='BLOCKED_MISSING_OR_CHANGED_SOURCES'


@pytest.mark.parametrize('mode', ['missing', 'mismatch'])
def test_reference_manifest_must_match_before_any_assembly_verification(sources, mode):
    root, raw, _, verified=sources
    if mode=='missing':(root/raw['source_artifact']['path']).unlink()
    else:raw['source_artifact']['sha256']='0'*64
    with pytest.raises(ContractError, match='Frozen reference manifest identity differs'):
        patch.audit_patch_sources(raw, root)
    assert not verified


@pytest.mark.parametrize('role', ['reference', 'input', 'fine'])
def test_path_escape_is_rejected_for_each_source_layer(sources, role):
    root, raw, manifest, _=sources
    escape=dict(path='../outside.json', sha256='0'*64)
    if role=='reference':raw['source_artifact']=escape
    elif role=='input':raw['input_identity']=escape
    else:manifest['populations']['HC']['coordinate']=escape
    with pytest.raises(ContractError, match='escapes the project'):
        patch.audit_patch_sources(raw, root)


def test_eccentricity_beyond_frozen_source_scope_is_rejected(sources):
    root, raw, _, _=sources; raw['eccentricity_deg']=6.01
    with pytest.raises(ContractError, match='exceeds source output scope'):
        patch.audit_patch_sources(raw, root)

@pytest.mark.parametrize('path', ['C:/within/project/file.json', 'C:drive-relative.json',
    r'\\server\share\file.json', '/absolute/file.json', r'\rooted\file.json'])
def test_binding_requires_unambiguous_relative_paths(path):
    raw=request(); raw['input_identity']['path']=path
    with pytest.raises(ContractError, match='relative path'):
        patch.validate_patch_config(raw)

@pytest.mark.parametrize('xy', [[True, 0.], [0., False], [float('nan'), 0.], [0., float('inf')]])
def test_explicit_rf_coordinates_reject_bool_and_nonfinite(xy):
    raw=request(); raw.pop('azimuth_deg'); raw['rf_center_deg']=xy
    with pytest.raises(ContractError):patch.validate_patch_config(raw)


def test_reference_binding_cannot_substitute_another_file_in_manifest_directory(sources):
    root, raw, _, verified=sources
    substitute=root/'reference'/'unrelated.json'
    substitute.write_text('same directory, unrelated source identity', encoding='utf8')
    raw['source_artifact']=dict(path='reference/unrelated.json', sha256=patch.digest(substitute))
    with pytest.raises(ContractError, match='actual frozen manifest'):
        patch.audit_patch_sources(raw, root)
    assert not verified
