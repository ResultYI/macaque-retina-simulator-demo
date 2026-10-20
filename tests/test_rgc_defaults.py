"""Default adoption preserves numerical scope and binds its own evidence."""
from pathlib import Path
import hashlib,json
import numpy as np
import pytest
from tests.test_calibration_modules import circuit
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.rgc_defaults import apply_rgc_defaults,historical_reference_file,historical_project_default

ROOT=Path(__file__).resolve().parents[1]


def test_adopted_defaults_change_only_accepted_output_coefficients(circuit):
    original,channels,audit=circuit
    revised,_,new_audit=apply_rgc_defaults(circuit,ROOT)
    accepted={r['component_id'] for r in revised.accepted_rgc_output_defaults}
    assert 'parasol_RGC_ON' in accepted
    assert canonical_hash(original.spec.links)==canonical_hash(revised.spec.links)
    assert canonical_hash(original.spec.gap_junctions)==canonical_hash(revised.spec.gap_junctions)
    for cid,c in original.components.items():
        new=revised.components[cid]
        assert new.mechanism==c.mechanism
        old_records={r.parameter_id:r for r in c.parameters.records}
        for r in new.parameters.records:
            if cid in accepted and r.parameter_id in ('alpha','mu','beta'):
                assert r.evidence_class=='project_approximation'
            else:assert canonical_hash(r)==canonical_hash(old_records[r.parameter_id])
        if cid in accepted:
            assert new.parameters.conditions['source_reference_candidate']['production_default']
            for key,value in dict(alpha=90.,mu=-.053,beta=.00167).items():
                np.testing.assert_array_equal(getattr(new.native_model,key),value)
        else:assert canonical_hash(c.parameters.profile)==canonical_hash(new.parameters.profile)
    assert new_audit['runtime_profile']['accepted_RGC_output_defaults']==revised.accepted_rgc_output_defaults
    assert apply_rgc_defaults(circuit,ROOT,False) is circuit


def test_only_exact_historical_main_can_resolve_to_archive():
    expected='a02224d6ce920daad22310e39577f45ec1562fb9146b4e201155dd3b8be9bd20'
    p=historical_reference_file(ROOT,'configurations/retina_platform_v1.json',expected)
    assert p.parent.name=='history' and hashlib.sha256(p.read_bytes()).hexdigest()==expected
    cfg=historical_project_default(ROOT)
    assert cfg.circuit_configuration_path=='configurations/atlas_isetbio_source_stop_shared_v1.json'
    with pytest.raises(ContractError):historical_reference_file(ROOT,'configurations/effective_dynamics_v1.json',expected)


@pytest.mark.parametrize('failure',['acceptance_hash','failed_type_check','evidence_hash'])
def test_invalid_acceptance_stops_before_default_merge(tmp_path,monkeypatch,failure):
    from retina_sim import rgc_defaults
    (tmp_path/'configurations').mkdir();recipe_path=tmp_path/'recipe.json';recipe_path.write_text('{}')
    acceptance=dict(schema_version='rgc-engineering-acceptance-1.0',component_id='parasol_RGC_ON',
        physiological_calibration=False,fit_performed=False,
        passed_checks={k:True for k in ('mapping','bright_dark','contrast_ordering','own_mapping_recovery','single_cell_spatial_or_spectral')},
        evidence_files=[dict(path='evidence.json',sha256='0'*64)])
    (tmp_path/'evidence.json').write_text('{}')
    if failure=='failed_type_check':acceptance['passed_checks']['own_mapping_recovery']=False
    acceptance_path=tmp_path/'acceptance.json';acceptance_path.write_text(json.dumps(acceptance))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    policy=dict(schema_version='rgc-engineering-defaults-1.0',accepted_types={'parasol_RGC_ON':dict(
        recipe_path='recipe.json',recipe_sha256=sha(recipe_path),
        acceptance=dict(path='acceptance.json',sha256='0'*64 if failure=='acceptance_hash' else sha(acceptance_path)))})
    (tmp_path/'configurations/retina_platform_v1.json').write_text(json.dumps(dict(rgc_output_defaults=policy)))
    monkeypatch.setattr(rgc_defaults,'read_source_candidate',lambda *unused:(dict(component_id='parasol_RGC_ON'),recipe_path))
    monkeypatch.setattr(rgc_defaults,'merge_source_reference_candidate',lambda *a,**k:pytest.fail('Invalid evidence reached default merge'))
    with pytest.raises(ContractError):apply_rgc_defaults((None,None,{}),tmp_path)


@pytest.mark.parametrize('case',['hash','path','unsupported_global'])
def test_fixed_rod_reference_identity_and_route_are_not_silently_ignored(tmp_path,case):
    from retina_sim.atlas_population_network import atlas_population_network
    (tmp_path/'configurations').mkdir()
    raw=(ROOT/'configurations/rod_fixed_dark_reference_v1.json').read_bytes()
    (tmp_path/'configurations/rod.json').write_bytes(raw)
    binding=dict(path='configurations/rod.json',sha256=hashlib.sha256(raw).hexdigest())
    if case=='hash':binding['sha256']='0'*64
    if case=='path':binding['path']='../outside.json'
    (tmp_path/'configurations/retina_platform_v1.json').write_text(json.dumps(dict(rod_reference=binding)))
    (tmp_path/'configurations/global.json').write_text(json.dumps(dict(profile_id='atlas_isetbio_geometry_v1',storage_kind='global_disk_graph_v1')))
    message={'hash':'policy identity','path':'inside project','unsupported_global':'current frozen local main circuit'}[case]
    with pytest.raises(ContractError,match=message):
        atlas_population_network(tmp_path,execution_profile='atlas_isetbio_geometry_v1',configuration_path='configurations/global.json')
