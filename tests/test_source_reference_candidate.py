from pathlib import Path
import copy
import json
import numpy as np
import pytest
from tests.test_calibration_modules import circuit
from retina_sim.calibration_modules import merge_parameter_package
from retina_sim.calibration_proposals import GROUPS
from retina_sim.source_reference_candidate import read_source_candidate
from retina_sim.source_reference_candidate import CROSS_TYPE_TARGETS,TRANSFER_EVIDENCE
from retina_sim.contracts import ContractError,SolverConfig,canonical_hash
from retina_sim.hybrid import HybridRunner

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'configurations/on_parasol_literature_candidate_v1.json'


def test_source_transfer_only_changes_named_rgc_and_rejects_old_checkpoint(circuit):
    old,_,audit=circuit;d,path=read_source_candidate(ROOT,PATH)
    runner=HybridRunner(old,compute_backend='python')
    checkpoint=runner.checkpoint(runner.initialize(SolverConfig(input_interpolation='zero_order_hold')))
    new=merge_parameter_package(old,audit['registry'],d,applicability=d['applicability'],
        allowed_fields=['alpha','mu','beta'],proposal_path=path)
    assert old.assembly_hash!=new.assembly_hash
    for cid,c in old.components.items():
        revised=new.components[cid]
        if cid!='parasol_RGC_ON':
            assert canonical_hash(c.parameters.records)==canonical_hash(revised.parameters.records)
            assert canonical_hash(c.parameters.profile)==canonical_hash(revised.parameters.profile)
        else:
            before={r.parameter_id:r for r in c.parameters.records}
            for r in revised.parameters.records:
                if r.parameter_id in d['changes']:
                    np.testing.assert_array_equal(r.value.values,np.full(r.value.values.shape,d['changes'][r.parameter_id]['value']))
                    assert r.evidence_class=='project_approximation'
                else:assert canonical_hash(before[r.parameter_id])==canonical_hash(r)
    with pytest.raises(ContractError):HybridRunner(new,compute_backend='python').restore(checkpoint)
    assert not new.parameter_package_merge['production_default']
    assert {'alpha','mu','beta'} not in GROUPS['HistoryConductanceRGC']


@pytest.mark.parametrize('edit',[
    lambda d:d.update(component_id='midget_RGC_ON'),
    lambda d:d['changes']['alpha'].update(value=91.),
    lambda d:d['changes']['mu'].update(unit='mV'),
    lambda d:d.update(default_enabled=True),
    lambda d:d['changes'].update(EL={'value':-.06,'unit':'V','meaning':'leak_reversal'})])
def test_source_route_rejects_scope_or_parameter_mutation(tmp_path,edit):
    d=json.loads(PATH.read_bytes());edit(d)
    # Use a workspace temporary path: candidate sources are required to stay local.
    p=ROOT/'tmp/source_candidate_test_mutation.json'
    try:
        p.write_text(json.dumps(d),encoding='utf-8')
        with pytest.raises(ContractError):read_source_candidate(ROOT,p)
    finally:p.unlink(missing_ok=True)


@pytest.mark.parametrize('cid',CROSS_TYPE_TARGETS)
def test_cross_type_candidate_keeps_all_other_profiles_and_requires_lower_evidence(circuit,cid):
    old,_,audit=circuit;path=ROOT/'configurations'/(cid+'_literature_candidate_v1.json')
    d,path=read_source_candidate(ROOT,path)
    assert d['transfer_evidence']==TRANSFER_EVIDENCE
    new=merge_parameter_package(old,audit['registry'],d,applicability=d['applicability'],
        allowed_fields=['alpha','mu','beta'],proposal_path=path)
    for key,c in old.components.items():
        if key!=cid:
            assert canonical_hash(c.parameters.records)==canonical_hash(new.components[key].parameters.records)
            assert canonical_hash(c.parameters.profile)==canonical_hash(new.components[key].parameters.profile)
    assert set(new.parameter_package_merge['changed_fields'])=={'alpha','mu','beta'}
    runner=HybridRunner(old,compute_backend='python');cp=runner.checkpoint(runner.initialize())
    with pytest.raises(ContractError):HybridRunner(new,compute_backend='python').restore(cp)


@pytest.mark.parametrize('edit',[lambda d:d['transfer_evidence'].update(target_transfer_support='A: fitted'),
    lambda d:d['source'].update(source_id='LATIMER2019_TRANSFERRED_OUTPUT_V1')])
def test_cross_type_route_rejects_evidence_promotion_or_ambiguous_identity(edit):
    d=json.loads((ROOT/'configurations/midget_RGC_ON_literature_candidate_v1.json').read_bytes());edit(d)
    p=ROOT/'tmp/cross_type_mutation.json'
    try:
        p.write_text(json.dumps(d),encoding='utf8')
        with pytest.raises(ContractError):read_source_candidate(ROOT,p)
    finally:p.unlink(missing_ok=True)
