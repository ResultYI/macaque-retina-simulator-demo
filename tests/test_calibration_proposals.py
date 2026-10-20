"""Restricted proposal plumbing tests; numerical changes are engineering QA only."""
from pathlib import Path
from dataclasses import replace
import copy
import json
import numpy as np
import pytest
from tests.test_calibration_modules import circuit, make_rgc, baselines, CONDITIONS
from tests.test_bc_ac_candidate import parent
from retina_sim.contracts import Quantity, ContractError, RunMode, canonical_hash, SolverConfig
from retina_sim.calibration_modules import build_module, gap_boundary_id, save_parameter_package, merge_parameter_package
from retina_sim.calibration_proposals import fitted_parameter_proposal
from retina_sim.bc_ac_candidate import CandidateProfile, migrate_bc_ac
from retina_sim.bc_ac_hypothesis import freeze_reference_coefficients, resolve_background_bindings, _legacy_reference_bindings
from retina_sim.assembly import RetinaBuilder
from retina_sim.hybrid import HybridRunner

ROOT=Path(__file__).resolve().parents[1]


def context(module,field,bounds):
    source=next(iter(module.registry.sources))
    return dict(observation_kind='engineering_reference',fit_performed=False,source_id=source,
        source_locator='Explicit synthetic contract QA, not observations',unit_index_status='verified',
        splits=dict(train=['a'*64],validation=['b'*64],test=[]),preprocessing='identity synthetic QA',
        metrics={'synthetic_error':0.},bounds={field:bounds},selection_history='No selection; contract test',
        uncertainty='No biological estimate or confidence interval')


def package(module,tmp_path,field,change,bounds):
    result=fitted_parameter_proposal(module,version='contract-qa-v1',new_profile_id='qa_changed_'+field,
        changes={field:change},allowed_fields=[field],applicability=CONDITIONS,fit_context=context(module,field,bounds))
    path=tmp_path/(field+'.json');save_parameter_package(result,path)
    return result,path


def test_mu_proposal_changes_only_named_field_and_invalidates_checkpoint(circuit,tmp_path):
    old,_,audit=circuit;module=make_rgc(circuit);cid='midget_RGC_ON';c=old.components[cid]
    q=c.parameters.quantity('mu','V','spike_voltage_offset')
    proposal,path=package(module,tmp_path,'mu',Quantity(q.values+.001,q.unit,q.meaning,q.evidence_tag),[-.09,-.04])
    runner=HybridRunner(old,compute_backend='python');checkpoint=runner.checkpoint(runner.initialize(SolverConfig(input_interpolation='zero_order_hold')))
    new=merge_parameter_package(old,audit['registry'],proposal,applicability=CONDITIONS,allowed_fields=['mu'],proposal_path=path)
    assert old.assembly_hash!=new.assembly_hash
    for key,component in old.components.items():
        if key!=cid:
            assert canonical_hash(component.parameters.records)==canonical_hash(new.components[key].parameters.records)
            assert canonical_hash(component.parameters.profile)==canonical_hash(new.components[key].parameters.profile)
        else:
            for row in component.parameters.records:
                updated=next(r for r in new.components[key].parameters.records if r.parameter_id==row.parameter_id)
                np.testing.assert_array_equal(updated.value.values,row.value.values+(.001 if row.parameter_id=='mu' else 0))
    assert not new.parameter_package_merge['system_acceptance']
    assert new.parameter_package_merge['invalidation']['checkpoint']
    again=merge_parameter_package(old,audit['registry'],proposal,applicability=CONDITIONS,allowed_fields=['mu'],proposal_path=path)
    assert again.assembly_hash==new.assembly_hash
    with pytest.raises(ContractError):HybridRunner(new,compute_backend='python').restore(checkpoint)
    with pytest.raises(ContractError,match='whitelist'):
        merge_parameter_package(old,audit['registry'],proposal,applicability=CONDITIONS,proposal_path=path)


def test_history_only_allows_amplitude_and_proposal_rejects_tampering(circuit,tmp_path):
    old,_,audit=circuit;module=make_rgc(circuit)
    p,path=package(module,tmp_path,'spike_history_kernel',{'scale':.9},[.5,1.5])
    new=merge_parameter_package(old,audit['registry'],p,applicability=CONDITIONS,
        allowed_fields=['spike_history_kernel'],proposal_path=path)
    np.testing.assert_array_equal(new.components['midget_RGC_ON'].native_model.history_kernel,
        old.components['midget_RGC_ON'].native_model.history_kernel*.9)
    bad=copy.deepcopy(p);bad['changes']['spike_history_kernel']['value'][0]*=1.1
    bad['content_hash']=canonical_hash({k:v for k,v in bad.items() if k!='content_hash'})
    with pytest.raises(ContractError,match='delta'):
        merge_parameter_package(old,audit['registry'],bad,applicability=CONDITIONS,
            allowed_fields=['spike_history_kernel'],proposal_path=path)
    with pytest.raises(ContractError,match='whitelist'):
        fitted_parameter_proposal(module,version='x',new_profile_id='bad_alpha',changes={'alpha':Quantity([1.],'Hz','x','x')},
            allowed_fields=['alpha'],applicability=CONDITIONS,fit_context=context(module,'alpha',[1,600]))


def test_split_overlap_wrong_units_and_wrong_surroundings_rejected(circuit,tmp_path):
    old,_,audit=circuit;module=make_rgc(circuit);q=old.components['midget_RGC_ON'].parameters.quantity('mu','V','spike_voltage_offset')
    ctx=context(module,'mu',[-.09,-.04]);ctx['splits']['test']=['a'*64]
    with pytest.raises(ContractError,match='disjoint'):
        fitted_parameter_proposal(module,version='v',new_profile_id='bad',changes={'mu':q},allowed_fields=['mu'],applicability=CONDITIONS,fit_context=ctx)
    with pytest.raises(ContractError):
        package(module,tmp_path,'mu',Quantity(q.values,'mV',q.meaning,q.evidence_tag),[-90,-40])
    p,path=package(module,tmp_path,'mu',Quantity(q.values+.001,q.unit,q.meaning,q.evidence_tag),[-.09,-.04])
    bad=copy.deepcopy(p);bad['base_snapshot']['system_compatibility']['connectivity_hash']='changed'
    base=bad['base_snapshot'];base['content_hash']=canonical_hash({k:v for k,v in base.items() if k!='content_hash'})
    bad['content_hash']=canonical_hash({k:v for k,v in bad.items() if k!='content_hash'})
    with pytest.raises(ContractError,match='Full-system'):
        merge_parameter_package(old,audit['registry'],bad,applicability=CONDITIONS,allowed_fields=['mu'],proposal_path=path)


def test_gE_change_preserves_frozen_K_half_and_K_change_initializes_with_new_K(parent,tmp_path):
    old,registry,channels=parent;old=RetinaBuilder(registry).build(old.spec,RunMode.REFERENCE)
    base=CandidateProfile(ROOT/'configurations/bc_ac_contact_first_hypothesis_v1.json');inputs={k:v.at(0) for k,v in channels.items()}
    targets=tuple(base.data['families']);ref=freeze_reference_coefficients(old,base,targets,inputs)
    before=resolve_background_bindings(old,base,targets,inputs,ref)
    data=copy.deepcopy(base.data);cid='midget_BC_OFF'
    data['families'][cid]['parameters']['gE']['value']*=1.4
    path=tmp_path/'changed_gE.json';path.write_text(json.dumps(data),encoding='utf8');changed=CandidateProfile(path)
    legacy=_legacy_reference_bindings(old,changed,targets,inputs)
    assert abs(legacy[cid]['release_half']-before[cid]['release_half'])>1e-4
    with pytest.raises(ContractError,match='frozen model coefficient'):
        migrate_bc_ac(old,registry,changed,background_inputs=inputs)
    new,reg,_=migrate_bc_ac(old,registry,changed,background_inputs=inputs,coefficient_reference=ref)
    p=new.components[cid].parameters
    np.testing.assert_array_equal(p.quantity('K','dimensionless','receptor_half_drive').values,before[cid]['K'])
    np.testing.assert_array_equal(p.quantity('release_half','V','activation_half_voltage').values,before[cid]['release_half'])
    assert np.max(abs(p.quantity('V0','V','initial_membrane_voltage').values-before[cid]['V0']))>1e-5
    data['families'][cid]['parameters']['K'].update(binding='broadcast',value=before[cid]['K']*2)
    path=tmp_path/'changed_K.json';path.write_text(json.dumps(data),encoding='utf8');changed=CandidateProfile(path)
    after=resolve_background_bindings(old,changed,targets,inputs,ref)
    assert np.max(abs(after[cid]['e0']-before[cid]['e0']))>1e-3
    assert after[cid]['release_half']==before[cid]['release_half']


def test_bc_proposal_rebuilds_adapters_without_reanchoring_half(parent,tmp_path):
    old,reg,channels=parent;old=RetinaBuilder(reg).build(old.spec,RunMode.REFERENCE)
    new,reg,_=migrate_bc_ac(old,reg,ROOT/'configurations/bc_ac_contact_first_hypothesis_v1.json',background_inputs={k:v.at(0) for k,v in channels.items()})
    cid='midget_BC_OFF';c=new.components[cid]
    module=build_module(new,reg,cid,kind='bc_clamped',mode='declared_input_clamps',conditions=CONDITIONS,baselines=baselines(c))
    q=c.parameters.quantity('gE','S','conductance_per_weighted_receptor_gate')
    p,path=package(module,tmp_path,'gE',Quantity(q.values*1.2,q.unit,q.meaning,q.evidence_tag),[1e-12,1e-7])
    result=merge_parameter_package(new,reg,p,applicability=CONDITIONS,allowed_fields=['gE'],proposal_path=path)
    for key in ('K','release_half','release_slope'):
        np.testing.assert_array_equal(next(r.value.values for r in result.components[cid].parameters.records if r.parameter_id==key),
            next(r.value.values for r in c.parameters.records if r.parameter_id==key))
    adapters=[x for x in result.spec.components if x.options.get('biological_owner')==cid]
    assert adapters and all(x.parameter_profile_id==p['new_profile_id'] for x in adapters)
    assert canonical_hash(new.spec.gap_junctions)==canonical_hash(result.spec.gap_junctions)
    assert canonical_hash(new.spec.links)==canonical_hash(result.spec.links)
