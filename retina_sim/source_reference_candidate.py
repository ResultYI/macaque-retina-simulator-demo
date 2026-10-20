"""Narrow source transfer using the existing profile/rebuild machinery.

This separate route is not a fitted proposal and does not expand fitting GROUPS.
The original 1.0 ON-parasol recipe stays frozen. Version 1.1 adds individually
named, unadopted cross-type candidates with explicit lower evidence scope.
Separate per-type acceptance can adopt those unchanged recipes as engineering
defaults; the source coefficients never become physiological measurements.
"""
from dataclasses import replace
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
from .contracts import ContractError, Quantity, canonical_hash
from .assembly import RetinaBuilder
from .mechanism_contracts import register_mechanism_profile
from .provenance import SourceRecord

SCHEMA='rgc-source-reference-candidate-1.0'
CROSS_TYPE_SCHEMA='rgc-source-reference-candidate-1.1'
CROSS_TYPE_TARGETS=('midget_RGC_ON','midget_RGC_OFF','parasol_RGC_OFF','small_bistratified_S_ON')
TRANSFER_EVIDENCE=dict(source_parameter_support='B: published ON-parasol output fit',
    target_transfer_support='D: cross-type transfer in retained project membrane/bin/history conditions',
    target_type_voltage_spike_calibration=False,production_adoption=False)
VALUES={'alpha':(90.,'Hz','spike_intensity_scale'),
        'mu':(-.053,'V','spike_voltage_offset'),
        'beta':(.00167,'V','spike_voltage_scale')}


def read_source_candidate(root,path):
    root=Path(root).resolve();path=Path(path)
    if not path.is_absolute():path=root/path
    path=path.resolve()
    if not path.is_relative_to(root):raise ContractError('Source candidate must remain in project')
    d=json.loads(path.read_bytes())
    legacy=d.get('schema_version')==SCHEMA and d.get('component_id')=='parasol_RGC_ON'
    cross_type=(d.get('schema_version')==CROSS_TYPE_SCHEMA and d.get('component_id') in CROSS_TYPE_TARGETS
        and d.get('transfer_evidence')==TRANSFER_EVIDENCE)
    if (not (legacy or cross_type) or
        d.get('default_enabled') is not False or d.get('fit_performed') is not False or
        d.get('physiological_calibration') is not False or set(d.get('changes',{}))!=set(VALUES)):
        raise ContractError('An explicitly scoped unadopted per-type source-transfer candidate is required')
    if cross_type and (d.get('profile_id')!=d['component_id']+'_latimer2019_cross_type_v1' or
                       d.get('source',{}).get('source_id')!='LATIMER2019_CROSS_TYPE_'+d['component_id']):
        raise ContractError('Cross-type candidate/profile/source identity must name the target type')
    for key,(value,unit,meaning) in VALUES.items():
        if d['changes'][key]!={'value':value,'unit':unit,'meaning':meaning}:
            raise ContractError('Source transfer must use the audited published value and units: '+key)
    for row in [d['base_main_configuration'],d['base_dynamics_candidate'],d['source']]:
        source=(root/row['path']).resolve()
        if (source.is_relative_to(root) and row is d['base_main_configuration'] and
                hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']):
            # B5 changes the one live main configuration. The original recipe
            # remains immutable and binds only this exact archived predecessor.
            prior=root/'configurations/history/retina_platform_pre_rgc_20261020.json'
            if row['path']=='configurations/retina_platform_v1.json' and prior.is_file():source=prior
        if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:
            raise ContractError('Source/main-configuration/candidate identity differs')
    if d['source']['sha256']!='09b5ce244f7abbdcc110046d0fc424f51c39055016e47c951098d49e2ec9bf9f':
        raise ContractError('Unexpected primary source for this transfer')
    return d,path


def merge_source_reference_candidate(parent,registry,package,*,applicability,allowed_fields,proposal_path,_engineering_acceptance=None):
    from .calibration_modules import _plain,affected_closure
    d,path=read_source_candidate(registry.root,proposal_path)
    legacy=d['schema_version']==SCHEMA
    if (canonical_hash(d)!=canonical_hash(package) or applicability!=d['applicability'] or
        allowed_fields is None or set(allowed_fields)!=set(VALUES) or len(allowed_fields)!=3):
        raise ContractError('Persisted source candidate/applicability/exact field boundary differs')
    cid=d['component_id'];old=parent.components[cid]
    if old.mechanism!='HistoryConductanceRGC':raise ContractError('Original RGC factory required')
    actual={r.parameter_id:r for r in old.parameters.records}
    if set(actual)-set(VALUES)!=set(d['retained_parameters']):raise ContractError('Retained RGC parameter contract differs')
    isolated=copy.copy(registry)
    for attr in ('sources','profiles','mechanism_profiles'):
        setattr(isolated,attr,dict(getattr(registry,attr)))
    source=d['source'];sid=source['source_id']
    if sid in isolated.sources:raise ContractError('Source-transfer identity already exists')
    isolated.register_source(SourceRecord(sid,'Latimer2019 output coefficients transferred to project conditions',
        ('K W Latimer','F Rieke','J W Pillow'),2019,source['url'],'paper',None,source['path'],source['locator'],
        source['sha256'],None,'CC BY4.0 original article','identity_verified',
        ('Published fit in original two-cell domain; this transfer remains project_approximation, not measurement or refit' if legacy else
         'Published fit in original ON-parasol two-cell domain; target transfer remains project_approximation, not measurement or refit'),
        '2026-10-20',{'source_recording':'Cafaro-Rieke2013 dynamic clamp; raw V-spike unavailable'}))
    rows=[]
    for r in old.parameters.records:
        if r.parameter_id in VALUES:
            value,unit,meaning=VALUES[r.parameter_id]
            r.value.require(unit,meaning)
            r=replace(r,value=Quantity(np.full(r.value.values.shape,value),unit,meaning,r.value.evidence_tag),
                source_id=sid,evidence_class='project_approximation',locator=source['locator']+'; '+r.parameter_id,
                supported_domain=copy.deepcopy(applicability),
                uncertainty='Literature transfer across membrane/bin/history conditions; not empirically calibrated')
        rows.append(r)
    adopted=_engineering_acceptance is not None
    pid=d['profile_id']+('_engineering_default' if adopted else '')
    profile=replace(old.parameters.profile,profile_id=pid,records=tuple(rows),
        source_ids=tuple(sorted(set(old.parameters.source_ids+(sid,)))),
        approval_reference=('Pro B5 accepted scoped engineering output default, 2026-10-20' if adopted else
                            'Authorized isolated ON-parasol source-transfer candidate, 2026-10-20' if legacy else
                            'Authorized isolated '+cid+' source-transfer candidate, 2026-10-20'),
        conditions=dict(old.parameters.conditions,source_reference_candidate=dict(
            candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),applicability=copy.deepcopy(applicability),
            fitted=False,production_default=adopted,
            **({} if legacy else dict(transfer_evidence=copy.deepcopy(d['transfer_evidence']))),
            **(dict(engineering_acceptance=copy.deepcopy(_engineering_acceptance)) if adopted else {}))))
    register_mechanism_profile(isolated,profile)
    runtime=getattr(isolated,'runtime_profile',None)
    if runtime is not None and runtime.modeling_scope is not None:
        runtime=copy.copy(runtime);scope=copy.copy(runtime.modeling_scope)
        scope.base_scope_sha256=getattr(scope,'base_scope_sha256',scope.sha256)
        scope.parameter_proposal_revisions=copy.deepcopy(getattr(scope,'parameter_proposal_revisions',{}))
        scope.parameter_proposal_revisions[cid]=dict(base_profile_id=old.parameters.profile.profile_id,
            new_profile_id=pid,content_hash=canonical_hash(d),changed_fields=sorted(VALUES),source_transfer=True)
        scope.sha256=canonical_hash(dict(base_scope_sha256=scope.base_scope_sha256,revisions=scope.parameter_proposal_revisions))
        runtime.modeling_scope=scope;isolated.runtime_profile=runtime
    populations=[]
    for p in parent.spec.populations:
        ids=tuple(pid if value==old.parameters.profile.profile_id else value for value in p.cells.parameter_profile_id)
        populations.append(replace(p,cells=replace(p.cells,parameter_profile_id=ids)) if ids!=p.cells.parameter_profile_id else p)
    specs=[]
    for s in parent.spec.components:
        # Existing conductance adapters share the owner's cell table, while
        # retaining their own coefficients and mechanism profile.
        if s.options.get('target_population_profile_id')==old.parameters.profile.profile_id:
            s=replace(s,options=dict(s.options,target_population_profile_id=pid))
        if s.component_id==cid:s=replace(s,parameter_profile_id=pid)
        specs.append(s)
    result=RetinaBuilder(isolated).build(replace(parent.spec,populations=tuple(populations),components=tuple(specs)),parent.run_mode)
    for key,c in parent.components.items():
        revised=result.components[key]
        if key!=cid and canonical_hash((_plain(c.parameters.records),_plain(c.parameters.profile)))!=canonical_hash((_plain(revised.parameters.records),_plain(revised.parameters.profile))):
            raise ContractError('Source transfer changed unrelated coefficients: '+key)
    for key in ('geometry_source_binding','input_geometry_binding_required','typed_view_domain',
                'frozen_typed_geometry_binding','global_disk_circuit','global_disk_view','canonical_spatial_recipe','candidate_dynamics',
                'accepted_rgc_output_defaults'):
        if hasattr(parent,key):setattr(result,key,copy.deepcopy(getattr(parent,key)))
    result.calibration_registry=isolated
    result.parameter_package_merge=dict(status='source_reference_engineering_default' if adopted else 'source_reference_transfer_unadopted',changed_component=cid,
        changed_fields=sorted(VALUES),new_profile_id=pid,content_hash=canonical_hash(d),
        fitted=False,production_default=adopted,initialization_transferred=False,background_validated=False,
        invalidation=dict(background=True,checkpoint=True,adapted_state=True,
            affected_neural_components=affected_closure(parent.spec,{cid})))
    return result
