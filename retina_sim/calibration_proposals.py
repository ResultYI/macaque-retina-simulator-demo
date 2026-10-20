"""Restricted, nondefault fitted proposals; never automatic physiological adoption.

Legacy snapshots retain their identity-only contract. This separate schema stores
an explicit parameter delta and rebuilds the real factories with new identities.
"""
from dataclasses import replace
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import numpy as np
from .contracts import ContractError, Quantity, canonical_hash
from .assembly import RetinaBuilder
from .mechanism_contracts import register_mechanism_profile
from .provenance import SourceRecord

SCHEMA = 'module-fitted-proposal-1.0'
GROUPS = {
    'HistoryConductanceRGC': ({'mu'}, {'mu','beta'}, {'C','gL'}, {'C'}, {'gL'}, {'spike_history_kernel'}),
    'ContactFirstConeBipolarReceptorMembraneV2': (
        {'tau_e'}, {'gE'}, {'gE','tau_e'}, {'K','hill'}, {'release_half','tau_release'}, {'release_half'}, {'tau_release'}),
    'ContactFirstGenericAmacrineReceptorMembraneV2': (
        {'tau_e'}, {'gE'}, {'gE','tau_e'}, {'K','hill'}, {'activation_half','activation_slope'}),
}
POSITIVE = {'C','gL','beta','tau_e','gE','K','hill','tau_release','activation_slope'}


def _fit_context(context, applicability):
    required=('observation_kind','source_id','source_locator','unit_index_status','splits','fit_performed',
              'preprocessing','metrics','bounds','selection_history','uncertainty')
    if any(k not in context or context[k] in (None,'',{}) for k in required):
        raise ContractError('Explicit fitted provenance, units, bounds, scores and split contract required')
    if context['observation_kind'] not in ('real_observation','engineering_reference'):
        raise ContractError('Distinguish real observations from engineering references')
    if type(context['fit_performed']) is not bool or context['observation_kind']=='real_observation' and not context['fit_performed']:
        raise ContractError('Real fitted proposals require an actual fit; engineering QA must declare its status')
    if context['unit_index_status'] not in ('verified','unverified_conditional'):
        raise ContractError('Explicit source unit/index status required')
    if not isinstance(applicability,dict) or not applicability:
        raise ContractError('Explicit fitted applicability required')
    splits=context['splits']
    if set(splits)!={'train','validation','test'} or not splits['train']:
        raise ContractError('Explicit train/validation/test groups required; test may be empty and unvalidated')
    seen=set()
    for groups in splits.values():
        if not isinstance(groups,list):raise ContractError('Split groups must be hash lists')
        for h in groups:
            if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) or h in seen:
                raise ContractError('Split identities must be unique disjoint SHA256 groups')
            seen.add(h)


def _delta(component, changes):
    fields=set(changes)
    if fields not in GROUPS.get(component.mechanism,()):
        raise ContractError('Parameter group outside approved restricted fitting whitelist')
    records={r.parameter_id:r for r in component.parameters.records}
    result={}
    for key,change in changes.items():
        if key not in records:raise ContractError('Unknown fitted coefficient '+key)
        old=records[key].value
        if key=='spike_history_kernel':
            if (not isinstance(change,dict) or set(change)!={'scale'} or
                    not np.isscalar(change['scale']) or not np.isfinite(change['scale']) or change['scale']<=0):
                raise ContractError('History permits only one positive amplitude scale, fixed shape and dt')
            value=old.values*float(change['scale']);operation=dict(kind='amplitude_scale',scale=float(change['scale']))
        else:
            if not isinstance(change,Quantity):raise ContractError('Fitted changes require typed quantities')
            value=change.require(old.unit,old.meaning,old.values.shape)
            if change.evidence_tag!=old.evidence_tag or not np.all(value==value.flat[0]):
                raise ContractError('Only a shared within-family coefficient; no per-cell fit freedom')
            operation=dict(kind='broadcast_coefficient')
        if not np.all(np.isfinite(value)) or key in POSITIVE and np.any(value<=0):
            raise ContractError('Fitted coefficient domain invalid')
        if np.array_equal(value,old.values):raise ContractError('Fitted proposal must change each listed coefficient')
        result[key]=dict(value=np.asarray(value).tolist(),unit=old.unit,meaning=old.meaning,
            old_value=old.values.tolist(),operation=operation)
    return result


def fitted_parameter_proposal(module, *, version, new_profile_id, changes, allowed_fields,
                              applicability, fit_context):
    from .calibration_modules import parameter_package
    if (not version or not new_profile_id or not allowed_fields or
            set(changes)!=set(allowed_fields) or len(allowed_fields)!=len(set(allowed_fields))):
        raise ContractError('Explicit exact field whitelist and new profile identity required')
    cid=module.manifest['component_id'];component=module.compiled.components[cid]
    if not hasattr(component.parameters.profile,'records'):
        raise ContractError('Native source reference profiles are immutable; fit only approved dynamic profiles')
    if new_profile_id in module.registry.profiles:raise ContractError('New fitted profile identity already exists')
    _fit_context(fit_context,applicability)
    if fit_context['source_id'] not in module.registry.sources:
        raise ContractError('Fitted dataset/reference source must be registered')
    module.registry.sources[fit_context['source_id']].verify(module.registry.root)
    delta=_delta(component,changes)
    if set(fit_context['bounds'])!=set(delta):raise ContractError('Bounds required for exactly the changed fields')
    for key,change in delta.items():
        bounds=fit_context['bounds'][key]
        if not isinstance(bounds,list) or len(bounds)!=2 or not all(np.isfinite(v) for v in bounds) or bounds[0]>=bounds[1]:
            raise ContractError('Finite ordered parameter bounds required')
        values=change['operation'].get('scale',change['value'])
        if np.any(np.asarray(values)<bounds[0]) or np.any(np.asarray(values)>bounds[1]):
            raise ContractError('Proposal coefficient outside declared bounds')
    base=parameter_package(module,version=version,source_id=fit_context['source_id'],applicability=applicability)
    payload=dict(schema_version=SCHEMA,status='fitted_proposal_unadopted',version=version,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        component_id=cid,new_profile_id=new_profile_id,base_snapshot=base,
        allowed_fields=sorted(changes),changes=delta,fit_context=copy.deepcopy(fit_context),
        input_source_identity=canonical_hash(module.registry.sources[fit_context['source_id']]),
        applicability=copy.deepcopy(applicability),system_acceptance=False,production_default=False)
    return dict(payload,content_hash=canonical_hash(payload))


def merge_fitted_proposal(parent, registry, package, *, applicability, allowed_fields, proposal_path):
    from .calibration_modules import merge_parameter_package, affected_closure, _plain
    if (package.get('schema_version')!=SCHEMA or package.get('status')!='fitted_proposal_unadopted' or
            package.get('content_hash')!=canonical_hash({k:v for k,v in package.items() if k!='content_hash'}) or
            package.get('applicability')!=applicability or package.get('system_acceptance') is not False or
            package.get('production_default') is not False or allowed_fields is None or
            set(allowed_fields)!=set(package.get('allowed_fields',())) or
            len(allowed_fields)!=len(set(allowed_fields)) or set(package['changes'])!=set(allowed_fields)):
        raise ContractError('Fitted proposal identity/applicability/exact whitelist mismatch')
    _fit_context(package['fit_context'],applicability)
    # Validate the complete old system, not merely the changed component. This
    # keeps snapshot compatibility strict while allowing only the named delta.
    merge_parameter_package(parent,registry,package['base_snapshot'],applicability=applicability)
    cid=package['component_id'];old=parent.components[cid]
    if package['base_snapshot']['component_id']!=cid:raise ContractError('Proposal component mismatch')
    source=registry.sources.get(package['fit_context']['source_id'])
    if source is None:raise ContractError('Unknown fitted observation source')
    source.verify(registry.root)
    if canonical_hash(source)!=package.get('input_source_identity'):
        raise ContractError('Fitted input source identity changed')
    typed={}
    for key,row in package['changes'].items():
        if row['operation']['kind']=='amplitude_scale':typed[key]={'scale':row['operation']['scale']}
        else:typed[key]=Quantity(row['value'],row['unit'],row['meaning'],old.evidence_tag)
    if canonical_hash(_delta(old,typed))!=canonical_hash(package['changes']):
        raise ContractError('Fitted delta value/unit/shape/operation mismatch')
    # Revalidate bounds/provenance at consumption too, including rehashed edits.
    bounds=package['fit_context']['bounds']
    if set(bounds)!=set(typed):raise ContractError('Fitted bounds/fields mismatch')
    for key,row in package['changes'].items():
        b=bounds[key];v=np.asarray(row['operation'].get('scale',row['value']))
        if len(b)!=2 or not np.isfinite(b).all() or b[0]>=b[1] or np.any(v<b[0]) or np.any(v>b[1]):
            raise ContractError('Fitted coefficient outside declared bounds')
    path=Path(proposal_path).resolve() if proposal_path is not None else None
    if path is None or not path.is_file():raise ContractError('Persist fitted proposal before rebuilding')
    import json
    if canonical_hash(json.loads(path.read_text(encoding='utf8')))!=canonical_hash(package):
        raise ContractError('Persisted fitted proposal differs from supplied content')
    isolated=copy.copy(registry)
    for name in ('sources','profiles','mechanism_profiles'):setattr(isolated,name,dict(getattr(registry,name)))
    pid=package['new_profile_id']
    if not pid or pid in isolated.profiles:raise ContractError('New fitted profile identity required')
    sid='FITTED_PROPOSAL::'+pid
    record=SourceRecord(sid,'Unadopted conditional parameter proposal',('Retina project calibration',),2026,
        'urn:retina:calibration:'+package['content_hash'],'data_table',None,str(path),'changes',
        hashlib.sha256(path.read_bytes()).hexdigest(),None,'Project proposal; upstream data retains original license',
        'identity_verified','Inferred effective coefficients; not direct animal measurements',
        # First engineering audit artifacts predate the timestamp field. Their
        # creation date is frozen, never replaced with today's merge date.
        package.get('created_at_utc','2026-10-20').split('T')[0],{'input_source':package['fit_context']['source_id']})
    isolated.register_source(record)
    rows=[]
    for r in old.parameters.records:
        if r.parameter_id in package['changes']:
            d=package['changes'][r.parameter_id]
            r=replace(r,value=Quantity(d['value'],d['unit'],d['meaning'],r.value.evidence_tag),
                evidence_class='software_fixture' if r.value.evidence_tag=='synthetic_fixture' else 'project_approximation',
                source_id=sid,locator='changes.'+r.parameter_id,supported_domain=copy.deepcopy(applicability),
                uncertainty=package['fit_context']['uncertainty'])
        rows.append(r)
    profile=replace(old.parameters.profile,profile_id=pid,records=tuple(rows),
        source_ids=tuple(sorted(set(old.parameters.source_ids+(sid,)))),
        conditions=dict(old.parameters.conditions,calibration_proposal=dict(content_hash=package['content_hash'],
            base_profile_id=old.parameters.profile.profile_id,fit_context=package['fit_context'],
            system_acceptance=False,initialization_parameters=['V0','e0','initial_spike_history'])))
    register_mechanism_profile(isolated,profile)
    runtime=getattr(isolated,'runtime_profile',None)
    if runtime is not None and runtime.modeling_scope is not None:
        runtime=copy.copy(runtime);scope=copy.copy(runtime.modeling_scope)
        scope.base_scope_sha256=getattr(scope,'base_scope_sha256',scope.sha256)
        scope.parameter_proposal_revisions=copy.deepcopy(getattr(scope,'parameter_proposal_revisions',{}))
        scope.parameter_proposal_revisions[cid]=dict(base_profile_id=old.parameters.profile.profile_id,
            new_profile_id=pid,content_hash=package['content_hash'],changed_fields=sorted(typed))
        scope.sha256=canonical_hash(dict(base_scope_sha256=scope.base_scope_sha256,revisions=scope.parameter_proposal_revisions))
        runtime.modeling_scope=scope;isolated.runtime_profile=runtime
    # The algebraic occupancy adapters use exactly the same owner profile. Never
    # leave the input K/hill transform on the old coefficient set.
    owners={cid}|{s.component_id for s in parent.spec.components if s.options.get('biological_owner')==cid}
    populations=[]
    for p in parent.spec.populations:
        ids=tuple(pid if v==old.parameters.profile.profile_id else v for v in p.cells.parameter_profile_id)
        populations.append(replace(p,cells=replace(p.cells,parameter_profile_id=ids)) if ids!=p.cells.parameter_profile_id else p)
    specs=[]
    for s in parent.spec.components:
        if s.options.get('target_population_profile_id')==old.parameters.profile.profile_id:
            s=replace(s,options=dict(s.options,target_population_profile_id=pid))
        if s.component_id in owners:
            options=copy.deepcopy(s.options)
            if 'candidate_dynamics' in options:
                options['candidate_dynamics']=dict(options['candidate_dynamics'],parameter_profile_id=pid,
                    calibration_proposal_hash=package['content_hash'],model_coefficients_reanchored=False)
            s=replace(s,parameter_profile_id=pid,options=options)
        specs.append(s)
    result=RetinaBuilder(isolated).build(replace(parent.spec,populations=tuple(populations),components=tuple(specs)),parent.run_mode)
    for key,c in parent.components.items():
        current=result.components[key].parameters
        # Native ParameterSet hashes include the entire registry's source
        # manifest. Adding this proposal source changes that provenance hash,
        # even when every native coefficient/profile is byte-for-byte identical.
        if key not in owners and canonical_hash((_plain(current.records),_plain(current.profile)))!=canonical_hash((_plain(c.parameters.records),_plain(c.parameters.profile))):
            raise ContractError('Fitted proposal changed an unrelated parameter profile: '+key)
    for key in ('geometry_source_binding','input_geometry_binding_required','typed_view_domain',
                'frozen_typed_geometry_binding','global_disk_circuit','global_disk_view','canonical_spatial_recipe','candidate_dynamics'):
        if hasattr(parent,key):setattr(result,key,copy.deepcopy(getattr(parent,key)))
    result.calibration_registry=isolated
    result.parameter_package_merge=dict(status='restricted_fitted_proposal_rebuilt_unadopted',
        content_hash=package['content_hash'],changed_component=cid,changed_fields=sorted(typed),
        new_profile_id=pid,initialization_transferred=False,background_validated=False,
        system_acceptance=False,production_default=False,
        invalidation=dict(background=True,checkpoint=True,adapted_state=True,
            affected_neural_components=affected_closure(parent.spec,owners),
            optical_input='Only reusable with identical optics/stimulus/receptor geometry identities'))
    # New parameter/population/model hashes make old checkpoints incompatible;
    # factories provide cold states. A feedback solve is required before use.
    return result
