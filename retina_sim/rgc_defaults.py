"""One shared main configuration selects individually accepted RGC mappings.

The historical source recipes and v1 numerical profile remain unchanged.
Adoption changes only output coefficients/identity, not membrane or graph.
"""
from pathlib import Path
import hashlib,json
from .contracts import ContractError,canonical_hash
from .source_reference_candidate import read_source_candidate,merge_source_reference_candidate,CROSS_TYPE_TARGETS


def _validate_adoption(accepted,cid,recipe):
    """Keep historical test acceptance distinct from an authorized D default."""
    common=(accepted.get('component_id')==cid and
            accepted.get('physiological_calibration') is False and
            accepted.get('fit_performed') is False)
    if accepted.get('schema_version')=='rgc-engineering-acceptance-1.0':
        valid=common and all(accepted.get('passed_checks',{}).get(k) is True for k in
            ('mapping','bright_dark','contrast_ordering','own_mapping_recovery','single_cell_spatial_or_spectral'))
    elif accepted.get('schema_version')=='rgc-engineering-adoption-2.0':
        valid=(common and cid in CROSS_TYPE_TARGETS and
            accepted.get('decision')=='adopt_source_limited_engineering_default' and
            accepted.get('evidence_class')=='project_approximation' and
            accepted.get('evidence_grade')=='D cross-type engineering transfer' and
            accepted.get('changed_parameters')==['alpha','mu','beta'] and
            accepted.get('values')==recipe.get('changes') and
            accepted.get('historical_failures_preserved') is True and
            accepted.get('general_image_video_validation_completed') is False and
            accepted.get('target_type_voltage_spike_calibration') is False and
            isinstance(accepted.get('authorization'),str) and bool(accepted['authorization'].strip()) and
            isinstance(accepted.get('scope'),str) and bool(accepted['scope'].strip()) and
            isinstance(accepted.get('limitations'),list) and bool(accepted['limitations']) and
            isinstance(accepted.get('evidence_files'),list) and bool(accepted['evidence_files']))
    else:valid=False
    if not valid:raise ContractError('Scoped per-type acceptance or explicit D engineering adoption required')


def historical_reference_file(root,relative,expected_sha256):
    """Resolve only the explicitly archived pre-B5 main config for old guards."""
    root=Path(root).resolve();p=(root/relative).resolve()
    if hashlib.sha256(p.read_bytes()).hexdigest()==expected_sha256:return p
    old=root/'configurations/history/retina_platform_pre_rgc_20261020.json'
    if (relative=='configurations/retina_platform_v1.json' and old.is_file() and
            hashlib.sha256(old.read_bytes()).hexdigest()==expected_sha256):return old
    raise ContractError('Historical protected asset changed: '+str(relative))


def historical_project_default(root):
    from retina_platform.config import RunConfig
    root=Path(root);path=root/'configurations/history/retina_platform_pre_rgc_20261020.json'
    if not path.exists():return RunConfig.project_default(root)
    if hashlib.sha256(path.read_bytes()).hexdigest()!='a02224d6ce920daad22310e39577f45ec1562fb9146b4e201155dd3b8be9bd20':
        raise ContractError('Pre-B5 main configuration identity differs')
    return RunConfig.from_dict(json.loads(path.read_bytes())['run_defaults'])


def apply_rgc_defaults(result,root,enabled=True):
    if not enabled:return result
    root=Path(root).resolve();main=json.loads((root/'configurations/retina_platform_v1.json').read_bytes())
    policy=main.get('rgc_output_defaults')
    if policy is None:return result
    if policy.get('schema_version')!='rgc-engineering-defaults-1.0':raise ContractError('Unknown RGC default policy')
    compiled,channels,audit=result;adoptions=[];absent=[]
    for cid,row in policy['accepted_types'].items():
        recipe,path=read_source_candidate(root,root/row['recipe_path'])
        acceptance_path=(root/row['acceptance']['path']).resolve()
        if (recipe['component_id']!=cid or hashlib.sha256(path.read_bytes()).hexdigest()!=row['recipe_sha256'] or
            not acceptance_path.is_relative_to(root) or hashlib.sha256(acceptance_path.read_bytes()).hexdigest()!=row['acceptance']['sha256']):
            raise ContractError('RGC default recipe/acceptance identity differs')
        accepted=json.loads(acceptance_path.read_bytes())
        _validate_adoption(accepted,cid,recipe)
        for binding in accepted['evidence_files']:
            p=(root/binding['path']).resolve()
            if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=binding['sha256']:
                raise ContractError('RGC acceptance evidence changed')
        if cid not in compiled.components:
            view=getattr(compiled,'global_disk_view',None)
            if view is None or cid not in view.get('empty_output_samples',()):
                raise ContractError('RGC default target absent without an explicit empty-owner declaration: '+cid)
            absent.append(dict(component_id=cid,status='EMPTY_OWNER_SAMPLE_DEFAULT_RETAINED_NOT_INSTANTIATED'))
            continue
        compiled=merge_source_reference_candidate(compiled,audit['registry'],recipe,
            applicability=recipe['applicability'],allowed_fields=['alpha','mu','beta'],proposal_path=path,
            _engineering_acceptance=dict(acceptance_sha256=row['acceptance']['sha256'],scope=accepted['scope'],
                evidence_grade=accepted['evidence_grade'],fit_performed=False,physiological_calibration=False,
                **(dict(adoption_schema=accepted['schema_version'],authorization=accepted['authorization'])
                   if accepted['schema_version']=='rgc-engineering-adoption-2.0' else {})))
        audit=dict(audit,registry=compiled.calibration_registry,spec=compiled.spec)
        adoptions.append(dict(component_id=cid,recipe_sha256=row['recipe_sha256'],acceptance=row['acceptance'],
            values=recipe['changes'],engineering_default=True,physiological_calibration=False))
    if adoptions:
        from .assembly import RetinaBuilder
        audit.update(runtime_profile=dict(audit['runtime_profile'],accepted_RGC_output_defaults=adoptions),
            resource_estimate=RetinaBuilder(audit['registry']).estimate(compiled.spec),
            population_records={key:pop.record() for key,pop in compiled.populations.items()})
        compiled.accepted_rgc_output_defaults=adoptions
    if absent:audit=dict(audit,rgc_defaults_absent_owner_components=absent)
    return compiled,channels,audit
