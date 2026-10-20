"""One shared main configuration selects individually accepted RGC mappings.

The historical source recipes and v1 numerical profile remain unchanged.
Adoption changes only output coefficients/identity, not membrane or graph.
"""
from pathlib import Path
import hashlib,json
from .contracts import ContractError,canonical_hash
from .source_reference_candidate import read_source_candidate,merge_source_reference_candidate


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
    compiled,channels,audit=result;adoptions=[]
    for cid,row in policy['accepted_types'].items():
        recipe,path=read_source_candidate(root,root/row['recipe_path'])
        acceptance_path=(root/row['acceptance']['path']).resolve()
        if (recipe['component_id']!=cid or hashlib.sha256(path.read_bytes()).hexdigest()!=row['recipe_sha256'] or
            not acceptance_path.is_relative_to(root) or hashlib.sha256(acceptance_path.read_bytes()).hexdigest()!=row['acceptance']['sha256']):
            raise ContractError('RGC default recipe/acceptance identity differs')
        accepted=json.loads(acceptance_path.read_bytes())
        if (accepted.get('schema_version')!='rgc-engineering-acceptance-1.0' or accepted.get('component_id')!=cid or
            accepted.get('physiological_calibration') is not False or accepted.get('fit_performed') is not False or
            not all(accepted.get('passed_checks',{}).get(k) is True for k in
                ('mapping','bright_dark','contrast_ordering','own_mapping_recovery','single_cell_spatial_or_spectral'))):
            raise ContractError('Scoped per-type engineering acceptance required before default adoption')
        for binding in accepted['evidence_files']:
            p=(root/binding['path']).resolve()
            if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=binding['sha256']:
                raise ContractError('RGC acceptance evidence changed')
        compiled=merge_source_reference_candidate(compiled,audit['registry'],recipe,
            applicability=recipe['applicability'],allowed_fields=['alpha','mu','beta'],proposal_path=path,
            _engineering_acceptance=dict(acceptance_sha256=row['acceptance']['sha256'],scope=accepted['scope'],
                evidence_grade=accepted['evidence_grade'],fit_performed=False,physiological_calibration=False))
        audit=dict(audit,registry=compiled.calibration_registry,spec=compiled.spec)
        adoptions.append(dict(component_id=cid,recipe_sha256=row['recipe_sha256'],acceptance=row['acceptance'],
            values=recipe['changes'],engineering_default=True,physiological_calibration=False))
    if adoptions:
        from .assembly import RetinaBuilder
        audit.update(runtime_profile=dict(audit['runtime_profile'],accepted_RGC_output_defaults=adoptions),
            resource_estimate=RetinaBuilder(audit['registry']).estimate(compiled.spec),
            population_records={key:pop.record() for key,pop in compiled.populations.items()})
        compiled.accepted_rgc_output_defaults=adoptions
    return compiled,channels,audit
