"""Explicit, uncalibrated release-domain borrowing of Souihel & Cessac (2021).

This is output adaptation, not a replacement for the retained BC FIR or membrane.
The profile and the enabled/disabled choice are part of component binding identity.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json
import numpy as np
from .contracts import ContractError, canonical_hash

SCHEMA = 'bc-release-gain-control-1.0'


@dataclass(frozen=True)
class BipolarOutputGainControl:
    enabled: bool = False
    tau_a_seconds: float = .1
    scale: float = .75
    profile_sha256: str = ''

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ContractError('BC output gain-control enabled must be boolean')
        if (isinstance(self.tau_a_seconds, bool) or not np.isfinite(self.tau_a_seconds)
                or self.tau_a_seconds <= 0 or isinstance(self.scale, bool)
                or not np.isfinite(self.scale) or not 0 <= self.scale <= .75):
            raise ContractError('BC adaptation requires positive seconds and declared bounded dimensionless scale [0,.75]')

    @classmethod
    def from_options(cls, options):
        row = options.get('bc_output_gain_control')
        if row is None:
            return cls()
        if row.get('schema_version') != SCHEMA or row.get('driver') != 'unadapted_terminal_release':
            raise ContractError('Unknown BC gain-control version or recursive adapted-release driver')
        return cls(row['enabled'], row['tau_a_seconds'], row['scale'], row['profile_sha256'])

    def gain(self, activity):
        a = np.asarray(activity)
        if not np.all(np.isfinite(a)) or np.any(a < -1e-10) or np.any(a > self.scale+1e-10):
            raise ContractError('BC release adaptation activity outside its invariant nonnegative bounded domain')
        return 1./(1.+a**6) if self.enabled else np.ones_like(a)

    @property
    def identity(self):
        return canonical_hash(dict(schema=SCHEMA, enabled=self.enabled, tau_a_seconds=self.tau_a_seconds,
            scale=self.scale, profile_sha256=self.profile_sha256,
            driver='unadapted_terminal_release', gain='1/(1+A^6), including G(0)=1',
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def selected_control(root, enabled=None):
    root = Path(root).resolve()
    main = json.loads((root/'configurations/retina_platform_v1.json').read_bytes())
    policy = main.get('bc_output_gain_control', {'enabled': False})
    choice = policy['enabled'] if enabled is None else enabled
    if type(choice) is not bool:
        raise ContractError('BC gain-control override must be boolean or None')
    if not choice:
        return None
    if policy.get('schema_version') != 'bc-output-gain-control-policy-1.0':
        raise ContractError('Unknown shared BC output policy')
    path = (root/policy['profile']['path']).resolve()
    if not path.is_relative_to(root):
        raise ContractError('BC gain-control profile must remain in the project')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != policy['profile']['sha256']:
        raise ContractError('BC gain-control profile identity differs')
    profile = json.loads(raw)
    if (profile.get('schema_version') != SCHEMA or profile.get('calibrated') is not False
            or profile.get('driver') != 'unadapted_terminal_release' or profile.get('gain_exponent') != 6
            or profile.get('gain_at_zero') != 1):
        raise ContractError('Uncalibrated original-release BC adaptation contract required')
    if profile['tau_a']['unit']!='s' or profile['scale']['unit']!='dimensionless':
        raise ContractError('BC adaptation profile must explicitly use seconds and dimensionless release scaling')
    row = dict(schema_version=SCHEMA, enabled=True, driver=profile['driver'],
        tau_a_seconds=profile['tau_a']['value'], scale=profile['scale']['value'],
        profile_sha256=digest, profile_id=profile['profile_id'])
    BipolarOutputGainControl.from_options({'bc_output_gain_control': row})
    return row


def apply_bc_gain_control(result, root, enabled=None):
    """Recompile the same graph with output adaptation; never prepare or fit it."""
    control = selected_control(root, enabled)
    parent, channels, audit = result
    previous = getattr(parent, 'bc_output_gain_control', None)
    if control is None:
        if previous is not None:
            raise ContractError('Disabling BC gain control requires a fresh model build; existing adapted state cannot be relabelled')
        return result
    if previous is not None:
        if previous['control'] != control:
            raise ContractError('BC adaptation already applied with a different identity')
        return result  # Never multiply the same output twice.
    targets = [s.component_id for s in parent.spec.components if s.mechanism == 'HistoryBipolarTerminal']
    if not targets:
        return result
    from .assembly import RetinaBuilder
    specs = tuple(replace(s, options=dict(s.options, bc_output_gain_control=control))
                  if s.component_id in targets else s for s in parent.spec.components)
    budget = getattr(parent, 'compile_resource_budget_bytes', None)
    builder = RetinaBuilder(audit['registry'], **({} if budget is None else {'max_compile_bytes': budget}))
    compiled = builder.build(replace(parent.spec, components=specs), parent.run_mode)
    for key, value in parent.__dict__.items():
        if key not in compiled.__dict__:
            setattr(compiled, key, value)
    receipt = dict(schema_version=SCHEMA, control=control, targets=targets,
        original_assembly_hash=parent.assembly_hash, assembly_hash=compiled.assembly_hash,
        driver='unadapted_terminal_release', output='activation aliases effective_release exactly once',
        parameters_calibrated=False, background_reinitialization_required=True,
        old_background_and_downstream_caches_valid=False, graph_or_other_parameters_changed=False)
    compiled.bc_output_gain_control = receipt
    audit = dict(audit, spec=compiled.spec, bc_output_gain_control=receipt,
        resource_estimate=builder.estimate(compiled.spec),
        runtime_profile=dict(audit.get('runtime_profile', {}), bc_output_gain_control=receipt))
    return compiled, channels, audit
