"""Continuous generic inhibition replacing, not following, legacy AC FIR.

The fixed DC/value-and-slope map is engineering calibration of an interface,
not a fit to neural recordings. It cannot preserve the old FIR transient or
the entire unbounded softplus transfer curve. Specialized AII is never selected.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np
from scipy.special import expit
from .assembly import PortSpec,RuntimeComponent,TimeContract
from .contracts import ContractError,Quantity,canonical_hash
from .inner import _profile,softplus
from .mechanism_contracts import MechanismStateSpec

KEY='EffectiveInhibitoryAC'
SCHEMA='generic-ac-effective-inhibition-1.0'


def implementation_hash():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def fixed_reference_mapping(profile,n):
    """Match legacy constant-reference activation and its DC first derivative."""
    kernel=profile.quantity('kernel','dimensionless','causal_residual_kernel').values
    bias=profile.quantity('bias','dimensionless','activation_bias',(n,)).values
    baseline=profile.quantity('baseline','dimensionless','input_baseline',(n,)).values
    history=profile.quantity('initial_history','dimensionless','causal_input_history',(len(kernel),n)).values
    if kernel.ndim!=1 or not len(kernel) or not np.all(np.isfinite(kernel)) or float(kernel.sum())<=0:
        raise ContractError('Continuous generic inhibition requires a positive legacy DC slope')
    if not np.all(history==history[0]):
        raise ContractError('A nonconstant legacy FIR prehistory has no single DC reference; fresh declared reference required')
    reference=baseline+history[0];z=bias+float(kernel.sum())*history[0]
    value=softplus(z);slope=float(kernel.sum())*expit(z)
    if np.any(reference<0) or np.any(value<=0) or not np.all(np.isfinite(value)):
        raise ContractError('Finite nonnegative reference input and positive reference activation required')
    amplitude=2*value;input_slope=2*slope/value
    for array in (reference,amplitude,input_slope):array.setflags(write=False)
    return reference,amplitude,input_slope,dict(legacy_kernel=kernel.tolist(),legacy_bias=bias.tolist(),
        reference_input=reference.tolist(),reference_activation=value.tolist(),reference_DC_derivative=slope.tolist(),
        mapping='h_target=sigmoid(input_slope*(c-reference_input)); activation=output_scale*h',
        normalization='Original incoming W and dimensionless activation units retained; no extra row rescaling',
        preserved='Fixed constant-reference value and first DC derivative only',
        not_preserved='Legacy FIR negative tail/transients, curvature and unbounded global output range',
        calibration='Engineering interface map, not data fitting')


def effective_ac_component(spec,populations,sources,mode):
    p,pop,tag=_profile(spec,populations,sources,mode);n=len(pop.cells.cell_id)
    if set(pop.cells.cell_type)!={'AC'} or spec.options.get('presynaptic_inhibition',False):
        raise ContractError('Effective inhibition is generic AC only; no new presynaptic gate or specialized AII replacement')
    control=spec.options.get('generic_ac_inhibition')
    if not isinstance(control,dict) or control.get('schema_version')!=SCHEMA or control.get('enabled') is not True:
        raise ContractError('Explicit versioned generic AC effective-inhibition contract required')
    tau=control.get('tau_A_seconds')
    if isinstance(tau,bool) or not isinstance(tau,(int,float)) or not np.isfinite(tau) or tau<=0:
        raise ContractError('Generic AC tau_A must be positive finite seconds, not a sampling interval')
    reference,amplitude,slope,mapping=fixed_reference_mapping(p,n)
    def target(drive):
        c=np.asarray(drive)
        if c.shape!=(n,) or not np.all(np.isfinite(c)) or np.any(c<0):
            raise ContractError('Generic AC requires finite nonnegative pooled BC effective release')
        return expit(slope*(c-reference))
    def validate_h(blocks):
        h=np.asarray(blocks[0])
        if h.shape!=(n,) or not np.all(np.isfinite(h)) or np.any(h< -1e-10) or np.any(h>1+1e-10):
            raise ContractError('Generic AC inhibitory activity outside finite invariant [0,1]')
        return h
    state=MechanismStateSpec('h',pop.cells.cell_id,(n,),'dimensionless','effective_inhibitory_activity',
        'Conditional fixed-reference h0=.5 seed; actual coupled background must be solved/prepared')
    def initial():return (Quantity(np.full(n,.5),'dimensionless','effective_inhibitory_activity',tag),)
    def derivative(t,blocks,incoming):
        h=validate_h(blocks)
        return (Quantity((target(incoming['drive'].values)-h)/tau,'dimensionless/s','effective_inhibitory_activity_derivative',tag),)
    def observe(port,t,blocks,incoming):
        h=validate_h(blocks)
        if port=='activation':return Quantity(amplitude*h,'dimensionless','effective_synaptic_activation',tag)
        value=target(incoming['drive'].values) if port=='target_inhibition' else h
        return Quantity(value,'dimensionless','effective_inhibitory_activity',tag)
    inputs={'drive':PortSpec('drive','dimensionless','effective_synaptic_activation',pop.cells.cell_id,tag)}
    outputs={name:PortSpec(name,'dimensionless','effective_synaptic_activation' if name=='activation'
        else 'effective_inhibitory_activity',pop.cells.cell_id,tag) for name in ('activation','inhibition_state','target_inhibition')}
    native=SimpleNamespace(tau_A_seconds=float(tau),reference_input=reference,input_slope=slope,
        output_scale=amplitude,target=target,mapping=mapping,generic_ac_inhibition=dict(control),
        model_hash=canonical_hash(dict(profile=p.parameter_hash,control=control,mapping=mapping,implementation=implementation_hash())),
        approximations=[dict(id='continuous_generic_inhibition',why='Unfitted10ms engineering state; fixed-reference value/slope map; no measured macaque AC kinetics or global FIR equivalence')])
    return RuntimeComponent(spec.component_id,KEY,p,native,inputs,outputs,(state,),initial,derivative,observe,
        {'activation':(),'inhibition_state':(),'target_inhibition':('drive',)},TimeContract('continuous'),
        ('AC','local_feedforward_AC','local_crossover_AC','widefield_AC'),(pop,),tag,p.source_ids,implementation_hash())


def selected_control(root,enabled=None):
    root=Path(root).resolve();main=json.loads((root/'configurations/retina_platform_v1.json').read_bytes())
    policy=main.get('generic_ac_inhibition',{'enabled':False});choice=policy['enabled'] if enabled is None else enabled
    if type(choice) is not bool:raise ContractError('Generic AC enabled override must be boolean or None')
    if not choice:return None
    if policy.get('schema_version')!='generic-ac-inhibition-policy-1.0':raise ContractError('Unknown generic AC policy')
    path=(root/policy['profile']['path']).resolve()
    if not path.is_relative_to(root):raise ContractError('Generic AC profile must remain in this project')
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=policy['profile']['sha256']:raise ContractError('Generic AC profile identity differs')
    profile=json.loads(raw)
    if (profile.get('schema_version')!=SCHEMA or profile.get('calibrated') is not False or
            profile.get('mapping')!='fixed_reference_value_and_DC_slope' or profile.get('replacement')!='legacy_AC_FIR_and_static_softplus'):
        raise ContractError('Explicit unfitted replacement and fixed-reference map required')
    times=profile.get('type_time_constants',{})
    if not times or any(row.get('unit')!='s' or isinstance(row.get('value'),bool) or
            not np.isfinite(row.get('value',np.nan)) or row['value']<=0 for row in times.values()):
        raise ContractError('Each generic AC role requires an explicit positive tau in seconds')
    return dict(schema_version=SCHEMA,enabled=True,profile_sha256=digest,type_time_constants=times,
        profile_id=profile['profile_id'],mapping=profile['mapping'])


def apply_ac_inhibition(result,root,enabled=None):
    parent,channels,audit=result;control=selected_control(root,enabled)
    previous=getattr(parent,'generic_ac_inhibition',None)
    if control is None:
        if previous is not None:raise ContractError('Disable generic AC on a fresh model build; adapted state cannot be relabelled')
        return result
    if previous is not None:
        if previous['control']!=control:raise ContractError('Generic AC already replaced with a different identity')
        return result
    targets=[s.component_id for s in parent.spec.components if s.mechanism=='ResidualAC']
    if not targets:return result
    specs=[]
    for s in parent.spec.components:
        if s.component_id not in targets:specs.append(s);continue
        group=s.required_layer if s.required_layer in control['type_time_constants'] else s.component_id
        if group not in control['type_time_constants']:group='generic'
        if group not in control['type_time_constants']:raise ContractError('Unmapped generic AC functional role: '+s.component_id)
        options=dict(s.options,generic_ac_inhibition=dict(schema_version=SCHEMA,enabled=True,
            tau_A_seconds=control['type_time_constants'][group]['value'],type_group=group,
            profile_sha256=control['profile_sha256'],evidence='project_approximation_not_fitted'))
        specs.append(replace(s,mechanism=KEY,options=options))
    from .assembly import RetinaBuilder
    budget=getattr(parent,'compile_resource_budget_bytes',None)
    builder=RetinaBuilder(audit['registry'],**({} if budget is None else {'max_compile_bytes':budget}))
    compiled=builder.build(replace(parent.spec,components=tuple(specs)),parent.run_mode)
    for key,value in parent.__dict__.items():
        if key not in compiled.__dict__:setattr(compiled,key,value)
    receipt=dict(schema_version=SCHEMA,control=control,targets=targets,original_assembly_hash=parent.assembly_hash,
        assembly_hash=compiled.assembly_hash,replaces='ResidualAC complete FIR/history+static output chain',
        specialized_AII_changed=False,connections_or_conductance_gains_changed=False,
        parameters_calibrated=False,background_reinitialization_required=True,old_background_and_downstream_caches_valid=False,
        mapping={cid:compiled.components[cid].native_model.mapping for cid in targets})
    compiled.generic_ac_inhibition=receipt
    audit=dict(audit,spec=compiled.spec,generic_ac_inhibition=receipt,resource_estimate=builder.estimate(compiled.spec),
        runtime_profile=dict(audit.get('runtime_profile',{}),generic_ac_inhibition=receipt))
    return compiled,channels,audit
