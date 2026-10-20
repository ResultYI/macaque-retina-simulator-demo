"""Contact-first v2 candidate factories shared by circuit and isolated module."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from scipy.special import expit
from .assembly import RuntimeComponent, PortSpec, TimeContract
from .contracts import ContractError, Quantity, canonical_hash
from .mechanism_contracts import MechanismStateSpec
from .bc_ac_receptor_components import BC_TYPES, AC_TYPES
from .contact_first_receptor import ROUTE, POOLING, MEANING, AGGREGATION, occupancy

BC_KEY = 'ContactFirstConeBipolarReceptorMembraneV2'
AC_KEY = 'ContactFirstGenericAmacrineReceptorMembraneV2'
STATIC_KEY = 'ContactFirstStaticOccupancyV2'


def _contract(spec):
    c = spec.options.get('candidate_dynamics', {})
    if (c.get('route') != ROUTE or c.get('pooling_order') != POOLING or
            c.get('aggregation_contract') != AGGREGATION or not c.get('profile_sha256') or
            c.get('glutamate_concentration_claim') is not False):
        raise ContractError('Explicit source-bound contact-first aggregation contract required')
    return c


def _hash():
    return canonical_hash({p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
        for p in ('bc_ac_contact_components.py', 'contact_first_receptor.py')})


def static_component(spec, populations, sources, mode):
    _contract(spec)
    from .inner import _profile
    p, pop, tag = _profile(spec, populations, sources, mode)
    ids = tuple(spec.options['carrier_cell_ids'])
    K = p.quantity('K','dimensionless','receptor_half_drive',positive=True).values
    hill = p.quantity('hill','dimensionless','receptor_hill_exponent',positive=True).values
    if not np.all(K == K[0]) or not np.all(hill == hill[0]):
        raise ContractError('Shared static input transform requires common K and hill across target cells')
    inputs = {'drive': PortSpec('drive','dimensionless',spec.options['input_meaning'],ids,tag)}
    outputs = {'occupied': PortSpec('occupied','dimensionless',MEANING,ids,tag)}
    def observe(name,t,b,i):
        u=i['drive'].require('dimensionless',inputs['drive'].meaning,(len(ids),))
        return Quantity(occupancy(u,K[0],hill[0]),'dimensionless',MEANING,tag)
    model=SimpleNamespace(model_hash=canonical_hash({'profile':p.parameter_hash,'route':ROUTE}),
        approximations=[{'id':ROUTE,'why':'Stateless computational transform before original W; not a new biological cell/contact'}])
    return RuntimeComponent(spec.component_id,STATIC_KEY,p,model,inputs,outputs,(),lambda:(),
        lambda t,b,i:(),observe,{'occupied':('drive',)},TimeContract('continuous'),(),(pop,),tag,p.source_ids,_hash())


def membrane_component(spec, populations, sources, mode):
    contract = _contract(spec)
    from .inner import _profile
    is_bc=spec.mechanism==BC_KEY
    if (spec.component_id not in (BC_TYPES if is_bc else AC_TYPES) or
            spec.required_layer not in (None,spec.component_id)):
        raise ContractError('Contact-first membrane requires an approved BC/AC target identity')
    p,pop,tag=_profile(replace(spec,mechanism='ResidualBC_'+BC_TYPES[spec.component_id]
        if is_bc else 'ResidualAC'),populations,sources,mode)
    n=len(pop.cells.cell_id);ids=pop.cells.cell_id
    def a(k,u,m,positive=False):return p.quantity(k,u,m,(n,),positive=positive).values
    C=a('C','F','membrane_capacitance',True);gL=a('gL','S','leak_conductance',True)
    EL=a('EL','V','leak_reversal');V0=a('V0','V','initial_membrane_voltage')
    gE=a('gE','S','conductance_per_weighted_receptor_gate',True);EE=a('EE','V','excitatory_reversal')
    tau=a('tau_e','s','excitatory_receptor_time_constant',True)
    e0=a('e0','dimensionless','initial_weighted_receptor_gate')
    totals=np.asarray(contract['row_sum_W'],dtype=float)
    if totals.shape!=(n,) or np.any(totals<=0) or not np.all(np.isfinite(totals)) or np.any((e0<0)|(e0>totals)):
        raise ContractError('Weighted initial aggregate requires actual positive row sums of W')
    if is_bc:
        half=a('release_half','V','activation_half_voltage');slope=a('release_slope','V','activation_voltage_slope',True)
        tau_release=a('tau_release','s','release_time_constant',True)
        names=[('V','V','physical_membrane_voltage'),('release','dimensionless','effective_release_activation'),
               ('e','dimensionless','excitatory_synaptic_gate')]
        initial_values=(V0,expit((V0-half)/slope),e0)
    else:
        half=a('activation_half','V','effective_output_half_voltage');slope=a('activation_slope','V','effective_output_voltage_slope',True)
        names=[('V','V','physical_membrane_voltage'),('e','dimensionless','excitatory_synaptic_gate')]
        initial_values=(V0,e0)
    states=tuple(MechanismStateSpec(k,ids,(n,),u,m,'Explicit initial weighted aggregate; no adaptation/contact-state reconstruction claim') for k,u,m in names)
    inputs={'drive':PortSpec('drive','dimensionless',MEANING,ids,tag)}
    off=is_bc and BC_TYPES[spec.component_id]=='OFF'
    if off:
        EI=a('EI','V','inhibitory_reversal');inputs['inhibition']=PortSpec('inhibition','S','inhibitory_conductance',ids,tag)
    outputs={'activation':PortSpec('activation','dimensionless','effective_synaptic_activation',ids,tag),
        'voltage':PortSpec('voltage','V','physical_membrane_voltage',ids,tag),
        'receptor_gate':PortSpec('receptor_gate','dimensionless','excitatory_synaptic_gate',ids,tag)}
    if is_bc:
        outputs['release']=replace(outputs['activation'],name='release')
        outputs['gap_current']=PortSpec('gap_current','A','electrical_gap_current',ids,tag)
        if off:outputs['glycine_conductance']=inputs['inhibition']
    def initial():return tuple(Quantity(v,s.unit,s.meaning,tag) for v,s in zip(initial_values,states))
    def derivative(t,b,i):
        V=b[0];e=b[-1];q=i['drive'].require('dimensionless',MEANING,(n,))
        if not np.all(np.isfinite(q)) or np.any(q < -1e-14) or np.any(q > totals+1e-14):
            raise ContractError('Weighted receptor occupancy outside [0,row_sum(W)]; supply Wf(u), not raw u')
        target=totals-q if is_bc and not off else q
        current=gL*(EL-V)+gE*e*(EE-V)
        if is_bc:current=current+i.get('__gap_current__',Quantity(np.zeros(n),'A','electrical_gap_current',tag)).values
        if off:
            gi=i['inhibition'].require('S','inhibitory_conductance',(n,))
            if np.any(gi<0):raise ContractError('Negative inhibitory conductance')
            current=current+gi*(EI-V)
        values=[current/C]
        if is_bc:values.append((expit((V-half)/slope)-b[1])/tau_release)
        values.append((target-e)/tau)
        return tuple(Quantity(v,s.unit+'/s',s.meaning+'_derivative',tag) for v,s in zip(values,states))
    def observe(name,t,b,i):
        if name in ('activation','release'):value=b[1] if is_bc else expit((b[0]-half)/slope)
        elif name=='voltage':value=b[0]
        elif name=='gap_current':value=np.zeros(n)
        elif name=='glycine_conductance':return i['inhibition']
        else:value=b[-1]
        port=outputs[name];return Quantity(value,port.unit,port.meaning,tag)
    model=SimpleNamespace(model_hash=canonical_hash({'profile':p.parameter_hash,'route':ROUTE}),
        approximations=[{'id':ROUTE,'why':'Shared-tau weighted gate reduction; uncalibrated passive membrane and effective output',
            'input_contract':contract,'whole_cell_max_excitatory_conductance':'gE * row_sum(W)'}])
    if is_bc:model.membrane_capacitance=C;model.gap_state_index=0
    return RuntimeComponent(spec.component_id,spec.mechanism,p,model,inputs,outputs,states,initial,derivative,
        observe,{k:('inhibition',) if k=='glycine_conductance' else () for k in outputs},
        TimeContract('continuous'),(spec.component_id,),(pop,),tag,p.source_ids,_hash())
