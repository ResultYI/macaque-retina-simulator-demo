"""Independent new-input equilibrium for the frozen finite HC v2 module.

Only input values and initial states change. Frozen H1/H2, BC/AC/RGC parameters
and held reservoir products are retained. This is not global optical closure.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from scipy.optimize import brentq
from .contracts import ContractError,canonical_hash
from .hc_lateral_interface import context,initialize_from_native


def prepare_aperture_background(base,parent_seed,hc_configuration,background,repo_root,protocol,*,input_identity):
    root=Path(repo_root);parent_hash=canonical_hash(parent_seed)
    prepared=deepcopy(parent_seed);state=prepared['states'];background=np.asarray(background,dtype=float)
    if (background.shape!=state['cones','analytic_RE','R'].shape or not np.isfinite(background).all()
            or np.any(background<=0) or not input_identity):raise ContractError('Explicit positive optical-equivalent background and identity required')
    assembly=context(base,prepared,hc_configuration,root)
    native=assembly.lateral_native_factory(assembly);p=native.p
    # The five-state phototransduction equilibrium algebra is exactly the source
    # VanHaterenModel.equilibrium first stage; no new fitted physiology.
    beta=p['c_beta']+p['k_beta']*background
    current=np.array([brentq(lambda c:c-(1/(b*(1+(p['a_C']*c)**p['n_C'])))**p['n_X'],
        0.,(1/b)**p['n_X'],xtol=1e-14,rtol=1e-14) for b in beta])
    precursor=current**(1/p['n_X'])
    voltage=(current/p['a_is'])**(1/(1+p['gamma']))
    conductance=p['a_is']*voltage**p['gamma']
    for name,value in [('X',precursor),('C',current),('Vis',voltage),('g_i',conductance),('Vbar',voltage)]:
        state['cones','reduced_native',name]=value.copy()
    state['cones','analytic_RE','R']=background.copy();state['cones','analytic_RE','E']=background.copy()
    prepared['input_reference']['native_background_td']=background.copy()
    prepared['native_inputs']={k:background.copy() for k in prepared['native_inputs']}
    prepared['optical_aperture_input']=dict(input_identity=input_identity,parent_seed_sha256=parent_hash,
        parent_initialized_model=parent_seed['joint_background_initialization']['initialized_model_id'],
        native_unit_semantics='Fixed LMS reference-equivalent engineering; not general photopicTd',
        held_reservoir_policy='Original exterior products retained in both paired finite models',
        physiological_parameters_changed=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    t=prepared['time_seconds']
    y,anchor=native.construction_point(SimpleNamespace(states=state,time_seconds=t),background,t+1.,0)
    y,equilibrium=native.equilibrate_native(y,anchor,t)
    result_assembly,seed,binding=initialize_from_native(base,prepared,hc_configuration,y,root,protocol)
    meta=seed['joint_background_initialization'];meta.pop('initialized_model_id')
    meta.update(optical_aperture_input=prepared['optical_aperture_input'],
        inherited_HC_parameter_module='HC_H1_H2_functional_v2',new_input_background=True,
        optical_whole_retina_closure=False)
    meta['initialized_model_id']=canonical_hash(meta)
    seed['lateral_hc_interface']['initialized_model_id']=meta['initialized_model_id']
    seed['optical_aperture_input']=prepared['optical_aperture_input']
    if canonical_hash(parent_seed)!=parent_hash:raise ContractError('Parent seed changed during optical initialization')
    return result_assembly,seed,binding,y,equilibrium
