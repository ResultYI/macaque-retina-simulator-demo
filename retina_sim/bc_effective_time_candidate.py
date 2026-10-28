"""Opt-in ON diffuse BC effective C binding; no production default changes."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from .contracts import ContractError, canonical_hash

SCHEMA = 'on-diffuse-bc-effective-membrane-time-1'
SELECTOR = 'bc.diffuse_ON.tau_eff_s'
POPULATION = 'diffuse_BC_ON'
CAPACITANCE = 'populations/' + POPULATION + '/C_total.npy'


class EffectiveTimeAssembly:
    """A real physical-total overlay, checked by GroupInnerDynamics as usual."""
    def __init__(self, source, seed, tau_seconds):
        tau = float(tau_seconds)
        if not np.isfinite(tau) or not .005 <= tau <= .030:
            raise ContractError('Approved ON diffuse BC effective time is 5..30ms')
        self.source = source
        self.manifest = deepcopy(source.manifest)
        self.runtime_factories = dict(source.runtime_factories)
        parameters = seed['inner_parameters']['populations'][POPULATION]
        gl = np.asarray(parameters['gL'], dtype=np.float64)
        old_c = np.asarray(parameters['C'], dtype=np.float64)
        if np.any(gl <= 0) or not np.allclose(old_c / gl, .020, rtol=0, atol=1e-17):
            raise ContractError('Approved original BC effective baseline is not20ms')
        # Preserve exact baseline bytes; the product can differ by one float ulp.
        effective_c = old_c.copy() if tau == .020 else gl * tau
        self.total_c = np.array(effective_c * source.populations[POPULATION].counts,
                                dtype=np.float64, copy=True)
        self.total_c.flags.writeable = False
        if CAPACITANCE not in self.manifest['files']:
            raise ContractError('Physical C_total validation asset must exist')
        self.binding = dict(schema=SCHEMA, selector=SELECTOR, population=POPULATION,
            tau_eff_seconds=tau, gL_fixed=True, effective_C_per_member_F=effective_c.tolist(),
            C_total_sha256=hashlib.sha256(self.total_c.tobytes()).hexdigest(),
            parent_artifact=source.artifact_sha256,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            interpretation='conditional effective membrane parameter, not measured capacitance',
            other_parameters_and_initial_states_fixed=True)
        # Keep the original source-file registry and explicitly declare the overlay.
        # array(C_total) supplies the actual changed totals; validation is not bypassed.
        self.manifest['effective_parameter_overlays'] = {CAPACITANCE: self.binding}
        self.artifact_sha256 = canonical_hash(self.binding)
        self.seed = deepcopy(seed)
        p = self.seed['inner_parameters']
        p['populations'][POPULATION]['C'] = effective_c.item() if effective_c.ndim == 0 else effective_c
        p['identity'] = canonical_hash(dict(parent=p['identity'], binding=self.binding))
        p['provenance'] = dict(parent=p.get('provenance'), effective_time=self.binding)
        self.seed['bc_effective_time_lineage'] = self.binding
        # C does not enter a constant equilibrium equation. Do not reprepare, or
        # reset, any native/inner state, contact, history, delay or RNG.

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'source'), name)

    def array(self, relative):
        return self.total_c if relative == CAPACITANCE else self.source.array(relative)

    def matrix(self, relative):
        return self.source.matrix(relative)
