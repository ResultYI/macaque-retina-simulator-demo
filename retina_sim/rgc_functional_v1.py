"""Research RGC readout over the retained, bidirectionally coupled AC network.

This interface adds no membrane, receptor, spike mechanism or connectivity.
Its presence alone does not qualify a background or complete a functional run.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .ac_functional_v1 import ACFunctionalV1
from .contracts import ContractError, canonical_hash
from .spatial_group_model import RGC


class RGCFunctionalV1:
    def __init__(self, root, configuration):
        self.root = Path(root)
        cfg = (json.loads((self.root / configuration).read_text(encoding='utf8'))
               if isinstance(configuration, (str, Path)) else dict(configuration))
        self.configuration = cfg
        if (cfg['schema'] != 'RGC-functional-effective-layer-1'
                or cfg['new_kinetic_mechanism'] is not False
                or cfg['new_connections'] is not False
                or tuple(cfg['types']) != tuple(RGC)
                or any(cfg[n] is not False for n in
                       ('physiologically_calibrated', 'production_adopted',
                        'background_qualified', 'FIT_READY'))):
            raise ContractError('Explicit retained research RGC identity required')
        self.ac = ACFunctionalV1(self.root, cfg['AC_configuration'])
        if self.ac.identity != cfg['AC_layer_identity']:
            raise ContractError('Frozen AC interface differs; no implicit requalification')
        self.model = self.ac.model
        self.populations = tuple(RGC)
        self.origin = self.ac.bc.origin
        if any(p not in self.model.counts or self.model.counts[p] <= 0
               for p in self.populations):
            raise ContractError('All existing RGC types must remain represented')
        self.fields = self._fields()
        self.types = {}
        for pop in self.populations:
            p = self.ac.assembly.populations[pop]
            self.types[pop] = dict(
                group_ids=list(map(str, p.group_ids)),
                groups=p.group_count, represented_members=p.counts.tolist(),
                total_represented_members=int(p.counts.sum()),
                independent_fine_cell_spikes=False)
        self.routes = [dict(name=r.name, pre=r.pre, post=r.post,
                            receptor='inhibitory' if r.pre in self.model.ac_populations
                            else 'excitatory')
                       for r in self.model.routes if r.post in self.populations]
        self.identity = canonical_hash(dict(
            configuration=cfg, AC_identity=self.ac.identity,
            source_inner=self.model.identity, fields=self.fields,
            types=self.types, routes=self.routes,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

    def _fields(self):
        specifications = (
            ('excitatory_conductance', 'S', 'Original total grouped gE'),
            ('inhibitory_conductance', 'S', 'Original total grouped gI'),
            ('voltage', 'V', 'Original graded membrane voltage'),
            ('excitatory_current', 'A', 'Original gE*(EE-V)'),
            ('inhibitory_current', 'A', 'Original gI*(EI-V)'),
            ('leak_current', 'A', 'Original grouped gL*(EL-V)'),
            ('net_current', 'A', 'Sum of original membrane currents'),
            ('voltage_derivative', 'V/s', 'Original net current / grouped C'),
            ('expected_hazard', 'Hz', 'Original binary-history mean hazard'),
            ('expected_binary_count_rate', 'Hz',
             'Original E[at least one event per 1ms bin]/1ms; not hazard'),
        )
        return {pop + '__' + name: dict(
            population=pop, count=self.model.counts[pop], unit=unit, meaning=meaning,
            layout='actual_group_rows',
            rate_semantics='one sampled group object; no multiplication by members')
            for pop in self.populations for name, unit, meaning in specifications}

    def initial(self):
        return self.ac.bc.initial()

    def initial_spike_states(self):
        states = deepcopy(self.ac.bc.seed['spike_states'])
        if set(states) != set(self.populations):
            raise ContractError('Original per-type spike histories and RNG required')
        return states

    def rhs(self, t, y):
        return self.model.pack(self.model.rhs(t, self.model.unpack(y, copy=False)))

    def commit(self, index, state, cone_release, rod_light, spike_states):
        """One original 100us source event; BC/RGC retain their 1ms clocks."""
        if type(index) is not int or index < 0:
            raise ContractError('Nonnegative integer original 100us event index required')
        tick = index % 10 == 0
        return self.model.commit_ticks(
            self.origin + index * .0001, state,
            cone_release=cone_release if tick else None, rod_light=rod_light,
            rod_tick=True, bc_tick=tick, rgc_tick=tick, spike_states=spike_states)

    def observe(self, t, state, cone_release):
        return self.observe_ports(self.model.outputs(t, state, cone_release))

    def observe_ports(self, out):
        """Same physical readout for original full state or explicit driven inputs."""
        result = {}
        for pop in self.populations:
            V, ge, gi = (out[pop, n] for n in ('voltage', 'gE', 'gI'))
            if np.any(ge < 0) or np.any(gi < 0):
                raise ContractError('Negative original RGC conductance; no clipping')
            ie = ge * (self.model.par(pop, 'EE') - V)
            ii = gi * (self.model.par(pop, 'EI') - V)
            il = self.model.gL[pop] * (self.model.par(pop, 'EL') - V)
            total = il + ie + ii
            values = dict(
                voltage=V, excitatory_conductance=ge, inhibitory_conductance=gi,
                excitatory_current=ie, inhibitory_current=ii, leak_current=il,
                net_current=total, voltage_derivative=total / self.model.C[pop],
                expected_hazard=out[pop, 'expected_hazard_Hz'],
                expected_binary_count_rate=out[pop, 'expected_rate_Hz'])
            for name, value in values.items():
                value = np.asarray(value)
                if value.shape != (self.model.counts[pop],) or not np.isfinite(value).all():
                    raise ContractError('Finite complete RGC readout required: ' + pop + '.' + name)
                result[pop + '__' + name] = value.copy()
            if (np.any(values['expected_hazard'] < 0)
                    or np.any(values['expected_binary_count_rate'] < 0)
                    or np.any(values['expected_binary_count_rate'] > 1000 + 1e-10)):
                raise ContractError('Original 1ms binary-output rate domain differs')
        if set(result) != set(self.fields):
            raise ContractError('Declared RGC field coverage differs')
        return result
