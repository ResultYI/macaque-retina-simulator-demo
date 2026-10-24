"""Versioned feed-forward AC/RGC replay using the original HybridRunner.

This is a numerical boundary cache, not a fitted/reduced cell model.  Cubic
Hermite interpolation is an approximation whose mesh must be checked for each
source protocol.  It does not make a 100 us observation cache exact.
Schema 1.2 binds the numerical content; older caches require recapture and review.
"""
from dataclasses import dataclass, fields, is_dataclass, replace
from copy import deepcopy
from enum import Enum
from pathlib import Path
from time import perf_counter
import hashlib
import json
import numpy as np
from scipy.interpolate import CubicHermiteSpline

from .contracts import ContractError, Quantity, SolverConfig, canonical_hash
from .assembly import ModelSpec, ComponentSpec, PortLink, ExternalInput, RetinaBuilder
from .hybrid import HybridRunner
from .network import InputBundle
from .spatial import InputChannel

SCHEMA = 'ac-rgc-boundary-replay-1.2'
AC = 'EffectiveInhibitoryAC'
RGC = 'HistoryConductanceRGC'
WHITELIST = ('ac_tau_scale', 'ac_output_gain', 'rgc_excitation_gain', 'rgc_inhibition_gain')


def implementation_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def runtime_identity():
    import scipy
    import sys
    return canonical_hash(dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        runtime={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(Path(__file__).parent.glob('*.py'))}))


def _classes():
    from .contracts import CellTable
    from .graph import ConnectionTable
    from .spatial import Population
    from .mechanism_contracts import EvidenceValue, MechanismProfile
    from .provenance import SourceRecord
    return {c.__name__: c for c in (ModelSpec, ComponentSpec, PortLink, ExternalInput,
        Quantity, CellTable, ConnectionTable, Population, EvidenceValue, MechanismProfile, SourceRecord)}


def _encode(value):
    """Data-only whitelist; no pickle, executable import or callable in cache."""
    if isinstance(value, Enum): return {'enum': value.value}
    if isinstance(value, np.ndarray):
        return {'array': value.tolist(), 'dtype': value.dtype.str, 'shape': list(value.shape)}
    if isinstance(value, np.generic): return value.item()
    if is_dataclass(value):
        if type(value).__name__ not in _classes(): raise ContractError('Unsupported replay data type')
        return {'class': type(value).__name__, 'fields': {f.name: _encode(getattr(value, f.name))
            for f in fields(value) if f.init}}
    if isinstance(value, tuple): return {'tuple': [_encode(v) for v in value]}
    if isinstance(value, list): return [_encode(v) for v in value]
    if isinstance(value, dict): return {k: _encode(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)): return value
    raise ContractError('Unsupported replay data value: '+type(value).__name__)


def _decode(value):
    if isinstance(value, list): return [_decode(v) for v in value]
    if not isinstance(value, dict): return value
    if 'class' in value:
        if value['class'] not in _classes(): raise ContractError('Unknown replay data class')
        return _classes()[value['class']](**{k: _decode(v) for k, v in value['fields'].items()})
    if 'array' in value:
        a = np.asarray(value['array'], dtype=value['dtype'])
        if list(a.shape) != value['shape']: raise ContractError('Replay array shape differs')
        return a
    if 'tuple' in value: return tuple(_decode(v) for v in value['tuple'])
    if 'enum' in value:
        from .contracts import RunMode
        return RunMode(value['enum'])
    return {k: _decode(v) for k, v in value.items()}


def group(spec):
    if spec.mechanism == AC: return spec.options['generic_ac_inhibition']['type_group']
    return spec.required_layer or spec.component_id


def candidate_spec(spec, candidate=None):
    """Type-shared overrides; unchanged edge membership, units and mechanisms.

    Output/input gains multiply existing nonnegative contact coefficients in a
    temporary spec.  Original profiles/evidence and the main config stay intact.
    """
    candidate = candidate or {}
    if set(candidate)-set(WHITELIST): raise ContractError('Parameter outside replay whitelist')
    specs = {s.component_id: s for s in spec.components}
    ac_groups = {group(s) for s in specs.values() if s.mechanism == AC}
    rgc_groups = {group(s) for s in specs.values() if s.mechanism == RGC}
    for key, mapping in candidate.items():
        allowed = ac_groups if key.startswith('ac_') else rgc_groups
        if not isinstance(mapping, dict) or set(mapping)-allowed:
            raise ContractError('Override must use an existing shared type group: '+key)
        if any(isinstance(x, bool) or not np.isfinite(x) or x < .5 or x > 2.
               for x in mapping.values()):
            raise ContractError('Replay multiplier outside declared engineering domain [.5,2]')
    def factor(key, cid): return float(candidate.get(key, {}).get(group(specs[cid]), 1.))
    components = []
    for s in spec.components:
        if s.mechanism == AC:
            control = dict(s.options['generic_ac_inhibition'])
            control['tau_A_seconds'] *= factor('ac_tau_scale', s.component_id)
            s = replace(s, options=dict(s.options, generic_ac_inhibition=control))
        components.append(s)
    links = []
    for link in spec.links:
        scale = 1.
        if specs[link.source_component].mechanism == AC and link.source_port == 'activation':
            scale *= factor('ac_output_gain', link.source_component)
        if specs[link.target_component].mechanism == RGC:
            key = 'rgc_excitation_gain' if link.target_port == 'excitation' else 'rgc_inhibition_gain'
            scale *= factor(key, link.target_component)
        table = replace(link.connections, weight=tuple(float(w)*scale for w in link.connections.weight))
        links.append(replace(link, connections=table))
    return replace(spec, components=tuple(components), links=tuple(links))


def _layout(compiled, mode):
    if mode not in ('ac_rgc', 'rgc_only'): raise ContractError('Unknown replay boundary')
    components = compiled.components
    rgcs = {cid for cid, c in components.items() if c.mechanism == RGC}
    acs = {cid for cid, c in components.items() if c.mechanism == AC} if mode == 'ac_rgc' else set()
    if not rgcs or (mode == 'ac_rgc' and not acs): raise ContractError('No continuous downstream targets')
    kept = rgcs | acs
    adapters = set()
    for l in compiled.links:
        s = l.specification
        if s.target_component in rgcs and s.target_port == 'inhibition' and mode == 'ac_rgc':
            if components[s.source_component].mechanism != 'SynapticConductance':
                raise ContractError('RGC inhibition must use existing AC conductance adapter')
            adapters.add(s.source_component)
    kept |= adapters
    for g in compiled.spec.gap_junctions:
        if g.left_component in kept or g.right_component in kept:
            raise ContractError('Electrical coupling crosses replay boundary')
    channels = []
    for cid in sorted(acs): channels.append(('c:'+cid, cid, 'drive'))
    for cid in sorted(rgcs):
        channels.append(('gE:'+cid, cid, 'excitation'))
        if mode == 'rgc_only': channels.append(('gI:'+cid, cid, 'inhibition'))
    boundary = {(cid, port) for _, cid, port in channels}
    for l in compiled.links:
        s = l.specification
        if s.source_component in kept and s.target_component not in kept:
            raise ContractError('Downstream feedback crosses replay boundary: '+s.source_component)
        if s.target_component in kept:
            if l.delayed_operator is not None: raise ContractError('Delayed replay edge unsupported')
            if (s.target_component, s.target_port) in boundary:
                # The only admissible changing boundary is adapted BC release.
                if s.target_component in acs:
                    if components[s.source_component].mechanism != 'HistoryBipolarTerminal' or s.source_port != 'activation':
                        raise ContractError('AC boundary must be pooled BC r_eff')
                else:
                    if components[s.source_component].mechanism != 'SynapticConductance':
                        raise ContractError('RGC boundary must be conductance, not activation')
            elif s.source_component not in kept:
                raise ContractError('Uncached upstream input to retained component')
        if s.target_component in adapters and s.source_component not in acs:
            raise ContractError('Inhibitory adapter has a non-generic-AC source')
    retained = tuple(s for s in compiled.spec.components if s.component_id in kept)
    populations = {pid for s in retained for pid in s.population_roles.values()}
    links = tuple(s for s in compiled.spec.links if s.source_component in kept and
                  s.target_component in kept and (s.target_component, s.target_port) not in boundary)
    template = replace(compiled.spec, populations=tuple(p for p in compiled.spec.populations
        if p.population_id in populations), components=retained, links=links,
        external_inputs=tuple(ExternalInput(*row) for row in channels),
        required_mechanisms=(), algebraic_closure=None, gap_junctions=())
    return template, channels


class BoundaryExtractor:
    """Only reads the BC/AC boundary states; never evaluates the cone-HC closure.

    The two scalar release derivatives are the identical existing terminal
    equations (also used by the compiled backend), with the original profiles.
    All upstream feedback is already reflected in the cached BC V/r/A states.
    """
    def __init__(self, runner, channels):
        self.runner=runner; self.channels=channels; self.release={}
        for cid,c in runner.compiled.components.items():
            if c.mechanism!='HistoryBipolarTerminal':continue
            p=c.parameters
            self.release[cid]=dict(half=p.quantity('release_half','V','activation_half_voltage').values,
                slope=p.quantity('release_slope','V','activation_voltage_slope').values,
                tau=p.quantity('tau_release','s','release_time_constant').values)

    def evaluate(self,t,y,*,derivatives=False):
        from scipy.special import expit
        model=self.runner.model; compiled=self.runner.compiled; memo={}; dots={}
        def incoming(cid,port,differential=False):
            target=compiled.components[cid].inputs[port]
            links=model._incoming.get((cid,port),())
            if not links:raise ContractError('Boundary needs explicit source links: '+cid+'.'+port)
            result=np.zeros(len(target.cell_ids))
            for l in links:
                if l.delayed_operator is not None:raise ContractError('Delayed boundary source unsupported')
                s=l.specification;result+=l.matrix@output(s.source_component,s.source_port,differential)
            return result
        def output(cid,port,differential=False):
            key=(cid,port); table=dots if differential else memo
            if key in table:return table[key]
            c=compiled.components[cid];b=model.local_blocks(cid,y)
            if c.mechanism=='HistoryBipolarTerminal' and port=='activation':
                n=c.native_model;k=n.release_state_index;r=b[k];control=n.bc_output_gain_control
                A=b[k+1] if control.enabled else np.zeros_like(r)
                g=control.gain(A)
                if differential:
                    p=self.release[cid]
                    dr=(expit((b[k-1]-p['half'])/p['slope'])-r)/p['tau']
                    dA=(control.scale*r-A)/control.tau_a_seconds if control.enabled else np.zeros_like(r)
                    result=dr*g-r*6*A**5*g*g*dA
                else:result=r*g
            elif c.mechanism=='SynapticConductance' and port=='conductance':
                gain=c.parameters.quantity('gain','S','synaptic_conductance_scale').values
                result=gain*incoming(cid,'activation',differential)
            elif c.mechanism==AC and port=='activation':
                n=c.native_model
                result=n.output_scale*((n.target(incoming(cid,'drive'))-b[0])/n.tau_A_seconds if differential else b[0])
            else:raise ContractError('Boundary is not adapted BC / continuous AC / original conductance: '+cid+'.'+port)
            table[key]=result
            return result
        values=np.concatenate([incoming(cid,port) for _,cid,port in self.channels])
        if not derivatives:return values
        return values,np.concatenate([incoming(cid,port,True) for _,cid,port in self.channels])


@dataclass(frozen=True)
class _PreparedOrigin:
    checkpoint_hash: str
    receipt: dict


class BoundaryRecorder:
    """Pass observer to source advance; record dense requested nodes and event sides.

    Source sampling/solver settings are not changed.  Call finish with its actual
    returned state.  Unadvanced capture requires a pre-draw seed.  Advanced
    capture is available only through from_prepared_background, which verifies
    the original background envelope and preserves its complete time/RNG phase.
    """
    def __init__(self, runner, initial_state, background_channels, *, identity, mode='ac_rgc', registry=None,
                 _prepared_origin=None):
        required = ('stimulus_hash', 'calibration_hash', 'spatial_mapping_hash', 'background_hash')
        if any(not isinstance(identity.get(k), str) or not identity[k] for k in required):
            raise ContractError('Explicit stimulus/calibration/spatial/background identity required')
        advanced=(initial_state.sample_index != 0 or initial_state.time_seconds != initial_state.grid_origin_seconds or
            any(a['last_bin_counts'] is not None or a['pending_events'] for a in initial_state.spike_states.values()))
        if advanced and _prepared_origin is None:
            raise ContractError('Advanced background requires from_prepared_background; no implicit trial/history reset')
        if _prepared_origin is not None:
            if not isinstance(_prepared_origin,_PreparedOrigin) or not runner.event_clock:
                raise ContractError('Verified prepared continuation and explicit source calendar required')
            if _prepared_origin.checkpoint_hash!=canonical_hash(json.loads(runner.checkpoint(initial_state))):
                raise ContractError('Prepared initial state changed after verified load')
        self.runner = runner
        self.template, self.channels = _layout(runner.compiled, mode)
        self.rows = []; self.mode = mode
        self.extractor=BoundaryExtractor(runner,self.channels)
        y = runner._flat(initial_state)
        if not background_channels or any(ch.input_contract.get('declared_constant_background') is not True
            for ch in background_channels.values()):
            raise ContractError('Explicit constant background identity required')
        self.background = self.extractor.evaluate(initial_state.time_seconds,y)
        # A fixed replay must not silently replace an unprepared source seed by
        # an equilibrium.  Check only the retained deterministic boundary, not
        # stochastic stationarity or the entire upstream retina.
        bg_by_key={key:self.background[sum(len(runner.compiled.components[c].inputs[p].cell_ids)
            for _,c,p in self.channels[:j]):sum(len(runner.compiled.components[c].inputs[p].cell_ids)
            for _,c,p in self.channels[:j+1])] for j,(key,_,_) in enumerate(self.channels)}
        gi_channels=[('gI:'+cid,cid,'inhibition') for cid,c in runner.compiled.components.items()
                     if c.mechanism==RGC]
        gi_values=BoundaryExtractor(runner,gi_channels).evaluate(initial_state.time_seconds,y)
        gi_offset=0;source_states={}
        for cid,c in runner.compiled.components.items():
            if c.mechanism not in (AC,RGC):continue
            actual=runner.model.local_blocks(cid,y)[0]
            if c.mechanism==AC and mode=='ac_rgc':
                expected=c.native_model.target(bg_by_key['c:'+cid])
            elif c.mechanism==RGC:
                n=c.native_model;e=bg_by_key['gE:'+cid];i=gi_values[gi_offset:gi_offset+len(actual)]
                gi_offset+=len(actual)
                expected=(n.gL*n.EL+e*n.EE+i*n.EI)/(n.gL+e+i)
            else:continue
            tolerance=initial_state.solver_config.atol+initial_state.solver_config.rtol*np.maximum(abs(actual),abs(expected))
            if not advanced and np.any(abs(actual-expected)>tolerance):
                raise ContractError('Source downstream background is not closed at its declared solver tolerance: '+cid)
            source_states[cid]=actual.tolist()
        valid_from={cid:float(initial_state.grid_origin_seconds+
            initial_state.component_tick_indices[cid]*runner.periods[cid])
            for cid in initial_state.spike_states}
        self.initial = dict(time=initial_state.time_seconds, origin=initial_state.grid_origin_seconds,
            sample_index=initial_state.sample_index, ticks=dict(initial_state.component_tick_indices),
            spikes=json.loads(json.dumps(initial_state.spike_states)),source_states=source_states,
            source_input_position=json.loads(json.dumps(initial_state.input_position)),
            initialization_policy='prepared_continuation' if advanced else 'unadvanced_equilibrium',
            first_new_draw_bin_seconds=valid_from,
            preparation=None if _prepared_origin is None else _prepared_origin.receipt)
        self.identity = dict(identity, source_assembly_hash=runner.compiled.assembly_hash,
            source_model_hash=runner.model.model_hash,
            source_checkpoint_hash=hashlib.sha256(runner.checkpoint(initial_state).encode()).hexdigest(),
            boundary_mode=mode, replay_implementation=implementation_identity(),runtime_identity=runtime_identity())
        self.profiles = tuple({c.parameters.profile_id: c.parameters for cid, c in
            runner.compiled.components.items() if cid in {s.component_id for s in self.template.components}}.values())
        if registry is None:
            from .core_sources import load_core_registry
            registry=load_core_registry(Path(__file__).resolve().parents[1])
        source_ids={source for p in self.profiles for source in p.source_ids}
        source_ids.update(r.source_id for p in self.profiles for r in p.records if r.source_id is not None)
        if source_ids-set(registry.sources):
            raise ContractError('Pass the original build audit registry to preserve dynamic source identities: '+str(sorted(source_ids-set(registry.sources))))
        self.source_records=tuple(registry.sources[s] for s in sorted(source_ids))
        self.identity['source_records_hash']=canonical_hash(self.source_records)
        self.clock = json.loads(json.dumps(runner.compiled.clock))
        self.config = initial_state.solver_config
        self.observation_dt = runner.observation_dt

    @classmethod
    def from_prepared_background(cls, store, path, background_channels, *, identity, mode='ac_rgc', registry=None):
        """Load the unchanged qualified background, then continue at absolute time.

        Returns (source_state, recorder).  Original threshold/window/coverage,
        geometry, clock, solver and full history checks remain in the store.
        No clock rebasing, projection, redraw or pending-event deletion occurs.
        """
        from .background_checkpoint import BackgroundCheckpointStore
        if not isinstance(store,BackgroundCheckpointStore):raise ContractError('Original background checkpoint store required')
        state,envelope=store.load(path,require_converged=True)
        receipt=dict(binding_hash=envelope['binding_hash'],complete_checkpoint_hash=envelope['complete_checkpoint_hash'],
            report_hash=envelope['report_hash'],absolute_time_seconds=envelope['absolute_time_seconds'],
            policy_id=store.policy.profile_id,status='verified_original_background_contract',
            time_policy='unchanged_absolute_continuation',history_policy='unchanged_conditioned_preparation_history')
        origin=_PreparedOrigin(envelope['complete_checkpoint_hash'],receipt)
        return state,cls(store.runner,state,background_channels,identity=identity,mode=mode,
                        registry=registry,_prepared_origin=origin)

    def observe(self, t, y, bundle):
        value, derivative = self.extractor.evaluate(t,y,derivatives=True)
        self.rows.append((float(t), value.copy(), derivative.copy()))

    def finish(self, final_state, *, stimulus_breakpoints=(), background_status='unqualified_conditional_background'):
        if len(self.rows) < 2: raise ContractError('Empty boundary cache')
        rows = []
        for row in self.rows:
            if rows and abs(row[0]-rows[-1][0]) < 1e-14:
                # BC r/A and AC h are continuous through old sampled transitions.
                if not np.array_equal(row[1], rows[-1][1]):
                    raise ContractError('Discontinuous boundary needs explicit left/right interpolants')
                rows[-1] = row
            else:
                if rows and row[0] < rows[-1][0]: raise ContractError('Boundary time goes backwards')
                rows.append(row)
        t = np.array([r[0] for r in rows]); v = np.stack([r[1] for r in rows]); d = np.stack([r[2] for r in rows])
        if abs(t[0]-self.initial['time']) > 1e-12 or abs(t[-1]-final_state.time_seconds) > 1e-12:
            raise ContractError('Incomplete boundary coverage')
        metadata = dict(schema_version=SCHEMA, mode=self.mode, identity=self.identity,
            identity_hash=canonical_hash(self.identity), template=_encode(self.template), profiles=_encode(self.profiles),
            source_records=_encode(self.source_records),
            run_mode=_encode(self.runner.compiled.run_mode),
            solver={f.name: getattr(self.config, f.name) for f in fields(self.config)},
            source_clock=self.clock, observation_dt=self.observation_dt, initial=self.initial,
            background_status=background_status,
            stochastic_background_claim='Conditioned explicit initial history and PCG64; no stationary spike-history claim',
            channels=[], stimulus_breakpoints=list(stimulus_breakpoints),
            interpolation='piecewise_cubic_Hermite_exact_boundary_derivatives_APPROXIMATE',
            raw_observer_count=len(self.rows), maximum_node_gap_seconds=float(np.max(np.diff(t))),
            implementation_sha256=implementation_identity())
        offset = 0
        for key, cid, port in self.channels:
            p = self.runner.compiled.components[cid].inputs[port]; n = len(p.cell_ids)
            metadata['channels'].append(dict(key=key, component=cid, port=port, start=offset, stop=offset+n,
                unit=p.unit, meaning=p.meaning, evidence_tag=p.evidence_tag, cell_ids=list(p.cell_ids)))
            offset += n
        return BoundaryCache(metadata, t, v, d, self.background.copy())


class BoundaryCache:
    def __init__(self, metadata, times, values, derivatives, background, *, expected_content_hash=None):
        self.metadata = deepcopy(metadata)
        metadata = self.metadata
        self.times = np.array(times, dtype=np.float64, copy=True)
        self.values = np.array(values, dtype=np.float64, copy=True)
        self.derivatives = np.array(derivatives, dtype=np.float64, copy=True)
        self.background = np.array(background, dtype=np.float64, copy=True)
        width = metadata['channels'][-1]['stop']
        if (metadata.get('schema_version') != SCHEMA or metadata.get('implementation_sha256') != implementation_identity() or
            metadata['identity'].get('runtime_identity') != runtime_identity() or
            metadata.get('identity_hash') != canonical_hash(metadata['identity']) or
            self.times.ndim != 1 or len(self.times) < 2 or np.any(np.diff(self.times) <= 0) or
            self.values.shape != (len(self.times), width) or self.derivatives.shape != self.values.shape or
            self.background.shape != (width,) or any(not np.all(np.isfinite(a)) for a in
                (self.times, self.values, self.derivatives, self.background))):
            raise ContractError('Boundary version/layout/finite-state contract differs')
        if np.any(self.values < 0) or np.any(self.background < 0): raise ContractError('Negative boundary signal')
        self.content_hash = self._content_hash()
        if expected_content_hash is not None and expected_content_hash != self.content_hash:
            raise ContractError('Boundary content identity differs')
        self.metadata['content_hash'] = self.content_hash
        for array in (self.times,self.values,self.derivatives,self.background):array.setflags(write=False)
        self.spline = CubicHermiteSpline(self.times.copy(), self.values, self.derivatives, extrapolate=False)

    def _content_hash(self):
        # Storage/compression checksums do not change the numerical boundary.
        metadata={k:v for k,v in self.metadata.items() if k not in ('content_hash','arrays_sha256','manifest_hash')}
        return canonical_hash(dict(metadata=metadata,times=self.times,values=self.values,
            derivatives=self.derivatives,background=self.background))

    def require_content_identity(self):
        if self.metadata.get('content_hash') != self.content_hash or self._content_hash() != self.content_hash:
            raise ContractError('Boundary content changed after identity verification')

    @property
    def identity_hash(self): return self.metadata['identity_hash']

    def save(self, directory):
        self.require_content_identity()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory/'boundary.npz', times=self.times, values=self.values,
            derivatives=self.derivatives, background=self.background)
        data = dict(self.metadata, arrays_sha256=hashlib.sha256((directory/'boundary.npz').read_bytes()).hexdigest())
        data['manifest_hash'] = canonical_hash(data)
        (directory/'manifest.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf8', newline='\n')

    @classmethod
    def load(cls, directory, *, expected_identity_hash):
        directory = Path(directory); m = json.loads((directory/'manifest.json').read_bytes())
        if m.get('schema_version') != SCHEMA or not isinstance(m.get('content_hash'),str):
            raise ContractError('Legacy or unsupported boundary cache; recapture and revalidate, no automatic migration')
        digest = m.pop('manifest_hash')
        if digest != canonical_hash(m) or m['identity_hash'] != expected_identity_hash or m['identity_hash'] != canonical_hash(m['identity']):
            raise ContractError('Boundary manifest/expected upstream identity differs')
        if m['arrays_sha256'] != hashlib.sha256((directory/'boundary.npz').read_bytes()).hexdigest():
            raise ContractError('Boundary array hash differs')
        with np.load(directory/'boundary.npz', allow_pickle=False) as z:
            return cls(m, *(z[k] for k in ('times', 'values', 'derivatives', 'background')),
                expected_content_hash=m['content_hash'])


class BoundaryReplay:
    def __init__(self, cache, root, *, candidate=None, compute_backend='python'):
        from .core_sources import load_core_registry
        from .mechanism_contracts import register_mechanism_profile
        started = perf_counter()
        cache.require_content_identity()
        self.cache = cache; self.candidate = candidate or {}
        registry = load_core_registry(root)
        sources=_decode(cache.metadata['source_records'])
        if canonical_hash(sources)!=cache.metadata['identity']['source_records_hash']:
            raise ContractError('Boundary source record identity differs')
        for record in sources:
            existing=registry.sources.get(record.source_id)
            if existing is not None:
                if canonical_hash(existing)!=canonical_hash(record):raise ContractError('Replay source identity differs')
            else:registry.register_source(record)
        for p in _decode(cache.metadata['profiles']):
            existing = registry.profiles.get(p.profile_id)
            if existing is not None:
                if existing.parameter_hash != p.parameter_hash: raise ContractError('Replay parameter profile identity differs')
            else: register_mechanism_profile(registry, p)
        self.template = _decode(cache.metadata['template'])
        spec = candidate_spec(self.template, self.candidate)
        compiled = RetinaBuilder(registry).build(spec, _decode(cache.metadata['run_mode']))
        # Calendar-only entries keep the original sampling/observation boundaries,
        # without upstream cells or transitions.  No extra RNG draw is introduced.
        compiled.clock = json.loads(json.dumps(cache.metadata['source_clock']))
        compiled.assembly_hash = canonical_hash(dict(downstream=compiled.assembly_hash,
            cache=cache.identity_hash, cache_content=cache.content_hash, calendar=compiled.clock, candidate=self.candidate,
            replay=implementation_identity(), interpolation=cache.metadata['interpolation']))
        self.runner = HybridRunner(compiled, compute_backend=compute_backend)
        self.runner.observation_dt = cache.metadata['observation_dt']
        config = SolverConfig(**cache.metadata['solver'])
        # Source input interpolation belongs to the upstream channels.  The replay
        # external signal is explicitly an analytic piecewise cubic approximation.
        # This changes no integration method/tolerance/dtype/event boundary.
        self.config = replace(config, input_interpolation='analytic')
        self.channels = self._channels(background=False)
        self.background_channels = self._channels(background=True)
        self.build_seconds = perf_counter()-started

    def _channels(self, *, background):
        m = self.cache.metadata; specs = {s.component_id:s for s in self.template.components}
        channels = {}
        for row in m['channels']:
            scale = 1.
            if row['port'] in ('excitation','inhibition'):
                key = 'rgc_excitation_gain' if row['port']=='excitation' else 'rgc_inhibition_gain'
                scale = self.candidate.get(key, {}).get(group(specs[row['component']]), 1.)
            sl = slice(row['start'],row['stop'])
            def evaluate(t, sl=sl, scale=scale, row=row):
                if t < self.cache.times[0]-1e-12 or t > self.cache.times[-1]+1e-12:
                    raise ContractError('Boundary extrapolation forbidden')
                value = (self.cache.background[sl] if background else self.cache.spline(t)[sl])*scale
                if not np.all(np.isfinite(value)) or np.any(value < 0):
                    raise ContractError('Boundary interpolation became negative/nonfinite; refine source mesh')
                return Quantity(value,row['unit'],row['meaning'],row['evidence_tag'])
            channels[row['key']] = InputChannel(float(self.cache.times[0]),float(self.cache.times[-1]),
                evaluate,row['unit'],row['meaning'],'analytic',row['evidence_tag'],
                dict(boundary_schema=SCHEMA,cache_identity=self.cache.identity_hash,cache_content_hash=self.cache.content_hash,candidate=self.candidate,
                     interpolation=m['interpolation'], declared_constant_background=background),
                () if background else tuple(float(t) for t in m['stimulus_breakpoints']
                    if self.cache.times[0] < t < self.cache.times[-1]))
        return channels

    def initialize(self):
        """Re-solve deterministic candidate background, preserve explicit prehistory/RNG."""
        init = self.cache.metadata['initial']; r = self.runner
        state = r.initialize(self.config, init['time'])
        state.grid_origin_seconds = init['origin']; state.sample_index = init['sample_index']
        state.component_tick_indices = dict(init['ticks'])
        state.spike_states = json.loads(json.dumps(init['spikes']))
        y = r._flat(state)
        bundle = InputBundle({k:ch.at(init['time']) for k,ch in self.background_channels.items()})
        for cid,c in r.compiled.components.items():
            if c.mechanism == AC:
                drive = bundle.channels['c:'+cid].values
                y[r.model.slices[cid]] = c.native_model.target(drive)
        incoming,_ = r.model.evaluate(init['time'],y,bundle)
        for cid,c in r.compiled.components.items():
            if c.mechanism != RGC: continue
            n = c.native_model; e=incoming[cid]['excitation'].values; i=incoming[cid]['inhibition'].values
            voltage = (n.gL*n.EL+e*n.EE+i*n.EI)/(n.gL+e+i)
            y[r.model.slices[cid]] = voltage
            if state.spike_states[cid]['last_bin_counts'] is None:
                state.spike_states[cid]['conditional_rate_Hz'] = n.intensity(voltage,
                    np.asarray(state.spike_states[cid]['history'])).tolist()
        if not self.candidate:
            # The exact baseline starts from the original retained background.
            for cid,values in init['source_states'].items():
                if cid in r.compiled.components:y[r.model.slices[cid]]=values
            for cid,aux in state.spike_states.items():
                if aux['last_bin_counts'] is None:
                    aux['conditional_rate_Hz']=r.compiled.components[cid].native_model.intensity(
                        y[r.model.slices[cid]],np.asarray(aux['history'])).tolist()
        state.input_position=dict(phase='boundary_replay',cache_identity=self.cache.identity_hash,cache_content_hash=self.cache.content_hash,
            source_context=init.get('source_input_position',{}),
            initialization_policy=init.get('initialization_policy','unadvanced_equilibrium'),
            first_new_draw_bin_seconds=init['first_new_draw_bin_seconds'])
        r._replace(state,y,init['time'])
        residual = r.model.rhs_flat(init['time'],y,bundle,r.model.parameters,r.model.graph)
        self.background_residual = float(np.max(abs(residual),initial=0.))
        return state

    def advance(self,state,stop,*,sample_times=None,recording_plan=None):
        return self.runner.advance(state,self.channels,stop,self.config,
            sample_times=sample_times,recording_plan=recording_plan)

    def checkpoint(self,state):
        self.cache.require_content_identity()
        return self.runner.checkpoint(state)
    def restore(self,payload):
        self.cache.require_content_identity()
        return self.runner.restore(payload,expected_input_context=dict(
            cache_identity=self.cache.identity_hash,cache_content_hash=self.cache.content_hash))


def teacher_forced_probability(voltage, observed_binary, native, *, initial_history, return_hazard=False):
    """Separate scoring path: observed strict-past history, never sampled history.

    Rows are bin starts; each column is a cell, bin width is the native interval.
    No trailing endpoint may be passed as an extra observed bin.
    """
    v=np.asarray(voltage,dtype=float); y=np.asarray(observed_binary)
    h=np.asarray(initial_history,dtype=float).copy()
    if (v.shape!=y.shape or v.ndim!=2 or h.shape!=(len(native.history_kernel),v.shape[1]) or
        np.any((y!=0)&(y!=1)) or np.any((h!=0)&(h!=1)) or not np.all(np.isfinite(v))):
        raise ContractError('Teacher-forced voltage/binary/history layout differs')
    result=np.empty_like(v)
    for t in range(len(v)):
        result[t]=native.intensity(v[t],h)*native.dt
        if len(h):h=np.vstack((y[t][None,:],h[:-1]))
    return result if return_hazard else -np.expm1(-result)


def absolute_trial_channels(channels, absolute_start_seconds):
    """Anchor a declared relative stimulus to the existing clock, never the state.

    This deterministic timestamp mapping is not an estimated latency.  Delays,
    sampled histories and RNG in the prepared source remain untouched.
    """
    if not channels or not np.isfinite(absolute_start_seconds):raise ContractError('Finite explicit trial origin required')
    if {ch.start_seconds for ch in channels.values()}!={0.}:
        raise ContractError('Trial channels must explicitly start at relative zero')
    out={}
    for name,ch in channels.items():
        shifted=np.asarray([absolute_start_seconds+x for x in ch.breakpoints_seconds])
        def at(t,ch=ch,shifted=shifted):
            relative=t-absolute_start_seconds
            # Preserve declared event sides even when subtraction loses an ULP.
            # Do not tolerance-snap a left limit onto the departing right value.
            if len(shifted):
                i=int(np.searchsorted(shifted,t,side='right'))
                if i and (t==shifted[i-1] or relative<ch.breakpoints_seconds[i-1]):relative=ch.breakpoints_seconds[i-1]
                if i<len(shifted) and relative>=ch.breakpoints_seconds[i]:
                    relative=np.nextafter(ch.breakpoints_seconds[i],-np.inf)
            return ch.at(relative)
        out[name]=InputChannel(absolute_start_seconds,absolute_start_seconds+ch.stop_seconds,at,
            ch.unit,ch.meaning,ch.interpolation,ch.evidence_tag,
            dict(ch.input_contract,absolute_trial_start_seconds=float(absolute_start_seconds),
                 time_mapping='source_time=relative_trial_time+explicit_origin; no state rebasing'),
            tuple(absolute_start_seconds+t for t in ch.breakpoints_seconds))
    return out


def expected_replay_count_rates(replay, result):
    """Exact eight-state expected *binary count rate*, not conditional hazard.

    Uses recorded exact native bin-start V and the saved preparation history;
    does not draw or change RNG.  An already drawn preparation bin is context,
    excluded mechanically.  It is not replaced by its sampled 0/1000-Hz trace.
    No stationary-history assumption or voltage interpolation is introduced.
    """
    from .rgc_expectation import STATES,NEXT0,NEXT1,INDEX
    times=np.asarray(result.outputs_with_units['time'].values)
    initial=replay.cache.metadata['initial'];outputs={}
    for cid,c in replay.runner.compiled.components.items():
        if c.mechanism!=RGC:continue
        n=c.native_model;dt=n.dt;aux=initial['spikes'][cid]
        if np.shape(n.history_kernel)!=(3,) or not n.deterministic_state_independent_of_spikes:
            raise ContractError('Expected-rate adapter requires the existing spike-independent three-tap sampler')
        first=initial['first_new_draw_bin_seconds'][cid]
        starts=np.arange(first,times[-1]-.5*dt,dt)
        edges=np.r_[starts,first+len(starts)*dt]
        if not len(starts) or abs(edges[-1]-times[-1])>1e-12:
            raise ContractError('Expected-rate output needs complete bins after the preparation-conditioned bin')
        indices=[]
        for t in starts:
            found=np.flatnonzero(abs(times-t)<1e-12)
            if len(found)!=1:raise ContractError('Exact native bin-start V missing; no interpolation')
            indices.append(int(found[0]))
        h=np.asarray(aux['history'])
        if aux['last_bin_counts'] is not None:
            h=np.vstack((np.asarray(aux['last_bin_counts'])[None,:],h[:-1]))
        if h.shape!=(3,n.n) or not np.isin(h,(0,1)).all():raise ContractError('Saved binary preparation history differs')
        law=np.zeros((n.n,8))
        for cell in range(n.n):law[cell,INDEX[tuple(int(x) for x in h[:,cell])]]=1.
        v=result.outputs_with_units[cid+'.voltage'].values[:,indices]
        rates=np.empty((len(starts),n.n));mass_error=0.
        for j in range(len(starts)):
            lam=n.alpha[:,None]*np.logaddexp(0.,(v[:,j,None]+(STATES@n.history_kernel)[None,:]-n.mu[:,None])/n.beta[:,None])
            p=-np.expm1(-lam*dt);rates[j]=(law*p).sum(axis=1)/dt
            next_law=np.zeros_like(law)
            for k in range(8):
                next_law[:,NEXT0[k]]+=law[:,k]*(1-p[:,k])
                next_law[:,NEXT1[k]]+=law[:,k]*p[:,k]
            law=next_law;mass_error=max(mass_error,float(np.max(abs(law.sum(axis=1)-1))))
        if mass_error>1e-12:raise ContractError('Expected history probability conservation failed')
        outputs[cid]=dict(rates_Hz=rates,bin_edges_seconds=edges,unit='Hz',
            quantity='expected_discrete_count_rate',cell_ids=list(c.state_specs[0].cell_ids),
            conditioning='saved preparation binary history; existing sampled bin omitted',
            omitted_prefix_seconds=float(first-initial['time']),maximum_probability_mass_error=mass_error)
    return outputs
