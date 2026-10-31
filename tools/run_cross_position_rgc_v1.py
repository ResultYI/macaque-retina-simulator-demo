"""Actual coupled inner-network probes from an independent near-center seed."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.cross_position_context_v1 import load_background, write
from retina_sim.rgc_functional_v1 import RGCFunctionalV1
from retina_sim.spatial_group_model import RGC
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_time_storage import save_bundle
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.spatial_group_graph import digest


class LocalBC:
    def __init__(self, model, seed, background):
        self.model, self.seed, self.background = model, seed, background
        self.origin = float(seed['time_seconds'])
        self.rod_background = float(seed['input_reference']['rod_background_rstar_s'])

    def initial(self):
        return {k: np.array(self.seed['states'][k], copy=True) for k in self.model.state_keys}

    def inputs(self, index):
        cone = .9 if 500 <= index < 2000 else 1.1 if 3500 <= index < 5000 else 1.
        rod = 2. if 6000 <= index < 7500 else 1.
        return self.background * cone, np.full(self.model.counts['rods'], self.rod_background * rod)

    def endpoint(self, index, state, cone, rod):
        before = state
        state, _, _ = self.model.commit_ticks(self.origin + index * .0001, state,
            cone_release=cone, rod_light=rod, rod_tick=True, bc_tick=True, rgc_tick=False)
        state.update(self.model._probability_tick_updates(before))
        return state


class LocalRGC(RGCFunctionalV1):
    """Bind existing readout methods to a new, explicitly identified local model."""
    def __init__(self, prepared, background):
        a, seed, source = load_background(prepared, background)
        self.model = a.runtime_factories['inner'](a, seed['inner_parameters'])
        self.populations = tuple(p for p in RGC if self.model.counts[p])
        with np.load(Path(background) / 'BACKGROUND.npz', allow_pickle=False) as z:
            release = z['cone_release'].copy()
        bc = LocalBC(self.model, seed, release)
        self.ac = SimpleNamespace(bc=bc, assembly=a)
        self.origin = bc.origin
        self.fields = self._fields()
        self.types = {p: dict(groups=a.populations[p].group_count,
            group_ids=list(map(str, a.populations[p].group_ids)),
            represented_members=a.populations[p].counts.tolist(),
            total_represented_members=int(a.populations[p].counts.sum())) for p in self.populations}
        self.identity = canonical_hash(dict(schema='explicit-cross-position-RGC-functional-binding-1',
            initialized_model=source['initialized_model_id'], source_inner=self.model.identity,
            fields=self.fields, types=self.types, shared_readout_sha256=digest(ROOT / 'retina_sim/rgc_functional_v1.py'),
            source_seed_unchanged=True, optical_background=digest(Path(background) / 'BACKGROUND.npz'),
            implementation=digest(Path(__file__)), FIT_READY=False, background_qualified=False, production_adopted=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--background', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', choices=['standard', 'tight'], required=True)
    parser.add_argument('--milliseconds', type=int, choices=[6, 900], required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(exist_ok=False)
    start = time.perf_counter(); memory = MemoryGuard(24 * 1024**3, 8 * 1024**3)
    status = 'FAILED'; nfev = 0; index = -1
    def guard():
        if time.perf_counter() - start > 820: raise TimeoutError('Coupled worker time protection')
        if not memory.check(): raise MemoryError(str(memory.stop_reason))
    try:
        guard()
        layer = LocalRGC(args.prepared, args.background)
        guard()
        model, bc = layer.model, layer.ac.bc
        state, auxiliary = layer.initial(), layer.initial_spike_states()
        saved = {f: [] for f in layer.fields}; law = {p: [] for p in layer.populations}
        rates = {p: [] for p in layer.populations}; counts = {p: [] for p in layer.populations}
        events = {p: [] for p in layer.populations}
        rtol, atol, step = ((1e-10, 1e-13, .000025) if args.precision == 'tight'
                            else (1e-6, 1e-9, .0001))
        write(out / 'IDENTITY.json', dict(RGC_identity=layer.identity, source_inner=model.identity,
            source_initialized_model=bc.seed['joint_background_initialization']['initialized_model_id'],
            fields=layer.fields, types=layer.types, full_inner_nstate=model.nstate,
            original_coupling=True, members_applied_once=True, no_new_equations=True,
            native_Cone_HC_held_local_background=True, complete_global_checkpoint=False,
            state_fields={'__'.join((b.population, b.mechanism, b.name)): dict(unit=b.unit, count=b.count)
                          for b in model.blocks},
            precision=args.precision, duration_ms=args.milliseconds, rtol=rtol, atol=atol, max_step=step))
        def capture(i, cone):
            values = layer.observe(bc.origin + i * .0001, state, cone)
            for f, value in values.items(): saved[f].append(value)
            for p in layer.populations:
                law[p].append(np.stack([state[p, 'RGC_expected_history', 'history_probability_' + str(j)] for j in range(8)], axis=1))
        def packet():
            result = dict(time_seconds=np.arange(len(next(iter(saved.values())))) * .001,
                          **{f: np.asarray(v).T for f, v in saved.items()})
            for p in layer.populations:
                result[p + '__history_law_after_tick'] = np.asarray(law[p]).transpose(1, 0, 2)
                result[p + '__expected_bin_count_rate'] = np.asarray(rates[p]).T
                result[p + '__sampled_bin_counts'] = np.asarray(counts[p]).T
                result[p + '__event_time_seconds'] = np.asarray([e['time_seconds'] - bc.origin for e in events[p]])
                result[p + '__event_group_index'] = np.asarray([e['cell_index'] for e in events[p]], dtype=np.int64)
            return result
        for index in range(args.milliseconds * 10):
            if index % 100 == 0: guard()
            cone, rod = bc.inputs(index); tick = index % 10 == 0
            if tick: before = layer.observe(bc.origin + index * .0001, state, cone)
            state, following, drawn = layer.commit(index, state, cone, rod, auxiliary if tick else None)
            if tick:
                auxiliary = following
                for p in layer.populations:
                    rates[p].append(before[p + '__expected_binary_count_rate'])
                    counts[p].append(auxiliary[p]['last_bin_counts'])
                    events[p] += drawn[p]
                capture(index, cone)
            left, right = bc.origin + index * .0001, bc.origin + (index + 1) * .0001
            sol = solve_ivp(layer.rhs, (left, right), model.pack(state), method='DOP853',
                rtol=rtol, atol=atol, max_step=step, first_step=min(step, right - left), t_eval=[right])
            if not sol.success: raise ContractError(sol.message)
            state = model.unpack(sol.y[:, -1]); nfev += sol.nfev
            if (index + 1) % 1000 == 0:
                ms = (index + 1) // 10
                np.savez_compressed(out / ('PARTIAL_%04d.npz' % ms), **packet())
                save_bundle(out / ('INNER_STATE_%04d' % ms), kind='validation_evidence',
                    identity=dict(RGC_identity=layer.identity, completed_boundary_seconds=right),
                    payload=dict(inner_flat=model.pack(state), spike_states=deepcopy(auxiliary), next_event_index=index + 1),
                    qualification=dict(default_adopted=False, fit_eligible=False,
                        scope='Inner held-native experiment evidence only; not complete global checkpoint'))
                print(dict(precision=args.precision, completed_ms=ms, wall_seconds=time.perf_counter() - start), flush=True)
        index = args.milliseconds * 10
        cone, rod = bc.inputs(index); state = bc.endpoint(index, state, cone, rod)
        capture(index, cone)
        np.savez_compressed(out / 'RESPONSE.npz', **packet())
        np.savez_compressed(out / 'STATE_END.npz', inner_flat=model.pack(state),
                            **{'__'.join(k): value for k, value in state.items()})
        status = 'PASS'
    except BaseException as exc:
        import traceback
        write(out / 'FAILURE.json', dict(error=str(exc), traceback=traceback.format_exc(), stopped_index=index))
        raise
    finally:
        write(out / 'RESOURCE.json', dict(status=status, precision=args.precision, duration_ms=args.milliseconds,
            last_event_index=index, nfev=nfev, wall_seconds=time.perf_counter() - start,
            memory=memory.last, memory_stop_reason=memory.stop_reason, automatic_retry=False,
            FIT_READY=False, background_qualified=False, production_adopted=False))


if __name__ == '__main__': main()
