"""Source RHS and member-unit checks at the actual new local operating point."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_cross_position_rgc_v1 import LocalRGC
from tools.cross_position_context_v1 import write
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.group_time_monitor import frozen_budget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--background', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(exist_ok=False)
    start = time.perf_counter(); memory = MemoryGuard(24 * 1024**3, 8 * 1024**3)
    if not memory.check(): raise MemoryError(str(memory.stop_reason))
    status = 'FAILED'
    try:
        layer = LocalRGC(args.prepared, args.background)
        if not memory.check(): raise MemoryError(str(memory.stop_reason))
        model, state = layer.model, layer.initial()
        ports = model.outputs(layer.origin, state, layer.ac.bc.background)
        observed = layer.observe_ports(ports)
        rhs = model.rhs(layer.origin, state, layer.ac.bc.background)
        rows = []
        for pop in layer.populations:
            key = (pop, 'RGC_spike_history', 'V')
            budget = frozen_budget(state[key], 'V')[2] / .0001
            error = abs(rhs[key] - observed[pop + '__voltage_derivative'])
            C_ok = np.array_equal(model.C[pop], model.members[pop] * model.par(pop, 'C'))
            gL_ok = np.array_equal(model.gL[pop], model.members[pop] * model.par(pop, 'gL'))
            instantaneous = -observed[pop + '__inhibitory_current'] / model.C[pop]
            row = dict(type=pop, groups=model.counts[pop],
                grouped_C_applied_once=bool(C_ok), grouped_gL_applied_once=bool(gL_ok),
                maximum_source_RHS_readout_error_V_per_s=float(error.max()),
                maximum_RHS_error_over_frozen_budget=float(np.max(error / budget)),
                inhibitory_current_negative=bool(np.all(observed[pop + '__inhibitory_current'] < 0)),
                predicted_noI_instantaneous_delta_dVdt_min=float(instantaneous.min()),
                predicted_noI_instantaneous_delta_dVdt_max=float(instantaneous.max()),
                inhibitory_removal_trajectory_run=False, scientific_parameters_changed=False)
            row['passed'] = bool(C_ok and gL_ok and np.all(error <= budget) and np.all(instantaneous > 0))
            rows.append(row)
        write(out / 'SOURCE_PORT_AND_UNIT_CHECK.json', dict(rows=rows, passed=all(r['passed'] for r in rows),
            scope='Actual source RHS and physical-current decomposition at independently initialized point; no extra ablation trajectory',
            complete_global_checkpoint=False, FIT_READY=False, background_qualified=False, production_adopted=False))
        if not all(r['passed'] for r in rows): raise ValueError('Actual source current/member-unit check failed')
        status = 'PASS'
    finally:
        write(out / 'RESOURCE.json', dict(status=status, wall_seconds=time.perf_counter() - start,
            memory=memory.last, memory_stop_reason=memory.stop_reason, neural_forward_seconds=0.))


if __name__ == '__main__': main()
