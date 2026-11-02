"""Compare actual original frame lookup; no solver or scientific state is run."""
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import numpy as np
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.contracts import ContractError
from tools.gin69_frame_index_support_v2 import IndexedGIN69Runner
from tools.durable_research_runtime_v1 import atomic_json


def answer(function, obj, t, side):
    try:
        return function(obj, t, side)
    except ContractError as error:
        return type(error).__name__, str(error)


def main(output):
    began = time.perf_counter()
    comparisons = 0
    for frames, origin in [(list(range(6750)), 0.), ([0, 1, 5, 6, 8], 0.), (list(range(11)), .123456789)]:
        obj = SimpleNamespace(seed={'native_inputs': dict.fromkeys(frames)},
            protocol={'input_origin_seconds': origin}, start=origin, stop=origin + (frames[-1] + 1) / 150)
        selected = frames if len(frames) < 100 else sorted(set(frames[::29] + frames[:3] + frames[-3:]))
        times = []
        for frame in selected:
            for boundary in (origin + frame / 150, origin + (frame + 1) / 150):
                times.extend([np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)])
            times.append(origin + (frame + .413) / 150)
        times.extend([float('nan'), float('inf'), -1., obj.stop + 1.])
        for t in times:
            for side in ('left', 'right', 'invalid'):
                original_side = 'left' if t == obj.stop and side == 'right' else side
                expected = answer(GroupTimeArm.frame_at, obj, t, original_side)
                actual = answer(IndexedGIN69Runner.frame_at, obj, t, side)
                if actual != expected:
                    raise AssertionError((t, side, actual, expected))
                comparisons += 1
        if len(frames) == 6750:
            # Every actual100us authoring node, including closed terminal.
            node_times = np.arange(450001) * .0001
            exact_starts = np.asarray([i / 150 for i in frames])
            expected = np.searchsorted(exact_starts, node_times, side='right') - 1
            expected[-1] = 6749
            actual = np.fromiter((IndexedGIN69Runner.frame_at(obj, t) for t in node_times), dtype=np.int64)
            np.testing.assert_array_equal(actual, expected)
    atomic_json(output / 'RESULT.json', dict(status='PASS', original_function_comparisons=comparisons,
        all_actual100us_nodes=450001, original_float_boundaries_and_nextafter=True,
        gapped_and_nonzero_origin_cases=True, errors_and_terminal_side_preserved=True,
        no_model_run=True, wall_seconds=time.perf_counter() - began))


if __name__ == '__main__':
    main(Path(sys.argv[2]))
