"""Cache immutable frame boundaries; preserve original exact float side rules."""
from bisect import bisect_right
import math
from retina_sim.contracts import ContractError
from tools.cell6_compiled_kernel_guard_v2 import GuardedCompiledCell6Runner


class IndexedGIN69Runner(GuardedCompiledCell6Runner):
    def frame_at(self, t, side='right'):
        # Preserve Cell6Runner's closed terminal witness, without adding an input.
        if t == self.stop and side == 'right':
            side = 'left'
        if not math.isfinite(t) or side not in ('left', 'right'):
            raise ContractError('Finite stimulus time and exact side required')
        if not hasattr(self, '_gin_frame_index'):
            frames = tuple(sorted(int(i) for i in self.seed['native_inputs']))
            origin = self.protocol['input_origin_seconds']
            starts = tuple(origin + i / 150 for i in frames)
            stops = tuple(origin + (i + 1) / 150 for i in frames)
            if any(b <= a for a, b in zip(starts, starts[1:])):
                raise ContractError('Frame index requires distinct increasing actual frame starts')
            self._gin_frame_index = frames, starts, stops
        frames, starts, stops = self._gin_frame_index
        position = bisect_right(starts, t) - 1
        candidates = []
        for j in range(max(0, position - 1), min(len(frames), position + 1)):
            if not (starts[j] <= t < stops[j] or side == 'left' and t == stops[j]):
                continue
            if side == 'left':
                if t == starts[j] and not (t == self.start and j == 0):
                    continue
            elif t == stops[j]:
                continue
            candidates.append(frames[j])
        if not candidates:
            raise ContractError('Cached frame does not cover the requested exact stimulus side')
        return min(candidates) if side == 'left' else max(candidates)
