"""Release unreachable SciPy solver cycles after a completed DOP853 slice.

No solver object is modified and no reachable model/state is reclaimed. Choose
the smallest generation covering collections that ran while the solver was live;
this avoids an unconditional whole-heap collection at every short slice.
"""
import gc
from time import perf_counter,process_time


def collection_epoch():
    return tuple(row['collections'] for row in gc.get_stats())


def release_solver_cycles(before):
    after=collection_epoch()
    # A generation0 collection can promote the live solver to generation1;
    # generation1/2 collection can promote it to the oldest generation.
    generation=2 if after[1:]!=before[1:] else 1 if after[0]!=before[0] else 0
    started=perf_counter();cpu=process_time()
    collected=gc.collect(generation)
    return dict(generation=generation,collected_objects=collected,
        wall_seconds=perf_counter()-started,CPU_seconds=process_time()-cpu)
