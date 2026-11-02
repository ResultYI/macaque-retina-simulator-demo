"""Conservative zero-energy screening; retain the original pair calculation."""
from types import SimpleNamespace
import time
import numpy as np
from retina_sim import hc_mosaic

ORIGINAL = hc_mosaic._point_energy


def screened_point_energy(index, point, local_lengths, positions, labels, lengths, spatial, amplitudes):
    neighbors = np.array(sorted(spatial.nearby(point)), dtype=np.int64)
    neighbors = neighbors[neighbors != index]
    if not len(neighbors):
        return 0.
    kinds = np.where(labels[neighbors] == labels[index], 0 if labels[index] == 'H1' else 1, 2)
    scale = (local_lengths[kinds]+lengths[neighbors, kinds])/2
    diff = positions[neighbors]-point
    distance2 = np.einsum('ij,ij->i', diff, diff)
    # A conservative margin retains every potentially nonzero original term.
    keep = neighbors[distance2 <= scale*scale*(1+1e-12)]
    proxy = SimpleNamespace(nearby=lambda _: keep)
    return ORIGINAL(index, point, local_lengths, positions, labels, lengths, proxy, amplitudes)


def self_check():
    random = np.random.default_rng(20261101)
    positions = random.uniform(-.2, .2, (5000, 2))
    labels = random.choice(['H1','H2'], len(positions))
    lengths = random.uniform(.003, .07, (len(positions), 3))
    spatial = hc_mosaic._SpatialIndex(positions, .071)
    amplitudes = np.array([4.,4.,4.])
    queries = [(int(i), positions[i], lengths[i]) for i in random.integers(0,len(positions),300)]
    queries += [(int(i),random.uniform(-.2,.2,2),random.uniform(.003,.07,3)) for i in range(100)]
    args = lambda q: (*q, positions, labels, lengths, spatial, amplitudes)
    start = time.perf_counter()
    original = [ORIGINAL(*args(q)) for q in queries]
    original_seconds = time.perf_counter()-start
    start = time.perf_counter()
    screened = [screened_point_energy(*args(q)) for q in queries]
    screened_seconds = time.perf_counter()-start
    assert np.array_equal(original,screened), np.max(abs(np.array(original)-screened))
    positions[1] = positions[0]
    spatial = hc_mosaic._SpatialIndex(positions,.071)
    assert np.isinf(screened_point_energy(0,positions[0],lengths[0],positions,labels,lengths,spatial,amplitudes))
    return dict(queries=len(queries),bitwise_equal=True,coincidence_inf_retained=True,
                original_seconds=original_seconds,screened_seconds=screened_seconds,
                scope='Pair-energy microcheck; not a complete generation timing claim')


if __name__ == '__main__':
    import json
    print(json.dumps(self_check()))
