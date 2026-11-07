"""Stream the approved fine H1 kernel into existing parent groups, preserving B."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import ctypes
from scipy import sparse
from scipy.spatial import cKDTree

ROOT = Path(r'D:\macaque retinal simulator')
sys.path.insert(0, str(ROOT))
from retina_sim.hc_biexponential import support_radius
from retina_sim.h1_h2_effective import matrix_id
from tools.run_hc_minimal_real_diagnostic_v1 import extend_labels, peak_memory_bytes


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)]+[
        (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page', 'available_page',
                                              'virtual', 'available_virtual', 'extended')]


def guard():
    status = MemoryStatus(); status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('Memory reserve query failed')
    if peak_memory_bytes() > 24*1024**3 or status.available < 8*1024**3:
        raise MemoryError('Original 24 GiB process / 8 GiB reserve guard')


def main():
    began = time.perf_counter()
    out = ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/joint_functional_assets'
    out.mkdir(exist_ok=False)
    source = ROOT/'reports/CONE_HC_OUTER_SOURCE_EXTENSION_20261102_v3/cones_01'
    native = ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/executions/assemble_native_01'
    record = json.loads((source/'RESULT.json').read_bytes())
    for name in ('cone_pedicle_mm.npy', 'official_source_rows.npy'):
        with (source/name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != record['files'][name]:
                raise ValueError('Source identity changed: '+name)
    xy = np.load(source/'cone_pedicle_mm.npy', mmap_mode='r', allow_pickle=False)
    official = np.load(source/'official_source_rows.npy', mmap_mode='r', allow_pickle=False)
    member = np.load(native/'MEMBERSHIP.npz', allow_pickle=False)
    receipt = json.loads((native/'NATIVE_ASSEMBLY_RESULT.json').read_bytes())
    hm = json.loads((ROOT/receipt['hc_table']['path']).read_bytes())
    hx = np.load(ROOT/hm['coordinates']['soma']['path'], allow_pickle=False)
    ht = np.load(ROOT/hm['extras']['subtype']['path'], allow_pickle=False).astype('U2')
    radius = support_radius(.300, .0001)
    center = hx.mean(axis=0)
    envelope = radius + np.linalg.norm(hx-center, axis=1).max()
    if np.linalg.norm(center)+envelope > record['IS_radius_mm']:
        raise ValueError('Known source extent cannot close finite kernel support')
    tree = cKDTree(xy)
    needed = np.asarray(tree.query_ball_point(center, envelope, return_sorted=True), int)
    labels = np.full(len(xy), '', dtype='U1')
    prior = np.load(ROOT/'reports/CONE_HC_MINIMAL_REAL_DIAGNOSTIC_20261102_v6/run_02/SOURCE_LABELS.npz', allow_pickle=False)
    labels[prior['cone_candidate_rows']] = prior['cone_types']
    cm = json.loads((ROOT/receipt['cone_table']['path']).read_bytes())
    keys = np.load(ROOT/cm['source_keys']['path'], allow_pickle=False).astype('U')
    fine_official = np.array([int(k.split(':')[-1]) for k in keys])
    mapped = np.searchsorted(official, fine_official)
    np.testing.assert_array_equal(official[mapped], fine_official)
    original_labels = np.load(ROOT/cm['extras']['subtype']['path'], allow_pickle=False).astype('U1')
    overlap = labels[mapped] != ''
    np.testing.assert_array_equal(labels[mapped][overlap], original_labels[overlap])
    labels[mapped] = original_labels
    unknown = needed[labels[needed] == '']
    if len(unknown):
        # Reuse the approved immutable-macro assignment, never invent a subtype.
        isxy = np.load(source/'cone_inner_segment_mm.npy', mmap_mode='r', allow_pickle=False)
        audit = extend_labels(ROOT, isxy, official, unknown, labels, record)
    else:
        audit = {'new_labels': 0}
    np.savez_compressed(out/'SOURCE_LABELS.npz', cone_candidate_rows=needed, cone_types=labels[needed])
    fine_to_parent = np.full(len(xy), -1, dtype=np.int64)
    fine_to_parent[mapped] = member['global_cone_fine_to_group']
    nh, nc = len(member['counts']), len(member['cone_counts'])
    final = np.zeros((nh, nc), dtype=np.float64)
    held = np.zeros(nh)
    uniform = sparse.load_npz(native/'A.npz').tocsr()
    hmap = member['fine_to_group']
    for row in np.flatnonzero(ht == 'H1'):
        guard()
        ids = np.asarray(tree.query_ball_point(hx[row], radius, return_sorted=True), int)
        if np.any(labels[ids] == ''):
            raise ValueError('Unlabelled functional source')
        ids = ids[labels[ids] != 'S']
        distance = np.linalg.norm(xy[ids]-hx[row], axis=1)
        short = np.exp(-(distance-distance.min())/.020); short /= short.sum()
        long = np.exp(-(distance-distance.min())/.300); long /= long.sum()
        weights = .15*short+.85*long
        group = int(hmap[row]); denom = member['counts'][group]
        target = fine_to_parent[ids]; valid = target >= 0
        final[group] += np.bincount(target[valid], weights=weights[valid], minlength=nc)/denom
        held[group] += weights[~valid].sum()/denom
        if row % 100 == 0:
            print(json.dumps({'fine_HC_row': int(row), 'elapsed_seconds': time.perf_counter()-began}), flush=True)
    h2 = member['HC_types'].astype('U2') == 'H2'
    final[h2] = uniform[h2].toarray()
    np.testing.assert_allclose(final.sum(axis=1)[~h2]+held[~h2], 1., rtol=0, atol=2e-14)
    A = sparse.csr_matrix(final)
    sparse.save_npz(out/'A_parent.npz', A)
    np.save(out/'A_exterior_mass.npy', held, allow_pickle=False)
    result = dict(status='SPATIAL_OPERATOR_PREPARED_BACKGROUND_UNBOUND',
        A_identity=matrix_id(A), B_unchanged_identity=matrix_id(sparse.load_npz(native/'typed_B.npz')),
        support_radius_mm=radius, fine_HC_count=len(hx), parent_shape=list(A.shape),
        H1_parameters=dict(lambda_s_mm=.020, lambda_l_mm=.300, w_s=.15),
        anatomical_contacts_changed=False, crop_renormalized=False,
        exterior_mass_range=[float(held[~h2].min()), float(held[~h2].max())],
        labels=audit, wall_seconds=time.perf_counter()-began,
        background_products_not_assumed=True, production_adopted=False)
    (out/'RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
