"""Build and audit eccentric-size Cone--HC operators on the actual fixed tables."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retina_sim.contracts import ContractError
from retina_sim.disk_graph_storage import FrozenSparseRoute, file_sha, save_json
from retina_sim.disk_hc_functional import freeze_functional_hc
from retina_sim.disk_tables import FrozenDiskTable


DIRECTORY = Path('data/atlas/hc_eccentric_size_w2000_v1')
HANDOUT = ROOT/'semi_reports/2026-11-02_HC'
PROFILE = 'configurations/hc_eccentric_size_w2000_v1.json'


def load_matrix(binding):
    route = FrozenSparseRoute(ROOT, binding)
    return sparse.vstack([route.block(block)[0] for block in route.record['blocks']], format='csr')


def degree_summary(values):
    values = np.asarray(values)
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                zero=int(np.count_nonzero(values == 0)), total=int(values.sum()),
                minimum=int(values.min()) if len(values) else None,
                maximum=int(values.max()) if len(values) else None)


def write_size_curves():
    from retina_sim.atlas_hc import AtlasHCProvider
    from retina_sim.contracts import Quantity
    provider = AtlasHCProvider(ROOT)
    for maximum, name in [(1.338, 'hc_size_curve.csv'), (6., 'hc_size_full_curve.csv')]:
        r = np.linspace(0, maximum, 301)
        points = np.c_[np.tile(r, 2), np.zeros(2*len(r))]
        query = provider.footprints(tuple(str(i) for i in range(len(points))),
            ['H1']*len(r)+['H2']*len(r), Quantity(points, 'mm', 'arbor_anchor_position', 'project', 'OPL'),
            mode='connection_preview', enable_central_candidate=True)
        d = np.array([row['radius_mm']*2000 for row in query['per_hc_coverage']]).reshape(2,-1)
        np.savetxt(HANDOUT/name, np.c_[r/.223, r, d.T], delimiter=',',
                   header='eccentricity_deg,eccentricity_mm,H1_diameter_um,H2_diameter_um', comments='')


def write_tex_results(report):
    lines = [r'同じ座標のH1 $82{,}456$個、H2 $20{,}597$個について接触を生成した。',
        r'収束数（Cone/HC）の平均は次の通りである。', r'\begin{center}\small',
        r'\begin{tabular}{c|rr}', r'物理偏心度（mm） & H1 & H2\\ \hline']
    for low, high in [(0., .5), (.5, 1.), (1., 1.338)]:
        pair = [next(row for row in report['bands'] if row['kind']==kind and row['low_mm']==low and row['high_mm']==high) for kind in ('H1','H2')]
        lines.append(f'${low:g}$--${high:g}$ & {pair[0]["mean"]:.2f} & {pair[1]["mean"]:.2f}'+r'\\')
    total = {kind:next(row for row in report['bands'] if row['kind']==kind and
                     row['low_mm']==0 and row['high_mm']==report['radius_mm']) for kind in ('H1','H2')}
    fmt = lambda n:format(n, ',').replace(',', '{,}')
    divergence = report['owner_OPL_divergence']
    lines += [r'\end{tabular}\end{center}',
        '$0$--$6^\\circ$内のHCを受け手として数えた接触は、H1 $'+fmt(total['H1']['total'])+
        '$辺、H2 $'+fmt(total['H2']['total'])+'$辺であった。',
        f'この範囲の接触0のHCはH1 ${total["H1"]["zero"]}$個、H2 ${total["H2"]["zero"]}$個であった。',
        '同じOPL円内のCone $'+fmt(report['owner_OPL_cones'])+'$個について、型別発散数の平均はL/MからH1が$'+
        f'{divergence["H1"]["mean"]:.2f}'+r'$、全ConeからH2が$'+f'{divergence["H2"]["mean"]:.2f}'+r'$であった。',
        '分型フィードバック先を持たないConeは$'+fmt(report['owner_OPL_typed_feedback']['zero'])+'$個として記録した。']
    (HANDOUT/'hc_connectivity_results.tex').write_text('\n'.join(lines)+'\n', encoding='utf8')


def audit_operators(operator, outer):
    cone, hc = (FrozenDiskTable(ROOT, outer[k]) for k in ('cones', 'HC'))
    cxy, hxy = cone.coordinates['opl_pedicle'], hc.coordinates['soma']
    ct, ht = cone.extras['subtype'].astype('U1'), hc.extras['subtype'].astype('U2')
    cr, hr = np.linalg.norm(cxy, axis=1), np.linalg.norm(hxy, axis=1)
    rf_radius = np.linalg.norm(cone.coordinates['rf'], axis=1)
    radii_binding = operator['operator_audit']['hc_radii']
    radii = np.load(ROOT/radii_binding['path'], allow_pickle=False)
    if file_sha(ROOT/radii_binding['path']) != radii_binding['sha256']:
        raise ContractError('HC radii identity changed')
    raw, A, B = (load_matrix(operator[k]) for k in ('raw_contacts', 'native_A', 'native_B'))
    raw = raw.tocsr()  # HC rows, cone columns throughout this audit.
    convergence = np.diff(raw.indptr)
    source_rows, cone_rows = raw.nonzero()
    distances = np.linalg.norm(hxy[source_rows] - cxy[cone_rows], axis=1)
    if np.any(distances > radii[source_rows] + 1e-12):
        raise ContractError('Actual contact outside its own HC size')
    if np.any((ht[source_rows] == 'H1') & (ct[cone_rows] == 'S')):
        raise ContractError('H1 unexpectedly contacts S')
    if np.any(raw.data != 1) or ((A.T != 0) != (B != 0)).nnz:
        raise ContractError('Duplicate/weighted raw contact or nonreciprocal native topology')
    for matrix in (A, B):
        sums = np.asarray(matrix.sum(1)).ravel()
        if not np.isfinite(matrix.data).all() or not np.allclose(sums[sums > 0], 1., atol=2e-15, rtol=0):
            raise ContractError('Native pooling lost finite unit-DC normalization')
    # Independent typed-tree counts plus the distance check certify every owner HC row.
    radius_mm = 6*.223
    owner_hc = hr <= radius_mm
    from retina_sim.henle_bridge import ScheinBoundedBridge
    from retina_sim.disk_source_geometry import schein_array
    bridge = ScheinBoundedBridge(ROOT, rf_transfer='rf_equals_inner_segment_mm_v1')
    bound = cone.record['source_bindings']['Schein']
    if (bridge.provenance['bridge_configuration_sha256'] != bound['bridge_configuration_sha256']
            or bridge.provenance['implementation_sha256'] != bound['implementation_sha256']
            or not np.array_equal(schein_array(bridge, cone.coordinates['rf']), cxy)):
        raise ContractError('Actual cone pedicles differ from the frozen OPL displacement')
    rf_support = cone.record['source_bindings']['fixed_RF_query_radius_mm']
    opl_support = min(float(np.interp(rf_support, *nodes)) for nodes in bridge.nodes.values())
    if np.any(hr[owner_hc]+radii[owner_hc] > opl_support):
        raise ContractError('Owner HC footprint crosses the complete cone-pedicle cohort')
    for kind in ('H1', 'H2'):
        rows = np.flatnonzero(owner_hc & (ht == kind))
        eligible = ct != 'S' if kind == 'H1' else np.ones(cone.count, bool)
        counts = cKDTree(cxy[eligible]).query_ball_point(hxy[rows], radii[rows]+1e-12,
                                                       return_length=True, workers=1)
        if not np.array_equal(counts, convergence[rows]):
            raise ContractError('Missing eligible contacts in complete owner HC disks')
    divergence = {kind:np.asarray(raw[ht == kind].sum(0)).ravel() for kind in ('H1', 'H2')}
    typed_degree = np.where(ct == 'S', divergence['H2'], divergence['H1'])
    bands = []
    for low, high in [(0., .5), (.5, 1.), (1., radius_mm), (0., radius_mm)]:
        for kind in ('H1', 'H2'):
            mask = (ht == kind) & (hr >= low) & (hr < high if high != radius_mm else hr <= high)
            bands.append(dict(kind=kind, low_mm=low, high_mm=high,
                              diameter_um_mean=float((2e3*radii[mask]).mean()),
                              **degree_summary(convergence[mask])))
    annuli = []
    for low in np.arange(0., 6., .5):
        high = low+.5
        for kind in ('H1', 'H2'):
            hm = (ht == kind) & (hr >= low*.223) & (hr < high*.223)
            eligible = ct != 'S' if kind == 'H1' else np.ones(cone.count, bool)
            cm = eligible & (cr >= low*.223) & (cr < high*.223)
            cstat, dstat = degree_summary(convergence[hm]), degree_summary(divergence[kind][cm])
            annuli.append(dict(kind=kind, low_deg=float(low), high_deg=float(high),
                HC_count=cstat['n'], mean_convergence=cstat['mean'], zero_HC=cstat['zero'],
                eligible_cones=dstat['n'], mean_divergence=dstat['mean'], zero_cones=dstat['zero'],
                mean_diameter_um=float((2e3*radii[hm]).mean())))
    owner_cones = cr <= radius_mm
    report = dict(schema='HC_eccentric_connectivity_validation_v1',
        implementation={name:file_sha(ROOT/name) for name in
            ('tools/build_eccentric_hc.py','retina_sim/disk_hc_functional.py','retina_sim/atlas_hc.py')},
        tables=outer, connection_profile=operator['profile'], size_profile=radii_binding['size_profile'],
        radius_deg=6, radius_mm=radius_mm, mm_per_degree=.223,
        coordinate_scope='HC soma co-located OPL anchors and actual cone pedicles; RF counts reported separately',
        cone_geometry='existing human-borrowed fixed cone mosaic with Schein OPL displacement; unchanged',
        total_HC=hc.count, total_cones=cone.count, total_edges=raw.nnz,
        owner_HC_counts={k:int(np.sum(owner_hc & (ht == k))) for k in ('H1', 'H2')},
        owner_OPL_cones=int(owner_cones.sum()), owner_RF_cones=int((rf_radius <= radius_mm).sum()),
        checks=dict(all_edges_inside_own_HC_radius=True,
            maximum_distance_over_radius=float(np.max(distances/radii[source_rows])),
            owner_HC_complete_against_independent_typed_trees=True,
            owner_HC_disks_within_complete_cone_outer_support=True,
            cone_pedicles_match_frozen_displacement=True,
            H1_S_edges=0, reciprocal_native_topology=True, finite_unit_DC_nonempty_rows=True,
            owner_cone_HC_cohort_complete=bool(radius_mm+radii.max() <= hc.record['source_bindings']['fixed_cohort_OPL_radius_mm']),
            single_HC_contact_quota_imposed=False, cone_divergence_quota_imposed=False,
            radius_expansion_or_nearest_fill=False),
        cone_support=dict(complete_RF_radius_mm=rf_support, conservative_OPL_outer_radius_mm=opl_support,
            maximum_owner_HC_disk_extent_mm=float(np.max(hr[owner_hc]+radii[owner_hc])),
            central_pedicle_free_radius_mm=bridge.limit,
            central_interpretation='empty pedicle region of the existing project displacement bridge; not a new physiological zero claim'),
        bands=bands, annuli=annuli,
        owner_OPL_divergence={k:degree_summary(v[owner_cones & (ct != 'S' if k == 'H1' else True)]) for k,v in divergence.items()},
        owner_OPL_typed_feedback=degree_summary(typed_degree[owner_cones]),
        owner_RF_typed_feedback=degree_summary(typed_degree[rf_radius <= radius_mm]),
        literature_comparison_central=dict(band_mm=[.5,1.], H1_reported=[6,7], H2_reported_about=15,
            interpretation='descriptive comparison on existing composite geometry; source intervals and model assumptions retained'),
        structural_checks_passed=True, physiological_validation=False,
        full_video_dynamics_rerun=False, previous_video_graph_changed=False)
    save_json(HANDOUT/'hc_connectivity_validation.json', report)
    with (HANDOUT/'hc_connectivity_annuli.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(annuli[0]))
        writer.writeheader(); writer.writerows(annuli)
    write_size_curves()
    write_tex_results(report)
    return report


def main():
    # The frozen route recipe binds exactly the tables used by the prior model.
    previous = json.loads((ROOT/'data/atlas/human_disk3mm_outer_connections_v1/native_A/manifest.json').read_bytes())['recipe']
    outer = {'cones': previous['pre_table'], 'HC': previous['post_table']}
    config = json.loads((ROOT/'configurations/atlas_human_borrowed_disk3mm_functional_hc_v1.json').read_bytes())
    reference = config['native_operator_representation']['anatomical_reference']
    operator = freeze_functional_hc(ROOT, outer, reference, DIRECTORY, profile_path=PROFILE)
    report = audit_operators(operator, outer)
    model = dict(schema='Cone_HC_eccentric_connection_model_v1', scope='actual finite Cone-HC operators',
        quantities={'HC_contact_footprint':'direct cone-pedicle support from the existing AtlasHCProvider',
                    'HC_functional_RF':'emergent direct input and independent homotypic dynamics; no separate RF diameter parameter'},
        connection_profile=operator['profile'], tables=outer,
        native_A=operator['native_A'], native_B=operator['native_B'], raw_contacts=operator['raw_contacts'],
        hc_radii=operator['operator_audit']['hc_radii'], validation={
            'path':(HANDOUT/'hc_connectivity_validation.json').relative_to(ROOT).as_posix(),
            'sha256':file_sha(HANDOUT/'hc_connectivity_validation.json')},
        zero_row_policy='retain zero; downstream unit-DC-required runtime must check coverage before simulation',
        structural_checks_passed=True, physiological_validation=False)
    response_path = HANDOUT/'hc_response_validation.json'
    if response_path.exists():
        response = json.loads(response_path.read_bytes())
        if response['new_connection_profile'] == operator['profile']:
            model['native_response_validation'] = {'path':response_path.relative_to(ROOT).as_posix(),
                                                   'sha256':file_sha(response_path)}
    save_json(ROOT/'configurations/atlas_hc_eccentric_size_w2000_v1.json', model)
    print(json.dumps(dict(edges=report['total_edges'], owner_HC=report['owner_HC_counts'],
        bands=report['bands'], typed_feedback=report['owner_OPL_typed_feedback'], checks=report['checks']), indent=2), flush=True)


if __name__ == '__main__':
    main()
