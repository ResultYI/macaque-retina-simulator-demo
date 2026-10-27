"""Independent interval-QA fixtures; no retina model, device or trajectory."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from retina_sim.continuous_state import AcceptedSeries, DiscreteHold, ExponentialForced, QuinticHermite
from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
from retina_sim.group_native_radau import NativeRadauSegment
from retina_sim.group_scc_dense_qa import compare_dense_interval


STOP = 2.**-14
INNER = ('synthetic_population', 'synthetic_membrane', 'V')


def _budget(key, initial, *, unit='V', meaning='voltage'):
    initial = np.array(initial, dtype=np.float64)
    spec = SimpleNamespace(population=key[0], mechanism=key[1], name=key[2],
                           count=len(initial), unit=unit, meaning=meaning)
    return dict(spec=spec, initial=initial.copy(), scale=np.maximum(abs(initial), 1.),
                budget=np.full(len(initial), 1e-8, dtype=np.float64))


def _inner_model(key=None, count=0):
    keys = () if key is None else (key,)
    return SimpleNamespace(state_keys=keys, slices={} if key is None else {key:slice(0, count)},
        sizes={} if key is None else {key:count}, nstate=count,
        parameters={'populations':{}}, p={})


def _arm(inner, *, native_slices=None, native_intervals=()):
    return SimpleNamespace(inner=inner, native_physics=SimpleNamespace(slices=native_slices or {}),
                           native_intervals=list(native_intervals))


def _dop_segment(start, stop, initial, slope, *, native=False):
    y0 = initial+slope*start
    coefficients = np.zeros((7, len(initial)), dtype=np.float64)
    coefficients[0] = slope*(stop-start)
    if native:
        return SimpleNamespace(t0=start, t1=stop, y0=y0, coefficients=coefficients,
                               kind='dop853_dense')
    return SimpleNamespace(t_old=start, t=stop, y_old=y0, F=coefficients)


def _quintic(start, stop, initial, slope, *, offset=0., ids=None):
    return QuinticHermite(t0=start, t1=stop,
        state_ids=np.arange(len(initial)) if ids is None else ids,
        y0=initial+slope*start+offset, ym=initial+slope*(start+(stop-start)/2)+offset,
        y1=initial+slope*stop+offset, f0=slope, fm=slope, f1=slope)


def _inner_fixture(*, key=INNER, initial=(.2, .6), slope=(.25, -.5), offset=0.,
                   unit='V', meaning='voltage'):
    initial, slope = np.array(initial, dtype=np.float64), np.array(slope, dtype=np.float64)
    inner = _inner_model(key, len(initial))
    ref_grid = (0., STOP/4, 3*STOP/4, STOP)
    candidate_grid = (0., STOP/2, STOP)
    reference = SimpleNamespace(t0=0., t1=STOP, model=inner,
        dense_segments=tuple(_dop_segment(a,b,initial,slope) for a,b in zip(ref_grid,ref_grid[1:])),
        representations={}, state={key:initial+slope*STOP})
    series = AcceptedSeries([_quintic(a,b,initial,slope,offset=offset)
        for a,b in zip(candidate_grid,candidate_grid[1:])])
    candidate = SimpleNamespace(t0=0., t1=STOP, model=inner, dense_segments=(),
        representations={key:[(np.arange(len(initial)),series)]},
        state={key:initial+slope*STOP+offset})
    return _arm(inner), _arm(inner), reference, candidate, {key:_budget(key,initial,unit=unit,meaning=meaning)}


def _compare(fixture, *, start=0., stop=STOP):
    ref_arm,cand_arm,ref_advance,cand_advance,budgets = fixture
    result = compare_dense_interval(ref_arm,cand_arm,ref_advance,cand_advance,start,stop,budgets)
    json.dumps(result, allow_nan=False)
    assert isinstance(result['issues'], list)
    assert isinstance(result['complete_coverage'], bool)
    assert isinstance(result['counters'], dict)
    return result


@pytest.mark.parametrize('slope', [(0.,0.), (.25,-.5)])
def test_same_complete_waveform_passes_with_different_accepted_partitions(slope):
    fixture = _inner_fixture(slope=slope)
    result = _compare(fixture)
    assert result['status'] == 'PASS', result
    assert result['complete_coverage'] is True
    assert result['issues'] == []


def test_uniform_offset_above_frozen_budget_is_numerical_fail():
    fixture = _inner_fixture(offset=.001)
    original_budget = fixture[-1][INNER]['budget'].copy()
    result = _compare(fixture)
    assert result['status'] == 'NUMERICAL_FAIL', result
    assert result['issues']
    np.testing.assert_array_equal(fixture[-1][INNER]['budget'], original_budget)


@pytest.mark.parametrize('missing', ['dense', 'key', 'ids'])
def test_missing_or_wrong_coordinate_coverage_is_inconclusive(missing):
    fixture = _inner_fixture()
    if missing == 'dense':
        fixture[2].dense_segments = ()
    elif missing == 'key':
        fixture[3].representations = {}
    else:
        old = fixture[3].representations[INNER][0][1]
        fixture[3].representations[INNER] = [(np.arange(2), AcceptedSeries([
            QuinticHermite(t0=segment.t0,t1=segment.t1,state_ids=np.array([10,11]),
                **{name:getattr(segment,name) for name in ('y0','f0','ym','fm','y1','f1')})
            for segment in old.representations]))]
    result = _compare(fixture)
    assert result['status'] == 'INCONCLUSIVE', result
    assert result['complete_coverage'] is False
    assert result['issues']


def test_unit_interval_domain_failure_is_not_hidden_by_equal_arms():
    fixture = _inner_fixture(initial=(1.2,1.3),slope=(0.,0.),
                             unit='dimensionless',meaning='release_probability')
    result = _compare(fixture)
    assert result['status'] == 'DOMAIN_FAIL', result
    assert result['issues']


def test_missing_frozen_budget_is_inconclusive_without_inventing_a_tolerance():
    fixture = _inner_fixture()
    fixture[-1][INNER]['budget'] = None
    result = _compare(fixture)
    assert result['status'] == 'INCONCLUSIVE', result
    assert fixture[-1][INNER]['budget'] is None


def _native_fixture(initial):
    key = ('cones','reduced_native','X')
    initial = np.array(initial,dtype=np.float64)
    slope = np.array([.25,-.5],dtype=np.float64)
    inner = _inner_model()
    reference_curve = SimpleNamespace(segments=(
        _dop_segment(0.,STOP/2,initial,slope,native=True),
        _dop_segment(STOP/2,STOP,initial,slope,native=True)))
    re_arrays = dict(R=np.array([.4,.6]), E=np.array([.3,.7]), light=np.array([.2,.8]))
    reference_anchor = AnalyticREAnchor(time_seconds=0.,input_stop_seconds=STOP,**re_arrays)
    candidate_anchor = AnalyticREAnchor(time_seconds=0.,input_stop_seconds=STOP,**re_arrays)
    reference_interval = SimpleNamespace(start=0.,stop=STOP,curves={'X':reference_curve},anchor=reference_anchor)
    segment = SimpleNamespace(start=0.,stop=STOP,y0=initial.copy(),f0=slope.copy(),
        ym=initial+slope*STOP/2,fm=slope.copy(),y1=initial+slope*STOP,f1=slope.copy())
    candidate_interval = SimpleNamespace(start=0.,stop=STOP,dense=SimpleNamespace(segments=(segment,)),anchor=candidate_anchor)
    native_slices = {'X':slice(0,len(initial))}
    reference_arm = _arm(inner,native_slices=native_slices,native_intervals=(reference_interval,))
    candidate_arm = _arm(inner,native_slices=native_slices,native_intervals=(candidate_interval,))
    empty_advance = SimpleNamespace(t0=0.,t1=STOP,state={},representations={},dense_segments=())
    budgets = {key:_budget(key,initial,unit='vH_native_precursor',meaning='native_precursor')}
    for name in ('R','E'):
        analytic_key = ('cones','analytic_RE',name)
        budgets[analytic_key] = _budget(analytic_key,re_arrays[name],unit='td',meaning='native_illuminance')
    return reference_arm,candidate_arm,empty_advance,empty_advance,budgets


def test_native_packed_quintic_rows_compare_to_per_field_reference_curves():
    result = _compare(_native_fixture((.2,.6)))
    assert result['status'] == 'PASS', result
    assert result['complete_coverage'] is True


def test_negative_native_precursor_is_domain_fail_even_for_matching_arms():
    result = _compare(_native_fixture((-.2,-.6)))
    assert result['status'] == 'DOMAIN_FAIL', result
    assert result['issues']


def test_cpu_exponential_evaluation_without_supported_arithmetic_bound_is_inconclusive():
    key = ('rods','rod_membrane','V')
    fixture = _inner_fixture(key=key,initial=(-.05,-.04),slope=(.001,.001))
    initial = fixture[-1][key]['initial']
    decay = np.array([10.,10.])
    representation = ExponentialForced.from_constant_forcing(t0=0.,t1=STOP,
        state_ids=np.arange(2),y0=initial,forcing=decay*initial+.001,decay=decay)
    fixture[3].representations[key] = [(np.arange(2),AcceptedSeries([representation]))]
    result = _compare(fixture)
    assert result['status'] == 'INCONCLUSIVE', result
    assert result['issues']


def test_held_packet_compares_exactly_to_zero_slope_reference():
    fixture = _inner_fixture(slope=(0.,0.))
    initial = fixture[-1][INNER]['initial']
    held = DiscreteHold(t0=0.,t1=STOP,state_ids=np.arange(2),y0=initial)
    fixture[3].representations[INNER] = [(np.arange(2),AcceptedSeries([held]))]
    result = _compare(fixture)
    assert result['status'] == 'PASS', result
    assert result['complete_coverage'] is True


def test_independent_qa_does_not_change_either_dense_owner_or_frozen_rows():
    fixture = _inner_fixture()
    reference = fixture[2].dense_segments
    candidate = fixture[3].representations[INNER][0][1].representations
    arrays = [array for segment in reference for array in (segment.y_old,segment.F)]
    arrays += [getattr(segment,name) for segment in candidate
               for name in ('y0','f0','ym','fm','y1','f1','state_ids')]
    arrays += [fixture[-1][INNER][name] for name in ('initial','scale','budget')]
    snapshots = [array.copy() for array in arrays]
    result = _compare(fixture)
    assert result['status'] == 'PASS', result
    for original, frozen in zip(arrays,snapshots):
        np.testing.assert_array_equal(original,frozen)


def test_analytic_row_count_must_match_both_original_native_coordinates():
    fixture = _native_fixture((.2,.6))
    key = ('cones','analytic_RE','E')
    fixture[-1][key] = _budget(key,[.3,.7,.9],unit='td',meaning='native_illuminance')
    result = _compare(fixture)
    assert result['status'] == 'INCONCLUSIVE', result
    assert result['complete_coverage'] is False


def test_empty_budget_row_cannot_omit_nonempty_original_state_coordinates():
    fixture = _inner_fixture()
    fixture[-1][INNER] = _budget(INNER,[])
    result = _compare(fixture)
    assert result['status'] == 'INCONCLUSIVE', result
    assert result['complete_coverage'] is False


def test_interior_difference_is_rejected_when_boundary_values_agree():
    fixture = _inner_fixture(initial=(0.,0.),slope=(0.,0.))
    bump = QuinticHermite(t0=0.,t1=STOP,state_ids=np.arange(2),
        y0=np.zeros(2),ym=np.full(2,.001),y1=np.zeros(2),
        f0=np.zeros(2),fm=np.zeros(2),f1=np.zeros(2))
    fixture[3].representations[INNER] = [(np.arange(2),AcceptedSeries([bump]))]
    result = _compare(fixture)
    assert result['status'] == 'NUMERICAL_FAIL', result
    assert result['issues']


def test_native_rounded_midpoint_override_cannot_pass_polynomial_only_qa():
    fixture = _native_fixture((2.,2.))
    start,stop = 1.,1.+3*np.spacing(1.)
    re_arrays = dict(R=np.array([.4,.6]),E=np.array([.3,.7]),light=np.array([.2,.8]))
    reference_anchor = AnalyticREAnchor(time_seconds=start,input_stop_seconds=2.,**re_arrays)
    candidate_anchor = AnalyticREAnchor(time_seconds=start,input_stop_seconds=2.,**re_arrays)
    coefficients = np.zeros((7,2)); coefficients[3] = 16.
    reference_segment = SimpleNamespace(t0=start,t1=stop,y0=np.full(2,2.),
        coefficients=coefficients,kind='dop853_dense')
    candidate_segment = NativeRadauSegment(start=start,stop=stop,y0=np.full(2,2.),
        ym=np.full(2,3.),y1=np.full(2,2.),f0=np.zeros(2),fm=np.zeros(2),f1=np.zeros(2),
        anchor=candidate_anchor,physics=None)
    midpoint = start+(stop-start)/2
    np.testing.assert_array_equal(candidate_segment.at(midpoint),[3.,3.])
    assert (midpoint-start)/(stop-start) == pytest.approx(2/3)
    fixture[0].native_intervals = [SimpleNamespace(start=start,stop=stop,anchor=reference_anchor,
        curves={'X':SimpleNamespace(segments=(reference_segment,))})]
    fixture[1].native_intervals = [SimpleNamespace(start=start,stop=stop,anchor=candidate_anchor,
        dense=SimpleNamespace(segments=(candidate_segment,)))]
    result = _compare(fixture,start=start,stop=stop)
    assert result['status'] != 'PASS', result
    assert result['elementary_arithmetic_complete'] is False
    assert result['issues']


def test_equal_analytic_anchors_do_not_hide_uncertain_cancellation_domain():
    fixture = _native_fixture((.2,.6))
    values = dict(R=np.zeros(2),E=np.zeros(2),light=np.full(2,1e308))
    for arm in fixture[:2]:
        arm.native_intervals[0].anchor = AnalyticREAnchor(time_seconds=0.,
            input_stop_seconds=STOP,**values)
    for name in ('R','E'):
        key = ('cones','analytic_RE',name)
        fixture[-1][key] = _budget(key,np.zeros(2),unit='td',meaning='native_illuminance')
    result = _compare(fixture,stop=1e-29)
    assert result['status'] == 'INCONCLUSIVE', result
    assert result['issues']
