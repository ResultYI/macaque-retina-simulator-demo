"""Read-only acceptance of the real 69#6 result and its existing Tk interface."""
from pathlib import Path
import hashlib
import json
import re
import sys
import tempfile
import time
import traceback
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_platform.internal_state_view import InternalStateApp, CURRENT_RESULT, trace_panel
from retina_platform.__main__ import enable_high_dpi
from retina_platform.plotting import scatter_plot


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_short_window(data, source):
    manifest = read(source / 'internal_states/manifest.json')
    assert manifest['status'] == 'COMPLETE'
    assert data.continuous_window == (5., 5.5)
    assert manifest['model_identity'] == data.identity
    assert manifest['background_identity'] == data.background_identity
    assert digest(source / 'CHECKPOINT_50000/manifest.json') == manifest['checkpoint_manifest_sha256']
    arrays = source / 'internal_states' / manifest['responses_file']
    assert digest(arrays) == manifest['responses_sha256']
    with np.load(arrays, allow_pickle=False) as saved:
        times = saved[manifest['time_array']]
        assert times.shape == (501,) and times[0] == 5. and times[-1] == 5.5
        np.testing.assert_allclose(np.diff(times), .001, rtol=0, atol=1e-12)
        for pop, meta in manifest['populations'].items():
            g = data.geometry[pop]
            assert list(map(str, meta['ids'])) == list(map(str, g['ids']))
            assert meta['members'] == g['members'].tolist()
            assert meta['coordinate_role'] == g['coordinate_role']
        for token, meta in manifest['fields'].items():
            pop, key, rows = meta['population'], meta['key'], np.asarray(meta['rows'])
            values = saved[meta['array']]
            assert values.shape == (len(rows), 501) and np.isfinite(values).all()
            assert data.variables(pop)[key]['unit'] == meta['unit']
            state = data.spatial(pop, key, 5.2)
            index = int(np.searchsorted(times, state['time']))
            assert state['time'] <= 5.2 and 5.2-state['time'] < .001000001
            np.testing.assert_array_equal(state['values'][rows], values[:, index])
            for source_row in sorted({0, len(rows)-1}):
                row = int(rows[source_row])
                series = data.series(pop, row, key, 5., 5.5, max_points=6000)
                if (pop, row) == data.target and key in ('V', 'gE', 'gI', 'expected_rate_Hz', 'glm_rate_Hz'):
                    continue
                np.testing.assert_array_equal(series['time'], times)
                np.testing.assert_array_equal(series['values'], values[source_row])
                assert series['recording'] == 'windowed'
                assert series['continuous_window'] == (5., 5.5)
            if not pop.endswith(('RGC_ON', 'RGC_OFF')) and pop != 'small_bistratified_S_ON':
                for outside in (4.999, 5.501):
                    assert np.isnan(data.spatial(pop, key, outside)['values']).all(), (pop, key, outside)
        for key in ('w', 'z2'):
            state = data.spatial('HC', key, 5.2)
            h2 = np.flatnonzero(data.geometry['HC']['subtypes'] == 'H2')
            assert np.isnan(state['values'][h2]).all()
    return manifest


def check_trace_gap(series):
    lines = []
    original_draw = ImageDraw.Draw

    class TrackingDraw:
        def __init__(self, image, *args, **kwargs):
            self.draw = original_draw(image, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.draw, name)

        def line(self, xy, *args, **kwargs):
            if kwargs.get('fill') == '#137f91':
                lines.append(np.asarray(xy))
            return self.draw.line(xy, *args, **kwargs)

    with patch('retina_platform.internal_state_view.ImageDraw.Draw', TrackingDraw):
        trace_panel((900, 400), [series], 0., 45., 5.2, 'Saved HC states', series['unit'])
    assert lines
    for line in lines:
        assert line.ndim == 2 and line.shape[1] == 2
        assert line[:, 0].min() >= 76 + 5 / 45 * (876-76) - 1e-10
        assert line[:, 0].max() <= 76 + 5.5 / 45 * (876-76) + 1e-10
    lines.clear()
    event = dict(time=np.array([5.2]), values=np.ones(1), recording='events')
    with patch('retina_platform.internal_state_view.ImageDraw.Draw', TrackingDraw):
        trace_panel((900, 400), [event], 5., 5.5, 5.2, 'Single saved spike', 'event')
    assert len(lines) == 1 and lines[0].shape == (4,)
    assert lines[0][0] == lines[0][2] and lines[0][3] - lines[0][1] > 200


def check_english_plot_text(app):
    texts = []
    original_draw = ImageDraw.Draw

    class TrackingText:
        def __init__(self, *args, **kwargs):
            self.draw = original_draw(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.draw, name)

        def text(self, xy, text, *args, **kwargs):
            texts.append(str(text))
            return self.draw.text(xy, text, *args, **kwargs)

    with patch('retina_platform.internal_state_view.ImageDraw.Draw', TrackingText):
        for index in range(1, 5):
            app.figure(index, (1200, 1000))
    assert texts and not any(re.search('[\u4e00-\u9fff]', text) for text in texts)
    assert 'Measured PSTH' in texts and 'Frozen GLM free-history prediction' in texts


def check_controls(app, update):
    app.notebook.select(0)
    app.short_button.invoke()
    update()
    assert app.bounds() == (5., 5.5) and app.clock.position == 5.
    assert app.playback_window == (5., 5.5)
    canvas = app.cards[0][0]
    population, points = app.maps[canvas]
    assert points
    s = app.layer_scale(canvas)
    center = np.array([(64*s+canvas.winfo_width()-107*s)/2, (43*s+canvas.winfo_height()-54*s)/2])
    x, y, _ = min(points, key=lambda p: np.linalg.norm(np.asarray(p[:2])-center))
    other_viewports = {name: (a.copy(), b.copy()) for name, (a, b) in app.viewports.items() if name != population}
    canvas.event_generate('<Enter>')
    canvas.event_generate('<MouseWheel>', x=round(x), y=round(y), delta=120)
    update()
    assert app.hover_canvas is canvas and app.zoom_levels[population] == 1.25
    assert set(app.viewports) == {population, *other_viewports}
    for name, bounds in other_viewports.items():
        np.testing.assert_array_equal(app.viewports[name], bounds)
    _, zoomed_points = app.maps[canvas]
    assert zoomed_points
    x, y, row = min(zoomed_points, key=lambda p: np.linalg.norm(np.asarray(p[:2])-center))
    canvas.event_generate('<Button-1>', x=round(x), y=round(y))
    update()
    assert app.cell == (population, row)
    assert str(app.data.geometry[population]['ids'][row]) in app.detail.get()
    viewport = tuple(a.copy() for a in app.viewports[population])
    app.jump_entry.focus_force()
    update()
    app.jump_entry.event_generate('<KeyPress-r>')
    update()
    np.testing.assert_array_equal(app.viewports[population], viewport)
    app.jump.set(f'{app.clock.position:.5f}')
    canvas.focus_force()
    canvas.event_generate('<Leave>')
    canvas.event_generate('<KeyPress-r>')
    update()
    np.testing.assert_array_equal(app.viewports[population], viewport)
    canvas.event_generate('<Enter>')
    canvas.event_generate('<KeyPress-R>')
    update()
    assert population not in app.viewports and population not in app.zoom_levels
    canvas.event_generate('<MouseWheel>', x=round(x), y=round(y), delta=120)
    update()
    viewport = tuple(a.copy() for a in app.viewports[population])
    selected, target, seconds = app.cell, app.target, app.clock.position
    for code, tab, play in [('ja', '層別状態', '再生'), ('en', 'Layer states', 'Play'), ('zh', '分层状态', '播放')]:
        app.change_language(code)
        update()
        assert app.language_code == code
        assert app.notebook.tab(0, 'text') == tab
        assert app.play_button.cget('text') == play
        assert app.cell == selected and app.target == target and app.clock.position == seconds
        np.testing.assert_array_equal(app.viewports[population], viewport)
        if app.curve_window and app.curve_window.winfo_exists():
            app.curve_window.destroy()
        app.curve_window = None
        app.show_curve()
        update()
        if code == 'en':
            assert not re.search('[\u4e00-\u9fff]', app.stimulus_caption.get())
            labels = [str(w.cget('text')) for w in app.curve_window.winfo_children() if 'text' in w.keys()]
            assert labels and not any(re.search('[\u4e00-\u9fff]', text) for text in labels)
            assert 'Selected object' in app.curve_window.title()
            check_english_plot_text(app)
        with patch('retina_platform.internal_state_view.filedialog.askdirectory', return_value='') as dialog:
            app.choose_directory()
            assert dialog.call_args.kwargs['title'] == app.tr('选择已有研究结果目录')
        with patch('retina_platform.internal_state_view.filedialog.asksaveasfilename', return_value='') as dialog:
            app.export()
            assert dialog.call_args.kwargs['title'] == app.tr('导出当前图…')
    app.show_short_window()
    app.clock.seek(5.49)
    app.toggle_play()
    deadline = time.monotonic()+1.
    while app.clock.playing and time.monotonic() < deadline:
        update(); time.sleep(.02)
    assert not app.clock.playing and app.clock.position == 5.5
    assert app.stimulus_frame['source_frame'] == app.data.stimulus(5.5)['source_frame']
    app.toggle_play()
    assert app.clock.playing and 5. <= app.clock.position < 5.5
    app.toggle_play()
    app.seek(5.2)
    return population


def check_display_and_stimulus(app, update):
    app.notebook.select(0)
    for seconds in (0., 5., 5.25, 5.5, 45.):
        app.seek(seconds)
        update()
        expected = app.data.stimulus(seconds)
        shown = app.stimulus_frame
        assert shown['time'] == expected['time'] <= seconds
        assert shown['source_frame'] == expected['source_frame']
        np.testing.assert_array_equal(shown['image'], expected['image'])
        canvas = app.stimulus_canvas
        assert app.images[canvas].size == (canvas.winfo_width(), canvas.winfo_height())
        assert f"{expected['time']:.5f}" in app.stimulus_caption.get()
    app.seek(5.25)
    update()
    for canvas, _, caption, _ in app.cards:
        assert app.images[canvas].size == (canvas.winfo_width(), canvas.winfo_height())
        assert app.photos[canvas].width() == canvas.winfo_width()
        assert app.photos[canvas].height() == canvas.winfo_height()
        label = next(w for w in canvas.master.winfo_children() if w.winfo_class() == 'TLabel' and str(w.cget('textvariable')) == str(caption))
        assert label.winfo_ismapped() and label.winfo_height() >= label.winfo_reqheight()
    pop = app.cards[0][1].get()
    xy = app.data.geometry[pop]['xy_mm']
    rows = np.asarray(app.network[pop], dtype=int)
    values = app.data.spatial(pop, app.primary(pop), 5.25)['values']
    _, normal = scatter_plot(xy, values, rows, (600, 400))
    _, dense = scatter_plot(xy, values, rows, (900, 600), pixel_scale=1.5)
    np.testing.assert_array_equal(np.asarray(normal)[:, 2], np.asarray(dense)[:, 2])
    np.testing.assert_allclose(np.asarray(normal)[:, :2]*1.5, np.asarray(dense)[:, :2], rtol=0, atol=1e-10)
    geometry = app.root.geometry()
    width, height = app.root.minsize()
    app.root.geometry(f'{width}x{height}')
    update(); app.render(); update()
    for canvas, pop, caption, _ in app.cards:
        assert app.images[canvas].size == (canvas.winfo_width(), canvas.winfo_height())
        assert app.maps[canvas][1], (pop.get(), canvas.winfo_width(), canvas.winfo_height())
        label = next(w for w in canvas.master.winfo_children() if w.winfo_class() == 'TLabel' and str(w.cget('textvariable')) == str(caption))
        assert label.winfo_ismapped() and label.winfo_height() >= label.winfo_reqheight()
    canvas = app.cards[0][0]
    population, points = app.maps[canvas]
    x, y, row = points[len(points)//2]
    app.click_cell(canvas, SimpleNamespace(x=x, y=y))
    assert app.cell == (population, row)
    app.root.geometry(geometry)
    update(); app.render(); update()
    display = dict(tk_dpi=float(app.root.winfo_fpixels('1i')), pixel_scale=app.display_scale,
                   raster_pixels_match_canvas=True, minimum_window_layers_and_hit_test=True)
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.WinDLL('user32')
        user32.GetWindowDpiAwarenessContext.argtypes = [wintypes.HWND]
        user32.GetWindowDpiAwarenessContext.restype = ctypes.c_void_p
        user32.GetAwarenessFromDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.GetAwarenessFromDpiAwarenessContext.restype = ctypes.c_int
        display['windows_awareness'] = user32.GetAwarenessFromDpiAwarenessContext(
            user32.GetWindowDpiAwarenessContext(app.root.winfo_id()))
        assert display['windows_awareness'] == 1
    return display


def check_merged_ac(app, update):
    app.notebook.select(0); app.seek(5.25)
    app.ac_combo.current(0); app.ac_combo.event_generate('<<ComboboxSelected>>')
    update()
    assert len(app.cards) == 5 and app.ac_filter.get() == 'AC_all'
    objects = app.ac_objects()
    expected = [('cross_ON_OFF_parasol_AC_v1', 3)] + [('AII_AC', int(row)) for row in app.network['AII_AC']]
    assert objects == expected and len(objects) == 11
    state = app.ac_state()
    assert state['key'] == 'V' and state['unit'] == 'V'
    np.testing.assert_array_equal(state['xy_mm'], [app.data.geometry[p]['xy_mm'][r] for p, r in expected])
    np.testing.assert_array_equal(state['values'], [app.data.spatial(p, 'V', 5.25)['values'][r] for p, r in expected])
    assert state['markers'] == ['square']+['circle']*10
    canvas = app.ac_canvas
    assert {cell for _, _, cell in app.maps[canvas][1]} == set(expected)
    for wanted in (expected[0], expected[1], expected[-1]):
        x, y, cell = next(point for point in app.maps[canvas][1] if point[2] == wanted)
        app.click_cell(canvas, SimpleNamespace(x=x, y=y))
        assert app.cell == wanted and str(app.data.geometry[cell[0]]['ids'][cell[1]]) in app.detail.get()
    app.select_cell(*expected[0])
    assert '解剖成员数未定义' in app.detail.get() and '代表成员：1' not in app.detail.get()
    for language in ('ja', 'en', 'zh'):
        app.change_language(language)
        if language == 'en': assert 'anatomical member count undefined' in app.detail.get()
    app.ac_combo.current(1); app.ac_combo.event_generate('<<ComboboxSelected>>'); update()
    assert app.ac_filter.get() == 'AII_AC' and len(app.ac_objects()) == 10
    assert app.cell[0] == 'AII_AC'
    assert all(cell[0] == 'AII_AC' for _, _, cell in app.maps[canvas][1])
    x, y, cell = app.maps[canvas][1][len(app.maps[canvas][1])//2]
    app.on_wheel(canvas, SimpleNamespace(x=x, y=y, delta=120)); update()
    assert app.zoom_levels['AII_AC'] == 1.25
    viewport = tuple(a.copy() for a in app.viewports['AII_AC'])
    for code in ('ja', 'en', 'zh'):
        selected = app.cell
        app.change_language(code); update()
        assert app.ac_filter.get() == 'AII_AC' and app.cell == selected
        np.testing.assert_array_equal(app.viewports['AII_AC'], viewport)
        if code == 'en':
            assert list(app.ac_combo.cget('values')) == ['All AC (including AII)', 'AII only']
    canvas.focus_force(); app.hover_canvas = canvas; update(); app.reset_zoom(); update()
    assert 'AII_AC' not in app.viewports
    app.ac_combo.current(0); app.ac_combo.event_generate('<<ComboboxSelected>>'); update()
    assert len(app.ac_objects()) == 11
    app.window_figure((1800, 1200))
    assert app.figure_references['AC']['population'] == 'cross_ON_OFF_parasol_AC_v1'
    assert app.figure_references['AII']['population'] == 'AII_AC'
    other = next(i for i, target in enumerate(app.targets) if target['population'] == 'small_bistratified_S_ON')
    app.rgc.set(app.target_labels[other]); app.select_target(); update()
    state = app.ac_state()
    local = [i for i, cell in enumerate(state['objects']) if cell[0] == 'local_feedforward_AC']
    assert local and state['unit'] == 'V' and np.isnan(state['values'][local]).all()
    p, r = state['objects'][local[0]]
    app.select_cell(p, r); app.render(); update()
    state = app.ac_state()
    assert state['key'] == 'h' and state['unit'] == app.data.variables(p)['h']['unit']
    np.testing.assert_array_equal(state['values'][local], [app.data.spatial(p, 'h', 5.25)['values'][state['objects'][i][1]] for i in local])
    assert np.isnan(state['values'][[i for i, cell in enumerate(state['objects']) if cell[0] == 'AII_AC']]).all()
    original = next(i for i, target in enumerate(app.targets) if target.get('is_target'))
    app.rgc.set(app.target_labels[original]); app.select_target(); update()
    assert len(app.ac_objects()) == 11


def main():
    source = ROOT / CURRENT_RESULT
    report = source / 'VISUALIZATION_CHECK.json'
    started = time.perf_counter()
    checks, durations, errors = {}, {}, []
    originals = [source / name for name in ('PREFIX_45s.npz', 'BACKGROUND.json',
        'readout/MODEL_FROZEN.json', 'readout/FIXED_DEVELOPMENT_PREDICTIONS.npz',
        'CHECKPOINT_50000/manifest.json')]
    before = {str(p.relative_to(ROOT)): digest(p) for p in originals}
    enable_high_dpi()
    root = tk.Tk()
    root.report_callback_exception = lambda *args: errors.append(''.join(traceback.format_exception(*args)))
    app = InternalStateApp(root, ROOT, source)
    root.title('69#6 内部状态 · 真实读取验收')
    root.geometry(f'{round(1500*app.display_scale)}x{round(960*app.display_scale)}')

    def update():
        root.update()
        if errors:
            raise AssertionError(errors[-1])

    def mark(name):
        checks[name] = True

    try:
        deadline = time.monotonic() + 180
        while app.data is None or app.busy:
            update()
            if app.status.get().startswith('加载失败'):
                raise AssertionError(app.status.get())
            if time.monotonic() > deadline:
                raise TimeoutError(app.status.get())
            time.sleep(.02)
        durations['desktop_import_seconds'] = time.perf_counter() - started
        update()
        data = app.data
        app.speed.set('2×'); app.start.set('5'); app.jump.set('5')
        app.loaded(data); update()
        assert app.clock.speed == 2. and app.bounds() == (0., data.duration)
        assert float(app.jump.get()) == data.duration
        app.speed.set('1×'); app.clock.speed = 1.
        mark('result_reload_preserves_speed_and_resets_time_controls')
        expected = read(source / 'NEURAL_RESULT.json')
        assert data.identity == expected['model_identity'] == 'b9d8f52c0d91ce06777f1f835b52ed807f0ccce0726b152407954f9ec57fdde6'
        assert data.target == ('parasol_RGC_OFF', expected['target_row']) == ('parasol_RGC_OFF', 3)
        target_geometry = data.geometry[data.target[0]]
        assert target_geometry['members'][3] == 1
        assert str(target_geometry['ids'][3]).endswith(':7730')
        assert digest(source / 'PREFIX_45s.npz') == read(source / 'PREFIX_45s.json')['sha256']
        mark('current_model_target_and_prefix_identity')

        for pop in ('midget_BC_ON', 'midget_BC_OFF', 'S_ON_BC', 'diffuse_BC_ON', 'diffuse_BC_OFF'):
            geometry = data.geometry[pop]
            assert np.all(geometry['members'] == 1)
            assert geometry['representation'] == 'independently_solved_BC_cell'
            assert len(np.unique(geometry['ids'])) == len(geometry['ids'])
            assert geometry['coordinate_role'] == 'rf'
        assert len(data.geometry['HC']['ids']) == 451
        assert np.count_nonzero(data.geometry['HC']['subtypes'] == 'H1') == 325
        assert np.count_nonzero(data.geometry['HC']['subtypes'] == 'H2') == 126
        assert data.geometry['cones']['representation'] == 'representative_group'
        assert all(data.geometry[p]['coordinate_role'] == 'effective_functional_RF_pool_not_anatomical_AC'
            for p in data.geometry if p.startswith(('same_', 'cross_')))
        mark('real_ids_members_and_separate_coordinate_roles')

        with np.load(source / 'PREFIX_45s.npz', allow_pickle=False) as raw:
            t = raw['time_source_s']
            assert t.shape == (450001,) and t[0] == 0 and t[-1] == 45
            for key, unit in (('V', 'V'), ('gE', 'S'), ('gI', 'S')):
                series = data.series(*data.target, key, 29.98, 30.04, max_points=127)
                indices = np.searchsorted(t, series['time'])
                np.testing.assert_array_equal(series['time'], t[indices])
                np.testing.assert_array_equal(series['values'], raw[key][indices])
                assert series['unit'] == unit and len(series['time']) <= 127
                full = data.series(*data.target, key, 0., 45., max_points=2000)
                assert len(full['time']) <= 2000 and np.all(np.diff(full['time']) > 0)
                assert full['values'].min() == raw[key].min() and full['values'].max() == raw[key].max()
            state = data.spatial(data.target[0], 'V', 12.34567)
            index = np.searchsorted(t, state['time'])
            assert state['time'] <= 12.34567
            assert state['values'][3] == raw['V'][index]
            base = raw['expected_rate_base_Hz'].reshape(-1, 20).mean(axis=1)
        mark('exact_saved_samples_units_and_bounded_curve_points')

        h1 = int(np.flatnonzero(data.geometry['HC']['subtypes'] == 'H1')[0])
        series = data.series('HC', h1, 'Vh')
        assert series['recording'] == 'windowed' and series['unit'] == 'vH_native_voltage'
        assert series['time'][0] == 0. and series['time'][-1] == 45.
        assert 5.0005 in {record[0] for record in data._snapshots['HC', 'Vh']}
        assert np.isfinite(data.spatial('HC', 'Vh', 5.)['values']).all()
        assert np.isfinite(data.spatial('HC', 'Vh', 5.0005)['values']).all()
        assert np.isfinite(data.spatial('HC', 'Vh', 45.)['values']).all()
        assert np.isnan(data.spatial('midget_BC_ON', 'V', 1.)['values']).all()
        assert np.isfinite(data.spatial('midget_BC_ON', 'V', 5.)['values']).all()
        assert np.count_nonzero(np.isfinite(data.spatial('cones', 'release', 45.)['values'])) == 233
        assert not len(data.series('HC', h1, 'Vh', 1., 2.)['time'])
        mark('native_and_inner_snapshot_times_not_interpolated_or_held')
        replay = check_short_window(data, source)
        mark('short_window_real_fields_ids_units_and_no_outside_hold')
        check_trace_gap(series)
        mark('continuous_window_does_not_connect_to_sparse_history')

        with np.load(source / 'readout/FIXED_DEVELOPMENT_PREDICTIONS.npz', allow_pickle=False) as raw:
            predicted = raw['probability'].reshape(-1, 20).sum(axis=1) / .02
            measured = raw['observed_counts'].mean(axis=0).reshape(-1, 20).sum(axis=1) / .02
            t = raw['time_bin_start_seconds'][::20]
            comparison = data.comparison(0., 45., 3000)
            np.testing.assert_array_equal(comparison['time'], t)
            np.testing.assert_allclose(comparison['measured'], measured, rtol=1e-14, atol=1e-12)
            np.testing.assert_allclose(comparison['glm'], predicted, rtol=1e-14, atol=1e-12)
            np.testing.assert_array_equal(comparison['base'], base)
            assert comparison['unit'] == 'Hz' and comparison['bin_width_s'] == .02
        mark('observed_psth_frozen_glm_original_readout_separated')

        checkpoint = read(source / 'CHECKPOINT_450000/manifest.json')
        payload = dict(checkpoint['payload']['dictionary'])
        controller = dict(payload['controller']['dictionary'])
        event_log = dict(controller['event_log']['dictionary'])
        events = [dict(e['dictionary']) for e in event_log[data.target[0]]]
        expected_spikes = np.asarray([e['time_seconds'] for e in events
            if e['cell_index'] == 3 and 0 <= e['time_seconds'] < 45])
        spikes = data.series(*data.target, 'spike', max_points=len(expected_spikes)+1)
        np.testing.assert_array_equal(spikes['time'], expected_spikes)
        assert len(expected_spikes) > 0 and spikes['recording'] == 'events'
        assert np.all(spikes['time'] < 45)
        del checkpoint, payload, controller, event_log, events
        mark('original_spikes_match_saved_event_log_no_future_pending_events')

        weights = data.h1_inputs(data.available_h1_rows[0])
        np.testing.assert_allclose(weights['short'] + weights['long'], weights['total'], rtol=0, atol=2e-17)
        np.testing.assert_allclose([weights['short'].sum(), weights['long'].sum()], [.15, .85], rtol=0, atol=2e-14)
        assert weights['coordinate_role'] == 'OPL_pedicle_mm'
        assert not weights['metadata']['response_independently_recorded']
        mark('saved_h1_complete_support_short_long_contributions')

        capture = data.optical_capture(12.345)
        stimulus = data.stimulus(12.345)
        assert capture['time'] == stimulus['time'] <= 12.345
        assert capture['values'].shape == capture['ids'].shape
        assert capture['xy_mm'].shape == (len(capture['ids']), 2)
        assert capture['unit'] == 'expected R*/frame'
        assert stimulus['image'].dtype == np.uint8 and stimulus['image'].shape[-1] == 3
        mark('saved_stimulus_and_optical_capture_shared_frame_timestamp')

        interaction = time.perf_counter()
        app.seek(5.)
        update()
        assert app.clock.position == 5. and not app.clock.playing
        app.jump.set('12.345')
        app.jump_to()
        update()
        assert app.clock.position == 12.345
        app.slider_changed('5.0005')
        update()
        assert app.clock.position == 5.0005
        app.toggle_play()
        end = time.monotonic() + .25
        while time.monotonic() < end:
            update()
            time.sleep(.01)
        app.toggle_play()
        assert app.clock.position > 5.0005 and not app.clock.playing
        app.seek(5.)
        canvas = app.cards[0][0]
        population, points = app.maps[canvas]
        assert points
        x, y, row = points[0]
        app.click_cell(canvas, SimpleNamespace(x=x, y=y))
        assert app.cell == (population, row)
        app.show_curve()
        app.variable.set('release')
        app.render_curve()
        update()
        assert app.curve_canvas in app.images
        app.show_info()
        update()
        for index in range(1, 5):
            app.notebook.select(index)
            update()
            app.render()
            assert app.tabs[index].canvas in app.images
        other = next(i for i, item in enumerate(app.targets) if item['population'] != app.target[0])
        app.rgc.set(app.target_labels[other])
        app.select_target()
        assert app.target == (app.targets[other]['population'], app.targets[other]['row'])
        assert app.target[1] in app.network[app.target[0]]
        target = next(i for i, item in enumerate(app.targets) if item.get('is_target'))
        app.rgc.set(app.target_labels[target])
        app.select_target()
        durations['interaction_checks_seconds'] = time.perf_counter() - interaction
        mark('desktop_select_click_curve_play_pause_slider_jump_all_tabs')
        zoom_population = check_controls(app, update)
        mark('mouse_wheel_local_zoom_hit_testing_and_R_reset')
        mark('three_languages_preserve_selection_time_zoom_and_new_dialogs')
        mark('english_plot_titles_legends_and_annotations_translated')
        mark('short_window_button_and_playback_stop_at_end')
        display = check_display_and_stimulus(app, update)
        mark('layer_stimulus_actual_frames_timestamps_and_resolution')
        mark('native_dpi_raster_dimensions_and_scaled_hit_coordinates')
        check_merged_ac(app, update)
        mark('merged_ac_scope_counts_roles_units_filter_and_real_id_picking')

        temporary_root = ROOT / 'tmp'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='internal_state_acceptance_', dir=temporary_root) as temporary:
            assert Path(temporary).resolve().is_relative_to(temporary_root.resolve())
            for index in (0, 1, 2, 3, 4):
                path = Path(temporary) / ('figure_'+str(index)+'.png')
                app.save_figure(path, index=index)
                with Image.open(path) as image:
                    provenance = json.loads(image.info['retina_provenance'])
                    assert provenance['model_identity'] == data.identity
                    assert provenance['time_seconds'] == app.clock.position
                    assert provenance['continuous_window'] == [5., 5.5]
                    assert image.width >= 1200 and image.height >= 800
                    assert provenance['ac_filter'] == app.ac_filter.get()
                    assert provenance['ac_scope'] == 'current selected RGC saved connection ancestors'
                    image.verify()
            for code in ('zh', 'ja', 'en'):
                app.change_language(code)
                path = Path(temporary) / ('language_'+code+'.png')
                app.save_figure(path, index=3)
                with Image.open(path) as image:
                    provenance = json.loads(image.info['retina_provenance'])
                    assert provenance['ui_language'] == code
                    assert zoom_population in provenance['layer_viewports']
                    assert provenance['continuous_manifest'].endswith('internal_states\\manifest.json') or provenance['continuous_manifest'].endswith('internal_states/manifest.json')
                    image.verify()
            app.change_language('zh')
        mark('native_png_export_contains_model_and_time_provenance')

        after = {str(p.relative_to(ROOT)): digest(p) for p in originals}
        assert after == before
        assert not any(name in sys.modules for name in ('tools.run_gin69_current_ac', 'tools.resume_gin69_current_ac'))
        assert not errors
        mark('scientific_artifacts_unchanged_no_runner_import_no_tk_errors')
        result = dict(status='PASS', model_identity=data.identity, checks=checks,
            performance_seconds=durations, wall_seconds=time.perf_counter()-started,
            display=display,
            original_artifact_sha256=before, scope='read-only desktop acceptance',
            simulation_runs_during_this_check=0, fits=0,
            short_window_seconds=list(data.continuous_window),
            short_window_manifest_sha256=digest(source / 'internal_states/manifest.json'),
            short_window_recorded_fields=len(replay['fields']),
            replay_source=dict(manifest='internal_states/manifest.json', recorded_seconds=.5,
                simulation_runs=1, executed_by_this_check=False),
            screenshot_validation='Separate native desktop visual inspection; this check does not capture the screen')
    except BaseException:
        result = dict(status='FAIL', checks=checks, performance_seconds=durations,
            wall_seconds=time.perf_counter()-started, error=traceback.format_exc(), callback_errors=errors,
            simulation_runs_during_this_check=0, fits=0)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    finally:
        app.close()
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
