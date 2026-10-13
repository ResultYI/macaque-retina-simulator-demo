"""Actual Tk widget/retina integration acceptance using labelled synthetic media.

File-picker return paths are injected; Tk buttons, canvas events, media decode,
worker and core run normally. Screenshots capture the visible native window.
"""
from pathlib import Path
import argparse, json, time, traceback, hashlib
from datetime import datetime, timezone
import tkinter as tk
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reuse-image-run', type=Path, help='Reuse a completed image acceptance run; do not rerun its simulation')
    args = parser.parse_args()
    project = args.project_root.resolve()
    sys.path.insert(0, str(project))
    from retina_platform.app import RetinaApp
    from retina_platform import app as app_module
    from retina_platform.results import ResultStore
    output = args.output or project / 'reports/platform_acceptance' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    root = tk.Tk()
    app = RetinaApp(root, project)
    root.geometry('1400x900+100+100')
    root.attributes('-topmost', True)
    checks, runs, timings = [], {}, []
    phase = 'begin'
    phase_started = time.monotonic()
    last_pulse = phase_started
    paused_hash = None
    paused_at = None
    pause_requested = False
    stop_requested = False
    original_dialogs = (app_module.filedialog.askopenfilename, app_module.filedialog.asksaveasfilename, app_module.filedialog.askdirectory)
    image = project / 'examples/platform_media/synthetic_rgb.png'
    video = project / 'examples/platform_media/synthetic_rgb_50fps.avi'

    def record(name, **extra):
        checks.append({'check': name, 'status': 'PASS', **extra})
        print(json.dumps(checks[-1]), flush=True)

    def finish(error=None):
        report = {'passed': error is None, 'checks': checks, 'runs': runs,
                  'ui_heartbeat_max_gap_seconds': max(timings, default=0.),
                  'input_class': 'synthetic_engineering_media', 'scientific_validation': 'NOT_RUN',
                  'interaction_method': 'actual Tk buttons/canvas events; file-picker paths injected',
                  'error': error}
        (output / 'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        app_module.filedialog.askopenfilename, app_module.filedialog.asksaveasfilename, app_module.filedialog.askdirectory = original_dialogs
        print(json.dumps({'output': str(output), 'passed': report['passed']}), flush=True)
        app.close()

    def set_params(**values):
        for key, value in values.items():
            app.variables[key].set(str(value))

    def click_file(button, path, dialog='open'):
        if dialog == 'open':
            app_module.filedialog.askopenfilename = lambda **kwargs: str(path)
        elif dialog == 'save':
            app_module.filedialog.asksaveasfilename = lambda **kwargs: str(path)
        else:
            app_module.filedialog.askdirectory = lambda **kwargs: str(path)
        app.buttons[button].invoke()

    def checked_run(name):
        store = app.result_store
        assert store is not None and store.windows
        summary = json.loads((store.path / 'summary.json').read_text(encoding='utf-8'))
        assert summary['status'] == 'completed'
        assert len(store.manifest['required_layers']) == 20
        assert summary['components'] == 39 and not summary['unimplemented_layers']
        assert not summary['state_reset_between_windows']
        assert store.manifest['scientific_validation'] == 'NOT_RUN'
        for i in range(len(store.windows)):
            window = store.read_window(i)
            assert all(np.all(np.isfinite(v)) for v in window['signals'].values())
        reused = name == 'image' and args.reuse_image_run is not None
        runs[name] = {'path': str(store.path), 'execution_reused_from_prior_acceptance': reused, **summary}
        record(name + '_actual_core', windows=len(store.windows), layers=20, components=39, execution_reused=reused)

    def image_exports():
        store = app.result_store
        for layer in store.manifest['required_layers']:
            app.layer_var.set(layer)
            app.layer_combo.event_generate('<<ComboboxSelected>>')
            root.update_idletasks()
            assert app.port_records, layer
        app.layer_var.set('\u5168\u90e8\u5c42')
        app.layer_combo.event_generate('<<ComboboxSelected>>')
        key = 'midget_RGC_ON.voltage'
        assert key in app.port_records
        app.port_var.set(key)
        app.port_combo.event_generate('<<ComboboxSelected>>')
        app._render_results()
        assert app.pointmap
        px, py, index = app.pointmap[len(app.pointmap)//2]
        app.scatter_canvas.event_generate('<Button-1>', x=int(px), y=int(py))
        app._render_results()
        assert app.selected_cell is not None
        meta = app._port_metadata()
        assert len(meta['geometry']['RF']['values']) == len(meta['cell_ids'])
        record('all_20_layers_selectable_and_actual_cell_click')
        click_file('export_curve', output / 'selected_cell.csv', 'save')
        click_file('export_png', output / 'response_export.png', 'save')
        assert (output / 'selected_cell.csv').is_file() and (output / 'response_export.png').is_file()
        history_t, history_v = store.history(key, app.selected_cell)
        import csv
        with (output / 'selected_cell.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.reader(stream))
        assert len(rows) - 1 == len(history_t)
        assert np.array_equal(np.array([float(row[1]) for row in rows[1:]]), history_v)
        bundle_parent = output / 'exports'
        bundle_parent.mkdir()
        click_file('export_bundle', bundle_parent, 'directory')
        bundle = bundle_parent / store.path.name
        assert bundle.is_dir()
        assert hashlib.sha256((bundle / 'final_checkpoint.json').read_bytes()).hexdigest() == hashlib.sha256((store.path / 'final_checkpoint.json').read_bytes()).hexdigest()
        click_file('load_results', bundle, 'directory')
        assert app.result_store.path == bundle
        app.port_var.set(key)
        app.port_combo.event_generate('<<ComboboxSelected>>')
        app.selected_cell = index
        app._render_results()
        record('config_results_csv_png_checkpoint_export_and_reload')

    def pulse():
        nonlocal phase, phase_started, last_pulse, paused_hash, paused_at, pause_requested, stop_requested
        now = time.monotonic()
        timings.append(now - last_pulse)
        last_pulse = now
        try:
            if now - phase_started > 900:
                raise TimeoutError('Acceptance phase exceeded 15 minutes: ' + phase)
            if phase != 'begin' and app.session is not None and app.session.status == 'error':
                raise RuntimeError(app.status_var.get())
            if phase == 'begin':
                assert not app.import_media(project / 'examples/platform_media/does_not_exist.png')
                click_file('import', image)
                assert app.media.info['width'] == 480 and app.media.info['height'] == 270
                assert tuple(app.media.frame_at(0)[20, 20]) == (255, 0, 0)
                set_params(duration_seconds=-.006)
                app.buttons['start'].invoke()
                assert app.session is None and app.status == 'error'
                set_params(duration_seconds=.006, warmup_seconds=.002, window_seconds=.002, sample_seconds=.001)
                click_file('save_config', output / 'image_config.json', 'save')
                set_params(effective_cones=80)
                click_file('load_config', output / 'image_config.json')
                assert int(app.variables['effective_cones'].get()) == 64
                record('actual_import_RGB_invalid_file_invalid_duration_config_roundtrip')
                if args.reuse_image_run:
                    assert app.load_results(args.reuse_image_run)
                    checked_run('image')
                    image_exports()
                    phase, phase_started = 'image_capture', now
                else:
                    app.buttons['start'].invoke()
                    assert app.session and app.session.is_alive()
                    phase, phase_started = 'image', now
            elif phase == 'image' and app.status == 'completed' and not app.session.is_alive():
                checked_run('image')
                image_exports()
                phase, phase_started = 'image_capture', now
            elif phase == 'image_capture' and now - phase_started > .6:
                app.screenshot(output / 'desktop_image.png')
                click_file('import', video)
                assert app.media.info['fps'] == 50. and app.media.info['frame_count'] == 6
                first, next_frame = app.media.frame_at(.02), app.media.frame_at(.04)
                assert not np.array_equal(first, next_frame)
                assert first[20, 20, 0] > 200 and first[20, 20, 2] < 30
                set_params(start_seconds=.02, duration_seconds=.045)
                click_file('save_config', output / 'video_config.json', 'save')
                app.buttons['start'].invoke()
                phase, phase_started = 'video', now
            elif phase == 'video':
                session = app.session
                if session._windows >= 1 and not pause_requested:
                    app.buttons['pause'].invoke()
                    pause_requested = True
                if app.status == 'paused' and paused_at is None:
                    paused_at = now
                    paused_hash = hashlib.sha256((session.path / 'latest_checkpoint.json').read_bytes()).hexdigest()
                    assert str(app.buttons['export_bundle']['state']) == 'normal'
                elif app.status == 'paused' and now - paused_at > .6:
                    assert paused_hash == hashlib.sha256((session.path / 'latest_checkpoint.json').read_bytes()).hexdigest()
                    app.buttons['resume'].invoke()
                    record('pause_preserves_checkpoint_and_resume')
                    phase, phase_started = 'video_resumed', now
            elif phase == 'video_resumed' and app.status == 'completed' and not app.session.is_alive():
                checked_run('video')
                logs = [json.loads(line) for line in (app.session.path / 'progress.jsonl').read_text(encoding='utf-8').splitlines()]
                frame_ids = {row['selected_frame_index'] for event in logs for row in event.get('input_summary', {}).get('frame_timestamps', [])}
                assert {1, 2, 3} <= frame_ids
                assert float(app.progress['value']) == 1.
                record('selected_video_interval_real_timestamps_cross_frame_continuity', frame_indices=sorted(frame_ids))
                app.port_var.set('midget_RGC_ON.voltage')
                app.port_combo.event_generate('<<ComboboxSelected>>')
                app._render_results()
                app.selected_cell = 0
                app._render_results()
                phase, phase_started = 'video_capture', now
            elif phase == 'video_capture' and now - phase_started > .6:
                app.screenshot(output / 'desktop_video.png')
                click_file('import', image)
                set_params(start_seconds=0., duration_seconds=.020)
                app.buttons['start'].invoke()
                phase, phase_started = 'stop', now
            elif phase == 'stop':
                if app.session._windows >= 1 and not stop_requested:
                    app.buttons['stop'].invoke()
                    stop_requested = True
                if app.status == 'stopped' and not app.session.is_alive():
                    summary = json.loads((app.session.path / 'summary.json').read_text(encoding='utf-8'))
                    assert 0 < summary['windows'] < 10
                    assert (app.session.path / 'final_checkpoint.json').is_file()
                    runs['stop'] = {'path': str(app.session.path), **summary}
                    record('repeat_run_stop_keeps_completed_windows_and_checkpoint', windows=summary['windows'])
                    assert str(app.buttons['start']['state']) == 'normal'
                    assert all(str(entry['state']) == 'normal' for entry in app.entries.values())
                    legacy = project / 'reports/full_target_runs/20261011T000323412125Z'
                    originals = {str(p.relative_to(legacy)): hashlib.sha256(p.read_bytes()).hexdigest() for p in legacy.rglob('*') if p.is_file()}
                    click_file('load_results', legacy, 'directory')
                    assert app.result_store.path == legacy
                    app.layer_var.set('\u5168\u90e8\u5c42')
                    app.layer_combo.event_generate('<<ComboboxSelected>>')
                    app.port_var.set('outer_native.Vh')
                    app.port_combo.event_generate('<<ComboboxSelected>>')
                    app._render_results()
                    metadata = app._port_metadata()
                    assert len(metadata['cell_ids']) == 32 and set(metadata['cell_types']) == {'HC'}
                    app.selected_cell = 0
                    app._render_results()
                    assert all(hashlib.sha256((legacy / name).read_bytes()).hexdigest() == sha for name, sha in originals.items())
                    record('read_only_existing_core_results_with_native_units_and_HC_geometry', original_files_unchanged=len(originals))
                    phase, phase_started = 'legacy_capture', now
            elif phase == 'legacy_capture' and now - phase_started > .6:
                    app.screenshot(output / 'desktop_legacy_result.png')
                    finish()
                    return
        except Exception:
            finish(traceback.format_exc())
            return
        root.after(120, pulse)

    root.after(600, pulse)
    root.mainloop()
    report = json.loads((output / 'acceptance.json').read_text(encoding='utf-8'))
    return 0 if report['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
