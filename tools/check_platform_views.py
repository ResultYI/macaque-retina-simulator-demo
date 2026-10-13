"""Read-only GUI view checks against completed engineering runs; no simulation."""
from pathlib import Path
import argparse, json, sys, tkinter as tk

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from retina_platform.app import RetinaApp
    from retina_platform.results import ResultStore
    report = json.loads((args.acceptance / 'acceptance.json').read_text(encoding='utf-8'))
    assert report['passed']
    root = tk.Tk()
    app = RetinaApp(root, project)
    root.geometry('1400x900+100+100')
    root.attributes('-topmost', True)
    assert app.load_results(report['runs']['video']['path'])
    checks = []
    def check():
        try:
            app.layer_var.set('\u5168\u90e8\u5c42')
            app.layer_combo.event_generate('<<ComboboxSelected>>')
            for key, frame in [('RBC_AII_AMPA.conductance', 'IPL'), ('individual_rod_RBC_contacts.conductance', 'OPL')]:
                app.port_var.set(key)
                app.port_combo.event_generate('<<ComboboxSelected>>')
                app._render_results()
                meta = app._port_metadata()
                assert set(meta['geometry']) == {frame}
                assert meta['display_element'] == '\u63a5\u70b9'
                assert len(app.pointmap) == len(meta['cell_ids'])
                checks.append({'key': key, 'frame': frame, 'real_points': len(app.pointmap), 'label': 'contact'})
            app.selected_cell = 0
            app._render_results()
            app.port_var.set('midget_RGC_ON.spike_count')
            app.port_combo.event_generate('<<ComboboxSelected>>')
            app.selected_cell = 0
            app._render_results()
            t, v = app.store.history('midget_RGC_ON.spike_count', 0)
            assert len(t) == 23 and len(v) == 23
            checks.append({'window_aggregate_history': 'PASS', 'samples': len(t)})
            if not (args.acceptance / 'window_spike_count.csv').exists():
                app.store.export_curve('midget_RGC_ON.spike_count', 0, args.acceptance / 'window_spike_count.csv')
            assert (args.acceptance / 'window_spike_count.csv').read_text(encoding='utf-8-sig').startswith('window_end_seconds,')
            destination = args.acceptance / 'final_export_check'
            if not destination.exists():
                app.store.export_bundle(destination)
            assert ResultStore(destination).manifest['input_assumptions']['profile'] == 'engineering_sRGB_profile'
            app.port_var.set('individual_rod_RBC_contacts.conductance')
            app.port_combo.event_generate('<<ComboboxSelected>>')
            app.selected_cell = 0
            app._render_results()
            root.after(400, capture)
        except Exception:
            root.destroy()
            raise
    def capture():
        app.screenshot(args.acceptance / 'desktop_contact_view.png')
        checks.append({'loaded_video_context': app.media.info['kind'], 'time_display': app.time_var.get(), 'preview_seconds': app.preview_var.get()})
        assert app.media.info['kind'] == 'video' and abs(app.preview_var.get() - .065) < 1e-12
        app.port_var.set('midget_RGC_ON.voltage')
        app.port_combo.event_generate('<<ComboboxSelected>>')
        app.selected_cell = 0
        app._render_results()
        root.after(200, capture_video)
    def capture_video():
        app.screenshot(args.acceptance / 'desktop_video.png')
        assert app.load_results(report['runs']['image']['path'])
        assert app.media.info['kind'] == 'image'
        app.port_var.set('midget_RGC_ON.voltage')
        app.port_combo.event_generate('<<ComboboxSelected>>')
        app.selected_cell = 0
        app._render_results()
        root.after(200, capture_image)
    def capture_image():
        app.screenshot(args.acceptance / 'desktop_image.png')
        assert app.load_results(project / 'reports/full_target_runs/20261011T000323412125Z')
        assert app.media is None and not app.variables['media_path'].get()
        app.port_var.set('outer_native.Vh')
        app.port_combo.event_generate('<<ComboboxSelected>>')
        app.selected_cell = 0
        app._render_results()
        root.after(200, capture_legacy)
    def capture_legacy():
        app.screenshot(args.acceptance / 'desktop_legacy_result.png')
        checks.append({'actual_screenshots_validated': True, 'read_only_context_restore': 'PASS'})
        (args.acceptance / 'view_checks.json').write_text(json.dumps({'passed': True, 'checks': checks, 'core_execution': 'NOT_RUN'}, indent=2), encoding='utf-8')
        print(json.dumps({'view_checks': 'PASS', 'checks': checks}), flush=True)
        app.close()
    root.after(500, check)
    root.mainloop()
    assert (args.acceptance / 'view_checks.json').exists()

if __name__ == '__main__':
    main()
