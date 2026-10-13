"""Export the single completed background job without running model equations."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def render(path, background, trace, cell_id):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 18)
    small = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 15)
    title = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 25)
    im = Image.new('RGB', (1280, 720), '#f6f8fa')
    draw = ImageDraw.Draw(im)
    draw.text((25, 20), 'Uniform background preparation and continuous stimulus', font=title, fill='#173244')
    draw.text((25, 62), f"Numerical criterion: {background['status']} | physiological validation NOT_RUN", font=font, fill='#435361')
    left, right = (90, 160, 590, 515), (740, 160, 1210, 515)
    for box in (left, right):
        draw.rectangle(box, outline='#93a2ad')
    rows = [r for r in background['windows'] if r.get('complete_window')]
    duration = max(background['elapsed_seconds'], 1.3)
    low = min(-3., np.floor(np.log10(min(max(r['maximum_norm'], 1e-12) for r in rows))))
    high = max(4., np.ceil(np.log10(max(r['maximum_norm'] for r in rows))))
    def point(t, norm):
        return (left[0] + t / duration * (left[2] - left[0]),
                left[3] - (np.log10(max(norm, 10. ** low)) - low) / (high - low) * (left[3] - left[1]))
    for power in range(int(low), int(high) + 1):
        y = point(0., 10. ** power)[1]
        draw.line((left[0], y, left[2], y), fill='#dce3e8')
        draw.text((28, y - 9), f'1e{power}', font=small, fill='#435361')
    y = point(0., 1.)[1]
    draw.line((left[0], y, left[2], y), fill='#29835b', width=2)
    draw.text((left[0] + 8, y - 22), 'pass <= 1 (existing tolerance)', font=small, fill='#29835b')
    minimum = background['timing_requirements']['minimum_required_seconds']
    x = point(minimum, 1.)[0]
    draw.line((x, left[1], x, left[3]), fill='#a0742d', width=2)
    pts = [point(r['stop_seconds'], r['maximum_norm']) for r in rows]
    if len(pts) > 1:
        draw.line(pts, fill='#347fac', width=3)
    for p, r in zip(pts, rows):
        draw.ellipse((p[0]-4, p[1]-4, p[0]+4, p[1]+4), fill='#29835b' if r['passed'] else '#347fac')
    draw.text((90, 108), 'Maximum full-window normalized residual', font=font, fill='#173244')
    draw.text((90, 134), 'Observed excursion / original RHS / discrete defect', font=small, fill='#435361')
    draw.text((90, 535), f"Background: 0 .. {background['elapsed_seconds']:.3f} s; minimum {minimum:g} s", font=font, fill='#435361')
    draw.text((740, 108), 'Actual owner midget RGC ON voltage', font=font, fill='#173244')
    if trace is not None:
        t, v = trace
        t = (t - background['stimulus_origin_seconds']) * 1000
        v = v * 1000
        lo, hi = float(v.min()), float(v.max())
        pad = max((hi-lo) * .1, .0001)
        pts = [(right[0] + tt/max(float(t[-1]), 1e-12)*(right[2]-right[0]),
                right[3] - (vv-lo+pad)/(hi-lo+2*pad)*(right[3]-right[1])) for tt, vv in zip(t, v)]
        draw.line(pts, fill='#ae532a', width=3)
        draw.text((740, 134), f'{cell_id}; {len(t)} actual samples', font=small, fill='#435361')
        draw.text((740, 535), f'0 .. {t[-1]:.1f} ms; {lo:.6f} .. {hi:.6f} mV', font=font, fill='#435361')
    else:
        draw.text((765, 295), 'Stimulus withheld: preparation not converged', font=font, fill='#8e452b')
    draw.text((25, 590), 'Same trajectory, event grid and RNG state; no state reset; source geometry and solver unchanged.', font=font, fill='#435361')
    draw.text((25, 624), 'sRGB128 input is a display approximation. BC/AC/RGC and contact population dynamics remain uncalibrated.', font=font, fill='#8e452b')
    draw.text((25, 658), 'Numerical convergence does not establish stochastic stationarity or a physiological steady state.', font=font, fill='#8e452b')
    im.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    job = args.job.resolve()
    outcome, request = read(job/'outcome.json'), read(job/'request.json')
    if outcome['execution_status'] not in ('completed', 'initialization_not_reached'):
        raise ValueError('Only a completed or budget-limited job can be finalized')
    run = Path(outcome['run_directory'])
    background = read(run/'background_initialization.json')
    initial = read(run/'initial_checkpoint.json')['payload']
    prepared = read(run/'prepared_checkpoint.json')['payload']
    final = read(run/'final_checkpoint.json')['payload']
    summary = read(run/'summary.json')
    assert initial['dynamic']['state']['time_seconds'] == 0.
    assert initial['grid_origin_seconds'] == prepared['grid_origin_seconds'] == final['grid_origin_seconds'] == 0.
    for key in ('model_hash', 'parameter_hash', 'graph_hash', 'state_spec_hash', 'solver_config'):
        assert initial['dynamic']['state'][key] == prepared['dynamic']['state'][key] == final['dynamic']['state'][key]
    profile = ROOT/'configurations/retina_contact_geometry_effective_v1.json'
    assert hashlib.sha256(profile.read_bytes()).hexdigest() == request['execution_profile_sha256']
    source_sha = hashlib.sha256((ROOT/'retina_sim/vh2005.py').read_bytes()).hexdigest()
    assert source_sha == 'cfc4ce4f9eb1015755cd790b180faaec83d2a9a7b8da083f3a471a00d36a4900'
    original_policy = read(ROOT/'configurations/bounded_uniform_background_v1.json')['policy']
    for key, value in background['policy'].items():
        if key not in ('max_sim_seconds', 'max_wall_seconds', 'require_converged'):
            assert original_policy[key] == value
    actual_arrays = final['dynamic']['state']['ordered_state_arrays']
    assert len(actual_arrays) == len(background['checked_states']) == 77 and not background['uncovered_states']
    state_scalars = sum(np.asarray(a).size for a in actual_arrays)
    assert all(np.isfinite(np.asarray(a)).all() for a in actual_arrays)
    assert [np.asarray(a).shape for a in initial['dynamic']['state']['ordered_state_arrays']] == [np.asarray(a).shape for a in actual_arrays]
    rng = {}
    for name, value in prepared['spike_states'].items():
        rng[name] = {'background_advanced': value['rng_state'] != initial['spike_states'][name]['rng_state'],
                     'stimulus_advanced': final['spike_states'][name]['rng_state'] != value['rng_state'],
                     'next_tick': final['component_tick_indices'][name]}
        assert rng[name]['background_advanced']
    trace, cell_id, numeric, samples = None, None, 0, 0
    if background['converged']:
        assert background['elapsed_seconds'] + 1e-12 >= background['timing_requirements']['minimum_required_seconds']
        last = background['windows'][-1]
        assert last['complete_window'] and last['passed'] and last['consecutive_passed_windows'] >= background['policy']['stable_windows']
        assert outcome['execution_status'] == 'completed' and summary['windows'] == 10
        assert not summary['state_reset_between_windows']
        assert abs(summary['simulation_time_seconds'] - background['elapsed_seconds'] - .020) < 1e-12
        assert all(v['stimulus_advanced'] for v in rng.values())
        from retina_platform.results import ResultStore
        store = ResultStore(run)
        info = store.port_info('midget_RGC_ON.voltage')
        index = info['is_owner'].index(True)
        cell_id = info['cell_ids'][index]
        trace = store.history('midget_RGC_ON.voltage', index)
        assert np.all(np.diff(trace[0]) > 0)
        store.export_curve('midget_RGC_ON.voltage', index, job/'prepared_owner_midget_ON_voltage.csv')
        samples = len(trace[0])
        for path in (run/'windows').glob('*.npz'):
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    if key == 'manifest_json':
                        continue
                    a = archive[key]
                    if np.issubdtype(a.dtype, np.number):
                        assert np.isfinite(a).all()
                        numeric += a.size
    else:
        assert summary['windows'] == 0 and outcome['stimulus_windows'] == 0
    last = background['windows'][-1]
    with (job/'final_background_state_residuals.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(last['states'][0]))
        writer.writeheader()
        writer.writerows(last['states'])
    figure = job/'background_and_response.png'
    render(figure, background, trace, cell_id)
    result = {**outcome, 'source_profile_sha256': request['execution_profile_sha256'], 'original_vh2005_sha256': source_sha,
              'final_state_arrays': len(actual_arrays), 'finite_final_state_scalars': state_scalars,
              'finite_recorded_numeric_values': numeric, 'owner_response_samples': samples,
              'owner_response_cell_id': cell_id, 'rng_continuity': rng,
              'last_window_maximum_norm': last['maximum_norm'], 'last_window_consecutive_passed': last['consecutive_passed_windows'],
              'figure': str(figure), 'per_state_csv': str(job/'final_background_state_residuals.csv'),
              'numerical_closure_scope': '77 registered deterministic state arrays; stochastic stationarity and physiological calibration not established',
              'checkpoint_model_hash': final['dynamic']['state']['model_hash'],
              'whole_network_steady_state_claim': background['whole_network_steady_state_claim'],
              'stochastic_status': background['stochastic_status']}
    (job/'closure.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    receipt_path = ROOT/'configurations/retina_contact_geometry_effective_v1.acceptance.json'
    receipt = read(receipt_path)
    receipt['uniform_background_closure'] = {'status': background['status'], 'closure_file': str(job/'closure.json'),
        'run_directory': str(run), 'elapsed_seconds': background['elapsed_seconds'],
        'continuous_stimulus_seconds': .020 if background['converged'] else 0.,
        'checkpoint_model_hash': result['checkpoint_model_hash'], 'physiology_validation': 'NOT_RUN'}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    progress_path = run/'background_progress.json'
    if progress_path.exists():
        progress = read(progress_path)
        progress.update(status=background['status'], checkpoint_file='prepared_checkpoint.json',
                        background_elapsed_seconds=background['elapsed_seconds'],
                        absolute_background_time_seconds=background['elapsed_seconds'],
                        background_wall_seconds=background['wall_seconds'])
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'summary'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
