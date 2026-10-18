"""Whitelist small validation receipts; never copy raw states/assets into Git."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]


def read(relative):
    return json.loads((ROOT / relative).read_bytes())


def main():
    full = read('reports/video_lossless_optimization_20261018T1723Z/complete_comparison_final.json')
    profile = read('reports/advance_exclusive_profile_20261018T230442Z/result.json')
    guard = read('reports/no_delay_publish_fix_20261018T232129Z/actual20ms/result.json')
    if (full['status'] != 'MATCHED_ACTUAL_VIDEO_05S_EQUIVALENCE_AND_PERFORMANCE_PASS'
            or profile['status'] != 'ACTUAL_20_TO_40MS_PAIRED_TIMING_AND_EQUIVALENCE_PASS'
            or guard['status'] != 'NO_DELAY_PUBLICATION_ACTUAL_20MS_EQUIVALENCE_AND_TIMING_PASS'):
        raise ValueError('Completed PASS receipts required')
    document = {
        'schema': 'sanitized_notebook_validation_receipt_v1', 'experiment_date_UTC': '2026-10-18',
        'scope': 'Recorded existing results only; no data, local environment or checkpoints bundled',
        'physiology': {'graph': 'original_HC_v7', 'dependency_nodes': 879437, 'registered_states': 4967023,
            'state_blocks': 77, 'background': '1.012s NOT_REACHED transient',
            'full_macaque_or_steady_state_acceptance': False, 'human_cone_optics_borrowing': True,
            'new_functional_HC_default_or_acceptance': False},
        'model_sha256': guard['model_hash'], 'assembly_sha256': guard['assembly_hash'],
        'input_manifest_sha256': guard['input_manifest_sha256'],
        'complete0_5s': {key: full[key] for key in ('status', 'duration_seconds_each', 'samples_each',
            'windows_each', 'full_state_scalars', 'checks', 'maximum_recorded_absolute_difference',
            'full_state_clock_RNG_history_identity_proof', 'baseline', 'optimized')},
        'complete0_5s_conditions': {'warm_optical_inputs': True, 'optical_preparation_excluded': True,
            'diagnostic_metadata_exceptions': ['compute_backend.plan_seconds', 'compute_backend.compile_or_cache_load_seconds'],
            'checkpoint_policy': 'Original>=60wall seconds after prior write completion at original2ms boundary',
            'rolling_counts_baseline_optimized': [35, 33],
            'window_file_bytes_baseline_optimized': [957554385, 7897957836],
            'full_graph_every1ms_checkpoint': False, 'observed_scalar_count_each_sample': 37449},
        'profile20_40ms': {key: profile[key] for key in ('status', 'timed_duration_seconds_each',
            'control_advance_seconds', 'profile_advance_seconds', 'paired_relative_difference_percent',
            'all_observed_numeric_bytes_units_scientific_event_metadata_equal',
            'complete_endpoint_state_clock_RNG_history_checkpoint_SHA256')},
        'profile20_40ms_exclusive_rows': profile['exclusive_profile']['rows'],
        'profile_caveat': 'Nested exclusive rows are additive; inclusive rows overlap. Outer function exit/boundary0.3745665s is separate. No full-run extrapolation.',
        'no_delay20_40ms': {key: guard[key] for key in ('status', 'timed_duration_seconds_each',
            'control_advance_seconds', 'patched_advance_seconds', 'advance_seconds_saved',
            'advance_reduction_percent', 'advance_speedup', 'control_solver_seconds_included',
            'patched_solver_seconds_included', 'control_outside_solver_seconds', 'patched_outside_solver_seconds',
            'solver_calls_each', 'nfev_each', 'all_paired_metadata_including_diagnostic_timers_equal',
            'original_numerical_source_files_unchanged', 'complete_endpoint_state_clock_RNG_history_checkpoint_SHA256',
            'execution_procedure')},
        'no_delay_caveat': 'One same-process sequential20ms pair only, no complete0.5s no-delay benchmark',
        'delay_fixture_regression': {'publication_calls_each': 98, 'flat_calls_original_patched': [200, 200],
            'observer_and_complete_state_equal': True},
        'paused_future_work': 'State/Quantity buffers and same-state pure-evaluate cache; not implemented or tested',
    }
    destination = ROOT / 'docs/validation_20261018.json'
    destination.write_bytes((json.dumps(document, indent=2, ensure_ascii=False) + '\n').encode('utf8'))
    print(destination.relative_to(ROOT))


if __name__ == '__main__':
    main()
