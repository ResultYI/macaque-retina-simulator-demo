"""Explicit video initialization choices; checkpoint identity checks always apply."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json


def add_background_arguments(parser):
    parser.add_argument('--background-mode', choices=('strict', 'transient'), default='strict',
                        help='strict requires convergence; transient is an explicitly unadapted throughput test')
    parser.add_argument('--background-source', type=Path,
                        help='Directory containing configuration.json and background_checkpoint.json')
    parser.add_argument('--restore-diagnostics', action='store_true',
                        help='Capture field hashes before backend setup and inside the unchanged strict loader')


def select_background_source(root, original, configuration, mode, source):
    if mode not in ('strict', 'transient'):
        raise ValueError('Unknown background initialization mode')
    if source is None:
        if mode != 'strict':
            raise ValueError('Transient mode requires an explicit --background-source')
        return configuration, original/'matched_response/background_checkpoint.json', None
    source = (root/source).resolve()
    checkpoint = source/'background_checkpoint.json'
    config_path = source/'configuration.json'
    if not source.is_relative_to(root.resolve()):
        raise ValueError('Background source must stay inside this project')
    config_bytes = config_path.read_bytes()
    configuration = json.loads(config_bytes)
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return configuration, checkpoint, {
        'mode': mode, 'source_directory': str(source), 'checkpoint_sha256': digest,
        'configuration_sha256': hashlib.sha256(config_bytes).hexdigest(),
    }


def restore_background_start(store, checkpoint, mode, *, diagnostic_path=None):
    """Relax only the convergence certificate in explicit transient mode."""
    from retina_sim.background import NUMERICAL_UNIT_SCALES, _history_consistency
    from retina_sim.contracts import ContractError
    import numpy as np

    if mode not in ('strict', 'transient'):
        raise ValueError('Unknown background initialization mode')
    state, envelope = store.load(checkpoint, require_converged=mode == 'strict', diagnostic_path=diagnostic_path)
    report = envelope.get('report') or {}
    if mode == 'transient':
        if report.get('status') != 'NOT_REACHED' or report.get('converged') is not False:
            raise ContractError('Transient mode requires an honestly unqualified NOT_REACHED checkpoint')
        coverage = store.coverage()
        if (coverage['missing'] or coverage['covered_blocks'] != coverage['state_blocks'] or
                report.get('uncovered_states') or report.get('checked_states') != coverage['states']):
            raise ContractError('Transient initialization requires complete registered state coverage')
        criteria = {k: v for k, v in asdict(store.policy).items() if k not in
                    ('max_wall_seconds', 'max_sim_seconds', 'max_observations', 'require_converged')}
        if (any(report.get('policy', {}).get(k) != v for k, v in criteria.items()) or
                report.get('numerical_unit_registry') != NUMERICAL_UNIT_SCALES):
            raise ContractError('Transient mode cannot alter background criteria or unit scales')
        if any(not np.isfinite(a).all() for a in state.ordered_state_arrays):
            raise ContractError('Transient checkpoint contains nonfinite registered states')
        failures = [x for x in _history_consistency(state, store.runner, store.policy)
                    if not x.endswith(':queue_not_constant_within_tolerance')]
        if failures or any(w.get('domain_failures') for w in report.get('windows', [])[-1:]):
            raise ContractError('Transient checkpoint has invariant/history failures: '+str(failures))
    annotation = {
        'mode': mode, 'status': report.get('status'), 'converged': report.get('converged'),
        'convergence_certificate_required': mode == 'strict',
        'background_seconds': state.time_seconds-envelope['preparation_origin_seconds'],
        'checkpoint': str(checkpoint), 'identity_checks': 'all original binding/hash/state checks retained',
        'whole_network_steady_state_claim': False,
        'test_scope': ('Functional pipeline and throughput at this transient initial state only; '
                       'not an adequately adapted physiological response') if mode == 'transient'
                      else 'Original sampled deterministic background convergence certificate required',
    }
    return state, envelope, annotation


def begin_media_stimulus(state, source_sha256, origin, initialization):
    """Switch external stimulus phase without resetting any dynamic or causal state."""
    if state.input_position.get('phase') not in ('uniform_background', 'media_stimulus'):
        raise ValueError('Video requires an explicitly restored background or media phase')
    state.input_position.update(
        phase='media_stimulus', stimulus_origin_seconds=origin,
        input_source_sha256=source_sha256,
        recorded_window_count=state.input_position.get('recorded_window_count', 0),
        video_initialization_mode=initialization['mode'],
    )
    if initialization['status'] is not None:
        state.input_position['background_initialization_status'] = initialization['status']
