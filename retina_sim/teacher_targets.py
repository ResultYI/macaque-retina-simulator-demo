"""Strict external-teacher rate import and fixed full-window scoring.

NPZ is the interchange boundary: exporters must bind actual teacher IDs to
model cell IDs explicitly.  This module never infers spikes from analog output,
reorders cells, interpolates times, estimates lag, fits a scale, or selects a
candidate.  A teacher score is not an independent physiological validation.
"""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import json
import re
import zipfile
import numpy as np

from .contracts import ContractError, canonical_hash

TARGET_SCHEMA = 'teacher-target-1.0'
PREDICTION_SCHEMA = 'teacher-rate-prediction-1.0'
SCORE_SCHEMA = 'teacher-fixed-score-1.0'
SPLIT_SCHEMA = 'teacher-split-validation-1.0'
TEACHER_ROLES = frozenset(('teacher_feature_fit', 'teacher_heldout'))
TARGET_QUANTITIES = {'repeat_spike_counts': 'count/bin', 'repeat_mean_rate': 'Hz'}
PREDICTION_QUANTITIES = frozenset(('expected_discrete_count_rate', 'repeat_mean_rate'))
BINNING = 'left_closed_right_open'


def _require(condition, message):
    if not condition:
        raise ContractError(message)


def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()), name + ' must be an explicit nonempty string')
    return value


def _digest(value, name, size=64):
    _require(isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{%d}' % size, value) is not None,
             name + ' must be a fixed %d-character hexadecimal identity' % size)
    return value.lower()


def _number(value, name, *, positive=False, nonnegative=False):
    _require(type(value) in (int, float) and np.isfinite(value), name + ' must be finite numeric metadata')
    _require(not positive or value > 0, name + ' must be positive')
    _require(not nonnegative or value >= 0, name + ' must be nonnegative')
    return float(value)


def _role(role):
    if role == 'real_independent_test':
        raise ContractError('A teacher target cannot replace real_independent_test; use independent real data')
    _require(isinstance(role, str) and role in TEACHER_ROLES,
             'Teacher role must be teacher_feature_fit or teacher_heldout')
    return role


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ContractError('Nonfinite JSON constant: ' + value)


def read_json_contract(path):
    """Read strict JSON and return its actual byte SHA256 alongside the object."""
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_json_object,
                           parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError('Invalid UTF-8 JSON contract: ' + str(path)) from exc
    _require(isinstance(value, dict), 'Contract JSON must contain an object')
    return value, hashlib.sha256(raw).hexdigest()


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ContractError('Contract metadata must be finite JSON without Python objects') from exc


def _repeat_inventory(manifest):
    count = manifest.get('repeat_count')
    ids = manifest.get('repeat_ids')
    _require(type(count) is int and count >= 2, 'A PSTH requires repeat_count >= 2; one spike train is not a PSTH')
    _require(isinstance(ids, list) and len(ids) == count and
             all(isinstance(x, str) and x.strip() for x in ids) and len(set(ids)) == count,
             'Explicit unique repeat_ids must match repeat_count')
    return count, tuple(ids)


def _bin_edges(contract):
    raw = contract.get('bin_edges_seconds')
    _require(isinstance(raw, list) and len(raw) >= 2 and
             all(type(x) in (int, float) for x in raw), 'Explicit bin_edges_seconds must be a numeric list')
    edges = np.asarray(raw, dtype=float)
    _require(np.all(np.isfinite(edges)) and np.all(np.diff(edges) > 0),
             'Bin edges must be finite, strictly increasing timestamps in seconds')
    _require(contract.get('time_unit') == 's', 'Timestamp unit must be explicitly s')
    edges.setflags(write=False)
    return edges


def _validate_cells(manifest):
    cells = manifest.get('cells')
    producer = manifest['producer']
    _require(isinstance(cells, list) and bool(cells), 'Explicit ordered cells are required')
    ids = []; teacher_ids = []
    for row in cells:
        _require(isinstance(row, dict), 'Each cell mapping must be an object')
        ids.append(_text(row.get('cell_id'), 'cell_id'))
        teacher_ids.append(_text(row.get('teacher_cell_id'), 'teacher_cell_id'))
        _text(row.get('model_target_type'), 'model_target_type')
        eccentricity = _number(row.get('eccentricity_degrees'), 'eccentricity_degrees', nonnegative=True)
        label = row.get('domain_label')
        if producer == 'synthetic_fixture':
            _require(label == 'synthetic_fixture_only', 'Synthetic cells must be labeled synthetic_fixture_only')
        elif eccentricity > 0:
            _require(label == 'unvalidated_eccentricity_extrapolation',
                     'Nonzero-eccentricity teacher use requires unvalidated_eccentricity_extrapolation')
        else:
            _require(label in ('teacher_central_model_only', 'unvalidated_eccentricity_extrapolation'),
                     'Central teacher cells need an explicit limited-domain label')
    _require(len(set(ids)) == len(ids) and len(set(teacher_ids)) == len(teacher_ids),
             'Model and teacher cell IDs must be unique; implicit order mapping is forbidden')
    _require(manifest.get('cell_mapping_method') == 'explicit_id_mapping',
             'Teacher-to-model cells require explicit_id_mapping, never guessed ordering')
    return tuple(ids)


def validate_teacher_manifest(manifest):
    """Validate metadata only; load_teacher_target additionally verifies NPZ bytes."""
    _require(isinstance(manifest, dict), 'Teacher manifest must be an object')
    _require(manifest.get('schema_version') == TARGET_SCHEMA, 'Unknown teacher target schema')
    _role(manifest.get('role'))
    producer = manifest.get('producer')
    _require(producer in ('virtual_retina', 'synthetic_fixture'), 'Explicit virtual_retina or synthetic_fixture producer required')
    identity = manifest.get('producer_identity')
    _require(isinstance(identity, dict), 'Explicit producer_identity is required')
    if producer == 'virtual_retina':
        _digest(identity.get('commit'), 'Virtual Retina commit', 40)
        _digest(identity.get('configuration_sha256'), 'Virtual Retina configuration_sha256')
        command = identity.get('command')
        executable = identity.get('executable_sha256')
        if executable is not None:
            _digest(executable, 'executable_sha256')
        command_ok = isinstance(command, list) and bool(command) and all(isinstance(x, str) and x for x in command)
        _require(command_ok or executable is not None,
                 'Virtual Retina needs fixed configuration plus a nonempty command or executable identity')
    else:
        _text(identity.get('fixture_id'), 'synthetic fixture_id')
    _digest(manifest.get('stimulus_hash'), 'stimulus_hash')
    _digest(manifest.get('input_condition_hash'), 'input_condition_hash')
    _text(manifest.get('stimulus_group'), 'stimulus_group')
    _text(manifest.get('experiment_id'), 'experiment_id')
    _number(manifest.get('pixels_per_degree'), 'pixels_per_degree', positive=True)
    limits = manifest.get('input_luminosity_range')
    _require(isinstance(limits, list) and len(limits) == 2, 'Explicit input_luminosity_range [minimum, maximum] required')
    lo = _number(limits[0], 'minimum input luminosity')
    hi = _number(limits[1], 'maximum input luminosity')
    _require(lo < hi, 'Input luminosity range must increase')
    _text(manifest.get('input_luminosity_unit'), 'input_luminosity_unit')
    _number(manifest.get('source_dt_seconds'), 'source_dt_seconds', positive=True)
    _repeat_inventory(manifest)
    _bin_edges(manifest)
    _validate_cells(manifest)
    quantity = manifest.get('quantity')
    _require(isinstance(quantity, str) and quantity in TARGET_QUANTITIES,
             'Teacher quantity must be repeat_spike_counts or repeat_mean_rate; analog -tr0/LIF voltage is forbidden')
    _require(manifest.get('unit') == TARGET_QUANTITIES[quantity], 'Teacher quantity has missing or incompatible units')
    data = manifest.get('data')
    _require(isinstance(data, dict) and data.get('key') == quantity, 'NPZ data key must explicitly equal the teacher quantity')
    _digest(data.get('sha256'), 'teacher NPZ sha256')
    reconstruction = manifest.get('reconstruction')
    _require(isinstance(reconstruction, dict), 'Actual reconstruction metadata is required')
    _text(reconstruction.get('tool'), 'reconstruction tool')
    _text(reconstruction.get('version'), 'reconstruction version')
    _require(isinstance(reconstruction.get('parameters'), dict), 'Actual reconstruction parameters must be recorded, including an empty object')
    source_units = {'spike_events': 's', 'repeat_spike_counts': 'count/bin', 'repeat_mean_rate': 'Hz'}
    source = reconstruction.get('source_quantity')
    _require(isinstance(source, str) and source in source_units and reconstruction.get('source_unit') == source_units[source],
             'Reconstruction must start from declared spike events/counts/rates, never analog or normalized LIF voltage')
    _require(quantity != 'repeat_spike_counts' or source != 'repeat_mean_rate',
             'A repeat-mean rate cannot be relabeled as individual repeat spike counts')
    _require(reconstruction.get('output_unit') == manifest['unit'], 'Reconstruction output unit must match imported data')
    latency = _number(reconstruction.get('latency_seconds'), 'Recorded reconstruction latency_seconds')
    policy = reconstruction.get('latency_policy')
    _require(policy in ('none', 'fixed_recorded_offset'), 'Latency policy must be none or fixed_recorded_offset')
    _require(policy != 'none' or latency == 0, 'Latency policy none requires latency_seconds=0')
    _require(reconstruction.get('binning') == BINNING,
             'Every spike bin must be left_closed_right_open, including the final bin')
    _require(reconstruction.get('source_output') != '-tr0', 'Analog -tr0 output is not a spike-rate target')
    _json_copy(manifest)
    return manifest


def _load_npz(path, data_contract):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    _require(actual == _digest(data_contract.get('sha256'), 'NPZ sha256'), 'Actual NPZ data hash mismatch')
    key = data_contract['key']
    try:
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            _require(set(archive.files) == {key}, 'NPZ must contain exactly the declared numeric array')
            values = archive[key]
            _require(values.dtype.kind in 'iuf' and values.dtype.kind != 'b', 'NPZ array must be numeric, without objects or booleans')
            values = np.array(values, dtype=float, copy=True)
    except (ValueError, TypeError, OSError, EOFError, KeyError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError('NPZ must be readable numeric arrays with allow_pickle=False') from exc
    _require(np.all(np.isfinite(values)) and np.all(values >= 0), 'Imported counts/rates must be finite and nonnegative')
    values.setflags(write=False)
    return values, actual


@dataclass(frozen=True)
class TeacherTarget:
    manifest: dict
    manifest_sha256: str
    data_sha256: str
    mean_rate_hz: np.ndarray
    bin_edges_seconds: np.ndarray
    cell_ids: tuple
    repeat_count: int
    role: str
    identity: str
    metadata_hash: str
    rate_hash: str


@dataclass(frozen=True)
class RatePrediction:
    contract: dict
    contract_sha256: str
    data_sha256: str
    rate_hz: np.ndarray
    identity: str
    metadata_hash: str
    rate_hash: str


def load_teacher_target(manifest_path, data_path, *, role=None):
    manifest, manifest_sha = read_json_contract(manifest_path)
    validate_teacher_manifest(manifest)
    if role is not None:
        _require(_role(role) == manifest['role'], 'Requested teacher role differs from target role')
    edges = _bin_edges(manifest); cells = _validate_cells(manifest)
    repeat_count, _ = _repeat_inventory(manifest)
    values, data_sha = _load_npz(data_path, manifest['data'])
    shape = (len(edges) - 1, len(cells))
    if manifest['quantity'] == 'repeat_spike_counts':
        _require(values.shape == (repeat_count,) + shape, 'Repeat spike counts must have shape (repeat,time,cell)')
        _require(np.all(values == np.floor(values)), 'Count/bin values must be integer event counts; multiple events are allowed')
        rate = np.mean(values, axis=0) / np.diff(edges)[:, None]
    else:
        _require(values.shape == shape, 'Repeat-mean rate must have shape (time,cell) in Hz')
        rate = values.copy()
    _require(np.all(np.isfinite(rate)), 'Count-to-Hz conversion produced nonfinite rates')
    rate.setflags(write=False)
    identity = canonical_hash(dict(manifest_sha256=manifest_sha, data_sha256=data_sha,
                                   schema_version=TARGET_SCHEMA))
    return TeacherTarget(manifest, manifest_sha, data_sha, rate, edges, cells, repeat_count,
                         manifest['role'], identity, canonical_hash(manifest), canonical_hash(rate))


def _target_unchanged(target):
    _require(isinstance(target, TeacherTarget), 'A verified TeacherTarget is required')
    _require(canonical_hash(target.manifest) == target.metadata_hash and
             canonical_hash(target.mean_rate_hz) == target.rate_hash,
             'Loaded teacher metadata or rates changed after identity verification')
    _require(np.array_equal(target.bin_edges_seconds, np.asarray(target.manifest['bin_edges_seconds'])) and
             target.cell_ids == tuple(row['cell_id'] for row in target.manifest['cells']) and
             target.role == target.manifest['role'] and target.repeat_count == target.manifest['repeat_count'],
             'Loaded target timestamps, cells, repeats or role changed after identity verification')


def load_prediction(contract_path, predictions_path, target):
    _target_unchanged(target)
    contract, contract_sha = read_json_contract(contract_path)
    _require(contract.get('schema_version') == PREDICTION_SCHEMA, 'Unknown prediction schema')
    _require(isinstance(contract.get('quantity'), str) and
             contract['quantity'] in PREDICTION_QUANTITIES and contract.get('unit') == 'Hz',
             'Prediction must explicitly be expected_discrete_count_rate or repeat_mean_rate in Hz; conditional intensity and voltage are forbidden')
    _require(_role(contract.get('role')) == target.role, 'Prediction role differs from teacher target role')
    _require(_digest(contract.get('target_manifest_sha256'), 'target_manifest_sha256') == target.manifest_sha256 and
             _digest(contract.get('target_data_sha256'), 'target_data_sha256') == target.data_sha256,
             'Prediction is bound to different target manifest/data identities')
    for key in ('stimulus_hash', 'input_condition_hash'):
        _require(contract.get(key) == target.manifest[key], 'Prediction ' + key + ' differs; no condition remapping')
    _require(contract.get('cell_ids') == list(target.cell_ids), 'Prediction cell IDs/order differ; no reordering')
    edges = _bin_edges(contract)
    _require(np.array_equal(edges, target.bin_edges_seconds), 'Prediction timestamps/bin edges differ; no interpolation or lag search')
    _require(contract.get('binning') == BINNING == target.manifest['reconstruction']['binning'],
             'Prediction and target require identical left_closed_right_open bins, including the final bin')
    model = contract.get('model_identity')
    _require(isinstance(model, dict), 'Explicit prediction model_identity is required')
    _digest(model.get('model_hash'), 'prediction model_hash')
    _digest(model.get('parameter_hash'), 'prediction parameter_hash')
    if contract['quantity'] == 'repeat_mean_rate':
        _repeat_inventory(contract)
    data = contract.get('data')
    _require(isinstance(data, dict) and data.get('key') == contract['quantity'], 'Prediction NPZ key must equal its declared quantity')
    rates, data_sha = _load_npz(predictions_path, data)
    _require(rates.shape == target.mean_rate_hz.shape, 'Prediction must have exact (time,cell) target shape')
    _json_copy(contract)
    identity = canonical_hash(dict(contract_sha256=contract_sha, data_sha256=data_sha,
                                   schema_version=PREDICTION_SCHEMA))
    return RatePrediction(contract, contract_sha, data_sha, rates, identity,
                          canonical_hash(contract), canonical_hash(rates))


def validate_teacher_splits(manifests):
    """Validate a supplied full inventory; whole stimulus groups stay in one role.

    This validates declared experiment membership, not the completeness of an
    external directory. Repeated aliases of identical stimulus/condition hashes
    cannot evade the role separation.
    """
    _require(isinstance(manifests, (list, tuple)) and bool(manifests), 'A nonempty teacher split inventory is required')
    groups = {}; stimuli = {}; identities = []
    for item in manifests:
        if isinstance(item, TeacherTarget):
            _target_unchanged(item); row = item.manifest; identity = item.manifest_sha256
        elif isinstance(item, (str, Path)):
            row, identity = read_json_contract(item)
        else:
            row = _json_copy(item); identity = canonical_hash(row)
        validate_teacher_manifest(row)
        role = row['role']; group = row['stimulus_group']
        repeats = tuple(sorted(row['repeat_ids']))
        if group in groups:
            _require(groups[group]['role'] == role, 'Complete stimulus_group repeats leak across fit/heldout: ' + group)
            _require(groups[group]['repeat_ids'] == repeats, 'Complete stimulus_group repeat inventory differs: ' + group)
        else:
            groups[group] = dict(role=role, repeat_ids=repeats)
        stimulus = (row['stimulus_hash'], row['input_condition_hash'])
        _require(stimulus not in stimuli or stimuli[stimulus] == role,
                 'Identical stimulus/condition repeats leak across fit/heldout under different group names')
        stimuli[stimulus] = role; identities.append(identity)
    return dict(schema_version=SPLIT_SCHEMA, supplied_inventory_valid=True,
                partition_unit='complete_stimulus_group', groups=groups,
                manifest_identities=identities, external_inventory_completeness='DECLARED_NOT_INDEPENDENTLY_VERIFIED')


def _mean(values):
    scale = float(np.max(np.abs(values)))
    return 0. if scale == 0 else scale * float(np.mean(values / scale))


def _rms(values):
    scale = float(np.max(np.abs(values)))
    return 0. if scale == 0 else scale * float(np.sqrt(np.mean((values / scale) ** 2)))


def _pearson(target, prediction):
    if np.ptp(target) == 0 or np.ptp(prediction) == 0:
        return None
    a = target - _mean(target); b = prediction - _mean(prediction)
    a = a / np.max(abs(a)); b = b / np.max(abs(b))
    return float(np.clip(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)), -1., 1.))


def _center(values):
    return np.zeros_like(values) if np.ptp(values) == 0 else values - _mean(values)


def _metrics(target, prediction):
    error = prediction - target
    centered_target = _center(target)
    centered_prediction = _center(prediction)
    modulation_error = _rms(centered_prediction - centered_target)
    std = _rms(centered_target)
    return dict(mean_error_hz=_mean(error), rmse_hz=_rms(error), pearson=_pearson(target, prediction),
                centered_modulation_rmse_hz=modulation_error, target_modulation_std_hz=std,
                centered_modulation_rmse_over_target_std=None if std == 0 else modulation_error / std)


def score_teacher_target(target, prediction, *, role):
    """Score every declared bin/cell once with no tunable alignment or scaling."""
    _target_unchanged(target)
    _require(_role(role) == target.role, 'Score role differs from target role')
    _require(isinstance(prediction, RatePrediction), 'A verified Hz RatePrediction is required')
    _require(canonical_hash(prediction.contract) == prediction.metadata_hash and
             canonical_hash(prediction.rate_hz) == prediction.rate_hash,
             'Loaded prediction metadata or rates changed after identity verification')
    _require(_digest(prediction.contract['target_manifest_sha256'], 'target_manifest_sha256') == target.manifest_sha256 and
             _digest(prediction.contract['target_data_sha256'], 'target_data_sha256') == target.data_sha256,
             'Prediction identity belongs to another target')
    actual = target.mean_rate_hz; predicted = prediction.rate_hz
    per_cell = [dict(cell_id=cid, **_metrics(actual[:, i], predicted[:, i]))
                for i, cid in enumerate(target.cell_ids)]
    correlations = [row['pearson'] for row in per_cell if row['pearson'] is not None]
    centered_actual = np.column_stack([_center(actual[:, i]) for i in range(len(target.cell_ids))])
    centered_predicted = np.column_stack([_center(predicted[:, i]) for i in range(len(target.cell_ids))])
    std = _rms(centered_actual); modulation_error = _rms(centered_predicted - centered_actual)
    aggregate = dict(mean_error_hz=_mean(predicted - actual), rmse_hz=_rms(predicted - actual),
        pearson=None if not correlations else float(np.mean(correlations)),
        pearson_aggregation='mean_of_defined_per_cell_Pearson', defined_pearson_cells=len(correlations),
        centered_modulation_rmse_hz=modulation_error, target_modulation_std_hz=std,
        centered_modulation_rmse_over_target_std=None if std == 0 else modulation_error / std)
    unvalidated = ['physiological_validity_not_established', 'real_independent_test_not_performed',
                   'split_inventory_not_verified', 'external_producer_and_reconstruction_declarations_not_executed_or_independently_verified',
                   'ON_OFF_peak_and_latency_features_not_scored']
    labels = sorted({row['domain_label'] for row in target.manifest['cells']})
    unvalidated.extend(labels)
    result = dict(schema_version=SCORE_SCHEMA, role=role, producer=target.manifest['producer'],
        producer_identity=target.manifest['producer_identity'], reconstruction=target.manifest['reconstruction'],
        target_quantity=target.manifest['quantity'], prediction_quantity=prediction.contract['quantity'], rate_unit='Hz',
        cells=target.manifest['cells'],
        input_metadata={key: target.manifest[key] for key in
            ('pixels_per_degree', 'input_luminosity_range', 'input_luminosity_unit', 'source_dt_seconds')},
        target_identity=target.identity, target_manifest_sha256=target.manifest_sha256,
        target_data_sha256=target.data_sha256, prediction_identity=prediction.identity,
        prediction_contract_sha256=prediction.contract_sha256, prediction_data_sha256=prediction.data_sha256,
        model_identity=prediction.contract['model_identity'], stimulus_hash=target.manifest['stimulus_hash'],
        input_condition_hash=target.manifest['input_condition_hash'], cell_ids=list(target.cell_ids),
        repeat_count=target.repeat_count, bin_edges_seconds=target.bin_edges_seconds.tolist(),
        per_cell=per_cell, aggregate=aggregate, domain_labels=labels, unvalidated=unvalidated,
        score_policy=dict(window='all_declared_bins', weighting='uniform_bins_and_cells', cell_order='exact',
            timestamp_alignment='exact_bin_edges_seconds', binning=BINNING, interpolation=False, lag_search=False,
            scaling=False, window_selection=False, parameter_search=False, candidate_selection=False),
        physiological_validation='NOT_PERFORMED', claim='Fixed teacher-feature score only; no physiological pass decision')
    return _json_copy(result)
