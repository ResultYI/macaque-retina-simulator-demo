"""Strictly reuse one completed video's exact, already bound input packet."""
from pathlib import Path
import json
from retina_sim.contracts import canonical_hash
from retina_sim.disk_graph_storage import file_sha
from retina_sim.streamed_input import load_streamed_input


def prepared_input_binding(root, directory, config, identity):
    root = Path(root).resolve()
    directory = (root / directory).resolve()
    if not directory.is_relative_to(root):
        raise ValueError('Prepared video input must remain inside the project')
    manifest_path, trial_path = directory / 'manifest.json', directory / 'trial.json'
    manifest, trial = json.loads(manifest_path.read_bytes()), json.loads(trial_path.read_bytes())
    if trial['status'] != 'COMPLETED' or trial['identity'] != manifest['identity']:
        raise ValueError('Prepared input requires a completed, consistently identified source run')
    previous = manifest['identity']
    for key in ('media_sha256', 'start_seconds', 'duration_seconds', 'source_manifest_sha256', 'compiler',
                'execution_environment', 'route', 'new_functional_HC_acceptance'):
        if previous[key] != identity[key]:
            raise ValueError('Prepared input source identity differs: ' + key)
    for key in ('mode', 'source_directory', 'checkpoint_sha256', 'configuration_sha256', 'selector_sha256'):
        if previous['background_start'][key] != identity['background_start'][key]:
            raise ValueError('Prepared input background source differs: ' + key)
    actual_config = config.to_dict()
    if canonical_hash(manifest['config']) != canonical_hash(actual_config):
        different = [key for key in set(manifest['config']) | set(actual_config)
                     if canonical_hash(manifest['config'].get(key)) != canonical_hash(actual_config.get(key))]
        raise ValueError('Prepared input configuration differs: ' + ','.join(sorted(different)))
    packet = directory / 'input_stream_shards/manifest.json'
    packet_sha = file_sha(packet)
    if packet_sha != manifest['input_assumptions']['source_sha256']:
        raise ValueError('Prepared source input packet differs from the completed run')
    return {'schema': 'strict_completed_video_input_reuse_v1', 'source_directory': str(directory),
            'source_manifest_sha256': file_sha(manifest_path), 'source_trial_sha256': file_sha(trial_path),
            'input_manifest_sha256': packet_sha, 'configuration_hash': canonical_hash(actual_config),
            'assembly_hash': manifest['assembly_hash'], 'model_hash': manifest['runtime']['model_hash']}


def load_prepared_input(binding, compiled, model_hash, config):
    directory = Path(binding['source_directory'])
    for path, expected in ((directory / 'manifest.json', binding['source_manifest_sha256']),
                           (directory / 'trial.json', binding['source_trial_sha256']),
                           (directory / 'input_stream_shards/manifest.json', binding['input_manifest_sha256'])):
        if file_sha(path) != expected:
            raise ValueError('Prepared source changed after admission: ' + str(path))
    if (compiled.assembly_hash != binding['assembly_hash'] or model_hash != binding['model_hash'] or
            canonical_hash(config.to_dict()) != binding['configuration_hash']):
        raise ValueError('Prepared input actual model/configuration differs')
    packet, stores = load_streamed_input(directory / 'input_stream_shards/manifest.json', compiled,
        config.light_context(), reference_id=compiled.global_disk_circuit.record['reference_id'])
    summary = dict(packet.source_summary, actual_ISET_call=False, cache_reused=True,
                   prepared_input_reused=True, prepared_source_directory=str(directory),
                   validation='Original strict loader: every geometry, binding, shard hash, value, unit and reference check retained')
    return packet, stores, summary
