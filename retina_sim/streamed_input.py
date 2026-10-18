"""Bounded storage for the existing held 100-us global receptor input.

Counts remain counts: each original binary64 window width is used at readout.
The optical exporter, reference bridge, author clocks and neural state are not
changed. A strict one-window global packet supplies the existing channel and
geometry contracts; all additional numeric shards are independently bound.
"""
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from .contracts import ContractError, Quantity
from .disk_graph_storage import file_sha
from .disk_input_packet import load_global_input
from .geometry_identity import compiled_geometry_identity
from .isetbio_input import reference_equivalent_drive
from .offline_input import OfflineInputs, unique_json

SCHEMA = 'bounded_global_receptor_input_v1'
BUDGET = 128 * 1024**2


class RowStore:
    """Immutable numeric shards with a byte-bounded LRU, never time x all cones."""
    def __init__(self, root, records, row_map, columns, *, cache_bytes):
        self.root = Path(root).resolve()
        self.records = records
        self.cache_bytes = int(cache_bytes)
        self.cache = OrderedDict()
        self.resident_bytes = self.peak_resident_bytes = 0
        self.columns = columns
        self.mapping = np.asarray(row_map)
        if self.mapping.dtype.kind not in 'iu' or self.mapping.ndim != 2 or self.mapping.shape[1] != 2:
            raise ContractError('Stream row map must contain integer shard/row pairs')
        if not records or np.any(self.mapping < 0) or np.any(self.mapping[:, 0] >= len(records)):
            raise ContractError('Stream row map is outside its frozen shards')
        self.paths = []
        for index, record in enumerate(records):
            path = (self.root / record['path']).resolve()
            if not path.is_relative_to(self.root):
                raise ContractError('Stream shard outside run-owned input directory')
            shape = tuple(record['shape'])
            if len(shape) != 2 or shape[1] != columns or shape[0] < 1:
                raise ContractError('Stream shard cell/time shape differs')
            if np.prod(shape, dtype=object) * 8 > min(BUDGET, self.cache_bytes):
                raise ContractError('One stream shard exceeds its resident capture budget')
            selected = self.mapping[self.mapping[:, 0] == index, 1]
            if len(selected) and np.any(selected >= shape[0]):
                raise ContractError('Stream row map exceeds declared shard rows')
            self.paths.append(path)
            self._read(index)  # Validate every shard, not just the first frame.

    def _read(self, index):
        path, record = self.paths[index], self.records[index]
        if path.stat().st_size > BUDGET or file_sha(path) != record['sha256']:
            raise ContractError('Frozen input shard bytes differ or exceed capture budget')
        # mmap first prevents a forged header from requesting an eager allocation.
        array = np.load(path, allow_pickle=False, mmap_mode='r')
        if array.dtype != np.dtype('float64') or array.shape != tuple(record['shape']):
            raise ContractError('Stream shard requires its declared binary64 layout')
        for start in range(0, len(array), 32):
            block = array[start:start + 32]
            if not np.all(np.isfinite(block)) or np.any(block < 0):
                raise ContractError('Stream shard must contain finite nonnegative input')
        result = np.array(array, copy=True)
        result.setflags(write=False)
        return result

    def row(self, index):
        shard, row = map(int, self.mapping[index])
        if shard not in self.cache:
            value = self._read(shard)
            while self.cache and self.resident_bytes + value.nbytes > self.cache_bytes:
                _, removed = self.cache.popitem(last=False)
                self.resident_bytes -= removed.nbytes
            self.cache[shard] = value
            self.resident_bytes += value.nbytes
            self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
        self.cache.move_to_end(shard)
        return self.cache[shard][row]


def held_channel(template, edges, store, *, counts=False, alpha=None):
    """Same searchsorted(right), endpoint clamping and operation order as v1."""
    edges = np.asarray(edges, dtype=np.float64)
    if (edges.ndim != 1 or len(edges) < 2 or edges[0] != 0 or
            not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0) or
            not np.allclose(np.diff(edges), .0001, rtol=0, atol=1e-12)):
        raise ContractError('Stream requires the original increasing 100-us input axis')
    if len(store.mapping) != len(edges) - int(counts):
        raise ContractError('Stream window/sample count differs from its row map')
    if template.interpolation != 'zero_order_hold':
        raise ContractError('This storage adapter only supports the actual held disk input')
    widths = np.diff(edges)
    if counts:
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (store.columns,) or not np.all(np.isfinite(alpha)) or np.any(alpha <= 0):
            raise ContractError('Fixed positive LMS bridge coefficients required')
    last = [None, None]
    def evaluate(t):
        t = float(np.clip(t, 0., edges[-1]))
        index = min(max(int(np.searchsorted(edges, t, side='right') - 1), 0), len(store.mapping) - 1)
        if last[0] != index:
            value = store.row(index)
            if counts:
                # Eager v1 does counts / np.diff(axis), then rates / alpha.
                # Dividing once by nominal 1e-4 or multiplying a rescale is NOT equivalent.
                value = (value[None, :] / widths[index]) / alpha[None, :]
                value = value[0]
            last[:] = [index, value]
        return Quantity(last[1], template.unit, template.meaning, template.evidence_tag)
    contract = dict(template.input_contract, stop_seconds=float(edges[-1]),
                    storage_schema=SCHEMA, source_grid_step_seconds=.0001,
                    integration_windows_unchanged=True)
    if counts:
        contract['conversion'] = dict(contract['conversion'], integration_window_edges_seconds=edges.tolist())
    return replace(template, start_seconds=0., stop_seconds=float(edges[-1]), evaluator=evaluate,
                   input_contract=contract, breakpoints_seconds=tuple(map(float, edges[1:-1])))


def load_streamed_input(path, compiled, light, *, reference_id):
    path = Path(path).resolve()
    if path.stat().st_size > 4 * 1024**2:
        raise ContractError('Stream manifest exceeds bounded metadata budget')
    manifest = json.loads(path.read_bytes(), object_pairs_hook=unique_json,
        parse_constant=lambda value: (_ for _ in ()).throw(ContractError('Nonfinite stream metadata '+value)))
    if manifest.get('schema') != SCHEMA or manifest.get('version') != '1.0':
        raise ContractError('Unknown global streamed input version')
    if (manifest['global_graph'] != compiled.global_disk_circuit.binding or
            manifest['geometry_identity'] != compiled_geometry_identity(compiled) or
            manifest['background_context_hash'] != light.background_context_hash or
            manifest['reference_id'] != reference_id):
        raise ContractError('Stream graph, geometry, background or reference differs')
    baseline = (path.parent / manifest['contract_packet']['path']).resolve()
    if not baseline.is_relative_to(path.parent) or file_sha(baseline) != manifest['contract_packet']['sha256']:
        raise ContractError('Stream strict contract packet differs')
    bundle = load_global_input(baseline, compiled, light, reference_id=reference_id)
    for record in manifest['exposure_records']:
        evidence = (path.parent / record['path']).resolve()
        if not evidence.is_relative_to(path.parent) or evidence.stat().st_size > 4*1024**2 or file_sha(evidence) != record['sha256']:
            raise ContractError('Stream optical exposure provenance changed')
    with np.load(baseline, allow_pickle=False) as data:
        contract = json.loads(data['manifest_json'].item())
    native = contract['channels']['native_light']
    types = compiled.populations['cones'].cells.subtype
    profile = native['reference_profile']
    # Validate the exact existing reference bridge before reusing its coefficients.
    _, bridge = reference_equivalent_drive(np.zeros((1, len(types))), types, profile,
        capture_scope_sha256=native['capture_scope_sha256'])
    alpha = np.array([bridge['alpha_rstar_s_per_reference_td_LMS'][kind] for kind in types])
    edges = np.asarray(manifest['edges_seconds'], dtype=np.float64)
    channels, stores = {}, {}
    if set(manifest['channels']) != set(bundle.channels):
        raise ContractError('Stream must supply exactly the compiled input channels')
    for name, template in bundle.channels.items():
        record = manifest['channels'][name]
        external = next(e for e in compiled.spec.external_inputs if e.channel == name)
        columns = len(compiled.components[external.target_component].inputs[external.target_port].cell_ids)
        native_channel = name == 'native_light'
        if record['semantics'] != ('expected_Rstar_per_original_window' if native_channel else 'sampled_original_rod_Rstar_s'):
            raise ContractError('Stream storage cannot change input semantics')
        store = RowStore(path.parent, record['shards'], record['row_map'], columns,
                         cache_bytes=(48 if native_channel else 16) * 1024**2)
        stores[name] = store
        channels[name] = held_channel(template, edges, store, counts=native_channel, alpha=alpha if native_channel else None)
    digest = file_sha(path)
    for channel in channels.values():
        channel.input_contract.update(source_sha256=digest, producer=manifest['producer'])
    summary = dict(bundle.source_summary, schema=SCHEMA, source_file=str(path), source_sha256=digest,
        producer=manifest['producer'], storage='exact count dictionary + original 100us index + bounded LRU',
        channels={name:dict(channel.input_contract) for name,channel in channels.items()},
        duration_seconds=float(edges[-1]), input_window_count=len(edges)-1,
        resident_shard_cache_limit_bytes=64*1024**2, physiological_or_numerical_method_changed=False)
    return OfflineInputs(channels, summary), stores
