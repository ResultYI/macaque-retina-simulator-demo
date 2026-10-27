"""Bounded reuse of exact numeric JSON blocks, keyed by every current byte."""
from collections import OrderedDict
import json

import numpy as np


class CanonicalBlockCache:
    def __init__(self, max_payload_bytes=128*1024**2):
        if type(max_payload_bytes) is not int or max_payload_bytes < 0:
            raise ValueError('Nonnegative integer canonical cache capacity required')
        self.max_payload_bytes = max_payload_bytes
        self.entries = OrderedDict()
        self.payload_bytes = self.peak_payload_bytes = 0
        self.hits = self.misses = self.evictions = self.bytes_read = 0
        self.deduplicated_blocks = 0

    def _encode(self, dtype, raw):
        values = np.frombuffer(raw, dtype=dtype)
        if len(values) >= 1024:
            unique, inverse = np.unique(values.view('V'+str(values.dtype.itemsize)), return_inverse=True)
            if len(unique)*2 <= len(values):
                scalars = np.frombuffer(unique.tobytes(), dtype=dtype).tolist()
                tokens = json.dumps(scalars, sort_keys=True, allow_nan=False, separators=(',', ':'))[1:-1].split(',')
                self.deduplicated_blocks += 1
                return ','.join(np.asarray(tokens, dtype=object)[inverse].tolist())
        return json.dumps(values.tolist(), sort_keys=True, allow_nan=False, separators=(',', ':'))[1:-1]

    def encode(self, dtype, raw):
        self.bytes_read += len(raw)
        key = (np.dtype(dtype).str, raw)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        encoded = self._encode(dtype, raw)
        size = len(raw)+len(encoded)
        if size <= self.max_payload_bytes:
            while self.entries and self.payload_bytes+size > self.max_payload_bytes:
                (old_dtype, old_raw), old_encoded = self.entries.popitem(last=False)
                self.payload_bytes -= len(old_raw)+len(old_encoded)
                self.evictions += 1
            self.entries[key] = encoded
            self.payload_bytes += size
            self.peak_payload_bytes = max(self.peak_payload_bytes, self.payload_bytes)
        return encoded

    def metrics(self):
        return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                    bytes_read=self.bytes_read, payload_bytes=self.payload_bytes,
                    peak_payload_bytes=self.peak_payload_bytes,
                    max_payload_bytes=self.max_payload_bytes,
                    deduplicated_blocks=self.deduplicated_blocks,
                    key='dtype_and_complete_current_bytes_not_a_digest_or_object_identity')
