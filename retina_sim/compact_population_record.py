"""Array-backed construction metadata; canonical IDs and table hash stay original.

This is not a new storage/identity format. Export through the existing plain
serializer when JSON lists are required. Numeric arrays remain read-only.
"""
import numpy as np
from collections.abc import Sequence
from operator import index as integer_index
from .canonical_stream import CanonicalSequence


class CompactPopulationRecord(dict):
    @classmethod
    def from_canonical(cls, table):
        record = cls({name: getattr(table, name) for name in
            ('atlas_id', 'atlas_version', 'source_version', 'population', 'reference_id')})
        record.update(source_keys=table.source_keys, global_cell_ids=table.global_cell_ids,
            table_hash=table.table_hash,
            coordinates={role: dict(values=q.values, unit=q.unit, meaning=q.meaning,
                frame=q.coordinate_frame, evidence_tag=q.evidence_tag)
                for role, q in table.coordinates.items()})
        return record

    def select(self, ids):
        wanted = ids if isinstance(ids, (set, frozenset)) else set(ids)
        rows = np.fromiter((i for i, gid in enumerate(self['global_cell_ids']) if gid in wanted), dtype=np.int64)
        if len(rows) == len(self['global_cell_ids']):
            return type(self)(self)
        coordinates = {}
        for role, quantity in self['coordinates'].items():
            values = quantity['values'][rows]
            values.setflags(write=False)
            coordinates[role] = dict(quantity, values=values)
        return type(self)(self, global_cell_ids=tuple(self['global_cell_ids'][i] for i in rows),
            source_keys=tuple(self['source_keys'][i] for i in rows), coordinates=coordinates)


class CompactContactEdges(CanonicalSequence, Sequence):
    """Original four-field contact records, without one dictionary per edge.

    This is construction/audit storage only. The compiled ConnectionTables and
    sparse numerical operators are not modified. Explicit iteration/export
    reconstructs the exact original records in their original order.
    """
    def __init__(self, pre_ids, post_ids, rows, columns, weights):
        self.pre_ids, self.post_ids = tuple(pre_ids), tuple(post_ids)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.columns = np.asarray(columns, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float64)
        if (self.rows.ndim != 1 or self.columns.shape != self.rows.shape
                or self.weights.shape != self.rows.shape or not np.isfinite(self.weights).all()
                or np.any(self.rows < 0) or np.any(self.rows >= len(self.post_ids))
                or np.any(self.columns < 0) or np.any(self.columns >= len(self.pre_ids))):
            raise ValueError('Original contact indices and finite weights required')
        for values in (self.rows, self.columns, self.weights):
            values.setflags(write=False)

    @classmethod
    def from_records(cls, pre_ids, post_ids, records):
        pre_ids, post_ids = tuple(pre_ids), tuple(post_ids)
        pi, po = {gid:i for i,gid in enumerate(pre_ids)}, {gid:i for i,gid in enumerate(post_ids)}
        if len(pi) != len(pre_ids) or len(po) != len(post_ids):
            raise ValueError('Unique original contact population IDs required')
        n = len(records)
        rows, columns, weights = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.float64)
        for i, record in enumerate(records):
            if set(record) != {'pre_id', 'post_id', 'weight', 'unit'} or record['unit'] != 'dimensionless':
                raise ValueError('Unreviewed contact record fields or unit')
            rows[i], columns[i], weights[i] = po[record['post_id']], pi[record['pre_id']], record['weight']
        return cls(pre_ids, post_ids, rows, columns, weights)

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        i = integer_index(key)
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        return dict(pre_id=self.pre_ids[self.columns[i]], post_id=self.post_ids[self.rows[i]],
                    weight=float(self.weights[i]), unit='dimensionless')

    def factory(self):
        return iter(self)

    def parents_for(self, post_ids):
        wanted = set(post_ids)
        selected = np.fromiter((gid in wanted for gid in self.post_ids), dtype=bool, count=len(self.post_ids))
        return (self.pre_ids[i] for i in self.columns[selected[self.rows]])

    def select(self, pre_ids=None, post_ids=None):
        pre = None if pre_ids is None else set(pre_ids)
        post = None if post_ids is None else set(post_ids)
        return [record for record in self if (pre is None or record['pre_id'] in pre)
                and (post is None or record['post_id'] in post)]

    def remap(self, pre_ids, post_ids):
        """Retain edge order with caller-ordered IDs, without expanding records."""
        pre_ids,post_ids=tuple(pre_ids),tuple(post_ids)
        if pre_ids==self.pre_ids and post_ids==self.post_ids:return self
        pre={gid:i for i,gid in enumerate(pre_ids)};post={gid:i for i,gid in enumerate(post_ids)}
        if len(pre)!=len(pre_ids) or len(post)!=len(post_ids):raise ValueError('Unique contact IDs required')
        columns=np.fromiter((pre.get(gid,-1) for gid in self.pre_ids),dtype=np.int64)
        rows=np.fromiter((post.get(gid,-1) for gid in self.post_ids),dtype=np.int64)
        r,c=rows[self.rows],columns[self.columns];keep=(r>=0)&(c>=0)
        return type(self)(pre_ids,post_ids,r[keep],c[keep],self.weights[keep])
