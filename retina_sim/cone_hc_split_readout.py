"""Compact native *state* dense readout; never endpoint-release interpolation.

The selected HC rows are evaluated before the original B row reduction. This
does not exchange a sparse reduction with polynomial evaluation. Every weight
and within-row column order comes from the complete original native B matrix.
No GPU operations occur on import. New B-class method, runtime NOT RUN.
"""
import numpy as np

from .contracts import ContractError, canonical_hash


class NativeDenseRelease:
    def __init__(self, runner):
        from .resident_cusparse import NativeCSRPlan

        part, torch = runner.partition, runner.torch
        native_model = runner.compiled.components[part.cid].native_model
        expected = ('R', 'E', 'X', 'C', 'Vis', 'g_i', 'Vbar', 'z1', 'z2', 'Vh')
        if (tuple(s.state_name for s in native_model.state_specs) != expected
                or tuple(part.native.sizes) != (part.native.nc,)*8+(part.native.nh,)*2
                or sum(part.native.sizes) != part.offset):
            raise ContractError('Native dense readout state layout is not the reviewed complete vH layout')
        full = native_model.B.copy()
        full.sort_indices()
        if not full.has_canonical_format or full.dtype != np.float64:
            raise ContractError('Dense native readout requires canonical original FP64 B rows')
        used = np.asarray(part.used, dtype=np.int64)
        if not len(used):
            raise ContractError('Split readout requires at least one actual native release consumer')
        selected = full[used].tocsr()
        support = np.unique(selected.indices).astype(np.int64)
        if not len(support):
            raise ContractError('Split native release requires actual HC support')
        indices = np.searchsorted(support, selected.indices).astype(np.int64)
        row_ptr = selected.indptr.astype(np.int64)
        self.torch, self.device = torch, part.gather.device
        self.count, self.hc_count = len(used), len(support)
        self.p = dict(part.native.p)
        # State order: R,E,X,C,V,gi,Vbar,z1 (nc each), z2,h (nh each).
        rows = np.concatenate((4*part.native.nc+used, 6*part.native.nc+used,
                               8*part.native.nc+part.native.nh+support))
        self.rows = torch.as_tensor(rows.copy(), dtype=torch.int64, device=self.device)
        matrix = torch.sparse_csr_tensor(
            torch.as_tensor(row_ptr.copy(), dtype=torch.int64, device=self.device),
            torch.as_tensor(indices.copy(), dtype=torch.int64, device=self.device),
            torch.as_tensor(selected.data.copy(), dtype=torch.float64, device=self.device),
            size=(len(used), len(support)), dtype=torch.float64, device=self.device,
            check_invariants=True)
        if not isinstance(part.native.B, NativeCSRPlan):
            raise ContractError('Split candidate requires the retained explicit CSR ALG2 native plan')
        self.B = NativeCSRPlan(matrix, part.native.B.api, torch)
        self.identity = dict(state_rows=rows, B_row_ptr=row_ptr, B_columns=indices,
                             B_values=selected.data.copy(), native_parameters=dict(self.p),
                             evaluation_order='dense_state_then_original_B_row_then_sigmoid_then_adaptation',
                             sparse_policy='original_retained_cusparse_CSR_ALG2_float64')
        self.sha256 = canonical_hash(self.identity)

    def project(self, accepted_segment):
        return accepted_segment.project_rows(self.rows)

    def at(self, projected_segment, t):
        torch, p = self.torch, self.p
        state = projected_segment.at(t)
        v, vbar, h = state.split((self.count, self.count, self.hc_count))
        if (not torch.isfinite(state).all().item() or (v < 0).any().item()
                or (vbar <= 0).any().item()):
            raise ContractError('Native dense readout finite/positive domain violated; no clipping')
        bh = torch.mv(self.B, h.contiguous())
        q = 1./(1.+torch.exp(-((v-bh-p['V_k'])/p['V_n'])))
        release = q/(vbar/p['V_I'])**p['mu']
        if not torch.isfinite(release).all().item():
            raise ContractError('Nonfinite nonlinear native dense release')
        return release


class AcceptedNativeCursor:
    """Accepted lookahead owned separately from the public inner-time state.

    Consumers must acknowledge time explicitly. The buffer prunes only through
    their minimum watermark; a native acceptance never advances a consumer.
    The native observation delay queue is NOT this buffer and never receives
    lookahead rows. Old segment tensors are immutable and safely shared by a
    functional state fork; controller metadata and consumer maps are copied.
    """
    CONSUMERS = ('inner', 'monitor', 'recording', 'observation')

    def __init__(self, solver, readout, source_binding):
        self.solver, self.readout = solver, readout
        self.source_binding = source_binding
        self.segments = []
        self.projected = []
        self.consumed = dict.fromkeys(self.CONSUMERS, solver.t)

    def fork(self):
        obj = type(self).__new__(type(self))
        obj.solver, obj.readout = self.solver.fork(), self.readout
        obj.source_binding = self.source_binding
        obj.segments, obj.projected = list(self.segments), list(self.projected)
        obj.consumed = dict(self.consumed)
        return obj

    def append(self, segment):
        previous = self.segments[-1].stop if self.segments else min(self.consumed.values())
        if segment.start != previous or segment.stop != self.solver.t:
            raise ContractError('Accepted native buffer is not contiguous with its solver/consumers')
        compact = self.readout.project(segment)
        self.segments.append(segment)
        self.projected.append(compact)

    def segment(self, t, *, left_limit=False):
        # The state is continuous at a genuine light discontinuity; the selected
        # segment/input epoch is still side-specific. Never round a query into
        # the next interval with an epsilon.
        rows = reversed(list(zip(self.segments, self.projected)))
        for full, compact in rows:
            if full.start <= t <= full.stop:
                if left_limit and t == full.start:
                    continue
                return full, compact
        if left_limit:
            for full, compact in zip(self.segments, self.projected):
                if full.start == t:
                    return full, compact
        raise ContractError('Native consumer query is outside retained accepted intervals')

    def state_at(self, t):
        if t == self.solver.t:
            return self.solver.y.clone()
        return self.segment(t)[0].at(t)

    def release_at(self, t, *, left_limit=False):
        return self.readout.at(self.segment(t, left_limit=left_limit)[1], t)

    def acknowledge(self, consumer, t):
        if consumer not in self.consumed or not self.consumed[consumer] <= t <= self.solver.t:
            raise ContractError('Invalid native consumer watermark')
        self.consumed[consumer] = float(t)
        floor = min(self.consumed.values())
        while self.segments and self.segments[0].stop <= floor:
            self.segments.pop(0)
            self.projected.pop(0)

    def payload(self):
        return dict(solver=self.solver.payload(), segments=[s.payload() for s in self.segments],
                    consumed=dict(self.consumed), source_binding=self.source_binding,
                    readout_sha256=self.readout.sha256)

    @classmethod
    def from_payload(cls, raw, runner, template, policy, public_time):
        from .cone_hc_split_solver import AcceptedNativeDense, PersistentNativeDop853
        if (raw['source_binding'] != runner.binding['binding_hash']
                or raw['readout_sha256'] != runner.dense_release.sha256):
            raise ContractError('Split checkpoint native input/readout identity differs')
        solver = PersistentNativeDop853.from_payload(raw['solver'], runner.torch, template, policy)
        obj = cls(solver, runner.dense_release, raw['source_binding'])
        marks = raw['consumed']
        if (set(marks) != set(cls.CONSUMERS) or
                any(not np.isfinite(t) or t != public_time for t in marks.values()) or solver.t < public_time):
            raise ContractError('Split checkpoint consumer ownership/time is incomplete')
        obj.consumed = dict(marks)
        previous = None
        for row in raw['segments']:
            segment = AcceptedNativeDense.from_payload(row, runner.torch, template)
            if previous is not None and segment.start != previous:
                raise ContractError('Split checkpoint accepted interval gap/overlap')
            if segment.stop <= public_time or segment.stop > solver.t:
                raise ContractError('Split checkpoint retained interval ownership differs')
            obj.segments.append(segment)
            obj.projected.append(obj.readout.project(segment))
            previous = segment.stop
        if solver.t > public_time:
            if not obj.segments or obj.segments[0].start > public_time or previous != solver.t:
                raise ContractError('Split checkpoint lost native lookahead needed by inner')
            if not runner.torch.equal(obj.segments[-1].y1.view(runner.torch.int64), solver.y.view(runner.torch.int64)):
                raise ContractError('Split checkpoint native accepted endpoint disagrees with controller')
        elif obj.segments:
            raise ContractError('Unexpected old native dense intervals at a fully consumed checkpoint')
        return obj
