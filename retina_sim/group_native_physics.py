"""Original reduced equations on group operators, with original K-node IDs."""
from types import SimpleNamespace
from pathlib import Path
import hashlib

import numpy as np

from .contracts import ContractError, canonical_hash
from .cone_hc_reduced_native import ReducedNative, AnalyticREAnchor, STATE_NAMES
from .vh2005 import SOURCE_VALUES


class GroupNativePhysics(ReducedNative):
    def __init__(self, assembly, *, torch=None, device=None, plans=None):
        cone, hc = assembly.populations['cones'], assembly.populations['HC']
        A = assembly.matrix('routes/cones_to_HC/mean_weights.npz')
        B = assembly.matrix('routes/HC_to_cones/mean_weights.npz')
        model = SimpleNamespace(p=SOURCE_VALUES, nc=cone.group_count, nh=hc.group_count,
            cones=SimpleNamespace(cells=SimpleNamespace(cell_id=tuple(cone.group_ids))),
            hcs=SimpleNamespace(cells=SimpleNamespace(cell_id=tuple(hc.group_ids))), A=A, B=B,
            model_hash=canonical_hash(dict(artifact=assembly.artifact_sha256, source_parameters=SOURCE_VALUES)))
        mapped = np.asarray(cone.recording_auxiliary_group_rows, dtype=np.int64)
        if torch is not None:
            if plans is None:
                from .resident_cusparse import CusparseAPI, NativeCSRPlan
                api = CusparseAPI(torch)
                def plan(matrix, role):
                    tensor = torch.sparse_csr_tensor(
                        torch.as_tensor(matrix.indptr.astype(np.int64), dtype=torch.int64, device=device),
                        torch.as_tensor(matrix.indices.astype(np.int64), dtype=torch.int64, device=device),
                        torch.as_tensor(matrix.data.copy(), dtype=torch.float64, device=device),
                        size=matrix.shape, device=device, dtype=torch.float64)
                    result = NativeCSRPlan(tensor, api, torch)
                    result._retipath_group_binding = (assembly.artifact_sha256, role)
                    result._retipath_group_signature = tuple((v.data_ptr(), v._version) for v in
                        (tensor.values(), tensor.crow_indices(), tensor.col_indices()))
                    return result
                plans = (plan(A, 'A'), plan(B, 'B'))
            for role, selected in zip(('A','B'), plans):
                matrix = selected.matrix
                signature = tuple((v.data_ptr(), v._version) for v in
                    (matrix.values(),matrix.crow_indices(),matrix.col_indices()))
                if (getattr(selected,'_retipath_group_binding',None) != (assembly.artifact_sha256,role)
                        or getattr(selected,'_retipath_group_signature',None) != signature):
                    raise ContractError('Externally supplied native plan is not the immutable approved group operator')
            super().__init__(model, torch=torch, device=device, A=plans[0], B=plans[1], aux_indices=np.unique(mapped))
        else:
            super().__init__(model, aux_indices=np.unique(mapped))
        # Distinct original auxiliary nodes may share one group drive; preserve
        # every selected node's own filter memory and original row identity.
        self.aux_indices = tuple(int(x) for x in mapped)
        self.aux_cell_ids = tuple('original_cone_row:'+str(int(i)) for i in cone.recording_auxiliary_fine_rows)
        self._aux = mapped.copy() if torch is None else torch.as_tensor(mapped.copy(), dtype=torch.int64, device=self.device)
        self.sizes = (self.nc,)*5+(self.nh,)*3+(len(mapped),)
        self.nstate = sum(self.sizes)
        self.slices, at = {}, 0
        for name, size in zip(STATE_NAMES, self.sizes):
            self.slices[name] = slice(at, at+size); at += size
        self.identity = dict(schema='retipath-group-native-equations-1', artifact_sha256=assembly.artifact_sha256,
            parameters=dict(SOURCE_VALUES), state_names=STATE_NAMES, state_sizes=self.sizes,
            selected_original_rows=cone.recording_auxiliary_fine_rows, selected_group_rows=mapped,
            original_auxiliary_node_count=len(mapped), distinct_auxiliary_group_drives=len(np.unique(mapped)),
            equation_source_sha256=hashlib.sha256(Path(__file__).with_name('cone_hc_reduced_native.py').read_bytes()).hexdigest(),
            group_binding_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            method='equation_provider_only_no_midpoint_step', execution='CPU' if torch is None else 'CUDA_CSR_ALG2')
        self.source_version = canonical_hash(self.identity)

    def construction_point(self, seed, light, input_stop_seconds, input_epoch):
        states = seed.states
        values = [states['cones','reduced_native',n] for n in ('X','C','Vis','g_i','Vbar')]
        values += [states['HC','reduced_native',n] for n in ('w','z2','Vh')]
        values += [states['cones','selected_original_z1_auxiliary','z1_aux']]
        y = self._array(np.concatenate(values))
        anchor = AnalyticREAnchor(seed.time_seconds,
            self._array(states['cones','analytic_RE','R']), self._array(states['cones','analytic_RE','E']),
            self._array(light), float(input_stop_seconds), int(input_epoch))
        self._validate(y, anchor, seed.time_seconds)
        return y, anchor

    def step(self, state):
        raise ContractError('Group equation provider cannot execute the rejected midpoint method')
