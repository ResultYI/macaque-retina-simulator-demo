"""Disabled B-class Cone--HC split proposal; no device work on import.

Implementation permission is not permission to run this new numerical method.
The checked-in candidate manifest deliberately refuses execution. Neither the
old resident gate nor a Pro-suggested larger error budget certifies this method.
"""
from pathlib import Path
import hashlib
import json
import re
import time
from copy import deepcopy

from .contracts import ContractError, canonical_hash
from .resident_contract import ResidentNativeContract


CANDIDATE = 'cone_hc_split_fp64_candidate_v1.json'
BACKEND = 'cuda_cone_hc_persistent_dop853_cpu_inner_dense_fp64_candidate_v2'
USEFUL_SCOPE = 'FIRST_USEFUL_SPLIT_CHUNKS_WITH_QUALIFIED_BACKGROUND'


class ConeHCSplitContract(ResidentNativeContract):
    def validate_reference(self, reference):
        a = self.runtime_authorization or {}
        if a.get('scope') == USEFUL_SCOPE and 'source_reference_graph_binding' in a:
            from .source_graph_contract import SourceGraphContract
            binding = a['source_reference_graph_binding']
            if binding['configuration_sha256'] != a['target_graph_receipt_sha256']:
                raise ContractError('Useful source graph receipt differs from approved targets')
            SourceGraphContract.from_binding(reference, binding).validate_reference(reference)
        else:
            super().validate_reference(reference)

    def __init__(self):
        super().__init__()
        self.reference_contract_sha256 = self.sha256
        path = Path(__file__).resolve().parent.parent / 'configurations' / 'numerical_equivalence' / CANDIDATE
        raw = path.read_bytes()
        self.candidate_sha256 = hashlib.sha256(raw).hexdigest()
        self.candidate = json.loads(raw)
        if self.candidate.get('schema') != 'cone-hc-split-fp64-candidate-1':
            raise ContractError('Unreviewed Cone-HC split candidate schema')
        # Stable scientific identity: runtime deadlines/permit IDs are NOT
        # equations, numerical settings, input content or cache identity.
        self.scientific_record = dict(schema='cone_hc_split_science_v2',
            reference_contract=self.reference_contract_sha256,
            **{k: deepcopy(self.candidate[k]) for k in ('dtype','native','inner','coupling','observation')})
        self.sha256 = canonical_hash(self.scientific_record)
        self.runtime_authorization = None

    @classmethod
    def for_bounded_comparison(cls, authorization):
        """One explicit diagnostic authorization; never edits frozen manifests.

        The caller must retain this prospective record with actual checkpoint,
        input and deadline identities. It does not authorize a production source
        or permanently relax a comparator. Both numerical budgets are reported.
        """
        obj = cls()
        a = deepcopy(authorization)
        if (a.get('scope') not in ('ONE_PAIRED_20MS_RESIDENT_SPLIT_ENGINEERING_CONTINUATION',
                                  'ONE_BOUNDED_DYNAMIC_SPLIT_ENGINEERING_ACCEPTANCE')
                or a.get('source_checkpoint_fit_eligible') is not False
                or a.get('default_backend_change') is not False
                or a.get('official_cache_source_change') is not False
                or a.get('compare_both_old_and_pro_budget') is not True
                or not 0 < a.get('max_wall_seconds', 0) <= 1800
                or not 0 < a.get('max_rss_bytes', 0) <= 12*1024**3
                or not 0 < a.get('max_vram_bytes', 0) <= 12*1024**3
                or not time.time() < a.get('expires_unix', 0)):
            raise ContractError('Invalid or expired one-off split comparison authorization')
        for key in ('source_checkpoint_sha256', 'source_resident_model_hash', 'baseline_binding_hash',
                    'input_policy_sha256', 'comparison_protocol_sha256'):
            if not isinstance(a.get(key), str) or not re.fullmatch('[0-9a-f]{64}', a[key]):
                raise ContractError('One-off split authorization is missing actual identity: '+key)
        if not a.get('user_authorization_reference'):
            raise ContractError('One-off split comparison requires explicit human authorization evidence')
        obj.runtime_authorization = a
        obj._runtime_started_unix = float(a.get('round_started_unix', a['expires_unix']-a['max_wall_seconds']))
        if not obj._runtime_started_unix <= time.time() < obj._runtime_started_unix+a['max_wall_seconds']:
            raise ContractError('One-off split total wall budget is exhausted or has not started')
        obj._authorization_sha256 = canonical_hash(a)
        obj.candidate = deepcopy(obj.candidate)
        obj.candidate.update(execution_enabled=True, status='APPROVED_FOR_BOUNDED_NUMERICAL_QA')
        obj.candidate['acceptance'].update(
            approved_contract_sha256=a['comparison_protocol_sha256'],
            approved_input_policy_sha256=a['input_policy_sha256'],
            approval_reference=a['user_authorization_reference'],
            approved_runtime_budget={k: a[k] for k in ('max_wall_seconds','max_rss_bytes','max_vram_bytes')})
        # Keep scientific SHA unchanged. The permit is still separately checked
        # before execution/restore and at each solver/event boundary.
        obj.require_execution_approval()
        return obj

    def require_execution_approval(self):
        """Fail before allocation/JIT; this branch is intentionally disabled."""
        c = self.candidate
        if self.runtime_authorization is not None:
            from .useful_runtime_budget import goal_only_authorization
            goal_only=goal_only_authorization(self.runtime_authorization)
            if (canonical_hash(self.runtime_authorization) != self._authorization_sha256
                    or (not goal_only and (time.time() >= self.runtime_authorization['expires_unix']
                    or time.time() >= self._runtime_started_unix+self.runtime_authorization['max_wall_seconds']))):
                raise ContractError('One-off split authorization changed or expired')
        a = c.get('acceptance', {})
        digest = lambda value: isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)
        if (c.get('execution_enabled') is not True or c.get('status') != 'APPROVED_FOR_BOUNDED_NUMERICAL_QA'
                or not digest(a.get('approved_contract_sha256'))
                or not digest(a.get('approved_input_policy_sha256'))
                or not a.get('approval_reference') or not a.get('approved_runtime_budget')):
            raise ContractError('CONE_HC_SPLIT_B_CONTRACT_PENDING_APPROVAL: no simulation, GPU/JIT or validation is enabled')
        if a.get('proposed_budget_enabled') is not False:
            raise ContractError('Pro-proposed enlarged error budget cannot be enabled by this implementation')
        # Even a future QA approval does not select an official source/default.
        if c.get('default_backend_change') or c.get('official_cache_source_change'):
            raise ContractError('Split QA implementation cannot change default/backend cache authorization')
        n, i = c['native'], c['inner']
        if (c.get('dtype') != 'float64' or n['method'] != 'DOP853' or i['method'] != 'DOP853'
                or (n['rtol'], n['atol'], n['max_step']) != (1e-6, 1e-9, .0005)
                or (i['rtol'], i['atol'], i['max_step']) != (1e-6, 1e-9, .0001)):
            raise ContractError('Unreviewed split numerical proposal; do not silently tune it')

    @classmethod
    def for_useful_chunks(cls, authorization):
        """Explicit bounded source permit, only after actual input/graph approval.

        Preparing this factory does not authorize a model run. Receipts and an
        explicit human authorization are required; checked-in defaults stay off.
        """
        obj = cls()
        a = deepcopy(authorization)
        required = ('source_resident_model_hash', 'baseline_binding_hash',
                    'input_policy_receipt_sha256', 'target_graph_receipt_sha256',
                    'execution_protocol_sha256')
        policies = a.get('approved_input_policy_sha256_by_phase', {})
        digest = lambda v: isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v)
        start, wall = a.get('round_started_unix', 0.), a.get('max_wall_seconds', 0.)
        from .useful_runtime_budget import goal_only_authorization
        goal_only=goal_only_authorization(a)
        if (a.get('scope') != USEFUL_SCOPE or a.get('input_policy_approved') is not True
                or a.get('target_graph_approved') is not True
                or a.get('default_backend_change') is not False
                or a.get('fit_or_upload_authorized') is not False
                or not a.get('user_authorization_reference')
                or any(not digest(a.get(k)) for k in required)
                or not policies or not set(policies).issubset({'uniform_background', 'movie'})
                or any(not digest(v) for v in policies.values())
                or (not goal_only and (not 0 < wall <= 3600
                    or not start <= time.time() < min(a.get('expires_unix', 0.), start+wall)))
                or any(not 0 < a.get(k, 0) <= 12*1024**3
                       for k in ('max_rss_bytes', 'max_vram_bytes'))):
            raise ContractError('Useful split execution requires approved actual input/graph and a live <=1h permit')
        obj.runtime_authorization = a
        obj._runtime_started_unix = float(start)
        obj._authorization_sha256 = canonical_hash(a)
        obj.candidate = deepcopy(obj.candidate)
        obj.candidate.update(execution_enabled=True, status='APPROVED_FOR_BOUNDED_NUMERICAL_QA')
        obj.candidate['acceptance'].update(
            approved_contract_sha256=a['execution_protocol_sha256'],
            approved_input_policy_sha256=policies['uniform_background'],
            approval_reference=a['user_authorization_reference'],
            approved_runtime_budget={k:a[k] for k in ('max_wall_seconds','max_rss_bytes','max_vram_bytes')})
        obj.require_execution_approval()
        return obj

    def require_input_policy(self, channels, *, phase=None):
        self.require_execution_approval()
        actual = canonical_hash({name: channel.input_contract for name, channel in channels.items()})
        a = self.runtime_authorization or {}
        if a.get('scope') == USEFUL_SCOPE:
            policies = a['approved_input_policy_sha256_by_phase']
            if phase is not None and phase not in policies:
                raise ContractError('Split input phase has no approved actual policy receipt: '+phase)
            allowed = ({policies[phase]} if phase is not None else set(policies.values()))
        else:
            allowed = {self.candidate['acceptance']['approved_input_policy_sha256']}
        if actual not in allowed:
            raise ContractError('Split actual channel input/calibration policy has no approved receipt')
        return actual

    def bind_inputs(self, reference, snapshot, reference_state, channels,
                    channel_data_hashes, monitor_policy):
        self.require_execution_approval()
        self.require_input_policy(channels)
        binding = super().bind_inputs(reference, snapshot, reference_state, channels,
                                      channel_data_hashes, monitor_policy)
        binding.pop('binding_hash')
        binding['split_candidate_sha256'] = self.candidate_sha256
        binding['binding_hash'] = canonical_hash(binding)
        return binding


def guard_block_triangular(runner):
    """Check actual assembled topology; never infer independence from names."""
    cid = runner.partition.cid
    c = runner.compiled
    m = runner.model
    if (c.components[cid].time_contract.kind != 'continuous' or cid in runner.periods
            or m.slices[cid].start != 0 or m.slices[cid].stop != runner.partition.offset):
        raise ContractError('Split native block must be the complete continuous leading state block')
    if set(m._context_providers) != {cid}:
        raise ContractError('Unreviewed native context dependency in split graph')
    for link in c.links:
        s = link.specification
        if s.target_component == cid:
            raise ContractError('Split graph has an incoming native link: inner-to-native independence is not established')
        if s.source_component == cid and s.source_port != 'release':
            raise ContractError('Split native outward port is not the reviewed release port')
        if link.delayed_operator is not None:
            raise ContractError('Explicit delayed links are outside this split interface scope')
    if any(cid in (a, b) for a, b, *_ in m._gap_plans):
        raise ContractError('Split native cross-component gap feedback is unsupported')
    if any(block.cyclic for block in c.output_order):
        raise ContractError('Split interface does not remove algebraic closures')
    external = [e for e in c.spec.external_inputs if e.target_component == cid]
    if len(external) != 1 or external[0].target_port != 'light':
        raise ContractError('Split native dependency must be the explicit original light channel alone')
    return dict(native_component=cid, native_state_count=runner.partition.offset,
                inner_state_count=runner.nstate-runner.partition.offset,
                complete_native_feedback_retained=True, all_inner_links_retained=True,
                no_inner_to_native=True, delayed_links_supported=False)
