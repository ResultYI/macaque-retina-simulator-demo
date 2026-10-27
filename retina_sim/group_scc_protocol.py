"""Immutable SCC-Radau contract and cumulative, fail-closed acceptance budgets."""
from contextlib import AbstractContextManager
from pathlib import Path
import hashlib
import json
import math
import os
import time
import uuid

from .contracts import ContractError, canonical_hash
from .atomic_io import replace_with_retry, read_bytes_with_retry

SCHEMA = 'retipath-scc-radau-acceptance-1'
LEGACY_BATCH_ID = 'scc-radau-20261025-authorization-v1'
LEGACY_REPORT_PARENT = 'reports/solver_scc_radau_20261025_v1'
LEGACY_LIMITS = {'cold': 600.0, 'feasibility': 180.0, 'paired': 1200.0}
_PROFILE = os.environ.get('RETIPATH_ACCEPTANCE_BATCH', 'legacy')
if _PROFILE not in ('legacy', 'overnight_20261026_v1', 'overnight_short_20261026_v1',
                    'overnight_snapshot_20261026_v1', 'overnight_snapshot_short_20261026_v1',
                    'overnight_event_20261026_v1','overnight_event_short_20261026_v1'):
    raise ContractError('Unknown explicit acceptance authorization profile')
_OVERNIGHT = _PROFILE != 'legacy'
_SHORT_DIAGNOSTIC = _PROFILE in ('overnight_short_20261026_v1', 'overnight_snapshot_short_20261026_v1','overnight_event_short_20261026_v1')
_SNAPSHOT_STAGE = _PROFILE.startswith('overnight_snapshot_')
_EVENT_STAGE = _PROFILE.startswith('overnight_event_')
BATCH_ID = ('gpu-event-20261026-authorization-v1' if _EVENT_STAGE else 'gpu-snapshot-20261026-authorization-v1' if _SNAPSHOT_STAGE else
            'gpu-overnight-20261026-authorization-v1' if _OVERNIGHT else LEGACY_BATCH_ID)
REPORT_PARENT = ('reports/solver_gpu_event_20261026_v1' if _EVENT_STAGE else 'reports/solver_gpu_snapshot_20261026_v1' if _SNAPSHOT_STAGE else
                 'reports/solver_gpu_overnight_20261026_v1' if _OVERNIGHT else LEGACY_REPORT_PARENT)
BUDGET_SHA256 = '3cfab9768a856c13354ab38cc65b506da60661fe3e3b0f05e46c647c08981a77'
LIMITS = {'cold': 1800.0, 'feasibility': 600.0, 'paired': 7200.0} if _OVERNIGHT else LEGACY_LIMITS
RESOURCE_LIMITS = dict(max_rss_bytes=16*1024**3, minimum_available_bytes=4*1024**3,
    max_whole_GPU_bytes=12*1024**3, minimum_GPU_free_bytes=512*1024**2)
GPU_ACTIVITY_POLICY = dict(version='windows_engine_activity_v1', external_compute_limit_percent=0.0,
    quiet_graphics_time_weighted_aggregate_percent=1.0, graphics_peak_aggregate_percent=5.0,
    close_gain_repeat_ratio=1.1, unknown_activity='INCONCLUSIVE',
    scope='Engineering timing qualification only; scientific and resource gates unchanged')
CANDIDATES = ('radau', 'gpu_dop853')
FALLBACK_ORIGIN = dict(output=LEGACY_REPORT_PARENT+'/feasibility_r5',
    run_id='2648c52a653946c6a5ced0d7e254aa13', status='NUMERICAL_FAIL',
    source_identity='e13b3fd2ca3738a3de69370511691ecabdd0d0e283952e50d87078e8666d812b',
    evidence_sha256={
        'failure.json':'9f65fb5112b813d2aef376cc7b593c158c261468bf33babcf43e73fc79deb5a1',
        'REQUEST_STOP':'844a397c2a3bce773294c1e3ae15ce4784b3090d0f22e3f2d0e4b57ad2e0556e',
        'prospective_identity.json':'8af8a4adf2bae7b68f1f044791664fc83f80bd820b954b756f88bd8247c8ce7f'})
FALLBACK_GATE = dict(scope_seconds=.0001, wall_limit_seconds=180., charge_category='paired',
    contained_in_paired_limit_seconds=1200., additional_feasibility_allowance_seconds=0.,
    requires='Full first interval dense, numerical, domain, event, monitor, recording and acknowledgement')
GPU_DOP853_CONTRACT = dict(classification='A', backend='full_group_gpu_dop853_v1',
    mathematical_method='original_DOP853', precision='float64',
    error_norm='All original inner rows including held rows',
    controller='Original DOP853 controller restarted at every original event-free interval',
    dense='Original seven-coefficient DOP853 polynomial at every accepted step',
    reference='Unchanged CPU-inner and GPU-native original DOP853')
FAILURES = {'NUMERICAL_FAIL', 'DOMAIN_FAIL', 'EVENT_FAIL', 'RESTORE_FAIL', 'RESOURCE_FAIL', 'INCONCLUSIVE'}


class AcceptanceFailure(ContractError):
    def __init__(self, status, message):
        if status not in FAILURES:
            raise ValueError('Unknown acceptance failure status')
        self.status = status
        super().__init__(message)


def read_json(path, *, permission_retry_seconds=0.):
    def invalid(value):
        raise ContractError('Nonfinite JSON number: '+value)
    raw=(read_bytes_with_retry(path,max_wait_seconds=permission_retry_seconds).decode('utf8')
        if permission_retry_seconds else Path(path).read_text(encoding='utf8'))
    return json.loads(raw, parse_constant=invalid)


def write_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name+'.writing')
    with pending.open('w', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    replace_with_retry(pending, path)


def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ContractError('Finite nonnegative '+name+' required')
    return float(value)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def scientific_source_preflight(root):
    """Check the actual frozen comparator dependencies without CUDA or model loading."""
    from .resident_derived_ports import ADDENDUM_FILENAME, ADDENDUM_SHA256, ROD_SOURCE_SHA256
    root=Path(root)
    required={
        'configurations/numerical_equivalence/resident_native_fp64_v1.json': BUDGET_SHA256,
        'configurations/numerical_equivalence/'+ADDENDUM_FILENAME: ADDENDUM_SHA256,
        'retina_sim/rod_pathway.py': ROD_SOURCE_SHA256,
        'configurations/numerical_equivalence/resident_native_source_pins_v6.json':
            '757112307dddb49e8c89ae49ed942d3eb6a2b33db6c59b90f3e337ef51024f3e',
    }
    checked=[]; issues=[]
    for relative,expected in required.items():
        row=dict(path=relative,expected_sha256=expected,actual_sha256=None)
        try:
            row['actual_sha256']=sha256(root/relative)
            row['status']='PASS' if row['actual_sha256']==expected else 'FROZEN_HASH_MISMATCH'
        except FileNotFoundError:
            row['status']='MISSING_FROZEN_SCIENTIFIC_DEPENDENCY'
        except OSError as error:
            row.update(status='UNREADABLE_FROZEN_SCIENTIFIC_DEPENDENCY',error=str(error))
        checked.append(row)
        if row['status']!='PASS':
            issues.append(row)
    return dict(status='INCONCLUSIVE' if issues else 'PASS',stage='SOURCE_PREFLIGHT',
        message='Required frozen scientific dependency missing, changed or unreadable' if issues else 'Required frozen scientific dependency hashes match',
        scientific_status='INCONCLUSIVE' if issues else 'DEPENDENCIES_PRESENT_ONLY',
        issues=issues,checked_files=checked,model_forward_started=False,cuda_initialized=False,
        adoptable=False,default_adopted=False,fit_eligible=False,budget_charged=False,
        note='Presence and exact frozen hashes only; no numerical or execution qualification')


def candidate_modes(candidate):
    if candidate not in CANDIDATES:
        raise ContractError('Unknown acceptance candidate')
    return ('reference', candidate)


def validate_candidate_phase(candidate, phase):
    candidate_modes(candidate)
    if phase not in ('feasibility', 'paired'):
        raise ContractError('Unknown acceptance phase')
    if candidate == 'gpu_dop853' and phase != 'paired':
        raise ContractError('GPU DOP853 fallback is only an integral full paired suite; no new feasibility allowance')


def historical_batch_receipt(root):
    """Bind new compute authorization to the unchanged archived accounting."""
    parent = Path(root)/LEGACY_REPORT_PARENT
    path = parent/'budget_ledger.json'
    expected = '9ea19ecab62abf3fa6cb1c3ebd9b699871c2f6a0de54fc36f477458d4f0da02f'
    archived, ledger = read_json(parent/'ARCHIVED.json'), read_json(path)
    witness = ledger.pop('ledger_hash', None)
    if (sha256(path) != expected or witness != canonical_hash(ledger)
            or archived.get('budget_ledger_sha256') != expected
            or archived.get('status') != 'USER_STOP_ARCHIVED'
            or archived.get('model_or_test_processes_running') is not False
            or ledger.get('active') is not None or ledger.get('batch_id') != LEGACY_BATCH_ID
            or ledger.get('limits') != LEGACY_LIMITS
            or archived.get('budget_used_seconds') != ledger.get('used')):
        raise ContractError('Archived accounting differs; new authorization cannot erase prior charges')
    return dict(batch_id=LEGACY_BATCH_ID, ledger_sha256=expected,
                archive_sha256=sha256(parent/'ARCHIVED.json'),
                original_root=ledger['root'], used=dict(ledger['used']), limits=dict(ledger['limits']))


def historical_batch_receipts(root):
    receipts = [historical_batch_receipt(root)]
    if _SNAPSHOT_STAGE or _EVENT_STAGE:
        path = Path(root)/'reports/solver_gpu_overnight_20261026_v1/budget_ledger.json'
        expected = 'fc4639c6529ad6d89b80c0d5c1a630a2fc11ad7af95ff4203bb577c4e2000377'
        prior = read_json(path)
        witness = prior.pop('ledger_hash', None)
        if (sha256(path) != expected or witness != canonical_hash(prior) or prior.get('active') is not None
                or prior.get('batch_id') != 'gpu-overnight-20261026-authorization-v1'
                or prior.get('used') != dict(cold=112.67233920097351, feasibility=0., paired=5007.933922767639)
                or prior.get('historical_batches') != receipts):
            raise ContractError('Previous overnight stage charges changed; no fresh snapshot allowance')
        receipts.append(dict(batch_id=prior['batch_id'], ledger_sha256=expected, original_root=prior['root'],
            used=dict(prior['used']), limits=dict(prior['limits']),
            ledger_path='reports/solver_gpu_overnight_20261026_v1/budget_ledger.json'))
    if _EVENT_STAGE:
        relative='reports/solver_gpu_snapshot_20261026_v1/budget_ledger.json'
        expected='898f5e707d667ebcc241a4eeba3de94950643037f3ade5f670e01baec6ac5d45'
        prior=read_json(Path(root)/relative);witness=prior.pop('ledger_hash',None)
        if (sha256(Path(root)/relative)!=expected or witness!=canonical_hash(prior)
                or prior.get('active') is not None or prior.get('stopped')
                or prior.get('batch_id')!='gpu-snapshot-20261026-authorization-v1'
                or prior.get('used')!=dict(cold=94.70864367485046,feasibility=0.,paired=4783.335707187653)
                or prior.get('limits')!=dict(cold=1800.,feasibility=600.,paired=7200.)
                or prior.get('historical_batches')!=receipts):
            raise ContractError('Snapshot-stage charges changed; new event allowance cannot reset them')
        prefix='reports/solver_gpu_snapshot_20261026_v1/paired_combined_r5_r1/'
        pins={prefix+name:digest for name,digest in {
            'failure.json':'36f2fc8842c25c94709c9264fc0b7cc4e86a480d1772a1e721084ef0424fab7b',
            'REQUEST_STOP':'aee37708fd2c53632dae4515bfbc0f76449690cca3b20ccc7129bb3dcece539e',
            'uninterrupted_result.json':'85a200fd50061f7cb2312a1ff352dd3b11209849e190918b687c5db97153ae5a',
            'session_checkpoint.json':'bd2b94f8145f44848c30dd66cdc1d0f9656ad1fa765373d00be48ec45b89c923',
            'uninterrupted_oracle.json':'5c1dfa9413966471ddcb0a469f26f1e5fa72ffccfb292e80d1700dcab6038bb0'}.items()}
        if any(sha256(Path(root)/name)!=digest for name,digest in pins.items()):
            raise ContractError('Original combined r5 restore failure evidence changed')
        receipts.append(dict(batch_id=prior['batch_id'],ledger_sha256=expected,original_root=prior['root'],
            used=dict(prior['used']),limits=dict(prior['limits']),ledger_path=relative,evidence_pins=pins))
    return receipts


def verify_fallback_origin(root, protocol, ledger=None):
    if protocol.get('execution_contract') != GPU_DOP853_CONTRACT:
        raise ContractError('GPU DOP853 A execution contract changed')
    if protocol.get('fallback_origin') != FALLBACK_ORIGIN or protocol.get('paired_entry_gate') != FALLBACK_GATE:
        raise ContractError('Frozen fallback lineage or integral paired entry gate changed')
    root=Path(root)
    origin=root/FALLBACK_ORIGIN['output']
    for name,expected in FALLBACK_ORIGIN['evidence_sha256'].items():
        if sha256(origin/name) != expected:
            raise ContractError('Failed Radau evidence changed: '+name)
    stop=read_json(origin/'REQUEST_STOP')
    failure=read_json(origin/'failure.json')
    if stop.get('run_id') != FALLBACK_ORIGIN['run_id'] or stop.get('status') != 'NUMERICAL_FAIL' or failure.get('status') != 'NUMERICAL_FAIL':
        raise ContractError('Failed Radau lineage is not the frozen numerical failure')
    if ledger is not None:
        recorded_origin = origin
        if _OVERNIGHT:
            receipt = historical_batch_receipt(root)
            if ledger.get('historical_batches') != historical_batch_receipts(root):
                raise ContractError('New batch must retain the archived budget receipt')
            ledger = read_json(root/LEGACY_REPORT_PARENT/'budget_ledger.json')
            recorded_origin = Path(receipt['original_root'])/FALLBACK_ORIGIN['output']
        matched=[a for a in ledger['attempts'] if a['run_id']==FALLBACK_ORIGIN['run_id']]
        if (len(matched)!=1 or matched[0]['status']!='NUMERICAL_FAIL' or matched[0]['phase']!='feasibility'
                or matched[0]['identity']!=FALLBACK_ORIGIN['source_identity']
                or Path(matched[0]['output']).resolve()!=recorded_origin.resolve()):
            raise ContractError('Fallback must retain the original failed Radau ledger attempt and charges')


def load_protocol(root, candidate='radau'):
    root = Path(root)
    candidate_modes(candidate)
    filename='group_scc_radau_fp64_v1.json' if candidate=='radau' else 'group_gpu_dop853_fp64_v1.json'
    p = read_json(root/'configurations/numerical_equivalence'/filename)
    if (p.get('schema') != SCHEMA or p.get('batch_id') != LEGACY_BATCH_ID
            or p.get('budget_limits_seconds') != LEGACY_LIMITS or p.get('resource_limits') != RESOURCE_LIMITS
            or (p.get('start_seconds'), p.get('checkpoint_seconds'), p.get('stop_seconds')) != (4.993, 4.997, 5.001)
            or p.get('reference') != dict(method='joint_DOP853', rtol=1e-6, atol=1e-9, max_step=.0001)
            or p.get('numerical_budget', {}).get('sha256') != BUDGET_SHA256):
        raise ContractError('Frozen SCC-Radau protocol identity or limits changed')
    if p.get('gpu_activity_timing_policy') != GPU_ACTIVITY_POLICY:
        raise ContractError('Prospectively frozen GPU timing qualification changed')
    expected_candidate = dict(method='SCC_RADAU_IIA5_TWO_HALF_HERMITE5', rtol=1e-6, atol=1e-9,
        max_step=.0001, fast_max_step=.0001, maximum_subdivision=6, max_attempts_per_100us=127,
        max_newton=8, gmres_restart=32, max_gmres=96)
    if candidate=='gpu_dop853':
        expected_candidate=dict(method='joint_DOP853', rtol=1e-6, atol=1e-9, max_step=.0001, fast_max_step=.0001)
        verify_fallback_origin(root,p)
    if p['candidate'] != expected_candidate:
        raise ContractError('Candidate solver bounds/tolerances changed')
    exact = dict(input_fps=150.0, input_origin_seconds=0.0, clock_origin_seconds=0.0,
        rod_step_seconds=.0001, BC_RGC_period_seconds=.001, monitor_step_seconds=.0001,
        recording_step_seconds=.0005, native_observation_delay_seconds=.00282)
    if any(p.get(k) != v for k,v in exact.items()):
        raise ContractError('Original event/input/observer semantics changed')
    if sha256(root/p['numerical_budget']['path']) != BUDGET_SHA256:
        raise ContractError('Frozen scientific allocation changed')
    if sha256(root/p['group_artifact']/'manifest.json') != p['group_manifest_sha256']:
        raise ContractError('Fixed full group graph changed')
    if _OVERNIGHT:
        p = dict(p, batch_id=BATCH_ID, budget_limits_seconds=dict(LIMITS),
                 historical_batches=historical_batch_receipts(root),
                 compute_authorization='User authorized continued overnight local implementation and compute; new stage only; old STOP and charges retained')
    if _SHORT_DIAGNOSTIC:
        p = dict(p, checkpoint_seconds=49931*.0001, stop_seconds=49932*.0001,
                 diagnostic_scope_only=True, diagnostic_wall_cap_seconds=600.,
                 diagnostic_note='200us full-graph plus fresh-process suffix; same cumulative ledger and all scientific gates; cannot qualify full8ms or adoption')
    return p


def resource_check(*, rss, available, gpu_used=None, gpu_total=None):
    for name, value in [('rss', rss), ('available', available)]:
        finite_nonnegative(value, name)
    if rss > RESOURCE_LIMITS['max_rss_bytes'] or available < RESOURCE_LIMITS['minimum_available_bytes']:
        raise AcceptanceFailure('RESOURCE_FAIL', 'Original RAM hard gate reached')
    if gpu_used is not None or gpu_total is not None:
        finite_nonnegative(gpu_used, 'GPU used'); finite_nonnegative(gpu_total, 'GPU total')
        cap = min(RESOURCE_LIMITS['max_whole_GPU_bytes'], gpu_total-RESOURCE_LIMITS['minimum_GPU_free_bytes'])
        if gpu_total <= 0 or gpu_used > cap:
            raise AcceptanceFailure('RESOURCE_FAIL', 'Whole-GPU hard gate or physical VRAM reserve reached')


def comparison_status(result):
    if result.get('status') == 'PASS':
        return 'PASS'
    gates = [str(x.get('gate', '')) for x in result.get('issues', [])]
    if any('domain' in gate for gate in gates):
        return 'DOMAIN_FAIL'
    if any(gate.startswith(('draw', 'aux', 'pending_events', 'queue', 'stamp')) for gate in gates):
        return 'EVENT_FAIL' if result.get('status') == 'FAIL' else 'INCONCLUSIVE'
    return 'NUMERICAL_FAIL' if result.get('status') == 'FAIL' else 'INCONCLUSIVE'


def performance_result(pairs, candidate='radau'):
    """A close single result requests a repeat; tenfold is only a milestone."""
    candidate_modes(candidate)
    if not pairs:
        return dict(status='INCONCLUSIVE', speedup=None, adoptable=False, reason='No full paired timing')
    ref = [finite_nonnegative(p['reference'], 'reference hot time') for p in pairs]
    cand = [finite_nonnegative(p[candidate], 'candidate hot time') for p in pairs]
    if min(ref+cand) <= 0:
        raise ContractError('Positive complete neural times required')
    import statistics
    ratio = statistics.median(ref)/statistics.median(cand)
    if ratio <= 1:
        return dict(status='NO_SPEEDUP', speedup=ratio, adoptable=False)
    separated = max(cand) < min(ref)
    if not separated or len(pairs) == 1 and ratio < 1.1:
        return dict(status='INCONCLUSIVE', speedup=ratio, adoptable=False,
            reason='TIMING_RECHECK_REQUIRED', note='10 percent is a repeat trigger, not an adoption speed threshold')
    return dict(status='PERFORMANCE_BELOW_10X' if ratio < 10 else 'PERFORMANCE_AT_LEAST_10X',
        speedup=ratio, adoptable=True, measured_pairs=len(pairs), observed_ranges_separated=separated)


class BudgetLedger(AbstractContextManager):
    """Only the supervisor writes this fixed ledger; stale ownership fails closed."""
    def __init__(self, root, *, clock=time.time):
        self.root = Path(root).resolve()
        self.parent = self.root/REPORT_PARENT
        self.path = self.parent/'budget_ledger.json'
        self.lock = self.parent/'budget_ledger.lock'
        self.clock = clock
        self.data = None
        self.owned = False

    def __enter__(self):
        self.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.lock.open('x', encoding='utf8') as stream:
                json.dump(dict(pid=os.getpid(), batch_id=BATCH_ID, started_unix=self.clock()), stream)
        except FileExistsError as exc:
            raise ContractError('Budget ledger is owned or orphaned; no unchecked stale-lock override') from exc
        self.owned = True
        try:
            if self.path.exists():
                self.data = read_json(self.path)
                witness = self.data.pop('ledger_hash', None)
                if witness != canonical_hash(self.data):
                    raise ContractError('Cumulative budget ledger integrity differs')
                if (self.data.get('schema') != SCHEMA or self.data.get('batch_id') != BATCH_ID
                        or self.data.get('root') != str(self.root) or self.data.get('limits') != LIMITS):
                    raise ContractError('Budget identity/path/limits changed')
                for k in LIMITS:
                    finite_nonnegative(self.data['used'][k], k)
                if self.data.get('active') is not None:
                    raise ContractError('Orphaned unfinished attempt cannot be resumed or reset')
            else:
                # Existing run evidence without the ledger must not acquire a fresh budget.
                if any(x.name not in ('budget_ledger.lock',) for x in self.parent.iterdir()):
                    raise ContractError('Existing batch evidence without its ledger; refusing budget reset')
                self.data = dict(schema=SCHEMA, batch_id=BATCH_ID, root=str(self.root), limits=dict(LIMITS),
                    used=dict.fromkeys(LIMITS, 0.0), attempts=[], active=None, stopped=False)
                if _OVERNIGHT:
                    self.data['historical_batches'] = historical_batch_receipts(self.root)
                self._save()
            if _OVERNIGHT and self.data.get('historical_batches') != historical_batch_receipts(self.root):
                raise ContractError('New batch lost its unchanged historical accounting receipt')
            if self.data.get('stopped') or (self.parent/'REQUEST_STOP').exists():
                raise AcceptanceFailure('RESOURCE_FAIL', 'Parent acceptance batch is stopped')
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _save(self):
        write_json(self.path, dict(self.data, ledger_hash=canonical_hash(self.data)))

    def begin(self, output, phase, identity):
        output = Path(output).resolve()
        if phase not in ('feasibility', 'paired') or output.parent != self.parent or output.exists():
            raise ContractError('Fresh immediate child output of the fixed batch directory required')
        if self.data['active'] is not None:
            raise ContractError('Concurrent acceptance attempt forbidden')
        if self.data['stopped']:
            raise AcceptanceFailure('RESOURCE_FAIL', 'Stopped batch cannot start another attempt')
        for attempt in self.data['attempts']:
            stop_path = Path(attempt['output'])/'REQUEST_STOP'
            if stop_path.exists():
                try:
                    stop = read_json(stop_path)
                    repairable = (isinstance(stop, dict) and stop.get('run_id') == attempt['run_id']
                        and stop.get('status') == attempt['status']
                        and stop.get('status') in FAILURES-{'RESOURCE_FAIL'})
                except (ValueError, ContractError):
                    repairable = False
                if not repairable:
                    self.stop('Previous explicit/protection STOP blocks this batch')
                    raise AcceptanceFailure('RESOURCE_FAIL', 'Previous explicit/protection STOP blocks this batch')
        same_phase = [a for a in self.data['attempts'] if a['phase'] == phase]
        if same_phase:
            prior = same_phase[-1]
            repeat_allowed = prior.get('retry_kind') == 'timing_recheck' or prior['status'] in ('PERFORMANCE_BELOW_10X', 'PERFORMANCE_AT_LEAST_10X', 'NO_SPEEDUP')
            if prior['identity'] == canonical_hash(identity) and not repeat_allowed:
                raise ContractError('Necessary retry requires changed actual source identity')
        for category in ('cold', phase):
            if self.data['used'][category] >= LIMITS[category]:
                self.stop('Exhausted '+category+' budget')
                raise AcceptanceFailure('RESOURCE_FAIL', 'Cumulative '+category+' budget exhausted')
        token = uuid.uuid4().hex
        attempt = dict(run_id=token, phase=phase, output=str(output), identity=canonical_hash(identity),
            started_unix=self.clock(), status='RUNNING', used=dict.fromkeys(LIMITS, 0.0))
        self.data['attempts'].append(attempt); self.data['active'] = token
        output.mkdir(); self._save()
        return token

    def charge(self, category, seconds):
        if category not in LIMITS or self.data['active'] is None:
            raise ContractError('Active known budget category required')
        seconds = finite_nonnegative(seconds, 'elapsed duration')
        self.data['used'][category] += seconds
        self.data['attempts'][-1]['used'][category] += seconds
        self._save()
        if self.data['used'][category] >= LIMITS[category]:
            self.stop('Cumulative '+category+' wall budget reached')
            raise AcceptanceFailure('RESOURCE_FAIL', 'Cumulative '+category+' wall budget reached')

    def stop(self, reason):
        self.data['stopped'] = True
        self.data['stop_reason'] = str(reason)
        write_json(self.parent/'REQUEST_STOP', dict(batch_id=BATCH_ID, reason=str(reason), used=self.data['used']))
        self._save()

    def finish(self, status, *, retry_kind=None):
        if self.data['active'] is None:
            raise ContractError('No active attempt to finish')
        if retry_kind not in (None, 'timing_recheck'):
            raise ContractError('Unknown retry qualification')
        self.data['attempts'][-1].update(status=status, finished_unix=self.clock(), retry_kind=retry_kind)
        self.data['active'] = None
        if status == 'RESOURCE_FAIL':
            self.stop('Resource protection stopped acceptance batch')
        self._save()

    def __exit__(self, exc_type, exc, traceback):
        if self.owned:
            self.lock.unlink(missing_ok=True)
            self.owned = False
        return False
