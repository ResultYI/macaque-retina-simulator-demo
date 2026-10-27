"""Opt-in, homogeneous source-model teacher work; never a production profile.

The strict NativeConeH1 is composed, not patched. Its complete ten-state RHS
is reused; only the final derivative is rescaled by source_tau_h/derived_tau_h.
This preserves dynamic pooled_aI, feedback, input units and the source lock.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import json
import re
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy import sparse

from .contracts import ContractError, Quantity, RunMode, canonical_hash
from .provenance import ParameterSet, ParameterRecord, Profile
from .vh2005 import NativeConeH1, PROFILE, SOURCE_VALUES, PARAMETER_UNITS

IDENTITY = 'MODEL_DERIVED_FROM_PUBLISHED_FIT_NOT_RAW'
BRANCH = 'homogeneous_native_source_model_teacher'
FREQUENCIES = np.array([.61, 1, 2, 4, 6, 8, 10, 20, 30, 40, 60, 78])
TRAIN = 'smith2001_fig2_a_1000td'
HELDOUT = tuple('smith2001_fig2_'+c+'_100td' for c in 'abc')
DIAGNOSTIC = ('smith2001_fig2_b_1000td', 'smith2001_fig2_c_1000td',
              *( 'smith2001_fig2_'+c+'_10td' for c in 'abc'))
SOURCE_FILES = {
    'smith2001_h1_calibration_manifest.json': (8713, '89a85ea7cb525a35d99211b3bcacf0941e7cbe93010b92c67a610b4810c08900'),
    'smith2001_fig2_published_fit_parameters.csv': (1561, 'b2fd01f36182537343d94e820c8d69057996ac6adfd4b89f45fff814daf914d1'),
    'smith2001_fig2_model_derived_targets.csv': (24848, '409cdcbc2198d3b69020afd05651604a75e28d806761db08dcb0a2dfa3c3d863'),
    'vh05_independent_steady_anchors.json': (6333, '87b155f66c565a7bdfab3c2a228342f1577e5590c63cca5377b1764561217590'),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source_files(directory):
    receipt = {}
    for name, (size, digest) in SOURCE_FILES.items():
        path = Path(directory)/name
        raw = path.read_bytes()
        actual = (len(raw), hashlib.sha256(raw).hexdigest())
        if actual != (size, digest):
            raise ContractError(f'{name}: expected bytes/SHA {(size, digest)}, actual {actual}')
        if raw.startswith(b'\xef\xbb\xbf'):
            raise ContractError('BOM in canonical source')
        receipt[name] = dict(bytes=size, sha256=digest)
    return receipt


def load_targets(directory):
    verify_source_files(directory)
    with (Path(directory)/'smith2001_fig2_model_derived_targets.csv').open(encoding='utf8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 108 or {r['provenance'] for r in rows} != {IDENTITY}:
        raise ContractError('Teacher provenance/row count differs')
    targets = {}
    for condition in (TRAIN, *HELDOUT, *DIAGNOSTIC):
        subset = [r for r in rows if r['condition_id'] == condition]
        f = np.array([float(r['frequency_Hz']) for r in subset])
        if not np.array_equal(f, FREQUENCIES):
            raise ContractError('Teacher grid/order differs: '+condition)
        targets[condition] = np.array([complex(float(r['complex_real_mV_per_td']),
                                             float(r['complex_imag_mV_per_td'])) for r in subset])
    return targets, rows


def normalized(transfer):
    transfer = np.asarray(transfer, dtype=complex)
    if not np.isfinite(transfer).all() or transfer[0] == 0:
        raise ContractError('Nonfinite or zero normalization anchor; no epsilon rescue')
    return transfer/transfer[0]


def teacher_loss(student, teacher):
    return float(np.mean(abs(normalized(student)[1:]-normalized(teacher)[1:])**2))


def smith_transfer(frequencies, row):
    """Source QA only: rounded published fit formula, never appended to student."""
    s = 2j*np.pi*np.asarray(frequencies)
    t1, t2 = float(row['tau1_ms'])*.001, float(row['tau2_ms'])*.001
    return (-float(row['sensitivity_mV_per_td'])*np.exp(-s*.002)/(1+s*.002)**3
            /(1+s*t1)/(1+s*t2)**3*(1+s*.12)/(1+s*.08)
            /(1+2*float(row['zeta_dimensionless'])*s*t2+(s*t2)**2))


def make_homogeneous_source(root, background_td):
    if not np.isfinite(background_td) or background_td < 0:
        raise ContractError('Explicit nonnegative background in td required')
    # Isolated transcription replay: no global registry admission and no claim
    # that the absent original PDF/erratum assets have been reverified.
    registry_path = Path(root)/'sources/core_registry.json'
    data = json.loads(registry_path.read_text(encoding='utf8'))
    profile_data = next(p for p in data['profiles'] if p['profile_id'] == PROFILE)
    profile = Profile(**dict(profile_data,parameter_ids=tuple(profile_data['parameter_ids'])))
    by_id = {r['parameter_id']:r for r in data['parameters']}
    records = tuple(ParameterRecord(**dict(by_id[key],shape=tuple(by_id[key]['shape']),
        allowed_modes=tuple(by_id[key]['allowed_modes']))) for key in profile.parameter_ids)
    values = {r.parameter_id.removeprefix('VH2005_'):float(r.value) for r in records}
    units = {r.parameter_id.removeprefix('VH2005_'):r.unit for r in records}
    if values != SOURCE_VALUES or units != PARAMETER_UNITS:
        raise ContractError('Pinned original source parameter transcription/units differs')
    parameters = ParameterSet(records,profile,RunMode.REFERENCE,canonical_hash(dict(
        registry_sha256=sha256(registry_path),admission='ISOLATED_TRANSCRIPTION_REPLAY_ONLY',
        original_asset_verification='NOT_REVERIFIED_MISSING_LOCAL_ASSETS')))
    tag = 'model_derived_source_teacher'
    def population(kind, subtype):
        return SimpleNamespace(cells=SimpleNamespace(cell_id=(BRANCH+':'+subtype,),
            cell_type=(kind,), subtype=(subtype,), evidence_tags=(tag,)))
    return NativeConeH1(parameters, population('cone', 'L'), population('HC', 'H1'),
        sparse.eye(1, format='csr'), sparse.eye(1, format='csr'),
        Quantity([background_td], 'td', 'native_retinal_illuminance', tag),
        pooling_policy='pool_Vbar_before_aI',
        geometry_frame='homogeneous_source_no_spatial_coordinates',
        observation_interval_seconds=.0001)


@dataclass(frozen=True)
class DerivedTeacherProfile:
    schema: str
    profile_id: str
    branch: str
    source_profile_id: str
    source_parameter_hash: str
    source_model_hash: str
    source_implementation_sha256: str
    tau_h_ms: float
    parent_candidate_hash: str
    protocol_sha256: str
    source_parameter_qualified: bool = False

    def __post_init__(self):
        if (self.schema != 'hc-derived-teacher-profile-1'
                or self.profile_id != 'VH2005_derived_HC_teacher_tau_h_v1'
                or self.branch != BRANCH or self.source_profile_id != PROFILE
                or self.source_parameter_qualified is not False
                or not np.isfinite(self.tau_h_ms) or not 10 <= self.tau_h_ms <= 40):
            raise ContractError('Invalid derived teacher profile schema/identity')
        for value in (self.source_parameter_hash,self.source_model_hash,
                      self.source_implementation_sha256,self.parent_candidate_hash,self.protocol_sha256):
            if not re.fullmatch('[0-9a-f]{64}',value):
                raise ContractError('Derived profile requires SHA256 source/lineage identities')


@dataclass(frozen=True)
class DerivedHomogeneousTeacher:
    source: NativeConeH1
    profile: DerivedTeacherProfile

    @classmethod
    def create(cls, source, tau_h_ms, protocol_sha256, parent_candidate_hash):
        if (source.parameters.profile.profile_id != PROFILE or dict(source.p) != SOURCE_VALUES
                or source.nc != 1 or source.nh != 1
                or source.geometry_frame != 'homogeneous_source_no_spatial_coordinates'
                or not np.array_equal(source.A.toarray(), [[1.]])
                or not np.array_equal(source.B.toarray(), [[1.]])):
            raise ContractError('Strict homogeneous source with intact source lock required')
        if not np.isfinite(tau_h_ms) or not 10 <= tau_h_ms <= 40:
            raise ContractError('Derived tau_h outside predeclared [10,40] ms')
        if not protocol_sha256 or not parent_candidate_hash:
            raise ContractError('Explicit protocol and candidate lineage required')
        profile = DerivedTeacherProfile('hc-derived-teacher-profile-1',
            'VH2005_derived_HC_teacher_tau_h_v1', BRANCH, PROFILE,
            source.parameters.parameter_hash, source.model_hash,
            sha256(Path(__file__).with_name('vh2005.py')), float(tau_h_ms),
            parent_candidate_hash, protocol_sha256)
        return cls(source, profile)

    @property
    def candidate_hash(self):
        return canonical_hash(asdict(self.profile))

    @property
    def equilibrium(self):
        return np.concatenate([q.values for q in self.source.initial_conditions()])

    def rhs(self, t_seconds, y, light_td):
        y = np.asarray(y)
        if y.shape != (10,) or not np.isfinite(y).all():
            raise ContractError('Complete finite homogeneous ten-state vector required')
        light = Quantity([light_td], 'td', 'native_retinal_illuminance', self.source.tag)
        derivative = self.source.derivative(t_seconds, tuple(y[i:i+1] for i in range(10)), {'light':light})
        result = np.concatenate([q.values for q in derivative])
        # Native RHS has already changed ms to SI once. Only tau_h changes here.
        result[-1] *= self.source.p['tau_h']/self.profile.tau_h_ms
        return result


def linearize(model, relative_step=1e-5):
    y = model.equilibrium
    light = float(model.source.background[0])
    if light <= 0:
        raise ContractError('Linearization requires positive background')
    scale = np.maximum(abs(y), 1.)
    steps = relative_step*scale
    jac = np.empty((10, 10))
    for i, step in enumerate(steps):
        delta = np.eye(10)[i]*step
        jac[:, i] = (model.rhs(0, y+delta, light)-model.rhs(0, y-delta, light))/(2*step)
    step = relative_step*light
    drive = (model.rhs(0, y, light+step)-model.rhs(0, y, light-step))/(2*step)
    poles = np.linalg.eigvals(jac)
    if not np.isfinite(jac).all() or np.max(poles.real) >= -1e-8:
        raise ContractError('Unstable/nonfinite candidate; no pole clipping')
    return jac, drive, poles


def frequency_response(model, relative_step=1e-5):
    jac, drive, poles = linearize(model, relative_step)
    output, residuals = [], []
    for frequency in FREQUENCIES:
        matrix = 2j*np.pi*frequency*np.eye(10)-jac
        state = np.linalg.solve(matrix, drive)
        residuals.append(float(np.linalg.norm(matrix@state-drive)/np.linalg.norm(drive)))
        # Vh is undelayed in feedback; exactly one fixed observation delay.
        output.append(state[-1]*np.exp(-2j*np.pi*frequency*model.source.p['observation_delay']*.001))
    return np.array(output), dict(max_solve_relative_residual=max(residuals),
        poles_per_second=[[float(z.real),float(z.imag)] for z in poles],
        maximum_pole_real_per_second=float(max(poles.real)),
        steady_rhs_max_native_per_second=float(max(abs(model.rhs(0, model.equilibrium, model.source.background[0])))))


def time_domain_qa(model, contrast, guard=lambda:None):
    """Finite integer-cycle check at 20Hz, exact delayed dense-output readout."""
    frequency, warmup, cycles, samples = 20., 2., 4, 128
    light = float(model.source.background[0])
    delay = model.source.p['observation_delay']*.001
    start = time.monotonic()
    evaluations = 0
    def rhs(t, y):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 300000 or time.monotonic()-start > 120:
            raise ContractError('Time-domain QA finite solve limit reached')
        if evaluations % 128 == 0: guard()
        return model.rhs(t, y, light*(1+contrast*np.cos(2*np.pi*frequency*t)))
    solution = solve_ivp(rhs, (0, warmup+cycles/frequency), model.equilibrium,
        method='DOP853', rtol=2e-9, atol=1e-11, max_step=1/(frequency*64), dense_output=True)
    if not solution.success:
        raise ContractError('Small-signal time solve failed: '+solution.message)
    times = warmup+np.arange(cycles*samples)/(samples*frequency)
    values = solution.sol(times-delay)[-1]
    def f1(t, v):
        return 2*np.mean((v-np.mean(v))*np.exp(-2j*np.pi*frequency*t))/(light*contrast)
    transfer = f1(times, values)
    half = len(times)//2
    convergence = abs(f1(times[:half],values[:half])-f1(times[half:],values[half:]))/abs(transfer)
    predicted = frequency_response(model)[0][7]
    return dict(contrast=contrast, frequency_Hz=frequency, warmup_seconds=warmup,
        integer_cycles=cycles, samples_per_cycle=samples, nfev=evaluations,
        wall_seconds=time.monotonic()-start,
        complex_native_per_td=[float(transfer.real),float(transfer.imag)],
        relative_complex_error=float(abs(transfer-predicted)/abs(predicted)),
        two_half_window_relative_difference=float(convergence))
