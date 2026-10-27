"""Prepare/freeze then fit the isolated, single-tau homogeneous HC teacher."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import sparse, __version__ as scipy_version
from scipy.optimize import minimize_scalar

from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.hc_teacher_calibration import (BRANCH, IDENTITY, TRAIN, HELDOUT,
    DIAGNOSTIC, FREQUENCIES, SOURCE_FILES, DerivedHomogeneousTeacher,
    verify_source_files, load_targets, sha256, make_homogeneous_source,
    smith_transfer, normalized, teacher_loss, frequency_response, time_domain_qa)

SOURCE = ROOT/'data/source_data/h1_model_teacher_v1'
CONFIG = ROOT/'configurations/calibration/hc_teacher_tau_h_v1.json'
CODE = ('retina_sim/hc_teacher_calibration.py', 'tools/calibrate_hc_teacher.py',
        'retina_sim/vh2005.py', 'retina_sim/hybrid.py', 'retina_sim/contracts.py',
        'retina_sim/core_sources.py', 'retina_sim/provenance.py',
        'sources/registry.json', 'sources/core_registry.json',
        'retina_sim/background_checkpoint.py', 'retina_sim/canonical_stream.py',
        'schemas/hc_derived_teacher_profile_v1.schema.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, data):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def model(source, tau, protocol_hash, parent=None):
    return DerivedHomogeneousTeacher.create(source, tau, protocol_hash,
        parent or canonical_hash(dict(source_model_hash=source.model_hash, source_tau_h_ms=20.)))


def spatial_check(config):
    binding = config['spatial_check']
    path = ROOT/binding['manifest']
    if sha256(path) != binding['sha256']:
        raise ContractError('Frozen spatial manifest differs')
    manifest = read(path)
    base = path.parent
    bindings = {}
    def bound(relative, matrix=False):
        p = base/relative
        entry = manifest['files'][relative]
        if p.stat().st_size != entry['bytes'] or sha256(p) != entry['sha256']:
            raise ContractError('Spatial bound asset differs: '+relative)
        bindings[relative] = entry
        if not matrix:return np.load(p, allow_pickle=False)
        with np.load(p,allow_pickle=False) as z:
            if set(z.files) != {'data','indices','indptr','shape'}:
                raise ContractError('Explicit frozen CSR layout required')
            result=sparse.csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']))
        result.check_format(full_check=True)
        return result
    A = bound('routes/cones_to_HC/mean_weights.npz', True)
    B = bound('routes/HC_to_cones/mean_weights.npz', True)
    for m in (A, B):
        if np.any(m.data < 0) or not np.isfinite(m.data).all() or not np.allclose(m.sum(1), 1., atol=1e-12, rtol=0):
            raise ContractError('Frozen spatial A/B fails original unit-DC contract')
    rf = bound('populations/cones/centroid_mm.npy')
    keys = bound('populations/HC/keys.npy')
    count = bound('populations/HC/counts.npy')
    ids = bound('populations/HC/group_ids.npy')
    scale = manifest['configuration']['mm_per_degree']
    h1 = manifest['populations']['HC']['subtype_labels'].index('H1')
    eligible = np.flatnonzero((keys[:,1] == h1) & (count == 1))
    incoming_rf = A@rf/scale
    row = int(eligible[np.argmin(np.linalg.norm(incoming_rf[eligible]-[4.73,0], axis=1))])
    center = incoming_rf[row]
    lo = bound('populations/cones/bounds_min_mm.npy')/scale-center
    hi = bound('populations/cones/bounds_max_mm.npy')/scale-center
    nearest = np.where(lo > 0, lo, np.where(hi < 0, hi, 0.))
    farthest = np.maximum(abs(lo),abs(hi))
    straddles = (np.linalg.norm(nearest,axis=1) < 2.5) & (np.linalg.norm(farthest,axis=1) > 2.5)
    background = bound('input/group_background_native_td.npy')
    fine = ROOT/manifest['configuration']['graph']['path']
    # Aggregate means cannot restore member dynamics at a disk boundary.
    return dict(status='SPATIAL_SOURCE_MATCH_BLOCKED', evaluation='NOT_EVALUATED',
        manifest=binding, assets_verified=bindings,
        selected_H1_group_id=str(ids[row]), selected_H1_group_row=row,
        selection='singleton H1 nearest engineering4.73deg using incoming cone RF mean, before any loss',
        incoming_cone_RF_center_deg=center.tolist(), disk_diameter_deg=5.,
        HC_coordinate_role=manifest['populations']['HC']['coordinate_role'],
        cone_coordinate_role=manifest['populations']['cones']['coordinate_role'],
        A_shape=list(A.shape), B_shape=list(B.shape), A_nnz=A.nnz, B_nnz=B.nnz,
        fine_graph_path=str(fine.relative_to(ROOT)), fine_graph_present=fine.is_file(),
        aggregate_cone_bounding_boxes_crossing_disk=int(straddles.sum()),
        existing_background_range_td=[float(background.min()),float(background.max())],
        requested_backgrounds_td=[1000,100],
        blockers=['Original fine-cell graph unavailable; frozen representative groups cannot be expanded to original A/B',
            'Disk boundary can split multi-member RF groups; no exact member-specific nonlinear dynamics certified',
            'HC soma is not a functional RF/IS disk center; A-weighted cone RF is only an engineering center, no Smith registration'],
        production_graph_modified=False, cropped=False, renormalized=False,
        homogeneous_equals_five_degree_disk=False)


def source_baseline(config, protocol_hash):
    receipt = verify_source_files(SOURCE)
    targets, rows = load_targets(SOURCE)
    with (SOURCE/'smith2001_fig2_published_fit_parameters.csv').open(encoding='utf8', newline='') as stream:
        parameters = list(csv.DictReader(stream))
    errors = []
    for p in parameters:
        condition = p['condition_id']
        predicted = smith_transfer(FREQUENCIES, p)
        supplied = targets[condition]
        subset = [r for r in rows if r['condition_id'] == condition]
        errors.append(dict(condition_id=condition,
            complex_relative_error=float(max(abs(predicted-supplied)/abs(supplied))),
            amplitude_relative_error=float(max(abs(abs(predicted)-[float(r['amplitude_mV_per_td']) for r in subset])/abs(predicted))),
            circular_phase_error_rad=float(max(abs(np.angle(predicted/np.exp(1j*np.array([float(r['phase_unwrapped_rad']) for r in subset])))))),
            normalized_amplitude_error=float(max(abs(abs(normalized(predicted))-[float(r['amplitude_normalized_to_0p61Hz']) for r in subset])))))
    maximum = max(r['complex_relative_error'] for r in errors)
    if maximum > config['numerical_thresholds']['source_complex_relative_error']:
        raise ContractError('Smith analytic source QA mismatch')
    anchors = read(SOURCE/'vh05_independent_steady_anchors.json')
    checks = []
    for anchor in anchors['steady_states']:
        source = make_homogeneous_source(ROOT, anchor['I_td'])
        student = model(source, 20., protocol_hash)
        y = student.equilibrium
        values = dict(X=y[2], C=y[3], Vis_mV=y[4], gi=y[5], Vis_prime_mV=y[6],
            U=y[7], Vb=y[8], Vh_mV=y[9], aI=(y[6]/source.p['V_I'])**source.p['mu'],
            Vs_mV=y[4]-y[9], It=y[7])
        error = max(abs(value-anchor[key])/max(abs(anchor[key]),1.) for key,value in values.items())
        rhs = max(abs(student.rhs(0,y,anchor['I_td'])))
        if error > config['numerical_thresholds']['steady_anchor_relative_error'] or rhs > config['numerical_thresholds']['steady_rhs_max_native_per_second']:
            raise ContractError('Independent vH steady anchor/RHS mismatch')
        checks.append(dict(background_td=anchor['I_td'], maximum_scaled_anchor_error=float(error),
            rhs_max_native_per_second=float(rhs), source_model_hash=source.model_hash,
            source_parameter_hash=source.parameters.parameter_hash))
    assets=[]
    for asset in read(ROOT/'sources/core_registry.json')['sources']:
        if not asset['source_id'].startswith('VH2005'):continue
        path=ROOT/asset['source_path']
        status='MISSING_NOT_REVERIFIED'
        if path.is_file():
            if sha256(path)!=asset['asset_sha256']:raise ContractError('Original vH asset identity mismatch')
            status='LOCAL_BYTES_MATCH_REGISTRY'
        assets.append(dict(source_id=asset['source_id'],path=asset['source_path'],expected_sha256=asset['asset_sha256'],status=status))
    return dict(status='PASS_SOURCE_QA_ONLY', identity=IDENTITY, canonical_files=receipt,
        source_parameter_admission='ISOLATED_PINNED_TRANSCRIPTION_REPLAY_NOT_GLOBAL_REGISTRY_VALIDATION',
        original_vH_assets=assets,
        smith_analytic_checks=errors, steady_anchor_checks=checks,
        warning='Anchor field names containing mV are retained source labels, not a calibrated native-to-physical-mV map. Steady anchors carry no tau_h information. Smith and vH share Smith evidence, not independent biological validation.')


def prepare(output, config):
    output.mkdir(parents=True, exist_ok=False)
    identity = sha256(CONFIG)
    spatial = spatial_check(config)
    baseline = source_baseline(config, identity)
    write(output/'SOURCE_BASELINE.json', baseline)
    write(output/'SPATIAL_SOURCE_MATCH.json', spatial)
    write(output/'PROTOCOL.json', config)
    write(output/'FROZEN.json', dict(frozen_utc=now(), optimization_evaluations=0,
        protocol_sha256=sha256(output/'PROTOCOL.json'), configuration_sha256=identity,
        source_files=verify_source_files(SOURCE), code_sha256={p:sha256(ROOT/p) for p in CODE},
        source_baseline_sha256=sha256(output/'SOURCE_BASELINE.json'),
        spatial_check_sha256=sha256(output/'SPATIAL_SOURCE_MATCH.json'),
        source_directory=str(SOURCE), source_contents_read_only=True))
    print(json.dumps(dict(stage='PREPARED_AND_FROZEN', output=str(output), source_baseline=baseline['status'], spatial=spatial['status'])))


def fit(output, config):
    frozen = read(output/'FROZEN.json')
    if frozen['configuration_sha256'] != sha256(CONFIG) or frozen['protocol_sha256'] != sha256(output/'PROTOCOL.json'):
        raise ContractError('Protocol changed after freeze')
    for p,digest in frozen['code_sha256'].items():
        if sha256(ROOT/p) != digest:
            raise ContractError('Code/source changed after freeze: '+p)
    for name,key in (('SOURCE_BASELINE.json','source_baseline_sha256'),('SPATIAL_SOURCE_MATCH.json','spatial_check_sha256')):
        if sha256(output/name) != frozen[key]:
            raise ContractError('Frozen preparation changed: '+name)
    verify_source_files(SOURCE)
    write(output/'FIT_STARTED.json', dict(started_utc=now(), pid=os.getpid(), protocol_sha256=frozen['protocol_sha256']))
    start, cpu = time.monotonic(), time.process_time()
    memory = MemoryGuard(max_rss_bytes=config['guard']['max_rss_bytes'],min_available_bytes=0)
    max_rss = 0
    def guard():
        nonlocal max_rss
        if any((output/name).exists() for name in config['guard']['stop_files']):
            raise ContractError('Task-local STOP requested')
        measured=memory.check()
        max_rss = max(max_rss, memory.peak_rss_bytes)
        if not measured:
            raise ContractError('Owned process RSS limit reached')
    protocol_hash = frozen['protocol_sha256']
    sources = {bg:make_homogeneous_source(ROOT,bg) for bg in (1000,100,10)}
    thresholds = config['numerical_thresholds']
    numerical, time_checks = [], []
    def qa(tau, time_check=False):
        for bg,source in sources.items():
            guard()
            student = model(source,tau,protocol_hash)
            primary,details = frequency_response(student)
            differences = []
            for step in config['jacobian_relative_steps']:
                other,check = frequency_response(student,step)
                differences.append(float(max(abs(normalized(other)-normalized(primary)))))
                if check['max_solve_relative_residual'] > thresholds['linear_solve_relative_residual']:
                    raise ContractError('Linear solve residual failed')
            if max(differences) > thresholds['jacobian_normalized_transfer_max_difference'] or details['steady_rhs_max_native_per_second'] > thresholds['steady_rhs_max_native_per_second']:
                raise ContractError('Jacobian scale/steady QA failed')
            lower = make_homogeneous_source(ROOT,bg*(1-1e-4))
            upper = make_homogeneous_source(ROOT,bg*(1+1e-4))
            dc = (upper.initial_conditions()[-1].values[0]-lower.initial_conditions()[-1].values[0])/(2*bg*1e-4)
            if dc >= 0:
                raise ContractError('Light increment does not hyperpolarize H1')
            numerical.append(dict(tau_h_ms=tau,background_td=bg,
                normalized_transfer_max_step_difference=max(differences),
                light_increment_DC_native_per_td=float(dc),**details))
            if time_check and bg in config['time_qa']['backgrounds_td']:
                for contrast in config['time_qa']['contrasts']:
                    result = time_domain_qa(student,contrast,guard)
                    time_checks.append(dict(tau_h_ms=tau,background_td=bg,**result))
                    print(json.dumps(dict(stage='TIME_QA',tau_h_ms=tau,background_td=bg,**result)),flush=True)
                    if result['relative_complex_error'] > thresholds['time_complex_relative_error'] or result['two_half_window_relative_difference'] > thresholds['time_half_window_relative_difference']:
                        raise ContractError('Predetermined time-domain QA threshold failed')
        write(output/('NUMERICAL_QA_tau_'+str(tau)+'.json'),dict(jacobian=numerical[-3:],time=[r for r in time_checks if r['tau_h_ms']==tau]))
    for tau in config['time_qa']['tau_points_ms']:
        qa(tau,True)
    targets,_ = load_targets(SOURCE)
    trace,cache = [],{}
    def objective(eta, role='optimizer'):
        guard()
        tick = time.monotonic()
        tau = float(20*np.exp(eta))
        if tau < 10 and tau > 10-1e-12: tau=10.
        if tau > 40 and tau < 40+1e-12: tau=40.
        if tau in cache:return cache[tau]
        if len(trace) >= config['optimizer']['max_objective_evaluations']:
            raise ContractError('Objective evaluation limit reached')
        try:
            transfer,check = frequency_response(model(sources[1000],tau,protocol_hash))
            loss = teacher_loss(transfer, targets[TRAIN])
            if not np.isfinite(loss):raise ContractError('Nonfinite objective')
            if time.monotonic()-tick > config['guard']['objective_max_wall_seconds']:
                raise ContractError('Objective per-call timeout')
        except Exception as error:
            with (output/'invalid_candidates.jsonl').open('a',encoding='utf8') as stream:
                stream.write(json.dumps(dict(eta=float(eta),tau_h_ms=tau,error=repr(error)))+'\n')
            raise
        row = dict(evaluation=len(trace)+1,role=role,eta=float(eta),tau_h_ms=tau,
            loss=loss,wall_seconds=time.monotonic()-tick,maximum_pole_real_per_second=check['maximum_pole_real_per_second'])
        trace.append(row);cache[tau]=loss
        with (output/'optimizer_trace.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(row)+'\n')
        return loss
    profile = [dict(tau_h_ms=tau,loss=objective(np.log(tau/20),'predeclared_loss_profile')) for tau in config['loss_profile_ms']]
    result = minimize_scalar(objective,method='bounded',bounds=(np.log(.5),np.log(2)),
        options=dict(xatol=config['optimizer']['xatol_eta'],maxiter=config['optimizer']['maxiter']))
    if not result.success:raise ContractError('Bounded scalar optimizer failed: '+str(result.message))
    selected_tau = min(cache,key=cache.get)
    if selected_tau not in config['time_qa']['tau_points_ms']:qa(selected_tau,True)
    baseline_model = model(sources[1000],20.,protocol_hash)
    candidate = model(sources[1000],selected_tau,protocol_hash,baseline_model.candidate_hash)
    write(output/'DERIVED_PROFILE.json',dict(status='VALID_DERIVED_ONLY_NOT_ADOPTED',
        baseline_profile=asdict(baseline_model.profile),profile=asdict(candidate.profile),
        candidate_hash=candidate.candidate_hash,source_parameter_qualified=False,
        overrides={'tau_h':{'initial_ms':20.,'final_ms':selected_tau}},
        production_written=False,downstream_cache_invalidation_required_on_adoption=True))
    conditions, predictions = {},[]
    loss_numerical_error = 0.
    for condition,target in targets.items():
        bg = int(condition.rsplit('_',1)[1].removesuffix('td'))
        values = {}
        for label,tau in (('baseline',20.),('candidate',selected_tau)):
            student = model(sources[bg],tau,protocol_hash)
            raw,_ = frequency_response(student)
            z,t = normalized(raw),normalized(target)
            loss = teacher_loss(raw,target)
            if condition==TRAIN:
                for step in config['jacobian_relative_steps']:
                    alternative,_=frequency_response(student,step)
                    loss_numerical_error=max(loss_numerical_error,abs(teacher_loss(alternative,target)-loss))
            values[label]=dict(loss=loss,mean_abs_amplitude_error=float(np.mean(abs(abs(z[1:])-abs(t[1:])))),
                mean_abs_circular_phase_error_rad=float(np.mean(abs(np.angle(z[1:]/t[1:])))))
            for i,f in enumerate(FREQUENCIES):
                predictions.append(dict(condition_id=condition,split='train' if condition==TRAIN else 'heldout' if condition in HELDOUT else 'diagnostic',
                    model=label,tau_h_ms=tau,frequency_Hz=float(f),is_normalization_anchor=i==0,
                    student_native_per_td_real=float(raw[i].real),student_native_per_td_imag=float(raw[i].imag),
                    teacher_normalized_real=float(t[i].real),teacher_normalized_imag=float(t[i].imag),
                    student_normalized_real=float(z[i].real),student_normalized_imag=float(z[i].imag),
                    normalized_amplitude_error=float(abs(z[i])-abs(t[i])),
                    circular_relative_phase_error_rad=float(np.angle(z[i]/t[i])),
                    squared_complex_error=float(abs(z[i]-t[i])**2)))
        conditions[condition]=values
    if loss_numerical_error > thresholds['loss_numerical_error_ceiling']:
        raise ContractError('Loss numerical error exceeds frozen ceiling')
    train = conditions[TRAIN]
    improvement = train['baseline']['loss']-train['candidate']['loss']
    required = max(config['acceptance']['minimum_training_absolute_improvement'],
        config['acceptance']['minimum_training_relative_improvement']*train['baseline']['loss'],
        config['acceptance']['improvement_over_observed_numerical_error_factor']*loss_numerical_error)
    train_pass = improvement>required
    holdout_pass = all(conditions[c]['candidate']['loss'] <= conditions[c]['baseline']['loss']+
        config['acceptance']['every_100td_holdout_must_not_degrade_beyond_absolute_loss'] for c in HELDOUT)
    write(output/'TRAIN.json',dict(status='IMPROVED_ABOVE_NUMERICAL_ERROR' if train_pass else 'NO_ACCEPTABLE_IMPROVEMENT',condition=TRAIN,
        **train,absolute_improvement=improvement,required_improvement=required,
        observed_Jacobian_step_loss_error=loss_numerical_error,training_frequencies_excluding_anchor=11))
    write(output/'HELDOUT.json',dict(status='NONDEGRADED_ALL_100TD' if holdout_pass else 'FAILED_SOURCE_CONDITION_GENERALIZATION',
        primary_condition=HELDOUT[0],conditions={c:conditions[c] for c in HELDOUT},
        interpretation=config['holdout_interpretation']))
    write(output/'DIAGNOSTIC.json',dict(status='REPORTED_NOT_USED_IN_OPTIMIZER',conditions={c:conditions[c] for c in DIAGNOSTIC}))
    write(output/'TEACHER_FIT.json',dict(status='ACCEPTED_CONDITIONAL_SOURCE_TEACHER' if train_pass and holdout_pass else 'FAILED_PREDECLARED_TEACHER_ACCEPTANCE',
        identity=IDENTITY,branch=BRANCH,initial_tau_h_ms=20.,candidate_tau_h_ms=selected_tau,
        nonzero_parameter_update=selected_tau!=20.,eta=float(np.log(selected_tau/20)),
        training_improved=train_pass,heldout_nondegraded=holdout_pass,
        optimizer=dict(success=bool(result.success),message=str(result.message),nfev=int(result.nfev),nit=int(result.nit),
            total_distinct_evaluations=len(trace),best_observed_loss=float(cache[selected_tau])),
        boundary=bool(selected_tau in (10.,40.)),loss_profile=profile,
        identifiability='One scalar engineering profile; no physiological confidence interval. Boundary optimum or univariate profile cannot identify other mechanisms.',
        candidate_adopted=False))
    write(output/'PHYSIOLOGICAL_CALIBRATION.json',dict(status='NOT_EVALUATED',reason=config['claim_limits']))
    with (output/'frequency_predictions_errors.csv').open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    write(output/'NUMERICAL_QA.json',dict(status='PASS',jacobian=numerical,time_domain=time_checks,
        observed_training_loss_numerical_error=loss_numerical_error,
        no_pole_clipping=True,fixed_readout_delay_ms=2.82,feedback_delay_ms=0,
        native_to_physical_mV_gain='UNCALIBRATED'))
    guard()
    write(output/'RESOURCES.json',dict(pid=os.getpid(),owned_process_only=True,active_workers_after_exit=0,
        start_utc=read(output/'FIT_STARTED.json')['started_utc'],finished_utc=now(),
        wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,max_rss_bytes=max_rss,
        python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,scipy=scipy_version,
        time_qa_total_nfev=sum(r['nfev'] for r in time_checks),
        numerical_guard=config['guard'],full_chain_or_long_movie_run=False))
    write(output/'COMPLETE.json',dict(status='COMPLETED_INDEPENDENT_TEACHER_ATTEMPT',
        fit_acceptance=read(output/'TEACHER_FIT.json')['status'],source_baseline='PASS_SOURCE_QA_ONLY',
        derived_profile='VALID_DERIVED_ONLY_NOT_ADOPTED',train=read(output/'TRAIN.json')['status'],
        heldout=read(output/'HELDOUT.json')['status'],spatial_source_match='SPATIAL_SOURCE_MATCH_BLOCKED',
        physiological_calibration='NOT_EVALUATED',production_written=False,active_workers=0,finished_utc=now(),
        frozen_sha256=sha256(output/'FROZEN.json'),reproduce='Use a NEW output directory: python tools/calibrate_hc_teacher.py prepare --output <new>; then fit --output <new>'))
    print(json.dumps(read(output/'COMPLETE.json')),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('prepare','fit'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'reports'):
        raise ContractError('New task output must be inside reports')
    config=read(CONFIG)
    if args.stage=='prepare':
        try:prepare(output,config)
        except BaseException as error:
            if output.is_dir():
                write(output/('PREPARATION_FAILURE_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                    dict(status='FAILED_PREPARATION_NO_OPTIMIZATION',utc=now(),error=repr(error),traceback=traceback.format_exc()))
            raise
        return
    lock=output/'OWNED_PROCESS.lock'
    with lock.open('x',encoding='utf8') as stream:stream.write(str(os.getpid()))
    try:
        fit(output,config)
    except BaseException as error:
        failure=output/('FAILURE_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
        write(failure,dict(status='FAILED_OR_STOPPED',utc=now(),pid=os.getpid(),error=repr(error),traceback=traceback.format_exc(),production_written=False))
        raise
    finally:
        lock.unlink()


if __name__=='__main__':main()
