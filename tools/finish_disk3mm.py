"""Finish the explicitly authorized production build after geometry is frozen.

This consumes an explicit authorization receipt. It does not generate cells,
change source/dynamics, restart failed work, or substitute a smaller ROI.
"""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / 'reports/disk3mm_build_20261016'
REQUEST = REPORT / 'authorized_delivery_request.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.writing')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(pending, path)


def local(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Delivery paths must stay inside the existing project')
    return path


def preflight():
    request = json.loads(REQUEST.read_bytes())
    if request['schema'] != 'authorized_disk3mm_delivery_v1' or request['status'] != 'APPROVED':
        raise ValueError('No matching approved delivery request')
    if Path(sys.executable).resolve() != Path(request['python_executable']).resolve():
        raise ValueError('Delivery must use the approved retipath_v1 environment')
    for relative, expected in request['frozen_identity_guards'].items():
        if sha(local(relative)) != expected:
            raise ValueError('Approved input or source identity changed: ' + relative)
    return request


def coverage(configuration):
    from retina_sim.disk_circuit import GlobalDiskCircuit
    from retina_sim.disk_geometry import DiskAtlasROI
    circuit = GlobalDiskCircuit(ROOT, configuration['global_graph'])
    record = circuit.record
    roi = DiskAtlasROI((-3., 3., -3., 3.), (-3., 3., -3., 3.), record['reference_id'],
                       owner_radius_mm=3., query_radius_mm=3.)
    routes = []
    for descriptor in record['routes']:
        route = circuit.route(descriptor).record
        routes.append(dict(
            route_id=descriptor['route_id'], kind=descriptor['kind'], binding=descriptor['binding'],
            edges=route['edges'], post_rows=route['recipe']['post_count'],
            missing_degree_rows=route['missing_degree_rows'], empty_rows=route['empty_rows']))
    blocked = {}
    for name, value in record['blocked_midget_candidates'].items():
        parent = circuit.table(name.replace('RGC', 'BC'))
        rows = value.get('parent_BC_rows', [])
        owner_count = int(roi.owner_mask(parent.coordinates['rf'][rows]).sum()) if rows else 0
        blocked[name] = {'count_including_support': value['count'], 'owner_count': owner_count,
                         'exact_parent_rows_in': configuration['global_graph']}
    owner_counts = {}
    for name in record['tables']:
        table=circuit.table(name)
        if record['tables'][name]['kind']=='disk_population':
            if table.record['reference_id']!=roi.reference_id:
                raise ValueError('Fixed population reference differs from owner domain')
            owner_counts[name]=int(roi.owner_mask(table.positions).sum())
        else:
            owner_counts[name]=len(table.query_indices(roi,owner=True)) if 'rf' in table.coordinates else None
    hc_operator=record.get('native_operator_representation')
    hc_summary=None if hc_operator is None else {
        'representation':hc_operator['representation'],'profile':hc_operator['profile'],
        'owner_functional_support_complete':hc_operator['operator_audit']['owner_complete'],
        'maximum_owner_functional_distance_mm':hc_operator['operator_audit']['maximum_owner_functional_distance_mm'],
        'anatomical_reference':hc_operator['anatomical_reference']['raw_contacts'],
        'anatomical_owner_no_feedback_cones':hc_operator['anatomical_reference']['owner_zero_feedback_cones'],
        'anatomical_central_coverage':'UNRESOLVED; independent original reference retained',
        'physiological_validation':False}
    midget_rules={}
    from retina_sim.disk_graph_storage import FrozenSparseRoute
    for descriptor in record['routes']:
        if descriptor['pre'].startswith('midget_BC') and descriptor['post'].startswith('midget_RGC'):
            candidate=circuit.route(descriptor).record['recipe'].get('frozen_candidate_pool')
            if candidate:
                value=FrozenSparseRoute(ROOT,candidate).record
                midget_rules[descriptor['post']]={'minimum_degree_rule':value['recipe'].get('minimum_degree_rule','ceil_target_v1'),
                    'original_ceil_deficit_rows':value['missing_degree_rows'],'population_average_target_satisfied':'NOT_ESTABLISHED',
                    'candidate_binding':candidate}
    return {
        'global_graph': configuration['global_graph'], 'owner_disk_radius_mm': 3.,
        'support_population_counts': {name: table['count'] for name, table in record['tables'].items()},
        'owner_population_counts': owner_counts,
        'absent_owner_coordinate_policy':'null means no RF coordinate supplied (widefield computational IPL grid); no physical-coordinate fallback',
        'functional_owner_output_counts': {
            name: owner_counts[name] for name in record['output_populations']},
        'routes': routes, 'native_owner_zero_feedback_cones': record['native_owner_zero_feedback_cones'],
        'blocked_midget_candidates': blocked,
        'HC_operator':hc_summary,'midget_minimum_rules':midget_rules,
        'boundary': 'finite canonical global graph; no exterior feedback or cropped-weight renormalization',
        'physical_soma_displacement': 'NOT_IMPLEMENTED; RF, OPL and IPL roles remain separate',
        'whole_retina_claim': False, 'whole_exterior_recurrent_closure': 'NOT_CERTIFIED',
        'physiological_validation': False,
    }


def adopt(request, result, directory):
    preflight()
    main = ROOT / 'configurations/retina_platform_v1.json'
    design = ROOT / 'docs/core_design.md'
    for path in (main, design):
        expected = request['adoption_identity_guards'][path.relative_to(ROOT).as_posix()]
        if sha(path) != expected:
            raise ValueError('Parallel edits preserved; default adoption needs reconciliation: ' + str(path))
    baseline = json.loads(main.read_bytes())
    original_design = design.read_bytes()
    previous = request['rollback_directory']
    circuit_path = request.get('circuit_configuration_path', 'configurations/atlas_disk3mm_v1.json')
    atlas_profile = request.get('atlas_profile_id', 'retina_atlas_disk3mm_v1')
    defaults = dict(baseline['run_defaults'])
    defaults.update(circuit_configuration_path=circuit_path,
                    owner_x_mm=0., owner_y_mm=0., owner_width_mm=.02, owner_height_mm=.02,
                    max_spatial_cells=2000000, background_checkpoint_path='')
    if request.get('resource_profile'):
        defaults.update(json.loads(local(request['resource_profile']).read_bytes())['run_config'])
    next_config = dict(baseline, run_defaults=defaults,
        active_run_status=request.get('geometry_claim','Frozen3mm finite disk')+'; three actual ISET/matched20ms static-input/background responses passed after unchanged required numerical background convergence; effective dynamics and mixed optics, no physiology or stochastic stationarity claim',
        previous_active_configuration={'path': previous + '/retina_platform_v1.json',
                                      'sha256': request['adoption_identity_guards']['configurations/retina_platform_v1.json']},
        atlas_target={'atlas_profile_id': atlas_profile,
            's_partition_profile': baseline['atlas_target']['s_partition_profile'],
            'production_assembler': 'retina_sim.atlas_population_network.atlas_population_network',
            'global_graph': result['coverage']['global_graph'],
            'active_circuit_configuration': {'path': circuit_path,
                                             'sha256': sha(ROOT / circuit_path)},
            'complete_replacement_status': 'REPRESENTATIVE_NUMERICALLY_PREPARED_RESPONSE_ACCEPTED',
            'accepted_run': directory.relative_to(ROOT).as_posix(), 'coverage': result['coverage'],
            'parameter_profile': 'effective_dynamics_v1',
            'source_levels': request.get('geometry_claim','Retained approved source geometry and explicit project transfers/effective dynamics')},
        background_preparation={'status': 'THREE_REPRESENTATIVE_DEPENDENCY_VIEWS_NUMERICALLY_CONVERGED',
            'representative_reports': {name: directory.relative_to(ROOT).as_posix()+'/'+name+'/matched_response/background_initialization.json'
                                       for name in result['regions']},
            'full_state_restore_verified': True, 'daily_run_requires_its_own_matching_checkpoint': True,
            'old_checkpoint_reused': False, 'whole_network_steady_state_claim': False,
            'stochastic_status': 'INSUFFICIENT_SHORT_TRANSIENT_WINDOWS'},
        active_input_frontend=dict(baseline['active_input_frontend'],
            scope=request.get('geometry_claim',baseline['active_input_frontend']['scope']),
            circuit_configuration_path=circuit_path))
    populations = '\n'.join('- `' + name + '`: ' + str(count) for name, count in
                             result['coverage']['support_population_counts'].items())
    if request.get('resource_profile'):
        next_config['execution_resource_profile']={'path':request['resource_profile'],'sha256':sha(local(request['resource_profile']))}
    routes = '\n'.join('- `' + route['route_id'] + '`: edges=' + str(route['edges']) +
                       ', missing-degree rows=' + str(route['missing_degree_rows']) +
                       ', empty rows=' + str(route['empty_rows']) for route in result['coverage']['routes'])
    section = (
        '\n\n## Active finite 3 mm disk (' + datetime.now(timezone.utc).strftime('%Y-%m-%d UTC') + ')\n\n'
        'The single main configuration `configurations/retina_platform_v1.json` selects `' + circuit_path + '`. '
        + request.get('geometry_claim', 'Frozen finite source geometry with explicit project approximations') + '.\n\n'
        'The official right-eye nominal58-degree human cone lattice is imported at its original positions: '
        'negate official retinal-micron coordinates and divide by1000, then crop the fixed complete support. '
        'No density resampling, warp, Packer fit or gain correction is applied. Owner RF radius3mm contains505162 cones; '
        'complete imported square support contains705259. This is a human-borrowed approximation on the existing composite macaque frame. '
        'Other frozen HC/rod/RBC/AII/parasol masters and physiological/effective parameters are retained. '
        'Cone-dependent identities, apertures, BC/midget geometry, contacts and optical ownership are rebuilt. '
        'S labels retain macaque absolute macro quotas and legacy Packer selection scores; L:M1:1 is a project rule. '
        'The Wool temporal iso-density adapter still uses Packer as its named project mapping, rather than claiming that '
        'imported human cone density is macaque density. Density differences are recorded in the public review supplement.\n\n'
        'Actual run: `' + directory.relative_to(ROOT).as_posix() + '`. Central(0,0)/0.02mm, transition(1,0)/0.08mm '
        'and near-peripheral(2.85,0)/0.1mm owners use exact dependency views of the same frozen graph. '
        'Each passed real MATLAB/ISET expected-input capture, unchanged required numerical background convergence, '
        'and two matched20ms static-image/background neural branches restored from the same complete prepared checkpoint. '
        'Absolute time, phase, RNG, history and delays are preserved. The static captured input rate is held for the20ms '
        'window; no new photon noise, gain, temporal calibration or physiological response threshold is added. '
        'Finite nonzero native release, actual BC terminal release and owner RGC voltage differences are numerical '
        'engineering evidence only. Whole3mm dynamics, physiological calibration, stochastic stationarity and selective '
        'pathway causal attribution remain unverified. Daily GUI runs must prepare their own matching background checkpoint.\n\n'
        'BC `activation` is the dynamic `terminal_release` of `HistoryBipolarTerminal`; AC/RGC consume this port. '
        '`generator_activation` remains diagnostic. Same-window raw terminal states and activation values are checked exactly. '
        'vH C denotes the author-scale calcium concentration in Supplement1 equations7-9; I_os=X^nX is the current. '
        'Legacy serialized C unit/meaning labels are retained for checkpoint compatibility and do not establish physical micromolar units. '
        'The2.82ms delay applies to observed_Vh; BC input is the pre-readout outer_native.release. '
        '`conditional_rate` is hazard lambda; Bernoulli-bin expected count rate is (1-exp(-lambda*dt))/dt; '
        '`window_rate` is realized count/window. These quantities differ.\n\n'
        'Wool Table1 uses D=Rc=sigma as radius and exp[-d^2/(2*sigma^2)] truncated at Rc. '
        'The retained project implementation uses exp[-(d/Rc)^2], finite support min(40um,Rc*sqrt(log100)), '
        'candidate selection, fractional boundary weights and normalization. Borrowing n/scale does not make that '
        'kernel source-equivalent. Source references: [vH Supplement1](https://pure.rug.nl/ws/files/2938731/2005JVisvHaterenSupp1.pdf), '
        '[Wool article](https://pmc.ncbi.nlm.nih.gov/articles/PMC5815352/).\n\n'
        'RBC=rod/18, AII=5000/mm^2, HC candidate geometry and effective_dynamics_v1 remain approved project approximations. '
        'S/H2 kinetics, full physical calcium/release calibration and physical soma displacement remain open. '
        'The canonical graph has finite support; exterior recurrent closure is not certified. Missing degrees and '
        'empty rows are reported, without nearest fill, support expansion or cropped-weight renormalization.\n\n'
        'Frozen support populations (including halo):\n\n' + populations + '\n\n'
        'Actual global routes:\n\n' + routes + '\n\n'
        'Coverage, blockers and acceptance: `' + REPORT.relative_to(ROOT).as_posix() + '/final_delivery.json`. '
        'Rollback configuration: `' + previous + '/retina_platform_v1.json`. '
        'The macaque generator and old automatic waiter were paused by the user at iteration5855; the verified immutable '
        'checkpoint, current/best/history, initialization and implementation snapshot remain in '
        '`reports/disk3mm_build_20261016/macaque_paused_20261017T140036Z/`. Resume requires the user choosing the fallback. '
        'Historical sections below remain historical evidence and earlier decisions.\n')
    if request.get('adoption_notes'):
        section+='\n'+request['adoption_notes']+'\n'
    old_text = original_design.decode('utf-8')
    title, separator, body = old_text.partition('\n')
    next_design = (title + section + '\n' + body).encode('utf-8')
    pending_main = main.with_name(main.name + '.disk3mm-writing')
    pending_design = design.with_name(design.name + '.disk3mm-writing')
    pending_main.write_text(json.dumps(next_config, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending_design.write_bytes(next_design)
    for path in (main, design):
        if sha(path) != request['adoption_identity_guards'][path.relative_to(ROOT).as_posix()]:
            raise ValueError('Parallel edit detected before switch; staged files retained')
    new_main_sha = sha(pending_main)
    os.replace(pending_main, main)
    try:
        os.replace(pending_design, design)
    except OSError:
        if sha(main) == new_main_sha:
            main.write_bytes(local(previous + '/retina_platform_v1.json').read_bytes())
        raise
    result.update(default_changed=True, rollback_configuration=previous + '/retina_platform_v1.json',
                  active_configuration_sha256=sha(main), core_design_sha256=sha(design))


def main():
    global REPORT, REQUEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, default=REQUEST)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    REQUEST = local(args.request)
    REPORT = REQUEST.parent
    request = preflight()
    if args.preflight:
        print('AUTHORIZED_DELIVERY_IDENTITIES_VERIFIED; no geometry generation or neural execution', flush=True)
        return
    if request.get('prerequisite_transition_receipt'):
        prerequisite=json.loads(local(request['prerequisite_transition_receipt']).read_bytes())
        if prerequisite.get('status')!='TRANSITION_RESPONSE_PASS_NUMERICALLY_PREPARED':
            raise ValueError('Actual original transition/background/response prerequisite has not passed')
    final = REPORT / 'final_delivery.json'
    if final.exists():
        raise FileExistsError('Previous delivery retained; inspect before any repair rerun')
    result = {'status': 'RUNNING_MINIMUM_ACCEPTANCE', 'default_changed': False,
              'physiological_validation': False, 'whole_network_steady_state_claim': False}
    save(final, result)
    try:
        configuration = json.loads(local(request.get('circuit_configuration_path','configurations/atlas_disk3mm_v1.json')).read_bytes())
        result['coverage'] = coverage(configuration)
        scientific_gaps=[]
        if result['coverage']['native_owner_zero_feedback_cones']:
            scientific_gaps.append('Owner cones lack native HC feedback under unchanged finite arbors')
        if any(value['owner_count'] for value in result['coverage']['blocked_midget_candidates'].values()):
            scientific_gaps.append('Owner midget source pools are unresolved or deficient under unchanged Wool/Packer rules')
        result['scientific_geometry_blockers']=scientific_gaps
        directory = local(request['validation_output'])
        if directory.exists():
            raise FileExistsError('Prior validation retained; no automatic repeated neural run')
        save(final, result)
        with (REPORT / 'runtime_acceptance.stdout.log').open('xb') as out, (REPORT / 'runtime_acceptance.stderr.log').open('xb') as err:
            command=[sys.executable, '-B', '-X', 'utf8', str(ROOT / 'tools/check_disk3mm.py'),
                            '--circuit-configuration', request.get('circuit_configuration_path','configurations/atlas_disk3mm_v1.json'),
                            '--output', str(directory)]
            # A known failing global scientific gate cannot be repaired by a
            # long background run on selected windows. Preserve exact regional
            # qualification; wait for the user's scientific decision instead.
            command += ['--qualification-only'] if scientific_gaps else [
                '--capture-transient','--figure','--check-pathway-response','--media',str(local(request['media_path']))]
            if request.get('resource_profile'):command+=['--resource-profile',str(local(request['resource_profile']))]
            environment=dict(os.environ)
            if request.get('resource_profile'):
                settings=json.loads(local(request['resource_profile']).read_bytes()).get('process_environment',{})
                if settings and settings!={'OPENBLAS_NUM_THREADS':'1'}:raise ValueError('Unsupported execution environment override')
                environment.update(settings)
            subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=True, env=environment)
        delivery = json.loads((directory / 'delivery.json').read_bytes())
        result.update(validation_directory=directory.relative_to(ROOT).as_posix(), regions=delivery['regions'])
        result['matched_pathway_responses'] = delivery.get('matched_pathway_responses', {})
        if scientific_gaps:
            result['neural_acceptance']='NOT_RUN: actual frozen global scientific geometry gates failed; no unapproved partner/support/target repair'
            raise RuntimeError('Scientific geometry decision required before neural acceptance/default adoption: '+'; '.join(scientific_gaps))
        expected = {'central', 'transition', 'near_peripheral'}
        if set(result['regions']) != expected or any(
            r.get('status') != 'ISET_NEURAL_MATCHED_RESPONSE_PASS_NUMERICALLY_PREPARED' or r.get('neural_windows') != 2
            or r.get('background_preparation') != 'CONVERGED_UNDER_EXISTING_NUMERICAL_CRITERIA'
            for r in result['regions'].values()):
            raise RuntimeError('Representative acceptance blocked; exact region errors and assets retained')
        responses = result['matched_pathway_responses']
        if set(responses) != expected or any(
            r.get('status') != 'MATCHED_STATIC_INPUT_PATHWAY_RESPONSE_PASS_NUMERICALLY_PREPARED'
            or r.get('background_preparation', {}).get('converged') is not True
            or set(r.get('branch_restores', {})) != {'background', 'stimulus'}
            or any(not branch.get('exact_restore') for branch in r['branch_restores'].values())
            for r in responses.values()):
            raise RuntimeError('Three prepared matched20ms pathway responses with exact full-state restores not established')
        if not (directory / 'response.png').is_file():
            raise RuntimeError('Actual response figure missing')
        if result['coverage']['native_owner_zero_feedback_cones']:
            raise RuntimeError('Actual global owner cones have no native feedback partners; scientific geometry decision needed, no fill')
        blocked = result['coverage']['blocked_midget_candidates']
        if any(value['owner_count'] for value in blocked.values()):
            raise RuntimeError('Unresolved global midget RF candidates retained; no full-domain default adoption')
        adopt(request, result, directory)
        result['status'] = 'MINIMUM_ACCEPTANCE_COMPLETE_DEFAULT_ADOPTED'
    except Exception as error:
        result.update(status='BLOCKED', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        result['updated_UTC'] = datetime.now(timezone.utc).isoformat()
        save(final, result)
        print(json.dumps({key: result.get(key) for key in ('status', 'default_changed', 'error', 'updated_UTC')},
                         ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
