"""Create the compact public review evidence after the actual disk delivery.

Numeric response arrays are retained exactly. JSON copies are explicitly
sanitized/compacted derivatives with original-file hashes, never original logs.
No source papers, complete checkpoints or generated population arrays are copied.
"""
from pathlib import Path
import argparse, hashlib, json, re, shutil, subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT/'reports/disk3mm_build_20261016'
HISTORICAL = ROOT/'reports/platform_runs/20261016T071417135148Z_0348ea84'
LOCAL_PROFILE = json.loads((ROOT/'configurations/iset_lcd_apple_mw_v1.json').read_bytes())
LOCAL_PATHS = {LOCAL_PROFILE[key]: '<'+key.upper()+'>' for key in ('matlab_executable','isetcam_root','isetbio_root')}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sanitize(value):
    if isinstance(value, str):
        value = re.sub(re.escape(str(ROOT)), '<PROJECT_ROOT>', value, flags=re.I)
        value = re.sub(re.escape(ROOT.as_posix()), '<PROJECT_ROOT>', value, flags=re.I)
        for actual, public in LOCAL_PATHS.items():
            value = re.sub(re.escape(actual), public, value, flags=re.I)
            value = re.sub(re.escape(actual.replace('\\', '/')), public, value, flags=re.I)
        value = re.sub(r'(?i)C:[\\/]Users[\\/][^\\/\s"\']+', '<USER_HOME>', value)
        value = re.sub(r'Sentinel_[0-9a-f]{32}', '<LOCAL_APPROVAL_RECORD>', value)
        return value
    if isinstance(value, dict):
        return {sanitize(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 256:
            raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
            result = {'omitted_list_length': len(value), 'canonical_JSON_sha256': hashlib.sha256(raw).hexdigest()}
            if all(isinstance(x, (int, float, bool)) for x in value):
                if all(x == value[0] for x in value):
                    result['exact_uniform_value'] = value[0]
                else:
                    result.update(minimum=min(value), maximum=max(value))
            return result
        return [sanitize(x) for x in value]
    return value


def cases():
    # Pro recovered these categories, not a complete historical A1-A7 contract.
    # Criteria below are actual current repository assertions, not invented
    # physiological thresholds or newly approved Pro stimulus protocols.
    return [
        ('A1', 'assets_and_inputs',
         'tests/test_disk3mm.py::test_compact_global_input_checks_rows_units_and_actual_background',
         'Reject wrong global rows, units, background or mutated input bytes; retain actual per-cone background.'),
        ('A2', 'spatial_coverage',
         'tests/test_disk3mm.py::test_disk_source_integrals_and_field_identity',
         'Original approved source integrals: AII rounded disk target158398 and parasolON target30726; source field identities remain distinct.'),
        ('A3', 'stable_queries',
         'tests/test_disk3mm.py::test_frozen_typed_arrays_share_ids_across_overlapping_disk_views',
         'Overlapping views return identical frozen IDs/coordinates; changed table inputs reject reuse.'),
        ('A4', 'mechanism_ownership',
         'tests/test_circuit_geometry.py::test_production_build_uses_frozen_pools_anchors_and_gaps',
         'Fixture retains frozen pools/contact anchors/gaps and owner closure; recording leaves canonical geometry unchanged. Actual3mm same-window observations require activation == recorded terminal_release with identical IDs and no generator_activation consumers; actual producer/port/consumer edges are archived.'),
        ('A5', 'dynamic_state',
         'tests/test_background_preparation.py::test_checkpoint_preserves_origin_delay_rng_and_continuation',
         'Restored continuation has array-exact states, RNG/delays/history/tick/input context and spike events; wrong context rejects. Original numerical convergence criteria remain in bounded_uniform_background_v1.json.'),
        ('A6', 'pathway_function',
         'tools/check_disk3mm.py --circuit-configuration configurations/atlas_human_borrowed_disk3mm_v1.json --check-pathway-response --figure --media docs/pro_review_supplement/controlled_image.png --output reports/new_review_run',
         'Existing3ROIs; required existing numerical background convergence; same complete state restored to both20ms branches; finite outputs and nonzero native.release, actual BC.activation terminal_release and owner RGC.voltage differences. Numerical engineering criterion, no biological response/latency threshold.'),
        ('A7', 'traceability',
         'tests/test_circuit_geometry.py::test_actual_weight_or_gap_change_is_rejected',
         'Mutation of actual weights or gap conductance invalidates frozen geometry identity; original source/frozen asset hashes are separately retained.'),
    ]


def main():
    global BUILD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=BUILD)
    parser.add_argument('--allow-blocked',action='store_true',help='Archive an actual scientific geometry blocker without claiming a passed runtime or default switch')
    parser.add_argument('--output', type=Path, default=Path('docs/pro_review_supplement'))
    args = parser.parse_args()
    BUILD = (ROOT/args.build).resolve()
    if not BUILD.is_relative_to(ROOT):
        raise ValueError('Build report must stay inside project')
    output = (ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Use a new output directory inside this project; prior evidence is retained')
    final = json.loads((BUILD/'final_delivery.json').read_bytes())
    blocked=final.get('status')=='BLOCKED' and bool(final.get('scientific_geometry_blockers'))
    if final.get('status') != 'MINIMUM_ACCEPTANCE_COMPLETE_DEFAULT_ADOPTED' and not (args.allow_blocked and blocked):
        raise ValueError('Actual authorized delivery has not passed; no completed public review package created')
    run = ROOT/final['validation_directory']
    request = json.loads((BUILD/'authorized_delivery_request.json').read_bytes())
    circuit_path = request.get('circuit_configuration_path','configurations/atlas_disk3mm_v1.json')
    configuration = json.loads((ROOT/circuit_path).read_bytes())
    output.mkdir(parents=True)
    inventory = []; missing = {}; visited = set(); pending = []

    def bindings(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                pending.append((value['path'], value['sha256']))
            for v in value.values():
                bindings(v)
        elif isinstance(value, list):
            for v in value:
                bindings(v)

    def archive_json(source, destination):
        data = json.loads(source.read_bytes())
        bindings(data)
        write(destination, {'original_local_sha256': sha(source),
            'copy_kind': 'sanitized_JSON_derivative; long lists summarized, no numerical values recalibrated',
            'content': sanitize(data)})
        inventory.append({'source': source.relative_to(ROOT).as_posix(),
                          'public': destination.relative_to(output).as_posix(), 'original_sha256': sha(source)})

    def archive_actual_parameters(region):
        """Expose executed values without publishing full native topology arrays."""
        source_dir=run/region/'global_definition'
        for metadata_file in sorted(source_dir.glob('*_actual_values.json')):
            metadata=json.loads(metadata_file.read_bytes())
            array_binding=metadata['array_file']
            arrays_file=Path(array_binding['path'])
            if not arrays_file.is_absolute():arrays_file=ROOT/arrays_file
            arrays_file=arrays_file.resolve()
            if not arrays_file.is_relative_to(ROOT) or sha(arrays_file)!=array_binding['sha256']:
                raise ValueError('Actual executed parameter arrays changed: '+str(arrays_file))
            destination=output/'actual_parameters'/region
            archive_json(metadata_file,destination/metadata_file.name)
            summaries={}
            with np.load(arrays_file,allow_pickle=False) as values:
                for key in values.files:
                    value=values[key]
                    item={'shape':list(value.shape),'dtype':str(value.dtype),
                          'raw_array_sha256':hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
                    if value.size and value.dtype.kind in 'biuf':
                        item.update(minimum=float(np.min(value)),maximum=float(np.max(value)))
                        if np.all(value==value.flat[0]):item['exact_uniform_value']=value.flat[0].item()
                        elif value.size<=256:item['exact_values']=value.tolist()
                        else:
                            distinct,counts=np.unique(value,return_counts=True)
                            if len(distinct)<=32:
                                item['exact_distinct_values_and_counts']=[{'value':v.item(),'count':int(n)} for v,n in zip(distinct,counts)]
                    summaries[key]=item
            write(destination/(metadata_file.stem+'_array_summary.json'),{
                'schema':'executed_parameter_array_summary_v1','original_file_sha256':sha(arrays_file),
                'rule':'Uniform values and small arrays are exact; other distributions are explicitly summarized. Metadata maps array keys to parameters or native matrix options.',
                'arrays':summaries})
            relative=arrays_file.relative_to(ROOT).as_posix()
            if arrays_file.stat().st_size<=2*1024**2:
                target=destination/arrays_file.name;shutil.copyfile(arrays_file,target)
                inventory.append({'source':relative,'public':target.relative_to(output).as_posix(),
                                  'original_sha256':sha(arrays_file),'copy_kind':'byte-exact executed parameter NPZ'})
            else:
                missing[relative]={'sha256':sha(arrays_file),'bytes':arrays_file.stat().st_size,
                    'reason':'Full native parameter/topology archive omitted; metadata, raw-array hashes and exact uniform/small values archived'}

    for name in ('final_delivery.json', 'status.json', 'geometry_pipeline.json', 'cone_master.json',
                 'population_build.json', 'primary_graph.json', 'contact_geometry.json','feedback_gap_diagnosis.json',
                 'outer_tables.json','outer_connections.json','bc_tables.json','midget_tables.json','inner_tables.json',
                 'aii_bc_connections.json','input_plan.json','global_graph.json'):
        p = BUILD/name
        if p.exists():
            archive_json(p, output/'production'/name)
    config_names=['retina_platform_v1.json',Path(circuit_path).name,request.get('atlas_profile_id','retina_atlas_disk3mm_v1')+'.json']
    if request.get('resource_profile'):config_names.append(Path(request['resource_profile']).name)
    if configuration.get('native_operator_representation'):config_names.append(Path(configuration['native_operator_representation']['profile']['path']).name)
    for name in config_names:
        archive_json(ROOT/'configurations'/name, output/'configuration'/name)
    performance=ROOT/'reports/runtime_acceleration_20261017/actual_native_benchmark.json'
    if performance.exists():archive_json(performance,output/'production/actual_native_execution_benchmark.json')
    for name in ('gpu_native_probe.json','gpu_dop853_probe.json','gpu_native_interval_probe.json','gpu_backend_verification.json','gpu_backend_verification_v2.json'):
        candidate=ROOT/'reports/runtime_acceleration_20261017'/name
        if candidate.exists():archive_json(candidate,output/'production'/name)
    for attempt in range(1,8):
        source=ROOT/('reports/human_disk3mm_midget_floor_20261017/transition_attempt_'+str(attempt)+'.json')
        if source.exists():archive_json(source,output/'preparation_history'/source.name)
    comparison=ROOT/'reports/disk3mm_build_20261016/density_comparison_20261017/comparison.json'
    if comparison.exists():archive_json(comparison,output/'production/human_Packer_density_comparison.json')
    for name in ('manifest.json', 'resolved_profile.json', 'background_initialization.json', 'summary.json'):
        archive_json(HISTORICAL/name, output/'historical_local'/name)
    for region in ('central', 'transition', 'near_peripheral'):
        for name in ('check.json', 'manifest.json', 'resolved_profile.json',
                     'matched_response/background_initialization.json', 'matched_response/response_check.json'):
            source=run/region/name
            if source.exists():archive_json(source, output/'regions'/region/name)
            elif not blocked:raise FileNotFoundError(source)
        archive_actual_parameters(region)
        for relative,reason in (
            ('input.npz','Actual complete physical-input packet omitted; optical reference and capture identities are in the run manifest'),
            ('matched_response/background_checkpoint.json','Complete prepared neural state/RNG/history/delay checkpoint retained locally; public matched-response arrays are exact')):
            source=run/region/relative
            if source.exists():missing[source.relative_to(ROOT).as_posix()]={
                'sha256':sha(source),'bytes':source.stat().st_size,'reason':reason}
        for branch in ('background', 'stimulus'):
            source = run/region/'matched_response'/branch/'window_00000000.npz'
            if blocked and not source.exists():continue
            destination = output/'raw_response'/region/(branch+'.npz');destination.parent.mkdir(parents=True, exist_ok=True)
            with np.load(source, allow_pickle=False) as original:
                arrays = {k: original[k] for k in original.files if k != 'manifest_json'}
                identities = {k: {'shape': list(a.shape), 'dtype': str(a.dtype),
                    'raw_array_sha256': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()} for k, a in arrays.items()}
                arrays['manifest_json'] = np.array(json.dumps(sanitize(json.loads(original['manifest_json'].item())), ensure_ascii=False))
                np.savez_compressed(destination, **arrays)
            inventory.append({'source': source.relative_to(ROOT).as_posix(), 'public': destination.relative_to(output).as_posix(),
                'original_sha256': sha(source), 'copy_kind': 'all recorded numeric arrays byte-exact; JSON metadata sanitized/compacted', 'arrays': identities})
    bindings(configuration)
    while pending:
        relative, expected = pending.pop()
        if (relative, expected) in visited:
            continue
        visited.add((relative, expected));p = (ROOT/relative).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file():
            continue
        if relative.startswith('data/atlas/'):
            if sha(p) != expected:
                raise ValueError('Actual referenced asset changed: '+relative)
            if p.suffix == '.json' and p.stat().st_size <= 2*1024**2:
                archive_json(p, output/'asset_manifests'/relative)
            else:
                missing[relative] = {'sha256': expected, 'bytes': p.stat().st_size, 'reason': 'Generated asset omitted from public source/evidence package'}
    figure_source=BUILD/'geometry_blockers.png' if blocked else run/'response.png'
    figure_name='geometry_blockers.png' if blocked else 'response.png'
    shutil.copyfile(figure_source, output/figure_name)
    request = json.loads((BUILD/'authorized_delivery_request.json').read_bytes())
    shutil.copyfile(ROOT/request['media_path'], output/'controlled_image.png')
    write(output/'review_index.json', {'schema': 'compact_public_retina_review_v1',
        'source_revision': 'Uncommitted local blocked candidate; identify execution code by the archived file hashes' if blocked else 'The Git commit containing this package; obtain with git rev-parse HEAD',
        'publication_status':final.get('publication_status','Inspect the Git commit containing this package'),
        'repository_HEAD_before_evidence_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'delivery_status': final['status'], 'default_configuration': 'configurations/retina_platform_v1.json',
        'geometry_claim': request.get('geometry_claim'),
        'focused_regressions': request.get('focused_regressions',{}),
        'source_and_acceptance_file_sha256': request['frozen_identity_guards'],
        'execution_code_sha256': {p.relative_to(ROOT).as_posix():sha(p) for folder in ('retina_sim','retina_platform') for p in sorted((ROOT/folder).glob('*.py'))},
        'evidence_inventory': inventory, 'omitted_generated_assets': missing,
        'A1_A7_original_contract': 'Only category names recovered from Pro text; exact historical cases/thresholds were not supplied. Below are current repository executable assertions, not a reconstructed approval.',
        'A1_A7': [{'id': a, 'category': b, 'command': ('python -B -X utf8 -m pytest -q -p no:cacheprovider '+c) if c.startswith('tests/') else 'python -B -X utf8 '+c.replace('configurations/atlas_human_borrowed_disk3mm_v1.json',circuit_path),
                   'current_repository_criterion': d,
                   'delivery_execution': ('NOT_RUN: actual global geometry gates failed before optical/neural execution' if blocked else 'Actual3mm matched responses archived') if a == 'A6' else
                       ('NOT_RUN_ON_NEW_HUMAN_GRAPH; no BC terminal alias or dynamic-port audit claimed' if blocked else 'Actual3mm BC terminal alias equality and producer-port-consumer audit archived; fixture assertion not rerun') if a == 'A4' else
                       'Passed focused restoration regression in this delivery' if a == 'A5' else
                       'Passed focused disk3mm fixture suite in this delivery; not full3mm dynamic validation'
                       if a in ('A1','A2','A3') and request.get('focused_regressions',{}).get('disk3mm_structural_input_and_identity_suite',{}).get('passed',0)>=15
                       else 'NOT_RERUN_AS_PART_OF_FINAL3mm_NEURAL_ACCEPTANCE; inspect existing fixture tests/scope'} for a,b,c,d in cases()],
        'new_human_route_runtime': 'NOT_RUN: scientific geometry gates failed' if blocked else 'Three prepared matched20ms representative response checks passed',
        'not_run': (['New human route MATLAB/ISET capture','New human route background initialization','New human route neural response'] if blocked else [])+['Independent animal physiological validation', 'Whole3mm dynamic execution',
            'Stochastic spike stationarity', 'Selective cone/rod or inhibitory causal isolation',
            'Pro proposed2s/10percent/2Hz and full combination protocols'],
        'raw_response_rule': 'NOT_CREATED: new human graph failed scientific gates before neural execution' if blocked else 'No averaging, gain or time rebasing of archived numeric arrays. Times are absolute; onset comes from response_check.json.',
        'figure_kind': 'Frozen geometry blockers; not neural responses' if blocked else 'Actual matched neural response differences',
        'figure_sha256': sha(output/figure_name)})
    (output/'README.md').write_text(
        '# Pro review evidence\n\n'
        + ('CURRENT RESULT: BLOCKED. The human geometry candidate was frozen and region-qualified; no new human optical capture, background preparation, neural response or default adoption is claimed. The figure is a geometry gap map. Historical results remain separate.\n\n' if blocked else '') +
        'Review the Git commit containing this package. GUI entry: `run_retina_gui.cmd`; single main configuration: '
        '`configurations/retina_platform_v1.json`. `review_index.json` lists original file hashes, executable current '
        'A1-A7 assertions, actual execution and unrun controls. Original historical A1-A7 thresholds were not supplied.\n\n'
        '`production/` and `regions/` document the current frozen3mm human-borrowed cone route and three real '
        'representative dependency views; inspect actual status before interpreting any runtime claim. `historical_local/` is earlier local-run evidence. JSON files are sanitized '
        'and compacted derivatives with original hashes. '
        + ('No new raw_response directory exists because the human route was not executed. ' if blocked else
        '`raw_response/` contains every originally recorded numeric array without averaging, calibration or time rebasing. Times are absolute; stimulus onset is in response_check.json. ') +
        'Numerical background convergence and20ms branch differences do not establish biological validation.\n\n'
        '`actual_parameters/` contains executed component metadata and exact small parameter archives. Array summaries expose exact uniform values, small arrays and small discrete distributions; large nonuniform arrays are identified by hashes and listed as omitted. Native connection matrices are not reinterpreted as calibrated physiological parameters.\n\n'
        'Official human Curcio/Watson cone positions are imported with the official sign/unit transform and fixed crop, '
        'without warp or Packer resampling. Other frozen populations and effective dynamics remain. Legacy Packer '
        'S-selection scores/Wool mapping and macaque S quotas remain explicit project rules; human/Packer density '
        'differences are archived. The macaque5855 checkpoint and implementation snapshot remain paused locally.\n\n'
        'BC activation is HistoryBipolarTerminal dynamic terminal_release and feeds AC/RGC. generator_activation is '
        'diagnostic. conditional_rate is hazard; bin expectation is (1-exp(-lambda*dt))/dt; window_rate is realized '
        'count/window. vH C is author-scale calcium concentration; legacy serialized labels remain.2.82ms is only '
        'the observed_Vh readout delay. Wool source radius/kernel differs from the retained project kernel and finite '
        'support. Full details and original source links are in `docs/core_design.md`.\n\n'
        'Large generated populations, CSR blocks, complete checkpoints, MATLAB/ISET installations and source papers '
        'are omitted. `omitted_generated_assets` records referenced paths, bytes and hashes. Exact local assets are '
        'needed to reproduce this frozen identity; rebuilding creates a new identity and needs its own acceptance. '
        'The official source deposit is https://purl.stanford.edu/vm447vf0975, right/nominal-fov-58deg.mat, '
        'SHA25654eb0a9ef90511ca72782a194a7c6f17d5404e05794f7e4872c808d138f45728. '
        'Copy `configurations/iset_lcd_apple_mw_v1.example.json` to its local non-example name and set installed paths. '
        'Pinned ISETCam:17146a9f0470cc89b58d583851f680187268506d; '
        'ISETBio:d2924c2fe1d41c7dc5af1006d8081735b08f1b35. Changing machine paths changes input-cache identity. '
        '`tools/build_human_disk3mm.py --help` describes the cone import/structural adapter; other frozen masters '
        'need the retained local assets or a separately identified approved rebuild.\n',encoding='utf-8')
    print('PUBLIC_REVIEW_PACKAGE_CREATED', output.relative_to(ROOT), 'status='+final['status'])


if __name__ == '__main__':
    main()
