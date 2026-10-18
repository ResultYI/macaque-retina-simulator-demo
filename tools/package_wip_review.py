"""Publishable WIP evidence, explicitly independent of final acceptance/defaults."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil,subprocess,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch',required=True)
    args=parser.parse_args()
    from tools.package_pro_review import sanitize,sha,write
    output=ROOT/'docs/pro_review_supplement'
    retained=ROOT/'reports/current_wip_review/previous_public_review'
    if output.exists():
        if output.resolve()!=ROOT/'docs/pro_review_supplement' or retained.exists() or not retained.resolve().is_relative_to(ROOT):
            raise ValueError('Prior evidence archive identity differs; do not overwrite')
        retained.parent.mkdir(parents=True,exist_ok=True)
        output.rename(retained)
    output.mkdir()
    inventory=[];omitted={}
    def archive(source,relative):
        source=Path(source);target=output/relative
        if not source.resolve().is_relative_to(ROOT):raise ValueError('Evidence outside project')
        write(target,{'original_local_sha256':sha(source),'copy_kind':'sanitized compact JSON derivative',
                      'content':sanitize(json.loads(source.read_bytes()))})
        inventory.append({'source':source.relative_to(ROOT).as_posix(),'public':relative,'sha256':sha(source)})
    build=ROOT/'reports/human_disk3mm_functional_hc_20261017'
    passed=ROOT/'reports/human_disk3mm_floor_transition_v7/transition'
    response=json.loads((passed/'matched_response/response_check.json').read_bytes())
    background=json.loads((passed/'matched_response/background_initialization.json').read_bytes())
    progress=json.loads((ROOT/'reports/human_disk3mm_floor_transition_v6/transition/matched_response/background_progress.json').read_bytes())
    passed_check=json.loads((passed/'check.json').read_bytes())
    passed_manifest=json.loads((passed/'manifest.json').read_bytes())
    components=[]
    for source in sorted((passed/'global_definition').glob('*_actual_values.json')):
        archive(source,'actual_parameters/'+source.name)
        metadata=json.loads(source.read_bytes());binding=metadata['array_file'];array_path=Path(binding['path'])
        if not array_path.is_absolute():array_path=ROOT/array_path
        if not array_path.resolve().is_relative_to(ROOT) or sha(array_path)!=binding['sha256']:
            raise ValueError('Actual parameter archive identity differs')
        components.append(source.stem.removesuffix('_actual_values'))
        if array_path.stat().st_size<=2*1024**2:
            target=output/'actual_parameters'/array_path.name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(array_path,target)
            inventory.append({'source':array_path.relative_to(ROOT).as_posix(),'public':target.relative_to(output).as_posix(),
                              'sha256':sha(array_path),'copy_kind':'byte-exact small executed parameter archive'})
        else:
            omitted[array_path.relative_to(ROOT).as_posix()]={'sha256':sha(array_path),'bytes':array_path.stat().st_size,
                'reason':'Large topology/parameter archive retained locally; source/profile values and metadata remain available'}
    for branch in ('background','stimulus'):
        source=passed/'matched_response'/branch/'window_00000000.npz'
        destination=output/'raw_response'/f'transition_{branch}.npz';destination.parent.mkdir(parents=True,exist_ok=True)
        with np.load(source,allow_pickle=False) as original:
            arrays={key:original[key] for key in original.files if key!='manifest_json'}
            signatures={key:{'shape':list(value.shape),'dtype':str(value.dtype),
                'raw_array_sha256':hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()} for key,value in arrays.items()}
            arrays['manifest_json']=np.array(json.dumps(sanitize(json.loads(original['manifest_json'].item())),ensure_ascii=False))
            np.savez_compressed(destination,**arrays)
        inventory.append({'source':source.relative_to(ROOT).as_posix(),'public':destination.relative_to(output).as_posix(),
                          'sha256':sha(source),'copy_kind':'recorded numeric arrays byte-exact; metadata sanitized','arrays':signatures})
    for relative in ('check.json','manifest.json','resolved_profile.json','matched_response/background_initialization.json',
                     'matched_response/response_check.json','matched_response/execution_backend_check.json'):
        archive(passed/relative,'passed_transition/'+relative)
    for attempt in range(1,8):
        source=ROOT/f'reports/human_disk3mm_midget_floor_20261017/transition_attempt_{attempt}.json'
        if source.exists():archive(source,'preparation_history/'+source.name)
    performance=ROOT/'reports/runtime_acceleration_20261017'
    for name in ('actual_native_benchmark.json','gpu_native_probe.json','gpu_dop853_probe.json',
                 'gpu_native_interval_probe.json','gpu_backend_verification_v2.json'):
        archive(performance/name,'performance/'+name)
    central=ROOT/'reports/human_disk3mm_functional_validation_v1/central/check.json'
    archive(central,'functional_HC_unaccepted/central_check_before_shape_fix.json')
    functional=ROOT/'reports/human_disk3mm_functional_validation_v1/transition/matched_response/execution_backend_check.json'
    if functional.exists():archive(functional,'functional_HC_unaccepted/transition_100us_execution_check.json')
    asset_paths=['data/atlas/isetbio_human_borrowed_cones_disk3mm_v1/manifest.json',
        'data/atlas/human_disk3mm_functional_hc_v1/circuit/manifest.json',
        'data/atlas/human_disk3mm_midget_floor_v1/circuit_input_v2/manifest.json']
    for relative in asset_paths:archive(ROOT/relative,'asset_manifests/'+relative)
    for source in (passed/'input.npz',passed/'matched_response/background_checkpoint.json'):
        omitted[source.relative_to(ROOT).as_posix()]={'sha256':sha(source),'bytes':source.stat().st_size,
            'reason':'Full input/checkpoint retained locally; raw recorded response arrays and immutable identities are public'}
    shutil.copyfile(passed.parent/'response.png',output/'transition_original_HC_response.png')
    shutil.copyfile(ROOT/'reports/cone_master_migration_20261016/gui_final/controlled_image.png',output/'controlled_image.png')
    integration=progress['integration_metrics']['integration_wall_seconds']
    wall=background['wall_seconds']
    state={'schema':'physiology_WIP_review_v1','status':'WIP_NOT_FINAL_ACCEPTANCE','UTC':datetime.now(timezone.utc).isoformat(),
        'review_branch':args.branch,'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'default_changed_for_this_review':False,'automatic_final_publisher':'STOPPED; human requested immediate WIP review',
        'human_geometry':{'raw_MAT_sha256':'54eb0a9ef90511ca72782a194a7c6f17d5404e05794f7e4872c808d138f45728',
            'rule':'Official human right-eye nominal58deg positions, exact negative raw/1000 and fixed crop; no Packer warp or coordinate resampling',
            'owner_cones':505162,'support_cones':545658,'labels':'Macaque absolute S macro-quota and L:M1:1 project split; not macaque density validation'},
        'functional_HC':{'profile':'configurations/hc_functional_divergence_w2000_v1.json','graph_sha256':'713aefa3573799f4de6c9a56e1161ffe2691255efeef32b6711e9d15ab003557',
            'source':'Wassle2000 divergence anchors at0.5-1mm and5-6mm',
            'anchors':{'H1':[3.7,7.5],'H2':[2.2,3.2]},
            'project_choices':['Representative radii0.75/5.5mm','H2 peripheral3.2 midpoint','Clipped linear radial transfer; central constant below0.5mm',
                'Nearest existing typed HC anchors; H1 excludes S','Stable floor/fractional integer allocation','Reciprocal functional associations and global unit-DC A/B normalization'],
            'existing_coordinates_and_densities_retained':True,'functional_associations':3577939,'owner_feedback_empty_rows':0,
            'maximum_owner_support_um':89.0147698955,'original_anatomical_feedback_empty_owner_cones':7770,
            'anatomical_gap_resolved':False,'independent_S_H2_kinetics':'NOT_CALIBRATED; shared H1 source values'},
        'midget_floor':{'minimum':'floor(n) for original effective n>=1','recovered_existing_candidates_each_polarity':533,
            'partners_support_weights':'unchanged','original_ceil_deficits':'retained','population_average_target_satisfied':'NOT_ESTABLISHED'},
        'cell_groups':{'photoreceptors':['L','M','S','rod'],'HC':['H1','H2'],
            'BC':['midget ON','midget OFF','diffuse ON','diffuse OFF','S-ON','RBC'],
            'AC':['AII','local feedforward','local crossover','widefield'],
            'RGC':['midget ON','midget OFF','parasol ON','parasol OFF','small bistratified S-ON'],
            'scope':'Effective groups, not all macaque cell subtypes; BC/AC/RGC and rod-inner parameters remain partly project effective'},
        'acceptance':{'original_HC_transition_v7':'PASS: required2.1s numerical background and matched20ms response; exact same full checkpoint restored to both branches',
            'functional_HC_central':'Initial empty RF owner array rank bug fixed with3 unit regressions; short reassembly now reaches AII_AC.excitation explicit positive-support failure; no artificial input added',
            'functional_HC_transition':'100us full-state CPU/CUDA comparison passed; further run stopped after central failure; no converged background/response claimed',
            'functional_HC_near_peripheral':'NOT_RUN','three_ROI_acceptance':'INCOMPLETE','physiological_validation':False},
        'response_test':{'stimulus':'One existing static controlled image/fixed gaze, expected ISET rate repeated20ms versus declared background',
            'variables':'Actual native release, BC terminal activation/state and chosen owner midget RGC voltage',
            'criteria':['All recorded outputs finite','Same complete checkpoint hash/time/RNG/history before both branches',
                'Nonzero native release, some actual BC activation and chosen owner RGC voltage differences','Actual BC activation equals terminal_release; generator activation has no consumers'],
            'not_established':['Biological amplitude/latency thresholds','Selective HC or rod causal isolation','Receptive field size tuning','Spike stationarity','All subtypes'],
            'duration_seconds':response['duration_seconds'],'origin_seconds':response['stimulus_origin_seconds'],
            'deltas':{key:response['port_differences'][key] for key in ('outer_native.release','midget_RGC_ON.voltage')}},
        'performance':{'actual_original_HC_transition_nodes':879437,'actual_scalar_states':4967023,
            'cones':530929,'HC':344287,'other_cells':4221,'background_sim_seconds':background['elapsed_seconds'],
            'passed_view_population_counts':passed_check['populations'],
            'passed_view_actual_links':passed_manifest['actual_links'],
            'functional_central_selected_nodes':894425,
            'functional_transition_short_check':json.loads(functional.read_bytes()) if functional.exists() else None,
            'background_wall_seconds':wall,'neural_integration_wall_seconds':integration,
            'outside_integration_wall_seconds':wall-integration,'outside_integration_fraction':1-integration/wall,
            'remaining_breakdown':'Not yet separately measured: original monitor/evaluate, selection/metadata, validation/conversion/checkpoint IO; CPU/GPU transfers are inside integration timing',
            'CPU_fusion':'Native-only RHS array-exact, measured3.16075x','CUDA':'Separate float64 joint DOP853 backend; native-only warm1ms benchmark26.499x, not whole-network speed',
            'same_identity_reuse':'v7 strictly reused converged v6 checkpoint; no new background integration. New functional HC graph has different weights/binding and cannot reuse the old-HC background',
            'next_action':'Short same-model profiler; prioritize high-cost exact metadata/selection/monitor/conversion work, not another multi-hour validation'},
        'dependencies':{'python_environment':'Existing retipath_v1; no upgrade','torch':'2.14.0+cu130','CUDA':'13.0','GPU':'RTX4070TiSUPER',
            'scipy':'1.17.1','numba':'0.67.0','OPENBLAS_NUM_THREADS':'1','dtype':'float64','fastmath_CPU':False,
            'ISETBio_commit':'d2924c2fe1d41c7dc5af1006d8081735b08f1b35','ISETCam_commit':'17146a9f0470cc89b58d583851f680187268506d',
            'cache_identity':'Full graph/population/weights/parameters/solver/model/background/schema/clock identity; no cross-graph fallback'},
        'review_priorities':['retina_sim/atlas_runtime.py:_owner_ids','retina_sim/atlas_primary_rod.py:attach_primary_rod',
            'retina_sim/assembly.py:positive support','retina_sim/hybrid.py:advance/record','retina_sim/background.py:_WindowMonitor.observe',
            'retina_sim/background_checkpoint.py:save/load','retina_sim/engine.py:validate_state','retina_sim/network.py:evaluate/execution_metadata',
            'retina_sim/recording.py:indices/delay payload','retina_sim/gpu_hybrid.py:CPU-GPU boundary','retina_sim/gpu_dop853.py:error norm synchronizations'],
        'next_scientific_stage_not_started':'After minimum acceptance, propose one ROI/same background with a small bounded bright/dark and spot-size set; cost separately. Quantitative comparison only where original paper conditions/units match.'}
    write(output/'review_status.json',state)
    write(output/'review_index.json',{'schema':'compact_WIP_review_index_v1','status':'WIP_NOT_FINAL_ACCEPTANCE',
        'review_branch':args.branch,'inventory':inventory,'omitted_generated_assets':omitted,
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for folder in ('retina_sim','retina_platform') for p in sorted((ROOT/folder).glob('*.py'))}})
    (output/'README.md').write_text(
        '# WIP physiology and performance review\n\n'
        'This is the current review snapshot, not final three-ROI acceptance. `review_status.json` is the concise status and performance entry point. '
        'The current default is unchanged; all automatic final publishers were stopped for this explicit WIP push.\n\n'
        '`transition_original_HC_response.png` shows the old anatomical-HC transition baseline: a20ms static image versus background, after2.1s required numerical preparation. '
        'It is not a result for the new functional HC operator. Raw numeric response arrays retain absolute times, original units and byte-exact values. '
        'Finite outputs plus nonzero native/BC/RGC differences establish limited numerical pathway operation, not physiological amplitude, latency or spike validation.\n\n'
        'The functional HC alternative keeps existing HC coordinates/densities and uses source divergence anchors with explicit project transfer/allocation/normalization choices. '
        'Its zero functional owner gaps do not repair the7770 anatomical-reference gaps. S/H2 kinetics, many inner dynamics and subtype completeness remain uncalibrated. '
        'The central empty-owner shape bug is repaired; reassembly exposed AII chemical excitation support missing for some cells. That condition remains blocked for review.\n\n'
        'Performance is an active blocker:2.1s of simulated background took about4.3h; only about21% was within the integrator. '
        'Residual checks/evaluation/metadata/conversion/checkpoint costs still need separate short profiling. Native CUDA speedups must not be presented as whole-network speedups.\n\n'
        'Large generated populations, full CSR arrays, complete checkpoints, source papers, vendor MATLAB/ISET code and machine profiles are excluded. '
        '`review_index.json` records exact omitted-file identities. The official human cone source is https://purl.stanford.edu/vm447vf0975 '
        '(right/nominal-fov-58deg.mat, SHA25654eb0a9ef90511ca72782a194a7c6f17d5404e05794f7e4872c808d138f45728). '
        'Use the pinned ISET versions in `review_status.json`; inspect `tools/build_human_disk3mm.py --help`, `tools/build_functional_hc.py --help` '
        'and `tools/build_disk3mm.py --help`. Copy the example optical profile to its ignored local name and set installed paths. '
        'Exact cold reproduction requires the retained hash-bound master assets; any regeneration creates a new identity and requires its own checks.\n',encoding='utf-8')
    print('WIP_REVIEW_PACKAGE_READY',len(inventory),'evidence entries; no default adopted',flush=True)


if __name__=='__main__':main()
