"""One requested position under the existing frozen near-center S0 contract; dry-run by default."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.spatial_group_graph import digest
from tools.prepare_cross_position_coverage_v1 import choose_geometry

NEAR=ROOT/'reports/layerwise_physiology/cross_position_functional_coverage_v1_20261029/restored_resume_v1'
SCENE=NEAR/'background_v1/OPTICAL_CONFIG.json'
TEMPLATE=NEAR/'local_static/selection_identity.json'
PARENT=NEAR/'parent_construction_seed'
ANCHOR_BINDING=ROOT/'reports/isetbio_native_aperture_convergence_20261029_v1/NEXT_STEP_PREPARATION.json'
STEPS=('construct','optical_setup','capture','initialize')
HISTORY=5536.059938100068


def read(path: Path):
    return json.loads(path.read_bytes())


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--center-deg',type=float,nargs=2,required=True,metavar=('X','Y'))
    result.add_argument('--output',type=Path,required=True)
    result.add_argument('--execute-step',choices=STEPS)
    result.add_argument('--plan-output',type=Path)
    result.add_argument('--historical-seconds',type=float,default=HISTORY)
    return result


def preflight(center, output: Path):
    out=output.resolve();reports=ROOT/'reports'
    if out==reports or not out.is_relative_to(reports):
        raise ContractError('Separate experiment output below reports required')
    if digest(SCENE)!=read(ANCHOR_BINDING)['existing_near_scene_anchor']['sha256']:
        raise ContractError('Existing frozen near-center S0 identity changed')
    template=read(TEMPLATE);anchor=read(SCENE);parent=read(PARENT/'manifest.json')
    cfg=template['configuration'];artifact=ROOT/cfg['source_artifact']['path']
    if digest(artifact)!=cfg['source_artifact']['sha256']:
        raise ContractError('Frozen parent graph manifest differs')
    if digest(PARENT/'manifest.json')!=cfg['input_identity']['sha256']:
        raise ContractError('Frozen near-center parent construction differs')
    seed=parent['identity']['seed']
    if (parent['identity']['artifact_sha256']!=template['parent_artifact']
        or seed.get('classification')!='SOURCE_CONSTRUCTION_IVP_NOT_PREPARED_BACKGROUND'
        or seed.get('neural_model_advanced') is not False or seed.get('time_seconds')!=0.):
        raise ContractError('Only the existing unadvanced parent construction t=0 is reusable')
    if any(not (PARENT/name).is_file() for name in parent['files']):
        raise ContractError('A declared parent construction array is absent')
    reference=Path(anchor['reference_capture_path'])
    if digest(reference)!=anchor['source_reference_sha256']:
        raise ContractError('Frozen S0 reference capture differs')
    raw=loadmat(reference,variable_names=['manifest_json','q_ref_LMS'])
    physical=read(SCENE);meta=json.loads(raw['manifest_json'].item())
    for key in ('fixed_radiance_gain','reference_rgb','reference_energy_spectrum_hex'):
        if canonical_hash(physical[key])!=canonical_hash(meta[key]):
            raise ContractError('S0 differs from its source reference: '+key)
    if not np.array_equal(raw['q_ref_LMS'].reshape(-1),anchor['q_ref_LMS']):
        raise ContractError('Frozen S0 fixed LMS conversion differs')
    aperture=ROOT/'configurations/isetbio_native_cone_aperture_v1.json'
    requires_native_background=canonical_hash(read(aperture))!=anchor['aperture_profile']
    geometry=choose_geometry(artifact.parent,center)
    prepared=out/'prepared';background=out/'background'
    paths={'scene_anchor':SCENE,'scene_anchor_binding':ANCHOR_BINDING,'near_selection_template':TEMPLATE,'parent_construction':PARENT/'manifest.json',
        'parent_graph':artifact,'aperture_profile':aperture,
        'initializer':ROOT/'retina_sim/h1_h2_background.py',
        'aperture_initializer':ROOT/'retina_sim/cone_aperture_hc_bridge.py',
        'boundary_rule':ROOT/'retina_sim/local_patch_boundary.py',
        'HC_parameters':ROOT/'configurations/research/HC_homotypic_lateral_only_v2.json',
        'RGC_candidate':ROOT/'configurations/research/RGC_FUNCTIONAL_V1_candidate.json',
        'position_builder':ROOT/'tools/prepare_cross_position_coverage_v1.py',
        'background_builder':ROOT/'tools/prepare_cross_position_background_v1.py',
        'optical_executor':ROOT/'tools/run_cross_position_optical_v1.py',
        'capture':ROOT/'tools/capture_cross_position_background_v1.m',
        'native_aperture':ROOT/'tools/cone_native_aperture.m',
        'entry':Path(__file__)}
    bindings={key:dict(path=str(path.relative_to(ROOT)),sha256=digest(path)) for key,path in paths.items()}
    input_reference=dict(parent['payload']['dictionary'])['input_reference']
    rod=dict(input_reference['dictionary'])['rod_background_rstar_s']
    plan=dict(schema='requested-position-frozen-near-S0-preparation-1',
        requires_native_aperture_background=requires_native_background,
        center_deg=list(map(float,center)),actual_eccentricity_deg=geometry['actual_eccentricity_deg'],
        target_type=geometry['selected_type'],target_parent_group_row=geometry['selected_parent_group_row'],
        core_RGC_centroid_counts={name:value['core_centroid_groups'] for name,value in geometry['geometric_presence'].items() if 'RGC' in name or name=='small_bistratified_S_ON'},
        output=str(out),prepared=str(prepared),background=str(background),bindings=bindings,
        parameter_provenance=dict(
            physical_S0=dict(source='scene_anchor, exact values checked against original reference capture',
                profile=anchor['profile'],fixed_radiance_gain=anchor['fixed_radiance_gain'],reference_rgb=anchor['reference_rgb'],
                reference_energy_spectrum_hex=anchor['reference_energy_spectrum_hex'],q_ref_LMS=anchor['q_ref_LMS']),
            coordinates=dict(source='parent graph retained actual members',mm_per_degree=read(artifact)['configuration']['mm_per_degree']),
            patch=dict(source='near_selection_template',geometry=cfg['patch_geometry'],native_radius_deg=template['native_radius']),
            boundary=dict(source='parent construction t=0, existing configure_boundary_runtime and H1H2 seed methods',
                policy=cfg['boundary_policy'],new_rule_created=False,old_far_equilibrium_annulus_reused=False,
                whole_retina_S0_equilibrium_claimed=False),
            rod=dict(source='parent construction input_reference; existing independent engineering pathway',background_rstar_s=rod),
            initialization=dict(source='existing initialize function; independent requested-position solve',historical_local_seed_copied=False)),
        preflight_scope='Declared source/parameter bindings and array existence; full graph/array SHA guards still run in existing construction/seed loaders',
        default_action='dry_run_only',neural_dynamics_step_available=False,paired_background_experiment_completed=False,
        FIT_READY=False,background_qualified=False,production_adopted=False)
    plan['experiment_id']=canonical_hash(plan)
    if out.exists():
        if not (out/'RUN_PLAN.json').is_file() or read(out/'RUN_PLAN.json')['experiment_id']!=plan['experiment_id']:
            raise ContractError('Existing output is not this experiment; historical results cannot be overwritten')
    return plan


def commands(plan, historical=HISTORY):
    base=[sys.executable];prepared=plan['prepared'];background=plan['background']
    bounded=['--max-rss-gib','24','--historical-seconds',str(historical)]
    return dict(
        construct=base+[str(ROOT/'tools/prepare_cross_position_coverage_v1.py'),'--output',prepared,
            '--center-deg',*[str(x) for x in plan['center_deg']],'--parent-construction',str(PARENT),
            '--selection-template',str(TEMPLATE),*bounded],
        optical_setup=base+[str(ROOT/'tools/prepare_cross_position_background_v1.py'),'--prepared',prepared,
            '--output',background,'--mode','optical_setup','--scene-anchor',str(SCENE),*bounded],
        capture=base+[str(ROOT/'tools/run_cross_position_optical_v1.py'),background,*bounded],
        initialize=base+[str(ROOT/'tools/prepare_cross_position_background_v1.py'),'--prepared',prepared,
            '--output',background,'--mode','initialize',*bounded])


def execute(plan, step, historical_base=HISTORY):
    out=Path(plan['output']);prepared=Path(plan['prepared']);background=Path(plan['background'])
    prerequisite={'construct':None,'optical_setup':prepared/'local_static/selection_identity.json',
        'capture':background/'OPTICAL_CONFIG.json','initialize':background/'capture.mat'}[step]
    expected={'construct':prepared/'local_static/selection_identity.json','optical_setup':background/'OPTICAL_CONFIG.json',
        'capture':background/'capture.mat','initialize':background/'seed/manifest.json'}[step]
    if prerequisite and not prerequisite.is_file():
        raise ContractError('Complete preceding explicit step first: '+str(prerequisite))
    if plan.get('requires_native_aperture_background'):
        raise ContractError('Generate a native-aperture background and scene anchor before executing a new position')
    forbidden={'construct':[prepared],'optical_setup':[background],
        'capture':[expected,background/'optical_capture_RESOURCE.json'],
        'initialize':[expected,background/'FIXED_LMS_BRIDGE.json',background/'initialize_RESOURCE.json']}[step]
    if any(path.exists() for path in forbidden) or (out/f'{step}_EXECUTION.json').exists():
        raise ContractError('Step output already exists; preserve it and inspect before a separate new attempt')
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'RUN_PLAN.json').exists():
        with (out/'RUN_PLAN.json').open('x',encoding='utf-8') as stream:json.dump(plan,stream,indent=2)
    history=out/'HISTORICAL_BASE.json'
    if history.exists():
        historical_base=read(history)['seconds']
    else:
        with history.open('x',encoding='utf-8') as stream:json.dump(dict(seconds=historical_base),stream)
    elapsed=sum(read(path)['wall_seconds'] for path in out.glob('*_EXECUTION.json'))
    historical=historical_base+elapsed;start=time.perf_counter();status='FAILED';code=None
    command=commands(plan,historical)[step]
    with (out/f'{step}_stdout.log').open('xb') as stdout,(out/f'{step}_stderr.log').open('xb') as stderr:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            code=process.wait(timeout=900)
            if code!=0 or not expected.is_file():
                raise ContractError('Explicit preparation step failed; see preserved logs')
            status='PASS'
        except BaseException:
            if process.poll() is None:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=15)
            raise
        finally:
            receipt=dict(status=status,step=step,returncode=code,command=command,
                experiment_id=plan['experiment_id'],wall_seconds=time.perf_counter()-start,
                historical_seconds=historical,recorded_utc=datetime.now(timezone.utc).isoformat(),
                no_downstream_dynamics=True,FIT_READY=False,background_qualified=False,production_adopted=False)
            with (out/f'{step}_EXECUTION.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2)
    return receipt


def main(argv=None):
    args=parser().parse_args(argv);plan=preflight(args.center_deg,args.output)
    if args.plan_output:
        target=args.plan_output.resolve()
        if not target.is_relative_to(ROOT/'reports'):
            raise ContractError('Plan artifact must remain under reports')
        with target.open('x',encoding='utf-8') as stream:json.dump(plan,stream,indent=2)
    if args.execute_step:
        print(json.dumps(execute(plan,args.execute_step,args.historical_seconds)))
    else:
        print(json.dumps(dict(status='DRY_RUN_PASS',plan=plan,explicit_step_commands=commands(plan)),indent=2))
    return plan


if __name__=='__main__':
    main()
