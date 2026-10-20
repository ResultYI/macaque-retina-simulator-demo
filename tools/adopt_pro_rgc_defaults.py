"""Adopt only completed per-type B5 checks into the existing single main config.

Consumes saved experiments without advancing a network or fitting parameters.
Original candidate recipes, source numerical profiles and prior results stay
unchanged. The acceptance files bind each type's own complete result bytes.
"""
from pathlib import Path
import argparse,difflib,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.source_reference_candidate import CROSS_TYPE_TARGETS,read_source_candidate
from retina_sim.contracts import ContractError

CHECKS=('mapping','bright_dark','contrast_ordering','own_mapping_recovery','single_cell_spatial_or_spectral')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encode(d):return (json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
def encode_preserving_unchanged_lines(original,document):
    old=original.splitlines(keepends=True);new=encode(document).splitlines(keepends=True)
    matcher=difflib.SequenceMatcher(None,[x.rstrip(b'\r\n') for x in old],[x.rstrip(b'\r\n') for x in new],autojunk=False)
    for a,b,n in matcher.get_matching_blocks():new[b:b+n]=old[a:a+n]
    return b''.join(new)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',type=Path,required=True);a=p.parse_args()
    reports=a.reports.resolve()
    if not reports.is_relative_to(ROOT):raise ContractError('Workspace evidence required')
    receipt=reports/'adoption.json'
    if receipt.exists():raise ContractError('Adoption already recorded; inspect existing receipt rather than repeating')
    main_path=ROOT/'configurations/retina_platform_v1.json';before=main_path.read_bytes();main_config=json.loads(before)
    rod=json.loads((reports/'rod_fixed_dark_regression/result.json').read_bytes())
    if rod.get('status')!='PASS_FIXED_DARK_REFERENCE_SHORT_REGRESSION' or rod.get('downstream_direction_regression_pass') is not True:
        raise ContractError('Complete approved rod short regression required before final adoption')
    decisions={};accepted_files={};new_entries={}
    for cid in CROSS_TYPE_TARGETS:
        result_path=reports/cid/'result.json';raw=result_path.read_bytes();result=json.loads(raw)
        recipe,path=read_source_candidate(ROOT,ROOT/'configurations'/(cid+'_literature_candidate_v1.json'))
        if (result.get('status')!='COMPLETE_OWN_TYPE_MAIN_ROUTE_ACCEPTANCE' or result.get('component')!=cid or
                result.get('recipe_sha256')!=sha(path.read_bytes()) or result.get('counts',{}).get('components')!=34 or
                set(result.get('passed_checks',{}))!=set(CHECKS)):
            raise ContractError('Completed own-type34-component evidence required: '+cid)
        passed=all(result['passed_checks'][key] is True for key in CHECKS)
        if result.get('all_checks_passed') is not passed:raise ContractError('Inconsistent per-type decision: '+cid)
        binding=dict(path=result_path.relative_to(ROOT).as_posix(),sha256=sha(raw))
        decisions[cid]=dict(adopted=passed,failed_checks=[k for k in CHECKS if not result['passed_checks'][k]],
            blocked_checks=result.get('blocked_checks',{}),own_result=binding,
            evidence_grade='D cross-type output transfer; relative engineering function only',
            gray_expected_count_rate_Hz=result['gray_expected_count_rate_Hz'])
        if not passed:continue
        relative='configurations/'+cid+'_engineering_acceptance_v1.json'
        if (ROOT/relative).exists():raise ContractError('Per-type acceptance file already exists: '+relative)
        accepted=dict(schema_version='rgc-engineering-acceptance-1.0',component_id=cid,
            physiological_calibration=False,fit_performed=False,
            evidence_grade='B original ON-parasol source output fit; D cross-type transfer to retained project membrane/bin/history',
            scope='Own type on actual34-component shared-main route with fixed-dark rod; relative center/ring/spectral, contrast and recovery only; no physiological amplitude sufficiency',
            passed_checks=result['passed_checks'],evidence_files=[binding],
            authorization='User authorized adoption of individually passed Pro B5 types;2026-10-20')
        data=encode(accepted);accepted_files[relative]=data
        new_entries[cid]=dict(recipe_path=path.relative_to(ROOT).as_posix(),recipe_sha256=sha(path.read_bytes()),
            acceptance=dict(path=relative,sha256=sha(data)))
    main_config['rgc_output_defaults']['accepted_types'].update(new_entries)
    main_config['active_run_status']='Current frozen local34-component main; individually accepted engineering RGC output defaults and fixed-dark rod reference; short integration only, fresh full background preparation required; not physiological calibration'
    history=main_config['background_preparation']
    history['historical_status_before_Pro_B5']=history['status']
    history['status']='HISTORICAL_BACKGROUND_INVALIDATED_BY_PRO_B5_PARAMETER_IDENTITIES'
    history['current_default_checkpoint_reuse_allowed']=False
    history['reason']='RGC output mapping and rod reference changed; old accepted run retained as history, not relabelled as adapted current default'
    main_config['pro_parameter_implementation']=dict(
        acceptance='Scoped engineering relative-function checks; physiological amplitude not fitted',
        decisions=decisions,rod_fixed_dark_regression=dict(path=(reports/'rod_fixed_dark_regression/result.json').relative_to(ROOT).as_posix(),
            sha256=sha((reports/'rod_fixed_dark_regression/result.json').read_bytes())),
        current_full_background_preparation_completed=False)
    # Complete all validation before writing the reviewable accepted policy.
    if main_path.read_bytes()!=before:raise ContractError('Main configuration changed concurrently; no overwrite')
    for relative,data in accepted_files.items():(ROOT/relative).write_bytes(data)
    pending=main_path.with_suffix('.pro_b5_writing');pending.write_bytes(encode_preserving_unchanged_lines(before,main_config));os.replace(pending,main_path)
    receipt.write_bytes(encode(dict(status='APPLIED_INDIVIDUALLY_PASSED_ENGINEERING_DEFAULTS',
        main_path=main_path.relative_to(ROOT).as_posix(),main_before_sha256=sha(before),main_after_sha256=sha(main_path.read_bytes()),
        accepted_RGC_types=list(main_config['rgc_output_defaults']['accepted_types']),decisions=decisions,
        physiological_calibration=False,fit_performed=False,matlab_launches=0,upload_performed=False)))
    print(receipt.read_text(encoding='utf-8'),flush=True)


if __name__=='__main__':main()
