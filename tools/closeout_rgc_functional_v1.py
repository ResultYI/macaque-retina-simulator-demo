"""Close this local conditional RGC stage only after actual coupled checks pass."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    out = p.parse_args().output.resolve()
    final_dir = out / 'source_coupled_v1/analysis'
    result = json.loads((final_dir / 'RESULT.json').read_bytes())
    ledger = json.loads((out / 'RESOURCE_LEDGER.json').read_bytes())
    types = json.loads((final_dir / 'TYPE_COMPLETION.json').read_bytes())
    if (not result['all_types_functional_completed'] or not result['numerical_passed']
            or result['source_900ms_endpoint_max_error_over_budget'] > 1
            or ledger['status'] != 'COMPLETED' or len(ledger['source_coupled_runs']) != 2
            or any(r['status'] != 'COMPLETED' for r in ledger['source_coupled_runs'])):
        raise ValueError('Required actual source-coupled evidence has not passed')
    if not all((final_dir / f).is_file() for f in
               ('functional_curves.png', 'native_group_spikes.png', 'index.html')):
        raise ValueError('Required actual curve deliverables missing')
    descriptor_path = ROOT / 'configurations/research/RGC_FUNCTIONAL_V1.json'
    if descriptor_path.exists() or (out / 'FINAL_RESULT.json').exists():
        raise FileExistsError('Preserve an existing final result or stage descriptor')
    sources = ['retina_sim/rgc_functional_v1.py', 'retina_sim/rgc_functional_replay.py',
        'tools/run_rgc_functional_v1.py', 'tools/rgc_functional_worker.py',
        'tools/run_rgc_s_type_probe.py', 'tools/run_rgc_coupled_qualification.py',
        'tools/rgc_functional_coupled.py', 'tools/analyze_rgc_functional_v1.py',
        'tools/plot_rgc_functional_v1.py', 'tools/document_rgc_functional_v1.py',
        'tools/closeout_rgc_functional_v1.py',
        'tests/test_rgc_functional_v1.py',
        'tests/test_rgc_functional_grid.py',
        'configurations/research/RGC_FUNCTIONAL_V1_candidate.json']
    dump(out / 'EXECUTION_SOURCES.json', {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources})
    current = dict(result, finalized_utc=datetime.now(timezone.utc).isoformat(),
        AII='All original AII-to-BC chemical/electrical mechanisms evolve in the main coupled assay; driven controls inherit their BC-cache contribution; no direct RGC edge added',
        final_report=str((final_dir / 'index.html').relative_to(ROOT)),
        final_curve_data=str((final_dir / 'ALL_TYPE_MEAN_TIME_COURSES.csv').relative_to(ROOT)),
        final_type_table=str((final_dir / 'TYPE_COMPLETION.csv').relative_to(ROOT)),
        source_coupled_standard='source_coupled_v1/standard/RESPONSE.npz',
        source_coupled_tight='source_coupled_v1/tight/RESPONSE.npz',
        model_wall_seconds_this_session=ledger['model_wall_seconds_this_session'],
        historical_plus_this_session_seconds=ledger['historical_plus_this_session_seconds'],
        new_output_bytes_before_closeout=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        earlier_cache_interpolation_failure_preserved=True,
        source_asset_hash_gates_bypassed=False, scientific_parameters_changed=False,
        subtype_output_mapping_evidence='Inherited ON-parasol source-limited mapping plus four existing D cross-type transfers',
        historical_ledger_unchanged=ledger['historical_ledger_unchanged'],
        fitting_started=False, automated_cross_eccentricity_started=False)
    dump(out / 'FINAL_RESULT.json', current)
    dump(out / 'REVIEW_WORK.json', dict(
        verdict='PASS_WITHIN_LOCAL_CONDITIONAL_FUNCTIONAL_SCOPE',
        goal='All five existing types,35 groups and50 E/I-to-membrane-to-output fields retained',
        correctness='Exact source algebra witnesses, original frozen numerical budgets and saved-source endpoint checks passed',
        runtime='Actual original6305-state standard/tight900ms runs plus declared background/inhibition/S controls completed',
        presentation='Actual five-type scientific curves and native group-object rasters inspected',
        security='Local artifacts only; no upload, installation, production edit or source-asset hash bypass',
        context='Existing source-limited/D-grade subtype mappings and all qualification-false flags preserved',
        no_independent_review_claimed=True,
        limitations=['No whole-retina background or cross-eccentricity calibration',
                     'No optical color-opponency or subtype physiological fit',
                     'Driven controls remain conditional;900ms main assay does not establish late rod recovery']))
    dump(descriptor_path, dict(name='RGC_FUNCTIONAL_V1', status=current['status'],
        entry='retina_sim.rgc_functional_v1.RGCFunctionalV1',
        runnable_configuration='configurations/research/RGC_FUNCTIONAL_V1_candidate.json',
        candidate_pre_migration_status_is_archival=True,
        validated_layer_identity=current['RGC_identity'],
        result=str((out / 'FINAL_RESULT.json').relative_to(ROOT)),
        source_coupled_inner_run_completed=True, all_existing_types_retained=True,
        parameter_fit=False, production_adopted=False, background_qualified=False, FIT_READY=False))
    lines = ['# RGC functional v1: local conditional closeout', '',
        'All five existing types completed their declared interface and functional checks. '
        'The final main assay evolves all6305 original coupled inner states for900ms, '
        'with unchanged AII coupling, original event clocks and native group-object sampling.', '',
        '| Type | Groups | Main expected rate range (Hz) | Native events | Status |',
        '|---|---:|---:|---:|---|']
    for r in types:
        lines.append(f"| {r['type']} | {r['groups']} | {r['source_rate_min_Hz']:.6f}–{r['source_rate_max_Hz']:.6f} | {r['actual_sampled_object_events']} | {r['status']} |")
    lines += ['', f"All50 standard/tight readouts passed; maximum error/allocation = {result['max_numerical_error_over_budget']:.9g}.",
        f"Saved full-inner900ms source endpoint maximum error/allocation = {result['source_900ms_endpoint_max_error_over_budget']:.9g}.",
        'New complete inner-state byte hashes are recorded separately and need not equal the earlier machine run. '
        'The old frozen assets were not modified and no source-asset SHA gate was bypassed.', '',
        'The earlier1ms cached replay remains explicitly conditional and retains its endpoint allocation exceedance. '
        'It was followed by this original coupled qualification; no acceptance threshold was relaxed.', '',
        'The S-bistratified type responds positively to selective S_ON_BC terminal input; its common-input net response remains mixed/OFF-like. '
        'Optical color opponency and subtype-specific physiology remain unvalidated. '
        'Constant-background, inhibition-removal and selective-S controls use the declared driven original equations.', '',
        'All five types retain their inherited90Hz/-53mV/1.67mV output mapping. Four existing subtype mappings remain D engineering transfers. '
        'No Cell6 normalization, target-PSTH rescaling, parameter optimization, new connection or mechanism was introduced.', '',
        'The original post-tick source expected rates, pre-bin expected binary counts, actual binary draws and native within-bin event times '
        'are separate data. The endpoint has no sampled future bin; no fine-cell spike trains are replicated.', '',
        'The900ms assay does not establish complete late rod-response recovery or temporal calibration. '
        'STATE_END.npz is a final inner-state comparison artifact, not a resumable sampler/native/global checkpoint.', '',
        f"This session model wall time: {ledger['model_wall_seconds_this_session']:.3f}s; historical plus this session: {ledger['historical_plus_this_session_seconds']:.3f}s.",
        'The old ledger and deadline remain archival. Each model subprocess kept its900s guard. '
        'No full optics or global retina simulation was launched.', '',
        'FIT_READY=false; background_qualified=false; production_adopted=false. '
        'No whole-retina, cross-eccentricity, subtype-calibration or natural-video validation is claimed.', '',
        'Next single step: return to the already agreed cross-eccentricity calibration stage; do not automatically start a large fit.', '']
    (out / 'CLOSEOUT.md').write_text('\n'.join(lines), encoding='utf8')
    ledger['new_output_bytes'] = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    ledger['session_status'] = 'CLOSED_AFTER_RGC_FUNCTIONAL_V1'
    ledger['latest_result'] = str((out / 'FINAL_RESULT.json').relative_to(ROOT))
    (out / 'RESOURCE_LEDGER.json').write_text(json.dumps(ledger, indent=2) + '\n', encoding='utf8')
    print(json.dumps(current))


if __name__ == '__main__':
    main()
