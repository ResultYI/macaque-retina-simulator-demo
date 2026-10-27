"""Restore named backup assets, verify real67#33 intake, run data-only checks.

Copies are exclusive and source-hashed. No movie decode, model advance or fit.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BACKUP = Path(r'D:\macaque_retina_simulator.backup_20261026_v1')
STAGE = ROOT/'reports/local_patch_fit_real_intake_v1'
DATA = 'data/source_data/schottdorf_lee2021_gin_v1/source'
REPORT = 'reports/cross_layer_functional_fit_20261021_v1'
from tools.run_local_patch import sha,read,write

ASSETS = [DATA+'/README.md', DATA+'/data/lSS01159.txt', DATA+'/data/lSS01160.txt',
    'reports/pro_current_implementation_20261025_v1/dataset_local_audit.json',
    'configurations/natural_movie_matched_cohort_v1.json',
    *[REPORT+'/'+name for name in ('source_spike_index.json','real_supervision_arrays.npz',
        'source_RF_index.json','movie_clock.json','source_movie_stream_metadata.json','data_contract.json',
        'functional_v1/protocol.json','functional_v1/protocol_disposition.json',
        'regional_v2/protocol.json','regional_v2/source_input_registration.json',
        'regional_v2/source_bindings.json','regional_v2/source_supervision.npz',
        'regional_v2/type_admission.json','regional_v2/native_parasol/result.json')],
    'tools/fit_regional_functional.py']


def execute():
    import numpy as np
    from retina_sim.natural_movie_supervision import (source_roster,read_raw_spike_tables,
        load_prepared_real_trial,bin_live_trial)
    started,cpu = time.time(),time.process_time()
    STAGE.mkdir()
    entries = []
    for relative in ASSETS:
        source,destination = BACKUP/relative,ROOT/relative
        expected = sha(source)
        if destination.exists():
            if sha(destination) != expected:
                raise ValueError('Existing destination differs; no overwrite: '+relative)
            action = 'IDENTICAL_EXISTING_PRESERVED'
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            with source.open('rb') as inp,destination.open('xb') as out:
                shutil.copyfileobj(inp,out)
            action = 'COPIED_EXCLUSIVE'
        assert sha(destination) == expected and sha(source) == expected
        entries.append(dict(path=relative,source=str(source),sha256=expected,
            bytes=destination.stat().st_size,action=action))
    write(STAGE/'RESTORE_RECEIPT.json',dict(backup_read_only=True,entries=entries,
        restored_bytes=sum(e['bytes'] for e in entries if e['action']=='COPIED_EXCLUSIVE'),
        movie_copied=False,movie_source=str(BACKUP/DATA/'stimuli/1x10_256.mpg'),
        default_parameter_or_environment_overwritten=False))
    directory = ROOT/REPORT
    roster = source_roster(ROOT/DATA/'README.md')
    index = read(directory/'source_spike_index.json')
    summaries = []
    for number in (1159,1160):
        metadata = roster[f'lss{number:05d}.txt']
        source = ROOT/DATA/f'data/lSS{number:05d}.txt'
        parsed = read_raw_spike_tables(source,metadata)
        unique = [p for p in parsed if p['exact_duplicate_of_segment'] is None]
        assert len(unique) == 1
        record_id = f'lSS{number:05d}_segment0'
        record = next(r for r in index['records'] if r['record_id']==record_id)
        assert sha(source) == record['file_sha256']
        assert record['source_cell_key']=='67#33' and record['eccentricity_deg']==4.73
        assert record['RF_source'] is None and record['project_RF_binding'] is None
        totals = []
        for trial in [t for t in record['trials'] if t['is_live_movie_repeat']]:
            ordinal = trial['column_zero_based']
            loaded = load_prepared_real_trial(directory,record_id,ordinal)
            raw = unique[0]['ticks'][ordinal]
            np.testing.assert_array_equal(raw,loaded['raw_context_ticks_100us'])
            np.testing.assert_array_equal(bin_live_trial(raw,record['live_seconds']),loaded['counts_1ms'])
            assert int(loaded['counts_1ms'].max()) <= 1
            totals.append(int(loaded['counts_1ms'].sum()))
        summaries.append(dict(record_id=record_id,trials=len(totals),
            counts_shape=[len(totals),record['live_seconds']*1000],live_events_per_trial=totals,
            raw_and_prepared_exact=True,multicount_bins=0,RF_source=None,project_RF_binding=None,
            duplicate_segments=[p['segment_index'] for p in parsed if p['exact_duplicate_of_segment'] is not None]))
    rejected = []
    for record_id,trial in [('lSS01159_segment1',0),('lSS01159_segment0',6),('lSS01160_segment1',0)]:
        try:
            load_prepared_real_trial(directory,record_id,trial)
        except ValueError:
            rejected.append([record_id,trial])
        else:
            raise AssertionError('Duplicate or maintained activity must not become a movie trial')
    protocol = read(directory/'regional_v2/protocol.json')
    disposition = read(directory/'functional_v1/protocol_disposition.json')
    assert all(c['source_cell_key'] != '67#33' for c in protocol['cases'])
    evidence = dict(status='REAL67_33_INTAKE_VERIFIED_ALIGNMENT_BLOCKED',records=summaries,
        source_units=dict(tick_seconds=.0001,bin_seconds=.001,source_playback_Hz=150),
        rejected_duplicate_or_maintained_activity=rejected,
        raw_spike_clipping_smoothing_or_synthesis=False,
        actual_live_origin_frame_zero_based=751,one_frame_physical_onset_discrepancy_retained=True,
        existing_patch_forward_interval=[4.993,4.9961],existing_patch_native_frames=[748,749,750],
        real_live_origin_in_existing_frame_clock_seconds=751/150,
        real_live_seconds_covered_by_existing_forward=0,
        genuine_real_spikes_paired_to_engineering_forward=False,
        functional_v1_disposition=disposition,
        regional_protocol_status=protocol['status'],target67_33_in_regional_protocol=False,
        methods={k:protocol[k] for k in ('data_selection','parameters','objective','optimizer','evaluation')},
        blockers=['67#33 has no source2D RF/project optical input binding',
            'Existing tested patch frame748..750 engineering inputs do not cover source liveframe751',
            '67#33 is outside the recovered8-cell regional_v2 scored cohort; do not invent a split',
            'No current graph/input qualified grouped background or fit-horizon numerical qualification'],
        note='Recovered prior methods are provenance, not automatic authorization or qualification to train the current patch',
        model_advance_calls=0,optimizer_steps=0)
    write(STAGE/'REAL_DATA_AND_PROTOCOL.json',evidence)
    # Invoke only the original driver data interface, never its fitting branch.
    output = STAGE/'original_driver_data_preflight'
    command = [sys.executable,'-I','-X','utf8','-B',str(ROOT/'tools/fit_regional_functional.py'),
        '--protocol',str(directory/'regional_v2/protocol.json'),
        '--bindings',str(directory/'regional_v2/source_bindings.json'),
        '--output-dir',str(output),'--check-inputs-only']
    with (STAGE/'original_driver_stdout.log').open('x',encoding='utf8') as stream:
        completed = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if completed.returncode:
        raise RuntimeError('Original data-only preflight failed; preserved log')
    write(STAGE/'RESOURCE_AND_COMPLETION.json',dict(status='COMPLETE_DATA_ONLY',
        wall_seconds=time.time()-started,parent_CPU_seconds=time.process_time()-cpu,
        copied_files=len(entries),model_workers=0,optimizer_steps=0,
        original_driver_command=command,exit_code=completed.returncode))
    write(STAGE/'REQUEST_STOP',dict(stage_closed=True,model_or_fit_runs_authorized=False))
    print(json.dumps(dict(status=evidence['status'],restored_files=len(entries),model_advanced=False)))


if __name__=='__main__':execute()
