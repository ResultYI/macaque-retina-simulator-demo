"""Inspect the actual Latimer assets for measured voltage/spike calibration.

Exports a minimal primary-data handoff contract when no paired voltage exists.
Does not fit, advance a network, download data, or alter any production profile.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.latimer2019 import read_cell6

BASE = ROOT/'reports/modeling_scope_20261020'
DATA = ROOT/'data/source_data/latimer2019_figshare_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def minimum_package():
    return dict(
        schema='measured_rgc_voltage_spike_handoff_1.0', status='AWAITING_PRIMARY_DATA_NOT_A_CANDIDATE',
        target_component='parasol_RGC_ON', other_RGC_classes_in_scope=False,
        requested_source='Raw two-cell dynamic-clamp records underlying Latimer2019 Figure3, or a new matched independent ON-parasol record',
        original_record_description=dict(seconds=19, repeat_counts=[18,24], Eexc_V=0., Einh_V=-.09,
                                         feedback_delay_s=.0001, raw_data_recovered=False),
        arrays=dict(
            time_s='Explicit sampled times, reference/trigger, sample rate and clock for each record',
            voltage_V='Raw physical membrane voltage per cell and repeat; no synthetic voltage inferred from currents',
            spike_times_s='Per-repeat detected spike times on the same physical clock, plus detector/export definition',
            trial_table='Cell, animal/species, trial, condition, waveform/repeat group, recording mode and source identities',
            injected_input='Applied gE/gI, Eexc/Einh, common scaling and holding/bias current when recorded; identify unavailable fields'),
        metadata_required=[
            'Primary source locator, file hash, author variable definitions and physical units',
            'Voltage origin, liquid-junction correction and whether already applied; no assumed 10mV shift',
            'Sampling, spike detection/index origin, filters, resampling, trigger and voltage/spike alignment',
            'Exact Fig3 cell IDs and mapping to animal/species, ON-parasol identification and recording conditions',
            'Dynamic-clamp reversal potentials, applied conductance scale and DC injection for each condition',
            'Original repeat-mean voltage preprocessing, spike treatment and interval exclusions'],
        proposed_analysis=dict(
            allowed_parameter_groups=[['mu'],['mu','beta']],
            fixed_parameters='Existing alpha=150Hz, C/gL/reversals, voltage-valued three-bin history and dt=1ms remain fixed',
            observation='Existing HistoryRGC softplus and 1-exp(-lambda*dt) binary likelihood',
            binning='Predeclare conversion of original times/voltage to retained bin sampling; reject multiple events/bin, no clipping or fit-selected time shifts',
            mean_voltage='Build the measured repeat-mean voltage independently within each declared split; never pool heldout targets into training preprocessing',
            split='Predeclare disjoint waveform/condition groups. If only one repeated 19s waveform exists, disjoint time blocks must be held together across repeats with history guard and the limited within-condition claim stated',
            independence='No reuse of the already-inspected Cell6 holdout as a new independent test; new cells/stimuli required for broader generalization claims',
            bounds='Declare justified units/bounds and selection criterion before fitting; do not expand after heldout inspection',
            candidate='Save source identities, exact coefficient delta, split hashes, training/validation/test likelihood and uncertainty; unadopted ON-parasol module only',
            standard_control='Only after candidate acceptance: compare the frozen short fullfield_flash_pair protocol under identical inputs and preparation; no new optical capture or long background without required authorization',
            physiological_claim='A measured-voltage conditional output fit does not calibrate passive membrane or upstream conductances'),
        completion=False, authors_contacted=False, communication_sent=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    before=json.loads((BASE/'standard_stimulus_rate_audit_v1/baseline_frozen_identity.json').read_bytes())
    for group in ('files','fixed_parameters_and_equations'):
        for rel,h in before[group].items():
            assert digest(ROOT/rel)==h, ('Protected baseline differs',rel)
    data=read_cell6(ROOT)
    fields=sorted(k for k in data if not k.startswith('__'))
    assert fields==sorted(['ExcCurrents','ExcStm','InhCurrents','InhStm','SpikeStm','SpikeTimes','fn'])
    metadata=json.loads((DATA/'article_metadata.json').read_bytes())
    rows=[]
    for f in metadata['files']:
        rows.append(dict(file=f['name'],declared_bytes=f['size'],declared_md5=f['computed_md5'],
                         inspected_local_bytes=f['name']=='Cell_6.mat',
                         description='Public metadata advertises currents, stimuli and spikes; no paired voltage advertised',
                         measured_voltage_field_in_local_file=False if f['name']=='Cell_6.mat' else 'NOT_DOWNLOADED_NOT_INSPECTED'))
    with (args.output/'public_file_inventory.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    paper=DATA/'latimer2019_PMC6989090.xml'
    assert digest(paper)=='09b5ce244f7abbdcc110046d0fc424f51c39055016e47c951098d49e2ec9bf9f'
    root=ET.parse(paper).getroot()
    dynamic=next(s for s in root.iter('sec') if s.findtext('title')=='Dynamic clamp recordings')
    text=' '.join(''.join(dynamic.itertext()).split())
    assert all(token in text for token in ('Cafaro','2013','100 μs','-90 mV','18 repeat','19 s','24 repeat'))
    figure=next(f for f in root.iter('fig') if f.attrib.get('id')=='fig3')
    source_tags=[e.tag for e in figure.iter() if e.tag in ('supplementary-material','media')]
    assert not source_tags
    supplementary=[dict(tag=e.tag,href=e.attrib.get('{http://www.w3.org/1999/xlink}href',''))
                   for e in root.iter() if e.tag in ('supplementary-material','media')]
    previous=BASE/'latimer_cell6_rgc_v1/calibration_decision.json'
    decision=json.loads(previous.read_bytes())
    assert decision['status']=='CALIBRATION_PROPOSAL_REJECTED_NO_PRODUCTION_ADOPTION'
    review=BASE/'rgc_voltage_hazard_source_audit_v1/primary_source_review.json'
    primary=json.loads(review.read_bytes())
    assert primary['paper']['recording_dynamic_clamp_conditions']['Ei_mV']==-90
    assert primary['paper']['reported_membrane_conditions']['Ei_mV']==-80
    old_scope=json.loads((BASE/'standard_stimulus_rate_audit_v1/scope_integrity.json').read_bytes())
    for rel,h in old_scope['source_sha256'].items():
        assert digest(ROOT/rel)==h,('Prior input/GUI source differs',rel)
    package=minimum_package();save(args.output/'minimum_primary_data_handoff.json',package)
    result=dict(schema='latimer_measured_voltage_spike_readiness_1.0',
        status='NO_MATCHED_PRIMARY_V_SPIKE_DATA_CALIBRATION_NOT_PERFORMED',
        useful_new_findings=dict(dynamic_clamp_Einh_mV=-90,CBEM_model_Einh_mV=-80,
            distinct_recording_cohorts=True,original_dynamic_clamp_seconds=19,
            original_repeat_counts=[18,24],original_source='Cafaro and Rieke2013, PMC3687519'),
        local_file=dict(path=str(DATA/'Cell_6.mat'),sha256=digest(DATA/'Cell_6.mat'),fields=fields,
                        contains_measured_voltage=False,contains_spikes=True,
                        voltage_cannot_be_reconstructed_as_an_independent_measurement=True),
        paper_attachments=supplementary,figure3_raw_data_attachment_found=False,
        six_file_scope='Six-file metadata inspected; only Cell6 bytes inspected. No claims about hidden fields in the five undownloaded files',
        previous_attempt=dict(path=str(previous),sha256=digest(previous),status=decision['status'],
            old_parameter_group=['alpha','mu'],old_conditional_fit_is_not_current_whitelist_calibration=True,
            old_holdout_already_inspected=True,new_optimization_performed=False),
        tested=['Cell6 identity and exact numeric schema','Primary XML dynamic-clamp cohort and conditions',
                'Fig3 attachment inventory','Prior rejected-fit status and holdout use',
                'Primary review model-versus-recording distinction','169 frozen video and 11 model/configuration identities',
                'Prior standard-input and GUI source identities'],
        untested=['Real V-spike intake: missing raw voltage','mu or mu+beta fit: not run',
                  'Independent candidate validation: not run','Candidate short standard-stimulus control: no candidate'],
        blockers=['Raw time-aligned per-repeat membrane voltage and spikes for the two Fig3 cells are not located in inspected public assets',
                  'Exact cell-specific voltage reference/corrections, applied conductance scale and DC bias are not supplied',
                  'No independent untouched matched data are available for a new adaptive fit/test'],
        current_authorization=dict(permission_requests=0,permission_denials=0,pending_actions=[],
            uncertain_executions=[],retried_denied_operations=0,authors_contacted=False),
        access_limits='eLife webpage challenge and inaccessible metadata/lab endpoints; public PMC text, Figshare description and GitHub source/metadata were readable. This does not establish that unpublished raw data do not exist',
        production_parameters_changed=False,production_defaults_changed=False,UI_changed=False,
        new_neural_seconds=0,matlab_launches=0,additional_datasets_downloaded=0,
        source_review_sha256=digest(review),intake_tool_sha256=digest(Path(__file__)),
        handoff_sha256=digest(args.output/'minimum_primary_data_handoff.json'))
    save(args.output/'result.json',result)
    print(json.dumps(dict(status=result['status'],fields=fields,tested_checks=len(result['tested']),
                          fit_performed=False,standard_control_performed=False,
                          protected_model_files=len(before['fixed_parameters_and_equations'])),indent=2))


if __name__=='__main__':main()
