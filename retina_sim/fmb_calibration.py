"""FMB assay constraints and recording intake, never automatic parameter values."""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
from .contracts import ContractError, Quantity


def identifiability_report():
    return {
        'voltage_only_scale': {
            'symmetry': 'C and all conductances/currents scaled by the same positive factor preserve V(t)',
            'conditions': 'No independent absolute injected-current/conductance reference; external currents/gaps must also scale',
            'identifiable': ['conductance/C combinations', 'time constants given a fixed input scale'],
            'not_separately_identifiable': ['C', 'gL', 'gE', 'unknown baseline inhibitory/gap conductances'],
            'needed': 'Same-condition absolute current or total input-admittance measurement; remaining current components still need separation'},
        'effective_drive_scale': {
            'symmetry': 'u -> b*u and K -> b*K preserve Hill occupancy for b>0',
            'needed': 'Declared/calibrated effective-drive scale; nominal puff concentration is not a cone-drive calibration'},
        'release': 'Voltage traces alone do not identify the voltage-to-release mapping and downstream gain',
        'fit_enabled': False,
    }


def receptor_shape_capability():
    return dict(model='one first-order receptor gate',
        assumptions=['constant delivered receptor drive during the step', 'fixed voltage clamp',
                     'initial gate equilibrated to prestimulus drive', 'OFF activation step increases drive'],
        supported='monotone approach to occupancy', early_peak_then_decay=False,
        caveat='A plateau/peak summary alone is insufficient without peak/plateau windows and delivered drive waveform',
        action='Report model/protocol limitation; do not infer tau or add desensitization/channels automatically')


def inverse_reported_mean_resistance(value_GOhm):
    if isinstance(value_GOhm, bool) or not np.isfinite(value_GOhm) or value_GOhm <= 0:
        raise ContractError('Positive finite reported input resistance required')
    return dict(value=1./value_GOhm, unit='nS',
        meaning='inverse_of_reported_mean_input_resistance', is_gL=False,
        population_mean_conductance=False, uncertainty='No inverse-error/CI or cellwise conductance distribution inferred')


def input_resistance_from_steady_iv(voltage_command, outward_current):
    """Total small-signal I-V slope from explicitly selected steady measurements."""
    if not isinstance(voltage_command,Quantity) or not isinstance(outward_current,Quantity):
        raise ContractError('Typed steady voltage/current observations required')
    V=voltage_command.require('V','voltage_clamp_command')
    I=outward_current.require('A','voltage_clamp_current_positive_outward',V.shape)
    if V.ndim!=1 or len(V)<2:raise ContractError('At least two steady I-V observations required')
    d=V-np.mean(V);denominator=float(d@d)
    if denominator<=0:raise ContractError('Distinct voltage commands required')
    conductance=float(d@(I-np.mean(I))/denominator)
    if conductance<=0:raise ContractError('Positive outward-current input slope required')
    return dict(input_resistance_Ohm=1./conductance,total_input_admittance_S=conductance,
        is_gL=False,capacitance_identified=False,points=len(V),
        interpretation='Measured total steady small-signal slope; no receptor/gap/leak separation')


def passive_resistance_ceiling(leak_S):
    """Conditional model bound, not a measured leak estimate or a fitting rule."""
    if isinstance(leak_S,bool) or not np.isfinite(leak_S) or leak_S<=0:
        raise ContractError('Positive finite leak conductance required')
    return dict(maximum_Rin_Ohm=1./leak_S,
        assumptions='Passive one-compartment steady response with fixed receptor gates and nonnegative additional input load',
        usage='Conditional compatibility check only; never set gL to inverse reported mean Rin')


def plateau_peak_ratio(times_s, current, *, baseline_window_s, peak_window_s,
                       plateau_window_s, inward_negative):
    """Observed current metric with explicit windows; no kinetic fit or resampling."""
    if not isinstance(current, Quantity):raise ContractError('Typed experimental current required')
    values=current.require('A','voltage_clamp_current_positive_outward')
    times=np.asarray(times_s,dtype=float)
    if (times.ndim!=1 or values.shape!=times.shape or len(times)<3 or
            not np.all(np.isfinite(times)) or np.any(np.diff(times)<=0) or type(inward_negative) is not bool):
        raise ContractError('Explicit increasing experimental sample times, current and sign required')
    def mask(window):
        if len(window)!=2 or not np.isfinite(sum(window)) or window[0]>=window[1]:
            raise ContractError('Explicit finite ordered observation window required')
        chosen=(times>=window[0])&(times<window[1])
        if not np.any(chosen):raise ContractError('Observation window has no samples; no interpolation')
        return chosen
    baseline=float(np.mean(values[mask(baseline_window_s)]))
    response=(baseline-values) if inward_negative else (values-baseline)
    peak=float(np.max(response[mask(peak_window_s)]));plateau=float(np.mean(response[mask(plateau_window_s)]))
    if peak<=0 or plateau<0:raise ContractError('Current polarity/baseline does not produce the declared response')
    return dict(metric='plateau_peak_ratio', value=plateau/peak, unit='dimensionless',
        baseline_A=baseline, peak_response_A=peak, plateau_response_A=plateau,
        peak_window_s=list(peak_window_s), plateau_window_s=list(plateau_window_s),
        interpretation='Observed waveform summary only; not tau, Hill K or a fitted coefficient')


class FMBConstraintCatalog:
    def __init__(self, root):
        self.root=Path(root).resolve()
        path=self.root/'configurations/fmb_calibration_constraints_v1.json'
        self.data=json.loads(path.read_bytes())
        if self.data.get('schema_version')!='fmb-constraints-1.0':raise ContractError('Unknown FMB constraint schema')
        for row in self.data['constraints']:
            if row['usage']!='observation_constraint_only' or row['runtime_parameter_binding'] is not None:
                raise ContractError('FMB assay constraints cannot become runtime coefficients')
        for asset in self.data['source_assets']:
            if hashlib.sha256((self.root/asset['path']).read_bytes()).hexdigest()!=asset['sha256']:
                raise ContractError('FMB source asset identity mismatch: '+asset['path'])
        self.records={row['constraint_id']:row for row in self.data['constraints']}

    def select(self, constraint_id):
        if constraint_id not in self.records:raise ContractError('Unknown FMB constraint')
        return copy.deepcopy(self.records[constraint_id])

    def runtime_parameter(self, *args, **kwargs):
        raise ContractError('FMB observation constraints require a generative assay/identifiability contract; no direct parameter assignment')

    def load_digitized_shape(self):
        """Phase-domain figure data only; never invent a time vector for replay."""
        row=self.select('FMB_DACEY2000_FIG4')['digitization']
        folder=(self.root/row['directory']).resolve()
        if not folder.is_relative_to(self.root):raise ContractError('Digitization must stay in project')
        provenance=folder/'provenance.json'
        if hashlib.sha256(provenance.read_bytes()).hexdigest()!=row['provenance_sha256']:
            raise ContractError('Digitization provenance identity changed')
        meta=json.loads(provenance.read_bytes())
        for name,expected in meta['files'].items():
            path=(folder/name).resolve()
            if not path.is_relative_to(folder) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ContractError('Digitized figure asset identity changed')
        if meta['representation']!='figure_digitization' or meta['raw_recording'] or meta['fit_enabled']:
            raise ContractError('Digitized shape cannot claim raw recording or calibration')
        with np.load(folder/'fig4B_phase_voltage.npz',allow_pickle=False) as archive:
            arrays={name:archive[name].copy() for name in archive.files}
        required=('phase_cycles','delta_voltage_V','voltage_digitization_error_bound_V')
        n=len(arrays[required[0]])
        if (any(arrays[k].shape!=(n,) or not np.all(np.isfinite(arrays[k])) for k in required) or
                np.any(np.diff(arrays['phase_cycles'])<=0) or
                np.any(arrays['voltage_digitization_error_bound_V']<0) or
                'time_seconds' in arrays):
            raise ContractError('Invalid phase-domain digitization; no implicit seconds')
        return arrays,meta

    def inspect_inputs(self, path):
        path=Path(path).resolve();data=json.loads(path.read_bytes())
        if data.get('schema_version')!='fmb-calibration-inputs-1.0':raise ContractError('Unknown FMB recording intake schema')
        reports={}
        for row in data['recordings']:
            constraint=self.select(row['constraint_id']);missing=[]
            for name in ('recording_id','target_type','species','recording_region','temperature_C','preparation','time_origin',
                         'boundary','baseline','observed_unit','observed_meaning','input_protocol'):
                if row.get(name) is None or row.get(name)=='' or row.get(name)=={}:missing.append(name)
            raw=row.get('data') or {}
            for name in ('path','sha256','time_key','response_key','representation'):
                if not raw.get(name):missing.append('data.'+name)
            if raw.get('time_unit')!='s':missing.append('data.time_unit_must_be_s')
            if row.get('observed_unit') not in ('A','V'):missing.append('observed_unit_must_be_explicit_SI_A_or_V')
            if raw.get('representation') not in ('raw_samples','figure_digitization','software_fixture'):
                missing.append('data.supported_representation')
            if row.get('target_type') and row['target_type'] not in ('FMB','OFF_midget_BC'):
                missing.append('target_type_not_FMB')
            context=constraint.get('conditions',{})
            allowed=context.get('species')
            if isinstance(allowed,str):allowed=[allowed]
            if allowed and row.get('species') and row['species'] not in allowed:
                missing.append('species_outside_source_context')
            region=row.get('recording_region')
            if region is not None:
                if (not isinstance(region,dict) or region.get('unit')!='mm' or
                        region.get('frame')!='BC_recording_site_anatomical_retina' or
                        not isinstance(region.get('distance'),(int,float)) or not np.isfinite(region['distance'])):
                    missing.append('recording_region_requires_source_frame_mm')
                else:
                    bounds=context.get('recording_region_mm')
                    if isinstance(bounds,list) and not bounds[0]<=region['distance']<=bounds[1]:
                        missing.append('recording_region_outside_source_context')
                    if isinstance(bounds,dict) and region['distance']<=bounds['lower_exclusive']:
                        missing.append('recording_region_outside_source_context')
            temperature=row.get('temperature_C')
            expected=context.get('temperature_C')
            if expected and temperature is not None and (not isinstance(temperature,(float,int)) or
                    not np.isfinite(temperature) or not expected[0]<=temperature<=expected[1]):
                missing.append('temperature_outside_source_context')
            if raw.get('representation')=='figure_digitization' and not raw.get('digitization_uncertainty'):
                missing.append('data.digitization_uncertainty')
            if raw.get('path') and raw.get('sha256'):
                file=(path.parent/raw['path']).resolve()
                if not file.is_file():missing.append('data.file_not_found')
                elif hashlib.sha256(file.read_bytes()).hexdigest()!=raw['sha256']:
                    raise ContractError('Experimental recording differs from frozen supplied identity')
            prediction=[]
            if constraint['assay'] in ('input_resistance','receptor_puff','glycine_event'):
                prediction.append('BC_experimental_voltage_clamp_boundary_and_current_readout_not_implemented')
            if not row.get('model_input_mapping'):prediction.append('model_input_mapping_and_scale')
            if constraint['assay']=='receptor_puff' and not row.get('delivered_receptor_waveform'):
                prediction.append('delivered_receptor_drive_not_just_puff_command')
            if constraint['assay']=='light_voltage' and not row.get('stimulus_response_synchronization'):
                prediction.append('stimulus_response_synchronization_and_spatial_mapping')
            prediction += ['candidate_source_bound_parameters', 'actual_W_tau_initial_mapping_for_candidate']
            analysis=row.get('analysis_protocol') or {}
            fields={'input_resistance':('steady_IV_selection_windows_s','voltage_command_key'),
                    'receptor_puff':('baseline_window_s','peak_window_s','plateau_window_s','holding_V'),
                    'glycine_event':('baseline_window_s','event_window_s','holding_V','filter_protocol'),
                    'light_voltage':('baseline_window_s','voltage_reference','cycle_windows_s')}
            analysis_missing=[name for name in fields[constraint['assay']] if analysis.get(name) is None]
            reports[row['constraint_id']]=dict(observation_extraction_ready=not missing and not analysis_missing,
                recording_arrays_ready=not missing,
                missing_assay_analysis_fields=analysis_missing,
                missing_recording_fields=missing, candidate_prediction_ready=False,
                candidate_prediction_blockers=prediction,
                source_binding_status=constraint['source_binding_status'], fit_enabled=False,
                data_representation=raw.get('representation'),physiological_validation_claim=False)
        return reports

    def load_recording(self, path, constraint_id):
        """Read a supplied single experimental trace only after its explicit contract."""
        path=Path(path).resolve();status=self.inspect_inputs(path)[constraint_id]
        if not status['recording_arrays_ready']:
            raise ContractError('FMB_RECORDING_NOT_READY: '+', '.join(status['missing_recording_fields']))
        contract=next(row for row in json.loads(path.read_bytes())['recordings'] if row['constraint_id']==constraint_id)
        raw=contract['data'];file=(path.parent/raw['path']).resolve()
        with np.load(file,allow_pickle=False) as arrays:
            if raw['time_key'] not in arrays or raw['response_key'] not in arrays:
                raise ContractError('Explicit experimental NPZ array keys missing')
            times=np.asarray(arrays[raw['time_key']],dtype=float)
            values=np.asarray(arrays[raw['response_key']],dtype=float)
        if times.ndim!=1 or values.shape!=times.shape or len(times)<2 or not np.all(np.isfinite(times)) or np.any(np.diff(times)<=0):
            raise ContractError('Finite single-trace samples and increasing seconds required')
        return times,Quantity(values,contract['observed_unit'],contract['observed_meaning'],
            'synthetic_fixture' if raw['representation']=='software_fixture' else 'experimental_observation'),contract


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Inspect FMB evidence and real-recording intake; no model run or fit')
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--inputs',type=Path)
    args=parser.parse_args()
    catalog=FMBConstraintCatalog(args.root)
    inputs=args.inputs or args.root/'configurations/fmb_calibration_inputs_v1.json'
    print(json.dumps(dict(status='FMB_OBSERVATION_INTAKE_ONLY_NO_FIT',
        recording_readiness=catalog.inspect_inputs(inputs),identifiability=identifiability_report(),
        receptor_capability=receptor_shape_capability()),ensure_ascii=False,indent=2))
