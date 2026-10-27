"""Freeze one training-input-only, no-loss engineering prefix before running it."""
from pathlib import Path
import copy, hashlib, json, math, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.contracts import canonical_hash
from retina_sim.group_time_storage import save_bundle
from tools.validate_ac_vs_native_support import load


def main():
    out=ROOT/'reports/bc_ac_fullchain_terra_v1/pilot_preparation_v2'
    out.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configurations/calibration/bc_ac_fullchain_candidate_v1.json').read_text(encoding='utf8'))
    _,source,ident=load(ROOT/config['prepared_model'])
    definition=json.loads((ROOT/config['targets']['directory']/'target_definition.json').read_text(encoding='utf8'))
    group=definition['groups'][0]
    if group['split']!='train':raise ValueError('First original group must be training')
    with np.load(ROOT/config['targets']['directory']/'targets_native_100us.npz',allow_pickle=False) as arrays:
        stimulus=arrays['g0_stimulus_raw'].copy()
    if stimulus.shape!=(71000,) or hashlib.sha256(stimulus.tobytes()).hexdigest()!=group['stimulus_sha256']:raise ValueError('Source stimulus identity mismatch')
    t0=source['time_seconds'];n=11000;stop=t0+n*.0001
    input_contract=dict(formula='cone_background_td * (1 + 0.05 * (raw_stimulus[j] - raw_stimulus[0]))',
        spatial_pattern='uniform on the selected native cones',rod='frozen source background R*/s',
        contrast_scale=.05,physical_stimulus_unit=None,physical_illuminance_mapping=None,identity_status='engineering conditional; not calibrated photometry')
    clock_contract=dict(source_export_origin_s=0.,source_dt_s=.0001,model_origin_s=t0,input_fps=150.,
        sampling='j=max(0,floor((global_frame_time-model_origin)/source_dt)); zero-order hold',
        relative_index='stimulus and E/I exported rows j share source sample grid',absolute_acquisition_latency_s=None,
        spike_index_origin='unresolved and unused',free_time_shift=False)
    input_id=canonical_hash(input_contract);clock_id=canonical_hash(clock_contract)
    first=math.floor((t0-ident['protocol']['input_origin_seconds'])*150.)
    last=math.floor((np.nextafter(stop,-np.inf)-ident['protocol']['input_origin_seconds'])*150.)
    native={};rod={};indexes={}
    base=np.asarray(source['input_reference']['native_background_td'],dtype=np.float64)
    rod_reference=np.asarray(source['input_reference']['rod_background_rstar_s'],dtype=np.float64)
    rod_base=np.broadcast_to(rod_reference,source['states']['rods','rod_membrane','V'].shape).copy()
    for frame in range(first,last+1):
        j=max(0,math.floor(((ident['protocol']['input_origin_seconds']+frame/150.)-t0)/.0001))
        if j>=n:raise ValueError('Input frame sampled beyond frozen prefix')
        drive=base*(1.+.05*(float(stimulus[j])-float(stimulus[0])))
        if not np.isfinite(drive).all() or (drive<0).any():raise ValueError('Invalid native drive; no clipping')
        native[frame]=drive;rod[frame]=rod_base.copy();indexes[str(frame)]=j
    payload=dict(stimulus_sha256=group['stimulus_sha256'],native_inputs=native,rod_inputs=rod,model_origin_s=t0,model_stop_s=stop,
        source_origin_s=0.,source_dt_s=.0001,source_samples=71000,input_driver_identity=input_id,clock_bridge_identity=clock_id,native_unit='td',rod_unit='R*/s')
    fields=ident['selection']['fields']
    names=['ports__cones:release','ports__HC:Vh','ports__diffuse_BC_ON:voltage','ports__diffuse_BC_ON:effective_release',
        'ports__cross_OFF_ON_parasol_AC_v1:voltage','ports__cross_OFF_ON_parasol_AC_v1:s','ports__cross_OFF_ON_parasol_AC_v1:gE',
        'ports__rod_BC:voltage','ports__AII_AC:voltage','ports__AII_AC:gap_current','ports__AII_AC:release',
        'ports__parasol_RGC_ON:gE','ports__parasol_RGC_ON:gI','ports__parasol_RGC_ON:gI_crossover','ports__parasol_RGC_ON:voltage']
    selected=[]
    for name in names:
        row=fields[name]
        if not row['ids']:raise ValueError('Empty fixed recording selection '+name)
        selected.append(dict(name=name,population=row['key'][0],port=row['key'][1],unit=row['unit'],global_ids=row['ids']))
    limits=dict(single_worker_wall_seconds=1800,process_RSS_limit_bytes=17179869184,available_RAM_floor_bytes=4294967296,
        rationale='~570 s estimated from measured 51.83 s per 0.1 s; 1800 s single worker safety limit, no cumulative limit')
    config['launch'].update(approved_contract=True,fixed_input_driver=dict(identity=input_id,stimulus_payload_sha256={group['stimulus_sha256']:canonical_hash(payload)},contract=input_contract),
        fixed_clock_bridge=dict(identity=clock_id,contract=clock_contract),fixed_response_polarity_mapping=dict(E='negative conductance proxy',I='positive conductance proxy',physical_current_unit=None,used_for_loss=False),
        fixed_output_selection=selected,initial_state_history_policy=dict(mode='fresh_verified_source',source_seed_sha256=canonical_hash(source),fixed_boundary='same saved conditional equilibrium reservoir'),
        forward_resource_contract=limits,final_parameter_choice_and_bounds=dict(status='all fixed for diagnostic only',fit_selection='pending training sensitivity and timing qualification'))
    config['diagnostic_pilot']=dict(purpose='TRAINING_INPUT_FORWARD_ONLY_NO_LOSS_OR_FIT',source_start_index=0,source_sample_count=n,
        complete_initial_baseline_s=1.,active_prefix_s=.1,source_original_samples=71000)
    save_bundle(out/'drive',kind='validation_evidence',identity=dict(input=input_id,clock=clock_id,stimulus=group['stimulus_sha256']),payload=payload,qualification=dict(default_adopted=False,fit_eligible=False))
    (out/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    (out/'values.json').write_text('{}\n',encoding='utf8')
    frozen=dict(input_contract=input_contract,clock_contract=clock_contract,source_stimulus_sha256=group['stimulus_sha256'],frame_sample_indices=indexes,
        native_frame_payload_sha256=canonical_hash(payload),source_seed_sha256=canonical_hash(source),sample_count=n,
        model_start_s=t0,model_stop_s=stop,limits=limits,optimizer_steps=0,scope='pre-registered training input forward only; no observed target used',
        absolute_acquisition_latency_s=None,old_historical_test_read=False)
    (out/'PRE_REGISTERED.json').write_text(json.dumps(frozen,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(prepared=str(out),frames=len(native),source_samples=n,stop=stop,payload_sha256=canonical_hash(payload))))

if __name__=='__main__':main()
