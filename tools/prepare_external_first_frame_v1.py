"""Bind the existing one-exposure optical fixture to both completed local S0 seeds."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.contracts import ContractError,canonical_hash
from retina_sim.isetbio_input import reference_equivalent_drive
from retina_sim.spatial_group_graph import digest
from tools.cross_position_context_v1 import read,write

VIDEO=ROOT/'reports/cone_video_113_4s_eccentric_aperture_0_6deg_20261029_v1'
NEAR=ROOT/'reports/layerwise_physiology/cross_position_functional_coverage_v1_20261029/restored_resume_v1'
FAR=ROOT/'reports/position_S0_far_originaltarget_20261029_resume_v2'
HISTORY=6494.377218400068


def locations():
    return dict(near=dict(prepared=NEAR,background=NEAR/'background_24g_v3',optical=NEAR/'background_v1'),
        far=dict(prepared=FAR/'prepared',background=FAR/'background',optical=FAR/'background'))


def prepare(out):
    start=time.perf_counter();out.mkdir(exist_ok=False)
    if not all(importlib.util.find_spec(x) for x in ['numpy','scipy','numba']):
        raise ContractError('An existing required numeric runtime is absent')
    source=VIDEO/'first_frame/capture.mat';geometry=VIDEO/'geometry.npz'
    done=read(VIDEO/'first_frame/RESULT.json');expected=next(v['sha256'] for k,v in done['output_files'].items() if k.replace('\\','/').endswith('first_frame/capture.mat'))
    if digest(source)!=expected:raise ContractError('Existing first-frame capture SHA differs')
    raw=loadmat(source,variable_names=['expected_counts','window_edges_seconds','manifest_json'])
    edges=raw['window_edges_seconds'].reshape(-1);duration=float(edges[-1]);meta=json.loads(raw['manifest_json'].item())
    if not np.array_equal(edges,[0.,1/30]) or meta['source_time_seconds']!=0. or meta['exposure_seconds']!=duration:
        raise ContractError('Only the existing complete first exposure,0..1/30s, is allowed')
    with np.load(geometry,allow_pickle=False) as z:g={k:z[k].copy() for k in z.files}
    if raw['expected_counts'].shape!=(1,len(g['source_table_rows'])):raise ContractError('Fixture cell/time axes differ')
    lookup={int(row):i for i,row in enumerate(g['source_table_rows'])}
    numerical_source=ROOT/'reports/ac_vs_spatial_common_v1/run_060_v1/model_identity.json'
    old=read(numerical_source)['common']['protocol']
    keys=['schema','BC_RGC_period_seconds','input_fps','native_observation_delay_seconds','recording_step_seconds',
        'rod_step_seconds','monitor_step_seconds','reference','candidate']
    protocol={key:deepcopy(old[key]) for key in keys}
    protocol.update(start_seconds=0.,stop_seconds=duration,clock_origin_seconds=0.,input_origin_seconds=0.,
        input_frame_indices=list(range(5)),checkpoint_seconds=duration)
    if old['input_fps']!=150. or duration*150!=5.:raise ContractError('Exact5-hold consumer partition required')
    bindings={};summaries={};source_pins={str(p.relative_to(ROOT)):digest(p) for p in [source,geometry,VIDEO/'optics_config.json',numerical_source]}
    for label,paths in locations().items():
        for path in [paths['prepared']/'local_static/selection_identity.json',paths['background']/'seed/manifest.json',
                paths['background']/'inner_boundary/manifest.json',paths['background']/'HC_POSITION_BINDING.json',
                paths['background']/'FIXED_LMS_BRIDGE.json',paths['optical']/'APERTURE_GEOMETRY.npz']:
            if not path.is_file():raise ContractError('Declared local dependency absent: '+str(path))
            source_pins[str(path.relative_to(ROOT))]=digest(path)
        with np.load(paths['optical']/'APERTURE_GEOMETRY.npz',allow_pickle=False) as z:
            fine=z['fine_rows'];indices=np.array([lookup[int(row)] for row in fine]);types=z['cone_types'].astype('U1');counts=z['local_group_counts']
            for left,right,name in [(g['xy_mm'][indices],z['positions_mm'],'positions'),(g['apertures_um'][indices],z['reference_diameter_um'],'diameters'),(g['types'][indices],types,'types')]:
                if not np.array_equal(left,right):raise ContractError('Fixture/local '+name+' differ')
        rates=raw['expected_counts'][:,indices]/duration
        bridge=deepcopy(read(paths['background']/'FIXED_LMS_BRIDGE.json')['profile'])
        scope=canonical_hash(dict(fixture=digest(source),fine_rows=fine,semantics='existing static first-frame radiance held over its sole declared exposure'))
        bridge['capture_scope_sha256']=scope
        drive,report=reference_equivalent_drive(rates,types.tolist(),bridge,capture_scope_sha256=scope)
        cuts=np.r_[0,np.cumsum(counts)]
        group=np.array([drive[0,cuts[i]:cuts[i+1]].mean() for i in range(len(counts))],dtype=np.float64)
        if counts.sum()!=len(fine) or not np.isfinite(group).all():raise ContractError('Existing member-mean mapping differs')
        folder=out/label;folder.mkdir();np.savez_compressed(folder/'DRIVE.npz',group_native_drive=group,fine_rates_rstar_s=rates,fine_rows=fine)
        binding=dict(position=label,prepared=str(paths['prepared']),background=str(paths['background']),
            fine_members=len(fine),cone_groups=len(group),fixed_LMS_bridge=bridge,bridge_result=report,
            member_mean_rule='Existing actual fine-member arithmetic mean once',
            five150Hz_inputs_are_identical=True,interpolation='zero_order_hold; no new intermediate values',
            source_frame_index=0,exposure_edges_seconds=edges.tolist(),endpoint_side='left, no invented post-exposure input',
            drive_sha256=digest(folder/'DRIVE.npz'),seed_states_not_reinitialized=True)
        write(folder/'BINDING.json',binding);bindings[label]=binding
        summaries[label]=dict(members=len(fine),groups=len(group),native_drive_range=[float(group.min()),float(group.max())])
    optics=read(VIDEO/'optics_config.json')
    source_pins.update({p:digest(ROOT/p) for p in ['retina_sim/group_time_runner.py','retina_sim/hc_lateral_interface.py',
        'retina_sim/native_multirate.py','retina_sim/isetbio_input.py','tools/validate_ac_vs_timescale.py','tools/validate_ac_vs.py']})
    descriptor=dict(schema='existing-first-frame-external-input-1',source_pins=source_pins,protocol=protocol,
        fixture=dict(source=str(source.relative_to(ROOT)),geometry=str(geometry.relative_to(ROOT)),
            input='Existing113.AVI first frame at0s',contrast='Unchanged existing image pixel radiance; no new scalar contrast',
            duration_seconds=duration,fixed_radiance_gain=optics['fixed_radiance_gain'],
            S0_seed_gain=read(FAR/'background/OPTICAL_CONFIG.json')['fixed_radiance_gain'],
            gain_difference_preserved_not_normalized=True,phototransduction_already_applied=False,eye_movements=False),
        exact_existing_time_rule='Constant first-frame scene over declared1/30s exposure; partition into5 identical150Hz consumer holds',
        numerical_rule_source=str(numerical_source.relative_to(ROOT)),bindings=bindings,
        rod_rule='Existing independent engineering background unchanged; no spectral Rod capture added',
        calibrated_visual_contrast_protocol=False,new_scientific_parameter=False,
        FIT_READY=False,background_qualified=False,production_adopted=False)
    descriptor['experiment_id']=canonical_hash(descriptor);write(out/'PREPARATION.json',descriptor)
    write(out/'PREFLIGHT.json',dict(status='READY_EXISTING_EXTERNAL_FIXTURE',positions=summaries,
        source_capture_sha_verified=True,all_members_covered=True,positions_types_D_exact=True,
        dynamic_entry='LateralHCResearchRunner -> unchanged GroupTimeArm; DirectACVSRHS CPU',
        numerical_protocol_clocks_unchanged=True,no_new_scalar_contrast_or_pulse_timing=True,
        hardware_qualification_claim=False,new_neural_runs=0,preparation_wall_seconds=time.perf_counter()-start,
        historical_cumulative_seconds=HISTORY))
    print(json.dumps(dict(status='READY',positions=summaries,experiment_id=descriptor['experiment_id'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT/'reports'):raise ContractError('Fresh reports output required')
    prepare(output)
