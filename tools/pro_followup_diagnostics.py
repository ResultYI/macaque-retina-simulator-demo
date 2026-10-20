"""Actual small-bistratified workpoint and local B4 intake, without fitting.

One original-graph reconstruction restores its own saved state. Public rod
arrays are inspected in their original units; no clipping, offsets, timing
search, source-parameter tuning or new acquisition is performed.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.rgc_defaults import historical_project_default,historical_reference_file
from tools.run_rgc_type_candidate import (BASE,RunConfig,atlas_population_network,read_source_candidate,
    merge_parameter_package,HybridRunner,constant_background_channels,save,sha)
from tools.check_pro_background_closure import values,stats
from retina_sim.network import InputBundle
from retina_sim.contracts import canonical_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args();d=a.output.resolve();d.mkdir(parents=True,exist_ok=False)
    cfg=historical_project_default(ROOT);cid='small_bistratified_S_ON'
    recipe,path=read_source_candidate(ROOT,ROOT/'configurations'/f'{cid}_literature_candidate_v1.json')
    original,_,audit=atlas_population_network(ROOT,apply_rgc_output_defaults=False,duration=.6,light_context=cfg.light_context(),execution_profile=cfg.execution_profile,
        configuration_path=cfg.circuit_configuration_path,owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,
        max_spatial_cells=cfg.max_spatial_cells,seed=cfg.seed,dynamics_candidate_path=ROOT/recipe['base_dynamics_candidate']['path'])
    compiled=merge_parameter_package(original,audit['registry'],recipe,applicability=recipe['applicability'],allowed_fields=['alpha','mu','beta'],proposal_path=path)
    runner=HybridRunner(compiled,compute_backend='python');own=BASE/'pro_b0_b2_v1'/cid/'matched_gray/final_checkpoint.json';cp=own.read_text(encoding='utf8');state=runner.restore(cp)
    assert runner.checkpoint(state)==cp
    bg=constant_background_channels(compiled,cfg.light_context(),.7,interpolation='zero_order_hold');bundle=InputBundle({k:v.at(state.time_seconds) for k,v in bg.items()})
    incoming,out=runner.model.evaluate(state.time_seconds,runner._flat(state),bundle)
    branches={};excit=[]
    for suffix,pre in [('excitatory_S_ON_BC','S_ON_BC'),('excitatory_diffuse_BC_OFF','diffuse_BC_OFF')]:
        adapter=cid+'_'+suffix;q=out[(adapter,'conductance')].values;excit.append(q)
        link=next(l for l in compiled.spec.links if l.target_component==adapter and l.target_port=='activation')
        assert link.source_component==pre and link.source_port=='activation'
        branches[adapter]=dict(presynaptic_component=pre,presynaptic_port=link.source_port,
            adapter_gain_nS=stats(values(compiled.components[adapter])['gain']*1e9),actual_conductance_nS=stats(q*1e9),
            connection_identity=canonical_hash(link.connections),port_cell_ids=list(compiled.components[adapter].outputs['conductance'].cell_ids))
    np.testing.assert_array_equal(excit[0]+excit[1],incoming[cid]['excitation'].values)
    c=compiled.components[cid];p0=values(c);V=runner.model.local_blocks(cid,runner._flat(state))[0]
    ge=incoming[cid]['excitation'].values;gi=incoming[cid]['inhibition'].values;G=p0['gL']+ge+gi
    ve=(p0['gL']*p0['EL']+ge*p0['EE']+gi*p0['EI'])/G
    work=dict(status='ACTUAL_DISTINCT_BRANCH_SUM_CONFIRMED',neural_seconds=0.,assembly_hash=compiled.assembly_hash,
        own_checkpoint_sha256=sha(own),branches=branches,total_excitation_nS=stats(ge*1e9),inhibition_nS=stats(gi*1e9),
        actual_voltage_mV=stats(V*1000),instantaneous_passive_equilibrium_mV=stats(ve*1000),
        membrane_C_pF=stats(p0['C']*1e12),leak_g_nS=stats(p0['gL']*1e9),leak_EL_mV=stats(p0['EL']*1000),
        current_workpoint_zero_history_hazard_Hz=stats(c.native_model.intensity(V,np.zeros((3,len(V))))),
        voltage_sensitivity_per_excitatory_nS_mV=stats((p0['EE']-V)/G*1e-6),
        interpretation='Two distinct S-ON and LM-OFF routes each retain 1nS adapter scale. Their sum raises the passive workpoint. No duplicate-port bug established; no divide-by-two or gain/default change.',
        calibrated_or_fitted=False,proposed_parameter_change=None)
    save(d/'small_bistratified_workpoint.json',work)
    data=ROOT/'data/source/chen2024_dryad_v3_325663/Figure2-SourceData1.mat';identity=sha(data)
    assert identity=='6724e2edd8e8708456f32900a14efae88192a5ba597365046d0031eb543d1331'
    raw=loadmat(data,variable_names=['PrimateRodResponses'],simplify_cells=True)['PrimateRodResponses'];records=[]
    for j,row in enumerate(raw):
        u=np.asarray(row['AdaptingStepStim']);v=np.asarray(row['AdaptingStepResponse']);bad=np.flatnonzero(u<0)
        records.append(dict(array_label=f'PrimateRodResponses{{{j+1}}}',biological_cell_id_confirmed=False,fields=list(row),
            samples=len(u),recording_dt_seconds=.001,nominal_recording_seconds=len(u)*.001,all_finite=bool(np.isfinite(u).all() and np.isfinite(v).all()),
            stimulus_unit='R*/s',response_unit='current / estimated absolute dark current; dimensionless',
            stimulus_range=[float(u.min()),float(u.max())],response_range=[float(v.min()),float(v.max())],
            negative_stimulus_samples=len(bad),first_negative_sample_0based=int(bad[0]),last_negative_sample_0based=int(bad[-1]),
            first_negative_index_seconds=float(bad[0]*.001),negative_value_treatment='NONE: raw values retained',
            data_file_sha256=identity,fit_performed=False))
    provenance=json.loads((ROOT/'data/source_data/dacey2000_fig4B_digitized_v1/provenance.json').read_bytes())
    data_report=dict(status='LOCAL_DATA_INTAKE_COMPLETE_NO_ELIGIBLE_MATCHED_NEW_FIT',matlab_launches=0,new_downloads=0,fits_performed=0,
        rod_records=records,rod_blockers=[
            'Input labelled total R*/s contains substantive negative values; no permitted rectification/background offset established',
            'README gives 1ms rod recording; exact 0.1ms source-step replay/hold and alignment contract not supplied',
            'Three stimulus families are concatenated under two generic fields; exact family boundaries, prehistory and independent group split not supplied'],
        rod_next_action='Resolve negative-input semantics, family boundaries and clock/prehistory contract; then fixed-source replay before shared sigma/eta/KGC and per-record gamma candidates',
        dacey=dict(available=True,source_sha256=provenance['source']['sha256'],context=provenance['experimental_context'],
            blockers=['Source eccentricity13.6mm and150-1200um annulus differs from current0.35mm small ROI',
                      'Panel4B seconds/frequency unconfirmed; phase digitization cannot identify tau',
                      'Effective cone drive from source annulus and absolute V not known; do not fit arbitrary amplitude scale'],
            next_action='Use only permitted relative phase/polarity comparison until stimulus/clock/input-domain mapping is resolved'),
        S_cone=dict(matched_binary_data_available=False,next_action='Provide paired S and LM flash/filter records with retina/eccentricity/background/clock and holdout grouping'),
        glycine=dict(matched_raw_events_available=False,next_action='Provide waveform arrays, holding potentials, acquisition filter/sample rate and cell groups before rise/decay fitting'),
        RGC=dict(new_matched_voltage_spike_available=False,next_action='Use existing V-spike intake handoff; Cell_6 remains development, no reopened fit'),
        workbook_rows_requested=[
            'Rod fit/input task: any explicit negative-stimulus semantics, concatenated family sample boundaries, 1ms-to0.1ms replay and initial-history rules',
            'Dacey Fig4B task/source: any verified panel-specific frequency, absolute/background-to-input mapping and region transfer rule',
            'Any numeric acceptance row defining sufficient recovery and single-cell center/surround functional criterion'],
        workbook_import_retried=False)
    save(d/'B4_data_readiness.json',data_report)
    print(json.dumps(dict(small_bistratified=work,rod_intake=records,B4_status=data_report['status'])),flush=True)


if __name__=='__main__':main()
