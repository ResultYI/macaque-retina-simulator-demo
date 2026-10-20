"""B3 checks on an existing accepted complete-network background; no advance.

Analytic local fixed points test the declared resource/chemical equations.
Residuals of the actual coupled state are reported separately. This is neither
a new physiological validation nor permission to change reference anchors.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.rgc_defaults import historical_project_default,historical_reference_file
from tools.run_rgc_type_candidate import RunConfig,atlas_population_network,HybridRunner,constant_background_channels,save,sha
from retina_sim.network import InputBundle
from retina_sim.vh2005 import SOURCE_VALUES
from retina_sim.rod_pathway import rod_source_values,phototransduction_equilibrium


def values(component):return {r.parameter_id:np.asarray(r.value.values) for r in component.parameters.records if hasattr(r.value,'values')}
def stats(v):
    v=np.asarray(v);return dict(mean=float(v.mean()),minimum=float(v.min()),maximum=float(v.max()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    d=args.output.resolve();d.mkdir(parents=True,exist_ok=False)
    previous=ROOT/'reports/modeling_scope_20261020/actual_video_halfsecond_v1'
    cfg=historical_project_default(ROOT)
    compiled,_,audit=atlas_population_network(ROOT,apply_rgc_output_defaults=False,duration=.6,light_context=cfg.light_context(),
        execution_profile=cfg.execution_profile,configuration_path=cfg.circuit_configuration_path,
        owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,
        seed=cfg.seed,dynamics_candidate_path=ROOT/'configurations/bc_ac_contact_first_hypothesis_v1.json')
    actual_cfg=json.loads((previous/'config.json').read_bytes())
    actual=RunConfig(**{k:v for k,v in actual_cfg.items() if k in RunConfig.__dataclass_fields__})
    runner=HybridRunner(compiled,compute_backend='python');cp=(previous/'background_checkpoint.json').read_text(encoding='utf8')
    state=runner.restore(cp)
    assert runner.checkpoint(state)==cp
    bg=constant_background_channels(compiled,actual.light_context(),state.time_seconds+.1,interpolation='zero_order_hold')
    bundle=InputBundle({k:v.at(state.time_seconds) for k,v in bg.items()})
    y=runner._flat(state);incoming,out=runner.model.evaluate(state.time_seconds,y,bundle)
    report=dict(status='READ_ONLY_B3_ACCEPTED_BACKGROUND_CHECKS',neural_seconds=0.,matlab_launches=0,
        checkpoint_sha256=sha(previous/'background_checkpoint.json'),assembly_hash=compiled.assembly_hash,
        exact_own_checkpoint_restore=True,model_time_seconds=state.time_seconds,
        acceptance=json.loads((previous/'background_initialization.json').read_bytes())['claim'],
        physiological_validation=False,source_groups_unchanged=True,resources={},chemical={},gap={},
        original_background_context='Actual saved ISET per-cone td vector; independent retained rod 5R*/s',
        current_units='pA converted once to A; event rates in events/s, never event/tick')
    assert dict(compiled.components['outer_native'].native_model.p)==SOURCE_VALUES
    rod=compiled.components['rod'].native_model
    assert dict(rod.p)==rod_source_values(ROOT)
    rod_blocks=runner.model.local_blocks('rod',y)
    rod_target=phototransduction_equilibrium(rod,bundle.channels['rod_light'].values)
    report['rod_source_fixed_point_errors']={s.state_name:float(np.max(abs(a-b.values))) for s,a,b in zip(rod.state_specs,rod_blocks,rod_target)}
    membrane=compiled.components['rod_membrane'];rp=values(membrane);rb=runner.model.local_blocks('rod_membrane',y)
    current=incoming['rod_membrane']['current'].values
    np.testing.assert_array_equal(current,out[('rod','current')].values*1e-12)
    dv=membrane.derivative(state.time_seconds,rb,incoming['rod_membrane'])[0].values
    expected_current=rp['gL']*(rp['EL']-rb[0])-(current-rp['current_anchor'])+incoming['rod_membrane']['__gap_current__'].values
    np.testing.assert_allclose(rp['C']*dv,expected_current,rtol=0,atol=1e-26)
    Idark=-rod.p['k']*rod.p['gdark']**rod.p['n']*1e-12
    fixed_dark_anchor=Idark-rp['gL']*(rp['EL']-(-.04))
    current_veq=rp['EL']-(current-rp['current_anchor'])/rp['gL']
    proposed_veq=rp['EL']-(current-fixed_dark_anchor)/rp['gL']
    proposed_release=expit((proposed_veq-rp['release_half'])/rp['release_slope'])
    report['rod_anchor_reference']=dict(status='REFERENCE_CONTRACT_DIFFERENCE_REQUIRES_DECISION',
        current_anchor_A=stats(rp['current_anchor']),proposed_fixed_dark_anchor_A=stats(fixed_dark_anchor),
        dark_current_pA=Idark*1e12,actual_background_current_pA=stats(current*1e12),
        current_no_gap_equilibrium_mV=stats(current_veq*1000),
        proposed_fixed_dark_no_gap_equilibrium_mV=stats(proposed_veq*1000),
        actual_release=stats(rb[1]),proposed_passive_release_at_equilibrium=stats(proposed_release),
        current_implementation='atlas_primary_rod.py binds current_anchor=current_eq(B)+gL*(V0-EL) at assembly',
        approved_change_performed=False,
        effect='Changes rod DC/reference and all downstream TRPM1/RBC/AII workpoints; proposed numbers are algebra, not a new coupled run')
    for cid in ('rod_BC','AII_AC'):
        c=compiled.components[cid];p0=values(c);b=runner.model.local_blocks(cid,y)
        attempts=p0['rmax']*expit((b[0]-p0['r_half'])/p0['r_slope'])
        xe=1/(1+p0['tau_recovery']*p0['U']*attempts)
        rate=p0['Npool']*p0['U']*attempts*b[2]
        np.testing.assert_array_equal(rate,out[(cid,'event_rate')].values)
        analytic=(b[0],b[1],xe)
        resource_rhs=c.derivative(state.time_seconds,analytic,incoming[cid])[2].values
        np.testing.assert_allclose(resource_rhs,0.,rtol=0,atol=4e-14)
        report['resources'][cid]=dict(actual_x=stats(b[2]),equilibrium_x_at_actual_V=stats(xe),
            maximum_actual_x_equilibrium_error=float(abs(b[2]-xe).max()),
            actual_expected_event_rate_per_s=stats(rate),
            maximum_local_analytic_resource_rhs_abs=float(abs(resource_rhs).max()),
            no_resource_one_or_release_half_initialization_assumed=True)
    for cid in ('RBC_AII_AMPA','AII_FMB_glycine','AII_DB1_glycine'):
        c=compiled.components[cid];p0=values(c);b=runner.model.local_blocks(cid,y);i=incoming[cid]
        nu=i['event_rate'].values;tr=p0['tau_rise'];td=p0['tau_decay']
        if 'postsynaptic_voltage' in i:
            td=p0['decay_minus100']+(p0['decay_0']-p0['decay_minus100'])*np.clip((i['postsynaptic_voltage'].values+.1)/.1,0,1)
        peak_t=np.log(td/tr)/(1/tr-1/td);Z=np.exp(-peak_t/td)-np.exp(-peak_t/tr)
        ae=nu*tr;be=nu*td;target=p0['ncontact']*p0['g_peak']*nu*(td-tr)/Z
        actual_g=out[(cid,'conductance')].values
        np.testing.assert_allclose(c.observe('conductance',state.time_seconds,(ae,be),i).values,target,rtol=2e-15,atol=1e-24)
        residual=c.derivative(state.time_seconds,(ae,be),i)
        assert max(float(abs(q.values).max()) for q in residual)<2e-12
        report['chemical'][cid]=dict(actual_event_rate_per_s=stats(nu),actual_conductance_nS=stats(actual_g*1e9),
            fixed_point_conductance_nS_at_actual_V_and_rate=stats(target*1e9),
            maximum_actual_conductance_fixed_point_error_nS=float(abs(actual_g-target).max()*1e9),
            maximum_actual_a_fixed_point_error=float(abs(b[0]-ae).max()),maximum_actual_b_fixed_point_error=float(abs(b[1]-be).max()),
            normalization='per-event peak normalized; ncontact applied once after a/b gates',
            voltage_decay='held-V interpolation is a project approximation' if 'postsynaptic_voltage' in i else 'fixed source/engineering tau')
    total_current=0.;power=0.;edge_power=0.;max_pair_error=0.
    for l,r,ix,iy,G in runner.model._gap_plans:
        vl=out[(l,'voltage')].values[ix];vr=out[(r,'voltage')].values[iy]
        I=G*(vr-vl);total_current+=float(I.sum()+(-I).sum());edge_power-=float(np.sum(G*(vr-vl)**2))
        max_pair_error=max(max_pair_error,float(np.max(abs(I+(-I)))))
    for cid,c in compiled.components.items():
        if (cid,'gap_current') in out:
            power+=float(np.sum(out[(cid,'voltage')].values*out[(cid,'gap_current')].values))
    np.testing.assert_allclose(power,edge_power,rtol=1e-13,atol=1e-24)
    report['gap']=dict(shared_undirected_conductance=True,total_pair_current_A=total_current,
        maximum_pair_antisymmetry_error_A=max_pair_error,voltage_times_current_power_W=power,
        negative_edge_dissipation_W=edge_power,power_identity_error_W=power-edge_power)
    report['actual_rhs_maxima']={}
    for cid in ('rod_membrane','rod_BC','AII_AC','RBC_AII_AMPA','AII_FMB_glycine','AII_DB1_glycine'):
        c=compiled.components[cid];deriv=c.derivative(state.time_seconds,runner.model.local_blocks(cid,y),incoming[cid])
        report['actual_rhs_maxima'][cid]={s.state_name:dict(maximum_abs=float(abs(q.values).max()),unit=q.unit)
            for s,q in zip(c.state_specs,deriv)}
    save(d/'result.json',report)
    save(d/'rod_anchor_proposal_not_applied.json',report['rod_anchor_reference'])
    print(json.dumps(dict(status=report['status'],rod_anchor=report['rod_anchor_reference'],resources=report['resources'],gap=report['gap'])),flush=True)


if __name__=='__main__':main()
