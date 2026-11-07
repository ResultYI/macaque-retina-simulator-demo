"""Bounded BC geometry audit and one 50ms independent-cell full-chain forward."""
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import time
import traceback

import numpy as np

from retina_sim.assembly import RetinaBuilder
from retina_sim.bc_independent import DEFAULT_CONFIGURATION, temporal_profile
from retina_sim.bc_current import current_bc
from retina_sim.bc_pedicle_density import RegisteredPedicleField, anchor_rows, cone_bc_densities
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.group_inner_dynamics import BC_MECH
from retina_sim.cone_hc_eccentric import EccentricHCFullChainRunner
from retina_sim.h1_h2_effective import matrix_id
from retina_sim.hc_eccentric import digest
from retina_sim.spatial_group_model import BC, RGC
from tools.accept_bc_physiology import dc_check, protected, write
from tools.validate_h1_h2_effective import tick
from tools.run_hc_minimal_real_diagnostic_v1 import peak_memory_bytes


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/BC_PHYSIOLOGY_IMPLEMENTATION_2026-11-02'


def spatial_audit(model):
    selected = current_bc(ROOT)
    cfg = json.loads((ROOT/selected['bindings']['parameters']['path']).read_bytes())
    cone = FrozenDiskTable(ROOT, cfg['owner_support']['cone_source_table'])
    field = RegisteredPedicleField(cone,cfg)
    owners = field.owner_rows
    ids = cone.rows(owners).global_cell_ids
    quotas = anchor_rows(ids,field.labels[owners],np.arange(len(owners)))
    canonical = {n:(owners[rows],slots) for n,(rows,slots) in quotas.items()}
    counts = {n:len(rows) for n,(rows,_) in canonical.items()}
    for name,table in model.assembly.tables.items():
        pairs = set(zip(*canonical[name]))
        if not all(pair in pairs for pair in zip(table.extras['cone_row'],table.extras['bc_anchor_slot'])):
            raise ContractError('Existing local BC identity disagrees with full-parent quota')
    regions = {}
    for label,e in (('central',.5),('peripheral',5.75)):
        ref = owners[np.argmin(np.linalg.norm(field.rf[owners]-[e*.223,0.],axis=1))]
        center = field.xy[ref]
        radii = np.linalg.norm(field.xy[owners]-center,axis=1)
        in_window = owners[radii<=field.window]
        complete_owners = np.array_equal(np.sort(in_window),np.sort(np.flatnonzero(np.linalg.norm(field.xy-center,axis=1)<=field.window)))
        if not complete_owners:
            raise ContractError('Representative density window crosses the output-owner boundary')
        lm,s = field.density(center,'LM'),field.density(center,'S')
        density = cone_bc_densities(lm,s)
        rows = {}
        for name,(anchors,slots) in canonical.items():
            take = np.flatnonzero(np.linalg.norm(field.xy[anchors]-center,axis=1)<=field.window)
            anchors = anchors[take]
            if not len(anchors): raise ContractError('Representative patch has no '+name)
            degrees, actual_radii, support_radii, external = [], [], [], []
            for owner in anchors:
                if name.startswith('midget'):
                    inputs = np.array([owner]); support = 0.
                else:
                    inputs, audit = field.contacts(field.xy[owner],name)
                    if not audit['status'].startswith('READY'):
                        raise ContractError('Representative BC contact support incomplete')
                    support = audit['radius_mm']
                allowed = ['S'] if name == 'S_ON_BC' else ['L','M','S'] if name == 'midget_BC_OFF' else ['L','M']
                if not np.all(np.isin(field.labels[inputs],allowed)):
                    raise ContractError('BC Cone selectivity changed')
                distances = np.linalg.norm(field.xy[inputs]-field.xy[owner],axis=1)
                radius = float(distances.max(initial=0.))
                if radius>support+1e-12: raise ContractError('BC contacts exceed retained support')
                degrees.append(len(inputs));actual_radii.append(radius);support_radii.append(support)
                external.append(int(np.sum(np.linalg.norm(field.rf[inputs],axis=1)>1.338)))
            ecc = np.linalg.norm(field.rf[anchors],axis=1)/.223
            profile = temporal_profile(model.config,name,ecc)
            rows[name] = dict(cells=len(anchors),density_cells_mm2=len(anchors)/(np.pi*field.window**2),
                local_expected_anchor_density_cells_mm2=float(density[name]),eccentricity_deg=[float(ecc.min()),float(ecc.max())],
                degree_min=int(min(degrees)),degree_max=int(max(degrees)),degree_mean=float(np.mean(degrees)),
                actual_radius_mm=[min(actual_radii),max(actual_radii)],support_radius_mm=[min(support_radii),max(support_radii)],
                profile_id=profile['profile_id'],private_owner_preserved=name.startswith('midget'),
                source_cones_outside_6deg=sum(external),type_selectivity_passed=True)
        regions[label] = dict(requested_eccentricity_deg=e,anchor_center_OPL_mm=center,
            actual_reference_eccentricity_deg=float(np.linalg.norm(field.rf[ref])/.223),window_radius_mm=field.window,
            complete_window=True,density_plane='OPL functional input anchors; not BC somata',types=rows)
    # The outer output annulus is deliberately queried against the complete
    # existing pedicle source, never against an RF<=6 degree presynaptic crop.
    boundary = {}
    for name in ('diffuse_BC_ON','diffuse_BC_OFF','S_ON_BC'):
        anchors = canonical[name][0]
        rf = field.rf[anchors]
        ecc = np.linalg.norm(rf,axis=1)/.223
        anchors = anchors[(ecc>=5.95)&(np.abs(rf[:,1])<.04)&(rf[:,0]>0)]
        edges,affected,max_e = 0,0,0.
        for owner in anchors:
            inputs,audit = field.contacts(field.xy[owner],name)
            if not audit['status'].startswith('READY'): raise ContractError('Outer BC ancestry incomplete')
            ie = np.linalg.norm(field.rf[inputs],axis=1)/.223
            outside = int(np.sum(ie>6.));edges+=outside;affected+=int(outside>0)
            max_e=max(max_e,float(ie.max(initial=0.)))
        boundary[name] = dict(output_cells=len(anchors),cells_with_external_inputs=affected,
            retained_external_cone_edges=edges,maximum_input_eccentricity_deg=max_e,
            policy='existing finite source and held exterior; no silent 6deg input truncation')
    if not sum(v['retained_external_cone_edges'] for v in boundary.values()):
        raise ContractError('Boundary probe did not exercise external ancestors')
    return dict(current_density_identity=selected['identity'],canonical_counts=counts,
        complete_owner_cones=len(owners),existing_local_quota_identity_verified=True,
        regions=regions,boundary=boundary,cone_geometry_regenerated=False,
        scope='two spatial probes on existing full parent; dynamics use the already frozen local network')


def static_runtime_checks(model,arm):
    inner=arm.inner;state=arm.inner_state
    ledger=model.seed['bc_physiology']['ledger']
    inner.parameters=dict(inner.parameters,provenance=dict(inner.parameters['provenance'],
        bc_targets_nS={n:r['effective_conductance_nS'] for n,r in ledger.items()}))
    dc=dc_check(inner,state)
    rhs=inner.rhs(0.,state)
    residual={n:{k:float(abs(rhs[n,BC_MECH,k]).max(initial=0.)) for k in ('V','terminal_release','A')} for n in BC}
    if max(v['V'] for v in residual.values())>1e-8: raise ContractError('Independent BC background residual')
    # A changed cell is independently integrated, even if its Cone input remains
    # in a shared upstream group. No copied group trajectory can pass this check.
    changed={k:v.copy() for k,v in state.items()}
    pop='midget_BC_ON';changed[pop,BC_MECH,'V'][0]+=.0001
    delta=inner.rhs(0.,changed)[pop,BC_MECH,'V']-rhs[pop,BC_MECH,'V']
    if delta[0]==0 or np.count_nonzero(delta)!=1: raise ContractError('BC states were merged')
    for n in BC:
        layout=model.assembly.populations[n]
        np.testing.assert_array_equal(layout.counts,1)
        np.testing.assert_array_equal(layout.fine_to_group[layout.member_global_rows],np.arange(layout.group_count))
        for field in model.assembly.recording_fields:
            if field.population==n and (field.identity!='independently_solved_cell'
                    or len(field.rows)!=layout.group_count):
                raise ContractError('BC readout identity is not its actual independent state layout')
    aii={}
    changed={k:v.copy() for k,v in state.items()}
    changed['AII_AC','ReceptorConductanceMembrane','V']+=.0001
    gap=inner.rhs(0.,changed)
    for n in ('midget_BC_ON','diffuse_BC_ON','S_ON_BC'):
        aii[n]=float(np.max(abs(gap[n,BC_MECH,'V']-rhs[n,BC_MECH,'V'])))
        if aii[n]==0: raise ContractError('AII-to-ON BC gap effect lost')
    for n,c in (('midget_BC_OFF','AII_FMB_glycine'),('diffuse_BC_OFF','AII_DB1_glycine')):
        changed={k:v.copy() for k,v in state.items()};changed['AII_AC',c,'b']*=1.001
        d=inner.rhs(0.,changed)[n,BC_MECH,'V']-rhs[n,BC_MECH,'V']
        if np.max(abs(d))==0: raise ContractError('AII-to-OFF BC glycine effect lost')
        aii[n]=float(np.max(abs(d)))
    native=arm.native_physics;up=model.upstream.native
    for attr in ('A','B'):
        if matrix_id(getattr(native,attr))!=matrix_id(getattr(up,attr)):
            raise ContractError('BC refinement changed accepted upstream '+attr)
    if (matrix_id(native.Q['H2'])!=matrix_id(up.Q['H2'])
            or dict(native.effective_h1_parameters)!=dict(up.effective_h1_parameters)
            or dict(native.parameters)!=dict(up.parameters)):
        raise ContractError('Accepted H1/H2 parameters changed')
    return dict(DC_conductance=dc,background_BC_rhs=residual,
        independently_perturbed_cell_changes_only_its_BC_derivative=True,
        AII_probe_delta_rhs=aii,snapshot_identity_metadata_passed=True,H1_H2_operators_parameters_preserved=True)


def forward(model,arm):
    rows=[arm.snapshot(0.,'right',queue_node=True)];restored=None;clock=time.perf_counter()
    arm.initial_commit();arm.ack(0.)
    checks=0;max_restore=0.
    try:
        for i in range(1,501):
            left,right=tick(arm,i*.0001,i)
            arm.inner.validate(arm.inner_state)
            if any(not np.isfinite(v).all() for s in ('states','rhs','ports') for v in right[s].values()):
                raise ContractError('Nonfinite independent BC full chain')
            if i%10==0:
                for n in BC:
                    drive=arm.inner.mean['cones_to_'+n] @ left['ports']['cones','release']
                    np.testing.assert_array_equal(drive,left['ports'][n,'drive'])
                    h=np.vstack([right['states'][n,BC_MECH,'history_'+str(k)] for k in range(3)])
                    sign=1 if n.endswith('_OFF') else -1
                    expected=np.logaddexp(0.,arm.inner.par(n,'bias')+sign*(np.asarray(arm.inner.p[n]['kernel'])@h))
                    np.testing.assert_allclose(expected,right['ports'][n,'generator_activation'],rtol=0,atol=1e-15)
                    checks+=1
                rows.append(right)
            if i==490:
                payload=arm.checkpoint_payload()
                restored=EccentricHCFullChainRunner.from_checkpoint(model.assembly,model.seed,model.protocol,payload)
                bad=deepcopy(payload);bad['controller']['BC_parameter_hash']='old-group-parameter-identity'
                try:EccentricHCFullChainRunner.from_checkpoint(model.assembly,model.seed,model.protocol,bad)
                except ContractError:pass
                else:raise ContractError('Mismatched BC checkpoint accepted')
            elif i>490:
                _,other=tick(restored,i*.0001,i)
                for s in ('states','rhs','ports'):
                    for k,v in right[s].items():max_restore=max(max_restore,float(np.max(abs(v-other[s][k]),initial=0.)))
                if canonical_hash(arm.spike_states)!=canonical_hash(restored.spike_states):
                    raise ContractError('RGC histories/RNG changed on restore')
            if i%100==0:print(f'BC full chain {i/10:.0f} ms, wall {time.perf_counter()-clock:.1f}s',flush=True)
        if max_restore!=0:raise ContractError('Independent BC checkpoint continuation differs')
        data={'time_seconds':np.array([r['time_seconds'] for r in rows])}
        for section in ('states','ports'):
            for key in rows[-1][section]:data[section+'__'+'__'.join(key)]=np.column_stack([r[section][key] for r in rows])
        responses={}
        for n in BC:
            delta={k:float(np.max(abs(data['ports__'+n+'__'+k]-data['ports__'+n+'__'+k][:,[0]])))
                   for k in ('drive','generator_activation','voltage','unadapted_release','adaptation_activity','effective_release')}
            if not all(v>0 for v in delta.values()):raise ContractError('Independent BC state did not advance: '+n)
            g=model.assembly.bc_cells[n]['original_group_rows']
            voltage=data['ports__'+n+'__voltage']
            differences=[float(np.max(np.ptp(voltage[g==i],axis=0))) for i in np.unique(g) if np.sum(g==i)>1]
            responses[n]=dict(maximum_absolute_change=delta,maximum_within_old_group_voltage_difference_V=max(differences,default=0.),
                independent_cells=len(g))
        required=[('cones','release'),('HC','h1'),('HC','h2'),('rods','voltage'),('rod_BC','voltage'),('AII_AC','voltage')]
        required += [(n,k) for n in RGC for k in ('voltage','gE','gI','expected_rate_Hz')]
        if any(k not in rows[-1]['ports'] for k in required):raise ContractError('Full-chain layer readout missing')
        return dict(status='CONDITIONAL_PASS',duration_seconds=.05,wall_seconds=time.perf_counter()-clock,
            BC_drive_and_typed_activation_checks=checks,checkpoint_restore_max_abs_error=max_restore,
            old_parameter_checkpoint_rejected=True,RGC_history_RNG_equal=True,responses=responses,
            full_chain_ports_present=True,BC_input='current accepted Cone release; no second HC kernel',
            boundary='accepted finite held exterior; unqualified whole-retina convergence',
            port_units={':'.join(k):v for k,v in rows[-1]['port_units'].items()}),data
    finally:
        if restored is not None:restored.close()


def plot(data,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,3,figsize=(13,12),sharex=True)
    for i,n in enumerate(BC):
        for j,(field,scale,label) in enumerate((('voltage',1e6,'Delta V (uV)'),('effective_release',1.,'Delta effective release'),('adaptation_activity',1.,'Delta adaptation A'))):
            values=data['ports__'+n+'__'+field];values=(values-values[:,[0]])*scale
            lo,hi=np.percentile(values,[10,90],axis=0)
            ax=axes[i,j];ax.fill_between(data['time_seconds']*1000,lo,hi,alpha=.2,label='cell 10-90%')
            ax.plot(data['time_seconds']*1000,values.mean(0),label='cell mean')
            ax.set_ylabel(label);ax.set_title(n,fontsize=9);ax.grid(alpha=.2)
            if i==4:ax.set_xlabel('Time (ms)')
    axes[0,0].legend(fontsize=8);fig.suptitle('Independent BC states | engineering full-chain response, not observed physiology')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--configuration',default=DEFAULT_CONFIGURATION)
    parser.add_argument('--resume-interface-repair',action='store_true')
    args=parser.parse_args()
    result_path=REPORT/'CALIBRATION_METRICS.json'
    previous=None
    if result_path.exists():
        previous=json.loads(result_path.read_bytes())
        if (not args.resume_interface_repair or previous.get('status')!='FAILED_RETAINED'
                or 'fullchain' in previous):
            raise ContractError('Current BC metrics already exist; completed forward must not be repeated')
    start=time.perf_counter();result=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat())
    if previous:
        result['prior_attempts']=previous.get('prior_attempts',[])+[{k:v for k,v in previous.items() if k!='prior_attempts'}]
    before=protected(ROOT)
    before.update({str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'retina_sim').glob('*hc*.py')})
    arm=None
    try:
        model=RetinaBuilder.build_research_bc(ROOT,args.configuration)
        result.update(model_identity=model.identity,upstream_identity=model.upstream.identity,build_seconds=time.perf_counter()-start)
        print('Independent BC model built',flush=True)
        if previous and previous.get('spatial',{}).get('current_density_identity')==current_bc(ROOT)['identity']:
            result['spatial']=previous['spatial'];result['spatial_checks_reused_from']=previous['started_utc']
        else:result['spatial']=spatial_audit(model)
        print('Central/peripheral geometry passed',flush=True)
        arm=model.runner();result['static']=static_runtime_checks(model,arm)
        result['fullchain'],data=forward(model,arm)
        np.savez_compressed(REPORT/'BC_FULLCHAIN_STATES.npz',**data)
        arrays={n+'__'+k:v for n,row in model.assembly.bc_cells.items() for k,v in row.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(REPORT/'BC_CELL_IDENTITIES.npz',**arrays)
        write(REPORT/'BC_PARAMETER_LEDGER.json',dict(configuration=args.configuration,configuration_sha256=digest(ROOT/args.configuration),
            model_identity=model.identity,upstream_identity=model.upstream.identity,temporal_profiles=model.temporal_bindings,
            membrane_binding=model.seed['bc_physiology'],route_projections=model.assembly.projection_ledger,
            actual_parameters={n:model.seed['inner_parameters']['populations'][n] for n in BC},
            identity_arrays='BC_CELL_IDENTITIES.npz',state_arrays='BC_FULLCHAIN_STATES.npz',
            source_targets='data/source_data/bc_physiology_targets.json',
            grouped_upstream_retained=True,independent_BC=True,physiological_dynamic_calibration=False))
        plot(data,REPORT/'BC_FULLCHAIN_RESPONSES.png')
        after={p:digest(ROOT/p) for p in before}
        if before!=after:raise ContractError('Protected existing source changed during acceptance')
        result.update(status='BC_ENGINEERING_ACCEPTANCE_PASS',protected_sources=after,
            calibration=dict(allowed_optimization_parameters=[],optimization_runs=0,
                baseline_metric=None,candidate_metric=None,new_temporal_parameters_accepted=False,
                decision='RETAIN_BASELINE_UNIDENTIFIED_MATCHED_LIGHT_RESPONSE',
                explanation='Published peripheral puff receptor currents, current-clamp channel responses and RF fits do not identify a central effective light-to-release FIR; no fabricated waveform or fit',
                membrane='Existing approved local conditional Rin constraints reapplied only to refined actual gap/glycine geometry',
                S_ON='Retain engineering membrane and temporal values'),
            FIT_READY=False,BC_engineering_freezable=True,whole_retina_physiology_qualified=False)
    except Exception as error:
        result.update(status='FAILED_RETAINED',error=str(error),traceback=traceback.format_exc());raise
    finally:
        if arm is not None:arm.close()
        result.update(wall_seconds=time.perf_counter()-start,peak_working_set_bytes=peak_memory_bytes(),ended_utc=datetime.now(timezone.utc).isoformat())
        write(result_path,result)
    print(result['status'],result['wall_seconds'],flush=True)


if __name__=='__main__':main()
