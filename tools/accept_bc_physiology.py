"""One source-bound local BC acceptance, with two independent luminance steps."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time
import traceback

import numpy as np
from scipy import sparse

from retina_sim.assembly import RetinaBuilder
from retina_sim.bc_physiology import BCPhysiologyRunner
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.disk_graph_storage import FrozenSparseRoute
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.group_inner_dynamics import BC_MECH, MEMBRANE
from retina_sim.group_parameter_binding import bc_glycine_dc_conductance
from retina_sim.group_time_storage import save_bundle, load_bundle
from retina_sim.h1_h2_effective import matrix_id
from retina_sim.spatial_group_model import BC, RGC
from tools.gin_hc2_bc1_context_support_v2 import protocol, background_directory
from tools.run_hc_minimal_real_diagnostic_v1 import peak_memory_bytes
from tools.validate_h1_h2_effective import tick


def write(path, value):
    def serial(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=serial), encoding='utf-8')


def protected(root):
    paths = [root/'configurations/retina_platform_v1.json', root/'configurations/effective_dynamics_v1.json',
        root/'configurations/research/BC_FUNCTIONAL_V1_candidate.json',
        root/'retina_sim/hc_biexponential.py', root/'retina_sim/hc_lateral_candidate.py',
        root/'retina_sim/ac_vs_model.py', root/'retina_sim/h1_h2_background.py',
        root/'reports/CELL6_RAW_REBUILD_CONDITIONAL_READOUT_V1/runtime_monitor_repair_v1/readout/FROZEN_IDENTITY.json']
    paths += list((background_directory()/'seed').rglob('*'))
    paths += list((root/'configurations/research').glob('*cone_hc*.json'))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def connectivity(root, assembly, out):
    cat = json.loads((root/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/BC_catalog.json').read_bytes())
    cones = assembly.populations['cones']
    table = FrozenDiskTable(root, assembly.manifest['populations']['cones']['source_table'])
    labels = table.extras['subtype'].astype('U1')
    result = {}; saved = {}
    for name in BC:
        pop = assembly.populations[name]
        if cat['tables'][name] != assembly.manifest['populations'][name]['source_table']:
            raise ContractError('BC member table differs from approved Builder')
        rows, w, _, _ = FrozenSparseRoute(root, cat['routes']['cones->'+name]).selected_rows(pop.member_global_rows)
        coo = w.tocoo(); pre = cones.fine_to_group[coo.col]; post = pop.fine_to_group[rows[coo.row]]
        if np.any(pre < 0) or np.any(post < 0):
            raise ContractError('Actual fine Cone/BC members are absent')
        types = labels[coo.col]
        allowed = {'S'} if name == 'S_ON_BC' else {'L','M','S'} if name == 'midget_BC_OFF' else {'L','M'}
        if not set(types) <= allowed or np.any(np.diff(w.indptr) == 0):
            raise ContractError('BC types or direct contact coverage differ')
        if name.startswith('midget') and np.any(np.diff(w.indptr) != 1):
            raise ContractError('Approved private parent was changed')
        total = sparse.csr_matrix((coo.data,(post,pre)),shape=(pop.group_count,cones.group_count))
        mean = (sparse.diags(1/pop.counts)@total).tocsr()
        actual = assembly.matrix('routes/cones_to_'+name+'/mean_weights.npz')
        delta = mean-actual
        error = float(np.max(abs(delta.data),initial=0.))
        if error > 1e-12:
            raise ContractError('Fine-member projection differs; no centroid replacement allowed')
        np.testing.assert_array_equal(np.diff(pop.member_offsets), pop.counts)
        np.testing.assert_array_equal(pop.fine_to_group[pop.member_global_rows],np.repeat(np.arange(pop.group_count),pop.counts))
        xy = FrozenDiskTable(root, cat['tables'][name]).coordinates['rf'][rows]/.223
        result[name] = dict(groups=pop.group_count, fine_members=len(rows), fine_edges=w.nnz,
            group_edges=actual.nnz, input_cone_types={t:int(np.sum(types==t)) for t in ('L','M','S')},
            fine_degree_min=int(np.diff(w.indptr).min()), fine_degree_max=int(np.diff(w.indptr).max()),
            eccentricity_deg=[float(np.linalg.norm(xy,axis=1).min()),float(np.linalg.norm(xy,axis=1).max())],
            member_projection_max_abs_error=error, matrix_identity=matrix_id(actual), reused_without_regeneration=True)
        saved.update({name+'__'+k:v for k,v in dict(post_fine_rows=rows[coo.row],cone_fine_rows=coo.col,
            post_groups=post,cone_groups=pre,weight=coo.data,member_counts=pop.counts).items()})
    np.savez_compressed(out/'CONNECTION_MEMBERS.npz',**saved)
    write(out/'CONNECTION_STATISTICS.json',result)
    return result


def dc_check(model, state):
    result = {}
    for pop in BC:
        np.testing.assert_array_equal(model.C[pop],20e-12*model.members[pop])
        np.testing.assert_array_equal(model.gL[pop],model.par(pop,'gL')*model.members[pop])
        eps = 1e-7
        currents = []
        for sign in (-1,1):
            s = {k:v.copy() for k,v in state.items()}
            s[pop,BC_MECH,'V'] += sign*eps
            if pop.endswith('_OFF'):
                component = 'AII_FMB_glycine' if pop == 'midget_BC_OFF' else 'AII_DB1_glycine'
                c = model.contacts[component]; rate = model.outputs(0.,s)['AII_AC','event_rate'][c['pre']]
                rise,decay,_ = model._kinetics(s,component)
                s['AII_AC',component,'a'] = rate*rise
                s['AII_AC',component,'b'] = rate*decay
            currents.append(model.rhs(0.,s)[pop,BC_MECH,'V']*model.C[pop])
        measured = -(currents[1]-currents[0])/(2*eps)/model.members[pop]
        result[pop] = dict(finite_difference_DC_nS=[float(measured.min()*1e9),float(measured.max()*1e9)])
        if pop != 'S_ON_BC':
            target = model.parameters['provenance']['bc_targets_nS'][pop]
            np.testing.assert_allclose(measured,target*1e-9,rtol=2e-8,atol=1e-18)
    return result


def advance_tick(arm, index):
    t = arm.protocol['clock_origin_seconds']+index*.0001
    _, right = tick(arm, t, index)
    return right if index%10 == 0 else None


def condition(model, factor, name, out):
    seed = deepcopy(model.seed); p = protocol(); p['stop_seconds'] = .05
    bg = seed['input_reference']['native_background_td']
    seed['native_inputs'] = {i:bg*(factor if i>=2 else 1.) for i in p['input_frame_indices']}
    arm = BCPhysiologyRunner(model.assembly,seed,p); restored = None
    clock = time.perf_counter(); rows = []
    try:
        initial = arm.snapshot(0.,'right',queue_node=True); rows.append(initial)
        arm.initial_commit(); arm.ack(0.)
        for pop in BC:
            for field in ('history_0','history_1','history_2','activation'):
                np.testing.assert_array_equal(arm.inner_state[pop,BC_MECH,field],initial['states'][pop,BC_MECH,field])
        restore_error = 0.
        for index in range(1,501):
            row = advance_tick(arm,index)
            if row is not None:
                for section in ('states','rhs','ports'):
                    if any(not np.isfinite(v).all() for v in row[section].values()):
                        raise ContractError('Nonfinite real-source trajectory')
                rows.append(row)
            if index == 490:
                folder = out/(name+'_checkpoint')
                identity = dict(model=model.identity,execution=arm.identity)
                save_bundle(folder,kind='complete_model_checkpoint',identity=identity,payload=arm.checkpoint_payload(),
                    qualification=dict(default_adopted=False,research_only=True,physiological_calibration=False,fit_eligible=False))
                payload = load_bundle(folder,expected_kind='complete_model_checkpoint',expected_identity=identity)['payload']
                restored = BCPhysiologyRunner.from_checkpoint(model.assembly,seed,p,payload)
                bad = deepcopy(payload); bad['controller']['bc_parameter_binding'] = 'wrong-BC-parameters'
                try:
                    BCPhysiologyRunner.from_checkpoint(model.assembly,seed,p,bad)
                except ContractError:
                    pass
                else:
                    raise ContractError('Checkpoint accepted changed BC parameter identity')
            elif index > 490:
                other = advance_tick(restored,index)
                for k,v in arm.inner_state.items():
                    restore_error = max(restore_error,float(np.max(abs(v-restored.inner_state[k]),initial=0.)))
                if canonical_hash(arm.spike_states) != canonical_hash(restored.spike_states):
                    raise ContractError('RGC histories/RNG did not restore exactly')
                if row is not None:
                    for section in ('states','ports'):
                        for k,v in row[section].items():
                            restore_error = max(restore_error,float(np.max(abs(v-other[section][k]),initial=0.)))
            if index%100 == 0:
                print(name, index/10, 'ms', round(time.perf_counter()-clock,2),'s',flush=True)
        if restore_error != 0.:
            raise ContractError('Checkpoint continuation changed states')
        traces = {'time_seconds':np.array([r['time_seconds'] for r in rows])}
        for section in ('states','ports'):
            for key in rows[-1][section]:
                traces[section+'__'+'__'.join(key)] = np.column_stack([r[section][key] for r in rows])
        np.savez_compressed(out/(name+'_TRAJECTORY.npz'),**traces)
        summary = {}
        for pop in BC:
            delta = {field:float(np.mean(rows[-1]['ports'][pop,field]-initial['ports'][pop,field]))
                     for field in ('voltage','unadapted_release','effective_release','adaptation_activity','drive','generator_activation')}
            polarity = 1 if pop.endswith('_OFF') else -1
            # Polarity is relative to actual Cone release, not illuminance:
            # the retained H2 feedback can reverse S release during this pulse.
            if not (delta['generator_activation']*delta['drive']*polarity>0
                    and delta['voltage']*delta['drive']*polarity>0
                    and delta['effective_release']*delta['drive']*polarity>0):
                raise ContractError('BC response polarity differs: '+pop+str(delta))
            summary[pop] = delta
        last = rows[-1]['ports']
        interfaces = dict(AII_voltage_finite=bool(np.isfinite(last['AII_AC','voltage']).all()),
            glycine_total_S={c:float(np.sum(last[c,'conductance'])) for c in ('AII_FMB_glycine','AII_DB1_glycine')},
            BC_gap_current_A={n:float(np.sum(last[n,'gap_current'])) for n in ('midget_BC_ON','diffuse_BC_ON','S_ON_BC')},
            AC_drives={n:float(np.mean(v)) for (n,k),v in last.items() if n in arm.inner.ac_populations and k=='drive'},
            RGC_gE_S={n:float(np.sum(last[n,'gE'])) for n in RGC},
            port_units={':'.join(k):v for k,v in rows[-1]['port_units'].items()})
        return dict(status='PASS',factor=factor,duration_seconds=.05,onset_seconds=2/150,
            wall_seconds=time.perf_counter()-clock,BC_responses=summary,interfaces=interfaces,
            restored_max_abs_difference=restore_error,RGC_history_RNG_equal=True,
            checkpoint_parameter_mismatch_rejected=True,runner_metrics=arm.metrics,
            identical_initial_state_sha256=canonical_hash(initial['states'])), traces
    finally:
        arm.close()
        if restored is not None: restored.close()


def plot(out, traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(out/'CONNECTION_MEMBERS.npz',allow_pickle=False) as source:
        members = {pop:source[pop+'__member_counts'].copy() for pop in BC}
    fig,axes = plt.subplots(5,4,figsize=(16,13),sharex=True)
    for i,pop in enumerate(BC):
        for j,(field,label,scale) in enumerate((('voltage','V (mV)',1000),('unadapted_release','terminal gate',1),
                ('effective_release','effective release',1),('adaptation_activity','adaptation A',1))):
            for name,data in traces.items():
                axes[i,j].plot(data['time_seconds']*1000,np.average(data['ports__'+pop+'__'+field],axis=0,weights=members[pop])*scale,label=name)
            axes[i,j].axvline(2/150*1000,color='gray',lw=.6,ls='--')
            axes[i,j].set_ylabel(label);axes[i,j].grid(alpha=.2)
            axes[i,j].ticklabel_format(axis='y',useOffset=False,style='plain')
            if j==0: axes[i,j].set_title(pop,loc='left',fontsize=10)
            if i==4: axes[i,j].set_xlabel('Time (ms)')
    axes[0,0].legend();fig.tight_layout();fig.savefig(out/'BC_RESPONSE_CURVES.png',dpi=180);plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); args = ap.parse_args()
    root = Path(__file__).resolve().parents[1];out = args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter(); result = dict(started_utc=datetime.now(timezone.utc).isoformat(),status='RUNNING')
    before = protected(root);write(out/'PROTECTED_BEFORE.json',before)
    try:
        model = RetinaBuilder.build_research_bc(root)
        result['model_identity'] = model.identity
        result['connectivity'] = connectivity(root,model.assembly,out)
        # Audit values are source metadata; they do not enter any equation.
        targets = {n:r['effective_conductance_nS'] for n,r in model.seed['bc_physiology']['ledger'].items()}
        inner = model.assembly.runtime_factories['inner'](model.assembly,model.seed['inner_parameters'])
        state = {k:model.seed['states'][k] for k in inner.state_keys}
        inner.parameters = dict(inner.parameters,provenance=dict(inner.parameters['provenance'],bc_targets_nS=targets))
        result['DC_conductance_check'] = dc_check(inner,state)
        rhs = inner.rhs(0.,state)
        result['background_BC_max_rhs'] = {n:{k:float(np.max(abs(rhs[n,BC_MECH,k]),initial=0.))
            for k in ('V','terminal_release','A')} for n in BC}
        if max(v['V'] for v in result['background_BC_max_rhs'].values()) > 1e-8:
            raise ContractError('New BC background is not compatible with its membrane equation')
        write(out/'PARAMETER_LEDGER.json',model.seed['bc_physiology'])
        save_bundle(out/'new_source_seed',kind='source_construction_seed',identity=dict(model=model.identity),
            payload=model.seed,qualification=dict(default_adopted=False,fit_eligible=False,research_only=True,quantitative_dynamic_fit_completed=False))
        result['conditions'] = {};all_traces = {}
        for name,factor in (('brighter',1.2),('dimmer',.8)):
            result['conditions'][name],all_traces[name] = condition(model,factor,name,out)
        assert len({c['identical_initial_state_sha256'] for c in result['conditions'].values()}) == 1
        plot(out,all_traces)
        if protected(root) != before:
            raise ContractError('Protected historical/source files changed')
        result.update(status='ENGINEERING_ACCEPTANCE_PASS',model_implementation_completed=True,
            all_five_types_readable=True,physiological_Rin_source_constraints=True,
            quantitative_BC_dynamic_fit_completed=False,physiologically_calibrated=False,
            historical_sources_unchanged=True,source_background=str(background_directory()),
            scope='One real fine-member local graph within 0-6 degrees; two 50ms native Td steps from identical background; no optics rerun, fit or natural movie')
    except Exception as error:
        result.update(status='FAILED_RETAINED',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        result.update(ended_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-start,
            peak_working_set_bytes=peak_memory_bytes())
        write(out/'RESULT.json',result)
    print(result['status'],result['wall_seconds'],flush=True)


if __name__=='__main__': main()
