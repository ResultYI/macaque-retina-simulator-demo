"""Read-only final evidence synthesis, including original contact identities."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
import difflib
import json
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from retina_sim.contracts import canonical_hash
from retina_sim.spatial_group_graph import digest
from retina_sim.group_time_monitor import frozen_budget
from retina_sim.group_time_storage import load_bundle
from tools.validate_ac_vs import causal_hash
from retina_sim.ac_vs_model import full_input_kernel
from scipy import sparse
from types import SimpleNamespace

OUT=ROOT/'reports/ac_vs_redesign_v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def contacts_compare(left,right,*,numerical):
    lreg,rreg=read(left/'trace_registry.json'),read(right/'trace_registry.json')
    lsel=np.load(OUT/'functional_static_halo_030_v1/selection.npz',allow_pickle=False)
    rsel=np.load(OUT/('functional_static_halo_030_v1' if numerical else 'functional_static_halo_060_v1')/'selection.npz',allow_pickle=False)
    parent=read(OUT/'functional_static_halo_030_v1/selection_identity.json')['configuration']['source_artifact']['path']
    base=(ROOT/parent).parent;manifest=read(base/'manifest.json')
    routes={row['contact']['component']:row for row in manifest['routes'] if row.get('contact',{}).get('component')}
    rows=[]
    with np.load(left/'traces.npz',allow_pickle=False) as l,np.load(right/'traces.npz',allow_pickle=False) as r:
        np.testing.assert_array_equal(l['time_seconds'],r['time_seconds'])
        for field in l.files:
            if field not in r.files or not (field.startswith('states__') or field.startswith('ports__')):continue
            category,text=field.split('__',1);key=text.split(':')
            component=key[1] if category=='states' else key[0]
            if component not in routes or (category=='ports' and key[-1]!='contact_conductance'):continue
            route=routes[component]
            lrows,rrows=lsel['contact__'+component],rsel['contact__'+component]
            if numerical:keep=np.arange(len(lrows))
            else:
                post=np.load(base/'routes'/route['name']/'contact_post_group.npy',allow_pickle=False)
                keep=np.flatnonzero(np.isin(post[lrows],lsel['core__'+route['post']]))
            mapping={int(row):i for i,row in enumerate(rrows)}
            pairs=[(int(i),mapping[int(lrows[i])]) for i in keep if int(lrows[i]) in mapping]
            if not pairs:continue
            a=l[field][:,[i for i,j in pairs]];b=r[field][:,[j for i,j in pairs]]
            unit=lreg['state_units'][text] if category=='states' else lreg['units'][text]
            if numerical:
                if category!='states':continue
                bound=frozen_budget(l[field][0],unit)[2][[i for i,j in pairs]][None,:]
            else:bound=1e-11 if unit=='S' else 1e-8+.001*np.maximum(np.abs(a),np.abs(b))
            norm=float(np.max(np.abs(a-b)/bound,initial=0.))
            rows.append(dict(field=field,rows=len(pairs),max_abs=float(np.max(np.abs(a-b),initial=0.)),maximum_norm=norm,passed=norm<=1))
    lsel.close();rsel.close()
    return rows


def export_routing_matrices(config):
    destination=OUT/'routing_matrices_v1';destination.mkdir(exist_ok=True)
    ledger=[]
    for halo,runname in [('030','run_030_base_v2'),('060','run_060_base_v1')]:
        selected=np.load(OUT/('functional_static_halo_'+halo+'_v1')/'selection.npz',allow_pickle=False)
        parent=(ROOT/read(OUT/('functional_static_halo_'+halo+'_v1')/'selection_identity.json')['configuration']['source_artifact']['path']).parent
        validated=read(OUT/runname/'routing.json');folder=destination/halo;folder.mkdir(exist_ok=True)
        def population(name):
            return SimpleNamespace(group_count=len(np.load(parent/'populations'/name/'counts.npy',allow_pickle=False)),
                counts=np.load(parent/'populations'/name/'counts.npy',allow_pickle=False),
                centroid_mm=np.load(parent/'populations'/name/'centroid_mm.npy',allow_pickle=False))
        for c,v in zip(config['channels'],validated):
            pre,post=population(c['pre']),population(c['target'])
            full,_=full_input_kernel(pre,post,config['geometry'])
            prows,qrows=selected['population__'+c['pre']],selected['population__'+c['target']]
            local=full[qrows][:,prows].tocsr()
            logical=canonical_hash(dict(data=local.data,indices=local.indices,indptr=local.indptr,shape=local.shape))
            if logical!=v['local_kernel_sha256']:raise AssertionError('Export differs from actual validated route')
            name=c['pool'];files={}
            for suffix,matrix in [('input_fraction_weights',local),('input_conductance_S',c['G_BA_S']*local),
                ('output_fraction_weights',sparse.eye(len(qrows),format='csr')),
                ('output_total_conductance_S',sparse.diags(post.counts[qrows]*c['G_AR_S'],format='csr'))]:
                path=folder/(name+'__'+suffix+'.npz');sparse.save_npz(path,matrix)
                files[path.name]=digest(path)
            path=folder/(name+'__RF_positions_mm.npy');np.save(path,post.centroid_mm[qrows],allow_pickle=False);files[path.name]=digest(path)
            ledger.append(dict(halo=halo,pool=name,local_logical_hash=logical,files=files,
                counts_applied='BC original member counts in full fractional kernel once; target counts in output_total_conductance_S once; effective pool count1',crop_renormalized=False))
        selected.close()
    write(destination/'manifest.json',dict(schema='ac-vs-exported-validated-routing-1',routes=ledger))


def main():
    base=OUT/'run_030_base_v2';half=OUT/'run_030_half_v1';halo=OUT/'run_060_base_v1';restored=OUT/'run_030_restored_v1'
    numerical=read(OUT/'numerical_base_half_v1/result.json');spatial=read(OUT/'spatial_030_060_v1/result.json')
    for report,left,right,mode in ((numerical,base,half,True),(spatial,base,halo,False)):
        extra=contacts_compare(left,right,numerical=mode)
        report['contact_comparisons_by_original_parent_identity']=extra
        report['maximum_norm']=max([report['maximum_norm'],*(row['maximum_norm'] for row in extra)])
        report['passed']=report['passed'] and bool(extra) and all(row['passed'] for row in extra)
        if mode:
            registry=read(left/'trace_registry.json')['state_units']
            with np.load(left/'traces.npz',allow_pickle=False) as l,np.load(right/'traces.npz',allow_pickle=False) as r:
                empty=[name for name in registry if l['states__'+name].shape[1]==0 and r['states__'+name].shape[1]==0]
                report['empty_state_blocks_covered']=[dict(state=name,shape=list(l['states__'+name].shape),
                    passed=bool(np.isfinite(l['states__'+name]).all() and np.array_equal(l['states__'+name],r['states__'+name]))) for name in empty]
            covered={row['field'].split('__',1)[1] for row in report['comparisons']+extra if row['field'].startswith('states__')}
            covered.update(empty)
            missing=set(registry)-covered
            report['complete_state_coverage']=dict(total_blocks=len(registry),covered_blocks=len(covered),missing=sorted(missing))
            report['passed']=report['passed'] and not missing
            report['refinement_scope']='New joint inner CPU DOP853 max step100us versus50us; native reference is unchanged and checked independently by dense defects, not independently half-refined here.'
        write(OUT/('NUMERICAL_COMPLETE.json' if mode else 'SPATIAL_COMPLETE.json'),report)
    continuous=load_bundle(base/'final_checkpoint',expected_kind='complete_model_checkpoint')['payload']
    resumed=load_bundle(restored/'final_checkpoint',expected_kind='complete_model_checkpoint')['payload']
    checked={field:causal_hash(continuous[field])==causal_hash(resumed[field]) for field in continuous}
    initial=read(restored/'result.json')
    restore=dict(category='RESTORE',passed=all(checked.values()) and initial['restored_start_hash']==initial['saved_checkpoint_hash'],
        complete_causal_sha256=causal_hash(continuous),per_field_exact_equality=checked,
        original_pid=read(base/'resource_receipt.json')['pid'],new_process_pid=read(restored/'resource_receipt.json')['pid'],
        excluded='noncausal metrics dictionaries only; all state/anchors/controllers/history/queues/RNG/events/readout cursors/monitor retained')
    write(OUT/'RESTORE_COMPLETE.json',restore)
    sources=read(base/'model_identity.json')['sources']
    if any(digest(ROOT/p)!=h for p,h in sources.items()):raise AssertionError('Validated runtime source changed')
    config=read(ROOT/'configurations/ac_vs_functional_v1.json')
    export_routing_matrices(config)
    routing=read(base/'routing.json');bg=read(OUT/'background_030_v1/result.json');isolated=read(OUT/'isolated_step_v1/result.json')
    runs={name:read(path/'result.json') for name,path in [('030_base',base),('030_half',half),('060_base',halo),('030_restore',restored)]}
    static=all(row['missing_edges']==0 and row['selected_empty_rows']==0 and not row['full_parent_outer_boundary_row_indices'] and row['effective_pool_count']>0 and not row['crop_renormalized'] for row in routing)
    dynamic=all(r['DYNAMIC_NUMERICAL_DEFECT']['passed'] and r['PHYSICAL_DOMAIN_AND_CONSERVATION']['passed'] and r['INPUT_EVENT_AND_HISTORY']['passed'] for r in runs.values()) and numerical['passed']
    status=dict(SOURCE='PASS_EXPLICIT_ENGINEERING_PROVENANCE_NOT_PHYSIOLOGICAL_CALIBRATION',
        MODEL='PASS_VOLTAGE_GATE_AND_LEGACY_ISOLATION' if isolated['passed'] else 'FAIL',
        ROUTING='PASS_FOUR_NONEMPTY_TARGET_PATHS' if static else 'FAIL',
        BACKGROUND='PASS_CONDITIONAL_HELD_EXTERIOR_LOCAL_EQUILIBRIUM' if bg['passed'] else 'FAIL',
        DYNAMIC_NUMERICAL='PASS_NEW_INNER_AND_DENSE_DEFECTS' if dynamic else 'FAIL',
        SPATIAL='PASS_FROZEN_3P1MS_CORE_ONLY' if spatial['passed'] else 'FAIL',
        RESTORE='PASS_NEW_PROCESS_COMPLETE_CAUSAL_STATE' if restore['passed'] else 'FAIL')
    raw_files=[*OUT.glob('*/failure.json'),*OUT.glob('*/resource_receipt.json')]
    receipt=dict(schema='ac-vs-final-engineering-qualification-1',created_utc=datetime.now(timezone.utc).isoformat(),
        status=status,source_identity=sources,config_sha256=digest(ROOT/'configurations/ac_vs_functional_v1.json'),
        validation_contract_sha256=digest(ROOT/'configurations/ac_vs_acceptance_v1.json'),
        dynamic_maximum_norms={k:v['DYNAMIC_NUMERICAL_DEFECT']['maximum_norm'] for k,v in runs.items()},
        numerical_maximum_norm=numerical['maximum_norm'],spatial_maximum_norm=spatial['maximum_norm'],
        background_maximum_norm=bg['absolute_fixed_point_report']['maximum_norm'],
        old_false_preserved={k:v['absolute_monitor_preserved']['passed'] for k,v in runs.items()},
        worker_and_failure_receipts={str(p.relative_to(OUT)):digest(p) for p in raw_files},
        physiological_calibration=False,FIT_READY=False,model_runtime_ownership_released=not (OUT/'ACTIVE_WORKER.json').exists(),
        independent_native_half_refinement=False,arbitrary_dynamic_exterior_or_natural_movie_qualified=False,
        background_automatically_used=False,all_parameters_frozen=True,optimizer_steps=0)
    write(OUT/'FINAL_QUALIFICATION.json',receipt)
    backup=Path('D:/macaque_retina_simulator.backup_pre_ac_redesign_20261027_v1')
    patches=[]
    for relative in ['retina_sim/group_inner_dynamics.py','retina_sim/group_time_runner.py']:
        patches.extend(difflib.unified_diff((backup/relative).read_text(encoding='utf8').splitlines(True),(ROOT/relative).read_text(encoding='utf8').splitlines(True),fromfile='pre_ac_backup/'+relative,tofile=relative))
    (OUT/'existing_source_changes.patch').write_text(''.join(patches),encoding='utf8')
    seeds={name:load_bundle(path/'construction_seed',expected_kind='source_construction_seed')['payload'] for name,path in [('030',base),('060',halo)]}
    write(OUT/'MODEL_IDENTITIES.json',dict(config=config,state_blocks={name:len(seed['states']) for name,seed in seeds.items()},
        state_scalars={name:sum(len(a) for a in seed['states'].values()) for name,seed in seeds.items()},
        runtime_identity={name:r['identity'] for name,r in runs.items()},new_route_ledger=routing,
        original_weights_archived_at=str(backup),backup_verification_receipt_sha256=digest(backup/'_LOCAL_BACKUP_METADATA/verification.json')))
    print(json.dumps(dict(status=status,worker_released=receipt['model_runtime_ownership_released'])))


if __name__=='__main__':main()
