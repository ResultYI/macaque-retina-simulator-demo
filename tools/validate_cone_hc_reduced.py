"""One sealed, bounded real-movie reduced-native comparison; never production.

Import is CPU-only. --prepare-only reads small manifests and seals the protocol;
--execute uses one owned worker. It does not regenerate light or background,
change old receipts, generate a source cache, fit, or select a default backend.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_sim.cone_hc_observation_queue import (
    queue_contract, upgrade_split_reference, queue_time_evidence, compare_queue_times,
)
REPORT = ROOT / 'reports/cone_hc_reduced_implementation_20261024_v1'
ORIGINAL = ROOT / 'reports/first_useful_split_capture_20261023_v1'
SCHEMA = 'cone-hc-reduced-real-short-protocol-2'
GIB = 1024**3


class ScientificGateFailure(RuntimeError):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024**2), b''):
            h.update(part)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf8')).hexdigest()


def write_json(path, value, *, replace_existing=False):
    path = Path(path)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf8')
    if not replace_existing:
        with path.open('xb') as stream:
            stream.write(raw)
        return
    temporary = path.with_name(path.name+'.writing-'+str(os.getpid()))
    with temporary.open('wb') as stream:
        stream.write(raw)
    os.replace(temporary, path)


def _project_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Validation assets must belong to this project')
    return path


def build_protocol(run_dir=ORIGINAL):
    """Small-metadata preflight only; no NumPy, model, CUDA or asset hashing."""
    run_dir = _project_path(run_dir)
    session_path = run_dir/'source_execution/movie_chunks/session.json'
    session = json.loads(session_path.read_bytes())
    binding = session['binding']
    chunk = run_dir/'source_execution/movie_chunks/chunk-000000'
    commit = json.loads((chunk/'commit.json').read_bytes())
    manifest_path = chunk/'checkpoint/manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    if (commit['index'] != 0 or commit['checkpoint_manifest_sha256'] != manifest['manifest_hash']
            or manifest['time_seconds'] != 6.4 or manifest.get('fit_eligible') is not False
            or binding['original_teacher_background_frames'] != 750
            or binding['qualification']['status'] != 'verified_original_background_contract'):
        raise ValueError('Expected authentic, qualified-lineage movie chunk0 at6.4s')
    stop = 6.5
    origin = float(binding['raw_zero_absolute_seconds'])
    frame_edges = [origin+i/150 for i in range(9752)
                   if manifest['time_seconds'] < origin+i/150 < stop]
    start = float(manifest['time_seconds'])
    grid = [i*.0001 for i in range(64000, 65001)]
    interior = [start+.000025, start+.00005, start+.000075]
    times = sorted(set([start, stop]+[t for t in grid if start<t<stop]+interior))
    points = sorted(set([start, stop, *interior,
        start+.025, start+.05, start+.075, *frame_edges]))
    seam = start+.000031
    times = sorted(set(times+[seam]))
    points = sorted(set(points+[seam]))
    files = {'source_session': session_path, 'source_commit': chunk/'commit.json',
        'source_checkpoint_manifest': manifest_path,
        'source_graph_binding': run_dir/'source_execution/source_graph_binding.json',
        'source_input_definition': run_dir/'source_execution/movie_input_definition.json',
        'input_policy_approval': run_dir/'input_policy_approval.json',
        'frozen_budget': ROOT/'configurations/numerical_equivalence/resident_native_fp64_v1.json'}
    native = manifest['native_cursor']['solver']
    protocol = dict(schema=SCHEMA, classification='B', default_backend_change=False,
        observation_queue=queue_contract(),
        production_fit_eligible=False, optical_or_background_forward=False,
        source_run_directory=run_dir.relative_to(ROOT).as_posix(),
        checkpoint_directory=(chunk/'checkpoint').relative_to(ROOT).as_posix(),
        checkpoint_manifest_hash=manifest['manifest_hash'], source_model_hash=manifest['model_hash'],
        science_sha256=manifest['scientific_contract_sha256'],
        source_input_binding_hash=manifest['input_binding_hash'],
        start_seconds=start, stop_seconds=stop, raw_zero_seconds=origin,
        checkpoint_seam_seconds=seam,
        movie_window_seconds=[.1, .2], frame_edges_seconds=frame_edges,
        sample_times_seconds=times, comparison_times_seconds=points,
        point_selection='last complete observer snapshot at each exact(t,input side); monitors still inspect every callback',
        auxiliary_cone_indices=[0, 272829, 545657],
        native_reference=dict(method='persistent DOP853',rtol=native['rtol'],atol=native['atol'],
            max_step=native['max_step']),
        native_candidate=dict(method='exponential_midpoint',dtype='float64',base_step=.0001,
            true_input_edges=True, interpolation='x0+theta*h*f0+theta**2*(x1-x0-h*f0)',
            R_E='exact two-stage ZOH filter with preserved anchors',w='A@old_z1'),
        inner=dict(method='DOP853',rtol=1e-6,atol=1e-9,max_step=.0001,
            held_slot_exclusion=False,rod_author_step=.0001,BC_RGC_step=.001),
        budgets=dict(original_strict='.01*min(E,D)',pro_candidate='min(E,.01*D)',
            E='1e-9+1e-6*S',D='registered_unit_atol+original_monitor_rtol*S',
            S='max(abs(original frozen reference initial),registered unit floor)',
            w='abs(A)@old_z1_allowance; no new free tolerance',
            native_outputs='original typed port budget from original initial output',
            events='original ResidentQAComparator.compare_draw; exact actual draws/history/PCG64, first divergence stops',
            same_candidate_bridge='original typed port allowance; CPU versus GPU direct same-trajectory release',
            no_post_result_adjustment=True),
        limits=dict(max_wall_seconds=1800,max_rss_bytes=12*GIB,max_vram_bytes=12*GIB,
            min_free_ram_bytes=8*GIB,min_free_disk_bytes=8*GIB),
        saved_metadata={key:dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p))
                        for key,p in files.items()},
        required_checks=['initial one-way migration','R/E exact input-side anchors',
            'native core states and original RHS domains','w versus original A@z1',
            'preselected auxiliary z1','CPU bridge versus same GPU trajectory',
            'inner states/ports and actual RGC draws/history/PCG64',
            'original2.82ms observation-only queue','new complete checkpoint restore',
            'full monitor on both arms','separate cold/native/transfer/inner/monitor/advance wall'],
        performance_scope='one same-window full neural advance; no fit/optics/cold preparation included',
        user_authorization='2026-10-24 direct human authorization: implement, bounded real GPU A/B, commit and normal push; production remains stopped',
        original_history_preserved=True)
    sources = sorted((ROOT/'retina_sim').glob('cone_hc_reduced_*.py'))+[
        ROOT/'retina_sim/cone_hc_observation_queue.py',Path(__file__)]
    protocol['implementation_sha256'] = {p.relative_to(ROOT).as_posix():digest(p) for p in sources}
    stop_file = run_dir/'REQUEST_STOP'
    protocol['production_stop'] = dict(path=stop_file.relative_to(ROOT).as_posix(),sha256=digest(stop_file))
    protocol['protocol_sha256'] = canonical(protocol)
    return protocol


def validate_protocol(protocol):
    item = deepcopy(protocol)
    digest_value = item.pop('protocol_sha256', None)
    if item.get('schema') != SCHEMA or digest_value != canonical(item):
        raise ValueError('Validation protocol mutated or unsealed')
    if item.get('observation_queue') != queue_contract():
        raise ValueError('Observation queue contract differs')
    if (item['start_seconds'],item['stop_seconds']) != (6.4,6.5):
        raise ValueError('Only the frozen100ms actual-movie interval is authorized here')
    if item['native_candidate']['base_step'] != .0001 or item['inner'] != dict(
            method='DOP853',rtol=1e-6,atol=1e-9,max_step=.0001,held_slot_exclusion=False,
            rod_author_step=.0001,BC_RGC_step=.001):
        raise ValueError('Frozen methods or event clocks changed')
    for record in item['saved_metadata'].values():
        path = _project_path(ROOT/record['path'])
        if digest(path) != record['sha256']:
            raise ValueError('Original saved validation metadata changed: '+record['path'])
    for path,sha in item['implementation_sha256'].items():
        if digest(_project_path(ROOT/path))!=sha:
            raise ValueError('Prospectively sealed implementation changed: '+path)
    if digest(ROOT/item['production_stop']['path'])!=item['production_stop']['sha256']:
        raise ValueError('Original production REQUEST_STOP changed')
    return item


def frozen_allowance(initial, unit, monitor_rtol, *, pro=False, units=None):
    """Same original frozen state/typed-output formula, never candidate-scaled."""
    import numpy as np
    if units is None:
        from retina_sim.background import NUMERICAL_UNIT_SCALES
        units = NUMERICAL_UNIT_SCALES
    from retina_sim.resident_contract import CONTRACT_SHA256
    path = ROOT/'configurations/numerical_equivalence/resident_native_fp64_v1.json'
    if digest(path) != CONTRACT_SHA256:
        raise ValueError('Frozen numerical formula source changed')
    original = json.loads(path.read_bytes())
    entry = units[unit]
    value = np.asarray(initial, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError('Allowance initial reference is nonfinite')
    scale = np.maximum(np.abs(value),entry['scale_floor'])
    config = original['solver_required']
    E = config['atol']+config['rtol']*scale
    D = entry['atol']+float(monitor_rtol)*scale
    f = original['state_budget_fraction']
    return np.minimum(E,f*D) if pro else f*np.minimum(E,D)


def propagate_w_allowance(A, z1_allowance):
    import numpy as np
    from scipy import sparse
    matrix = sparse.csr_matrix(A,copy=True,dtype=np.float64)
    radius = np.asarray(z1_allowance,dtype=np.float64)
    if (radius.shape != (matrix.shape[1],) or not np.isfinite(radius).all()
            or np.any(radius<0) or not np.isfinite(matrix.data).all()):
        raise ValueError('Finite original matrix and nonnegative z1 allowances required')
    matrix.data = np.abs(matrix.data)
    return np.asarray(matrix@radius).ravel()


def compare_vector(reference,candidate,allowance,*,gate):
    import numpy as np
    a,b,r=map(lambda x:np.asarray(x,dtype=np.float64),(reference,candidate,allowance))
    if a.shape!=b.shape or a.shape!=r.shape:
        return dict(status='FAIL',gate=gate,reason='shape differs')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(r).all() or np.any(r<0):
        return dict(status='FAIL',gate=gate,reason='nonfinite or negative allowance')
    delta=np.abs(a-b)
    ratio=np.divide(delta,r,out=np.where(delta==0.,0.,np.inf),where=r>0)
    bad=np.flatnonzero(delta>r)
    zero_failure = bool(np.any((r==0)&(delta>0)))
    return dict(status='FAIL' if len(bad) else 'PASS',gate=gate,
        max_abs=float(delta.max(initial=0.)),
        max_normalized=None if zero_failure else float(ratio.max(initial=0.)),
        zero_allowance_violation=zero_failure,
        first_bad_index=None if not len(bad) else int(bad[0]))


def _movie_channels(reference,base,directory,session,catalog):
    """Same source functions and arithmetic as run_vr_movie_source; no capture."""
    import numpy as np
    from retina_sim.contracts import ContractError,Quantity,canonical_hash
    from retina_sim.spatial import InputChannel
    from retina_sim.source_graph_contract import source_rf_mm
    from tools.run_vr_useful_split import rod_input_breakpoints
    from tools.vr_teacher_handoff_v1.fit_integration_v1.capture_preparation_v2.linear_gray import LinearGrayScene
    raw_origin=float(session['binding']['raw_zero_absolute_seconds'])
    adapter=LinearGrayScene(ROOT,catalog.approval)
    channels,hashes,rod_cache={},{},{}
    grid_origin=0.
    for external in reference.compiled.spec.external_inputs:
        component=reference.compiled.components[external.target_component]
        port=component.inputs[external.target_port];old=base.channels[external.channel]
        metadata=dict(old.input_contract,declared_constant_background=False,
            raw_zero_absolute_seconds=raw_origin,recorded_background_frames=750,
            vr_policy_sha256=catalog.approval['policy_sha256'],exact_input_definition_sha256=catalog.definition_hash,
            actual_prepared_output_receipts_required_per_block=True,future_optical_outputs_are_not_certified=True)
        if port.unit=='td':
            def evaluate(t,port=port,old=old):return catalog.native(t,port,old.evidence_tag)
            channels[external.channel]=InputChannel(0.,catalog.edges[-1],evaluate,port.unit,port.meaning,
                'zero_order_hold',old.evidence_tag,metadata,tuple(catalog.edges[:-1]))
        elif component.time_contract.kind=='author_discrete' and external.target_port=='light':
            population=reference.compiled.populations['rod'];lookup={x:i for i,x in enumerate(population.cells.cell_id)}
            xy=source_rf_mm(population)[[lookup[x] for x in port.cell_ids]]
            def evaluate_rod(t,xy=xy,port=port,old=old):
                tick=np.floor((t-grid_origin)/.0001);absolute=grid_origin+tick*.0001
                if absolute<raw_origin:return old.at(t)
                i=int(np.searchsorted(catalog.edges,absolute,side='right')-1)
                if not 0<=i<9751:raise ContractError('Original raw rod input/guard exhausted')
                if i not in rod_cache:
                    rod_cache.clear();rod_cache[i]=Quantity(adapter.independent_rod_rates(catalog.reader.frame(i),xy),
                        port.unit,port.meaning,old.evidence_tag)
                return rod_cache[i]
            metadata.update(grid_step_seconds=.0001,includes_background=True)
            channels[external.channel]=InputChannel(0.,catalog.edges[-1],evaluate_rod,port.unit,port.meaning,
                'zero_order_hold',old.evidence_tag,metadata,rod_input_breakpoints(grid_origin,0.,catalog.edges[-1]))
        else:
            if old.input_contract.get('declared_constant_background') is not True:
                raise ContractError('Unreviewed extra nonconstant input')
            value=old.at(old.start_seconds)
            channels[external.channel]=InputChannel(0.,catalog.edges[-1],lambda t,v=value:v,
                old.unit,old.meaning,old.interpolation,old.evidence_tag,dict(old.input_contract),())
        hashes[external.channel]=canonical_hash(dict(channel=external.channel,
            actual_raw_and_frozen_transform=catalog.definition,input_contract=channels[external.channel].input_contract))
    return channels,hashes


def prepare_actual_context(directory,output,protocol,guard,authorization):
    """Rebuild/restore the authentic reference without any background advance."""
    import numpy as np
    from retina_platform.config import RunConfig
    from retina_sim.contracts import ContractError,canonical_hash,SolverConfig
    from retina_sim.atlas_population_network import atlas_population_network
    from retina_sim.gpu_compact import CompactCudaHybridRunner
    from retina_sim.resident_cusparse import make_candidate
    from retina_sim.source_graph_contract import SourceGraphContract,source_rf_mm
    from retina_sim.resident_background import bind_resident_background
    from retina_sim.cone_hc_split_contract import ConeHCSplitContract,USEFUL_SCOPE
    from retina_sim.cone_hc_split_runner import ConeHCSplitRunner
    from retina_sim.cone_hc_split_checkpoint import load_complete
    from retina_sim.split_chunk_store import SplitChunkStore
    from tools.run_vr_useful_split import require_frontend
    from tools.vr_exact_input_catalog import ExactInputCatalog
    from tools.vr_teacher_handoff_v1.fit_integration_v1.capture_preparation_v2.handoff import approved_policy
    validate_protocol(protocol);directory=_project_path(directory);output=Path(output)
    begun=time.perf_counter();guard.require(force=True)
    _,frontend=require_frontend(directory)
    approval=json.loads((directory/'input_policy_approval.json').read_bytes())
    handoff,objects=approved_policy(ROOT,approval)
    session_path=directory/'source_execution/movie_chunks'
    raw_session=json.loads((session_path/'session.json').read_bytes())
    store=SplitChunkStore(session_path,expected_session_hash=raw_session['receipt_sha256'])
    session=store.session
    ref=json.loads((directory/'actual_global_reference.json').read_bytes())
    if canonical_hash(ref)!=frontend['global_reference_numeric_sha256']:
        raise ContractError('Actual ISET reference differs from saved frontend')
    input_dir=directory/'linear_inputs'
    background=np.load(input_dir/'cone_background_counts.npy',allow_pickle=False,mmap_mode='r')
    table=json.loads((ROOT/'data/atlas/human_disk3mm_typed_tables_v1/cones/manifest.json').read_bytes())
    types=np.load(ROOT/table['extras']['subtype']['path'],allow_pickle=False,mmap_mode='r')
    alpha=np.asarray(ref['q_ref_LMS_rstar_s'])[[{'L':0,'M':1,'S':2}[t.decode('ascii')] for t in types]]/100.
    native_background=np.asarray(background)/(1/150.)/alpha
    if not np.array_equal(native_background,np.load(input_dir/'cone_background_native_td.npy',allow_pickle=False,mmap_mode='r')):
        raise ContractError('Original actual background capture bridge differs')
    cfg=RunConfig.project_default(ROOT);policy=replace(cfg.background_policy(),max_wall_seconds=3600.)
    cfg.native_background_values=tuple(native_background)
    cfg.native_background_td=objects['input_policy']['outside']['uniform_scene_photopic_reference_td_before_cell_capture']
    cfg.rod_background_rstar_s=objects['input_policy']['rod']['outside_rate_Rstar_per_s']
    cfg.cone_background_rstar_s=ref['q_ref_LMS_rstar_s'][0]*(19/255/.5)
    cfg.frontend_signature=canonical_hash(frontend);light=cfg.light_context()
    config_path=handoff['references']['capture_configuration']['path'];configuration=objects['capture_configuration']
    print('assemble_actual_original_graph_no_forward',flush=True)
    compiled,_,audit=atlas_population_network(ROOT,duration=12.,light_context=light,
        owner_bounds_mm=configuration['owner_bounds_mm'],halo_mm=configuration['halo_mm'],
        max_spatial_cells=600000,seed=cfg.seed,execution_profile='atlas_isetbio_geometry_v1',configuration_path=config_path)
    guard.require(force=True)
    print('verify_original_graph_and_bind_inputs',flush=True)
    if not np.array_equal(compiled.global_disk_view['selected_rows']['cones'],np.arange(len(alpha))):
        raise ContractError('Original global cone order differs')
    for cid,required in configuration['owner_global_rows'].items():
        if not np.array_equal(np.sort(compiled.global_disk_view['selected_rows'][cid]),np.sort(required)):
            raise ContractError('Approved original8+4 owner set differs')
    for cell in objects['cells']['cells']:
        pop=compiled.populations[cell['student_component']];ids=list(pop.cells.cell_id)
        if cell['student_cell_id'] not in ids or not np.array_equal(source_rf_mm(pop)[ids.index(cell['student_cell_id'])],cell['rf_mm']):
            raise ContractError('Approved comparison cell/RF identity differs')
    compact=CompactCudaHybridRunner(compiled);solver=SolverConfig(**audit['resolved_execution_profile']['solver'])
    snapshot=compact.initialize(solver)
    source_contract=SourceGraphContract(compact,configuration_path=ROOT/config_path,
        configuration_sha256=handoff['references']['capture_configuration']['sha256'])
    saved_graph=json.loads((directory/'source_execution/source_graph_binding.json').read_bytes())
    if source_contract.source_graph_binding!=saved_graph:raise ContractError('Authentic saved source graph changed')
    retained=make_candidate(compact,numerical_contract=source_contract,purpose='approved_source_binding')
    base=bind_resident_background(retained,snapshot,light,policy)
    raw_origin=session['binding']['raw_zero_absolute_seconds']
    catalog=ExactInputCatalog(directory,raw_origin=raw_origin)
    saved_definition=json.loads((directory/'source_execution/movie_input_definition.json').read_bytes())
    if catalog.definition!=saved_definition:raise ContractError('Original exact actual input definition differs')
    channels,hashes=_movie_channels(retained,base,directory,session,catalog)
    proof=catalog.require_block(protocol['start_seconds'],protocol['stop_seconds'])
    actual_policy=canonical_hash({k:v.input_contract for k,v in channels.items()})
    a=dict(authorization,scope=USEFUL_SCOPE,input_policy_approved=True,target_graph_approved=True,
        default_backend_change=False,fit_or_upload_authorized=False,
        input_policy_receipt_sha256=digest(input_dir/'frontend_receipt.json'),
        target_graph_receipt_sha256=handoff['references']['capture_configuration']['sha256'],
        execution_protocol_sha256=protocol['protocol_sha256'],source_reference_graph_binding=saved_graph,
        source_resident_model_hash=retained.model.model_hash,baseline_binding_hash=retained.binding['binding_hash'],
        approved_input_policy_sha256_by_phase=dict(uniform_background=canonical_hash({k:v.input_contract for k,v in base.channels.items()}),movie=actual_policy))
    split=ConeHCSplitRunner.from_resident(retained,purpose='bounded_source',
        numerical_contract=ConeHCSplitContract.for_useful_chunks(a))
    split.import_resident_state(base.state,retained=retained,reference_state=base.reference_state,
        channels=base.channels,channel_data_hashes={k:canonical_hash(v.at(0.)) for k,v in base.channels.items()},monitor_policy=policy)
    saved_binding=deepcopy(session['binding']['source_input_binding'])
    plain=dict(saved_binding);sealed=plain.pop('binding_hash')
    stable=('assembly_hash','graph_hash','parameter_hash','state_spec_hash','initial_state_and_aux_hash',
        'monitor_policy_hash','monitor_reference_hash','solver_hash','source_pins_hash','contract_hash',
        'split_candidate_sha256','reference_model_hash','actual_import_source_model_hash')
    if (canonical_hash(plain)!=sealed or any(saved_binding[k]!=split.binding[k] for k in stable)
            or hashes!=saved_binding['channel_data_hashes'] or session['binding']['model_hash']!=split.model.model_hash
            or session['binding']['science_sha256']!=split.contract.sha256):
        raise ContractError('Saved movie phase cannot be restored under the authentic current graph/input')
    split.contract.validate_channels(channels,saved_binding)
    split._binding=saved_binding
    split._channel_evaluators={k:v.evaluator for k,v in channels.items()}
    split.runtime_guard=guard.require
    print('restore_original_complete_checkpoint_no_warmup',flush=True)
    state=load_complete(split,ROOT/protocol['checkpoint_directory'],expected_manifest_hash=protocol['checkpoint_manifest_hash'])
    if state.time_seconds!=protocol['start_seconds'] or state.grid_origin_seconds!=0.:
        raise ContractError('Authentic source start/calendar differs')
    legacy_model_hash = split.model.model_hash
    split,state = upgrade_split_reference(split,state)
    save=dict(cold_preparation_seconds=time.perf_counter()-begun,source_graph=saved_graph,
        legacy_model_hash=legacy_model_hash,observation_queue=queue_contract(),
        reference_model_hash=split.model.model_hash,restored_checkpoint_manifest_hash=protocol['checkpoint_manifest_hash'],
        actual_input_proof=proof,old_production_files_written=False,old_background_integrated=False)
    write_json(output/'prepared_reference.json',save)
    return dict(reference=split,state=state,retained=retained,base=base,channels=channels,policy=policy,
        registry=audit['registry'],catalog=catalog,authorization=a,audit=audit,preparation=save)


def _cpu(value):
    import numpy as np
    return value.detach().cpu().numpy().copy() if hasattr(value,'detach') else np.array(value,copy=True)


def _selected_delay(delay,t,device_rows):
    return _cpu(delay.at(t).index_select(0,device_rows))


def _collect_reference_recording(recorded,native_id,samples,delays,events):
    values = recorded.outputs_with_units[native_id+'.observed_Vh'].values
    if values.shape[1]!=len(samples):
        raise ValueError('Original selected observation coverage differs')
    for i,t in enumerate(samples):
        delays[float(t)] = values[:,i]
    events.extend(dict(component=cid,**e) for cid,rows in recorded.metadata['spike_events'].items() for e in rows)


def _snapshot(runner,t,y,bundle,aux_indices,*,reduced):
    """Complete retained state and actual typed outputs at a single exact side."""
    import numpy as np
    evidence = runner.evaluate(t,y,bundle,require_rhs=True)
    part = runner.partition
    native = runner.compiled.components[part.cid].native_model
    if reduced:
        from retina_sim.cone_hc_reduced_evaluation import state_arrays
        blocks = state_arrays(runner,y)
        result = {'native:'+k:_cpu(v) for k,v in blocks.items()}
        outputs = evidence.native_outputs
    else:
        blocks = dict(zip((s.state_name for s in native.state_specs),y[:part.offset].split(part.native.sizes)))
        result = {'native:'+k:_cpu(v) for k,v in blocks.items() if k!='z1'}
        result['native:w'] = _cpu(part.native.A.mv(blocks['z1'].contiguous()))
        result['native:z1_aux'] = _cpu(blocks['z1'][list(aux_indices)])
        outputs = dict(evidence.native_outputs)
        outputs['Bh'] = part.native.B.mv(blocks['Vh'].contiguous())
    for name in ('q','release','Ios','Bh'):
        result['native_port:'+name] = _cpu(outputs[name])
    for spec in runner.model.state_specs[len(native.state_specs):]:
        cid = spec.state_name.split('.',1)[0]
        names = [s.state_name for s in runner.compiled.components[cid].state_specs]
        index = names.index(spec.state_name.split('.',1)[1])
        result['inner:'+spec.state_name] = _cpu(evidence.local[cid][index])
    for (cid,name),quantity in evidence.outputs.items():
        result['port:'+cid+'.'+name] = quantity.values.copy()
    return result


def _freeze_budgets(context,initial):
    import numpy as np
    from retina_sim.background import NUMERICAL_UNIT_SCALES
    ref,base,policy = context['reference'],context['base'],context['policy']
    native = ref.compiled.components[ref.partition.cid].native_model
    old = dict(zip((s.state_name for s in native.state_specs),base.reference_state.ordered_state_arrays))
    units = {s.state_name:s.unit for s in native.state_specs}
    state_specs = {s.state_name:s for s in ref.model.state_specs}
    budgets = {'strict':{},'B':{}}
    unregistered = []
    for label,pro in (('strict',False),('B',True)):
        target = budgets[label]
        for name in ('X','C','Vis','g_i','Vbar','z2','Vh','R','E'):
            target['native:'+name] = frozen_allowance(old[name],units[name],policy.relative_tolerance,pro=pro)
        z1 = frozen_allowance(old['z1'],units['z1'],policy.relative_tolerance,pro=pro)
        target['native:w'] = propagate_w_allowance(native.A,z1)
        target['native:z1_aux'] = z1[context['protocol']['auxiliary_cone_indices']]
        for key,value in initial.items():
            if key in target:
                continue
            if key.startswith('inner:'):
                name = key.split(':',1)[1]
                index = next(i for i,s in enumerate(ref.model.state_specs) if s.state_name==name)
                target[key] = frozen_allowance(base.reference_state.ordered_state_arrays[index],
                    state_specs[name].unit,policy.relative_tolerance,pro=pro)
            elif key.startswith('native_port:'):
                name = key.split(':',1)[1]
                unit = 'vH_native_voltage' if name=='Bh' else ref.compiled.components[ref.partition.cid].outputs[name].unit
                target[key] = frozen_allowance(value,unit,policy.relative_tolerance,pro=pro)
            elif key.startswith('port:'):
                cid,name = key.split(':',1)[1].rsplit('.',1)
                unit = ('A' if name=='gap_current' else ref.compiled.components[cid].outputs[name].unit)
                if unit in NUMERICAL_UNIT_SCALES:
                    target[key] = frozen_allowance(value,unit,policy.relative_tolerance,pro=pro)
                elif label=='strict':
                    unregistered.append(key)
    return budgets,unregistered


def _draw_gate(expected,context,*,pro,sink):
    """Only the original actual-draw comparison; no full QA-plan certificate."""
    from types import MethodType
    from retina_sim.resident_qa import ResidentQAComparator
    from retina_sim.resident_contract import CONTRACT_SHA256
    from retina_sim.background import NUMERICAL_UNIT_SCALES
    c = ResidentQAComparator.__new__(ResidentQAComparator)
    c.np = __import__('numpy')
    c.plan = dict(draws=expected,unit_scales=NUMERICAL_UNIT_SCALES,
                  monitor_relative_tolerance=context['policy'].relative_tolerance)
    c.contract = json.loads((ROOT/'configurations/numerical_equivalence/resident_native_fp64_v1.json').read_bytes())
    c.states = {s.state_name:dict(initial=a,unit=s.unit,cell_ids=s.cell_ids)
        for s,a in zip(context['reference'].model.state_specs,context['base'].reference_state.ordered_state_arrays)}
    c.sink,c.issues,c.worst_status,c.closed = sink,[],'PASS',False
    c.seen,c.max_issues,c.issue_overflow,c.spike_cells = {'draws':[]},256,False,set()
    if pro:
        def budget(self,spec,section):
            a = spec['initial'][section]
            entry = NUMERICAL_UNIT_SCALES[spec['unit']]
            return c.np.maximum(c.np.abs(a),entry['scale_floor']),frozen_allowance(
                a,spec['unit'],context['policy'].relative_tolerance,pro=True)
        c._budget = MethodType(budget,c)
    return c


def _compare_ribbon(context,a,b,budgets,key):
    from retina_sim.resident_derived_ports import ribbon_descriptors, compare_cell, validate_descriptor
    descriptor = ribbon_descriptors(context['reference'].compiled).get(key.split(':',1)[1])
    if descriptor is None:
        return dict(gate=key,status='INCONCLUSIVE',reason='unregistered original derived output')
    params = validate_descriptor(descriptor)
    rows = []
    for i in range(len(a[key])):
        V,x = 'inner:'+descriptor['voltage_state'],'inner:'+descriptor['resource_state']
        row = compare_cell(**{k:float(params[n][i]) for k,n in (
            ('pool','Npool'),('U','U'),('rmax','rmax'),('r_half','r_half'),('r_slope','r_slope'))},
            V_reference=float(a[V][i]),V_candidate=float(b[V][i]),
            x_reference=float(a[x][i]),x_candidate=float(b[x][i]),
            rate_reference=float(a[key][i]),rate_candidate=float(b[key][i]),
            B_V=float(budgets[V][i]),B_x=float(budgets[x][i]))
        rows.append(row)
    return dict(gate=key,status='PASS' if all(r['passed'] for r in rows) else 'FAIL',cells=rows)


def run_worker(output):
    import traceback
    import platform
    import numpy as np
    import scipy
    import numba
    import torch
    from retina_sim.contracts import canonical_hash, ContractError
    from retina_sim.network import NetworkStimulus
    from retina_sim.recording import RecordingPlan
    from retina_sim.resident_qa_harness import _schedule
    from retina_sim.split_execution_guard import SplitExecutionGuard
    from retina_sim.cone_hc_reduced_runner import ReducedConeHCRunner
    from retina_sim.cone_hc_reduced_monitor import ReducedWindowMonitor
    from retina_sim.cone_hc_reduced_checkpoint import save_complete,load_complete,_payload
    from tools.benchmark_resident_continuation import actual_draw_capture
    output = Path(output).resolve()
    protocol = json.loads((output/'protocol.json').read_bytes())
    validate_protocol(protocol)
    write_json(output/'worker_started.json',dict(pid=os.getpid(),time_unix=time.time()))
    begun = time.time()
    limits = protocol['limits']
    a = dict(round_started_unix=begun,max_wall_seconds=limits['max_wall_seconds'],
        expires_unix=begun+limits['max_wall_seconds'],max_rss_bytes=limits['max_rss_bytes'],
        max_vram_bytes=limits['max_vram_bytes'],
        user_authorization_reference=protocol['user_authorization'])
    guard = SplitExecutionGuard(a,stop_file=output/'REQUEST_STOP',output_directory=output,
        min_free_ram=limits['min_free_ram_bytes'],min_free_disk=limits['min_free_disk_bytes'])
    def phase(name):
        guard.require(force=True)
        write_json(output/'progress.json',dict(phase=name,pid=os.getpid(),elapsed_seconds=time.time()-begun),replace_existing=True)
        print(name,flush=True)
    result = dict(status='INCONCLUSIVE',candidate_default=False,fit_eligible=False,
                  protocol_sha256=protocol['protocol_sha256'])
    try:
        phase('restore_actual_graph_and_cached_input')
        context = prepare_actual_context(ROOT/protocol['source_run_directory'],output,protocol,guard,a)
        context['protocol'] = protocol
        ref,source,channels = context['reference'],context['state'],context['channels']
        start,stop,seam = protocol['start_seconds'],protocol['stop_seconds'],protocol['checkpoint_seam_seconds']
        stimulus = NetworkStimulus(ref.compiled,channels,source,stop,source.solver_config.input_interpolation,ref.model.evidence_tag)
        initial = _snapshot(ref,start,source.y,stimulus.at(start),protocol['auxiliary_cone_indices'],reduced=False)
        original_time = context['base'].state.time_seconds
        original_initial = _snapshot(context['retained'],original_time,context['base'].state.y,
            stimulus.at(original_time),protocol['auxiliary_cone_indices'],reduced=False)
        budgets,unregistered = _freeze_budgets(context,original_initial)
        del original_initial
        np.savez(output/'frozen_budgets.npz',**{label+'|'+k:v for label,rows in budgets.items() for k,v in rows.items()})
        phase('construct_reduced_and_explicitly_migrate')
        reduced = ReducedConeHCRunner.from_resident(context['retained'],
            aux_indices=protocol['auxiliary_cone_indices'],runtime_guard=guard.require)
        candidate = reduced.migrate(source,source_runner=ref,channels=channels)
        migrated = _snapshot(reduced,start,candidate.point,stimulus.at(start),protocol['auxiliary_cone_indices'],reduced=True)
        migration = [compare_vector(initial[k],migrated[k],budgets['strict'][k],gate=k)
                     for k in budgets['strict']]
        write_json(output/'migration.json',migration)
        if any(r['status']!='PASS' for r in migration):
            raise ContractError('Initial legal one-way migration failed')
        times = np.array(protocol['sample_times_seconds'])
        _,_,ticks = _schedule(ref,source,stimulus,times,stop)
        expected = []
        for t,due in ticks:
            for cid in source.spike_states:
                if cid not in due:
                    continue
                n = ref.compiled.components[cid].native_model
                for i,cell in enumerate(n.state_specs[0].cell_ids):
                    expected.append(dict(id='draw_%05d'%len(expected),component=cid,cell_id=cell,
                        bin_start_seconds=float(t),dt=float(n.dt),voltage_state=cid+'.'+n.state_specs[0].state_name,
                        cell_index=i,alpha=float(n.alpha[i]),beta=float(n.beta[i])))
        write_json(output/'draw_schedule.json',expected)
        environment = dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,
            torch=torch.__version__,torch_cuda=torch.version.cuda,platform=platform.platform(),
            gpu=torch.cuda.get_device_name(reduced.native.device),
            cusparse_dll=str(reduced.native.A.api.path),cusparse_dll_sha256=reduced.native.A.api.sha256,
            cone_count=reduced.native.nc,hc_count=reduced.native.nh,components=len(ref.compiled.components),
            C_star=reduced.selection.count,H_star=reduced.selection.hc_count,
            packet_bytes=reduced.selection.packet_bytes,original_inner_scalars=reduced.partition.inner_size)
        write_json(output/'environment.json',environment)
        demo = None
        if protocol.get('demo_recording'):
            from tools.cone_hc_demo_recording import DemoRecorder
            demo = DemoRecorder(reduced,protocol,output)
        monitors = dict(reference=ref.make_monitor(context['base'].reference_state,context['policy']),
                        candidate=ReducedWindowMonitor(reduced,context['base'].reference_state,context['policy']))
        receipts,draws,costs,measurements = {},{}, {},{}
        actual_events = {'reference':[],'candidate':[]}
        critical = set(protocol['comparison_times_seconds'])
        bridge_checks = []
        selected_h = np.unique(np.linspace(0,reduced.native.nh-1,5,dtype=int))
        selected_h_device = torch.as_tensor(selected_h,dtype=torch.int64,device=reduced.native.device)
        observed_delay = {'reference':{},'candidate':{}}
        rgc_cells = tuple(cell for cid in source.spike_states for cell in ref.compiled.components[cid].state_specs[0].cell_ids)
        native_id = ref.partition.cid
        h_ids = tuple(reduced.native.model.hcs.cells.cell_id[i] for i in selected_h)
        recording = RecordingPlan(states=(),ports=(native_id+'.observed_Vh',),cell_ids=rgc_cells+h_ids)
        for arm,runner,state in (('reference',ref,source),('candidate',reduced,candidate)):
            phase('advance_'+arm)
            arm_dir = output/arm
            arm_dir.mkdir()
            receipts[arm],draws[arm] = {},[]
            costs[arm] = dict(entire_wrapped_RGC_seconds_upper_bound=0.,RGC_calls=0)
            online_witnesses = {'strict':[],'B':[]}
            online_gates = {label:_draw_gate(expected,context,pro=label=='B',sink=online_witnesses[label].append)
                            for label in ('strict','B')} if arm=='candidate' else {}
            if arm=='candidate':
                class ActualCandidateDraws(list):
                    def append(self,row):
                        index = len(self)
                        super().append(row)
                        if index>=len(draws['reference']):
                            raise ScientificGateFailure('Unexpected additional candidate draw')
                        ar = {k:v for k,v in draws['reference'][index].items() if k!='actual_voltage_V'}
                        br = {k:v for k,v in row.items() if k!='actual_voltage_V'}
                        for gate in online_gates.values():
                            gate.compare_draw(ar,br)
                        if online_gates['B'].worst_status!='PASS':
                            raise ScientificGateFailure('First actual B event/RNG/rate/history divergence at '+str(row['bin_start_seconds']))
                draws[arm] = ActualCandidateDraws()
            monitor = monitors[arm]
            copy_seconds = 0.
            counter = 0
            def observe(t,y,bundle):
                nonlocal counter,copy_seconds
                guard.require()
                monitor.observe(t,y,bundle)
                if t not in critical:
                    return
                tick = time.perf_counter()
                values = _snapshot(runner,t,y,bundle,protocol['auxiliary_cone_indices'],reduced=arm=='candidate')
                file = 'point_%04d.npz'%counter
                counter += 1
                np.savez(arm_dir/file,**values)
                receipts[arm][(float(t),bool(bundle.delay_left_limit))] = dict(file=file,sha256=digest(arm_dir/file))
                if arm=='candidate' and runner._active_cursor is not None and any(
                        p.start<=t<=p.stop for p in runner._active_cursor.packets):
                    packet = runner._active_cursor.packet_at(t,left=bundle.delay_left_limit)
                    direct = values['native_port:release'][runner.selection.cone_rows]
                    allowance = budgets['B']['native_port:release'][runner.selection.cone_rows]
                    bridge_checks.append(dict(t=t,left=bool(bundle.delay_left_limit),**compare_vector(
                        direct,packet.release_at(t),allowance,gate='same_trajectory_CPU_GPU_release')))
                copy_seconds += time.perf_counter()-tick
            def record(t,point,evidence,st):
                observed_delay['candidate'][float(t)] = _selected_delay(st.delay_states[native_id],t,selected_h_device)
                if demo is not None:
                    demo.record(t,point,evidence,st)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            advance_wall = 0.
            advance_start = None
            failure = None
            try:
                with actual_draw_capture(ref.compiled,expected,draws[arm],costs[arm]):
                    for end in (seam,stop):
                        samples = times[(times>=state.time_seconds)&(times<=end)]
                        advance_start = time.perf_counter()
                        if arm=='reference':
                            state,recorded = runner.advance(state,channels,end,sample_times=samples,
                                recording_plan=recording,state_observer=observe)
                            _collect_reference_recording(recorded,native_id,samples,
                                observed_delay['reference'],actual_events['reference'])
                        else:
                            state = runner.advance(state,channels,end,sample_times=samples,
                                state_observer=observe,record=record,
                                event_sink=lambda cid,e:actual_events['candidate'].append(dict(component=cid,**e)))
                        torch.cuda.synchronize()
                        advance_wall += time.perf_counter()-advance_start
                        advance_start = None
                        if arm=='candidate':
                            if end==seam:
                                phase('save_restore_candidate_with_pending_history')
                                checkpoint_start = time.perf_counter()
                                receipt = save_complete(runner,state,output/'candidate_checkpoint')
                                def plain(v):
                                    if hasattr(v,'detach'):
                                        return _cpu(v)
                                    if isinstance(v,dict):
                                        return {k:plain(x) for k,x in v.items()}
                                    if isinstance(v,(list,tuple)):
                                        return [plain(x) for x in v]
                                    return v
                                before = canonical_hash(plain(_payload(runner,state)))
                                restored = load_complete(runner,output/'candidate_checkpoint',expected_manifest_hash=receipt['manifest_hash'])
                                after = canonical_hash(plain(_payload(runner,restored)))
                                receipt.update(bitwise_payload_equal=before==after,
                                    unconsumed_segments=len(restored.cursor.segments),
                                    resume_uses_restored_state=True,seconds=time.perf_counter()-checkpoint_start)
                                write_json(output/'checkpoint_restore.json',receipt)
                                if before!=after:
                                    raise ContractError('Restored complete reduced payload differs')
                                state = restored
                torch.cuda.synchronize()
                wall = time.perf_counter()-tick
            except Exception as exc:
                failure = exc
                torch.cuda.synchronize()
                if advance_start is not None:
                    advance_wall += time.perf_counter()-advance_start
                wall = time.perf_counter()-tick
            finally:
                write_json(output/(arm+'_queue_times.json'),dict(
                    queue_time_evidence(state.delay_states,time_seconds=state.time_seconds),
                    state_scope='last_successfully_returned_state' if failure is not None else 'completed_arm',
                    arm_completed=failure is None))
                write_json(output/(arm+'_actual_draws.json'),draws[arm])
                write_json(output/(arm+'_actual_events.json'),actual_events[arm])
                if online_gates:
                    write_json(output/'online_draw_gates.json',{label:dict(status=g.worst_status,
                        issues=[r.__dict__ for r in g.issues]) for label,g in online_gates.items()})
                    for label,rows in online_witnesses.items():
                        write_json(output/(label+'_online_draw_witnesses.json'),rows)
            report = monitor.report(stop-start)
            write_json(output/(arm+'_monitor.json'),report)
            np.savez(output/(arm+'_monitor_cells.npz'),**{
                str(i)+'|'+key:_cpu(row[key]) for i,row in enumerate(monitor.rows)
                for key in ('lo','hi','rhs','defect','den')})
            write_json(output/(arm+'_points.json'),[dict(t=t,left=left,**row) for (t,left),row in receipts[arm].items()])
            measurements[arm] = dict(full_neural_advance_and_evidence_seconds=wall,
                neural_advance_seconds=advance_wall,
                critical_snapshot_and_hash_seconds=copy_seconds,monitor_observations=monitor.count,
                actual_draws=len(draws[arm]),draw_capture=costs[arm],
                integration=dict(runner.split_metrics) if arm=='reference' else dict(runner.metrics),
                transfer={} if arm=='reference' else dict(runner.bridge.metrics))
            write_json(output/(arm+'_measurement.json'),measurements[arm])
            if failure is not None:
                result.update(partial_measurements=measurements,checkpoint_restored=(output/'checkpoint_restore.json').exists(),
                              bridge_checks_before_stop=bridge_checks)
                raise failure
            if arm=='candidate':
                candidate = state
            else:
                source = state
            if report['domain_failures']:
                raise ContractError(arm+' complete monitor domain failure')
        phase('compare_frozen_budgets_without_tuning')
        verdicts = {'strict':[],'B':[]}
        if set(receipts['reference'])!=set(receipts['candidate']):
            raise ContractError('Exact common-point/input-side coverage differs')
        for point,ar in receipts['reference'].items():
            br = receipts['candidate'][point]
            with np.load(output/'reference'/ar['file'],allow_pickle=False) as av,np.load(output/'candidate'/br['file'],allow_pickle=False) as bv:
                if set(av.files)!=set(bv.files):
                    raise ContractError('Retained state/output coverage differs')
                for label in verdicts:
                    for key,radius in budgets[label].items():
                        row = compare_vector(av[key],bv[key],radius,gate=key)
                        verdicts[label].append(dict(t=point[0],left=point[1],**row))
                    for key in unregistered:
                        verdicts[label].append(dict(t=point[0],left=point[1],**_compare_ribbon(context,av,bv,budgets[label],key)))
        delay_checks = []
        if set(observed_delay['reference'])!=set(observed_delay['candidate']):
            raise ContractError('HC observation request coverage differs')
        np.savez(output/'delayed_HC_observations.npz',times=np.array(sorted(observed_delay['reference'])),
            reference=np.stack([observed_delay['reference'][t] for t in sorted(observed_delay['reference'])]),
            candidate=np.stack([observed_delay['candidate'][t] for t in sorted(observed_delay['reference'])]))
        for t,value in observed_delay['reference'].items():
            delay_checks.append(dict(t=t,**compare_vector(value,observed_delay['candidate'][t],
                budgets['B']['native:Vh'][selected_h],gate='original_2.82ms_delayed_HC_observation')))
        draw_verdicts = {}
        for label,pro in (('strict',False),('B',True)):
            witnesses = []
            c = _draw_gate(expected,context,pro=pro,sink=witnesses.append)
            if len(draws['reference'])!=len(expected) or len(draws['candidate'])!=len(expected):
                c._issue('FAIL','draws','Actual original draw count differs from frozen schedule')
            for ar,br in zip(draws['reference'],draws['candidate']):
                c.compare_draw({k:v for k,v in ar.items() if k!='actual_voltage_V'},
                               {k:v for k,v in br.items() if k!='actual_voltage_V'})
                if c.worst_status!='PASS':
                    break
            draw_verdicts[label] = dict(status=c.worst_status,issues=[r.__dict__ for r in c.issues])
            write_json(output/(label+'_draw_witnesses.json'),witnesses)
        write_json(output/'numeric_gates.json',verdicts)
        write_json(output/'bridge_gates.json',bridge_checks)
        write_json(output/'delay_gates.json',delay_checks)
        write_json(output/'draw_gates.json',draw_verdicts)
        queue_comparison = compare_queue_times(
            queue_time_evidence(source.delay_states,time_seconds=source.time_seconds),
            queue_time_evidence(candidate.delay_states,time_seconds=candidate.time_seconds))
        write_json(output/'delay_queue_comparison.json',queue_comparison)
        exact_aux = dict(component_ticks=source.component_tick_indices==candidate.component_tick_indices,
            sample_index=source.sample_index==candidate.sample_index,
            rng_history=all(canonical_hash({k:v for k,v in source.spike_states[cid].items()
                if k in ('rng_state','history','last_bin_counts')})==canonical_hash({k:v for k,v in candidate.spike_states[cid].items()
                if k in ('rng_state','history','last_bin_counts')}) for cid in source.spike_states),
            delay_times=queue_comparison['status']=='PASS',
            consumed_event_identities=sorted((e['component'],e['cell_index']) for e in actual_events['reference'])==
                sorted((e['component'],e['cell_index']) for e in actual_events['candidate']),
            pending_event_identities=all([e['cell_index'] for e in source.spike_states[cid]['pending_events']]==
                [e['cell_index'] for e in candidate.spike_states[cid]['pending_events']] for cid in source.spike_states))
        write_json(output/'final_auxiliary_gates.json',exact_aux)
        statuses = {label:'PASS' if all(r['status']=='PASS' for r in rows) and draw_verdicts[label]['status']=='PASS'
                    else 'FAIL' for label,rows in verdicts.items()}
        passed = (statuses['B']=='PASS' and bool(bridge_checks) and all(r['status']=='PASS' for r in bridge_checks+delay_checks)
                  and all(exact_aux.values()))
        result.update(status='PASS_BOUNDED_CANDIDATE' if passed else 'FAIL_CANDIDATE_RETAINED',
            numeric_status=statuses,draw_status=draw_verdicts,
            full_window_seconds=stop-start,environment=environment,measurements=measurements,
            reference_over_candidate_wall_ratio=measurements['reference']['neural_advance_seconds']/
                measurements['candidate']['neural_advance_seconds'],
            completed_phases=['real_GPU_native','original_CPU_inner','input_boundaries','full_monitors',
                              'actual_draws','same_trajectory_bridge','delayed_observation','checkpoint_resume'],
            production_stop_preserved=digest(ROOT/protocol['production_stop']['path'])==protocol['production_stop']['sha256'])
    except ScientificGateFailure as exc:
        result.update(status='FAIL_CANDIDATE_RETAINED',error=str(exc),
                      stop_reason='first_actual_event_gate_divergence',full_100ms_comparison_completed=False,
                      traceback=traceback.format_exc())
        print(result['traceback'],flush=True)
    except Exception as exc:
        result.update(status='INCOMPLETE_EXECUTION',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        print(result['traceback'],flush=True)
    finally:
        result.update(total_worker_seconds=time.time()-begun,resource=guard.last)
        if protocol.get('demo_recording') and result.get('completed_phases'):
            try:
                result['desktop_replay'] = demo.finish(result,actual_events['candidate'])
            except Exception as exc:
                result['desktop_replay_error'] = type(exc).__name__+': '+str(exc)
                print(traceback.format_exc(),flush=True)
        write_json(output/'outcome.json',result)
    return 0 if result['status'] in ('PASS_BOUNDED_CANDIDATE','FAIL_CANDIDATE_RETAINED') else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--demo-recording',action='store_true',help='Prepare selected desktop replay recording; no model change')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-only',action='store_true')
    group.add_argument('--execute',action='store_true')
    group.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = _project_path(args.output)
    if args.prepare_only:
        output.mkdir(parents=True,exist_ok=False)
        protocol = build_protocol()
        if args.demo_recording:
            protocol['demo_recording'] = 'retipath-reduced-demo-1'
            module = ROOT/'tools/cone_hc_demo_recording.py'
            protocol['implementation_sha256'][module.relative_to(ROOT).as_posix()] = digest(module)
            protocol.pop('protocol_sha256')
            protocol['protocol_sha256'] = canonical(protocol)
        validate_protocol(protocol)
        write_json(output/'protocol.json',protocol)
        print('Prepared CPU-only protocol:',output/'protocol.json')
        return 0
    if args.worker:
        return run_worker(output)
    protocol = json.loads((output/'protocol.json').read_bytes())
    validate_protocol(protocol)
    if (output/'worker_started.json').exists():
        raise ValueError('This sealed execution was already attempted; retain its evidence')
    with (output/'worker.log').open('xb') as log:
        try:
            completed = subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),
                '--worker','--output',str(output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                timeout=protocol['limits']['max_wall_seconds'],
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except subprocess.TimeoutExpired:
            write_json(output/'supervisor_timeout.json',dict(status='STOPPED',reason='whole_worker_wall_limit'))
            return 2
    print('Worker exit:',completed.returncode,'Evidence:',output/'outcome.json')
    return completed.returncode


if __name__=='__main__':
    raise SystemExit(main())

