"""Source-identical movie sampling and explicitly unscored runtime feasibility."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
import hashlib
import time
import numpy as np
from scipy.ndimage import map_coordinates
from retina_sim.contracts import Quantity,canonical_hash


def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def source_movie_channels(compiled,channels,frames,case,origin,duration,*,inside_only=False,verify_first_frame=False):
    off=np.array([.01451,.005123,.02612]);gain=np.array([.9855,.9949,.9739]);power=np.array([2.3122,2.2752,2.2818])
    LMS=np.array([[2.74,3.4,1.34],[1.21*1.06,1.21*3.58,1.21*2.07],[.212,8.28,285.]])
    ROD=np.array([.2126,.7152,.0722]);gray=off+gain*(75/256.)**power
    cone_xy=compiled.populations['cones'].cells.rf_position.values;rod_xy=compiled.populations['rod'].cells.rf_position.values
    cone_type=np.array([{'L':0,'M':1,'S':2}[x] for x in compiled.populations['cones'].cells.subtype])
    count_index=np.arange(len(cone_type));cone_reference=(LMS@gray)[cone_type];rod_reference=float(ROD@gray)
    relative_offset=np.array([case['source_RF']['parameters'][k] for k in ('xoff','yoff')])*255/256
    target=np.asarray(case['target_RF_mm']);maps={k:127.5+relative_offset+(xy-target)/.223*256/4.6 for k,xy in [('native_light',cone_xy),('rod_light',rod_xy)]}
    outside={k:np.any((v<0)|(v>255),axis=1) for k,v in maps.items()};rows={k:OrderedDict() for k in maps}
    hashes={k:{} for k in maps};selected={k:inside_only for k in maps};proof={}
    def input_row(name,index):
        cache=rows[name]
        if index not in cache:
            linear=off+gain*(np.asarray(frames[index],dtype=float)/256.)**power;xy=maps[name]
            def evaluate(only_inside):
                if only_inside:
                    sample=np.broadcast_to(gray,(len(xy),3)).copy();keep=~outside[name];points=xy[keep]
                    sample[keep]=np.column_stack([map_coordinates(linear[:,:,j],[points[:,1],points[:,0]],order=1,mode='nearest',prefilter=False) for j in range(3)])
                else:
                    sample=np.column_stack([map_coordinates(linear[:,:,j],[xy[:,1],xy[:,0]],order=1,mode='nearest',prefilter=False) for j in range(3)])
                    sample[outside[name]]=gray
                return 100*(sample@LMS.T)[count_index,cone_type]/cone_reference if name=='native_light' else 5*(sample@ROD)/rod_reference
            if verify_first_frame and index==0:
                original=evaluate(False);optimized=evaluate(True);equal=np.array_equal(original,optimized)
                selected[name]=equal;value=optimized if equal else original
                proof[name]=dict(bitwise_equal=equal,original_sha256=array_sha(original),optimized_sha256=array_sha(optimized),
                    full_input_cells=len(xy),inside_display_cells=int((~outside[name]).sum()),fallback_original=not equal)
            else:value=evaluate(selected[name])
            digest=array_sha(value)
            if str(index) in hashes[name] and hashes[name][str(index)]!=digest:
                raise RuntimeError('Resumed source input row differs from its frozen original hash')
            cache[index]=value;hashes[name][str(index)]=digest
            while len(cache)>2:cache.popitem(last=False)
        return cache[index]
    frame_boundaries=np.asarray([origin+j/150 for j in range(1,int(round(duration*150)))])
    movie={}
    for name,template in channels.items():
        if name not in maps:movie[name]=template;continue
        q=template.at(origin)
        def evaluator(t,name=name,q=q):
            # Use the actual calendar; an epsilon must not move a left limit into the next frame.
            index=int(np.searchsorted(frame_boundaries,t,side='right'))
            return Quantity(input_row(name,index),q.unit,q.meaning,q.evidence_tag)
        movie[name]=replace(template,evaluator=evaluator,interpolation='zero_order_hold',
            breakpoints_seconds=tuple(frame_boundaries),
            input_contract=dict(template.input_contract,declared_constant_background=False,source_case=case['case_id'],
                input_class='source_CRT_relative_display_approximation',calibrated_retinal_flux=False,
                frame_time_contract='absolute_150Hz_left_closed_right_open_v1',
                outside_display='fixed mean-gray RGB75; not fitted',first_live_frame_zero_based=751,physical_fps=150))
    if verify_first_frame:
        for name in maps:input_row(name,0)
    return movie,hashes,proof


def runtime_only_pilot(compiled,runner,state,channels,frames,case,target,duration,directory,save,update,reader_factory,phase_hook=None):
    from retina_sim.retained_background import close_retained_background
    from retina_sim.background_advance import choose_observed_runner
    from retina_sim.recording import RecordingPlan
    if duration!=.025 or case['case_id']!='cell_67_14':raise ValueError('Only the authorized fixed25ms/67#14 runtime pilot is allowed')
    if phase_hook is not None:phase_hook('PREPARING_DERIVED_STATE',{})
    before=canonical_hash(state);seed_started=time.monotonic()
    pilot,closure=close_retained_background(runner,deepcopy(state),channels)
    closure_seconds=time.monotonic()-seed_started;origin=pilot.time_seconds
    movie,row_hashes,proof=source_movie_channels(compiled,channels,frames,case,origin,duration,verify_first_frame=True)
    save(directory/'input_sampling_equivalence.json',dict(source_frame_zero_based=751,proof=proof,
        original_sampling_definition_unchanged=True,full_LMS_and_rod_matrix_products_retained=True))
    cid,index,_=target;reader=reader_factory(compiled,runner,cid,index);transport,procedure=choose_observed_runner(runner)
    samples={0:reader(origin,runner._flat(pilot))};started=time.monotonic();cpu=time.process_time();old_metrics=dict(runner.integration_metrics)
    if phase_hook is not None:phase_hook('ADVANCE_BEGIN',{'advance_started_monotonic':started})
    last_progress=[started];final_state_finite=None
    class PilotWallLimit(RuntimeError):pass
    def observe(t,y,bundle):
        tick=(t-origin)/.0001
        if abs(tick-round(tick))<1e-7:samples[int(round(tick))]=reader(t,y)
        if phase_hook is not None:
            phase_hook('OBSERVATION_MEMORY',{'observed_neural_seconds':float(t-origin)})
        if phase_hook is not None and time.monotonic()-last_progress[0]>=.5:
            phase_hook('OBSERVATION',{'last_observed_neural_seconds':max(samples)*.0001})
            last_progress[0]=time.monotonic()
        if time.monotonic()-started>60.:raise PilotWallLimit('60s runtime-only budget reached at an original observation point')
    status='RUNTIME_ONLY_PILOT_COMPLETE_NOT_FIT_ELIGIBLE';reason=None
    try:
        end=origin+duration;times=origin+np.arange(251)*.0001;times[-1]=end
        result,_=transport.advance(pilot,movie,end,pilot.solver_config,sample_times=times.tolist(),
            recording_plan=RecordingPlan(ports=(),states=(),cell_ids=(),max_bytes=1024**2),state_observer=observe)
        samples[250]=reader(end,runner._flat(result))
        final_state_finite=bool(np.all(np.isfinite(runner._flat(result))))
    except PilotWallLimit as exc:
        status='RUNTIME_ONLY_PILOT_PARTIAL_NOT_FIT_ELIGIBLE';reason=str(exc)
    finally:
        metric_keys=('solver_calls','nfev','njev','nlu','integration_wall_seconds')+tuple(
            k for k in runner.integration_metrics if k.startswith('solver_lifetime_gc_') or k=='solver_result_copy_wall_seconds')
        metrics={k:runner.integration_metrics.get(k,0)-old_metrics.get(k,0) for k in metric_keys}
        runner.integration_metrics=old_metrics
        if phase_hook is not None:phase_hook('ADVANCE_END',dict(advance_finished_monotonic=time.monotonic(),
            last_observed_neural_seconds=max(samples)*.0001,final_state_finite=final_state_finite,
            integration_metrics=metrics))
    ticks=np.array(sorted(samples));values=np.asarray([samples[int(k)] for k in ticks])
    np.savez_compressed(directory/'runtime_pilot_trace_NOT_FIT_ELIGIBLE.npz',relative_time=ticks*.0001,
        gE_S=values[:,0],gI_S=values[:,1],V_native_V=values[:,2])
    unchanged=canonical_hash(state)==before
    report=dict(status=status,fit_eligible=False,background_convergence_performed=False,NLL_or_fit_performed=False,
        case_id=case['case_id'],target_component=cid,target_cell_id=case['target_cell_id'],
        actual_observed_neural_seconds=float(ticks[-1]*.0001),requested_neural_seconds=duration,
        wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
        derived_fixedpoint_seconds=closure_seconds,derived_fixedpoint_report=closure,
        original_main_state_reference_RNG_fingerprint=before,main_state_reference_RNG_unchanged=unchanged,
        input_sampling_equivalence=proof,input_row_sha256=row_hashes,
        integration_metrics=metrics,execution_procedure=procedure,stop_reason=reason,
        trace_file='runtime_pilot_trace_NOT_FIT_ELIGIBLE.npz',final_state_finite=final_state_finite,
        interpretation='Compute feasibility only; all original background criteria remain required before fitting')
    save(directory/'runtime_pilot_NOT_FIT_ELIGIBLE.json',report)
    if not unchanged:raise RuntimeError('Runtime pilot changed the real main initial state')
    update(runtime_only_pilot=report)
    return bool(proof) and all(p['bitwise_equal'] for p in proof.values()),pilot
