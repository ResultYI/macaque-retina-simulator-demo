"""One bounded old/new input check; no circuit, background preparation or fitting."""
from pathlib import Path
import sys,json,hashlib,gc,tracemalloc
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from retina_sim.atlas_runtime import _sampled_native_light_channel,q,TAG
    from retina_sim.spatial import InputChannel
    from retina_sim.runtime_profiles import LightContext
    from retina_sim.background import rebase_stimulus_channels
    from retina_sim.contracts import ContractError
    out=root/'reports/native_light_input_check_20261012.json'
    if out.exists():raise FileExistsError('The bounded input validation is already complete')
    baseline=json.loads((root/'reports/native_light_input_baseline_20261012.json').read_bytes())
    duration=.00404;light=LightContext(native_background_td=137.);rf_x=np.array([-.17,0.,.004,.35,1.2,3.,7.1])
    grid=np.arange(int(round(duration/.0001))+1)*.0001
    rates=light.native_background_td*(1+.1*np.sin(2*np.pi*80*grid[:,None]+rf_x[None,:]/.223))
    def old_evaluate(t):
        position=np.clip(t/.0001,0,len(grid)-1);i=min(int(np.floor(position)),len(grid)-1)
        fraction=position-i
        return q((1-fraction)*rates[i]+fraction*rates[min(i+1,len(grid)-1)],'td','native_retinal_illuminance')
    old=InputChannel(0,duration,old_evaluate,'td','native_retinal_illuminance','linear',TAG,
        {'source':'declared_short_synthetic_stimulus','background_context_hash':light.background_context_hash,'not_physically_calibrated':True})
    new=_sampled_native_light_channel(duration,light,rf_x)
    for field in ('start_seconds','stop_seconds','unit','meaning','interpolation','evidence_tag','input_contract','breakpoints_seconds'):
        assert getattr(old,field)==getattr(new,field)
    def cache(channel):
        return dict(zip(channel.evaluator.__code__.co_freevars,(c.cell_contents for c in channel.evaluator.__closure__)))
    times=np.unique(np.concatenate((grid,(grid[:-1]+grid[1:])/2,np.nextafter(grid,np.inf),
        np.nextafter(grid,-np.inf),[duration,-5e-13,duration+5e-13])))
    times=times[(times>=-1e-12)&(times<=duration+1e-12)]
    compared=0;maximum_error=0.
    # Reverse and repeated visits exercise cache replacement, not only monotonic playback.
    for t in np.concatenate((times,times[::-1],times[::9])):
        t=float(t);a=old.at(t);b=new.at(t)
        assert np.array_equal(a.values,b.values),(t,a.values,b.values)
        maximum_error=max(maximum_error,float(np.max(np.abs(a.values-b.values))))
        expected=min(int(np.floor(np.clip(t/.0001,0,len(grid)-1))),len(grid)-1)
        state=cache(new);assert state['cached_index']==expected and state['cached_rows'].shape==(2,len(rf_x))
        assert np.array_equal(state['cached_rows'],rates[[expected,min(expected+1,len(grid)-1)]])
        compared+=1
    for t in [-2e-12,duration+2e-12]:
        for channel in (old,new):
            try:channel.at(t)
            except ContractError:pass
            else:raise AssertionError('Out-of-interval input accepted')
    t=.00102;new.at(t);rows=cache(new)['cached_rows'];new.at(.00108)
    assert cache(new)['cached_rows'] is rows
    background=InputChannel(0.,1.,lambda t:q(np.full(len(rf_x),light.native_background_td),'td','native_retinal_illuminance'),
        'td','native_retinal_illuminance','analytic',TAG,{'source':'validation-only declared uniform background'})
    origin=.1;shifted_old=rebase_stimulus_channels({'native_light':old},{'native_light':background},
        original_origin_seconds=0.,stimulus_origin_seconds=origin,stop_seconds=origin+duration)['native_light']
    shifted_new=rebase_stimulus_channels({'native_light':new},{'native_light':background},
        original_origin_seconds=0.,stimulus_origin_seconds=origin,stop_seconds=origin+duration)['native_light']
    boundaries=[origin,origin+.001,origin+.003,origin+duration]
    shifted_checks=0
    for t in [0.,origin-.0001]+[x for b in boundaries for x in [np.nextafter(b,-np.inf),b,np.nextafter(b,np.inf)]]:
        assert np.array_equal(shifted_old.at(t).values,shifted_new.at(t).values);shifted_checks+=1
    memory=[];memory_x=np.linspace(-.17,7.1,1465)
    for horizon in [11.,1100.]:
        gc.collect();tracemalloc.start();channel=_sampled_native_light_channel(horizon,light,memory_x)
        for t in [0.,.00005,horizon/2,horizon]:channel.at(t)
        current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        state=cache(channel);retained_rows=state['cached_rows'].nbytes
        assert retained_rows==2*1465*8 and state['phase'].nbytes==1465*8 and peak<1024*1024
        memory.append({'duration_seconds':horizon,'cells':1465,'retained_sample_rows':2,
            'retained_rows_bytes':retained_rows,'phase_bytes':state['phase'].nbytes,
            'traced_current_bytes':current,'traced_peak_bytes':peak,
            'previous_full_array_bytes_estimate':(int(round(horizon/.0001))+1)*1465*8})
        del channel,state
    after={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in baseline['protected_files_sha256']}
    assert after==baseline['protected_files_sha256']
    report={'status':'PASS','test_scope':'one seven-cell input fixture; no model/circuit/background runs',
        'value_comparisons':compared,'maximum_absolute_error_td':maximum_error,'bitwise_values_equal':True,
        'source_grid_step_seconds':.0001,'grid_rows':len(grid),'duration_seconds':duration,
        'grid_index_checks':compared,'backward_and_repeated_queries':True,'same_index_cache_reused':True,
        'input_contract_and_bounds_unchanged':True,'background_rebase_and_time_boundaries_checked':shifted_checks,
        'memory':memory,'memory_scope':'tracemalloc Python/NumPy allocations in channel creation and four queries; not full-model RSS',
        'protected_files_unchanged':after,'previous_atlas_runtime_sha256':baseline['previous_atlas_runtime_sha256'],
        'current_atlas_runtime_sha256':hashlib.sha256((root/'retina_sim/atlas_runtime.py').read_bytes()).hexdigest(),
        'full_domain_generation_or_model_runs':0}
    out.write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:report[k] for k in ['status','value_comparisons','maximum_absolute_error_td','background_rebase_and_time_boundaries_checked','memory']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
