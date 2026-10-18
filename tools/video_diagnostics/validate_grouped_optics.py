"""Two existing tiles x two frames; unchanged MATLAB exporter, exact comparisons."""
from pathlib import Path
import datetime,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.io import loadmat
from retina_platform.grouped_disk_capture import group_missing_jobs,read_group_frame
from retina_platform.disk_frontend import read_disk_capture


def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=ROOT/'reports'/('grouped_optics_validation_'+stamp);output.mkdir()
    def save(name,value):(output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    (ROOT/'reports/active_grouped_validation.json').write_text(
        json.dumps({'directory':str(output),'worker_pid':os.getpid()})+'\n',encoding='utf8')
    old=ROOT/'reports/notebook_transient_video05s_diag_20261018T100402Z/input_stream_shards'
    old_jobs=json.loads(next(old.glob('batch_*.json')).read_bytes())
    entries=json.loads((ROOT/'reports/notebook_video05s_20261018T063214Z/background/reference_input_shards/exposure_000000.json').read_bytes())['tile_captures']
    indices=[0,min(range(len(entries)),key=lambda i:sum(x*x for x in entries[i]['tile_index']))]
    cfg=json.loads((ROOT/'configurations/atlas_human_borrowed_disk3mm_floor_input_v2.json').read_bytes())
    plan=json.loads((ROOT/cfg['input_plan']['path']).read_bytes())
    tiles={tuple(t['tile_index']):t for t in plan['tiles']}
    sources,labels,baselines={}, {}, {}
    jobs=[];profile=scope=None
    for name,index in zip(('edge','central'),indices):
        first=Path(entries[index]['directory'])
        source_configs=[first/'export.json',Path(old_jobs[index]['configuration'])]
        for frame,path in enumerate(source_configs):
            c=json.loads(path.read_bytes());profile=c['profile'];scope=c['background_signature']
            cache=path.parent;capture=cache/'capture.mat'
            if not capture.exists():capture=Path(old_jobs[index]['output'])
            raw,meta=read_disk_capture(capture,c['request_signature'])
            baselines[(name,frame)]=(raw,meta)
            jobs.append({'configuration':str(path),'output':'unused'})
            labels[str(path.resolve())]={'tile':tiles[tuple(entries[index]['tile_index'])],
                'exposure_index':frame,'video_time_seconds':frame*.0334,'decoded_frame_index':frame,'gaze_deg':[0.,0.]}
            sources[(name,frame)]=str(path)
    grouped,views,summary=group_missing_jobs(ROOT,profile,jobs,labels,scope)
    if len(grouped)!=2:raise ValueError('Fresh two-tile validation requires exactly two new grouped exports')
    save('grouped_requests.json',summary)
    cases=[{'label':'warmup','configuration':sources[('edge',0)],'output':str(output/'warmup.mat')}]
    for name in ('edge','central'):
        for frame in (0,1):
            cases.append({'label':name+'_single_'+str(frame),'configuration':sources[(name,frame)],
                          'output':str(output/(name+'_single_'+str(frame)+'.mat'))})
        view=views[str(Path(sources[(name,0)]).parent.resolve())]
        job=next(j for j in grouped if str(Path(j['configuration']).parent)==view['directory'])
        cases.append(dict(job,label=name+'_group'))
    cases=[{'label':c['label'],'configuration':c['configuration'],'output':c['output']} for c in cases]
    save('cases.json',cases)
    matlab="""function benchmark_grouped(casesPath,timingsPath)
batch=jsondecode(fileread(casesPath));rows=repmat(struct('label','','seconds',0),1,numel(batch));
for k=1:numel(batch)
    if iscell(batch),job=batch{k};else,job=batch(k);end
    started=tic;export_disk_mixed_reference(job.configuration,job.output);
    rows(k).label=job.label;rows(k).seconds=toc(started);
    marker=fopen([job.output '.ready'],'w');assert(marker>0);fclose(marker);
end
f=fopen(timingsPath,'w');assert(f>0);fprintf(f,'%s',jsonencode(rows));fclose(f);
end
"""
    (output/'benchmark_grouped.m').write_text(matlab,encoding='utf8')
    quote=lambda p:str(p).replace("'","''")
    expression="addpath('"+quote(ROOT/'tools')+"');addpath('"+quote(output)+"');benchmark_grouped('"+quote(output/'cases.json')+"','"+quote(output/'timings.json')+"');"
    preferences=output/'matlab_preferences';preferences.mkdir()
    temporary=output/'matlab_temp';temporary.mkdir()
    env=dict(os.environ,MATLAB_PREFDIR=str(preferences),TEMP=str(temporary),TMP=str(temporary))
    save('result.json',{'status':'MATLAB_RUNNING','cases':len(cases),'tile_indices':[entries[i]['tile_index'] for i in indices]})
    started=time.perf_counter()
    with (output/'matlab_stdout.log').open('wb') as stdout,(output/'matlab_stderr.log').open('wb') as stderr:
        completed=subprocess.run([profile['matlab_executable'],'-batch',expression],cwd=ROOT,env=env,
            stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=300)
    if completed.returncode:
        save('result.json',{'status':'MATLAB_FAILED','returncode':completed.returncode})
        raise RuntimeError('Grouped equivalence MATLAB failed: '+str(output))
    timings={r['label']:r['seconds'] for r in json.loads((output/'timings.json').read_bytes())}
    comparisons=[];speeds=[]
    for name in ('edge','central'):
        for frame in (0,1):
            view=views[str(Path(sources[(name,frame)]).parent.resolve())]
            raw,meta=read_group_frame(view);baseline,old_meta=baselines[(name,frame)]
            control,_=read_disk_capture(output/(name+'_single_'+str(frame)+'.mat'),old_meta['request_signature'])
            fields={}
            for key in ('expected_counts','background_counts','q_ref_LMS','window_edges_seconds','diameters_um','fixed_radiance_gain_numeric'):
                a,b,c=np.asarray(baseline[key]),np.asarray(raw[key]),np.asarray(control[key])
                fields[key]={'group_equal':a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),
                             'single_control_equal':a.shape==c.shape and a.dtype==c.dtype and a.tobytes()==c.tobytes(),
                             'maximum_absolute_difference':float(np.max(np.abs(a-b)))}
            fixed=('global_reference_definition_sha256','reference_energy_spectrum_hex','fixed_radiance_gain',
                   'wave_nm','OD_LMS','peak_efficiency_LMS','lens_density','macular_density','pupil_diameter_mm')
            metadata_equal=all(meta[k]==old_meta[k] for k in fixed)
            comparisons.append({'tile':name,'frame':frame,'fields':fields,'fixed_metadata_equal':metadata_equal})
            if not metadata_equal or not all(v['group_equal'] and v['single_control_equal'] for v in fields.values()):
                save('result.json',{'status':'FAIL_EXACT_EQUIVALENCE','comparisons':comparisons,'timings':timings})
                raise RuntimeError('Exact grouped optical equivalence failed; no tolerance relaxation')
        singles=timings[name+'_single_0']+timings[name+'_single_1'];group=timings[name+'_group']
        speeds.append({'tile':name,'two_single_seconds':singles,'two_frame_group_seconds':group,'measured_speedup':singles/group})
    result={'status':'PASS_BITWISE_EQUIVALENCE','comparisons':comparisons,'timings_seconds':timings,
            'measured_two_frame_speedups':speeds,'MATLAB_process_wall_seconds':time.perf_counter()-started,
            'scope':'Two existing tiles x two existing frames only; original exporter unchanged; no full-graph benchmark',
            'original_video_axis':'5000 original 100-us bins unchanged; group axis labels independent captures only'}
    save('result.json',result);print(json.dumps({'directory':str(output),'status':result['status'],'speedups':speeds}),flush=True)


if __name__=='__main__':main()
