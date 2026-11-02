function capture_gin69_formal_optics_v1(manifestPath,runPath,prefixSeconds)
% Reuse the frozen exporter and cache; commit each exact-frame batch atomically.
m=jsondecode(fileread(manifestPath));
assert(ismember(prefixSeconds,[30 45]) && m.frame_count==6750);
record(fullfile(runPath,'MATLAB_SELF.json'),struct('pid',feature('getpid')));
needed=[];
for k=1:round(prefixSeconds*150)
    if iscell(m.frames),frame=m.frames{k};else,frame=m.frames(k);end
    if isfield(frame,'job') && ~isempty(frame.job)
        needed(end+1)=frame.job; %#ok<AGROW>
    end
end
needed=unique(needed); started=tic; completed=0; newCount=0;
for k=1:numel(needed)
    assert(~isfile(fullfile(runPath,'REQUEST_STOP')),'Owned explicit stop');
    job=m.jobs(needed(k)+1); folder=fileparts(job.output);
    receipt=fullfile(folder,'COMPLETE.json');
    if isfile(receipt)
        r=jsondecode(fileread(receipt));
        assert(isfile(job.output) && strcmp(r.config_sha256,job.config_sha256));
        completed=completed+1; continue
    end
    assert(~isfile(job.output),'Unreceipted capture requires explicit recovery');
    partial=fullfile(folder,'CAPTURE.partial.mat');
    assert(~isfile(partial),'Previous incomplete capture retained');
    record(fullfile(runPath,'PROGRESS.json'),struct('stage','capturing','job',job.job, ...
        'completed_jobs',completed,'required_jobs',numel(needed),'prefix_seconds',prefixSeconds, ...
        'elapsed_seconds',toc(started),'utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss.SSS''Z'''))));
    phase=tic; export_gin_crt_tile_v1(job.config,partial); seconds=toc(phase);
    movefile(partial,job.output);
    record(receipt,struct('status','COMPLETE','job',job.job,'frames',job.count, ...
        'config_sha256',job.config_sha256,'source_bank_sha256',job.bank_sha256, ...
        'capture_seconds',seconds,'same_persistent_exporter',true));
    completed=completed+1;newCount=newCount+job.count;
    fprintf('FORMAL_OPTICS_COMMITTED job=%d frames=%d wall=%.3f required=%d/%d\n',job.job,job.count,seconds,completed,numel(needed));
end
record(fullfile(runPath,'COMPLETE.json'),struct('status','COMPLETE_PREFIX_OPTICS', ...
    'prefix_seconds',prefixSeconds,'required_jobs',numel(needed),'completed_jobs',completed, ...
    'new_frames',newCount,'wall_seconds',toc(started),'neural_run',false));
end
function record(path,value)
tmp=[path '.tmp'];f=fopen(tmp,'w');assert(f>=0);fprintf(f,'%s',jsonencode(value));fclose(f);movefile(tmp,path,'f');
end
