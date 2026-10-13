% Existing MATLAB, only historical nonlinear kernel and init function.
% Synthetic numerical fixtures, NOT measured Dryad data. No demo/inverse.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'data','source_snapshots','chen2024_1bcf7c9a'));
initial=initPhotoreceptorParams('peripheralPrimateCone');
names={'dark','background','weak_flash','step_flash','high_rate'};
lengths=[5000,40000,18000,30000,6000];
oracle=struct();
for idx=1:numel(names)
    n=lengths(idx); rates=zeros(1,n);
    if idx==2
        rates(:)=5000;
    elseif idx==3
        rates(6001:6100)=1000;
    elseif idx==4
        rates(:)=5000;
        rates(10001:20000)=10000;
        rates(15001:15100)=15000;
        rates(20001:25000)=1000;
    elseif idx==5
        rates(:)=49000;
    end
    params=initial;
    params.biophysFlag=1;
    params.tme=(0:n-1)*params.timeStep;
    params.stm=rates*params.timeStep; % public rate -> count once
    params=BiophysModel(params);
    oracle.(names{idx})=struct('rates_Rstar_per_s',rates,'absolute_current_pA',params.response,'time_seconds',params.tme,'evidence_tag','synthetic_fixture','source_input_counts',params.stm);
end
metadata=struct('commit','1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2','evidence_tag','synthetic_fixture','branch','nonlinear','dt_seconds',initial.timeStep,'source_kernel','unmodified historical BiophysModel.m','license','MIT','matlab_version',version);
save(fullfile(root,'reports','chen2024_matlab_oracle.mat'),'oracle','metadata','initial','-v7');
disp(jsonencode(struct('fixtures',numel(names),'branch','nonlinear','evidence_tag','synthetic_fixture','output','reports/chen2024_matlab_oracle.mat')));
