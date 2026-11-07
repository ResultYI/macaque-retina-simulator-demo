function validate_native_aperture_v1(reportPath)
% Current native aperture checks; historical Gaussian comparisons stay archived.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'tools'));
for name={'uniform','spatial'}
    item=name{1};export_cone_video_aperture_tile(fullfile(reportPath,[item '.json']), ...
        fullfile(reportPath,[item '_native.mat']));
end
validate_native_cone_sampling(fullfile(reportPath,'spatial.json'),fullfile(reportPath,'VALIDATION.json'));
end
