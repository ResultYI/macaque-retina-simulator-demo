function validate_native_aperture_v1(reportPath)
% Two bounded, frozen optical fixtures; no neural forward or parameter fit.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'tools'));addpath(fullfile(root,'tools','matlab'));
c=jsondecode(fileread(fullfile(reportPath,'spatial.json')));p=c.profile;
addpath(genpath(p.isetcam_root));addpath(genpath(p.isetbio_root));
legacy=fullfile(reportPath,'legacy','tools');
addpath(legacy,'-begin');addpath(fullfile(legacy,'matlab'),'-begin');
clear export_cone_video_aperture_tile cone_video_gaussian_sampling sample_cone_aperture_zones
assert(strcmp(which('export_cone_video_aperture_tile'),fullfile(legacy,'export_cone_video_aperture_tile.m')));
for name={'uniform','spatial'}
    item=name{1};export_cone_video_aperture_tile(fullfile(reportPath,[item '.json']), ...
        fullfile(reportPath,[item '_legacy.mat']));
end
rmpath(fullfile(legacy,'matlab'));rmpath(legacy);
clear export_cone_video_aperture_tile cone_video_gaussian_sampling sample_cone_aperture_zones
assert(strcmp(which('export_cone_video_aperture_tile'),fullfile(root,'tools','export_cone_video_aperture_tile.m')));
results=struct();tolerance=3e-6;
for name={'uniform','spatial'}
    item=name{1};export_cone_video_aperture_tile(fullfile(reportPath,[item '.json']), ...
        fullfile(reportPath,[item '_native.mat']));
    a=load(fullfile(reportPath,[item '_legacy.mat']));b=load(fullfile(reportPath,[item '_native.mat']));
    captureError=max(abs(a.expected_counts(:)-b.expected_counts(:))./max(abs(a.expected_counts(:)),1e-20));
    backgroundError=max(abs(a.background_counts(:)-b.background_counts(:))./max(abs(a.background_counts(:)),1e-20));
    assert(captureError<tolerance && backgroundError<tolerance,'Native/legacy full-path capture mismatch');
    results.(item)=struct('maximum_capture_relative_error',captureError, ...
        'maximum_background_relative_error',backgroundError, ...
        'native_manifest',jsondecode(b.manifest_json),'legacy_manifest',jsondecode(a.manifest_json));
end
d=load(c.geometry_mat);n=numel(d.types);model=load(fullfile(c.optics_cache,'model.mat'));
pigment=cPhotoPigment('wave',model.wave,'opticalDensity',p.optical_density_LMS,'peakEfficiency',p.peak_efficiency_LMS);
macular=Macular('wave',model.wave,'density',p.macular_density);
data=struct('positionUnits','microns','positions',d.is_mm*1e3,'types',d.types(:), ...
    'lightGatheringApertureDiameters',d.reference_diameter_um(:),'blurApertureDiameterMicrons',mean(d.reference_diameter_um));
cm=cone_optics_mosaic(data,pigment,macular,p,1/30);
before=cm.coneApertureDiametersMicrons;xy=cm.coneRFpositionsMicrons;os=cm.computeEffectiveOSlengthAttenuationFactors(1:n);
x=model.x+c.center_mm(1)*1e3;y=model.y+c.center_mm(2)*1e3;
[matrix,area,diagnostic]=cone_video_gaussian_sampling(cm,xy,d.reference_diameter_um,d.zone_diameter_um,x,y);
assert(isequal(before,cm.coneApertureDiametersMicrons) && isequal(xy,cm.coneRFpositionsMicrons));
assert(isequal(os,cm.computeEffectiveOSlengthAttenuationFactors(1:n)));
% A constant retinal density must retain its scale through the complete sampler.
density=repmat(reshape([11 17 23],1,1,3),numel(y),numel(x),1);
sampled=matrix*reshape(density,[],3);chosen=sampled(sub2ind(size(sampled),(1:n)',d.types(:)));
capture=chosen.*area.*os';expected=[11 17 23];expected=expected(d.types(:))'.*area.*os';
scaleError=max(abs(capture-expected)./expected);
assert(scaleError<1e-12 && max(abs(area-2*pi*(.204*d.reference_diameter_um(:)*1e-6).^2)./area)<1e-12);
% Real OI/QE and the direct sampling path must also preserve imported state.
% Reuse native spatial support from the cached optical geometry, not a new eye model.
photons=ones(numel(y),numel(x),numel(model.wave),'single');
oi=model.baseOI;oi=oiSet(oi,'photons',photons);oi=oiSet(oi,'wave',model.wave);
[direct,directDiagnostic]=sample_cone_aperture_zones(cm,oi,xy,d.types(:), ...
    d.reference_diameter_um,d.zone_diameter_um,'macaque_eccentricity_aperture_v1',c.center_mm(:)'*1e3,3);
assert(isequal(before,cm.coneApertureDiametersMicrons) && isequal(os,cm.computeEffectiveOSlengthAttenuationFactors(1:n)));
assert(all(isfinite(direct)) && all(direct>0));
uniformDensity=sum(cm.qe*oiGet(oi,'bin width'),1);uniformExpected=uniformDensity(d.types(:))'.*area;
directScaleError=max(abs(direct-uniformExpected)./uniformExpected);
assert(directScaleError<tolerance,'Direct OI/QE/area scale regression');
boundaryRejected=false;
try
    cone_video_gaussian_sampling(cm,xy,d.reference_diameter_um,d.zone_diameter_um,x(1:2),y(1:2));
catch
    boundaryRejected=true;
end
assert(boundaryRejected && isequal(before,cm.coneApertureDiametersMicrons),'Boundary/state restoration regression');
results.state_and_scale=struct('cones',n,'type_counts',arrayfun(@(t)sum(d.types==t),1:3), ...
    'imported_diameters_positions_OS_unchanged',true,'constant_density_scale_relative_error',scaleError, ...
    'constant_photon_OI_capture_scale_relative_error',directScaleError, ...
    'boundary_rejected_and_state_restored',boundaryRejected,'sampling',diagnostic,'direct_sampling',directDiagnostic);
results.status='PASS';results.full_path_relative_tolerance=tolerance;
results.FIT_READY=false;results.background_qualified=false;results.production_adopted=false;
fid=fopen(fullfile(reportPath,'VALIDATION.json'),'w');assert(fid>=0);
cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(results));
fprintf('NATIVE_APERTURE_CONVERGENCE_PASS %d cones; uniform %.12g; spatial %.12g\n', ...
    n,results.uniform.maximum_capture_relative_error,results.spatial.maximum_capture_relative_error);
end
