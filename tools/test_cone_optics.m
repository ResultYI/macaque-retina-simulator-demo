function test_cone_optics(configPath, outputPath)
% Small capture-only checks: typed OS factors, exact aperture integration, no circuit.
c=jsondecode(fileread(configPath)); p=c.profile;
addpath(genpath(p.isetcam_root)); addpath(genpath(p.isetbio_root));
pigment=cPhotoPigment('opticalDensity',p.optical_density_LMS,'peakEfficiency',p.peak_efficiency_LMS);
macular=Macular('wave',pigment.wave,'density',p.macular_density);
data=struct('positionUnits','microns','positions',zeros(3,2),'types',[1;2;3], ...
    'lightGatheringApertureDiameters',[2;3;4],'blurApertureDiameterMicrons',3);
[foveal,ev]=cone_optics_mosaic(data,pigment,macular,p,.01);
assert(max(abs(foveal.computeEffectiveOSlengthAttenuationFactors(1:3)-1))<1e-14);
assert(isequal(foveal.coneApertureDiametersMicrons(:),[2;3;4]));
assert(max(abs(ev.foveal_OS_length_um_LMS-[35.5;35.5;35.5*.95]))<1e-12);
assert(max(abs(ev.specific_density_per_um_LMS.*ev.foveal_OS_length_um_LMS-pigment.opticalDensity(:)))<1e-14);
data.positions=repmat([2*p.microns_per_degree,0],3,1);
[peripheral,ev]=cone_optics_mosaic(data,pigment,macular,p,.01);
[lengths,fov]=cMosaic.outerSegmentLengthFromEccentricity(2);
for k=1:3
    a=pigment.absorbance(:,k); od=pigment.opticalDensity(k);
    opticalDepth=log(10)*a*od; ratios=ones(size(a))*lengths/fov;
    valid=opticalDepth>1e-8;
    ratios(valid)=(1-exp(-opticalDepth(valid)*lengths/fov))./(1-exp(-opticalDepth(valid)));
    expected=mean(ratios);
    assert(abs(ev.OS_attenuation(k)-expected)<1e-7);
end
assert(abs(ev.OS_attenuation(3)-ev.OS_attenuation(1))>1e-4);
displayPigment=cPhotoPigment('wave',380:4:780,'opticalDensity',p.optical_density_LMS);
[~,resampled]=cone_optics_mosaic(data,displayPigment,macular,p,.01);
assert(max(abs(resampled.OS_attenuation-ev.OS_attenuation))<1e-14);
diameters=[1.2;1.8;2.4;3.2;4.0;5.0]; n=numel(diameters);
% Remove spatial macular filtering only in this area/kernel isolation test.
macular=Macular('wave',pigment.wave,'density',0);
data=struct('positionUnits','microns','positions',repmat([300 0],n,1), ...
    'types',ones(n,1),'lightGatheringApertureDiameters',diameters,'blurApertureDiameterMicrons',mean(diameters));
cm=cone_optics_mosaic(data,pigment,macular,p,.01);
f=p.microns_per_degree*1e-6/tan(pi/180);
oi=oiCreate('human mw'); oi=oiSet(oi,'optics',opticsCreate('human mw',.0015,f));
oi=oiSet(oi,'wave',pigment.wave);
pixels=65; pitch=.5e-6;
oi=oiSet(oi,'photons',ones(pixels,pixels,numel(pigment.wave))*1e15);
oi=oiSet(oi,'fov',2*atan(pixels*pitch/(2*f))*180/pi);
center=[300/p.microns_per_degree 0];
uniform=capture_cone_optics(cm,oi,center,n,p);
assert(max(abs(uniform./uniform(1)-(diameters'/diameters(1)).^2))<1e-10);
[x,y]=meshgrid(-32:32,-32:32);
pattern=1+.8*cos(x*pi/2).*cos(y*pi/3);
oi=oiSet(oi,'photons',repmat(pattern,[1 1 numel(pigment.wave)])*1e15);
grouped=capture_cone_optics(cm,oi,center,n,p); independent=zeros(1,n);
cached=capture_cone_optics(cm,oi,center,n,p);
assert(isequal(grouped,cached),'ConeOptics:Cache','Exact kernel cache changed captures');
for k=1:n
    single=data; single.positions=data.positions(k,:); single.types=data.types(k);
    single.lightGatheringApertureDiameters=diameters(k); single.blurApertureDiameterMicrons=diameters(k);
    one=cone_optics_mosaic(single,pigment,macular,p,.01);
    [value,noisy,current,currentNoise]=one.compute(oi,'nTrials',1,'nTimePoints',1, ...
        'withFixationalEyeMovements',false,'opticalImagePositionDegs','mosaic-centered','padOIwithZeros',false);
    assert(isempty(noisy)&&isempty(current)&&isempty(currentNoise)); independent(k)=value;
end
relativeError=max(abs(grouped-independent)./independent);
assert(relativeError<1e-12);
exportDirectory=[outputPath '.exports']; assert(~isfolder(exportDirectory)); mkdir(exportDirectory);
rgb_frames=ones(65,65,3,2)*.7353569830524495; rgb_frames(:,:,:,2)=rgb_frames(:,:,:,2)*.5;
window_edges_seconds=[0 .01 .02]; frame_indices=[1 2]; reference_rgb=repmat(rgb_frames(1,1,1,1),1,3);
is_mm=[-5 -3; -3 4; 0 -2; 2 4; 4 -3; 5 2]/1000;
types=[1;2;3;1;2;3]; frozen_apertures_um=[2;2.5;3;3.2;3.5;4]; image_center_deg=[0 0];
input=fullfile(exportDirectory,'input.mat');
save(input,'rgb_frames','window_edges_seconds','frame_indices','reference_rgb','is_mm','types', ...
    'frozen_apertures_um','image_center_deg');
config=struct('profile',p,'input_mat',input,'patch_fov_deg',2*atan(65*1e-6/(2*f))*180/pi, ...
    'request_signature','optical_exporter_smoke_origin','background_signature','optical_exporter_smoke', ...
    'global_reference',struct('reference_diameter_um',3,'blur_um',3,'definition_sha256','optical_exporter_smoke'));
configFile=fullfile(exportDirectory,'origin.json'); writeJSON(configFile,config);
originFile=fullfile(exportDirectory,'origin.mat'); export_disk_mixed_reference(configFile,originFile);
origin=load(originFile); originMeta=jsondecode(origin.manifest_json);
% ISET scene energy->photon conversion uses single precision; gain order differs for the background.
uniformSceneError=max(abs(origin.expected_counts(1,:)./origin.background_counts-1));
assert(uniformSceneError<1e-6);
assert(all(origin.expected_counts(2,:)<origin.expected_counts(1,:)));
assert(isequal(origin.diameters_um(:),frozen_apertures_um));
is_mm=is_mm+[.223 0]; image_center_deg=[1 0];
input=fullfile(exportDirectory,'shifted_input.mat'); config.input_mat=input;
save(input,'rgb_frames','window_edges_seconds','frame_indices','reference_rgb','is_mm','types', ...
    'frozen_apertures_um','image_center_deg');
config.global_reference.fixed_radiance_gain=originMeta.fixed_radiance_gain;
config.global_reference.reference_energy_spectrum_hex=originMeta.reference_energy_spectrum_hex;
config.global_reference.q_ref_LMS_rstar_s=origin.q_ref_LMS;
config.request_signature='optical_exporter_smoke_shifted';
configFile=fullfile(exportDirectory,'shifted.json'); writeJSON(configFile,config);
shiftedFile=fullfile(exportDirectory,'shifted.mat'); export_disk_mixed_reference(configFile,shiftedFile);
shifted=load(shiftedFile); shiftedMeta=jsondecode(shifted.manifest_json);
assert(isequal(shifted.q_ref_LMS,origin.q_ref_LMS));
assert(shiftedMeta.fixed_radiance_gain==originMeta.fixed_radiance_gain);
assert(strcmp(shiftedMeta.cone_optics.revision,p.cone_optics_revision));
summary=struct('typed_OS_formula_pass',true,'foveal_factors_equal_one',true, ...
    'existing_foveal_OD_preserved',true,'OS_invariant_to_display_wave_resampling',true,'uniform_area_scaling_pass',true, ...
    'grouped_vs_independent_relative_error',relativeError,'type_diameter_multiplier_used',false, ...
    'fixed_tile_export_origin_and_shifted_pass',true,'global_reference_gain_and_LMS_preserved',true, ...
    'uniform_scene_relative_error',uniformSceneError, ...
    'downstream_circuit_run',false,'phototransduction_run',false);
assert(~isfile(outputPath),'ConeOptics:Output','Do not overwrite previous validation');
fid=fopen(outputPath,'w','n','UTF-8'); assert(fid~=-1); cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(summary)); fprintf('CONE_OPTICS_TEST_PASS\n');
end
function writeJSON(file,value)
fid=fopen(file,'w','n','UTF-8'); assert(fid~=-1); guard=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(value));
end
