function capture_cross_position_background_v1(configPath, outputPath)
% Single existing neutral reference, frozen aperture sampler, no neural run.
c=jsondecode(fileread(configPath));p=c.profile;d=load(c.geometry_mat);
assert(~isfile(outputPath),'Preserve prior capture');
addpath(genpath(p.isetcam_root));addpath(genpath(p.isetbio_root));
addpath(fullfile(fileparts(mfilename('fullpath')),'matlab'));
display=displayCreate('LCD-Apple');wave=double(displayGet(display,'wave'));
gamma=displayGet(display,'gamma');spd=displayGet(display,'spd');axis=linspace(0,1,size(gamma,1));
linear=zeros(1,3);
for k=1:3,linear(k)=interp1(axis,gamma(:,k),c.reference_rgb(k),'linear');end
spectrum=double(linear*spd');
assert(isequal(string(cellstr(num2hex(spectrum(:)))),string(c.reference_energy_spectrum_hex(:))));
energy=repmat(reshape(spectrum,1,1,[]),c.samples,c.samples,1)*c.fixed_radiance_gain;
scene=sceneCreate('empty',[],wave);scene=sceneSet(scene,'bit depth',64);
scene=sceneSet(scene,'energy',energy);scene=sceneSet(scene,'fov',c.patch_fov_deg);
scene=sceneSet(scene,'distance',1e6);
f=.223e-3/tan(pi/180);oi=oiCreate('human mw');
oi=oiSet(oi,'optics',opticsCreate('human mw',p.pupil_diameter_mm*.5e-3,f));
oi=oiCompute(oi,scene);
pigment=cPhotoPigment('wave',wave,'opticalDensity',p.optical_density_LMS,'peakEfficiency',p.peak_efficiency_LMS);
macular=Macular('wave',wave,'density',p.macular_density);
data=struct('positionUnits','microns','positions',d.is_mm*1e3,'types',d.types(:), ...
 'lightGatheringApertureDiameters',d.reference_diameter_um(:),'blurApertureDiameterMicrons',mean(d.reference_diameter_um));
cm=cone_optics_mosaic(data,pigment,macular,p,.005);
[rates,diagnostic]=sample_cone_aperture_zones(cm,oi,d.is_mm*1e3,d.types(:), ...
 d.reference_diameter_um,d.zone_diameter_um,'macaque_eccentricity_aperture_v1',c.center_mm(:)'*1e3,3);
os=cm.computeEffectiveOSlengthAttenuationFactors(1:numel(d.types))';
rates_rstar_s=double(rates).*os;
assert(all(isfinite(rates_rstar_s)) && all(rates_rstar_s>0));
manifest_json=jsonencode(struct('schema','cross-position-constant-aperture-capture-2', ...
 'aperture_backend','isetbio_native_aperture_v1', ...
 'fixed_global_reference_used',true,'per_position_normalization',false,'area_applications',1, ...
 'OS_applications',1,'diagnostic',diagnostic,'q_ref_LMS',c.q_ref_LMS, ...
 'neural_model_run',false,'background_qualified',false,'FIT_READY',false,'production_adopted',false));
save(outputPath,'rates_rstar_s','manifest_json','-v7');
fprintf('LOCAL_BACKGROUND_CAPTURE_PASS %d cones\n',numel(d.types));
end
