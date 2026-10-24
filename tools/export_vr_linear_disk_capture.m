function export_vr_linear_disk_capture(configPath,outputPath)
% Input-only linear gray adapter; original MW/cMosaic helpers copied unchanged.
% Original producer SHA256: 034c22330e154798588eb57d69e7cb80545d33caa97e35205380936eadf46ca7
% Gamma is used only to recreate the EXISTING neutral reference, never on gray8.
c=jsondecode(fileread(configPath));p=c.profile;d=load(c.input_mat);
assert(strcmp(p.profile_id,'iset_lcd_apple_mw_v1'),'MixedInput:Profile','Unknown active frontend');
oldPath=path;oldRng=rng;cleanup=onCleanup(@()restore(oldPath,oldRng)); %#ok<NASGU>
addpath(genpath(p.isetcam_root));addpath(genpath(p.isetbio_root));
display=displayCreate('LCD-Apple');wave=double(displayGet(display,'wave'));
gamma=displayGet(display,'gamma');spd=displayGet(display,'spd');
assert(size(gamma,2)==3 && size(spd,2)==3 && all(isfinite(gamma(:))) && all(spd(:)>=0), ...
    'MixedInput:Display','Measured LCD gamma/SPD required');
assert(isfield(c,'global_reference') && isfield(d,'frozen_apertures_um'),'DiskInput:Reference','Frozen global reference/apertures required');
globalReference=c.global_reference;diameters_um=d.frozen_apertures_um(:)';blur_um=globalReference.blur_um;
assert(all(isfinite(diameters_um)) && isfinite(blur_um) && blur_um>0,'DiskInput:Aperture','Invalid fixed global apertures');
assert(all(isfinite(diameters_um)) && all(diameters_um>0),'MixedInput:Aperture','Invalid full-support cone spacing');
pigment=cPhotoPigment('wave',wave,'opticalDensity',p.optical_density_LMS, ...
    'peakEfficiency',p.peak_efficiency_LMS);
macular=Macular('wave',wave,'density',p.macular_density);
f=p.microns_per_degree*1e-6/tan(pi/180);
oi=oiCreate('human mw');optics=opticsCreate('human mw',p.pupil_diameter_mm*.5e-3,f);
oi=oiSet(oi,'optics',optics);lens=opticsGet(optics,'lens');
assert(strcmp(opticsGet(optics,'name'),'human-MW') && lens.density==1 && macular.density==.35, ...
    'MixedInput:Optics','Expected explicit MW/human Lens/Macular conventions differ');
types=d.types(:);n=numel(types);t=numel(d.window_edges_seconds)-1;
assert(isequal(size(d.is_mm),[n 2]) && numel(d.frame_indices)==t && isfield(d,'scene_multipliers'), ...
    'MixedInput:Shape','Frozen cone/frame dimensions differ');
dt=diff(d.window_edges_seconds(:));assert(all(dt>0) && all(abs(dt-dt(1))<1e-12),'MixedInput:Time','Uniform integration windows required');
data=struct('positionUnits','microns','positions',d.is_mm*1e3,'types',types, ...
    'lightGatheringApertureDiameters',diameters_um,'blurApertureDiameterMicrons',blur_um);
mosaic=makeMosaic(data,pigment,macular,p,dt(1));
assert(isequal(mosaic.coneRFpositionsMicrons,d.is_mm*1e3) && isequal(mosaic.coneTypes,types), ...
    'MixedInput:Identity','Frozen cone positions/types changed');
center=d.image_center_deg(:)';referenceRGB=d.reference_rgb(:)';
referenceImage=repmat(reshape(referenceRGB,[1 1 3]),[size(d.scene_multipliers,1),size(d.scene_multipliers,2),1]);
backgroundScene=makeScene(referenceImage,display,spd,gamma,wave,c.patch_fov_deg,1);
rawLuminance=sceneGet(backgroundScene,'mean luminance');
referenceLinear=zeros(1,3);gammaAxis=linspace(0,1,size(gamma,1));
for channel=1:3,referenceLinear(channel)=interp1(gammaAxis,gamma(:,channel),referenceRGB(channel),'linear');end
referenceSpectrumHex=cellstr(num2hex(double(referenceLinear*spd')));
targetLuminance=100/(pi*(3/2)^2);
if isfield(globalReference,'fixed_radiance_gain')
    % ISET's single-precision luminance evaluation can differ by scene size.
    % Bind identical emitted reference spectra, then reuse the ONE foveal gain.
    % A tile never recomputes or fits a gain from its rounded luminance.
    assert(isfield(globalReference,'reference_energy_spectrum_hex') && ...
        isequal(string(referenceSpectrumHex(:)),string(globalReference.reference_energy_spectrum_hex(:))), ...
        'DiskInput:ReferenceSpectrum','Uniform emitted reference spectrum differs across fixed tiles');
    gain=globalReference.fixed_radiance_gain;
else
    assert(all(center==0),'DiskInput:Reference','Global reference must be constructed at the fixed foveal origin');
    gain=targetLuminance/rawLuminance;
end
assert(isfinite(gain)&&gain>0,'MixedInput:Reference','Reference LCD luminance must be positive');
backgroundScene=sceneSet(backgroundScene,'photons',sceneGet(backgroundScene,'photons')*gain);
backgroundOI=oiCompute(oi,backgroundScene);
res=oiGet(backgroundOI,'height spatial resolution')*1e6;
assert(ceil(2*blur_um/res)>=3,'MixedInput:Resolution','Scene would skip collecting aperture blur');
support=oiGet(backgroundOI,'spatial support');x=squeeze(support(1,:,1))*1e6;y=squeeze(support(:,1,2))*1e6;
x=x-mean(x)+center(1)*223;y=y-mean(y)+center(2)*223;
kernel=mosaic.generateApertureKernel(blur_um,res,false);margin=(size(kernel,1)-1)/2*res+res;
xy=d.is_mm*1e3;
assert(all(xy(:,1)>min(x)+margin & xy(:,1)<max(x)-margin & xy(:,2)>min(y)+margin & xy(:,2)<max(y)-margin), ...
    'MixedInput:Coverage','OI/aperture halo is insufficient; no capture extrapolation');
reference_counts=capture(mosaic,backgroundOI,center,n);
assert(strcmp(c.input_policy_sha256,c.approved_policy_sha256),'VRGray:Policy','Approved exact policy required');
addpath(c.linear_gray_adapter_directory);
backgroundMappedScene=make_gray_reference_scene(backgroundScene,ones(size(d.scene_multipliers,1),size(d.scene_multipliers,2))*c.outside_reference_multiplier,c.input_policy_sha256,c.approved_policy_sha256);
backgroundMappedOI=oiCompute(oi,backgroundMappedScene);
background_counts=capture(mosaic,backgroundMappedOI,center,n);
if isfield(globalReference,'q_ref_LMS_rstar_s')
    q_ref_LMS=globalReference.q_ref_LMS_rstar_s(:)';
else
    referenceData=struct('positionUnits','microns','positions',zeros(3,2),'types',[1;2;3], ...
        'lightGatheringApertureDiameters',repmat(globalReference.reference_diameter_um,[3 1]),'blurApertureDiameterMicrons',blur_um);
    referenceMosaic=makeMosaic(referenceData,pigment,macular,p,dt(1));
    q_ref_LMS=capture(referenceMosaic,backgroundOI,'mosaic-centered',3)/dt(1);
end
assert(all(q_ref_LMS>0),'MixedInput:Reference','Positive fixed LMS reference captures required');
expected_counts=zeros(t,n);uniqueCounts=zeros(size(d.scene_multipliers,3),n);
actualOIcomputations=0;
for k=1:size(d.scene_multipliers,3)
    % Same exact scene on a fixed tile has the same memoryless expected capture
    % (noise and eye movements are disabled). Preserve every source frame index;
    % only reuse identical input on this tile, never merge near-equal frames.
    identical=0;
    for j=1:k-1
        if isequal(d.scene_multipliers(:,:,k),d.scene_multipliers(:,:,j))
            identical=j;break;
        end
    end
    if identical>0
        uniqueCounts(k,:)=uniqueCounts(identical,:);continue;
    end
    scene=make_gray_reference_scene(backgroundScene,double(d.scene_multipliers(:,:,k)),c.input_policy_sha256,c.approved_policy_sha256);
    oiFrame=oiCompute(oi,scene);
    uniqueCounts(k,:)=capture(mosaic,oiFrame,center,n);
    actualOIcomputations=actualOIcomputations+1;
end
for k=1:t,expected_counts(k,:)=uniqueCounts(d.frame_indices(k),:);end
assert(all(isfinite(expected_counts(:))) && all(expected_counts(:)>=0),'MixedInput:Values','Capture must be finite/nonnegative');
summary=struct('schema','approved_vr_linear_capture_v1','matlab_version',version, ...
    'profile_id',p.profile_id,'scope',p.scope,'request_signature',c.request_signature, ...
    'background_signature',c.background_signature,'display_name',displayGet(display,'name'), ...
    'wave_nm',wave,'gamma_shape',size(gamma),'gamma_applied_once',true,'fixed_radiance_gain',gain, ...
    'raw_reference_luminance_cd_m2',rawLuminance,'reference_luminance_cd_m2',targetLuminance, ...
    'I_ref_td',100,'reference_rgb',referenceRGB,'q_ref_LMS_rstar_s',q_ref_LMS, ...
    'reference_diameter_um',globalReference.reference_diameter_um,'blur_diameter_um',blur_um, ...
    'diameter_range_um',[min(diameters_um),max(diameters_um)], ...
    'global_reference_definition_sha256',globalReference.definition_sha256, ...
    'spacing_policy','Frozen global master median of six nearest; D=0.79*spacing; no tile/ROI aperture recomputation', ...
    'optics_name',opticsGet(optics,'name'),'focal_length_m',f,'pupil_diameter_mm',3, ...
    'OD_LMS',pigment.opticalDensity,'peak_efficiency_LMS',pigment.peakEfficiency, ...
    'lens_density',lens.density,'macular_density',macular.density, ...
    'macular_rule','ISET human eccentricity rule in capture; human Lens applied only in optical image', ...
    'OS_factors',mosaic.computeEffectiveOSlengthAttenuationFactors(1:n), ...
    'phototransduction_applied',false,'photon_noise','none','units','expected R*/window', ...
    'coordinates','Frozen IS from named RF=IS project transfer; OI internally flips rows, RGB patch top maps positive functional y', ...
    'finite_optical_support','Existing halo_mm crop + ISET OI padding; aperture coverage asserted; finite optical support is a project numerical approximation');
summary.reference_energy_spectrum_hex=referenceSpectrumHex;
summary.input_policy_sha256=c.input_policy_sha256;summary.gray8_gamma_applied=false;
summary.outside_reference_multiplier=c.outside_reference_multiplier;
summary.actual_distinct_OI_computations=actualOIcomputations;
summary.background_is_actual_outside_capture=true;summary.reference_only_gamma_applied=true;
summary.reference_gain_rule='one foveal gain; exact emitted-spectrum identity across all tiles, no tile luminance refit';
summary.local_uniform_scene_luminance_after_fixed_gain_cd_m2=double(rawLuminance)*gain;
manifest_json=jsonencode(summary);window_edges_seconds=d.window_edges_seconds;fixed_radiance_gain_numeric=gain;
assert(~isfile(outputPath),'MixedInput:Output','Choose a fresh output path');
save(outputPath,'manifest_json','expected_counts','background_counts','q_ref_LMS','window_edges_seconds', ...
    'diameters_um','fixed_radiance_gain_numeric','-v7');
fprintf('MIXED_CAPTURE_PASS %d cones, %d windows, %d actual OI frames\n',n,t,size(d.scene_multipliers,3));
end
function cm=makeMosaic(data,pigment,macular,p,dt)
cm=cMosaic('coneData',data,'pigment',pigment,'macular',macular,'wave',pigment.wave, ...
    'micronsPerDegree',223,'whichEye','right eye','integrationTime',dt,'noiseFlag','none', ...
    'randomSeed',1,'useParfor',false,'eccVaryingConeAperture',true,'eccVaryingConeBlur',false, ...
    'eccVaryingOuterSegmentLength',true,'eccVaryingMacularPigmentDensity',true, ...
    'eccVaryingMacularPigmentDensityDynamic',false,'anchorAllEccVaryingParamsToTheirFovealValues',false, ...
    'rodIntrusionAdjustedConeAperture',false,'coneCouplingLambda',0,'coneApertureModifiers',struct('shape','Pillbox'));
end
function counts=capture(cm,oi,center,n)
[counts,noisy,current,currentNoise]=cm.compute(oi,'nTrials',1,'nTimePoints',1,'withFixationalEyeMovements',false, ...
    'opticalImagePositionDegs',center,'padOIwithZeros',false);
assert(isempty(noisy)&&isempty(current)&&isempty(currentNoise),'MixedInput:Ownership','Capture only, no upstream transduction/noise');
counts=reshape(counts,[1 n]);
end
function scene=makeScene(rgb,display,spd,gamma,wave,fov,gain)
assert(all(rgb(:)>=0 & rgb(:)<=1),'MixedInput:RGB','Digital display RGB must be [0,1]');
rgbLinear=zeros(size(rgb));axis=linspace(0,1,size(gamma,1));
for k=1:3,rgbLinear(:,:,k)=interp1(axis,gamma(:,k),rgb(:,:,k),'linear');end
energy=reshape(reshape(rgbLinear,[],3)*spd',[size(rgb,1),size(rgb,2),numel(wave)]);
scene=sceneCreate('empty',[],wave);scene=sceneSet(scene,'bit depth',64);
scene=sceneSet(scene,'energy',energy*gain);scene=sceneSet(scene,'fov',fov);scene=sceneSet(scene,'distance',1e6);
end
function restore(p,r),path(p);rng(r);end
