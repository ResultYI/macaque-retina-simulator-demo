function export_mixed_reference(configPath,outputPath)
% Approved production frontend. Digital RGB -> measured LCD gamma/SPD -> MW -> cMosaic.
c=jsondecode(fileread(configPath));p=c.profile;d=load(c.input_mat);
assert(~isfile(outputPath),'MixedInput:Output','Choose a fresh output path');
assert(strcmp(p.profile_id,'iset_lcd_apple_mw_v1'),'MixedInput:Profile','Unknown active frontend');
oldPath=path;oldRng=rng;cleanup=onCleanup(@()restore(oldPath,oldRng)); %#ok<NASGU>
addpath(genpath(p.isetcam_root));addpath(genpath(p.isetbio_root));
display=displayCreate('LCD-Apple');wave=double(displayGet(display,'wave'));
gamma=displayGet(display,'gamma');spd=displayGet(display,'spd');
assert(size(gamma,2)==3 && size(spd,2)==3 && all(isfinite(gamma(:))) && all(spd(:)>=0), ...
    'MixedInput:Display','Measured LCD gamma/SPD required');
assert(isfield(d,'native_apertures_um'),'ConeOptics:Geometry','Prepare native apertures on complete support before capture');
diameters_um=d.native_apertures_um(:)';blur_um=mean(diameters_um);
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
assert(isequal(size(d.is_mm),[n 2]) && size(d.rgb_frames,3)==3 && numel(d.frame_indices)==t, ...
    'MixedInput:Shape','Frozen cone/frame dimensions differ');
dt=diff(d.window_edges_seconds(:));assert(all(dt>0) && all(abs(dt-dt(1))<1e-12),'MixedInput:Time','Uniform integration windows required');
withEyeMovements=isfield(d,'em_positions_arcmin');
if withEyeMovements
    assert(size(d.em_positions_arcmin,1)==t && size(d.em_positions_arcmin,3)==2 && ...
        all(isfinite(d.em_positions_arcmin(:))),'MixedInput:EyePath','Invalid eye/video alignment');
    substeps=size(d.em_positions_arcmin,2); probes=d.fine_probe_indices(:);
    probeCounts=zeros(t*substeps,numel(probes)); convergence=[];
    checkFrames=unique([1 floor(t/2)+1 t]);
    if isfield(d,'microsaccade_onset_seconds')
        checkFrames=unique([checkFrames(:);floor(d.microsaccade_onset_seconds(:)/dt(1))+1]);
        checkFrames=checkFrames(checkFrames>=1 & checkFrames<=t);
    end
end
data=struct('positionUnits','microns','positions',d.is_mm*1e3,'types',types, ...
    'lightGatheringApertureDiameters',diameters_um,'blurApertureDiameterMicrons',blur_um);
[mosaic,coneOptics]=cone_optics_mosaic(data,pigment,macular,p,dt(1));
assert(isequal(mosaic.coneRFpositionsMicrons,d.is_mm*1e3) && isequal(mosaic.coneTypes,types), ...
    'MixedInput:Identity','Frozen cone positions/types changed');
center=d.image_center_deg(:)';referenceRGB=d.reference_rgb(:)';
referenceImage=repmat(reshape(referenceRGB,[1 1 3]),[size(d.rgb_frames,1),size(d.rgb_frames,2),1]);
backgroundScene=makeScene(referenceImage,display,spd,gamma,wave,c.patch_fov_deg,1);
rawLuminance=sceneGet(backgroundScene,'mean luminance');
targetLuminance=100/(pi*(3/2)^2);gain=targetLuminance/rawLuminance;
assert(isfinite(gain)&&gain>0,'MixedInput:Reference','Reference LCD luminance must be positive');
backgroundScene=sceneSet(backgroundScene,'photons',sceneGet(backgroundScene,'photons')*gain);
backgroundOI=oiCompute(oi,backgroundScene);
res=oiGet(backgroundOI,'height spatial resolution')*1e6;
assert(ceil(2*blur_um/res)>=3,'MixedInput:Resolution','Scene would skip collecting aperture blur');
support=oiGet(backgroundOI,'spatial support');x=squeeze(support(1,:,1))*1e6;y=squeeze(support(:,1,2))*1e6;
x=x-mean(x)+center(1)*223;y=y-mean(y)+center(2)*223;
kernel=mosaic.generateApertureKernel(max(diameters_um),res,false);margin=(size(kernel,1)-1)/2*res+res;
xy=d.is_mm*1e3;
assert(all(xy(:,1)>min(x)+margin & xy(:,1)<max(x)-margin & xy(:,2)>min(y)+margin & xy(:,2)<max(y)-margin), ...
    'MixedInput:Coverage','OI/aperture halo is insufficient; no capture extrapolation');
background_counts=capture_cone_optics(mosaic,backgroundOI,center,n,p);
referenceData=struct('positionUnits','microns','positions',repmat(center*223,[3 1]),'types',[1;2;3], ...
    'lightGatheringApertureDiameters',repmat(mean(diameters_um),[3 1]),'blurApertureDiameterMicrons',blur_um);
referenceMosaic=cone_optics_mosaic(referenceData,pigment,macular,p,dt(1));
q_ref_LMS=capture_cone_optics(referenceMosaic,backgroundOI,'mosaic-centered',3,p)/dt(1);
assert(all(q_ref_LMS>0),'MixedInput:Reference','Positive fixed LMS reference captures required');
expected_counts=zeros(t,n);uniqueCounts=zeros(size(d.rgb_frames,4),n);
for k=1:size(d.rgb_frames,4)
    scene=makeScene(double(d.rgb_frames(:,:,:,k)),display,spd,gamma,wave,c.patch_fov_deg,gain);
    oiFrame=oiCompute(oi,scene);
    if withEyeMovements
        eyePath=squeeze(d.em_positions_arcmin(k,:,:));
        [uniqueCounts(k,:),fine]=capture_cone_optics(mosaic,oiFrame,center,n,p,eyePath);
        probeCounts((k-1)*substeps+(1:substeps),:)=fine(:,probes);
        if ismember(k,checkFrames)
            coarseSteps=ceil(substeps/2);
            coarseTimes=d.window_edges_seconds(k)+((0:coarseSteps-1)'+.5)*dt(1)/coarseSteps;
            coarsePath=interp1(d.eye_time_seconds(:),d.eye_position_arcmin,coarseTimes,'linear');
            coarse=capture_cone_optics(mosaic,oiFrame,center,n,p,coarsePath);
            errors=sort(abs(coarse-uniqueCounts(k,:))./max(uniqueCounts(k,:),1e-12));
            convergence(end+1,:)=[k sum(abs(coarse-uniqueCounts(k,:)))/sum(uniqueCounts(k,:)), ...
                errors(ceil(.95*n)),max(errors)]; %#ok<AGROW>
        end
    else
        uniqueCounts(k,:)=capture_cone_optics(mosaic,oiFrame,center,n,p);
    end
end
for k=1:t,expected_counts(k,:)=uniqueCounts(d.frame_indices(k),:);end
assert(all(isfinite(expected_counts(:))) && all(expected_counts(:)>=0),'MixedInput:Values','Capture must be finite/nonnegative');
summary=struct('schema','approved_mixed_capture_v1','matlab_version',version, ...
    'profile_id',p.profile_id,'scope',p.scope,'request_signature',c.request_signature, ...
    'background_signature',c.background_signature,'display_name',displayGet(display,'name'), ...
    'wave_nm',wave,'gamma_shape',size(gamma),'gamma_applied_once',true,'fixed_radiance_gain',gain, ...
    'raw_reference_luminance_cd_m2',rawLuminance,'reference_luminance_cd_m2',targetLuminance, ...
    'I_ref_td',100,'reference_rgb',referenceRGB,'q_ref_LMS_rstar_s',q_ref_LMS, ...
    'reference_diameter_um',mean(diameters_um),'blur_diameter_um',blur_um, ...
    'blur_diameter_semantics','Legacy mean metadata only; actual integration uses each cone aperture kernel', ...
    'cone_optics',coneOptics, ...
    'diameter_range_um',[min(diameters_um),max(diameters_um)], ...
    'spacing_policy','ISET positionsToSpacings median of six nearest on frozen full support; D=0.79*spacing', ...
    'optics_name',opticsGet(optics,'name'),'focal_length_m',f,'pupil_diameter_mm',3, ...
    'OD_LMS',pigment.opticalDensity,'peak_efficiency_LMS',pigment.peakEfficiency, ...
    'lens_density',lens.density,'macular_density',macular.density, ...
    'macular_rule','ISET human eccentricity rule in capture; human Lens applied only in optical image', ...
    'OS_factors',mosaic.computeEffectiveOSlengthAttenuationFactors(1:n), ...
    'phototransduction_applied',false,'photon_noise','none','units','expected R*/window', ...
    'coordinates','Frozen IS from named RF=IS project transfer; OI internally flips rows, RGB patch top maps positive functional y', ...
    'finite_optical_support','Existing halo_mm crop + ISET OI padding; aperture coverage asserted; finite optical support is a project numerical approximation');
summary.eye_movements_applied=withEyeMovements;
if withEyeMovements
    summary.eye_substep_seconds=dt(1)/substeps;
    summary.eye_substeps_per_frame=substeps;
    summary.eye_macula_rule='Native cMosaic mean-eye-position macular map per held video frame';
    summary.eye_quadrature_checks=struct('columns',{{'frame_index_1based','relative_L1_error','relative_error_95th_percentile','maximum_relative_error'}}, ...
        'coarse_vs_fine',convergence);
    expected_counts_substeps=probeCounts;
    substep_times_seconds=reshape(d.subsample_times_seconds',[],1);
    substep_seconds=dt(1)/substeps; probe_indices=probes;
    probe_positions_mm=d.is_mm(probes,:); probe_types=d.types(probes);
    assert(~isfile([outputPath '.eye_substeps.mat']),'MixedInput:Output','Preserve the prior temporal probes');
    save([outputPath '.eye_substeps.mat'],'expected_counts_substeps','substep_times_seconds','substep_seconds', ...
        'probe_indices','probe_positions_mm','probe_types','-v7');
end
manifest_json=jsonencode(summary);window_edges_seconds=d.window_edges_seconds;
assert(~isfile(outputPath),'MixedInput:Output','Choose a fresh output path');
save(outputPath,'manifest_json','expected_counts','background_counts','q_ref_LMS','window_edges_seconds', ...
    'diameters_um','-v7');
fprintf('MIXED_CAPTURE_PASS %d cones, %d windows, %d actual OI frames\n',n,t,size(d.rgb_frames,4));
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
