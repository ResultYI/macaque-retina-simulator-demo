function prepare_cone_video_aperture_optics(configPath)
c=jsondecode(fileread(configPath));p=c.profile;out=c.cache_directory;
addpath(genpath(p.isetcam_root));addpath(genpath(p.isetbio_root));addpath(fullfile(fileparts(mfilename('fullpath')),'matlab'));
display=displayCreate('LCD-Apple');wave=double(displayGet(display,'wave'));
gamma=displayGet(display,'gamma');spd=displayGet(display,'spd');
f=p.microns_per_degree*1e-6/tan(pi/180);
baseOI=oiCreate('human mw');baseOI=oiSet(baseOI,'optics',opticsCreate('human mw',.0015,f));
assert(strcmp(oiGet(baseOI,'diffuser method'),'skip'));
grey=c.reference_rgb;rgb=repmat(reshape(grey,[1 1 3]),c.samples,c.samples);
scene=makeScene(rgb,display,spd,gamma,wave,c.patch_fov_deg,c.fixed_radiance_gain);
baseOI=oiSet(baseOI,'wangular',sceneGet(scene,'wangular'));baseOI=oiSet(baseOI,'wave',wave);
nativeOI=oiCompute(baseOI,scene);
padSize=round([c.samples c.samples]/8);padSize(3)=0;
support=oiGet(nativeOI,'spatial support');x=squeeze(support(1,:,1))*1e6;y=squeeze(support(:,1,2))*1e6;
x=x-mean(x);y=y-mean(y);
pigment=cPhotoPigment('wave',wave,'opticalDensity',p.optical_density_LMS,'peakEfficiency',p.peak_efficiency_LMS);
macular=Macular('wave',wave,'density',p.macular_density);
data=struct('positionUnits','microns','positions',zeros(3,2),'types',[1;2;3], ...
    'lightGatheringApertureDiameters',repmat(c.foveal_reference_diameter_um,3,1), ...
    'blurApertureDiameterMicrons',c.foveal_reference_diameter_um);
referenceMosaic=cone_optics_mosaic(data,pigment,macular,p,1/30);
qe=referenceMosaic.qe;binWidth=oiGet(nativeOI,'bin width');
photons=flipud(oiGet(nativeOI,'photons'));sz=size(photons);
backgroundDensity=reshape(reshape(photons,[],numel(wave))*(qe*binWidth),sz(1),sz(2),3);
model=struct('baseOI',baseOI,'otf',[],'wave',wave,'padSize',padSize);
[fastOI,otf]=cone_video_cached_oi(scene,model);a=oiGet(nativeOI,'photons');b=oiGet(fastOI,'photons');
rawPointwiseRelative=max(abs(double(a(:))-double(b(:)))./max(abs(double(a(:))),1e-30));
relativeL1=sum(abs(double(a(:))-double(b(:))))/sum(abs(double(a(:))));
error=0;
for k=1:numel(wave)
    delta=abs(double(a(:,:,k))-double(b(:,:,k)));reference=double(a(:,:,k));
    error=max(error,max(delta(:))/max(max(abs(reference(:))),1e-30));
end
fprintf('CACHE_COMPARISON wave_peak_relative=%.12g L1=%.12g raw_pointwise=%.12g\n',error,relativeL1,rawPointwiseRelative);
% Zero-padded spectral tails approach zero; compare at their native single-precision scale.
assert(error<1e-6 && relativeL1<1e-6,'ConeVideo:Cache','Cached native OI operations differ');
assert(~isfolder(out),'ConeVideo:Output','Optics cache already exists');mkdir(out);
for k=1:numel(wave)
    otf_slice=otf(:,:,k);save(fullfile(out,sprintf('otf_%03d.mat',k)),'otf_slice','-v7');
end
fixedGain=c.fixed_radiance_gain;samples=c.samples;patchFov=c.patch_fov_deg;
save(fullfile(out,'model.mat'),'baseOI','wave','padSize','gamma','spd','qe','binWidth','x','y', ...
    'backgroundDensity','fixedGain','samples','patchFov','-v7');
summary=struct('cached_vs_native_wave_peak_relative_error',error,'relative_L1_error',relativeL1, ...
    'raw_pointwise_relative_including_near_zero_padding',rawPointwiseRelative,'source_scene_samples',samples, ...
    'OI_samples',[numel(y) numel(x)],'wave_nm',wave,'fixed_radiance_gain',fixedGain, ...
    'native_operations','Irradiance -> cos4 -> zero padding -> per-wave abs(ifft2(OTF*fft2))', ...
    'photon_noise','none','phototransduction_run',false,'downstream_circuit_run',false);
fid=fopen(fullfile(out,'metadata.json'),'w');fprintf(fid,'%s',jsonencode(summary));fclose(fid);
fprintf('CONE_VIDEO_OPTICS_CACHE_READY %.3g\n',error);
end
function scene=makeScene(rgb,display,spd,gamma,wave,fov,gain)
linear=zeros(size(rgb));axis=linspace(0,1,size(gamma,1));
for k=1:3,linear(:,:,k)=interp1(axis,gamma(:,k),rgb(:,:,k),'linear');end
energy=reshape(reshape(linear,[],3)*spd',[size(rgb,1),size(rgb,2),numel(wave)]);
scene=sceneCreate('empty',[],wave);scene=sceneSet(scene,'bit depth',64);
scene=sceneSet(scene,'energy',energy*gain);scene=sceneSet(scene,'fov',fov);scene=sceneSet(scene,'distance',1e6);
end
