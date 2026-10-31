function export_cone_video_aperture_tile(configPath,outputPath)
% Cone-only capture: pinned MW spectrum, frozen Gaussian D(e), no neural stage.
persistent model modelPath sourceBank sourcePath sampler samplerKey area os types diag cm
c=jsondecode(fileread(configPath));p=c.profile;
assert(~isfile(outputPath),'ConeVideo:Output','Preserve prior capture');
addpath(genpath(p.isetcam_root));addpath(genpath(p.isetbio_root));addpath(fullfile(fileparts(mfilename('fullpath')),'matlab'));
if isempty(modelPath) || ~strcmp(modelPath,c.optics_cache)
    model=load(fullfile(c.optics_cache,'model.mat'));
    for k=1:numel(model.wave)
        d=load(fullfile(c.optics_cache,sprintf('otf_%03d.mat',k)));
        if k==1,model.otf=zeros([size(d.otf_slice) numel(model.wave)],'like',d.otf_slice);end
        model.otf(:,:,k)=d.otf_slice;
    end
    modelPath=c.optics_cache;
end
if isempty(sourcePath) || ~strcmp(sourcePath,c.source_bank)
    bank=load(c.source_bank);sourceBank=bank.source_rgb;sourcePath=c.source_bank;
end
d=load(c.geometry_mat);n=numel(d.types);
if isempty(samplerKey) || ~strcmp(samplerKey,c.tile_signature)
    pigment=cPhotoPigment('wave',model.wave,'opticalDensity',p.optical_density_LMS, ...
        'peakEfficiency',p.peak_efficiency_LMS);
    macular=Macular('wave',model.wave,'density',p.macular_density);
    data=struct('positionUnits','microns','positions',d.is_mm*1e3,'types',d.types(:), ...
        'lightGatheringApertureDiameters',d.reference_diameter_um(:), ...
        'blurApertureDiameterMicrons',mean(d.reference_diameter_um));
    cm=cone_optics_mosaic(data,pigment,macular,p,1/30);
    x=model.x+c.center_mm(1)*1e3;y=model.y+c.center_mm(2)*1e3;
    [sampler,area,diag]=cone_video_gaussian_sampling(cm,d.is_mm*1e3, ...
        d.reference_diameter_um,d.zone_diameter_um,x,y);
    os=cm.computeEffectiveOSlengthAttenuationFactors(1:n)';types=d.types(:);
    samplerKey=c.tile_signature;
end
axis=((0:model.samples-1)-(model.samples-1)/2)*.001;
[xx,yy]=meshgrid(axis+c.center_mm(1),-axis+c.center_mm(2));
height=size(sourceBank,1);width=size(sourceBank,2);bounds=c.plot_bounds_mm;
pitch=max((bounds(2)-bounds(1))/(width-1),(bounds(4)-bounds(3))/(height-1));
row=(height-1)/2-(yy-(bounds(3)+bounds(4))/2)/pitch+1;
col=(width-1)/2+(xx-(bounds(1)+bounds(2))/2)/pitch+1;
row(abs(row-1)<1e-8)=1;row(abs(row-height)<1e-8)=height;
col(abs(col-1)<1e-8)=1;col(abs(col-width)<1e-8)=width;
background_counts=sampleDensity(model.backgroundDensity,sampler,area,os,types)/30;
frames=c.frame_indices(:);expected_counts=zeros(numel(frames),n);control=[];
display=displayCreate('LCD-Apple');gammaAxis=linspace(0,1,size(model.gamma,1));
for j=1:numel(frames)
    rgb=zeros(model.samples,model.samples,3,'single');
    for k=1:3
        source=double(sourceBank(:,:,k,frames(j)))/255;
        rgb(:,:,k)=single(interp2(source,col,row,'linear',c.reference_rgb(k)));
    end
    linear=zeros(size(rgb));
    for k=1:3,linear(:,:,k)=interp1(gammaAxis,model.gamma(:,k),double(rgb(:,:,k)),'linear');end
    energy=reshape(reshape(linear,[],3)*model.spd',[model.samples model.samples numel(model.wave)]);
    scene=sceneCreate('empty',[],model.wave);scene=sceneSet(scene,'bit depth',64);
    scene=sceneSet(scene,'energy',energy*model.fixedGain);scene=sceneSet(scene,'fov',model.patchFov);
    scene=sceneSet(scene,'distance',1e6);
    oi=cone_video_cached_oi(scene,model);photons=flipud(oiGet(oi,'photons'));sz=size(photons);
    density=reshape(reshape(photons,[],numel(model.wave))*(model.qe*model.binWidth),sz(1),sz(2),3);
    rates=sampleDensity(density,sampler,area,os,types);
    expected_counts(j,:)=rates'/30;
    if c.native_control && j==1
        nativeOI=oiCompute(model.baseOI,scene);
        native=sample_cone_aperture_zones(cm,nativeOI,d.is_mm*1e3,types,d.reference_diameter_um, ...
            d.zone_diameter_um,'macaque_eccentricity_aperture_v1',c.center_mm(:)'*1e3,3);
        native=double(native).*os;
        relative=max(abs(rates-native)./max(native,1e-20));
        assert(relative<3e-6,'ConeVideo:NativeControl','Gaussian sparse/native capture differs');
        control=struct('frame',frames(j),'cones',n,'maximum_relative_error',relative, ...
            'native_oiCompute_and_zone_sampler_used',true);
    end
end
assert(all(isfinite(expected_counts(:))) && all(expected_counts(:)>=0) && all(background_counts>0));
window_edges_seconds=c.window_edges_seconds;diameters_um=d.reference_diameter_um(:)';
meta=struct('schema','cone_eccentric_aperture_video_tile_v2','aperture_backend','isetbio_native_aperture_v1','request_signature',c.request_signature, ...
    'tile_signature',c.tile_signature,'aperture_profile_sha256',c.aperture_profile_sha256, ...
    'kernel','Gaussian sigma=0.204*reference diameter; unit sum;0.05um zones', ...
    'area','2*pi*sigma^2 per actual cone, once','units','expected R*/frame', ...
    'sampling',diag,'native_control',control,'phototransduction_run',false, ...
    'downstream_circuit_run',false,'eye_movements',false,'background_qualified',false,'FIT_READY',false,'production_adopted',false);
manifest_json=jsonencode(meta);
save(outputPath,'expected_counts','background_counts','window_edges_seconds','diameters_um','manifest_json','-v7');
fprintf('ECCENTRIC_CAPTURE_PASS %d cones, %d frames\n',n,numel(frames));
end
function rates=sampleDensity(density,matrix,area,os,types)
sampled=matrix*double(reshape(density,[],3));
if isa(density,'single'),sampled=single(sampled);end
chosen=sampled(sub2ind(size(sampled),(1:numel(types))',types));
rates=double(chosen.*area).*os;
end
