function validate_native_cone_sampling(configPath,outputPath)
% Compare the integration adapter with cMosaic.compute, including spatial blur.
c=jsondecode(fileread(configPath));p=c.profile;d=load(c.geometry_mat);
addpath(genpath(p.isetcam_root));addpath(genpath(p.isetbio_root));
addpath(fullfile(fileparts(mfilename('fullpath')),'matlab'));
model=load(fullfile(c.optics_cache,'model.mat'));
rows=unique(round(linspace(1,numel(d.types),128)));
xy=d.is_mm(rows,:)*1000;types=d.types(rows);diameters=d.reference_diameter_um(rows);
pigment=cPhotoPigment('wave',model.wave,'opticalDensity',p.optical_density_LMS,'peakEfficiency',p.peak_efficiency_LMS);
macular=Macular('wave',model.wave,'density',0);
data=struct('positionUnits','microns','positions',xy,'types',types, ...
    'lightGatheringApertureDiameters',diameters,'blurApertureDiameterMicrons',mean(diameters));
cm=cone_optics_mosaic(data,pigment,macular,p,.01);
cm.eccVaryingOuterSegmentLength=false;cm.eccVaryingMacularPigmentDensity=false;
x=model.x+c.center_mm(1)*1000;y=model.y+c.center_mm(2)*1000;dx=x(2)-x(1);
[matrix,area,diagnostic]=cone_video_gaussian_sampling(cm,xy,diameters,diameters,x,y);
[groups,kernels,representatives]=native_cone_kernel_groups(cm,diameters,dx);
errors=zeros(2,1);scaleError=max(abs(full(sum(matrix,2))-1));
nativeArea=cm.computeApertureAreasMetersSquared(1:numel(rows));
assert(scaleError<1e-12 && isequal(area,nativeArea(:)));
for scenario=1:2
    [xx,yy]=meshgrid(linspace(-1,1,numel(x)),linspace(-1,1,numel(y)));
    image=ones(size(xx));
    if scenario==2,image=2+sin(xx*31).*cos(yy*23);end
    oi=model.baseOI;oi=oiSet(oi,'photons',zeros(model.samples,model.samples,numel(model.wave),'single'));
    oi=oiPadValue(oi,model.padSize,'zero photons',1e6);
    oi=oiSet(oi,'photons',repmat(single(image),1,1,numel(model.wave)));
    support=oiGet(oi,'spatial support')*1e6;
    sx=squeeze(support(1,:,1));sy=squeeze(support(:,1,2));
    sx=sx-mean(sx)+c.center_mm(1)*1000;sy=sy-mean(sy)+c.center_mm(2)*1000;
    [groups,kernels,representatives]=native_cone_kernel_groups(cm,diameters,oiGet(oi,'height spatial resolution')*1e6);
    [actualMatrix,~,~]=cone_video_gaussian_sampling(cm,xy,diameters,diameters,sx,sy);
    direct=sample_cone_aperture_zones(cm,oi,xy,types,diameters,diameters, ...
        'isetbio_native_pillbox_v1',c.center_mm(:)'*1000,3);
    photons=flipud(oiGet(oi,'photons'));
    density=reshape(photons,[],numel(model.wave))*(cm.qe*oiGet(oi,'bin width'));
    sampled=actualMatrix*double(density);
    sampled=sampled(sub2ind(size(sampled),(1:numel(rows))',types(:))).*area;
    assert(max(abs(sampled-direct)./max(direct,1e-30))<3e-6);
    native=zeros(numel(rows),1);
    for g=1:numel(groups)
        data.blurApertureDiameterMicrons=representatives(g);
        sub=cone_optics_mosaic(data,pigment,macular,p,.01);
        sub.eccVaryingOuterSegmentLength=false;sub.eccVaryingMacularPigmentDensity=false;
        value=sub.compute(oi,'nTrials',1,'nTimePoints',1,'withFixationalEyeMovements',false, ...
            'opticalImagePositionDegs',c.center_mm(:)'/.223,'padOIwithZeros',false);
        native(groups{g})=value(groups{g})/.01;
    end
    errors(scenario)=max(abs(direct-native)./max(native,1e-30));
    fprintf('NATIVE_COMPARISON scenario=%d error=%.12g first=[%.12g %.12g]\n',scenario,errors(scenario),direct(1),native(1));
    assert(errors(scenario)<3e-6,'ConeOptics:Native','Adapter differs from cMosaic.compute');
end
boundaryRejected=false;
try
    cone_video_gaussian_sampling(cm,xy,diameters,diameters,x(1:2),y(1:2));
catch
    boundaryRejected=true;
end
assert(boundaryRejected);
result=struct('status','PASS','cones',numel(rows),'native_kernel_groups',numel(kernels), ...
    'uniform_relative_error',errors(1),'spatial_relative_error',errors(2), ...
    'unit_sum_error',scaleError,'native_area_exact',true,'boundary_rejected',true, ...
    'comparator','cMosaic.compute; native per-kernel blur and per-cone area', ...
    'OS_and_macular_variation_disabled_only_for_aperture_isolation',true,'sampling',diagnostic);
fid=fopen(outputPath,'w');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(result));fprintf('NATIVE_SAMPLING_PASS %.12g %.12g\n',errors);
end
