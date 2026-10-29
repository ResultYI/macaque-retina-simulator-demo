function [rates, diagnostic] = sample_cone_aperture_zones(mosaic, oi, positionsMicrons, types, referenceDiameters, zoneDiameters, mode, imageCenterMicrons, legacyBlurDiameter)
% Project adapter over pinned ISETCam OI / ISETBio pigment and kernel math.
% Unit-sum kernel -> spectral absorption density -> sample -> exact area once.
% This does not call computeAbsorptionRate again or reapply an optical PSF.
assert(ismember(mode,{'legacy_fixed_cone_blur','macaque_eccentricity_aperture_v1'}));
n=size(positionsMicrons,1);assert(size(positionsMicrons,2)==2 && numel(types)==n);
assert(numel(referenceDiameters)==n && all(referenceDiameters>0) && numel(zoneDiameters)==n);
photons=oiGet(oi,'photons');wave=oiGet(oi,'wave');mosaic.wave=wave;
binWidth=oiGet(oi,'bin width');qe=mosaic.qe;
assert(size(qe,1)==numel(wave) && size(qe,2)==3 && isscalar(binWidth));
photons=flipud(photons); % Same x-right/y-up registration as cMosaic.compute.
sz=size(photons);density=reshape(reshape(photons,[],numel(wave))*(qe*binWidth),sz(1),sz(2),3);
support=oiGet(oi,'spatial support');
x=squeeze(support(1,:,1))*1e6;y=squeeze(support(:,1,2))*1e6;
x=x-mean(x)+imageCenterMicrons(1);y=y-mean(y)+imageCenterMicrons(2);
dx=abs(x(2)-x(1));dy=abs(y(2)-y(1));assert(abs(dx-dy)<1e-6*dx);
oldModifiers=mosaic.coneApertureModifiers;
cleanup=onCleanup(@()restoreModifiers(mosaic,oldModifiers)); %#ok<NASGU>
if strcmp(mode,'legacy_fixed_cone_blur')
    mosaic.coneApertureModifiers=struct('shape','Pillbox');
    zones=ones(n,1)*legacyBlurDiameter;
    area=pi*(referenceDiameters(:)*.5e-6).^2;
else
    mosaic.coneApertureModifiers=struct('shape','Gaussian','sigma',.204);
    zones=zoneDiameters(:);
    area=2*pi*(.204*referenceDiameters(:)*1e-6).^2;
end
rates=zeros(n,1);diameters=unique(zones);kernelSums=zeros(numel(diameters),2);radii=zeros(numel(diameters),1);
for iz=1:numel(diameters)
    d=diameters(iz);assert(ceil(2*d/dx)>=3,'Aperture would be skipped at this OI resolution');
    kernel=mosaic.generateApertureKernel(d,dx,false);
    kernelSums(iz,1)=sum(kernel(:));kernel=kernel/sum(kernel(:));kernelSums(iz,2)=sum(kernel(:));
    radius=(size(kernel,1)-1)*dx/2;radii(iz)=radius;
    select=find(zones==d);pos=positionsMicrons(select,:);
    assert(all(pos(:,1)>min(x)+radius+dx & pos(:,1)<max(x)-radius-dx & ...
        pos(:,2)>min(y)+radius+dy & pos(:,2)<max(y)-radius-dy),'Missing aperture halo; extrapolation disallowed');
    for kind=1:3
        chosen=select(types(select)==kind);if isempty(chosen),continue;end
        filtered=conv2(density(:,:,kind),kernel,'same');
        F=griddedInterpolant({y(:),x(:)},filtered,'linear','none');
        q=F([positionsMicrons(chosen,2),positionsMicrons(chosen,1)]);
        assert(all(isfinite(q)) && all(q>=0),'Invalid cone interpolation');
        rates(chosen)=q.*area(chosen);
    end
end
% Supplied OS factors are explicit and applied once in the caller; no defaults here.
assert(all(isfinite(rates)) && all(rates>=0));
diagnostic=struct('mode',mode,'unit','R*/s','spectral_bin_width_nm',binWidth, ...
    'kernel_diameters_um',diameters,'kernel_sum_before_after_normalization',kernelSums, ...
    'kernel_support_radii_um',radii,'pixel_size_um',dx,'effective_area_m2',area, ...
    'wavelength_nm',wave,'spectral_quantal_efficiency',qe,'area_applications',1, ...
    'optical_PSF_reapplied',false,'extrapolation','none','ISETBio_import_zones_bypassed',true);
end
function restoreModifiers(mosaic,value),mosaic.coneApertureModifiers=value;end
