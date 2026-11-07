function capture_gin69_gray_extension(configPath)
% Frozen 69#6 Gaussian flat gray reservoir; no movie frames or dynamic owners.
c=jsondecode(fileread(configPath));p=c.profile;
f=fopen(fullfile(c.extension_run,'MATLAB_SELF.json'),'w');assert(f>=0);
fprintf(f,'%s',jsonencode(struct('pid',feature('getpid'))));fclose(f);
assert(~isfile(c.extension_output),'Existing static background retained');
addpath(genpath(p.isetcam_root));addpath(genpath(p.isetbio_root));
m=load(fullfile(c.optics_cache,'model.mat'),'backgroundDensity','wave','x');
d=load(c.extension_input);n=numel(d.types);
pigment=cPhotoPigment('wave',m.wave,'opticalDensity',p.optical_density_LMS,'peakEfficiency',p.peak_efficiency_LMS);
macular=Macular('wave',m.wave,'density',p.macular_density);
data=struct('positionUnits','microns','positions',zeros(3,2),'types',[1;2;3], ...
 'lightGatheringApertureDiameters',ones(3,1)*c.foveal_reference_diameter_um, ...
 'blurApertureDiameterMicrons',c.foveal_reference_diameter_um);
cm=cone_optics_mosaic(data,pigment,macular,p,c.exposure_seconds);
cm.coneApertureModifiers=struct('shape','Gaussian','sigma',.204);
% The approved static extension is a spatially uniform incident grey reservoir.
% Its spectral OI is the cached gray optical-axis field, not an average or 1000td.
% The optical PSF/QE have already been applied in backgroundDensity, exactly once.
mid=(size(m.backgroundDensity,1)+1)/2;assert(mid==round(mid));
dx=m.x(2)-m.x(1);sample=zeros(n,1);
for zone=unique(d.zones_um(:))'
 kernel=cm.generateApertureKernel(zone,dx,false);kernel=kernel/sum(kernel(:));
 radius=(size(kernel,1)-1)/2;
 for kind=1:2
  rows=find(d.zones_um(:)==zone & d.types(:)==kind);
  patch=m.backgroundDensity(mid-radius:mid+radius,mid-radius:mid+radius,kind);
  sample(rows)=sum(double(patch(:)).*kernel(:));
 end
end
% Exact type-specific Beer-Lambert OS rule from cone_optics_mosaic, unchanged.
[lengths,foveal]=cMosaic.outerSegmentLengthFromEccentricity(d.eccentricity_deg(:));
relative=lengths/foveal;
attenuationPigment=cPhotoPigment('opticalDensity',pigment.opticalDensity);
factors=zeros(n,1);
for kind=1:2
 rows=find(d.types(:)==kind);absorbance=attenuationPigment.absorbance(:,kind)';
 reference=-expm1(-log(10)*absorbance*pigment.opticalDensity(kind));valid=reference>0;
 for first=1:4096:numel(rows)
  rr=rows(first:min(first+4095,numel(rows)));
  local=-expm1(-log(10)*(pigment.opticalDensity(kind)*relative(rr))*absorbance(valid));
  factors(rr)=mean(local./reference(valid),2);
 end
end
rates=sample.*d.area_m2(:).*factors;
assert(all(isfinite(rates)) && all(rates>0));
manifest_json=jsonencode(struct('scope','approved static grey exterior; no dynamic stimulus', ...
 'optical_reservoir','frozen pre-live gray optical-axis OI; unit-sum frozen Gaussian kernel', ...
 'area_applications',1,'OS_applications',1,'native_inputs_equivalent_not_photopic_td',true, ...
 'same_QE_spectrum_pupil_gain',true,'physical_peripheral_illumination_calibrated',false));
save(c.extension_output,'rates','factors','manifest_json','-v7');
fprintf('STATIC_GRAY_EXTENSION_COMPLETE %d sources\n',n);
end
