function export_isetbio_capture(configPath, outputPath)
% Explicit scene -> ISETCam optics -> ISETBio expected cone counts. No GUI/default changes.
% Input MAT arrays and JSON units are documented by make_isetbio_fixture.py.
c = jsondecode(fileread(configPath));
required = {'schema','scope','source_id','input_mat','input_mat_sha256', ...
    'isetcam_root','isetbio_root','scene','optics','capture','coordinates','cones','time'};
requireFields(c, required, 'config');
assert(strcmp(c.schema,'isetbio_capture_export_v1'), 'ISETExport:Version','Unknown export schema');
assert(ismember(c.scope,{'engineering_fixture','explicit_calibration_candidate'}), ...
    'ISETExport:Scope','Explicit scope required; no physiological validation claim');
requireFields(c.scene,{'unit','axes','horizontal_fov_deg','distance_m','source_id'},'scene');
requireFields(c.optics,{'mode','pupil_diameter_mm','focal_length_m','offaxis_method','source_id'},'optics');
requireFields(c.capture,{'source_id','blur_diameter_um','aperture_shape','optical_density_LMS', ...
    'peak_efficiency_LMS','macular_density','pigment_diameter_m'},'capture');
requireFields(c.coordinates,{'reference_id','registration','microns_per_degree', ...
    'which_eye','optical_image_position_deg','is_role','is_frame','rf_role','rf_frame', ...
    'is_binding_id','is_binding_source_sha256'},'coordinates');
requireFields(c.cones,{'cell_ids','types','population_id','evidence_tag'},'cones');
requireFields(c.time,{'source_origin_seconds','window_edges_seconds','frame_semantics'},'time');
assert(strcmp(c.scene.unit,'photons/(s m^2 sr nm)') && ...
    isequal(c.scene.axes,{'row';'column';'wavelength';'window'}), ...
    'ISETExport:Unit','Scene must be absolute spectral photon radiance, row/column/wavelength/window');
assert(strcmp(c.time.frame_semantics,'constant_radiance_within_each_window'), ...
    'ISETExport:Time','Explicit held-frame integration required');
assert(strcmp(c.coordinates.is_role,'inner_segment') && strcmp(c.coordinates.is_frame,'anatomical_retina') && ...
    strcmp(c.coordinates.rf_role,'rf') && strcmp(c.coordinates.rf_frame,'functional_visual_field'), ...
    'ISETExport:Coordinates','Separate IS capture coordinates and RF binding required');
assert(strcmp(c.capture.aperture_shape,'Pillbox'),'ISETExport:Aperture', ...
    'v1 uses an explicit common pillbox blur with per-cone collecting diameters');
assert(ismember(c.optics.offaxis_method,{'skip','cos4th'}),'ISETExport:Optics','Explicit offaxis method required');
assert(ismember(c.coordinates.which_eye,{'left eye','right eye'}),'ISETExport:Coordinates','Explicit eye required');
positive([c.scene.horizontal_fov_deg,c.scene.distance_m,c.optics.pupil_diameter_mm, ...
    c.optics.focal_length_m,c.capture.blur_diameter_um,c.capture.pigment_diameter_m, ...
    c.coordinates.microns_per_degree],'positive physical dimensions');
assert(c.scene.horizontal_fov_deg < 180,'ISETExport:Scene','FOV must be below 180 degrees');
assert(isfinite(c.time.source_origin_seconds),'ISETExport:Time','Explicit finite source origin required');
edges = c.time.window_edges_seconds(:)'; dt = diff(edges);
assert(numel(edges)>=2 && edges(1)==0 && all(isfinite(edges)) && all(dt>0) && ...
    all(abs(dt-dt(1))<=1e-12),'ISETExport:Time','All windows must have the declared uniform positive spacing from zero');
assert(strcmp(shaFile(c.input_mat),c.input_mat_sha256),'ISETExport:Identity','Input MAT hash differs');
d = load(c.input_mat);
requireFields(d,{'scene_photons','wave_nm','is_mm','rf_mm','diameters_um','os_factors', ...
    'absorbance_LMS','macular_unit_density','lens_transmittance'},'input MAT');
wave = d.wave_nm(:); n = numel(c.cones.cell_ids); t = numel(dt);
assert(n>0 && numel(unique(c.cones.cell_ids))==n && numel(c.cones.types)==n, ...
    'ISETExport:IDs','Frozen unique IDs/types required');
assert(all(ismember(c.cones.types,{'L','M','S'})),'ISETExport:Types','Only explicit L/M/S types are supported');
assert(numel(wave)>=3 && all(isfinite(wave)) && all(wave>0) && ...
    all(diff(wave)>0) && all(abs(diff(wave)-diff(wave(1:2)))<=1e-10), ...
    'ISETExport:Wavelength','Pinned API requires uniform positive wavelength support in nm');
assert(size(d.scene_photons,1)>=3 && size(d.scene_photons,2)>=3 && ...
    size(d.scene_photons,3)==numel(wave) && size(d.scene_photons,4)==t && ndims(d.scene_photons)<=4, ...
    'ISETExport:Shape','Scene dimensions differ from wavelength/windows');
nonnegative(d.scene_photons,'scene photons');
assert(isequal(size(d.is_mm),[n 2]) && isequal(size(d.rf_mm),[n 2]) && ...
    all(isfinite(d.is_mm(:))) && all(isfinite(d.rf_mm(:))),'ISETExport:Coordinates','IS/RF arrays must be N x 2 finite mm');
assert(numel(d.diameters_um)==n && numel(d.os_factors)==n,'ISETExport:Shape','Per-cone diameters/OS factors required');
positive(d.diameters_um,'diameters'); positive(d.os_factors,'OS factors');
assert(isequal(size(d.absorbance_LMS),[numel(wave) 3]) && numel(d.macular_unit_density)==numel(wave) && ...
    numel(d.lens_transmittance)==numel(wave),'ISETExport:Shape','Explicit spectral capture/transmission arrays required');
nonnegative(d.absorbance_LMS,'absorbance'); nonnegative(d.macular_unit_density,'macular absorbance');
nonnegative(d.lens_transmittance,'lens transmission');
assert(all(d.lens_transmittance<=1),'ISETExport:Optics','Lens transmission must be at most one');
positive(c.capture.optical_density_LMS,'optical densities');
positive(c.capture.peak_efficiency_LMS,'peak efficiencies');
assert(numel(c.capture.optical_density_LMS)==3 && numel(c.capture.peak_efficiency_LMS)==3 && ...
    all(c.capture.peak_efficiency_LMS<=1) && isfinite(c.capture.macular_density) && c.capture.macular_density>=0, ...
    'ISETExport:Capture','Explicit valid LMS density/efficiency and macular density required');

% Restore only this MATLAB process's path/RNG even after an API failure.
oldPath = path; oldRng = rng; cleanup = onCleanup(@()restoreSession(oldPath,oldRng)); %#ok<NASGU>
assert(strcmp(gitHead(c.isetbio_root),'d2924c2fe1d41c7dc5af1006d8081735b08f1b35') && ...
    strcmp(gitHead(c.isetcam_root),'17146a9f0470cc89b58d583851f680187268506d'), ...
    'ISETExport:Version','Exporter targets the installed pinned ISET commits');
addpath(genpath(c.isetcam_root)); addpath(genpath(c.isetbio_root));
assert(startsWith(which('cMosaic'),[c.isetbio_root filesep]) && ...
    startsWith(which('oiCompute'),[c.isetcam_root filesep]),'ISETExport:Path','Another toolbox shadows pinned APIs');
oi = oiCreate('diffraction limited');
switch c.optics.mode
    case 'diffraction_limited_explicit'
        requireFields(c.optics,{'model_assumption'},'optics');
        optics = oiGet(oi,'optics'); % replaced below by explicit physical settings
    case 'supplied_iset_optics'
        requireFields(d,{'iset_optics'},'input MAT');
        optics = d.iset_optics;
        requireFields(optics,{'type','model','focalLength','fNumber','offaxis'},'supplied optics');
        assert(abs(optics.focalLength-c.optics.focal_length_m)<1e-12 && ...
            abs(optics.fNumber-c.optics.focal_length_m/(c.optics.pupil_diameter_mm*1e-3))<1e-10, ...
            'ISETExport:Optics','Supplied optics disagree with explicit focal length/pupil');
    otherwise
        error('ISETExport:Optics','No default human optics; provide calibrated ISET optics or an explicit diffraction model');
end
% Lens transmission belongs to optics, macular/pigment capture to cMosaic.
if isfield(optics,'lens'), optics=rmfield(optics,'lens'); end
optics = opticsSet(optics,'focal length',c.optics.focal_length_m);
optics = opticsSet(optics,'fnumber',c.optics.focal_length_m/(c.optics.pupil_diameter_mm*1e-3));
optics.transmittance.wave = wave; optics.transmittance.scale = d.lens_transmittance(:);
optics.offaxis = c.optics.offaxis_method;
oi = oiSet(oi,'optics',optics); oi = oiSet(oi,'diffuser method','skip');
pigment = cPhotoPigment('wave',wave,'absorbance',d.absorbance_LMS, ...
    'opticalDensity',c.capture.optical_density_LMS(:)', ...
    'peakEfficiency',c.capture.peak_efficiency_LMS(:)','diameter',c.capture.pigment_diameter_m);
macular = Macular('wave',wave,'density',c.capture.macular_density, ...
    'unitDensity',d.macular_unit_density(:),'name',c.capture.source_id);
types = zeros(n,1);
for k=1:n, types(k)=find(strcmp(c.cones.types{k},{'L','M','S'})); end
coneData = struct('positionUnits','microns','positions',d.is_mm*1e3,'types',types, ...
    'lightGatheringApertureDiameters',d.diameters_um(:), ...
    'blurApertureDiameterMicrons',c.capture.blur_diameter_um, ...
    'outerSegmentLengthAttenationFactors',d.os_factors(:));
mosaic = cMosaic('coneData',coneData,'pigment',pigment,'macular',macular,'wave',wave, ...
    'micronsPerDegree',c.coordinates.microns_per_degree,'whichEye',c.coordinates.which_eye, ...
    'integrationTime',dt(1),'noiseFlag','none','randomSeed',1,'useParfor',false, ...
    'eccVaryingConeAperture',true,'eccVaryingConeBlur',false,'eccVaryingOuterSegmentLength',true, ...
    'eccVaryingMacularPigmentDensity',false,'eccVaryingMacularPigmentDensityDynamic',false, ...
    'anchorAllEccVaryingParamsToTheirFovealValues',false,'rodIntrusionAdjustedConeAperture',false, ...
    'coneCouplingLambda',0,'coneApertureModifiers',struct('shape','Pillbox'));
assert(isequal(mosaic.coneTypes,types) && isequal(mosaic.coneRFpositionsMicrons,d.is_mm*1e3) && ...
    isequal(mosaic.coneApertureDiametersMicrons(:),d.diameters_um(:)), ...
    'ISETExport:Mutation','ISET must preserve frozen types/IS/apertures');
imagePosition = c.coordinates.optical_image_position_deg(:)';
% Pinned cMosaic's numeric-position getter divides position spans. Avoid its
% singular zero-span case only when the caller explicitly chose this center.
if norm(max(d.is_mm,[],1)-min(d.is_mm,[],1))==0
    assert(all(abs(imagePosition*c.coordinates.microns_per_degree-mosaic.eccentricityMicrons)<1e-9), ...
        'ISETExport:Coordinates','Degenerate mosaic requires its explicitly declared image center');
    imagePosition='mosaic-centered';
end
expected_counts = zeros(t,n); retinal_photons = []; spatial_support_m = []; luminance_cd_m2=zeros(t,1);
for frame=1:t
    scene = sceneCreate('empty',[],wave);
    scene = sceneSet(scene,'bit depth',64);
    scene = sceneSet(scene,'photons',double(d.scene_photons(:,:,:,frame)));
    scene = sceneSet(scene,'fov',c.scene.horizontal_fov_deg);
    scene = sceneSet(scene,'distance',c.scene.distance_m);
    luminance_cd_m2(frame)=sceneGet(scene,'mean luminance');
    oiFrame = oiCompute(oi,scene);
    support = oiGet(oiFrame,'spatial support'); res = oiGet(oiFrame,'height spatial resolution')*1e6;
    assert(ceil(c.capture.blur_diameter_um*2/res)>=3,'ISETExport:Resolution', ...
        'ISET would skip aperture blur at this resolution; supply a finer scene grid');
    kernel = mosaic.generateApertureKernel(c.capture.blur_diameter_um,res,false);
    margin = (size(kernel,1)-1)/2*res + res;
    x = squeeze(support(1,:,1))*1e6; y = squeeze(support(:,1,2))*1e6;
    center = c.coordinates.optical_image_position_deg(:)'*c.coordinates.microns_per_degree;
    x=x-mean(x)+center(1); y=y-mean(y)+center(2); pos=d.is_mm*1e3;
    assert(all(pos(:,1)>=min(x)+margin & pos(:,1)<=max(x)-margin & ...
        pos(:,2)>=min(y)+margin & pos(:,2)<=max(y)-margin), ...
        'ISETExport:Coverage','OI needs a halo around every IS; no ISET padding/nearest extrapolation permitted');
    [counts,noisy,current,currentNoise,apiTimes] = mosaic.compute(oiFrame,'nTrials',1,'nTimePoints',1, ...
        'withFixationalEyeMovements',false,'opticalImagePositionDegs',imagePosition, ...
        'padOIwithZeros',false,'lowOpticalImageResolutionWarning',true);
    assert(isempty(noisy) && isempty(current) && isempty(currentNoise) && apiTimes==0, ...
        'ISETExport:Stage','Expected capture only; no upstream photocurrent/noise');
    expected_counts(frame,:) = reshape(counts,[1 n]);
    photons = oiGet(oiFrame,'photons');
    if frame==1
        retinal_photons=zeros([size(photons),t]); spatial_support_m=support;
    end
    retinal_photons(:,:,:,frame)=photons;
end
nonnegative(expected_counts,'expected counts');
window_edges_seconds=edges; is_mm=d.is_mm; rf_mm=d.rf_mm;
scope = struct('optics',c.optics,'resolved_optics',optics,'capture',c.capture,'coordinates',c.coordinates, ...
    'wave_nm',wave,'absorbance_LMS',d.absorbance_LMS,'macular_unit_density',d.macular_unit_density, ...
    'lens_transmittance',d.lens_transmittance);
meta = c; meta.schema='isetbio_cone_capture_v1'; meta.data_axes={'time','cell'};
meta.unit='R*/window'; meta.value_semantics='expected'; meta.capture_scope_sha256=shaText(jsonencode(scope));
meta.matlab_version=version; meta.isetbio_commit=gitHead(c.isetbio_root); meta.isetcam_commit=gitHead(c.isetcam_root);
meta.exporter_sha256=shaFile(which('export_isetbio_capture'));
meta.api_source_sha256=struct('cMosaic',shaFile(which('cMosaic')), ...
    'oiCompute',shaFile(which('oiCompute')),'compute',shaFile(fullfile(fileparts(which('cMosaic')),'compute.m')));
meta.frame_transform='ISETBio compute flips OI rows internally; supplied IS is x-right/y-up, never relabeled RF';
meta.aperture_policy='per-cone collecting diameter; one explicitly supplied common pillbox blur diameter';
meta.scene_luminance_cd_m2=luminance_cd_m2;
meta.boundary_policy='reject missing aperture halo and blur-skipping resolution';
meta.mechanism_ownership=struct('optics','ISETCam','photopigment_capture','ISETBio', ...
    'phototransduction','consumer','photon_noise','none','neural_noise','consumer');
meta.validation='ENGINEERING_INTERFACE_PASS_NOT_PHYSIOLOGICAL_CALIBRATION';
manifest_json=jsonencode(meta); wave_nm=wave;
assert(~isfile(outputPath),'ISETExport:Output','Output already exists; choose a fresh path');
save(outputPath,'manifest_json','expected_counts','window_edges_seconds','is_mm','rf_mm', ...
    'wave_nm','retinal_photons','spatial_support_m','-v7');
fprintf('ISET capture PASS: %d windows x %d frozen cones -> %s\n',t,n,outputPath);
end

function requireFields(s, names, label)
missing = names(~isfield(s,names));
assert(isempty(missing),'ISETExport:MissingFields','Missing %s fields: %s',label,strjoin(missing,', '));
end
function positive(v,label)
assert(isnumeric(v) && ~isempty(v) && all(isfinite(v(:))) && all(v(:)>0), ...
    'ISETExport:Value','%s must be finite and positive',label);
end
function nonnegative(v,label)
assert(isnumeric(v) && ~isempty(v) && all(isfinite(v(:))) && all(v(:)>=0), ...
    'ISETExport:Value','%s must be finite and nonnegative',label);
end
function h=shaFile(p)
fid=fopen(p,'rb'); assert(fid>=0,'ISETExport:File','Cannot open %s',p);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
h=shaBytes(fread(fid,inf,'*uint8'));
end
function h=shaText(s), h=shaBytes(unicode2native(s,'UTF-8')); end
function h=shaBytes(bytes)
digest=java.security.MessageDigest.getInstance('SHA-256'); digest.update(bytes);
h=lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2)',1,[]));
end
function h=gitHead(root)
git=fullfile(root,'.git'); h=strtrim(fileread(fullfile(git,'HEAD')));
if startsWith(h,'ref: '), h=strtrim(fileread(fullfile(git,h(6:end)))); end
end
function restoreSession(p,r), path(p); rng(r); end
