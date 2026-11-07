function [counts, substepCounts] = capture_cone_optics(cm, oi, center, n, p, eyePathArcMin)
% Use each actual aperture's ISETBio kernel, sharing only bit-identical kernels.
if nargin<6, eyePathArcMin=[]; end
withEyeMovements=~isempty(eyePathArcMin);
if withEyeMovements
    assert(size(eyePathArcMin,2)==2 && all(isfinite(eyePathArcMin(:))), ...
        'ConeOptics:EyePath','Finite sample-by-2 arcminute path required');
    samples=size(eyePathArcMin,1);
else
    samples=1;
end
substepCounts=zeros(samples,n);
assert(n==numel(cm.coneTypes) && all(cm.coneApertureDiametersMicrons>0), ...
    'ConeOptics:Shape','Frozen cone dimensions differ');
if ischar(center)
    assert(strcmp(center,'mosaic-centered'),'ConeOptics:Center','Unknown optical image center');
elseif all(max(cm.coneRFpositionsMicrons,[],1)==min(cm.coneRFpositionsMicrons,[],1))
    % ISETBio's extent-based degree getter is singular for coincident reference probes.
    assert(norm(center*p.microns_per_degree-cm.eccentricityMicrons)<1e-9, ...
        'ConeOptics:Center','A coincident probe requires its actual mosaic-centered OI');
    center = 'mosaic-centered';
end
res = oiGet(oi,'height spatial resolution')*1e6;
diameters = cm.coneApertureDiametersMicrons(:);
if withEyeMovements
    opticalSupport=oiGet(oi,'spatial support')*1e6;
    sx=squeeze(opticalSupport(1,:,1)); sy=squeeze(opticalSupport(:,1,2));
    if ischar(center), opticalCenter=cm.eccentricityMicrons; else, opticalCenter=center*p.microns_per_degree; end
    sx=sx-mean(sx)+opticalCenter(1); sy=sy-mean(sy)+opticalCenter(2);
    shift=eyePathArcMin*p.microns_per_degree/60;
    largestKernel=cm.generateApertureKernel(max(diameters),res,false);
    margin=(size(largestKernel,1)-1)/2*res+res;
    assert(min(cm.coneRFpositionsMicrons(:,1))+min(shift(:,1))>min(sx)+margin && ...
        max(cm.coneRFpositionsMicrons(:,1))+max(shift(:,1))<max(sx)-margin && ...
        min(cm.coneRFpositionsMicrons(:,2))+min(shift(:,2))>min(sy)+margin && ...
        max(cm.coneRFpositionsMicrons(:,2))+max(shift(:,2))<max(sy)-margin, ...
        'ConeOptics:EyeCoverage','Eye path requires larger optical support; no automatic padding or extrapolation');
end
persistent cachedDiameters cachedResolution cachedGroups cachedKernels
if isequal(diameters,cachedDiameters) && isequal(res,cachedResolution)
    groups=cachedGroups; kernels=cachedKernels;
else
[uniqueDiameters,~,membership] = unique(diameters);
diameterRows=accumarray(membership,(1:n)',[],@(rows){rows});
groups = {}; kernels = {};
for k=1:numel(uniqueDiameters)
    assert(ceil(2*uniqueDiameters(k)/res)>=3, ...
        'ConeOptics:Resolution','OI sampling would skip a collecting aperture');
    kernel = cm.generateApertureKernel(uniqueDiameters(k),res,false);
    rows = diameterRows{k};
    if ~isempty(kernels) && isequal(kernel,kernels{end})
        groups{end} = [groups{end};rows]; %#ok<AGROW>
    else
        kernels{end+1} = kernel; groups{end+1} = rows; %#ok<AGROW>
    end
end
cachedDiameters=diameters; cachedResolution=res; cachedGroups=groups; cachedKernels=kernels;
end
counts = zeros(1,n); assigned = false(1,n);
if ~withEyeMovements
    [~,noisy,current,currentNoise]=cm.compute(oi,'nTrials',1,'nTimePoints',1, ...
        'withFixationalEyeMovements',false,'opticalImagePositionDegs',center,'padOIwithZeros',false);
    assert(isempty(noisy)&&isempty(current)&&isempty(currentNoise)&&isempty(cm.shadowMask), ...
        'ConeOptics:Ownership','Expected capture with no additional shadow mask required');
    support=cm.absorptionsDensitySpatialSupportMicrons;
    os=cm.computeEffectiveOSlengthAttenuationFactors(1:n);
    for g=1:numel(groups)
        rows=groups{g};
        for type=1:3
            cells=rows(cm.coneTypes(rows)==type); if isempty(cells),continue;end
            filtered=conv2(cm.absorptionsDensityFullMap(:,:,type),kernels{g},'same');
            F=griddedInterpolant(support,filtered,'linear','none');
            xy=cm.coneRFpositionsMicrons(cells,:);
            area=cm.computeApertureAreasMetersSquared(cells);area=area(:);
            % Native sampling stores the area product in a double array before OS/time scaling.
            counts(cells)=double(F([xy(:,2) xy(:,1)]).*area)'.*os(cells)*cm.integrationTime;
        end
        assert(~any(assigned(rows)),'ConeOptics:Identity','Cone assigned twice');
        assigned(rows)=true;
    end
    assert(all(assigned)&&all(isfinite(counts))&&all(counts>=0),'ConeOptics:Capture','Invalid capture');
    substepCounts=counts;
    return;
end
for g=1:numel(groups)
    rows = groups{g};
    % Keep the full geometry: subsetting can change ISETBio's coordinate conversion/centering.
    data = struct('positionUnits','microns','positions',cm.coneRFpositionsMicrons, ...
        'types',cm.coneTypes,'lightGatheringApertureDiameters',diameters, ...
        'blurApertureDiameterMicrons',diameters(rows(1)));
    sub = cone_optics_mosaic(data,cm.pigment,cm.macular,p,cm.integrationTime/samples);
    if withEyeMovements
        % Match ISETBio's existing mean-eye-position macular approximation for this frame.
        meanPath=mean(eyePathArcMin,1);
        if ischar(center), baseCenter=cm.eccentricityMicrons/p.microns_per_degree; else, baseCenter=center; end
        assert(any(max(cm.coneRFpositionsMicrons,[],1)>min(cm.coneRFpositionsMicrons,[],1)), ...
            'ConeOptics:EyeGeometry','Eye capture requires a nondegenerate frozen mosaic');
        % Translating the OI center constructs the native frame-mean macular map without private setters.
        mapCenter=baseCenter-meanPath/60;
        [~,noisy,current,currentNoise]=sub.compute(oi,'nTrials',1,'nTimePoints',1, ...
            'withFixationalEyeMovements',false,'opticalImagePositionDegs',mapCenter,'padOIwithZeros',false);
        support=sub.absorptionsDensitySpatialSupportMicrons;
        shift=(eyePathArcMin-meanPath)*p.microns_per_degree/60;
        margin=(size(kernels{g},1)-1)/2*res+res;
        x=sub.coneRFpositionsMicrons(rows,1)+shift(:,1)';
        y=sub.coneRFpositionsMicrons(rows,2)+shift(:,2)';
        assert(all(x(:)>min(support{2})+margin & x(:)<max(support{2})-margin) && ...
            all(y(:)>min(support{1})+margin & y(:)<max(support{1})-margin), ...
            'ConeOptics:EyeCoverage','Eye path leaves valid optical/aperture support; no extrapolation allowed');
        os=sub.computeEffectiveOSlengthAttenuationFactors(1:n);
        for type=1:3
            selected=find(sub.coneTypes(rows)==type); if isempty(selected), continue; end
            cells=rows(selected);
            % The density image is constant within the held frame; reuse its native kernel/interpolant.
            filtered=conv2(sub.absorptionsDensityFullMap(:,:,type),kernels{g},'same');
            F=griddedInterpolant(support,filtered,'linear','none');
            xx=x(selected,:); yy=y(selected,:);
            density=reshape(F([yy(:) xx(:)]),numel(cells),samples);
            area=cm.computeApertureAreasMetersSquared(cells);area=area(:);
            substepCounts(:,cells)=(density.*area.*os(cells)')'*(cm.integrationTime/samples);
        end
        value=sum(substepCounts,1);
    end
    assert(isempty(noisy)&&isempty(current)&&isempty(currentNoise), ...
        'ConeOptics:Ownership','Only expected light capture is allowed');
    assert(~any(assigned(rows)),'ConeOptics:Identity','Cone assigned twice');
    value=reshape(value,1,[]); counts(rows) = value(rows); assigned(rows) = true;
end
assert(all(assigned) && all(isfinite(counts)) && all(counts>=0), ...
    'ConeOptics:Capture','Missing or invalid cone captures');
end
