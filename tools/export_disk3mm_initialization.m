function export_disk3mm_initialization(requestPath,outputPath)
% Same pinned initializer and approved complete-ellipse coverage repair.
c=load(requestPath);addpath(genpath(c.isetbio_root));
F=griddedInterpolant({c.grid_x_mm(:),c.grid_y_mm(:)},c.density_per_mm2,'linear','none');
p=retinalattice.configure(c.fov_degs,'cones','right eye');
p.rfSpacingExactFunction=@spacing;p.lambdaMinMicrons=c.minimum_spacing_um;p.rng=c.seed;
rawCheckpoint=[outputPath '.raw.mat'];
if isfile(rawCheckpoint)
 load(rawCheckpoint,'initial_raw_um','radius_um','oldCandidateCount','complete_candidates','extra_candidates', ...
  'phaseError','rng_before_extra','rng_after_extra');
 fprintf('RESUMED_EXACT_INITIALIZATION %d\n',size(initial_raw_um,1));
else
[base,radius_um]=retinalattice.initialize(c.fov_degs,'right eye',p,false,tic, ...
 'customDegsToMMsConversionFunction',@(deg)deg*c.mm_per_deg);
fprintf('PINNED_INITIALIZATION_RETURNED %d radius_um %.12g\n',size(base,1),radius_um);
r0=1e3*((c.fov_degs*1.33/2)*c.mm_per_deg);rows=2*r0;cols=rows;
scale=sqrt(3)/2;extraCols=round(cols/scale)-cols;
[X,Y]=meshgrid(1:(cols+extraCols),1:rows);X=X*scale;
for k=1:size(Y,2),Y(:,k)=Y(:,k)-mod(k-1,2)*.5;end
X=X*p.lambdaMinMicrons;Y=Y*p.lambdaMinMicrons;
keep=X>=-.1&X<=cols+.1&Y>=-.1&Y<=rows+.1;
center=[mean(X(keep)) mean(Y(keep))];legacyGrid=[X(keep)-center(1),Y(keep)-center(2)];
oldDomain=legacyGrid(p.domainFunction(legacyGrid,radius_um)<p.borderTolerance,:);
oldCandidateCount=size(oldDomain,1);
baseJ=round((base(:,1)+center(1))/(scale*p.lambdaMinMicrons));
baseI=round((base(:,2)+center(2))/p.lambdaMinMicrons+mod(baseJ-1,2)*.5);
baseOnPhase=[(baseJ*scale)*p.lambdaMinMicrons-center(1), ...
 (baseI-mod(baseJ-1,2)*.5)*p.lambdaMinMicrons-center(2)];
phaseError=max(sqrt(sum((base-baseOnPhase).^2,2)));
assert(phaseError<64*eps*max(abs(base(:))),'Pinned source hex phase reconstruction failed');
[oldSpacing,~]=spacing(oldDomain,'right eye',false);
maxDensity=max(RGCmodels.Watson.convert.spacingToDensityForHexGrid(oldSpacing));
clear X Y legacyGrid oldDomain oldSpacing keep baseJ baseI baseOnPhase
dx=scale*p.lambdaMinMicrons;R=radius_um+p.borderTolerance;
j=floor((center(1)-R)/dx)-1:ceil((center(1)+R)/dx)+1;
i=floor((center(2)-1.2247*R)/p.lambdaMinMicrons)-1:ceil((center(2)+1.2247*R)/p.lambdaMinMicrons)+1;
[J,I]=meshgrid(j,i);sx=(J*scale)*p.lambdaMinMicrons;sy=(I-mod(J-1,2)*.5)*p.lambdaMinMicrons;
wasLegacy=J>=1&J<=floor(cols+extraCols)&I>=1&I<=floor(rows)&sx>=-.1&sx<=cols+.1&sy>=-.1&sy<=rows+.1;
extendedXY=[sx(:)-center(1),sy(:)-center(2)];
within=p.domainFunction(extendedXY,radius_um)<p.borderTolerance;
assert(sum(wasLegacy(:)&within)==oldCandidateCount,'Source candidate phase/count mismatch');
extra=extendedXY(~wasLegacy(:)&within,:);
assert(oldCandidateCount+size(extra,1)==sum(within),'Incomplete ellipse candidates');
complete_candidates=sum(within);extra_candidates=size(extra,1);
clear J I sx sy wasLegacy extendedXY within
[extraSpacing,~]=spacing(extra,'right eye',false);
prob=RGCmodels.Watson.convert.spacingToDensityForHexGrid(extraSpacing)/maxDensity;
assert(all(prob>=0&prob<=1));rng_before_extra=rng;
selected=rand(size(extra,1),1)<prob;rng_after_extra=rng;
initial_raw_um=[base;extra(selected,:)];assert(size(unique(initial_raw_um,'rows'),1)==size(initial_raw_um,1));
fprintf('COMPLETE_INITIALIZATION_RETURNED %d candidates %d\n',size(initial_raw_um,1),complete_candidates);
clear base extra extraSpacing prob selected
save(rawCheckpoint,'initial_raw_um','radius_um','oldCandidateCount','complete_candidates','extra_candidates', ...
 'phaseError','rng_before_extra','rng_after_extra','-v7');
end
% Preserve the pinned129x129 logarithmic lookup coordinates and k9 operator.
% Remote rectangle corners can lie outside the density source although the
% entire ellipse is supported. Keep those values explicitly undefined instead
% of extrapolating; the relaxation checks every selected k9 value at every step.
ecc=logspace(log10(2),log10(max(sqrt(sum(initial_raw_um.^2,2)))+2),64);
ecc=[-fliplr(ecc) 0 ecc];[X,Y]=meshgrid(ecc);lookup_xy_um=[X(:) Y(:)];
rho=F(-lookup_xy_um(:,1)/1000,-lookup_xy_um(:,2)/1000);
lookup_spacing_um=nan(size(rho));valid=isfinite(rho)&rho>0;
lookup_spacing_um(valid)=sqrt(2e6./(sqrt(3)*rho(valid)));
metadata=struct('source_hex_phase_error_um',phaseError,'source_ellipse_candidates',oldCandidateCount, ...
 'added_ellipse_candidates',extra_candidates,'complete_ellipse_candidates',complete_candidates,'missing_candidates',0, ...
 'total_count',size(initial_raw_um,1),'radius_um',radius_um,'ellipse_axes',[1 1.2247], ...
 'lambda_min_um',p.lambdaMinMicrons,'seed',c.seed,'point_deletion_or_density_scale_change',false, ...
 'sampling_order','pinned source draws first; missing ellipse candidates next with continued same RNG', ...
 'lookup_coordinates','unchanged pinned logarithmic129x129','unsupported_lookup_values',sum(~valid), ...
 'unsupported_lookup_policy','NaN; selected k9 nodes must all be supported, checked every force evaluation');
initialization_json=jsonencode(metadata);
save(outputPath,'initial_raw_um','radius_um','lookup_spacing_um','lookup_xy_um', ...
 'initialization_json','rng_before_extra','rng_after_extra','-v7');
fprintf('COVERED_INITIALIZATION %s\n',initialization_json);
 function [s,e]=spacing(raw,~,~)
  rho=F(-raw(:,1)/1000,-raw(:,2)/1000);assert(all(isfinite(rho))&&all(rho>0));
  s=sqrt(2e6./(sqrt(3)*rho));e=sqrt(sum(raw.^2,2));
 end
end
