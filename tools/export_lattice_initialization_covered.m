function export_lattice_initialization_covered(requestPath,legacyDiagnosticPath,outputPath)
% Approved same-ellipse coverage repair, independent of the active model.
% Source hex phase, legacy draws, density and ellipse are retained; added
% candidates continue the original Bernoulli sampler with no reset or quota.
c=load(requestPath);legacy=load(legacyDiagnosticPath);addpath(genpath(c.isetbio_root));
F=griddedInterpolant({c.grid_x_mm(:),c.grid_y_mm(:)},c.density_per_mm2,'linear','none');
p=retinalattice.configure(c.fov_degs,'cones','right eye');
p.rfSpacingExactFunction=@spacing;p.lambdaMinMicrons=c.minimum_spacing_um;p.rng=c.seed;
[base,radius_um]=retinalattice.initialize(c.fov_degs,'right eye',p,false,tic, ...
 'customDegsToMMsConversionFunction',@(deg)deg*c.mm_per_deg);
assert(isequal(base,legacy.initial_raw_um)&&radius_um==legacy.radius_um,'Legacy initialization parity failed');
% Reconstruct the source's pre-crop grid phase, without changing its points.
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
assert(phaseError<64*eps*max(abs(base(:))),'Original source grid phase reconstruction failed');
[oldSpacing,~]=spacing(oldDomain,'right eye',false);
maxDensity=max(RGCmodels.Watson.convert.spacingToDensityForHexGrid(oldSpacing));
clear X Y legacyGrid oldDomain oldSpacing keep
% Continue the SAME hex phase outside the old square, only inside the already
% declared ellipse. The original random draws/cells are kept; source rng state
% continues for the omitted candidates, with the unchanged density probability.
dx=scale*p.lambdaMinMicrons;R=radius_um+p.borderTolerance;
j=floor((center(1)-R)/dx)-1:ceil((center(1)+R)/dx)+1;
i=floor((center(2)-1.2247*R)/p.lambdaMinMicrons)-1:ceil((center(2)+1.2247*R)/p.lambdaMinMicrons)+1;
[J,I]=meshgrid(j,i);sx=(J*scale)*p.lambdaMinMicrons;sy=(I-mod(J-1,2)*.5)*p.lambdaMinMicrons;
wasLegacy=J>=1&J<=floor(cols+extraCols)&I>=1&I<=floor(rows)&sx>=-.1&sx<=cols+.1&sy>=-.1&sy<=rows+.1;
extendedXY=[sx(:)-center(1),sy(:)-center(2)];
within=p.domainFunction(extendedXY,radius_um)<p.borderTolerance;
assert(sum(wasLegacy(:)&within)==oldCandidateCount,'Legacy candidate coverage/count mismatch');
extra=extendedXY(~wasLegacy(:)&within,:);
assert(oldCandidateCount+size(extra,1)==sum(within),'Initialization candidate domain has a gap');
assert(min(extendedXY(:,1))<-R&&max(extendedXY(:,1))>R);
assert(min(extendedXY(:,2))<-1.2247*R&&max(extendedXY(:,2))>1.2247*R);
[extraSpacing,~]=spacing(extra,'right eye',false);
prob=RGCmodels.Watson.convert.spacingToDensityForHexGrid(extraSpacing)/maxDensity;
assert(all(prob>=0&prob<=1));rng_before_extra=rng;
selected=rand(size(extra,1),1)<prob;rng_after_extra=rng;
initial_raw_um=[base;extra(selected,:)];assert(size(unique(initial_raw_um,'rows'),1)==size(initial_raw_um,1));
[lookup_spacing_um,lookup_xy_um]=retinalattice.compute.rfSpacingLookUpTables(initial_raw_um,'right eye',false,@spacing,64);
[springs,index,triangles]=retinalattice.compute.updatedTrussConnectivity(initial_raw_um,p.domainFunction,radius_um,p.borderTolerance);
[official_force,official_desired]=retinalattice.compute.updatedTrussForces(initial_raw_um,lookup_xy_um,lookup_spacing_um, ...
 springs,index,[],p.rfSpacingFastFunction,true,false);
metadata=struct('legacy_cells_preserved',size(base,1),'omitted_hex_candidates',size(extra,1), ...
 'legacy_ellipse_candidates',oldCandidateCount,'complete_ellipse_candidates',sum(within), ...
 'missing_ellipse_candidates_after_union',0,'legacy_source_hex_phase_error_um',phaseError, ...
 'candidate_rectangle_raw_um',[min(extendedXY,[],1);max(extendedXY,[],1)], ...
 'complete_candidate_domain_raw_um',[min(extendedXY(within,:),[],1);max(extendedXY(within,:),[],1)], ...
 'extra_candidate_range_raw_um',[min(extra,[],1);max(extra,[],1)], ...
 'added_by_unchanged_density_sampler',sum(selected),'sampler_expected_extra_count',sum(prob), ...
 'sampler_extra_count_variance',sum(prob.*(1-prob)),'total_count',size(initial_raw_um,1), ...
 'radius_um_unchanged',radius_um,'ellipse_axes',[1 1.2247],'hex_phase_center',center, ...
 'lambda_min_um_unchanged',p.lambdaMinMicrons,'seed_unchanged',c.seed, ...
 'candidate_grid_coverage_changed',true,'analytic_domain_changed',false,'target_density_changed',false, ...
 'sampling_order','exact legacy source draws first; omitted candidates next in source column-major hex order', ...
 'point_deletion_or_LMS_change',false);
initialization_json=jsonencode(metadata);
save(outputPath,'initial_raw_um','radius_um','lookup_spacing_um','lookup_xy_um','springs','triangles', ...
 'official_force','official_desired','initialization_json','rng_before_extra','rng_after_extra','-v7');
fprintf('COVERED_INITIALIZATION %s\n',initialization_json);
 function [s,e]=spacing(raw,~,~)
  rho=F(-raw(:,1)/1000,-raw(:,2)/1000);assert(all(isfinite(rho))&&all(rho>0));
  s=sqrt(2e6./(sqrt(3)*rho));e=sqrt(sum(raw.^2,2));
 end
end
