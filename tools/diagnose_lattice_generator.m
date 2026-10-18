function diagnose_lattice_generator(requestPath,frozenPath,outputPath)
% One initialization/force diagnostic, no relaxation sequence or neural work.
c=load(requestPath);saved=load(frozenPath);addpath(genpath(c.isetbio_root));
F=griddedInterpolant({c.grid_x_mm(:),c.grid_y_mm(:)},c.density_per_mm2,'linear','none');
params=retinalattice.configure(c.fov_degs,'cones','right eye');
params.rfSpacingExactFunction=@spacing;params.lambdaMinMicrons=c.minimum_spacing_um;params.rng=c.seed;
[initial_raw_um,radius_um]=retinalattice.initialize(c.fov_degs,'right eye',params,false,tic, ...
 'customDegsToMMsConversionFunction',@(deg)deg*c.mm_per_deg);
[lookup_spacing_um,lookup_xy_um]=retinalattice.compute.rfSpacingLookUpTables(initial_raw_um,'right eye',false,@spacing,64);
frozen_raw_um=-saved.positions_um;
[springs,indices,triangles]=retinalattice.compute.updatedTrussConnectivity(frozen_raw_um,params.domainFunction,radius_um,params.borderTolerance);
[official_force,official_desired]=retinalattice.compute.updatedTrussForces(frozen_raw_um,lookup_xy_um,lookup_spacing_um, ...
 springs,indices,[],params.rfSpacingFastFunction,true,false);
[nn_idx,nn_distance_um]=knnsearch(frozen_raw_um,frozen_raw_um,'K',2);
[minimum_nn_um,closest]=min(nn_distance_um(:,2));partner=nn_idx(closest,2);
pair_before=frozen_raw_um([closest partner],:);
pair_after_euler=pair_before+.2*official_force([closest partner],:);
pair_distance_after_euler_um=norm(pair_after_euler(1,:)-pair_after_euler(2,:));
steps=.2*official_force;actual_move=sqrt(sum(steps.^2,2));
reported_metric=sqrt(sum(.2*official_force.^2,2));
diagnostic=struct('radius_um',radius_um,'minimum_nn_um',minimum_nn_um,'closest_indices_one_based',[closest partner], ...
 'pair_distance_after_one_fresh_graph_euler_um',pair_distance_after_euler_um, ...
 'actual_max_step_um',max(actual_move),'actual_p30_step_um',prctile(actual_move,30), ...
 'source_logged_p30_metric_um',prctile(reported_metric,30),'dTolerance_um',params.dTolerance, ...
 'borderTolerance_um',params.borderTolerance,'minimum_spacing_um',params.lambdaMinMicrons);
diagnostic_json=jsonencode(diagnostic);
save(outputPath,'initial_raw_um','radius_um','lookup_spacing_um','lookup_xy_um','frozen_raw_um', ...
 'springs','triangles','official_force','official_desired','diagnostic_json','-v7');
fprintf('LATTICE_DIAGNOSTIC %s\n',diagnostic_json);
 function [s,e]=spacing(raw,~,~)
  rho=F(-raw(:,1)/1000,-raw(:,2)/1000);assert(all(isfinite(rho))&&all(rho>0));
  s=sqrt(2e6./(sqrt(3)*rho));e=sqrt(sum(raw.^2,2));
 end
end
