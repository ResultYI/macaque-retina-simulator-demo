function export_isetbio_geometry(requestPath, outputPath)
% Only geometry. Supplied macaque density replaces both Watson spacing defaults.
c=load(requestPath);oldPath=path;oldRng=rng;
cleanup=onCleanup(@()restoreState(oldPath,oldRng)); %#ok<NASGU>
addpath(genpath(c.isetbio_root));
F=griddedInterpolant({c.grid_x_mm(:),c.grid_y_mm(:)},c.density_per_mm2,'linear','none');
params=struct('sourceLatticeSizeDegs',c.fov_degs,'neuronType','cones', ...
    'whichEye','right eye','exportHistoryToFile',false,'visualizeConvergence',false, ...
    'useParfor',false,'maxIterations',c.max_iterations,'eccentricityLookUpTableSamplesNum',64);
positions_um=retinalattice.generatePatch(params,'randomSeed',c.seed, ...
    'customDegsToMMsConversionFunction',@(deg)deg*c.mm_per_deg, ...
    'customRFspacingFunction',@spacing,'customMinRFspacing',c.minimum_spacing_um);
assert(size(positions_um,2)==2 && all(isfinite(positions_um(:))),'MacaqueGeometry:Positions','Invalid generated positions');
metadata=struct('matlab_version',version,'generator','retinalattice.generatePatch', ...
    'density_species','Macaca nemestrina','human_density_used',false, ...
    'polarity_rule','density evaluated at minus raw coordinates; source reverses output polarity', ...
    'source_precision','upstream smoothGrid stores single history; generatePatch returns double', ...
    'eye','right eye','position_role','functional input RF, not soma','count',size(positions_um,1));
metadata_json=jsonencode(metadata);save(outputPath,'positions_um','metadata_json','-v7');
    function [spacing_um,eccentricity_um]=spacing(raw_um,~,~)
        rho=F(-raw_um(:,1)/1000,-raw_um(:,2)/1000);
        assert(all(isfinite(rho)) && all(rho>0),'MacaqueGeometry:Density','Macaque density grid unsupported; no human fallback');
        spacing_um=sqrt(2e6./(sqrt(3)*rho));
        eccentricity_um=sqrt(sum(raw_um.^2,2));
    end
end
function restoreState(p,r)
path(p);rng(r);
end
