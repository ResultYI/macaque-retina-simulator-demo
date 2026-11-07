function [groups,kernels,representatives]=native_cone_kernel_groups(mosaic,diameters,resolution)
% Exact native kernels; grouping never rounds or changes a Cone diameter.
[values,~,membership]=unique(diameters(:));
rows=accumarray(membership,(1:numel(diameters))',[],@(r){r});
groups={};kernels={};representatives=[];
for k=1:numel(values)
    assert(ceil(2*values(k)/resolution)>=3,'ConeOptics:Resolution','Aperture sampling too coarse');
    kernel=mosaic.generateApertureKernel(values(k),resolution,false);
    if ~isempty(kernels) && isequal(kernel,kernels{end})
        groups{end}=[groups{end};rows{k}];
    else
        kernels{end+1}=kernel;groups{end+1}=rows{k};representatives(end+1)=values(k);
    end
end
end
