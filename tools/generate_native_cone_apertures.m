function generate_native_cone_apertures(inputPath, outputPath)
% Python selects complete neighbour support; cMosaic owns the aperture model.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root,'vendor_pinned','isetcam')));
addpath(genpath(fullfile(root,'vendor_pinned','isetbio')));
d=load(inputPath);diameters=zeros(d.count,1);areas=diameters;
for b=1:numel(d.positions)
    xy=d.positions{b}; rows=d.query_rows{b}; local=d.local_rows{b};
    [native,area]=native_cone_apertures_from_positions(xy);
    diameters(rows)=native(local);areas(rows)=area(local);
    if mod(b,25)==0, fprintf('NATIVE_APERTURES %d/%d blocks\n',b,numel(d.positions));end
end
assert(all(isfinite(diameters)) && all(diameters>0) && all(areas>0));
engine='cMosaic.computeConeApertures via public coneApertureModifiers setter';
save(outputPath,'diameters','areas','engine','-v7');
end
