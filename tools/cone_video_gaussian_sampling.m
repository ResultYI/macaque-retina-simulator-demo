function [matrix,area,diagnostic]=cone_video_gaussian_sampling(mosaic,xy,diameters,zones,x,y)
% Compatibility entry name; native Pillbox kernels and areas at actual Cone positions.
n=size(xy,1);height=numel(y);width=numel(x);dx=x(2)-x(1);dy=y(2)-y(1);
zones=zones(:);diameters=diameters(:);
assert(dx>0 && dy>0 && abs(dx-dy)<1e-6*dx);
col=(xy(:,1)-x(1))/dx+1;row=(xy(:,2)-y(1))/dy+1;
col0=floor(col);row0=floor(row);fx=col-col0;fy=row-row0;
rowParts={};columnParts={};weightParts={};at=0;kernelSums=[];
[area,restoreNative]=cone_native_aperture(mosaic,diameters, ...
    struct('shape','Pillbox')); %#ok<NASGU>
assert(isequal(zones,diameters),'ConeVideo:Diameter','Rounded/custom kernel zones are retired');
[groups,kernels,representatives]=native_cone_kernel_groups(mosaic,diameters,dx);
for g=1:numel(groups)
    d=representatives(g);kernel=kernels{g};
    selected=groups{g};ry=(size(kernel,1)-1)/2;rx=(size(kernel,2)-1)/2;
    assert(all(row0(selected)>ry & row0(selected)+1+ry<=height & ...
               col0(selected)>rx & col0(selected)+1+rx<=width),'ConeVideo:Coverage','Missing optical halo');
    for iy=1:size(kernel,1)
        for ix=1:size(kernel,2)
            if kernel(iy,ix)==0,continue;end
            for sy=0:1
                for sx=0:1
                    at=at+1;
                    rowParts{at}=selected; %#ok<AGROW>
                    columnParts{at}=row0(selected)+sy+iy-ry-1+ ...
                        (col0(selected)+sx+ix-rx-2)*height; %#ok<AGROW>
                    wy=(1-fy(selected))*(1-sy)+fy(selected)*sy;
                    wx=(1-fx(selected))*(1-sx)+fx(selected)*sx;
                    weightParts{at}=kernel(iy,ix)*wy.*wx; %#ok<AGROW>
                end
            end
        end
    end
    kernelSums(end+1,:)=[d sum(kernel(:))]; %#ok<AGROW>
end
matrix=sparse(vertcat(rowParts{:}),vertcat(columnParts{:}),vertcat(weightParts{:}),n,height*width);
mass=full(sum(matrix,2));assert(max(abs(mass-1))<1e-12 && all(nonzeros(matrix)>=0));
diagnostic=struct('aperture_backend','isetbio_native_aperture_v1', ...
    'area_engine','cMosaic.computeApertureAreasMetersSquared','unit_sum_error',max(abs(mass-1)),'nnz',nnz(matrix), ...
    'kernel_diameter_and_sum',kernelSums,'area_applications',1,'optical_PSF_reapplied',false);
end
