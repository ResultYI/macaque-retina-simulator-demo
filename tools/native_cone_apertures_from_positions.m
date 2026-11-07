function [diameters,areas,cm]=native_cone_apertures_from_positions(xy)
% Complete neighbour support in microns; no externally assigned diameters.
assert(size(xy,2)==2 && size(xy,1)>=7 && all(isfinite(xy(:))));
data=struct('positionUnits','microns','positions',xy,'types',ones(size(xy,1),1), ...
    'lightGatheringApertureDiameters',ones(size(xy,1),1));
% Bootstrap the import; the setter replaces these placeholders from geometry.
evalc("cm=cMosaic('coneData',data,'useParfor',false,'noiseFlag','none',"+ ...
    "'eccVaryingConeAperture',true,'eccVaryingConeBlur',false,"+ ...
    "'rodIntrusionAdjustedConeAperture',false,"+ ...
    "'anchorAllEccVaryingParamsToTheirFovealValues',false);");
evalc("cm.coneApertureModifiers=struct('shape','Pillbox');");
diameters=cm.coneApertureDiametersMicrons(:);
areas=cm.computeApertureAreasMetersSquared(1:numel(diameters));areas=areas(:);
end
