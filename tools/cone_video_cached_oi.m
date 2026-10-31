function [oi,otf]=cone_video_cached_oi(scene,model)
% Repeat pinned ISET SI/MW operations, reusing the unchanged spectral OTF.
oi=model.baseOI;
oi=oiSet(oi,'photons',oiCalculateIrradiance(scene,oi));
method=lower(opticsGet(oiGet(oi,'optics'),'offaxis method'));
if strcmp(method,'cos4th')
    oi=opticsCos4th(oi);
else
    assert(ismember(method,{'skip','none',''}),'ConeVideo:Optics','Unknown off-axis method');
end
oi=oiPadValue(oi,model.padSize,'zero photons',sceneGet(scene,'distance'));
photons=oiGet(oi,'photons');
if isempty(model.otf),otf=oiCalculateOTF(oi,model.wave,'mm');else,otf=model.otf;end
assert(isequal(size(photons),size(otf)),'ConeVideo:OTF','Cached optical grid changed');
for k=1:numel(model.wave)
    photons(:,:,k)=abs(ifft2(otf(:,:,k).*fft2(photons(:,:,k))));
end
oi=oiSet(oi,'photons',photons);
end
