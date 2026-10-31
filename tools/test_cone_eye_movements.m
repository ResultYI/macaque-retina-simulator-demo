function test_cone_eye_movements(configPath, outputPath, filterOraclePath)
% Compare the compatibility filter, unmodified trajectory and native eye-path captures.
c=jsondecode(fileread(configPath)); p=c.profile;
addpath(genpath(p.isetcam_root)); addpath(genpath(p.isetbio_root));
assert(~isfile(outputPath)); out=fileparts(outputPath);
baseline=visualizableFixationalEM(); baseline.randomSeed=c.seed;
try
    baseline.compute(c.duration_seconds+.002,.001,1,false,'centerPathsAtSpecificTimeMsec',0);
    expectedPosition=squeeze(baseline.emPosArcMin(1,:,:));
catch failure
    assert(contains(failure.message,'sgolayfilt') && exist('sgolayfilt','file')==0);
    snapshot=struct(baseline);
    raw=snapshot.emPosTimeSeries;
    stabilization=snapshot.stabilizationStepsNum;
    steps=snapshot.tStepsNum;
    kept=raw(:,stabilization+1:steps-1); kept=kept-kept(:,1);
    expectedPosition=kept'*snapshot.scalarToArcMin;
end
aPath=fullfile(out,'eye_test_a.mat'); bPath=fullfile(out,'eye_test_b.mat');
prepare_cone_eye_movements(configPath,aPath); prepare_cone_eye_movements(configPath,bPath);
a=load(aPath); b=load(bPath);
assert(isequal(a.eye_position_arcmin,b.eye_position_arcmin) && isequal(a.em_positions_arcmin,b.em_positions_arcmin));
assert(isequal(a.eye_position_arcmin,expectedPosition),'ConeEM:Trajectory','Velocity compatibility altered the native generated positions');
addpath(fullfile(fileparts(mfilename('fullpath')),'isetbio_compat'),'-begin');
oracle=load(filterOraclePath); filterError=0;
for k=1:numel(oracle.windows)
    filtered=sgolayfilt(oracle.values,3,oracle.windows(k));
    filterError=max(filterError,max(abs(filtered(:)-oracle.expected(:,k))));
end
assert(filterError<1e-10);
pigment=cPhotoPigment('opticalDensity',p.optical_density_LMS); macular=Macular('wave',pigment.wave,'density',.35);
data=struct('positionUnits','microns','positions',[298 -1;300 1;302 -1;301 2], ...
    'types',[1;2;3;1],'lightGatheringApertureDiameters',[1.2;2.4;3.4;4.8],'blurApertureDiameterMicrons',3);
cm=cone_optics_mosaic(data,pigment,macular,p,.01); f=p.microns_per_degree*1e-6/tan(pi/180);
oi=oiCreate('human mw'); oi=oiSet(oi,'optics',opticsCreate('human mw',.0015,f)); oi=oiSet(oi,'wave',pigment.wave);
[x,y]=meshgrid(-32:32,-32:32); pattern=1+.5*x/32+.1*y/32;
oi=oiSet(oi,'photons',repmat(pattern,[1 1 numel(pigment.wave)])*1e15);
oi=oiSet(oi,'fov',2*atan(65*.5e-6/(2*f))*180/pi); center=[300/p.microns_per_degree 0];
path=[linspace(-.5,1.2,11)' .3+.2*sin(linspace(0,pi,11)')];
[optimized,substep]=capture_cone_optics(cm,oi,center,4,p,path);
independent=zeros(11,4);
for k=1:4
    referenceData=data; referenceData.blurApertureDiameterMicrons=data.lightGatheringApertureDiameters(k);
    native=cone_optics_mosaic(referenceData,pigment,macular,p,.01/11);
    native.emGenSequence(.01,'microsaccadeType','none','randomSeed',1,'nTrials',1);
    em=native.fixEMobj; em.emPosArcMin=reshape(path,[1 11 2]); em.emPosMicrons=em.emPosArcMin*p.microns_per_degree/60;
    values=native.compute(oi,'withFixationalEyeMovements',true,'nTrials',1, ...
        'opticalImagePositionDegs',center,'padOIwithZeros',false);
    independent(:,k)=reshape(values(1,:,k),[],1);
end
nativeError=max(abs(substep(:)-independent(:))./max(abs(independent(:)),1e-30));
assert(nativeError<1e-12 && max(abs(optimized-sum(independent,1)))<1e-10);
fixed=capture_cone_optics(cm,oi,center,4,p);
zero=capture_cone_optics(cm,oi,center,4,p,zeros(11,2));
zeroError=max(abs(fixed-zero)./fixed); assert(zeroError<1e-12);
positive=capture_cone_optics(cm,oi,center,4,p,repmat([.5 0],11,1)); assert(positive(1)>fixed(1));
try
    capture_cone_optics(cm,oi,center,4,p,[1e6 1e6]); error('ConeEM:Test','Missing coverage rejection');
catch failure
    assert(strcmp(failure.identifier,'ConeOptics:EyeCoverage'));
end
summary=struct('passed',true,'native_position_trajectory_unchanged',true,'fixed_seed_reproducible',true, ...
    'filter_vs_scipy_max_error',filterError,'substep_vs_native_relative_error',nativeError, ...
    'zero_eye_vs_fixed_relative_error',zeroError,'positive_sampling_shift_sign_pass',true, ...
    'out_of_support_rejected',true,'phototransduction_run',false,'downstream_circuit_run',false);
fid=fopen(outputPath,'w','n','UTF-8'); assert(fid~=-1); guard=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(summary)); fprintf('CONE_EM_TEST_PASS\n');
end
