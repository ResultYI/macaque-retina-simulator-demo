function export_vr_linear_disk_batch(batchPath)
% Sequential input capture only. No model, fitting, noise or parameter changes.
batch=jsondecode(fileread(batchPath));
for k=1:numel(batch)
    export_vr_linear_disk_capture(batch(k).configuration,batch(k).output);
    marker=fopen([batch(k).output '.ready'],'w');
    assert(marker>0,'VRGray:Marker','Cannot commit completed capture marker');
    fclose(marker);
end
end
