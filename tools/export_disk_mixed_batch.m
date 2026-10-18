function export_disk_mixed_batch(batchPath)
% Sequential capture of fixed global tiles in one pinned MATLAB/ISET session.
batch=jsondecode(fileread(batchPath));
for k=1:numel(batch)
    export_disk_mixed_reference(batch(k).configuration,batch(k).output);
    marker=fopen([batch(k).output '.ready'],'w');
    assert(marker>0,'DiskInput:Marker','Cannot write completed-capture marker');
    fclose(marker);
end
end
