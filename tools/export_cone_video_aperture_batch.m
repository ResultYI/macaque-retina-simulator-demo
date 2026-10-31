function export_cone_video_aperture_batch(path)
jobs=jsondecode(fileread(path));
for k=1:numel(jobs)
    export_cone_video_aperture_tile(jobs(k).configuration,jobs(k).output);
    fid=fopen([jobs(k).output '.ready'],'w');assert(fid>0);fclose(fid);
end
end
