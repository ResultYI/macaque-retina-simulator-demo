function render_cone_video(configPath, capturePath)
% Render saved optical captures on one fixed absolute scale; no model computation.
c=jsondecode(fileread(configPath)); d=load(c.input_mat); r=load(capturePath);
imageOnly=isfield(c,'image_only') && c.image_only;
meta=jsondecode(r.manifest_json);
withEyeMovements=isfield(d,'frame_eye_midpoint_arcmin');
assert(strcmp(meta.request_signature,c.request_signature) && ~meta.phototransduction_applied);
blocked=isfield(c,'capture_blocks');
if blocked
    edges=c.window_edges_seconds(:);dt=diff(edges);frames=numel(dt);
    blockIndex=1;blockFirst=c.capture_blocks(1).first+1;blockLast=c.capture_blocks(1).stop;
    rates=r.expected_counts./dt(blockFirst:blockLast);maximum=c.maximum_rate_Rstar_per_cone_s;
else
    edges=r.window_edges_seconds(:);dt=diff(edges);rates=r.expected_counts./dt;
    frames=size(rates,1);maximum=max(rates(:));
end
duration=edges(end)-edges(1);
assert((frames>1 || (imageOnly && frames==1)) && duration>0 && all(abs(dt-dt(1))<1e-12));
assert(all(isfinite(rates(:))) && all(rates(:)>=0));
step=10^floor(log10(maximum))/2; upper=ceil(maximum/step)*step;
xy=d.is_mm/.223; radii=r.diameters_um(:)/(2*223); theta=(0:19)*2*pi/20;
vx=xy(:,1)+radii*cos(theta); vy=xy(:,2)+radii*sin(theta);
vertices=[reshape(vx',[],1),reshape(vy',[],1)]; faces=reshape(1:numel(vx),20,[])';
palette=[.82 .19 .25; .10 .57 .34; .18 .39 .80]; bounds=d.plot_bounds_mm/.223;
out=fileparts(capturePath);
if isfield(c,'output_directory'),out=c.output_directory;end
highDefinition=isfield(c,'high_definition') && c.high_definition;
pipeEncoding=isfield(c,'encoder_arguments');
videoFile=fullfile(out,sprintf('cone_video_comparison_%gs.mp4',duration));
if highDefinition,videoFile=fullfile(out,sprintf('cone_video_comparison_%gs_4K.mp4',duration));end
rasterPath=fullfile(out,'cone_raster.mat'); rasterized=isfile(rasterPath);
if rasterized
    raster=load(rasterPath);
    assert(isequal(raster.diameters_um(:),r.diameters_um(:)) && ...
        isequal(raster.bounds_mm(:),d.plot_bounds_mm(:)) && max(raster.cone_ids(:))<=numel(d.types));
    ids=raster.cone_ids; scale=raster.supersampling; responseColors=parula(256);
    typeImage=rasterImage([1 1 1;palette(d.types,:)],ids,scale);
end
assert(~isfile(videoFile),'ConeVideo:Output','Preserve the previous rendered video');
set(groot,'defaultTextInterpreter','none','defaultAxesTickLabelInterpreter','none','defaultLegendInterpreter','none');
fig=figure('Visible','off','Color','white','Position',[50 50 1620 600],'PaperPositionMode','auto');
resolution=100;fontSize=11;
if highDefinition
    resolution=200;
    set(fig,'PaperUnits','inches','PaperPosition',[0 0 19.2 7.2],'PaperSize',[19.2 7.2],'PaperPositionMode','manual');
end
if isfield(c,'render_size')
    resolution=c.render_dpi;fontSize=c.render_font_size;
    set(fig,'PaperUnits','inches','PaperPosition',[0 0 c.render_size(:)'/resolution], ...
        'PaperSize',c.render_size(:)'/resolution,'PaperPositionMode','manual');
end
figureGuard=onCleanup(@()close(fig)); panels=[.055 .20 .255 .69; .365 .20 .255 .69; .675 .20 .255 .69];
for p=1:3
    ax=axes(fig,'Position',panels(p,:)); hold(ax,'on');
    if p==1
        inputHandle=image(ax,'XData',d.plot_x_mm([1 end])/.223,'YData',d.plot_y_mm([1 end])/.223, ...
            'CData',rgbFrame(d.rgb_frames,1));
        if withEyeMovements, title(ax,'A  Retinal input'); else, title(ax,'A  Input frame'); end
    elseif p==2
        if rasterized
            image(ax,'XData',bounds(1:2),'YData',bounds([4 3]),'CData',typeImage);
        else
            patch(ax,'Faces',faces,'Vertices',vertices,'FaceVertexCData',palette(d.types,:), ...
                'FaceColor','flat','EdgeColor','none');
        end
        title(ax,'B  L/M/S cone mosaic'); handles=gobjects(3,1);
        for t=1:3, handles(t)=plot(ax,nan,nan,'o','MarkerFaceColor',palette(t,:),'MarkerEdgeColor','none','MarkerSize',fontSize*.55); end
        legend(ax,handles,{'L','M','S'},'Orientation','horizontal','Position',[.42 .055 .16 .04],'Box','off','FontSize',fontSize);
    else
        if rasterized
            responseHandle=image(ax,'XData',bounds(1:2),'YData',bounds([4 3]),'CData', ...
                responseImage(rates(1,:),upper,responseColors,ids,scale));
        else
            responseHandle=patch(ax,'Faces',faces,'Vertices',vertices,'FaceVertexCData',rates(1,:)', ...
                'FaceColor','flat','EdgeColor','none');
        end
        title(ax,'C  Cone photon catches'); colormap(ax,parula(256)); clim(ax,[0 upper]);
        bar=colorbar(ax,'Position',[.943 .24 .012 .58]); bar.Ticks=linspace(0,upper,5);
        bar.Label.String='Expected isomerizations (R*/cone/s)'; bar.Label.FontName='Arial';bar.FontSize=fontSize;
        if imageOnly || pipeEncoding
            bar.AxisLocation='out';bar.Label.Units='normalized';bar.Label.Position=[4 .5 0];
        end
    end
    xlim(ax,bounds(1:2)); ylim(ax,bounds(3:4));
    set(ax,'YDir','normal','DataAspectRatio',[1 1 1],'FontName','Arial','FontSize',fontSize,'Box','on','Layer','top');
    xlabel(ax,'x (deg)'); ylabel(ax,'y (deg)'); ax.Position=panels(p,:);
end
selected=unique([1 floor(frames/2)+1 frames]);
if imageOnly
    assert(~pipeEncoding);
elseif pipeEncoding
    assert(rasterized && ~isfile(c.rgb_master) && ~isfile(c.playback_video));
    encoderArgs=java.util.ArrayList;
    for i=1:numel(c.encoder_arguments),encoderArgs.add(java.lang.String(c.encoder_arguments{i}));end
    builder=java.lang.ProcessBuilder(encoderArgs);
    environment=builder.environment;environment.put('PATH',[strjoin(c.encoder_dll_paths,';') ';' getenv('PATH')]);
    builder.redirectError(java.io.File(fullfile(out,'encoder_stderr.log')));
    builder.redirectOutput(java.io.File(fullfile(out,'encoder_stdout.log')));
    encoder=builder.start;stream=encoder.getOutputStream;
    encoderGuard=onCleanup(@()stopEncoder(encoder,stream));
else
    writer=VideoWriter(videoFile,'MPEG-4');writer.FrameRate=1/dt(1);writer.Quality=95;
    if highDefinition,writer.Quality=100;end
    open(writer);writerGuard=onCleanup(@()close(writer));
end
if highDefinition && ~pipeEncoding && ~imageOnly
    masterFile=fullfile(out,sprintf('cone_video_comparison_%gs_4K_RGB.avi',duration));
    assert(~isfile(masterFile),'ConeVideo:Output','Preserve the previous RGB master');
    master=VideoWriter(masterFile,'Uncompressed AVI');master.FrameRate=1/dt(1);
    open(master);masterGuard=onCleanup(@()close(master));
end
for k=1:frames
    if blocked && k>blockLast
        blockIndex=blockIndex+1;blockFirst=c.capture_blocks(blockIndex).first+1;blockLast=c.capture_blocks(blockIndex).stop;
        block=load(c.capture_blocks(blockIndex).path,'expected_counts');rates=block.expected_counts./dt(blockFirst:blockLast);
        assert(all(isfinite(rates(:))) && all(rates(:)>=0));
    end
    if blocked,currentRate=rates(k-blockFirst+1,:);else,currentRate=rates(k,:);end
    inputHandle.CData=rgbFrame(d.rgb_frames,k);
    if rasterized
        responseHandle.CData=responseImage(currentRate,upper,responseColors,ids,scale);
    else
        responseHandle.FaceVertexCData=currentRate';
    end
    if withEyeMovements
        inputHandle.XData=d.plot_x_mm([1 end])/.223-d.frame_eye_midpoint_arcmin(k,1)/60;
        inputHandle.YData=d.plot_y_mm([1 end])/.223-d.frame_eye_midpoint_arcmin(k,2)/60;
    end
    pixels=print(fig,'-RGBImage',sprintf('-r%d',resolution));
    if mod(size(pixels,1),2), pixels(end+1,:,:)=255; end
    if mod(size(pixels,2),2), pixels(:,end+1,:)=255; end
    if imageOnly
        assert(size(pixels,2)==c.render_size(1) && size(pixels,1)==c.render_size(2));
    elseif pipeEncoding
        assert(size(pixels,2)==c.render_size(1) && size(pixels,1)==c.render_size(2));
        bytes=typecast(reshape(permute(pixels,[3 2 1]),[],1),'int8');
        for firstByte=1:1048576:numel(bytes)
            lastByte=min(firstByte+1048575,numel(bytes));chunk=bytes(firstByte:lastByte);
            stream.write(chunk,0,numel(chunk));
        end
    else
        writeVideo(writer,pixels);
        if highDefinition,writeVideo(master,pixels);end
    end
    if ismember(k,selected)
        name=sprintf('frame_%03d',k-1); imwrite(pixels,fullfile(out,[name '.png']));
        if highDefinition
            % PNG retains the exact RGB pixels also written to the uncompressed master.
        elseif numel(d.types)>10000
            exportgraphics(fig,fullfile(out,[name '.pdf']),'ContentType','image','Resolution',160);
        else
            exportgraphics(fig,fullfile(out,[name '.pdf']),'ContentType','vector');
        end
    end
    fprintf('CONE_VIDEO_RENDER %d/%d\n',k,frames);
end
if imageOnly
    imwrite(imresize(pixels,[1440 3840],'lanczos3'),fullfile(out,'frame_000_4K.png'));
    summary=struct('frame_count',1,'source_time_seconds',0,'exposure_seconds',dt(1), ...
        'video_width',size(pixels,2),'video_height',size(pixels,1),'color_range',[0 upper], ...
        'language','English','subtitles',false,'captions',false,'eye_movements_enabled',withEyeMovements, ...
        'phototransduction_run',false,'downstream_circuit_run',false,'cone_count',numel(d.types), ...
        'full_resolution_image',fullfile(out,'frame_000.png'),'overview_image',fullfile(out,'frame_000_4K.png'));
    fid=fopen(fullfile(out,'render_metadata.json'),'w','n','UTF-8');assert(fid>0);
    fprintf(fid,'%s',jsonencode(summary));fclose(fid);fprintf('CONE_FIRST_FRAME_EXPORT_COMPLETE\n');return;
end
if pipeEncoding
    stream.close;status=encoder.waitFor;assert(status==0,'ConeVideo:Encoder','RGB encoder failed; see log');
    clear encoderGuard;
    summary=struct('frame_count',frames,'frame_rate',1/dt(1),'duration_seconds',frames*dt(1), ...
        'video_width',size(pixels,2),'video_height',size(pixels,1),'color_range',[0 upper], ...
        'language','English','subtitles',false,'captions',false,'eye_movements_enabled',withEyeMovements, ...
        'rgb_master',c.rgb_master,'rgb_master_codec','FFV1 lossless BGR0/RGB444', ...
        'playback_video',c.playback_video,'phototransduction_run',false,'downstream_circuit_run',false, ...
        'rasterized_apertures',true,'aperture_raster_samples',size(ids,1)/scale,'aperture_raster_supersampling',scale);
    fid=fopen(fullfile(out,'render_metadata.json'),'w','n','UTF-8');fprintf(fid,'%s',jsonencode(summary));fclose(fid);
    fprintf('CONE_VIDEO_RGB_EXPORT_COMPLETE %dx%d, %d frames\n',size(pixels,2),size(pixels,1),frames);
    return;
end
clear writerGuard;
if highDefinition
    clear masterGuard;
    nominalDuration=frames*dt(1);
    summary=struct('frame_count',frames,'frame_rate',1/dt(1),'duration_seconds',nominalDuration, ...
        'video_width',size(pixels,2),'video_height',size(pixels,1),'color_range',[0 upper], ...
        'maximum_rate_Rstar_per_cone_s',maximum,'fixed_color_scale_all_frames',true, ...
        'language','English','subtitles',false,'captions',false,'phototransduction_run',false, ...
        'downstream_circuit_run',false,'eye_movements_enabled',withEyeMovements, ...
        'optical_capture_recomputed',false,'rgb_master',masterFile,'rgb_master_codec','Uncompressed RGB AVI', ...
        'playback_video',videoFile,'aperture_raster_samples',size(ids,1)/scale, ...
        'aperture_raster_supersampling',scale);
    fid=fopen(fullfile(out,'render_metadata.json'),'w','n','UTF-8');assert(fid~=-1);
    fprintf(fid,'%s',jsonencode(summary));fclose(fid);
    fprintf('CONE_VIDEO_HD_EXPORT_COMPLETE %dx%d, %d frames\n',size(pixels,2),size(pixels,1),frames);
    return;
end
reader=VideoReader(videoFile); decoded=0;
while hasFrame(reader), readFrame(reader); decoded=decoded+1; end
% This reader reports 0.9999666 s for 30 frames at 30 fps; validate the full frame schedule.
nominalDuration=decoded/reader.FrameRate;
assert(decoded==frames && abs(reader.FrameRate-1/dt(1))<1e-6 && ...
    abs(nominalDuration-duration)<1e-9 && abs(reader.Duration-nominalDuration)<1e-3, ...
    'ConeVideo:Duration','Encoded frame count/FPS/duration differs');
summary=struct('frame_count',decoded,'frame_rate',reader.FrameRate,'duration_seconds',nominalDuration, ...
    'reader_reported_duration_seconds',reader.Duration, ...
    'video_width',reader.Width,'video_height',reader.Height,'color_range',[0 upper], ...
    'maximum_rate_Rstar_per_cone_s',maximum,'fixed_color_scale_all_frames',true, ...
    'language','English','subtitles',false,'captions',false,'phototransduction_run',false,'downstream_circuit_run',false);
summary.eye_movements_enabled=withEyeMovements;
summary.rasterized_apertures=rasterized;
if rasterized,summary.aperture_raster_supersampling=scale;end
if withEyeMovements
    summary.input_panel_eye_position='Video-frame midpoint; capture panel integrates all within-frame substeps';
    eye=load(fullfile(out,'eye_movements.mat'));
    pathFig=figure('Visible','off','Color','white','Position',[50 50 1100 430]); pathGuard=onCleanup(@()close(pathFig));
    ax=subplot(1,2,1,'Parent',pathFig); h=plot(ax,eye.time_seconds,eye.eye_position_arcmin,'LineWidth',1);
    hold(ax,'on');
    for onset=eye.microsaccade_onset_seconds(:)', xline(ax,onset,':','Color',[.55 .55 .55],'HandleVisibility','off'); end
    xlabel(ax,'Time (s)'); ylabel(ax,'Eye displacement (arcmin)'); xlim(ax,[0 duration]);
    title(ax,'A  Eye displacement'); legend(ax,h,{'Horizontal','Vertical'},'Box','off'); grid(ax,'on');
    ax=subplot(1,2,2,'Parent',pathFig); keep=eye.time_seconds<=duration;
    positions=eye.eye_position_arcmin(keep,:); plot(ax,positions(:,1),positions(:,2),'LineWidth',1); hold(ax,'on');
    plot(ax,positions(1,1),positions(1,2),'go','MarkerFaceColor','g');plot(ax,positions(end,1),positions(end,2),'rx');
    xlabel(ax,'Horizontal displacement (arcmin)');ylabel(ax,'Vertical displacement (arcmin)');
    title(ax,'B  Eye trajectory');axis(ax,'equal');grid(ax,'on');
    set(findall(pathFig,'Type','axes'),'FontName','Arial','FontSize',11);
    exportgraphics(pathFig,fullfile(out,'eye_movement_path.png'),'Resolution',180);
    exportgraphics(pathFig,fullfile(out,'eye_movement_path.pdf'),'ContentType','vector');
end
fid=fopen(fullfile(out,'render_metadata.json'),'w','n','UTF-8'); assert(fid~=-1); fileGuard=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(summary)); fprintf('CONE_VIDEO_RENDER_PASS %.6f seconds, %d frames\n',reader.Duration,decoded);
end
function rgb=rgbFrame(frames,index)
rgb=double(frames(:,:,:,index));
if isa(frames,'uint8'),rgb=rgb/255;end
end
function rgb=responseImage(rates,upper,colors,ids,scale)
indices=min(256,max(1,fix(rates(:)/upper*256)+1));
rgb=rasterImage([1 1 1;colors(indices,:)],ids,scale);
end
function rgb=rasterImage(colors,ids,scale)
height=size(ids,1)/scale;width=size(ids,2)/scale;
colors=single(colors);
rgb=reshape(colors(ids(:)+1,:),[size(ids) 3]);
rgb=reshape(mean(reshape(rgb,scale,height,scale,width,3),[1 3]),height,width,3);
end
function stopEncoder(process,stream)
try,stream.close;catch,end
if process.isAlive,process.destroy;end
end
