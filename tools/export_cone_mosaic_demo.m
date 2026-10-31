function export_cone_mosaic_demo(configPath, outputPath)
% Reuse the pinned production capture; figures contain no downstream dynamics.
c=jsondecode(fileread(configPath));
% Keep class destructors available when the production exporter restores its path.
addpath(genpath(c.profile.isetcam_root)); addpath(genpath(c.profile.isetbio_root));
export_mixed_reference(configPath, outputPath);
d=load(c.input_mat); r=load(outputPath);
meta=jsondecode(r.manifest_json);
assert(~meta.phototransduction_applied && strcmp(meta.photon_noise,'none'));
dt=diff(r.window_edges_seconds(:)); rates=r.expected_counts./dt;
assert(size(rates,1)==2 && all(isfinite(rates(:))) && all(rates(:)>=0));
maximum=max(rates(:)); assert(maximum>0);
step=10^floor(log10(maximum))/2; upper=ceil(maximum/step)*step;
xy=d.is_mm/.223; radii=r.diameters_um(:)/(2*223);
theta=(0:19)*2*pi/20;
vx=xy(:,1)+radii*cos(theta); vy=xy(:,2)+radii*sin(theta);
vertices=[reshape(vx',[],1),reshape(vy',[],1)];
faces=reshape(1:numel(vx),20,[])';
palette=[.82 .19 .25; .10 .57 .34; .18 .39 .80];
typeColors=palette(d.types,:); bounds=d.plot_bounds_mm/.223;
names={'synthetic','natural'};
out=fileparts(outputPath);
set(groot,'defaultTextInterpreter','none','defaultAxesTickLabelInterpreter','none','defaultLegendInterpreter','none');
for k=1:2
    fig=figure('Visible','off','Color','white','Position',[50 50 1620 575]);
    guard=onCleanup(@()close(fig));
    panels=[.055 .20 .255 .69; .365 .20 .255 .69; .675 .20 .255 .69];
    axesList=gobjects(3,1);
    for p=1:3
        ax=axes(fig,'Position',panels(p,:)); axesList(p)=ax; hold(ax,'on');
        if p==1
            image(ax,'XData',d.plot_x_mm([1 end])/.223,'YData',d.plot_y_mm([1 end])/.223, ...
                'CData',double(d.rgb_frames(:,:,:,k)));
            title(ax,'A  Input image');
        elseif p==2
            patch(ax,'Faces',faces,'Vertices',vertices,'FaceVertexCData',typeColors, ...
                'FaceColor','flat','EdgeColor','none');
            title(ax,'B  L/M/S cone mosaic');
            handles=gobjects(3,1);
            for t=1:3, handles(t)=plot(ax,nan,nan,'o','MarkerFaceColor',palette(t,:),'MarkerEdgeColor','none'); end
            legend(ax,handles,{'L','M','S'},'Orientation','horizontal','Position',[.42 .055 .16 .04],'Box','off');
        else
            patch(ax,'Faces',faces,'Vertices',vertices,'FaceVertexCData',rates(k,:)', ...
                'FaceColor','flat','EdgeColor','none');
            colormap(ax,parula(256)); clim(ax,[0 upper]);
            bar=colorbar(ax,'Position',[.943 .24 .012 .58]);
            bar.Ticks=linspace(0,upper,5);
            bar.Label.String='Expected isomerizations (R*/cone/s)'; bar.Label.FontName='Arial';
            title(ax,'C  Cone photon catches');
        end
        xlim(ax,bounds(1:2)); ylim(ax,bounds(3:4));
        set(ax,'YDir','normal','DataAspectRatio',[1 1 1],'FontName','Arial','FontSize',11,'Box','on','Layer','top');
        xlabel(ax,'x (deg)'); ylabel(ax,'y (deg)');
        ax.Position=panels(p,:);
    end
    exportgraphics(fig,fullfile(out,[names{k} '_comparison.png']),'Resolution',200);
    exportgraphics(fig,fullfile(out,[names{k} '_comparison.pdf']),'ContentType','vector');
    clear guard;
end
plot_cone_aperture_check(out);
fprintf('CONE_MOSAIC_FIGURES_PASS shared color scale [0, %.9g] R*/cone/s\n',upper);
fid=fopen(fullfile(out,'render_metadata.json'),'w','n','UTF-8'); assert(fid~=-1);
cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(struct('color_range',[0 upper],'data_max_rate_rstar_per_cone_s',maximum, ...
    'language','English','subtitles',false,'captions',false)));
end
