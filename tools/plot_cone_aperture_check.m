function plot_cone_aperture_check(directory)
% Inspect saved geometry/apertures only; never recompute capture or a circuit.
d=load(fullfile(directory,'input.mat')); r=load(fullfile(directory,'capture.mat'));
c=jsondecode(fileread(fullfile(directory,'export.json'))); meta=jsondecode(r.manifest_json);
assert(strcmp(meta.request_signature,c.request_signature));
xy=d.is_mm; diam=r.diameters_um(:); types=d.types(:); ecc=sqrt(sum(xy.^2,2))/.223;
assert(numel(diam)==size(xy,1) && all(isfinite(diam)) && all(diam>0));
png=fullfile(directory,'aperture_size_check.png'); pdf=fullfile(directory,'aperture_size_check.pdf');
assert(~isfile(png) && ~isfile(pdf),'Choose a fresh diagnostic name; do not overwrite existing figures');
palette=[.82 .19 .25; .10 .57 .34; .18 .39 .80];
fig=figure('Visible','off','Color','white','Position',[50 50 1540 525]); cleanup=onCleanup(@()close(fig));
set(groot,'defaultTextInterpreter','none','defaultAxesTickLabelInterpreter','none','defaultLegendInterpreter','none');
theta=(0:23)*2*pi/24; panels=[.065 .17 .25 .72; .385 .17 .25 .72; .72 .17 .23 .72];
targetX=min(xy(:,1))+(max(xy(:,1))-min(xy(:,1)))*[.15 .85];
windowCenters=zeros(2,2); windowMedian=zeros(2,1); windowCount=zeros(2,1);
for k=1:3
    ax=axes(fig,'Position',panels(k,:)); hold(ax,'on');
    if k<=2
        [~,index]=min((xy(:,1)-targetX(k)).^2+(xy(:,2)-median(xy(:,2))).^2);
        center=xy(index,:); local=(xy-center)*1000; keep=all(abs(local)<=10,2);
        p=local(keep,:); radius=diam(keep)/2;
        vx=p(:,1)+radius*cos(theta); vy=p(:,2)+radius*sin(theta);
        vertices=[reshape(vx',[],1),reshape(vy',[],1)]; faces=reshape(1:numel(vx),24,[])';
        patch(ax,'Faces',faces,'Vertices',vertices,'FaceVertexCData',palette(types(keep),:),'FaceColor','flat','EdgeColor','none');
        xlim(ax,[-10 10]); ylim(ax,[-10 10]); daspect(ax,[1 1 1]);
        windowCenters(k,:)=center; windowMedian(k)=median(diam(keep)); windowCount(k)=sum(keep);
        title(ax,sprintf('%c  Local window %d',char('A'+k-1),k));
        xlabel(ax,'Local x (µm)'); ylabel(ax,'Local y (µm)');
    else
        for t=1:3
            keep=types==t; scatter(ax,ecc(keep),diam(keep),9,palette(t,:),'filled');
        end
        title(ax,'C  Cone aperture diameter');
        xlabel(ax,'Eccentricity (deg)'); ylabel(ax,'Aperture diameter (µm)');
        grid(ax,'on');
        legend(ax,{'L','M','S'},'Location','southeast','Box','off');
    end
    set(ax,'FontName','Arial','FontSize',11,'Box','on','Layer','top'); ax.Position=panels(k,:);
end
exportgraphics(fig,png,'Resolution',200); exportgraphics(fig,pdf,'ContentType','vector');
summary=struct('cell_count',numel(diam),'diameter_range_um',[min(diam) max(diam)], ...
    'median_diameter_um',median(diam),'eccentricity_range_degrees',[min(ecc) max(ecc)], ...
    'max_min_diameter_ratio',max(diam)/min(diam),'max_min_aperture_area_ratio',(max(diam)/min(diam))^2, ...
    'window_centers_mm',windowCenters,'window_median_diameter_um',windowMedian,'window_counts',windowCount, ...
    'window_size_um',[20 20],'measured_anatomical_diameters',false,'type_specific_diameter_rule',false, ...
    'capture_recomputed',false,'downstream_circuit_run',false);
fid=fopen(fullfile(directory,'aperture_size_check.json'),'w','n','UTF-8'); assert(fid~=-1); closeFile=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(summary));
fprintf('APERTURE_SIZE_CHECK_PASS diameters %.9g to %.9g um, medians %.9g / %.9g um\n',min(diam),max(diam),windowMedian);
end
