function filtered = sgolayfilt(values, order, window)
% Polynomial least-squares smoothing for fixationalEM's velocity diagnostic only.
% This local process fallback supports exactly the vector/three-argument call used there.
assert(isvector(values) && isfloat(values) && all(isfinite(values(:))), ...
    'ConeEM:Filter','Finite floating-point vector required');
assert(isscalar(order) && order==fix(order) && order>=0 && isscalar(window) && ...
    window==fix(window) && mod(window,2)==1 && window>order && numel(values)>=window, ...
    'ConeEM:Filter','Valid polynomial order and odd window length required');
order=double(order); window=double(window);
half=(window-1)/2; t=(-half:half)'; design=t.^(0:order);
projection=design*pinv(design);
x=double(values(:)); y=conv(x,projection(half+1,:),'same');
y(1:half)=projection(1:half,:)*x(1:window);
y(end-half+1:end)=projection(end-half+1:end,:)*x(end-window+1:end);
filtered=reshape(cast(y,'like',values),size(values));
end
