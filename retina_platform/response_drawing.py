"""Pillow spatial point maps; colors use a fixed range over the complete clip."""
import numpy as np
from PIL import Image, ImageDraw
from .plotting import font


def color(value, lower, upper, delta):
    fraction = .5 if upper == lower else max(0., min(1., (value-lower)/(upper-lower)))
    stops = [(30,90,175),(242,245,247),(190,38,45)] if delta else [(48,18,87),(33,145,140),(250,226,60)]
    part = min(1, int(fraction*2)); local = fraction*2-part
    return tuple(round(a+(b-a)*local) for a,b in zip(stops[part], stops[part+1]))


def spatial_image(data, populations, index, network, direct, target, selected, size, delta, all_patch, rgc_rate=False):
    width, height = max(200,size[0]), max(140,size[1])
    image = Image.new('RGB',(width,height),'#101b28'); draw = ImageDraw.Draw(image)
    entries=[]; ranges=[]; unit=None
    for pop in populations:
        if pop not in data.populations: continue
        key = pop+'__expected_binary_count_rate' if rgc_rate and pop.endswith(('ON','OFF')) or rgc_rate and pop=='small_bistratified_S_ON' else data.populations[pop]['primary']
        if key is None or key not in data.values: continue
        value = data.values[key]
        lower,upper,scale,shown_unit=data.field_range(pop,key,delta)
        shown=(value[:,index]-value[:,0] if delta else value[:,index])*scale
        if unit is not None and unit!=shown_unit: raise ValueError('禁止把不同单位放入同一色标')
        unit=shown_unit;ranges.append((lower,upper))
        entries += [(pop,row,data.geometry[pop]['xy'][row],shown[row]) for row in range(len(value))]
    if not entries:
        draw.text((14,40),'未保存可空间映射状态',font=font(12),fill='#cbd5e1')
        return image, [], '缺失：不重建、不插值'
    low=min(a[0] for a in ranges); high=max(a[1] for a in ranges)
    if delta: high=max(abs(low),abs(high));low=-high
    xy=np.asarray([e[2] for e in entries]); lower=xy.min(axis=0);upper=xy.max(axis=0)
    span=np.maximum(upper-lower,1e-7); scale=min((width-32)/span[0],(height-64)/span[1])
    center=(upper+lower)/2; midpoint=np.array([width/2,(height-34)/2])
    pointmap=[]; shown_count=0
    for pop,row,point,value in entries:
        related=row in network.get(pop,set())
        if not all_patch and not related: continue
        pixel=(point-center)*scale*np.array([1,-1])+midpoint
        x,y=map(float,pixel);radius=2.4 if related else 1.5
        fill=color(float(value),low,high,delta) if related else '#384553'
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=fill)
        if (pop,row) in direct: draw.ellipse((x-3.4,y-3.4,x+3.4,y+3.4),outline='#b5c5d8')
        if (pop,row)==target: draw.ellipse((x-7,y-7,x+7,y+7),outline='#fbd45c',width=2)
        if (pop,row)==selected: draw.rectangle((x-5,y-5,x+5,y+5),outline='white',width=1)
        pointmap.append((x,y,pop,row));shown_count+=int(related)
    for x in range(width-24):
        value=low+(high-low)*x/max(1,width-25)
        draw.line((12+x,height-27,12+x,height-20),fill=color(value,low,high,delta))
    draw.text((12,height-17),f'{low:.5g}',font=font(9),fill='#cbd5e1')
    right=f'{high:.5g} {unit}';draw.text((width-12-draw.textlength(right,font=font(9)),height-17),right,font=font(9),fill='#cbd5e1')
    label=f"{'Δ 相对t0' if delta else '绝对状态'} · 全片固定色标 · 祖先 {shown_count}/{len(entries)} 组\nx {lower[0]:.3g}…{upper[0]:.3g} / y {lower[1]:.3g}…{upper[1]:.3g} °"
    if shown_count==0:draw.text((12,12),'当前RGC无此层已保存祖先',font=font(10),fill='#cbd5e1')
    return image,pointmap,label


def trace_image(time, values, seconds, size, title, unit, events=()):
    width,height=max(500,size[0]),max(260,size[1]);image=Image.new('RGB',(width,height),'#f8fafc');draw=ImageDraw.Draw(image)
    left,top,right,bottom=72,42,width-25,height-50
    low,high=float(values.min()),float(values.max());pad=max((high-low)*.08,1e-12)
    low-=pad;high+=pad
    draw.text((14,10),title,font=font(12),fill='#243447');draw.rectangle((left,top,right,bottom),outline='#cbd5e1')
    points=[(left+t/time[-1]*(right-left),bottom-(v-low)/(high-low)*(bottom-top)) for t,v in zip(time,values)]
    if len(points)>1:draw.line(points,fill='#2563eb',width=1)
    marker=left+seconds/time[-1]*(right-left);draw.line((marker,top,marker,bottom),fill='#dc2626')
    draw.text((5,top),f'{high:.6g}',font=font(9),fill='#334155');draw.text((5,bottom-15),f'{low:.6g}',font=font(9),fill='#334155')
    draw.text((5,top+20),unit,font=font(10),fill='#334155')
    for t in events:
        x=left+t/time[-1]*(right-left);draw.line((x,bottom+8,x,bottom+14),fill='#334155')
    draw.text((left,height-23),f'0 — {time[-1]:.3f} s · 原保存样本 / 组对象事件',font=font(10),fill='#475569')
    return image
