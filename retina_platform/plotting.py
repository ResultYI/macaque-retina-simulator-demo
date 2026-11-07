"""Raster plots of actual cell samples; no interpolation or fabricated history."""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = "#ffffff"
INK = "#263640"
MUTED = "#6d7880"
GRID = "#edf0f2"
AXIS = "#cbd2d7"


@lru_cache(maxsize=64)
def font(size=13):
    for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def format_tick(value, span=None):
    """Readable display precision; the underlying samples remain unchanged."""
    value = float(value)
    if value == 0:
        return '0'
    span = abs(float(span if span is not None else value))
    span = span or abs(value)
    span_order = int(math.floor(math.log10(max(span, 1e-30))))
    decimals = max(0, 2-span_order)
    if decimals > 6 or abs(value) >= 1e5:
        value_order = int(math.floor(math.log10(abs(value))))
        significant = min(16, max(5, value_order-span_order+3))
        return f'{value:.{significant}g}'
    text = f'{value:.{decimals}f}'
    text = text.rstrip('0').rstrip('.') if '.' in text else text
    return '0' if text == '-0' else text


def draw_text_fitted(draw, xy, text, width, *, size=12, fill=INK, pixel_scale=1.):
    face = font(max(1, round(size*pixel_scale)))
    text = str(text)
    if width <= 0:
        return
    if draw.textlength(text, font=face) > width:
        while text and draw.textlength(text+'…', font=face) > width:
            text = text[:-1]
        text += '…'
    draw.text(xy, text, font=face, fill=fill)


def colors(values, low, high):
    # Piecewise perceptual palette, evaluated only at measured cell samples.
    anchors = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)
    t = np.clip((np.asarray(values, float) - low) / max(high - low, 1e-30), 0, 1)
    t = np.nan_to_num(t, nan=0.0)
    p = t * 4
    i = np.minimum(p.astype(int), 3)
    out = anchors[i] * (1 - (p - i)[:, None]) + anchors[i + 1] * (p - i)[:, None]
    return [tuple(row.astype(int)) for row in out]


def empty_plot(size, message, pixel_scale=1.):
    s = pixel_scale
    im = Image.new("RGB", (max(1, int(size[0])), max(1, int(size[1]))), BG)
    draw = ImageDraw.Draw(im)
    draw.text((22*s, 22*s), message, fill=MUTED, font=font(round(14*s)))
    return im


def scatter_bounds(xy, indices, size, viewport=None, pixel_scale=1.):
    """Visible data bounds for an aspect-preserving scatter viewport."""
    s = pixel_scale
    width, height = max(1, int(size[0])), max(1, int(size[1]))
    if viewport is None:
        points = np.asarray(xy)[np.asarray(indices, dtype=int), :2]
        lower, upper = points.min(axis=0), points.max(axis=0)
    else:
        lower, upper = np.asarray(viewport[0], float), np.asarray(viewport[1], float)
    center = (lower+upper)/2
    span = np.maximum(upper-lower, 1e-6)
    pixels = np.array([max(1, width-171*s), max(1, height-97*s)], float)
    scale = float(np.min(pixels/span))
    return center-pixels/(2*scale), center+pixels/(2*scale)


def scatter_plot(xy, values, indices, size, *, coordinate="RF", coordinate_unit="mm", value_unit="", selected=None, fixed_range=None, title="", viewport=None, translate=lambda text: text, pixel_scale=1., markers=None):
    s = pixel_scale
    im = empty_plot(size, "", s)
    draw = ImageDraw.Draw(im)
    w, h = im.size
    if w <= 190*s or h <= 170*s:
        return empty_plot(size, translate('窗口过小，请放大'), s), []
    xy = np.asarray(xy, float)
    values = np.asarray(values, float).reshape(-1)
    indices = np.asarray(indices, int)
    if xy.ndim != 2 or xy.shape[1] < 2 or not len(indices):
        return empty_plot(size, translate("无可用细胞坐标"), s), []
    valid = np.all(np.isfinite(xy[indices, :2]), axis=1)
    indices = indices[valid]
    if not len(indices):
        return empty_plot(size, translate("无有限坐标"), s), []
    points = xy[indices, :2]
    vals = values[indices]
    finite = vals[np.isfinite(vals)]
    lo, hi = fixed_range if fixed_range is not None else ((float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0))
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
        raise ValueError("色标范围须为有限数且最大值不小于最小值")
    mins, maxs = (points.min(axis=0), points.max(axis=0)) if viewport is None else scatter_bounds(xy, indices, size, viewport, s)
    span = np.maximum(maxs - mins, 1e-6)
    left, top, right, bottom = 64*s, 43*s, w - 107*s, h - 54*s
    scale = min((right - left) / span[0], (bottom - top) / span[1])
    midpoint = (mins + maxs) / 2
    display = (points - midpoint) * np.array([scale, -scale]) + np.array([(left + right) / 2, (top + bottom) / 2])
    draw_text_fitted(draw, (16*s, 8*s), title, w-32*s, size=12, pixel_scale=s)
    visible_span = np.array([right-left, bottom-top])/scale
    lower, upper = midpoint-visible_span/2, midpoint+visible_span/2
    tick_face = font(round(9*s))
    fractions = (0., .5, 1.) if right-left >= 130*s else (0., 1.)
    for fraction in fractions:
        x, y = left+fraction*(right-left), bottom-fraction*(bottom-top)
        draw.line((x, top, x, bottom), fill=GRID, width=max(1, round(s)))
        draw.line((left, y, right, y), fill=GRID, width=max(1, round(s)))
        xt = format_tick(lower[0]+fraction*visible_span[0], visible_span[0])
        yt = format_tick(lower[1]+fraction*visible_span[1], visible_span[1])
        draw.text((x-draw.textlength(xt, font=tick_face)/2, bottom+5*s), xt, font=tick_face, fill=MUTED)
        draw.text((left-7*s-draw.textlength(yt, font=tick_face), y-6*s), yt, font=tick_face, fill=MUTED)
    draw.rectangle((left, top, right, bottom), outline=AXIS, width=max(1, round(s)))
    axis_label = f'{coordinate} x / {coordinate_unit}'.lstrip()
    draw_text_fitted(draw, (left, h-23*s), axis_label, right-left, size=10, fill=MUTED, pixel_scale=s)
    draw.text((left, top-18*s), f'y / {coordinate_unit}', font=font(round(9*s)), fill=MUTED)
    radius = s*max(2, min(6, int(160 / math.sqrt(max(len(indices), 1)))))
    mapping = []
    for idx, (px, py), color in zip(indices, display, colors(vals, lo, hi)):
        if viewport is not None and not (left <= px <= right and top <= py <= bottom):
            continue
        color = color if np.isfinite(values[idx]) else (148, 163, 184)
        marker = markers[idx] if markers is not None else 'circle'
        box = (px-radius, py-radius, px+radius, py+radius)
        if marker == 'square':
            draw.rectangle(box, fill=color, outline='#ffffff')
        elif marker == 'diamond':
            draw.polygon(((px, py-radius), (px+radius, py), (px, py+radius), (px-radius, py)), fill=color, outline='#ffffff')
        else:
            draw.ellipse(box, fill=color, outline="#ffffff")
        if int(idx) == selected:
            ring = (px-radius-3*s, py-radius-3*s, px+radius+3*s, py+radius+3*s)
            if marker == 'square': draw.rectangle(ring, outline='#ef4444', width=max(1, round(2*s)))
            else: draw.ellipse(ring, outline="#ef4444", width=max(1, round(2*s)))
        mapping.append((float(px), float(py), int(idx)))
    bx, by, bh = w - 75*s, 58*s, max(round(30*s), round(h - 146*s))
    for row in range(bh):
        color = colors(np.array([hi - (hi-lo) * row / max(bh-1, 1)]), lo, hi)[0]
        draw.line((bx, by+row, bx+15*s, by+row), fill=color)
    draw_text_fitted(draw, (bx-12*s, by-18*s), format_tick(hi, hi-lo), w-bx+8*s, size=9, pixel_scale=s)
    draw_text_fitted(draw, (bx-12*s, by+bh+3*s), format_tick(lo, hi-lo), w-bx+8*s, size=9, pixel_scale=s)
    unit_face = font(round(9*s))
    unit_lines, line = [], ''
    for character in value_unit:
        if line and draw.textlength(line+character, font=unit_face) > w-bx+8*s:
            unit_lines.append(line); line = ''
        line += character
    unit_lines.append(line)
    draw.text((bx-12*s, by+bh+19*s), '\n'.join(unit_lines), fill=MUTED, font=unit_face, spacing=0)
    draw_text_fitted(draw, (bx-12*s, h-20*s), translate("固定" if fixed_range is not None else "自动"), w-bx+8*s, size=9, fill=MUTED, pixel_scale=s)
    return im, mapping


def curve_plot(time, values, size, *, title="", unit="", window_values=False):
    im = empty_plot(size, "")
    draw = ImageDraw.Draw(im)
    w, h = im.size
    if w <= 180 or h <= 120:
        return empty_plot(size, '窗口过小，请放大')
    t, y = np.asarray(time, float).reshape(-1), np.asarray(values, float).reshape(-1)
    n = min(len(t), len(y))
    mask = np.isfinite(t[:n]) & np.isfinite(y[:n])
    t, y = t[:n][mask], y[:n][mask]
    draw_text_fitted(draw, (16, 8), title, w-32, size=12)
    if not len(t):
        draw.text((20, 44), "选择细胞查看记录", fill=MUTED, font=font(13))
        return im
    left, top, right, bottom = 66, 42, w - 22, h - 43
    tlo, thi, ylo, yhi = float(t.min()), float(t.max()), float(y.min()), float(y.max())
    yspan = max(yhi-ylo, abs(yhi)*0.02, 1e-12)
    ylo, yhi = ylo-0.05*yspan, yhi+0.05*yspan
    for fraction in (0., .5, 1.):
        x, yy = left+fraction*(right-left), bottom-fraction*(bottom-top)
        draw.line((left, yy, right, yy), fill=GRID)
        draw.text((5, yy-6), format_tick(ylo+fraction*(yhi-ylo), yhi-ylo), fill=MUTED, font=font(9))
        label = format_tick(tlo+fraction*(thi-tlo), thi-tlo)
        draw.text((x-draw.textlength(label, font=font(9))/2, bottom+5), label, fill=MUTED, font=font(9))
    draw.rectangle((left, top, right, bottom), outline=AXIS)
    px = left + (t-tlo) / max(thi-tlo, 1e-12) * (right-left)
    py = bottom - (y-ylo) / (yhi-ylo) * (bottom-top)
    if len(t) > 1 and not window_values:
        # Display envelope via min/max pairs in pixel bins, retaining narrow transients.
        if len(t) > 4 * (right-left):
            bins = np.clip(px.astype(int), left, right)
            points = []
            for bucket in np.unique(bins):
                rows = np.flatnonzero(bins == bucket)
                for index in sorted((int(rows[np.argmin(y[rows])]), int(rows[np.argmax(y[rows])]))):
                    points.append((float(px[index]), float(py[index])))
        else:
            points = list(zip(px.tolist(), py.tolist()))
        if len(points) > 1:
            draw.line(points, fill="#167e9a", width=2)
    else:
        for x, yy in zip(px, py):
            draw.ellipse((x-3, yy-3, x+3, yy+3), fill="#167e9a")
    draw.text((right-7, h-18), 's', fill=MUTED, font=font(9))
    draw_text_fitted(draw, (left, top-18), unit + (" · 窗口统计" if window_values else ""), right-left, size=9, fill=MUTED)
    return im
