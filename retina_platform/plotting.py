"""Raster plots of actual cell samples; no interpolation or fabricated history."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = "#f8fafc"
INK = "#243447"
MUTED = "#64748b"


def font(size=13):
    for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def colors(values, low, high):
    # Piecewise perceptual palette, evaluated only at measured cell samples.
    anchors = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)
    t = np.clip((np.asarray(values, float) - low) / max(high - low, 1e-30), 0, 1)
    t = np.nan_to_num(t, nan=0.0)
    p = t * 4
    i = np.minimum(p.astype(int), 3)
    out = anchors[i] * (1 - (p - i)[:, None]) + anchors[i + 1] * (p - i)[:, None]
    return [tuple(row.astype(int)) for row in out]


def empty_plot(size, message):
    im = Image.new("RGB", (max(240, int(size[0])), max(160, int(size[1]))), BG)
    draw = ImageDraw.Draw(im)
    draw.text((22, 22), message, fill=MUTED, font=font(14))
    return im


def scatter_plot(xy, values, indices, size, *, coordinate="RF", coordinate_unit="mm", value_unit="", selected=None, fixed_range=None, title=""):
    im = empty_plot(size, "")
    draw = ImageDraw.Draw(im)
    w, h = im.size
    xy = np.asarray(xy, float)
    values = np.asarray(values, float).reshape(-1)
    indices = np.asarray(indices, int)
    if xy.ndim != 2 or xy.shape[1] < 2 or not len(indices):
        return empty_plot(size, "无可用细胞坐标"), []
    valid = np.all(np.isfinite(xy[indices, :2]), axis=1)
    indices = indices[valid]
    if not len(indices):
        return empty_plot(size, "无有限坐标"), []
    points = xy[indices, :2]
    vals = values[indices]
    finite = vals[np.isfinite(vals)]
    lo, hi = fixed_range if fixed_range is not None else ((float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0))
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
        raise ValueError("色标范围须为有限数且最大值不小于最小值")
    mins, maxs = points.min(axis=0), points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    left, top, right, bottom = 64, 43, w - 107, h - 54
    scale = min((right - left) / span[0], (bottom - top) / span[1])
    midpoint = (mins + maxs) / 2
    display = (points - midpoint) * np.array([scale, -scale]) + np.array([(left + right) / 2, (top + bottom) / 2])
    draw.rectangle((left, top, right, bottom), outline="#cbd5e1")
    draw.text((16, 11), title, font=font(13), fill=INK)
    draw.text((left, h - 31), f"{coordinate} x / {coordinate_unit}  [{mins[0]:.4g}, {maxs[0]:.4g}]", font=font(11), fill=MUTED)
    draw.text((left, top + 3), f"y / {coordinate_unit}  [{mins[1]:.4g}, {maxs[1]:.4g}]", font=font(10), fill=MUTED)
    radius = max(2, min(6, int(160 / math.sqrt(max(len(indices), 1)))))
    mapping = []
    for idx, (px, py), color in zip(indices, display, colors(vals, lo, hi)):
        color = color if np.isfinite(values[idx]) else (148, 163, 184)
        draw.ellipse((px-radius, py-radius, px+radius, py+radius), fill=color, outline="#ffffff")
        if int(idx) == selected:
            draw.ellipse((px-radius-3, py-radius-3, px+radius+3, py+radius+3), outline="#ef4444", width=2)
        mapping.append((float(px), float(py), int(idx)))
    bx, by, bh = w - 75, 58, max(50, h - 126)
    for row in range(bh):
        color = colors(np.array([hi - (hi-lo) * row / max(bh-1, 1)]), lo, hi)[0]
        draw.line((bx, by+row, bx+15, by+row), fill=color)
    draw.text((bx-12, by-19), f"{hi:.4g}", fill=INK, font=font(10))
    draw.text((bx-12, by+bh+3), f"{lo:.4g}", fill=INK, font=font(10))
    draw.text((bx-12, by+bh+20), value_unit, fill=MUTED, font=font(10))
    draw.text((bx-12, h-25), "固定" if fixed_range is not None else "自动", fill=MUTED, font=font(11))
    return im, mapping


def curve_plot(time, values, size, *, title="", unit="", window_values=False):
    im = empty_plot(size, "")
    draw = ImageDraw.Draw(im)
    w, h = im.size
    t, y = np.asarray(time, float).reshape(-1), np.asarray(values, float).reshape(-1)
    n = min(len(t), len(y))
    mask = np.isfinite(t[:n]) & np.isfinite(y[:n])
    t, y = t[:n][mask], y[:n][mask]
    draw.text((16, 10), title, fill=INK, font=font(13))
    if not len(t):
        draw.text((20, 44), "选择细胞查看记录", fill=MUTED, font=font(13))
        return im
    left, top, right, bottom = 66, 42, w - 22, h - 43
    draw.rectangle((left, top, right, bottom), outline="#cbd5e1")
    tlo, thi, ylo, yhi = float(t.min()), float(t.max()), float(y.min()), float(y.max())
    yspan = max(yhi-ylo, abs(yhi)*0.02, 1e-12)
    ylo, yhi = ylo-0.05*yspan, yhi+0.05*yspan
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
    draw.text((left, bottom+9), f"{tlo:.5g} — {thi:.5g} s", fill=MUTED, font=font(11))
    draw.text((5, top), f"{yhi:.4g}", fill=MUTED, font=font(10))
    draw.text((5, bottom-12), f"{ylo:.4g}", fill=MUTED, font=font(10))
    draw.text((left+4, top+4), unit + (" · 窗口统计" if window_values else ""), fill=MUTED, font=font(11))
    return im
