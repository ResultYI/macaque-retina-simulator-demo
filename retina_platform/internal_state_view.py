"""Saved internal states in the existing Tk desktop; never invokes a model runner."""
from pathlib import Path
import json
import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk, PngImagePlugin

from .plotting import (font, scatter_plot, scatter_bounds, empty_plot, format_tick,
                       draw_text_fitted, BG, INK, MUTED, GRID, AXIS)
from .replay import PlaybackClock
from .internal_state_i18n import LANGUAGES, translate

CURRENT_RESULT = 'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/current_ac_source'
AC_POPULATIONS = ('same_ON_midget_AC_v1', 'same_OFF_midget_AC_v1', 'cross_OFF_ON_parasol_AC_v1',
                  'cross_ON_OFF_parasol_AC_v1', 'local_feedforward_AC', 'AII_AC')
AC_FILTERS = {'AC_all': '全体 AC（含 AII）', 'AII_AC': '仅 AII'}
LAYERS = [('Cone', ('cones',)), ('H1 / H2', ('HC',)),
          ('Bipolar', ('midget_BC_ON', 'midget_BC_OFF', 'diffuse_BC_ON', 'diffuse_BC_OFF', 'S_ON_BC')),
          ('AC', tuple(AC_FILTERS)), ('RGC', ('parasol_RGC_OFF', 'parasol_RGC_ON', 'midget_RGC_OFF', 'midget_RGC_ON', 'small_bistratified_S_ON'))]
TABS = ('分层状态', 'Cone 光学采样', 'H1 空间权重', '同窗响应', 'PSTH 对照')


def plot_title(draw, text, width, pixel_scale=1.):
    draw_text_fitted(draw, (18*pixel_scale, 10*pixel_scale), text, width,
                     size=12, pixel_scale=pixel_scale)


def trace_panel(size, series, start, stop, seconds, title, unit, message='未记录该时间窗口的状态', translate=lambda text: text, pixel_scale=1., subtitle=''):
    """Draw saved nodes only; sparse snapshots deliberately remain unconnected."""
    scale = pixel_scale
    image = Image.new('RGB', tuple(max(1, int(v)) for v in size), BG)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    if width <= 120*scale or height <= 115*scale:
        return empty_plot(size, translate('窗口过小，请放大'), scale)
    left, top, right, bottom = 76*scale, 54*scale, width-24*scale, height-48*scale
    plot_title(draw, translate(title), width-36*scale, scale)
    if subtitle:
        draw_text_fitted(draw, (76*scale, 30*scale), translate(subtitle), width-100*scale,
                         size=9, fill=MUTED, pixel_scale=scale)
    usable = [s for s in series if len(s.get('time', []))]
    if not usable:
        draw_text_fitted(draw, (25*scale, 70*scale), translate(message), width-50*scale,
                         size=11, fill=MUTED, pixel_scale=scale)
        return image
    factor, shown_unit = {'V': (1000., 'mV'), 'S': (1e9, 'nS'), 'A': (1e12, 'pA')}.get(unit, (1., unit))
    values = np.concatenate([s['values'] for s in usable])*factor
    finite = values[np.isfinite(values)]
    if not len(finite):
        draw_text_fitted(draw, (25*scale, 70*scale), translate(message), width-50*scale,
                         size=11, fill=MUTED, pixel_scale=scale)
        return image
    low, high = float(finite.min()), float(finite.max())
    pad = max((high-low)*.08, abs(high)*.00001, 1e-9)
    low -= pad; high += pad
    for fraction in (0., .5, 1.):
        y = bottom-fraction*(bottom-top)
        draw.line((left, y, right, y), fill=GRID, width=max(1, round(scale)))
        label = format_tick(low+fraction*(high-low), high-low)
        draw.text((left-8*scale-draw.textlength(label, font=font(round(9*scale))), y-6*scale),
                  label, font=font(round(9*scale)), fill=MUTED)
    draw.line((left, top, left, bottom, right, bottom), fill=AXIS, width=max(1, round(scale)))
    draw_text_fitted(draw, (4*scale, top-21*scale), shown_unit, 66*scale, size=9, fill=MUTED, pixel_scale=scale)
    colors = ('#137f91', '#dc7c31', '#6c63ac', '#c34f65')
    for i, item in enumerate(usable):
        t, y = np.asarray(item['time']), np.asarray(item['values'])*factor
        keep = (t >= start) & (t <= stop) & np.isfinite(y)
        points = list(zip(left+(t[keep]-start)/max(stop-start, 1e-9)*(right-left),
                          bottom-(y[keep]-low)/(high-low)*(bottom-top)))
        color = colors[i % len(colors)]
        if item.get('recording') == 'windowed':
            begin, end = item['continuous_window']
            inside = (t[keep] >= begin) & (t[keep] <= end)
            continuous = [p for p, yes in zip(points, inside) if yes]
            if len(continuous) > 1: draw.line(continuous, fill=color, width=max(1, round(2*scale)))
            for (x, y), yes in zip(points, inside):
                if not yes or len(continuous) == 1:
                    draw.ellipse((x-3*scale, y-3*scale, x+3*scale, y+3*scale), fill=color)
        elif item.get('recording') == 'events':
            for x, _ in points: draw.line((x, top+5*scale, x, bottom), fill=color, width=max(1, round(scale)))
        elif item.get('recording') == 'snapshot' or len(points) < 2:
            for x, y in points:
                draw.ellipse((x-3*scale, y-3*scale, x+3*scale, y+3*scale), fill=color)
        else:
            draw.line(points, fill=color, width=max(1, round(2*scale)))
        legend_width = (right-left)/len(usable)
        legend_x = left+i*legend_width
        draw.rectangle((legend_x, height-17*scale, legend_x+12*scale, height-15*scale), fill=color)
        draw_text_fitted(draw, (legend_x+18*scale, height-23*scale), translate(item.get('label', '')),
                         legend_width-24*scale, size=9, fill=MUTED, pixel_scale=scale)
    if start <= seconds <= stop:
        x = left+(seconds-start)/(stop-start)*(right-left)
        draw.line((x, top, x, bottom), fill='#ad5760', width=max(1, round(scale)))
    for fraction in (0., .5, 1.):
        x = left+fraction*(right-left)
        label = format_tick(start+fraction*(stop-start), stop-start)+' s'
        face = font(round(9*scale)); tw = draw.textlength(label, font=face)
        tx = min(max(left, x-tw/2), right-tw)
        draw.text((tx, bottom+5*scale), label, font=face, fill=MUTED)
    return image


class InternalStateApp:
    def __init__(self, root, project_root, result=None):
        self.root, self.project_root = root, Path(project_root)
        self.display_scale = max(1., root.winfo_fpixels('1i')/96)
        self.data = None; self.busy = False; self.closed = False
        self.queue = queue.Queue(); self.photos = {}; self.images = {}; self.maps = {}
        self.clock = PlaybackClock(45.)
        self.playback_window = None
        self.cell = None; self.network = {}; self.cards = []; self.h1_cache = {}
        self.viewports = {}; self.zoom_levels = {}; self.hover_canvas = None
        self.language_code = 'zh'; self.language = tk.StringVar(root, '中文')
        self.translatables = []; self.info_windows = []
        self.status_raw = '正在读取已有结果…'
        self.status = tk.StringVar(root, '正在读取已有结果…')
        self.identity = tk.StringVar(root, '69#6 · OFF-parasol · 3.88°')
        self.detail = tk.StringVar(root, '点击空间点，选择已保存的变量。')
        self.object_name = tk.StringVar(root, '—'); self.object_id = tk.StringVar(root, '—')
        self.object_summary = tk.StringVar(root, '点击空间点，选择已保存的变量。')
        self.coverage_key = None; self.coverage = None
        self.notes_open = False
        self.rgc = tk.StringVar(root); self.variable = tk.StringVar(root)
        self.time = tk.DoubleVar(root, 0.); self.jump = tk.StringVar(root, '0')
        self.time_label = tk.StringVar(root, '0.00000 / 45.00000 s')
        self.start = tk.StringVar(root, '0'); self.stop = tk.StringVar(root, '45')
        self.snapshot = tk.StringVar(root); self.h1 = tk.StringVar(root)
        self.speed = tk.StringVar(root, '1×')
        root.title('Macaque Retina · 内部状态浏览')
        max_width = root.winfo_screenwidth()-round(60*self.display_scale)
        max_height = root.winfo_screenheight()-round(90*self.display_scale)
        root.geometry(f'{min(round(1500*self.display_scale), max_width)}x{min(round(960*self.display_scale), max_height)}')
        root.minsize(min(round(1120*self.display_scale), max_width), min(round(760*self.display_scale), max_height))
        self._build()
        self.register_labels(self.root)
        root.bind('<KeyPress-r>', self.reset_zoom, add='+')
        root.bind('<KeyPress-R>', self.reset_zoom, add='+')
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.timer = root.after(60, self._tick)
        self.root.after(80, lambda: self.load(result or self.project_root/CURRENT_RESULT))

    def _build(self):
        style = ttk.Style(self.root); style.theme_use('clam')
        style.configure('.', font=('Microsoft YaHei UI', 10), background='#f5f6f7', foreground=INK)
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=INK)
        style.configure('TButton', padding=(10, 5), background='#f7f8f9', borderwidth=1, relief='flat')
        style.map('TButton', background=[('active', '#e9eef0'), ('pressed', '#dde6e9')])
        style.configure('Accent.TButton', background='#246979', foreground='white', borderwidth=0)
        style.map('Accent.TButton', background=[('disabled', '#e4eaed'), ('active', '#1b5968')],
                  foreground=[('disabled', '#8b979e')])
        style.configure('TCombobox', fieldbackground=BG, background='#f0f3f4', padding=4, arrowsize=13)
        style.map('TCombobox', fieldbackground=[('readonly', BG)], selectbackground=[('readonly', BG)],
                  selectforeground=[('readonly', INK)])
        style.configure('TEntry', fieldbackground=BG, padding=4)
        style.configure('Internal.TFrame', background='#f5f6f7')
        style.configure('Internal.TLabel', background='#f5f6f7', foreground=MUTED)
        style.configure('Muted.TLabel', background=BG, foreground=MUTED, font=('Microsoft YaHei UI', 9))
        style.configure('Section.TLabel', background=BG, foreground=INK, font=('Microsoft YaHei UI', 10, 'bold'))
        style.configure('TNotebook', background='#f5f6f7', borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure('TNotebook.Tab', padding=(14, 7), font=('Microsoft YaHei UI', 10), borderwidth=0)
        style.map('TNotebook.Tab', background=[('selected', BG), ('!selected', '#edf0f2')],
                  foreground=[('selected', '#246979'), ('!selected', MUTED)])
        style.configure('TLabelframe', background=BG, borderwidth=1, relief='solid', bordercolor='#e2e7ea')
        style.configure('TLabelframe.Label', background=BG, foreground=INK, font=('Microsoft YaHei UI', 10, 'bold'))
        style.configure('Horizontal.TScale', background=BG, troughcolor='#e7ecef', borderwidth=0)
        shell = ttk.Frame(self.root, padding=10, style='Internal.TFrame'); shell.pack(fill='both', expand=True)
        header = ttk.Frame(shell, style='Internal.TFrame'); header.pack(fill='x')
        ttk.Label(header, text='Retina', font=('Microsoft YaHei UI', 14, 'bold'), foreground=INK, style='Internal.TLabel').pack(side='left')
        ttk.Label(header, text='内部状态可视化', style='Internal.TLabel').pack(side='left', padx=12)
        ttk.Button(header, text='导出…', command=self.export).pack(side='right', padx=(5, 0))
        self.more_button = ttk.Menubutton(header, text='更多', width=5)
        self.more_menu = tk.Menu(self.more_button, tearoff=False, font=('Microsoft YaHei UI', 10))
        self.more_menu.add_command(label=self.tr('其他结果回放…'), command=self.legacy)
        self.more_menu.add_command(label=self.tr('身份、来源与缺失项'), command=self.show_info)
        self.more_button.configure(menu=self.more_menu); self.more_button.pack(side='right', padx=5)
        ttk.Button(header, text='打开…', command=self.choose_directory).pack(side='right', padx=5)
        self.language_combo = ttk.Combobox(header, textvariable=self.language, values=list(LANGUAGES.values()), state='readonly', width=9)
        self.language_combo.pack(side='right', padx=8)
        self.language_combo.bind('<<ComboboxSelected>>', lambda _: self.change_language())
        selection = ttk.Frame(shell, padding=(10, 7)); selection.pack(fill='x', pady=(8, 8))
        ttk.Label(selection, text='目标 RGC').pack(side='left', padx=(0, 10))
        self.rgc_combo = ttk.Combobox(selection, textvariable=self.rgc, state='readonly', width=42)
        self.rgc_combo.pack(side='left', fill='x', expand=True)
        self.rgc_combo.bind('<<ComboboxSelected>>', lambda _: self.select_target())
        ttk.Label(selection, textvariable=self.identity, style='Muted.TLabel').pack(side='right', padx=(12, 0))
        footer = ttk.Frame(shell, padding=(10, 5)); footer.pack(side='bottom', fill='x', pady=(8, 0))
        self.coverage_canvas = tk.Canvas(footer, height=round(55*self.display_scale), bg=BG, highlightthickness=0)
        self.coverage_canvas.pack(fill='x')
        self.coverage_canvas.bind('<Configure>', lambda _: self.render_coverage())
        self.coverage_canvas.bind('<Button-1>', self.coverage_seek)
        controls = ttk.Frame(footer); controls.pack(fill='x')
        self.play_button = ttk.Button(controls, text='播放', width=6, style='Accent.TButton', command=self.toggle_play); self.play_button.pack(side='left')
        self.slider = ttk.Scale(controls, from_=0, to=45, variable=self.time, command=self.slider_changed)
        self.slider.pack(side='left', fill='x', expand=True, padx=8)
        ttk.Label(controls, textvariable=self.time_label, width=24, style='Muted.TLabel').pack(side='left', padx=5)
        self.jump_entry = ttk.Entry(controls, textvariable=self.jump, width=9); self.jump_entry.pack(side='left', padx=3)
        self.jump_entry.bind('<Return>', lambda _: self.jump_to())
        ttk.Button(controls, text='跳转 / s', command=self.jump_to).pack(side='left')
        speed = ttk.Combobox(controls, textvariable=self.speed, values=['0.1×', '0.5×', '1×', '2×'], state='readonly', width=5)
        speed.pack(side='left', padx=5); speed.bind('<<ComboboxSelected>>', lambda _: setattr(self.clock, 'speed', float(self.speed.get()[:-1])))
        ttk.Label(footer, textvariable=self.status, style='Muted.TLabel', anchor='w').pack(fill='x', pady=(4, 0))
        body = ttk.Panedwindow(shell, orient='horizontal'); body.pack(fill='both', expand=True)
        side_shell = ttk.Frame(body, width=round(250*self.display_scale)); body.add(side_shell, weight=0)
        side_canvas = tk.Canvas(side_shell, width=round(242*self.display_scale), bg=BG, highlightthickness=0)
        side_scroll = ttk.Scrollbar(side_shell, orient='vertical', command=side_canvas.yview)
        side_scroll.pack(side='right', fill='y'); side_canvas.pack(side='left', fill='both', expand=True)
        side_canvas.configure(yscrollcommand=side_scroll.set)
        sidebar = ttk.Frame(side_canvas, padding=10)
        side_item = side_canvas.create_window(0, 0, anchor='nw', window=sidebar)
        side_canvas.bind('<Configure>', lambda event: side_canvas.itemconfigure(side_item, width=event.width))
        def wrap_sidebar(event):
            side_canvas.configure(scrollregion=side_canvas.bbox('all'))
            available = max(round(120*self.display_scale), event.width-20)
            for widget in (self.summary_label, self.notes_label):
                widget.configure(wraplength=available)
        sidebar.bind('<Configure>', wrap_sidebar)
        ttk.Label(sidebar, text='选中对象', style='Muted.TLabel').pack(anchor='w')
        ttk.Label(sidebar, textvariable=self.object_name, style='Section.TLabel', wraplength=220).pack(fill='x', pady=(4, 6))
        ttk.Label(sidebar, textvariable=self.object_id, style='Muted.TLabel').pack(anchor='w')
        self.summary_label = ttk.Label(sidebar, textvariable=self.object_summary, style='Muted.TLabel', justify='left', wraplength=220)
        self.summary_label.pack(fill='x', pady=(5, 7))
        ttk.Button(sidebar, text='复制 ID', command=self.copy_object_id).pack(anchor='w')
        ttk.Label(sidebar, text='变量', style='Muted.TLabel').pack(anchor='w', pady=(10, 3))
        self.var_combo = ttk.Combobox(sidebar, textvariable=self.variable, state='readonly', width=25)
        self.var_combo.pack(fill='x', pady=(0, 7))
        self.var_combo.bind('<<ComboboxSelected>>', lambda _: self.render())
        ttk.Button(sidebar, text='响应曲线', command=self.show_curve).pack(fill='x')
        ttk.Separator(sidebar).pack(fill='x', pady=12)
        ttk.Label(sidebar, text='时间范围 / s', style='Section.TLabel').pack(anchor='w')
        bounds = ttk.Frame(sidebar); bounds.pack(fill='x', pady=5)
        bounds.columnconfigure(0, weight=1); bounds.columnconfigure(2, weight=1)
        ttk.Entry(bounds, textvariable=self.start, width=7).grid(row=0, column=0, sticky='ew')
        ttk.Label(bounds, text='  至  ', style='Muted.TLabel').grid(row=0, column=1)
        ttk.Entry(bounds, textvariable=self.stop, width=7).grid(row=0, column=2, sticky='ew')
        ttk.Button(sidebar, text='应用时间窗口', command=self.refresh).pack(fill='x')
        shortcuts = ttk.Frame(sidebar); shortcuts.pack(fill='x', pady=(5, 0))
        self.short_button = ttk.Button(shortcuts, text='短窗口', command=self.show_short_window, state='disabled')
        self.short_button.pack(side='left', fill='x', expand=True, padx=(0, 4))
        ttk.Button(shortcuts, text='全片', command=self.show_full_window).pack(side='left', fill='x', expand=True)
        ttk.Label(sidebar, text='保存快照 / s', style='Muted.TLabel').pack(anchor='w', pady=(12, 3))
        self.snapshot_combo = ttk.Combobox(sidebar, textvariable=self.snapshot, state='readonly', width=25)
        self.snapshot_combo.pack(fill='x'); self.snapshot_combo.bind('<<ComboboxSelected>>', lambda _: self.seek(float(self.snapshot.get())))
        ttk.Separator(sidebar).pack(fill='x', pady=12)
        ttk.Label(sidebar, text='灰色 = 未记录 · 不插值', style='Muted.TLabel', wraplength=220).pack(fill='x')
        ttk.Button(sidebar, text='显示说明', command=self.toggle_notes).pack(fill='x', pady=(7, 0))
        self.notes_label = ttk.Label(sidebar, text='点击 H1/H2 或 Cone 可加入同窗响应。\nRGC 原读出与冻结 GLM 单独标记。\n滚轮：缩放当前层 · R：重置',
            style='Muted.TLabel', justify='left', wraplength=220)
        self.notebook = ttk.Notebook(body); body.add(self.notebook, weight=1)
        self.tabs = []
        for title in TABS:
            tab = ttk.Frame(self.notebook, padding=7); self.notebook.add(tab, text=title); self.tabs.append(tab)
        self.notebook.bind('<<NotebookTabChanged>>', lambda _: self.render())
        stimulus_strip = ttk.Frame(self.tabs[0], padding=(5, 4))
        stimulus_strip.pack(fill='x', padx=3, pady=(0, 7))
        preview_size = round(76*self.display_scale)
        self.stimulus_canvas = tk.Canvas(stimulus_strip, width=preview_size, height=preview_size,
            bg='#17212b', highlightthickness=0)
        self.stimulus_canvas.pack(side='left')
        self.stimulus_caption = tk.StringVar(self.root)
        stimulus_label = ttk.Label(stimulus_strip, textvariable=self.stimulus_caption,
            justify='left', style='Muted.TLabel')
        stimulus_label.pack(side='left', fill='x', expand=True, padx=12)
        stimulus_label.bind('<Configure>', lambda event: stimulus_label.configure(wraplength=max(100, event.width-8)))
        self.stimulus_canvas.bind('<Configure>', lambda _: self.schedule_render())
        grid = ttk.Frame(self.tabs[0]); grid.pack(fill='both', expand=True)
        for c in range(3): grid.columnconfigure(c, weight=1, uniform='columns')
        for r in range(2): grid.rowconfigure(r, weight=1, uniform='rows')
        for i, (title, names) in enumerate(LAYERS):
            frame = ttk.LabelFrame(grid, text=title, padding=4)
            frame.grid(row=i//3, column=2 if title == 'RGC' else i%3,
                columnspan=2 if title == 'AC' else 1, sticky='nsew', padx=3, pady=3)
            pop = tk.StringVar(self.root, names[0])
            if title == 'AC':
                self.ac_filter = pop; self.ac_choice = tk.StringVar(self.root, self.tr(AC_FILTERS[pop.get()]))
                combo = ttk.Combobox(frame, textvariable=self.ac_choice, state='readonly',
                    values=[self.tr(label) for label in AC_FILTERS.values()], width=20)
                self.ac_combo = combo
                combo.bind('<<ComboboxSelected>>', lambda _: self.choose_ac_filter())
            else:
                combo = ttk.Combobox(frame, textvariable=pop, state='readonly', values=names, width=20)
                combo.bind('<<ComboboxSelected>>', lambda _: self.render())
            combo.pack(fill='x')
            canvas = tk.Canvas(frame, bg=BG, highlightthickness=0); canvas.pack(fill='both', expand=True)
            caption = tk.StringVar(self.root)
            caption_label = ttk.Label(frame, textvariable=caption, wraplength=round(280*self.display_scale), font=('Microsoft YaHei UI', 8), foreground=MUTED)
            caption_label.pack(side='bottom', fill='x', before=canvas)
            caption_label.bind('<Configure>', lambda event, label=caption_label: label.configure(wraplength=max(100, event.width-8)))
            canvas.bind('<Button-1>', lambda e, c=canvas: self.click_cell(c, e))
            canvas.bind('<Configure>', lambda _: self.schedule_render())
            canvas.bind('<Enter>', lambda _, c=canvas: setattr(self, 'hover_canvas', c))
            canvas.bind('<Leave>', lambda _, c=canvas: self.leave_canvas(c))
            canvas.bind('<MouseWheel>', lambda event, c=canvas: self.on_wheel(c, event))
            self.cards.append((canvas, pop, caption, combo))
            if title == 'AC': self.ac_canvas = canvas
        for i in range(1, len(TABS)):
            if i == 2:
                bar = ttk.Frame(self.tabs[i]); bar.pack(fill='x')
                ttk.Label(bar, text='已保存完整输入行的真实 H1').pack(side='left', padx=4)
                self.h1_combo = ttk.Combobox(bar, textvariable=self.h1, state='readonly', width=20); self.h1_combo.pack(side='left')
                self.h1_combo.bind('<<ComboboxSelected>>', lambda _: self.render())
            canvas = tk.Canvas(self.tabs[i], bg=BG, highlightthickness=0); canvas.pack(fill='both', expand=True)
            canvas.bind('<Configure>', lambda _: self.schedule_render())
            self.tabs[i].canvas = canvas
        self.curve_window = None; self.render_pending = False

    def tr(self, text):
        return translate(str(text), self.language_code)

    def toggle_notes(self):
        self.notes_open = not self.notes_open
        if self.notes_open: self.notes_label.pack(fill='x', pady=8)
        else: self.notes_label.pack_forget()

    def copy_object_id(self):
        if not self.data or not self.cell: return
        pop, row = self.cell
        self.root.clipboard_clear(); self.root.clipboard_append(str(self.data.geometry[pop]['ids'][row]))
        self.set_status('已复制对象 ID')

    def show_full_window(self):
        if not self.data: return
        self.playback_window = None
        self.start.set('0'); self.stop.set(f'{self.data.duration:g}'); self.refresh()

    def recording_coverage(self):
        if not self.data or not self.cell or not self.variable.get(): return None
        cache_key = (id(self.data), self.cell, self.variable.get())
        if cache_key == self.coverage_key: return self.coverage
        pop, row = self.cell; key = self.variable.get()
        max_points = max(2, len(self.data._events.get(pop, []))) if key == 'spike' else 512
        series = self.data.series(pop, row, key, 0., self.data.duration, max_points=max_points)
        valid = np.isfinite(series['values'])
        times = np.asarray(series['time'])[valid]
        kind = series['recording']; intervals = []; snapshots = []
        if kind == 'continuous' and len(times): intervals = [(0., self.data.duration)]
        elif kind == 'events': intervals = [(0., self.data.duration)]
        elif kind == 'windowed':
            begin, end = series['continuous_window']
            if np.any((times >= begin) & (times <= end)): intervals = [(begin, end)]
            snapshots = times[(times < begin) | (times > end)].tolist()
        else: snapshots = times.tolist()
        self.coverage_key = cache_key
        self.coverage = dict(intervals=intervals, snapshots=snapshots, kind=kind,
                             events=times.tolist() if kind == 'events' else [], variable=key)
        return self.coverage

    def render_coverage(self):
        canvas = self.coverage_canvas
        canvas.delete('all')
        if not self.data or not self.cell: return
        coverage = self.recording_coverage()
        if coverage is None: return
        s = self.display_scale; width = canvas.winfo_width()
        left, right = 8*s, width-10*s
        if right <= left: return
        top, bottom = 24*s, 33*s
        duration = self.data.duration
        x_at = lambda t: left+np.clip(t/duration, 0., 1.)*(right-left)
        self.coverage_track = (left, right, top, bottom)
        label = self.tr('记录覆盖')+' · '+coverage['variable']
        if coverage['kind'] == 'events': label += ' · '+self.tr('事件记录')
        elif coverage['intervals'] == [(0., duration)]: label += ' · '+self.tr('全程连续')
        elif coverage['intervals']:
            a, b = coverage['intervals'][0]; label += f' · {a:g}–{b:g} s'
        elif not coverage['snapshots']: label += ' · '+self.tr('未保存该变量')
        canvas.create_text(left, 5*s, text=label, anchor='nw', font=('Microsoft YaHei UI', 9), fill=INK)
        legend = self.tr('连续记录')+'  /  '+self.tr('快照')+'  /  '+self.tr('未记录')
        if coverage['kind'] == 'events': legend = self.tr('事件时刻；无事件不表示缺失')
        canvas.create_text(right, 5*s, text=legend, anchor='ne', font=('Microsoft YaHei UI', 8), fill=MUTED)
        canvas.create_rectangle(left, top, right, bottom, fill='#edf0f2', outline='')
        for a, b in coverage['intervals']:
            canvas.create_rectangle(x_at(a), top, max(x_at(a)+2*s, x_at(b)), bottom,
                                    fill='#88adb7' if coverage['kind'] != 'events' else '#c6d9de', outline='')
        for x in sorted({round(x_at(t)) for t in coverage['events']}):
            canvas.create_line(x, top, x, bottom, fill='#507e89')
        for t in coverage['snapshots']:
            x = x_at(t); canvas.create_polygon(x, top-2*s, x+3*s, (top+bottom)/2,
                x, bottom+2*s, x-3*s, (top+bottom)/2, fill='#927846', outline='')
        x = x_at(self.clock.position)
        canvas.create_line(x, top-4*s, x, bottom+3*s, fill='#246979', width=max(1, round(2*s)))
        for t in (0., duration/2, duration):
            canvas.create_text(x_at(t), bottom+4*s, text=f'{t:g} s',
                anchor='nw' if t == 0 else 'ne' if t == duration else 'n', font=('Microsoft YaHei UI', 8), fill=MUTED)

    def coverage_seek(self, event):
        if not self.data or not hasattr(self, 'coverage_track'): return
        left, right, top, bottom = self.coverage_track
        if not left <= event.x <= right or not top-7*self.display_scale <= event.y <= bottom+7*self.display_scale: return
        t = (event.x-left)/(right-left)*self.data.duration
        snaps = self.recording_coverage()['snapshots']
        nearest = min(snaps, key=lambda v: abs(v-t)) if snaps else None
        if nearest is not None and abs(nearest-t)*(right-left)/self.data.duration < 6*self.display_scale: t = nearest
        self.seek(t)

    def set_status(self, text):
        self.status_raw = str(text); self.status.set(self.status_text())

    def status_text(self):
        prefix = '已导出：'
        if self.status_raw.startswith(prefix): return self.tr(prefix)+self.status_raw[len(prefix):]
        return self.tr(self.status_raw)

    def register_labels(self, parent):
        for widget in parent.winfo_children():
            keys = widget.keys()
            if 'text' in keys and ('textvariable' not in keys or not widget.cget('textvariable')):
                raw = str(widget.cget('text'))
                if raw:
                    self.translatables.append((widget, raw)); widget.configure(text=self.tr(raw))
            self.register_labels(widget)

    def change_language(self, code=None):
        code = code or next(k for k, value in LANGUAGES.items() if value == self.language.get())
        if code not in LANGUAGES: raise ValueError('Unknown UI language')
        self.language_code = code; self.language.set(LANGUAGES[code])
        self.more_menu.entryconfigure(0, label=self.tr('其他结果回放…'))
        self.more_menu.entryconfigure(1, label=self.tr('身份、来源与缺失项'))
        self.root.title(self.tr('Macaque Retina · 内部状态浏览'))
        for widget, raw in self.translatables:
            if widget.winfo_exists(): widget.configure(text=self.tr(raw))
        for index, title in enumerate(TABS): self.notebook.tab(index, text=self.tr(title))
        self.ac_combo.configure(values=[self.tr(label) for label in AC_FILTERS.values()])
        self.ac_choice.set(self.tr(AC_FILTERS[self.ac_filter.get()]))
        if self.data:
            selected = next(i for i, item in enumerate(self.targets) if (item['population'], item['row']) == self.target)
            self.target_labels = [self.tr(x['label']) for x in self.targets]
            self.rgc_combo.configure(values=self.target_labels); self.rgc.set(self.target_labels[selected])
            item = self.targets[selected]
            self.identity.set(self.tr(f"{item.get('eccentricity_role', '')} · {item['eccentricity_deg']:.3f}°"))
            if self.cell: self.update_cell_detail(*self.cell)
        if self.curve_window and self.curve_window.winfo_exists(): self.curve_window.title(self.tr('保存响应 · 选中对象'))
        for window, widget, raw in self.info_windows:
            if window.winfo_exists():
                window.title(self.tr('数据身份与覆盖')); widget.configure(state='normal')
                widget.delete('1.0', 'end'); widget.insert('1.0', self.tr(raw)); widget.configure(state='disabled')
        self.status.set(self.status_text()); self.render()

    def leave_canvas(self, canvas):
        if self.hover_canvas is canvas: self.hover_canvas = None

    def layer_scale(self, canvas):
        return min(self.display_scale, max(1., min(canvas.winfo_width()/260, canvas.winfo_height()/210)))

    def choose_ac_filter(self):
        self.ac_filter.set(next(key for key, label in AC_FILTERS.items() if self.tr(label) == self.ac_choice.get()))
        objects = self.ac_objects()
        if objects and self.cell and self.cell[0] in AC_POPULATIONS and self.cell not in objects:
            self.select_cell(*objects[0])
        self.render()

    def ac_populations(self, mode=None):
        mode = mode or self.ac_filter.get()
        return ('AII_AC',) if mode == 'AII_AC' else AC_POPULATIONS

    def ac_objects(self, mode=None):
        return [(pop, int(row)) for pop in self.ac_populations(mode) for row in self.network.get(pop, [])]

    def ac_state(self, mode=None):
        objects = self.ac_objects(mode)
        key = self.variable.get() if self.cell in objects else 'V'
        unit = self.data.variables(self.cell[0])[key]['unit'] if self.cell in objects else 'V'
        values = []; times = []; markers = []; xy = []
        states = {}
        for pop, row in objects:
            meta = self.data.variables(pop).get(key)
            if pop not in states:
                states[pop] = self.data.spatial(pop, key, self.clock.position) if meta and meta['unit'] == unit else None
            state = states[pop]
            value = state['values'][row] if state is not None else np.nan
            values.append(value)
            if state is not None and np.isfinite(value): times.append(float(state['time']))
            g = self.data.geometry[pop]; xy.append(g['xy_mm'][row])
            markers.append('circle' if pop == 'AII_AC' else 'square' if g['representation'] == 'effective_functional_pool' else 'diamond')
        return dict(objects=objects, xy_mm=np.asarray(xy).reshape(-1, 2), values=np.asarray(values),
                    key=key, unit=unit, times=sorted(set(times)), markers=markers)

    def on_wheel(self, canvas, event):
        if not self.data or canvas not in self.maps or not event.delta: return
        pop = self.maps[canvas][0]
        xy = self.ac_state(pop)['xy_mm'] if canvas is self.ac_canvas else self.data.geometry[pop]['xy_mm']
        rows = np.arange(len(xy)) if canvas is self.ac_canvas else self.network.get(pop, [])
        if not len(rows): return
        s = self.layer_scale(canvas)
        width, height = canvas.winfo_width(), canvas.winfo_height()
        bounds = scatter_bounds(xy, rows, (width, height), self.viewports.get(pop), s)
        left, top, right, bottom = 64*s, 43*s, width-107*s, height-54*s
        if not (left <= event.x <= right and top <= event.y <= bottom): return
        previous = self.zoom_levels.get(pop, 1.)
        level = min(64., max(1., previous * 1.25**(event.delta/120)))
        if level == previous: return 'break'
        lower, upper = bounds
        fraction = np.array([(event.x-left)/(right-left), (bottom-event.y)/(bottom-top)])
        anchor = lower+fraction*(upper-lower)
        span = (upper-lower)*previous/level
        if level <= 1.:
            self.viewports.pop(pop, None); self.zoom_levels.pop(pop, None)
        else:
            self.viewports[pop] = (anchor-fraction*span, anchor+(1-fraction)*span)
            self.zoom_levels[pop] = level
        self.hover_canvas = canvas; canvas.focus_set(); self.render()
        return 'break'

    def reset_zoom(self, event=None):
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text)): return
        canvas = self.hover_canvas
        if canvas not in self.maps: return
        pop = self.maps[canvas][0]
        self.viewports.pop(pop, None); self.zoom_levels.pop(pop, None); self.render()
        return 'break'

    def show_short_window(self):
        window = getattr(self.data, 'continuous_window', None) if self.data else None
        if window is None: self.set_status('当前结果没有短窗口连续记录'); return
        self.playback_window = tuple(window)
        self.start.set(f'{window[0]:g}'); self.stop.set(f'{window[1]:g}')
        self.seek(window[0]); self.set_status('已加载短窗口连续状态')

    def schedule_render(self):
        if not self.render_pending:
            self.render_pending = True; self.root.after(100, self._scheduled_render)

    def _scheduled_render(self):
        self.render_pending = False
        if not self.closed: self.render()

    def load(self, path):
        if self.busy: return
        self.clock.pause(); self.busy = True; self.set_status('读取模型身份、成员映射与保存记录…')
        def work():
            try:
                from .internal_state_data import SavedNetworkData
                self.queue.put(('loaded', SavedNetworkData(self.project_root, path)))
            except Exception as error: self.queue.put(('error', str(error)))
        threading.Thread(target=work, daemon=True).start()

    def loaded(self, data):
        if self.data: self.data.close()
        self.data = data; self.clock = PlaybackClock(data.duration); self.clock.seek(data.duration)
        self.coverage_key = None; self.coverage = None
        self.clock.speed = float(self.speed.get()[:-1])
        self.start.set('0'); self.jump.set(f'{data.duration:.5f}')
        self.playback_window = None
        self.h1_cache.clear()
        self.viewports.clear(); self.zoom_levels.clear()
        self.slider.configure(to=data.duration); self.stop.set(f'{data.duration:g}')
        self.targets = data.rgcs
        self.target_labels = [self.tr(x['label']) for x in self.targets]
        self.rgc_combo.configure(values=self.target_labels)
        target = next((i for i, x in enumerate(self.targets) if x.get('is_target')), 0)
        self.rgc.set(self.target_labels[target])
        self.snapshot_combo.configure(values=[f'{x:.7g}' for x in data.snapshot_times])
        self.snapshot.set(f'{data.duration:g}')
        self.h1_combo.configure(values=list(data.available_h1_rows))
        if data.available_h1_rows: self.h1.set(str(data.available_h1_rows[0]))
        for _, pop, _, combo in self.cards:
            if combo is self.ac_combo: continue
            names = [x for x in combo['values'] if x in data.geometry]
            combo.configure(values=names)
            if names: pop.set(names[0])
        self.references = {}
        self.short_button.configure(state='normal' if getattr(data, 'continuous_window', None) is not None else 'disabled')
        self.select_target()
        self.set_status('已加载短窗口连续状态' if getattr(data, 'continuous_window', None) is not None else '已加载当前 69#6 · 45 秒 RGC 连续记录 · 各层仅保存时刻可见 · 无模型计算')

    def select_target(self):
        if not self.data or self.rgc.get() not in self.target_labels: return
        item = self.targets[self.target_labels.index(self.rgc.get())]
        self.target = (item['population'], item['row'])
        self.cards[-1][1].set(item['population'])
        network = self.data.network(self.target)
        self.network = network[0] if isinstance(network, tuple) else network
        self.references = {key: ref for key, ref in self.references.items() if ref[1] in self.network.get(ref[0], [])}
        for _, pop, _, combo in self.cards:
            if combo is self.ac_combo: continue
            if not len(self.network.get(pop.get(), [])):
                available = [name for name in combo['values'] if len(self.network.get(name, []))]
                if available: pop.set(available[0])
        self.identity.set(self.tr(f"{item.get('eccentricity_role', '')} · {item['eccentricity_deg']:.3f}°"))
        self.select_cell(*self.target); self.render()

    def select_cell(self, pop, row):
        self.cell = (pop, int(row)); g = self.data.geometry[pop]
        self.update_cell_detail(pop, row)
        variables = self.data.variables(pop)
        self.var_combo.configure(values=list(variables))
        if variables: self.variable.set(self.primary(pop))
        else: self.variable.set('')
        if pop == 'cones': self.references['Cone'] = (pop, row)
        if pop == 'HC': self.references[str(g['subtypes'][row])] = (pop, row)
        if pop.endswith('_BC_ON') or pop.endswith('_BC_OFF') or pop == 'S_ON_BC': self.references['BC'] = (pop, row)
        if pop in AC_POPULATIONS and pop != 'AII_AC': self.references['AC'] = (pop, row)
        if pop == 'AII_AC': self.references['AII'] = (pop, row)
        self.render_curve()

    def update_cell_detail(self, pop, row):
        g = self.data.geometry[pop]
        xy = g['xy_mm'][row]; members = g['members'][row]
        role = self.role_label(g['coordinate_role'])
        membership = ('功能池；解剖成员数未定义' if g.get('representation') == 'effective_functional_pool'
            else f"代表成员：{members}" + (' · 独立 BC 状态' if g.get('representation') == 'independently_solved_BC_cell' else ' · 组/池状态'))
        self.detail.set(self.tr(f"{g['ids'][row]}\n{g['subtypes'][row]}\n({xy[0]:.5f}, {xy[1]:.5f}) mm\n{role}\n{membership}"))
        identifier = str(g['ids'][row])
        self.object_name.set(str(g['subtypes'][row]))
        self.object_id.set(identifier if len(identifier) <= 28 else '…'+identifier[-27:])
        self.object_summary.set(self.tr(f"({xy[0]:.5f}, {xy[1]:.5f}) mm\n{role}\n{membership}"))

    def bounds(self):
        start, stop = float(self.start.get()), float(self.stop.get())
        if not np.isfinite([start, stop]).all() or not 0 <= start < stop <= self.data.duration:
            raise ValueError('时间窗口必须在 0–45 秒内，且起点小于终点。')
        return start, stop

    def refresh(self):
        try: self.bounds(); self.render()
        except ValueError as error: self.set_status(str(error))

    def put(self, canvas, image):
        self.images[canvas] = image; self.photos[canvas] = ImageTk.PhotoImage(image, master=self.root)
        canvas.delete('all'); canvas.create_image(0, 0, image=self.photos[canvas], anchor='nw')

    def render(self):
        if not self.data or self.closed: return
        self.time.set(self.clock.position)
        self.time_label.set(f'{self.clock.position:.3f} / {self.data.duration:g} s')
        self.play_button.configure(text=self.tr('暂停' if self.clock.playing else '播放'))
        index = self.notebook.index(self.notebook.select())
        s = self.display_scale
        try:
            self.render_coverage()
            if index == 0:
                self.render_stimulus()
                for canvas, pop, caption, _ in self.cards:
                    size = (canvas.winfo_width(), canvas.winfo_height())
                    if min(size) <= 1: continue
                    image, mapping, text = self.layer_image(pop.get(), size, self.layer_scale(canvas))
                    self.put(canvas, image); self.maps[canvas] = (pop.get(), mapping); caption.set(text)
            else:
                canvas = self.tabs[index].canvas
                size = (canvas.winfo_width(), canvas.winfo_height())
                if min(size) > 1: self.put(canvas, self.figure(index, size, s))
            self.render_curve()
        except (ValueError, KeyError, OSError) as error: self.set_status('读取提示：'+str(error))

    def render_stimulus(self):
        self.stimulus_frame = self.data.stimulus(self.clock.position)
        frame = self.stimulus_frame
        width, height = max(1, self.stimulus_canvas.winfo_width()), max(1, self.stimulus_canvas.winfo_height())
        preview = Image.new('RGB', (width, height), '#17212b')
        if frame.get('image') is None:
            self.stimulus_caption.set(self.tr(frame.get('status', '原刺激帧文件不可用')))
        else:
            source = Image.fromarray(frame['image'])
            original_width, original_height = source.size
            source.thumbnail((width, height), Image.Resampling.LANCZOS)
            preview.paste(source, ((width-source.width)//2, (height-source.height)//2))
            self.stimulus_caption.set(
                f"{self.tr('原刺激帧 ')}{frame['time']:.5f} s\n"
                f"{self.tr('帧索引（从 0 开始）')}: {frame['source_frame']} · "
                f"{self.tr('原始分辨率')}: {original_width} × {original_height} px\n"
                +self.tr('与底部时间轴同步 · 保持原始比例'))
        self.put(self.stimulus_canvas, preview)

    def layer_image(self, pop, size, pixel_scale=1.):
        s = pixel_scale
        if pop in AC_FILTERS: return self.ac_image(pop, size, pixel_scale)
        if pop not in self.data.geometry: return empty_plot(size, self.tr('未保存空间映射'), s), [], self.tr('未记录')
        g = self.data.geometry[pop]; variables = self.data.variables(pop)
        key = self.variable.get() if self.cell and self.cell[0] == pop else self.primary(pop)
        state = self.data.spatial(pop, key, self.clock.position) if key else {'values': np.full(len(g['ids']), np.nan), 'status': '未记录', 'unit': ''}
        values = np.asarray(state['values'])
        rows = np.asarray(self.network.get(pop, []), dtype=int)
        if not len(rows): return empty_plot(size, self.tr('该类型不在当前 RGC 的连接祖先内'), s), [], self.role_label(g['coordinate_role'])
        viewport = self.viewports.get(pop)
        all_rows = rows.copy()
        if viewport is not None:
            lower, upper = scatter_bounds(g['xy_mm'], rows, size, viewport, s)
            visible = ((g['xy_mm'][rows] >= lower) & (g['xy_mm'][rows] <= upper)).all(axis=1)
            rows = rows[visible]
            if not len(rows):
                return empty_plot(size, self.tr('该位置没有细胞；滚轮缩小或按 R 重置'), s), [], self.role_label(g['coordinate_role'])
        if len(rows) > 2500: rows = rows[np.linspace(0, len(rows)-1, 2500, dtype=int)]
        selected = self.cell[1] if self.cell and self.cell[0] == pop else None
        if selected is not None and selected in self.network.get(pop, []) and selected not in rows: rows = np.append(rows, selected)
        unit = state['unit']; factor = 1000 if unit == 'V' else 1
        shown_unit = 'mV' if unit == 'V' else self.unit_label(unit)
        finite = values[all_rows][np.isfinite(values[all_rows])]*factor
        fixed_range = (float(finite.min()), float(finite.max())) if viewport is not None and len(finite) else None
        image, mapping = scatter_plot(g['xy_mm'], values*factor, rows, size, value_unit=shown_unit,
            selected=selected, title=self.tr(variables.get(key, {}).get('label', pop)), coordinate='', coordinate_unit='mm',
            viewport=viewport, fixed_range=fixed_range, translate=self.tr, pixel_scale=s)
        if not np.isfinite(values[rows]).any():
            draw = ImageDraw.Draw(image)
            draw.rectangle((size[0]-105*s, 35*s, size[0], size[1]-20*s), fill=BG)
            draw.text((size[0]-100*s, 65*s), self.tr('未记录'), font=font(round(11*s)), fill='#64748b')
        timestamp = f" · 样本 {state['time']:.5f}s" if np.isfinite(values[rows]).any() and 'time' in state else ''
        caption = self.tr(f"{state.get('status', '')}{timestamp}\n{self.role_label(g['coordinate_role'])} · {len(mapping)}/{len(self.network.get(pop, []))} 个对象")
        if viewport is not None: caption += f" · {self.zoom_levels[pop]:.2g}×"
        return image, mapping, caption

    def ac_image(self, mode, size, pixel_scale=1.):
        s = pixel_scale; state = self.ac_state(mode); objects = state['objects']
        if not objects: return empty_plot(size, self.tr('该类型不在当前 RGC 的连接祖先内'), s), [], self.tr('当前 RGC 网络内无 AC 对象')
        xy = state['xy_mm']; values = state['values']; rows = np.arange(len(objects))
        viewport = self.viewports.get(mode)
        unit = state['unit']; factor = 1000 if unit == 'V' else 1
        finite = values[np.isfinite(values)]*factor
        selected = objects.index(self.cell) if self.cell in objects else None
        image, points = scatter_plot(xy, values*factor, rows, size,
            value_unit='mV' if unit == 'V' else self.unit_label(unit), selected=selected,
            fixed_range=(float(finite.min()), float(finite.max())) if viewport is not None and len(finite) else None,
            title=self.tr('膜电位') if state['key'] == 'V' else state['key'], coordinate='RF',
            viewport=viewport, translate=self.tr, pixel_scale=s, markers=state['markers'])
        if not len(finite):
            draw = ImageDraw.Draw(image)
            draw.rectangle((size[0]-105*s, 35*s, size[0], size[1]-20*s), fill=BG)
            draw.text((size[0]-100*s, 65*s), self.tr('未记录'), font=font(round(11*s)), fill='#64748b')
        mapping = [(x, y, objects[row]) for x, y, row in points]
        aii = sum(pop == 'AII_AC' for pop, _ in objects)
        labels = []
        if 'square' in state['markers']: labels.append(self.tr('□ 普通 AC 功能池（非解剖细胞）'))
        if 'diamond' in state['markers']: labels.append(self.tr('◇ S 通路 AC 代表组 RF'))
        if aii: labels.append(self.tr('○ AII 代表组 RF'))
        time_label = ', '.join(f'{t:.5f}' for t in state['times'])
        caption = (self.tr('当前 RGC 连接范围')+f" · AC {len(objects)-aii} + AII {aii} · "
            +self.tr('可见 / 总数')+f" {len(mapping)}/{len(objects)} · {state['key']} "
            +self.tr('已记录')+f" {len(finite)}/{len(objects)}"
            +(self.tr(' · 样本 ')+time_label+' s' if time_label else '')+'\n'+' · '.join(labels))
        if len(finite) < len(objects): caption += '\n'+self.tr('灰点：该变量或该时刻未记录')
        if viewport is not None: caption += f" · {self.zoom_levels[mode]:.2g}×"
        return image, mapping, caption

    def primary(self, pop):
        variables = self.data.variables(pop)
        return next((key for key in ('Vh', 'V', 'native_input', 'release', 's', 'h') if key in variables), next(iter(variables), None))

    def role_label(self, role):
        return self.tr({'rf': '功能 RF 锚点（非胞体）', 'soma': '胞体位置', 'cone_inner_segment': 'Cone 内节位置',
                'effective_functional_RF_pool_not_anatomical_AC': '功能 RF 池（非解剖 AC）',
                'opl_pedicle': 'Cone OPL 突触位置'}.get(role, role))

    @staticmethod
    def unit_label(unit):
        return {'vH_native_voltage': 'native', 'dimensionless effective release': 'effective (1)',
                'reference-equivalent native drive': 'native drive'}.get(unit, unit)

    def figure(self, index, size, pixel_scale=1.):
        if index == 1: return self.optics_figure(size, pixel_scale)
        if index == 2: return self.h1_figure(size, pixel_scale)
        if index == 3: return self.window_figure(size, pixel_scale)
        return self.comparison_figure(size, pixel_scale)

    def optics_figure(self, size, pixel_scale=1.):
        s = pixel_scale
        capture = self.data.optical_capture(self.clock.position)
        image = empty_plot(size, '', s)
        panel_width = int(size[0]*.7)
        panel, _ = scatter_plot(capture['xy_mm'], capture['values'], np.arange(0, len(capture['values']), 3), (panel_width, size[1]-round(35*s)),
            title=self.tr(f"ISETBio Cone 期望光捕获 · t = {capture['time']:.6f} s"), value_unit='R*/frame', coordinate=self.tr('真实 Cone IS'), coordinate_unit='mm', translate=self.tr, pixel_scale=s)
        image.paste(panel, (0, 0))
        stimulus = self.data.stimulus(self.clock.position)
        draw = ImageDraw.Draw(image)
        x = panel_width+round(15*s)
        if stimulus.get('image') is not None:
            thumb = Image.fromarray(stimulus['image']); thumb.thumbnail((max(1, size[0]-x-round(20*s)), round(310*s)), Image.Resampling.LANCZOS)
            image.paste(thumb, (x, round(55*s)))
            draw.text((x, 65*s+thumb.height), self.tr(f"原刺激帧 {stimulus['time']:.5f} s"), font=font(round(11*s)), fill='#475569')
        draw.text((x, min(size[1]-160*s, 400*s)), self.tr('独立 Cone 光捕获\n9266 个真实 IS 位置\n显示每 3 个中的 1 个\n每帧曝光 1/150 s\n冻结 Gaussian 孔径\n借用的人 Cone 晶格'), font=font(round(11*s)), fill='#475569', spacing=round(7*s))
        draw.text((20*s, size[1]-22*s), self.tr(capture.get('status', '')), font=font(round(10*s)), fill='#64748b')
        return image

    def h1_figure(self, size, pixel_scale=1.):
        s = pixel_scale
        if not self.h1.get(): return empty_plot(size, self.tr('没有保存完整 H1 短、长分量输入行'), s)
        selected = int(self.h1.get())
        if selected not in self.h1_cache: self.h1_cache[selected] = self.data.h1_inputs(selected)
        row = self.h1_cache[selected]
        meta = row['metadata']
        image = empty_plot(size, '', s)
        width = size[0]//3
        indices = np.unique(np.r_[np.linspace(0, len(row['ids'])-1, min(2400, len(row['ids'])), dtype=int),
            np.argsort(row['total'])[-300:], np.argsort(row['short'])[-300:]])
        for i, (key, label) in enumerate((('short', '短分量'), ('long', '长分量'), ('total', '总权重'))):
            shown = np.flatnonzero(row['distance_mm'] <= max(10*meta['lambda_short_mm'], .025)) if key == 'short' else indices
            panel, _ = scatter_plot(row['xy_mm'], row[key], shown, (width, size[1]-round(90*s)),
                title=self.tr(f"{label}{' · 近中心放大' if key == 'short' else ''} · Σw = {np.sum(row[key]):.5f}"), value_unit='1', coordinate='Cone OPL', coordinate_unit='mm', translate=self.tr, pixel_scale=s)
            image.paste(panel, (i*width, 0))
        draw = ImageDraw.Draw(image)
        draw.text((20*s, size[1]-78*s), self.tr(f"H1 {meta['hc_source_key']} · λ短 {meta['lambda_short_mm']*1000:.3f} µm / λ长 {meta['lambda_long_mm']*1000:.3f} µm · 完整输入 {len(row['ids']):,}"), font=font(round(12*s)), fill='#173745')
        draw.text((20*s, size[1]-52*s), self.tr(f"各分量按完整支持归一化后，以 0.15 / 0.85 混合 · 短分量仅放大显示；长/总抽样 {len(indices):,} 点并保留最大权重"), font=font(round(11*s)), fill='#64748b')
        draw.text((20*s, size[1]-27*s), self.tr(f"Cone OPL 坐标，含静态外域功能输入 · H1 响应属于 {meta['group_members']} 成员代表组；无独立单细胞响应。"), font=font(round(11*s)), fill='#64748b')
        return image

    def window_figure(self, size, pixel_scale=1.):
        start, stop = self.bounds(); image = empty_plot(size, '', pixel_scale)
        self.figure_references = {}
        requests = [('Cone', 'release'), ('H1', 'Vh'), ('H2', 'Vh'), ('BC', 'V'), ('AC', 'V'), ('AII', 'V'), ('RGC', 'V')]
        for i, (label, key) in enumerate(requests):
            cell = self.target if label == 'RGC' else self.references.get(label)
            if cell is None and label == 'Cone' and 'cones' in self.data.geometry:
                rows = self.network.get('cones', []); cell = ('cones', int(rows[0])) if len(rows) else None
            if cell is None and label in ('H1', 'H2') and 'HC' in self.data.geometry:
                g = self.data.geometry['HC']; rows = [r for r in self.network.get('HC', []) if str(g['subtypes'][r]) == label]
                cell = ('HC', int(rows[0])) if rows else None
            if cell is None and label in ('BC', 'AC', 'AII'):
                if label == 'BC':
                    pop = self.cards[2][1].get(); rows = self.network.get(pop, [])
                    cell = (pop, int(rows[0])) if len(rows) else None
                else:
                    candidates = [item for item in self.ac_objects('AC_all') if (item[0] == 'AII_AC') == (label == 'AII')]
                    cell = candidates[0] if candidates else None
            series = []
            if cell and key in self.data.variables(cell[0]):
                s = self.data.series(*cell, key, start, stop); series = [s]
                self.figure_references[label] = dict(population=cell[0], row=int(cell[1]), id=str(self.data.geometry[cell[0]]['ids'][cell[1]]), variable=key)
            unit = series[0]['unit'] if series else ''
            title = f"{label} · {key}"
            subtitle = ''
            if series and series[0]['recording'] == 'snapshot': subtitle = '仅保存快照'
            if series and series[0]['recording'] == 'windowed': subtitle = '短窗口连续记录'
            if cell:
                identifier = str(self.data.geometry[cell[0]]['ids'][cell[1]])
                subtitle += (' · ' if subtitle else '')+(identifier if len(identifier) <= 42 else '…'+identifier[-39:])
            width = size[0] if label == 'RGC' else size[0]//2
            panel = trace_panel((width, size[1]//4), series, start, stop, self.clock.position, title, self.unit_label(unit),
                '该窗口未保存连续记录；快照仅以圆点显示，不连接、不插值', translate=self.tr, pixel_scale=pixel_scale, subtitle=subtitle)
            image.paste(panel, ((i%2)*(size[0]//2), (i//2)*(size[1]//4)))
        return image

    def comparison_figure(self, size, pixel_scale=1.):
        start, stop = self.bounds()
        result = self.data.comparison(start, stop)
        series = [dict(time=result['time'], values=result[key], label=label, recording='continuous')
                  for key, label in [('measured', '实测 PSTH'), ('glm', '冻结 GLM 自由预测'), ('base', '原 RGC 期望率')]]
        return trace_panel(size, series, start, stop, self.clock.position,
            '69#6 · 20 ms PSTH · 实测 / 冻结 GLM 自由预测 / 原 RGC', 'Hz', translate=self.tr, pixel_scale=pixel_scale)

    def click_cell(self, canvas, event):
        pop, points = self.maps.get(canvas, ('', []))
        if not points: return
        x, y, row = min(points, key=lambda p: (p[0]-event.x)**2+(p[1]-event.y)**2)
        if (x-event.x)**2+(y-event.y)**2 <= (12*self.layer_scale(canvas))**2:
            canvas.focus_set(); self.select_cell(*(row if canvas is self.ac_canvas else (pop, row))); self.render()

    def show_curve(self):
        if not self.data or not self.cell: return
        if self.curve_window and self.curve_window.winfo_exists(): self.curve_window.lift(); return
        self.curve_window = tk.Toplevel(self.root); self.curve_window.title(self.tr('保存响应 · 选中对象'))
        width = min(round(980*self.display_scale), self.root.winfo_screenwidth()-round(60*self.display_scale))
        height = min(round(460*self.display_scale), self.root.winfo_screenheight()-round(90*self.display_scale))
        self.curve_window.geometry(f'{width}x{height}')
        ttk.Label(self.curve_window, text='变量在主窗口左侧选择；稀疏检查点只显示散点。').pack(anchor='w', padx=12, pady=6)
        self.curve_canvas = tk.Canvas(self.curve_window, bg=BG, highlightthickness=0); self.curve_canvas.pack(fill='both', expand=True)
        self.curve_canvas.bind('<Configure>', lambda _: self.render_curve())
        ttk.Button(self.curve_window, text='导出此曲线…', command=lambda: self.export(curve=True)).pack(pady=5)
        self.register_labels(self.curve_window)
        self.render_curve()

    def render_curve(self):
        if not self.data or not self.curve_window or not self.curve_window.winfo_exists() or not self.variable.get(): return
        try:
            start, stop = self.bounds(); pop, row = self.cell; key = self.variable.get()
            series = self.data.series(pop, row, key, start, stop)
            size = (self.curve_canvas.winfo_width(), self.curve_canvas.winfo_height())
            if min(size) <= 1: return
            s = self.display_scale
            image = trace_panel(size, [series], start, stop, self.clock.position,
                f"{self.data.geometry[pop]['subtypes'][row]} · {key}", self.unit_label(series['unit']), translate=self.tr, pixel_scale=s,
                subtitle=f"{self.data.geometry[pop]['ids'][row]} · {series.get('status', '')}")
            self.put(self.curve_canvas, image)
        except (ValueError, KeyError) as error: self.set_status(str(error))

    def seek(self, seconds):
        if not np.isfinite(seconds): raise ValueError('时间必须是有限数')
        if self.playback_window and not self.playback_window[0] <= seconds <= self.playback_window[1]: self.playback_window = None
        self.clock.pause(); self.clock.seek(seconds); self.jump.set(f'{self.clock.position:.5f}'); self.render()

    def jump_to(self):
        try: self.seek(float(self.jump.get()))
        except ValueError: self.set_status('请输入有效的秒数。')

    def slider_changed(self, value):
        if abs(float(value)-self.clock.position) > 1e-8: self.seek(float(value))

    def toggle_play(self):
        if not self.data: return
        if not self.clock.playing and self.playback_window and self.clock.position >= self.playback_window[1]:
            self.clock.seek(self.playback_window[0])
        self.clock.pause() if self.clock.playing else self.clock.play(); self.render()

    def _tick(self):
        if self.closed: return
        try:
            while True:
                kind, value = self.queue.get_nowait(); self.busy = False
                if kind == 'loaded': self.loaded(value)
                else: self.set_status('加载失败：'+value)
        except queue.Empty: pass
        if self.clock.playing:
            self.clock.advance()
            if self.playback_window and self.clock.position >= self.playback_window[1]:
                self.clock.pause(); self.clock.seek(self.playback_window[1])
            self.render()
        self.timer = self.root.after(80, self._tick)

    def choose_directory(self):
        path = filedialog.askdirectory(parent=self.root, initialdir=self.project_root/'reports', title=self.tr('选择已有研究结果目录'))
        if path: self.load(path)

    def legacy(self):
        from .response_view import open_response_window
        open_response_window(self.root, self.project_root)

    def show_info(self):
        if not self.data: return
        window = tk.Toplevel(self.root); window.title(self.tr('数据身份与覆盖')); window.geometry('920x620')
        text = tk.Text(window, wrap='word', font=('Microsoft YaHei UI', 10)); text.pack(fill='both', expand=True)
        info = {'model_identity': self.data.identity, 'source': str(self.data.source),
                'snapshot_times_seconds': list(self.data.snapshot_times), 'variables': {p: self.data.variables(p) for p in self.data.populations},
                'continuous_window': getattr(self.data, 'continuous_window', None),
                'continuous_manifest': str(self.data.source/'internal_states/manifest.json') if getattr(self.data, 'continuous_window', None) is not None else None,
                'limitations': getattr(self.data, 'limitations', [])}
        raw = json.dumps(info, ensure_ascii=False, indent=2, default=str)
        text.insert('1.0', self.tr(raw)); text.configure(state='disabled'); self.info_windows.append((window, text, raw))

    def export(self, curve=False):
        if not self.data: return
        index = self.notebook.index(self.notebook.select())
        folder = self.project_root/'semireport'/'2026-11-06-内部状态可视化'
        folder.mkdir(parents=True, exist_ok=True)
        name = self.tr('对象响应' if curve else TABS[index])
        path = filedialog.asksaveasfilename(parent=self.root, title=self.tr('导出当前图…'), initialdir=folder, initialfile=name+'.png', defaultextension='.png', filetypes=[(self.tr('PNG 图像'), '*.png')])
        if path: self.save_figure(Path(path), index, curve)

    def save_figure(self, path, index=None, curve=False):
        index = self.notebook.index(self.notebook.select()) if index is None else index
        stimulus_metadata = None
        if curve:
            start, stop = self.bounds(); pop, row = self.cell; key = self.variable.get()
            series = self.data.series(pop, row, key, start, stop)
            image = trace_panel((1800, 800), [series], start, stop, self.clock.position,
                f"{self.data.geometry[pop]['ids'][row]} · {key}", series['unit'], translate=self.tr)
        elif index == 0:
            image = empty_plot((1800, 1280), '')
            frame = self.data.stimulus(self.clock.position)
            draw = ImageDraw.Draw(image)
            draw.text((20, 12), self.tr('输入刺激'), font=font(15), fill='#173745')
            if frame.get('image') is not None:
                preview = Image.fromarray(frame['image'])
                original_size = preview.size
                preview.thumbnail((130, 130), Image.Resampling.LANCZOS)
                image.paste(preview, (20, 40))
                stimulus_metadata = dict(time_seconds=frame['time'], source_frame_zero_based=frame['source_frame'],
                    source_resolution_pixels=list(original_size))
                caption = (f"{self.tr('原刺激帧 ')}{frame['time']:.5f} s\n"
                    f"{self.tr('帧索引（从 0 开始）')}: {frame['source_frame']} · "
                    f"{self.tr('原始分辨率')}: {original_size[0]} × {original_size[1]} px")
            else: caption = self.tr(frame.get('status', '原刺激帧文件不可用'))
            draw.text((180, 55), caption, font=font(13), fill='#475569', spacing=9)
            for i, (_, pop, _, _) in enumerate(self.cards):
                width = 1200 if i == 3 else 600
                x = 1200 if i == 4 else (i%3)*600; y = 180+(i//3)*550
                panel, _, text = self.layer_image(pop.get(), (width, 510))
                image.paste(panel, (x, y))
                ImageDraw.Draw(image).text((x+15, y+510), text, font=font(9), fill='#64748b')
        else: image = self.figure(index, (1800, 1200))
        info = PngImagePlugin.PngInfo()
        info.add_text('retina_provenance', json.dumps({'model_identity': self.data.identity, 'result': str(self.data.source),
            'time_seconds': self.clock.position, 'window_seconds': self.bounds(), 'selected_object': self.cell,
            'reference_objects': self.figure_references if index == 3 and not curve else self.references, 'h1_row': self.h1.get(),
            'ui_language': self.language_code,
            'ac_filter': self.ac_filter.get(),
            'ac_scope': 'current selected RGC saved connection ancestors',
            'stimulus': stimulus_metadata,
            'layer_viewports': {key: [low.tolist(), high.tolist()] for key, (low, high) in self.viewports.items()},
            'continuous_window': getattr(self.data, 'continuous_window', None),
            'continuous_manifest': str(self.data.source/'internal_states/manifest.json') if getattr(self.data, 'continuous_window', None) is not None else None,
            'h1_metadata': self.h1_cache.get(int(self.h1.get()), {}).get('metadata') if index == 2 else None,
            'display': 'actual saved samples; snapshots unconnected; no model execution'}, ensure_ascii=False))
        image.save(path, pnginfo=info); self.set_status('已导出：'+str(path)); return path

    def close(self):
        self.closed = True; self.clock.pause()
        if self.timer: self.root.after_cancel(self.timer)
        if self.data: self.data.close()
        self.root.destroy()


def open_internal_state_window(root, project_root):
    window = tk.Toplevel(root)
    return InternalStateApp(window, project_root)
