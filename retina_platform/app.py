from __future__ import annotations

import json
from pathlib import Path
import queue
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageGrab, ImageTk

from .backend import SimulationSession
from .config import RunConfig
from .media import MediaSource
from .results import ResultStore
from .plotting import curve_plot, empty_plot, scatter_plot
from retina_sim.regional_support import load_regional_support


class RetinaApp:
    """Tk widgets are updated on the main thread; simulation owns its state."""

    PARAMS = [
        ("execution_profile", "运行模式", str),
        ("roi_preset", "目标位置", str),
        ("hc_geometry_profile", "HC 空间模式", str),
        ("hc_arbor_profile", "HC arbor 来源", str),
        ("ipl_geometry_profile", "IPL 几何", str),
        ("requested_owner_output", "owner 输出", str),
        ("hc_h1_radius_mm", "H1 arbor / mm", float),
        ("hc_h2_radius_mm", "H2 arbor / mm", float),
        ("rgb_adapter", "输入 adapter", str),
        ("reference_linear_value", "reference u / linear", float),
        ("owner_x_mm", "ROI 中心 x / mm", float),
        ("owner_y_mm", "ROI 中心 y / mm", float),
        ("owner_width_mm", "ROI 宽 / mm", float),
        ("owner_height_mm", "ROI 高 / mm", float),
        ("halo_mm", "halo / mm", float),
        ("start_seconds", "开始 / s", float),
        ("duration_seconds", "记录 / s", float),
        ("warmup_seconds", "背景预运行 / s", float),
        ("background_initialization_mode", "背景初始化", str),
        ("background_max_sim_seconds", "适应上限 / s", float),
        ("window_seconds", "窗口 / s", float),
        ("sample_seconds", "采样间隔 / s", float),
        ("effective_cones", "preview cone 数", int),
        ("effective_rods", "preview rod 数", int),
        ("field_width_deg", "视场宽 / °", float),
        ("native_background_td", "native 背景 / td", float),
        ("cone_background_rstar_s", "cone 背景 / R*/s", float),
        ("rod_background_rstar_s", "rod 背景 / R*/s", float),
        ("gaze_x_deg", "固定眼位 x / °", float),
        ("gaze_y_deg", "固定眼位 y / °", float),
        ("seed", "种子", int),
    ]

    def __init__(self, root, project_root):
        self.root = root
        self.project_root = Path(project_root).resolve()
        self.regional_support=load_regional_support(self.project_root)
        self.roi_labels={key:row['label'] for key,row in self.regional_support['roi_presets'].items()}
        self.root.title("Macaque Retina · 本地模拟器")
        width = min(1400, root.winfo_screenwidth() - 50)
        height = min(900, root.winfo_screenheight() - 90)
        root.geometry(f"{width}x{height}")
        root.minsize(1080, 680)
        self.config = RunConfig.project_default(self.project_root)
        self.media = None
        self.store = None
        self.session = None
        self.state = "idle"
        self.atlas_variant=tk.StringVar(value='严格来源')
        self.params = {}
        self.entries = {}
        self.buttons = {}
        self.status_var = tk.StringVar(value="就绪")
        self.media_var = tk.StringVar(value="未导入媒体")
        self.time_var = tk.StringVar(value="0 s")
        self.layer_var = tk.StringVar(value="全部层")
        self.port_var = tk.StringVar()
        self.type_var = tk.StringVar(value="全部类型")
        self.coordinate_var = tk.StringVar(value="RF")
        self.scope_var = tk.StringVar(value="全部")
        self.window_var = tk.IntVar(value=0)
        self.live_var = tk.BooleanVar(value=True)
        self.range_var = tk.StringVar(value="自动")
        self.min_var, self.max_var = tk.StringVar(value="0"), tk.StringVar(value="1")
        self.preview_var = tk.DoubleVar(value=0)
        self.cell_var = tk.StringVar(value="未选择细胞")
        self.selected_cell = None
        self.port_records = {}
        self.pointmap = []
        self._photos = {}
        self._images = {}
        self._window_cache = None
        self._current_index = None
        self._closing = False
        self._render_pending = False
        self.event_log = []
        self.last_event = None
        self._build_ui()
        self._set_config_vars(self.config)
        self.variables = self.params
        self.buttons["start"] = self.buttons["run"]
        self.buttons["export_plot"] = self.buttons["export_png"]
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(120, self._poll_events)

    @property
    def result_store(self):
        return self.store

    @property
    def status(self):
        return self.state

    def _button(self, parent, key, text, command):
        button = ttk.Button(parent, text=text, command=command)
        self.buttons[key] = button
        return button

    def _build_ui(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10))
        style.configure("TFrame", background="#f1f5f9")
        style.configure("TLabel", background="#f1f5f9", foreground="#243447")
        style.configure("TButton", padding=(9, 5))
        self.root.configure(background="#f1f5f9")
        shell = ttk.Frame(self.root, padding=10)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(0, weight=1)
        left = ttk.Frame(shell, width=300)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(5, weight=1)
        files = ttk.Frame(left)
        files.grid(row=0, column=0, sticky="ew")
        self._button(files, "import", "导入图像 / 视频", self._choose_media).pack(side="left")
        self._button(files, "load_results", "载入结果", self._choose_results).pack(side="right")
        ttk.Label(left, textvariable=self.media_var, wraplength=292).grid(row=1, column=0, sticky="ew", pady=(7, 4))
        self.preview_canvas = tk.Canvas(left, height=153, bg="#18212c", highlightthickness=0)
        self.preview_canvas.grid(row=2, column=0, sticky="ew")
        self.preview_canvas.bind("<Configure>", lambda _: self._render_preview())
        preview_row = ttk.Frame(left)
        preview_row.grid(row=3, column=0, sticky="ew", pady=3)
        preview_row.columnconfigure(0, weight=1)
        self.preview_scale = ttk.Scale(preview_row, from_=0, to=1, variable=self.preview_var, command=self._preview_changed)
        self.preview_scale.grid(row=0, column=0, sticky="ew")
        self.preview_time = ttk.Label(preview_row, text="0 s", width=9)
        self.preview_time.grid(row=0, column=1)
        ttk.Label(left, text="工程 RGB 输入 · 来源约束核心", foreground="#167e9a").grid(row=4, column=0, sticky="w", pady=(5, 5))
        parameter_box = ttk.Frame(left)
        parameter_box.grid(row=5, column=0, sticky="nsew")
        parameter_canvas = tk.Canvas(parameter_box, highlightthickness=0, background="#f1f5f9", width=275)
        parameter_canvas.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(parameter_box, command=parameter_canvas.yview)
        scroll.pack(side="right", fill="y")
        parameter_canvas.configure(yscrollcommand=scroll.set)
        entries = ttk.Frame(parameter_canvas)
        window = parameter_canvas.create_window((0, 0), window=entries, anchor="nw")
        entries.bind("<Configure>", lambda _: parameter_canvas.configure(scrollregion=parameter_canvas.bbox("all")))
        parameter_canvas.bind("<Configure>", lambda e: parameter_canvas.itemconfigure(window, width=e.width))
        entries.columnconfigure(1, weight=1)
        for row, (name, label, _) in enumerate(self.PARAMS):
            var = tk.StringVar()
            self.params[name] = var
            ttk.Label(entries, text=label).grid(row=row, column=0, sticky="w", pady=4)
            choices={'execution_profile':['来源人口修订'],
                'hc_geometry_profile':['联合HC合成'],
                'hc_arbor_profile':['W2000_equal_area_bounded_edges_047_104'],
                'ipl_geometry_profile':['retina_contact_geometry_effective_v1'],
                'requested_owner_output':['midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON'],
                'background_initialization_mode':['fixed_prerun','bounded_convergence'],
                'rgb_adapter':['contrast_reference_v1','legacy_rgb_proxy_v1'],
                'roi_preset':list(self.roi_labels.values())}
            entry = (ttk.Combobox(entries,textvariable=var,values=choices[name],state='readonly',width=12)
                if name in choices else ttk.Entry(entries, textvariable=var, width=12))
            entry.grid(row=row, column=1, sticky="ew", padx=(6, 2), pady=4)
            self.entries[name] = entry
            if name=='roi_preset':entry.bind('<<ComboboxSelected>>',self._choose_roi_preset)
            if name=='hc_geometry_profile':entry.bind('<<ComboboxSelected>>',self._choose_hc_geometry_profile)
            if name=='hc_arbor_profile':entry.bind('<<ComboboxSelected>>',self._choose_hc_arbor_profile)
            if name=='execution_profile':entry.bind('<<ComboboxSelected>>',self._choose_execution_profile)
        row = len(self.PARAMS)
        self.params["media_path"] = tk.StringVar()
        self.params["gaze_csv"] = tk.StringVar()
        self.params['hc_geometry_path']=tk.StringVar()
        ttk.Label(entries, text="眼位轨迹 CSV").grid(row=row, column=0, sticky="w", pady=5)
        self._button(entries, "gaze_csv", "选择 / 清除", self._choose_gaze).grid(row=row, column=1, sticky="ew")
        self.gaze_label = ttk.Label(entries, text="固定眼位", wraplength=270)
        self.gaze_label.grid(row=row+1, column=0, columnspan=2, sticky="w")
        self._button(entries,'hc_geometry_json','HC 几何 JSON',self._choose_hc_geometry_json).grid(row=row+2,column=0,columnspan=2,sticky='ew')
        self._button(entries,'atlas_preview','来源 atlas 几何预览',self.preview_atlas).grid(row=row+3,column=0,columnspan=2,sticky='ew')
        self._button(entries,'atlas_connections','atlas 连接预览',self.preview_atlas_connections).grid(row=row+4,column=0,columnspan=2,sticky='ew')
        settings = ttk.Frame(left)
        settings.grid(row=6, column=0, sticky="ew", pady=(7, 0))
        self._button(settings, "save_config", "保存配置", self._choose_save_config).pack(side="left")
        self._button(settings, "load_config", "载入配置", self._choose_load_config).pack(side="right")

        right = ttk.Frame(shell)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=3)
        right.rowconfigure(4, weight=2)
        selectors = ttk.Frame(right)
        selectors.grid(row=0, column=0, sticky="ew")
        self.layer_combo = ttk.Combobox(selectors, textvariable=self.layer_var, values=["全部层"], state="readonly", width=16)
        self.layer_combo.pack(side="left", padx=(0, 6))
        self.layer_combo.bind("<<ComboboxSelected>>", lambda _: self._update_ports())
        self.port_combo = ttk.Combobox(selectors, textvariable=self.port_var, state="readonly", width=45)
        self.port_combo.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.port_combo.bind("<<ComboboxSelected>>", lambda _: self._port_changed())
        self.type_combo = ttk.Combobox(selectors, textvariable=self.type_var, values=["全部类型"], state="readonly", width=15)
        self.type_combo.pack(side="left")
        self.type_combo.bind("<<ComboboxSelected>>", lambda _: self._schedule_render())
        view = ttk.Frame(right)
        view.grid(row=1, column=0, sticky="ew", pady=6)
        ttk.Label(view, text="坐标").pack(side="left")
        self.coordinate_combo = ttk.Combobox(view, textvariable=self.coordinate_var, values=[], state="readonly", width=7)
        self.coordinate_combo.pack(side="left", padx=5)
        self.coordinate_combo.bind("<<ComboboxSelected>>", lambda _: self._schedule_render())
        scope_combo=ttk.Combobox(view,textvariable=self.scope_var,values=['全部','owner','halo'],state='readonly',width=6)
        scope_combo.pack(side='left',padx=4)
        scope_combo.bind('<<ComboboxSelected>>',lambda _:self._port_changed())
        ttk.Label(view, text="色标").pack(side="left", padx=(12, 0))
        range_combo = ttk.Combobox(view, textvariable=self.range_var, values=["自动", "固定"], state="readonly", width=6)
        range_combo.pack(side="left", padx=5)
        range_combo.bind("<<ComboboxSelected>>", lambda _: self._schedule_render())
        for var in (self.min_var, self.max_var):
            entry = ttk.Entry(view, textvariable=var, width=8)
            entry.pack(side="left", padx=2)
            entry.bind("<Return>", lambda _: self._schedule_render())
        ttk.Label(view, text="窗口").pack(side="left", padx=(12, 4))
        self.window_spin = ttk.Spinbox(view, from_=0, to=0, textvariable=self.window_var, width=5, command=self._window_selected)
        self.window_spin.pack(side="left")
        self.window_spin.bind("<Return>", lambda _: self._window_selected())
        ttk.Checkbutton(view, text="跟随", variable=self.live_var, command=self._follow_changed).pack(side="left", padx=5)
        self.scatter_canvas = tk.Canvas(right, highlightthickness=0, background="#f8fafc")
        self.scatter_canvas.grid(row=2, column=0, sticky="nsew")
        self.scatter_canvas.bind("<Configure>", lambda _: self._schedule_render())
        self.scatter_canvas.bind("<Button-1>", self._select_cell)
        ttk.Label(right, textvariable=self.cell_var).grid(row=3, column=0, sticky="w", pady=(7, 4))
        self.curve_canvas = tk.Canvas(right, highlightthickness=0, background="#f8fafc")
        self.curve_canvas.grid(row=4, column=0, sticky="nsew")
        self.curve_canvas.bind("<Configure>", lambda _: self._schedule_render())
        exports = ttk.Frame(right)
        exports.grid(row=5, column=0, sticky="ew", pady=(7, 0))
        for key, text, handler in (("export_bundle", "导出结果包", self._choose_export_bundle), ("export_curve", "导出所选曲线", self._choose_export_curve), ("export_png", "导出当前图", self._choose_export_png)):
            self._button(exports, key, text, handler).pack(side="left", padx=(0, 6))
        controls = ttk.Frame(shell)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self._button(controls, "cell6_replay", "Cell6 预计算回放", self.open_cell6_replay).pack(side="left", padx=(0, 7))
        for key, text, handler in (("run", "运行", self.start_run), ("pause", "暂停", self.pause), ("resume", "继续", self.resume), ("stop", "停止", self.stop)):
            self._button(controls, key, text, handler).pack(side="left", padx=(0, 7))
        self.progress = ttk.Progressbar(controls, mode="determinate", maximum=1)
        self.progress.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Label(controls, textvariable=self.time_var, width=20).pack(side="right")
        ttk.Label(shell, textvariable=self.status_var, wraplength=1200).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._update_buttons()

    def open_cell6_replay(self):
        from .replay import ReplayApp
        window = tk.Toplevel(self.root)
        ReplayApp(window, self.project_root)

    def _error(self, error):
        self.status_var.set(f"错误：{error}")

    def _set_config_vars(self, config):
        for name, var in self.params.items():
            value=getattr(config,name,"")
            var.set('' if value is None else str(value))
        self.params['execution_profile'].set({'retina_contact_geometry_effective_v1':'研究原型','atlas_midget_research_v1':'atlas midget研究','atlas_population_revision_v1':'来源人口修订','source_constrained_approximation':'源约束混合'}.get(config.execution_profile,'工程预览'))
        self.params['roi_preset'].set(self.roi_labels[config.roi_preset])
        self.params['hc_geometry_profile'].set({'hc_joint_synthetic_v1':'联合HC合成','atlas_fixed_joint_master_v2':'固定atlas HC'}.get(config.hc_geometry_profile,'单动物HC patch'))
        self.gaze_label.configure(text=Path(config.gaze_csv).name if config.gaze_csv else "固定眼位")
        self._atlas_runtime_controls()

    def _read_config(self):
        data = self.config.to_dict()
        for name, _, kind in self.PARAMS:
            text = self.params[name].get().strip()
            try:
                data[name] = None if name in ('hc_h1_radius_mm','hc_h2_radius_mm') and not text else kind(text)
                if name=='execution_profile':data[name]={'研究原型':'retina_contact_geometry_effective_v1','atlas midget研究':'atlas_midget_research_v1','来源人口修订':'atlas_population_revision_v1','源约束混合':'source_constrained_approximation','工程预览':'engineering_preview'}.get(text,text)
                if name=='roi_preset':data[name]={value:key for key,value in self.roi_labels.items()}.get(text,text)
                if name=='hc_geometry_profile':data[name]={'单动物HC patch':'source_patch_v1','联合HC合成':'hc_joint_synthetic_v1','联合HC合成·未跑':'hc_joint_synthetic_v1','固定atlas HC':'atlas_fixed_joint_master_v2'}.get(text,text)
            except ValueError:
                raise ValueError(f"{name} 参数格式错误：{text}") from None
        data["media_path"] = self.params["media_path"].get()
        data["gaze_csv"] = self.params["gaze_csv"].get()
        data['hc_geometry_path']=self.params['hc_geometry_path'].get()
        preset=self.regional_support['roi_presets'].get(data['roi_preset'],{})
        if 'center_mm' in preset:
            requested=[data[k] for k in ('owner_x_mm','owner_y_mm','owner_width_mm','owner_height_mm','halo_mm')]
            named=preset['center_mm']+preset['size_mm']+[preset['halo_mm']]
            if requested!=named:
                data['roi_preset']='custom'
                self.params['roi_preset'].set(self.roi_labels['custom'])
        config = RunConfig.from_dict(data)
        config.validate(self.media)
        return config

    def _choose_roi_preset(self,event=None):
        label=self.params['roi_preset'].get()
        key=next(k for k,v in self.roi_labels.items() if v==label)
        preset=self.regional_support['roi_presets'][key]
        if key=='custom':
            self.status_var.set('自定二维ROI · 未运行');return
        x,y=preset['center_mm'];width,height=preset['size_mm']
        for name,value in [('owner_x_mm',x),('owner_y_mm',y),('owner_width_mm',width),('owner_height_mm',height),('halo_mm',preset['halo_mm'])]:
            self.params[name].set(str(value))
        message='HC场待补' if key in ('center_design','central_author_anchor') else '区域动态仍为共享近似'
        self.status_var.set(label+' · '+message+' · 未运行')

    def _choose_hc_geometry_json(self):
        if self.session is not None and self.session.is_alive():
            self._error('运行中不能替换HC几何配置');return
        path=filedialog.askopenfilename(title='有限HC footprint / 形态 / canonical配置',filetypes=[('JSON','*.json')])
        if path:
            self.params['hc_geometry_path'].set(path)
            self.params['hc_geometry_profile'].set('联合HC合成')
            self.params['hc_arbor_profile'].set('explicit_footprints')
            self.status_var.set('已选HC几何 · 未运行')

    def preview_atlas(self):
        self.preview_atlas_connections()

    def _choose_hc_geometry_profile(self,event=None):
        if self.params['hc_geometry_profile'].get()=='单动物HC patch':
            self.params['hc_arbor_profile'].set('explicit_footprints')
            self.params['hc_geometry_path'].set('')
            self.params['hc_h1_radius_mm'].set('');self.params['hc_h2_radius_mm'].set('')
            self.status_var.set('单动物HC patch')
        else:
            self.status_var.set('联合HC合成 · 需显式arbor · 未运行')

    def preview_atlas_connections(self):
        try:
            from datetime import datetime,timezone
            from .population_preview import export_population_preview
            x,y,w,h,halo=[float(self.params[k].get()) for k in
                ('owner_x_mm','owner_y_mm','owner_width_mm','owner_height_mm','halo_mm')]
            path=self.project_root/'reports'/'population_preview'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            result=export_population_preview(self.project_root,path,(x-w/2,x+w/2,y-h/2,y+h/2),
                halo_mm=halo,max_cells=int(self.params['max_spatial_cells'].get()))
            window=tk.Toplevel(self.root);window.title('当前人口 / 连接')
            photo=ImageTk.PhotoImage(Image.open(result['image']))
            label=ttk.Label(window,image=photo);label.image=photo;label.pack()
            self.status_var.set('已显示当前冻结图 · 网络状态未更新')
        except Exception as error:self._error(error)

    def _atlas_options(self):
        target=json.loads((self.project_root/'configurations/retina_platform_v1.json').read_bytes())['atlas_target']
        return {key:target[key] for key in ('atlas_profile_id','s_partition_profile')}

    def _choose_execution_profile(self,event=None):
        if self.params['execution_profile'].get()=='atlas midget研究':
            preset=RunConfig.from_dict({'execution_profile':'atlas_midget_research_v1'})
            for name in ('duration_seconds','warmup_seconds','window_seconds','sample_seconds','owner_x_mm','owner_y_mm',
                'owner_width_mm','owner_height_mm','halo_mm','max_spatial_cells','hc_arbor_profile','ipl_geometry_profile','hc_geometry_path'):
                self.params[name].set(str(getattr(preset,name)))
            self.params['hc_geometry_profile'].set('固定atlas HC')
            self.params['hc_h1_radius_mm'].set('');self.params['hc_h2_radius_mm'].set('')
            self.params['background_initialization_mode'].set('fixed_prerun')
            self.params['requested_owner_output'].set('midget_RGC_ON')
            self.params['roi_preset'].set(self.roi_labels['custom'])
            self.atlas_variant.set('Schein桥接 · mulatta有效')
            self.config.parameter_profile='effective_dynamics_v1'
            self.config.adaptation_policy='uniform_declared_background'
            self.status_var.set('atlas midget · 局部研究 · rod/AII未接入')
            self._atlas_runtime_controls();return
        research=self.params['execution_profile'].get()=='研究原型'
        engineering=self.params['execution_profile'].get()=='工程预览'
        self.params['ipl_geometry_profile'].set('retina_contact_geometry_effective_v1' if research else 'legacy_point_ipl_v1')
        if research:
            self.params['hc_h1_radius_mm'].set('');self.params['hc_h2_radius_mm'].set('')
            for name,value in [('hc_geometry_profile','联合HC合成'),('hc_arbor_profile','W2000_equal_area_bounded_edges_047_104'),
                ('hc_geometry_path','configurations/hc_research_prototype_geometry_v1.json'),('owner_x_mm',.35),('owner_y_mm',0.),
                ('owner_width_mm',.02),('owner_height_mm',.02),('halo_mm',.04),('max_spatial_cells',60000)]:self.params[name].set(str(value))
            self.params['roi_preset'].set(self.roi_labels['custom'])
            self.status_var.set('研究原型 · 动力学未标定')
        else:
            self.params['hc_geometry_profile'].set('单动物HC patch');self.params['hc_arbor_profile'].set('explicit_footprints')
            self.params['hc_geometry_path'].set('')
            if not engineering:
                self.params['roi_preset'].set(self.roi_labels['peripheral_anchor']);self._choose_roi_preset()
        self._atlas_runtime_controls()

    def _atlas_runtime_controls(self):
        atlas=self.params['execution_profile'].get() in ('atlas midget研究','来源人口修订')
        for name in ('execution_profile','hc_geometry_profile','hc_arbor_profile','ipl_geometry_profile','hc_h1_radius_mm','hc_h2_radius_mm'):
            if name in self.entries:self.entries[name].configure(state='disabled')
        for name in (
                     'effective_cones','effective_rods','background_initialization_mode','warmup_seconds'):
            if name in self.entries:
                entry=self.entries[name]
                entry.configure(state='disabled' if atlas else 'readonly' if isinstance(entry,ttk.Combobox) else 'normal')
        self.buttons['hc_geometry_json'].configure(state='disabled')

    def _choose_hc_arbor_profile(self,event=None):
        if self.params['hc_arbor_profile'].get()=='W2000_equal_area_bounded_edges_047_104':
            self.params['execution_profile'].set('研究原型');self._choose_execution_profile();return
        if self.params['hc_arbor_profile'].get()=='central_OPL_equal_area_W2000':
            self.params['hc_geometry_profile'].set('联合HC合成')
            self.params['execution_profile'].set('源约束混合')
            self.params['ipl_geometry_profile'].set('p_chart_typed_effective_ipl_v1')
            self.params['hc_geometry_path'].set('')
            self.params['hc_h1_radius_mm'].set('');self.params['hc_h2_radius_mm'].set('')
            self.params['roi_preset'].set(self.roi_labels['central_OPL_support'])
            self._choose_roi_preset()
            self.status_var.set('中央P-chart有效几何 · 未运行')

    def import_media(self, path):
        if self.session is not None and self.session.is_alive():
            self._error("运行中不能替换媒体")
            return False
        candidate = None
        try:
            candidate = MediaSource(path)
            candidate.frame_at(0)
            if self.media is not None:
                self.media.close()
            self.media = candidate
            self.params["media_path"].set(str(Path(path).resolve()))
            info = self.media.info
            suffix = f" · {info['fps']:g} FPS · {info['duration_seconds']:.3f} s" if info['kind'] == "video" else " · 图像"
            self.media_var.set(f"{Path(info['path']).name}\n{info['width']} × {info['height']}{suffix}")
            self.preview_scale.configure(to=max(info['duration_seconds'] or 0, 0.001), state="normal" if info['kind'] == "video" else "disabled")
            self.preview_var.set(0)
            self._render_preview()
            self.status_var.set("媒体已导入")
            if self.state == "error":
                self.state = "idle"
            self._update_buttons()
            return True
        except Exception as exc:
            if candidate is not None and candidate is not self.media:
                candidate.close()
            self._error(exc)
            return False

    def _choose_media(self):
        path = filedialog.askopenfilename(title="导入媒体", filetypes=[("图像 / 视频", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.mp4 *.avi *.mov *.mkv *.webm"), ("所有文件", "*.*")])
        if path:
            self.import_media(path)

    def _preview_changed(self, _=None):
        self.preview_time.configure(text=f"{self.preview_var.get():.3f} s")
        self._render_preview()

    def _render_preview(self):
        if self.media is None:
            return
        try:
            seconds = float(self.preview_var.get())
            if self.media.info['kind'] == "video":
                seconds = min(seconds, max(0, self.media.info['duration_seconds'] - 1 / self.media.info['fps']))
            frame = self.media.frame_at(seconds)
            width = max(self.preview_canvas.winfo_width(), 10)
            height = max(self.preview_canvas.winfo_height(), 10)
            im = Image.fromarray(frame)
            im.thumbnail((width, height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(im)
            self._photos["preview"] = photo
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(width/2, height/2, image=photo)
        except Exception as exc:
            self._error(exc)

    def _choose_gaze(self):
        if self.params["gaze_csv"].get():
            self.params["gaze_csv"].set("")
            self.gaze_label.configure(text="固定眼位")
            return
        path = filedialog.askopenfilename(title="眼位 CSV", filetypes=[("CSV", "*.csv")])
        if path:
            self.params["gaze_csv"].set(path)
            self.gaze_label.configure(text=Path(path).name)

    def save_config(self, path):
        try:
            config = self._read_config()
            Path(path).write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            self.config = config
            self.status_var.set(f"配置已保存：{path}")
            return True
        except Exception as exc:
            self._error(exc)
            return False

    def load_config(self, path):
        if self.session is not None and self.session.is_alive():
            self._error("运行中不能替换配置")
            return False
        try:
            candidate = RunConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))
            if candidate.execution_profile != RunConfig.project_default(self.project_root).execution_profile:
                raise ValueError('配置不属于当前冻结模型；历史运行可通过载入结果查看')
            candidate.validate()
            if candidate.media_path and not self.import_media(candidate.media_path):
                return False
            self.config = candidate
            self._set_config_vars(candidate)
            self.status_var.set(f"配置已载入：{Path(path).name}")
            if self.state == "error":
                self.state = "idle"
            self._update_buttons()
            return True
        except Exception as exc:
            self._error(exc)
            return False

    def _choose_save_config(self):
        path = filedialog.asksaveasfilename(title="保存配置", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            self.save_config(path)

    def _choose_load_config(self):
        path = filedialog.askopenfilename(title="载入配置", filetypes=[("JSON", "*.json")])
        if path:
            self.load_config(path)

    def start_run(self):
        if self.session is not None and self.session.is_alive():
            self._error("已有运行任务")
            return False
        try:
            config = self._read_config()
            if self.media is None:
                raise ValueError("请先导入图像或视频")
            self.session = SimulationSession(config, self.project_root)
            self.config = config
            self.store = None
            self._window_cache = None
            self._current_index = None
            self.selected_cell = None
            self.port_records = {}
            self.port_combo.configure(values=[])
            self.port_var.set("")
            self.cell_var.set("未选择细胞")
            self._schedule_render()
            self.state = "building"
            self.progress["value"] = 0
            self.time_var.set("0 s")
            self.status_var.set("构建中")
            self.session.start()
            self._update_buttons()
            return True
        except Exception as exc:
            self.state = "error"
            self._error(exc)
            self._update_buttons()
            return False

    def pause(self):
        if self.session is not None and self.session.is_alive():
            self.session.pause()
            self.state = "pause_requested"
            self.status_var.set("等待当前窗口完成后暂停")
            self._update_buttons()

    def resume(self):
        if self.session is not None and self.session.is_alive():
            self.session.resume()
            self.state = "running"
            self.status_var.set("运行中")
            self._update_buttons()

    def stop(self):
        if self.session is not None and self.session.is_alive():
            self.session.stop()
            self.state = "stopping"
            self.status_var.set("停止中 · 保留已完成窗口")
            self._update_buttons()

    def _poll_events(self):
        if self.session is not None:
            while True:
                try:
                    event = self.session.events.get_nowait()
                except queue.Empty:
                    break
                self._handle_event(event)
        self._update_buttons()
        if self._closing:
            if self.session is None or not self.session.is_alive():
                if self.media is not None:
                    self.media.close()
                self.root.destroy()
                return
        self.root.after(120, self._poll_events)

    def _handle_event(self, event):
        self.last_event = dict(event)
        self.event_log.append(dict(event))
        del self.event_log[:-100]
        names = {"building": "构建中", "running": "运行中", "pause_requested": "等待窗口结束", "paused": "已暂停", "stopping": "停止中", "stopped": "已停止", "completed": "已完成", "initialization_not_reached":"适应未达到", "error": "错误"}
        status = event.get("status")
        if status:
            self.state = status
            self.status_var.set(names.get(status, status))
        if event.get("kind") == "error":
            self.state = "error"
            self._error(event.get("message", "后台运行失败"))
        if "time" in event:
            t = float(event["time"])
            total = float(event.get("total", self.config.warmup_seconds + self.config.duration_seconds))
            elapsed = float(event.get("elapsed", max(0, t)))
            self.time_var.set(f"{t:.5g} / {total:.5g} s")
            self.progress["value"] = min(1, max(0, elapsed/max(total, 1e-30)))
        if event.get("path") and event.get("kind") in ("window", "finished"):
            self.load_results(event["path"], live=True)
        if self.media is not None and "stimulus_time" in event:
            self.preview_var.set(max(0, float(event["stimulus_time"])))
            self.preview_time.configure(text=f"{self.preview_var.get():.3f} s")
            self._render_preview()
        if event.get("message") and event.get("kind") != "error":
            self.status_var.set(names.get(status, "") + " · " + str(event["message"]))
        self._update_buttons()

    def _update_buttons(self):
        active = self.session is not None and self.session.is_alive()
        for name in ("run", "import", "load_config", "load_results", "gaze_csv", "hc_geometry_json", "atlas_preview", "atlas_connections"):
            self.buttons[name].configure(state="disabled" if active else "normal")
        for entry in self.entries.values():
            entry.configure(state="disabled" if active else "readonly" if isinstance(entry,ttk.Combobox) else "normal")
        if not active:self._atlas_runtime_controls()
        self.buttons["pause"].configure(state="normal" if active and self.state in ("running", "building") else "disabled")
        self.buttons["resume"].configure(state="normal" if active and self.state in ("paused", "pause_requested") else "disabled")
        self.buttons["stop"].configure(state="normal" if active and self.state not in ("stopping", "stopped") else "disabled")
        for name in ("export_bundle", "export_png"):
            self.buttons[name].configure(state="normal" if self.store is not None and len(self.store.windows) else "disabled")
        if active and self.state != "paused":
            self.buttons["export_bundle"].configure(state="disabled")
        self.buttons["export_curve"].configure(state="normal" if self.store is not None and self.selected_cell is not None else "disabled")

    def load_results(self, path, live=False):
        if not live and self.session is not None and self.session.is_alive():
            self._error("运行中不能切换结果")
            return False
        try:
            store = ResultStore(path)
            self.store = store
            self._window_cache = None
            self._current_index = None
            layers = store.manifest.get("required_layers", [])
            self.layer_combo.configure(values=["全部层"] + list(layers))
            if self.layer_var.get() not in ["全部层"] + list(layers):
                self.layer_var.set("全部层")
            self._update_ports(reset_cell=not live)
            self.window_spin.configure(to=max(0, len(store.windows)-1))
            if not live or self.live_var.get():
                self.window_var.set(max(0, len(store.windows)-1))
            self._schedule_render()
            if not live:
                self._restore_result_input(store)
                self.status_var.set(f"结果已载入 · {len(store.windows)} 个窗口 · {Path(path).name}")
                if self.state == "error":
                    self.state = "idle"
            self._update_buttons()
            return True
        except Exception as exc:
            self._error(exc)
            return False

    def _restore_result_input(self, store):
        """Restore recorded context; never show an unrelated old input preview."""
        if self.media is not None:
            self.media.close()
        self.media = None
        self.preview_canvas.delete("all")
        self._photos.pop("preview", None)
        self.preview_var.set(0)
        self.preview_time.configure(text="0 s")
        self.preview_scale.configure(state="disabled")
        self.params["media_path"].set("")
        assumptions = store.manifest.get("input_assumptions", {})
        input_class = assumptions.get("input_class", store.manifest.get("input_class", "历史输入"))
        self.media_var.set(str(input_class)[:90])
        candidate = None
        try:
            value = store.manifest.get("config")
            if isinstance(value, dict):
                candidate = RunConfig.from_dict(value)
        except (TypeError, ValueError):
            candidate = None
        last_time = 0.0
        if store.windows:
            last_record = store.read_window(-1)
            times = np.asarray(last_record["time"])
            if len(times):
                last_time = float(times[-1])
            self._window_cache = last_record
            self._current_index = len(store.windows)-1
        controlled=store.manifest.get('input',{})
        if controlled.get('class')=='explicit_independent_consumer_uniform_step':
            label=f"均匀阶跃 · native {controlled['native_td']:.4g} td · rod {controlled['rod_Rstar_s']:.4g} R*/s"
            self.media_var.set(label)
            self.preview_canvas.create_text(16,20,anchor='nw',text=label,fill='#64748b')
            self.time_var.set(f"{last_time:.5g} s · 阶跃 {controlled['duration_seconds']:.4g} s")
            self.progress['value']=1.
            # This result is a scalar-input experiment, so retain the active
            # Run configuration and never display its unused RGB fixture.
            return
        total = None
        if candidate is not None:
            try:
                value = float(candidate.warmup_seconds) + float(candidate.duration_seconds)
                if np.isfinite(value) and value > 0:
                    total = value
            except (TypeError, ValueError):
                pass
            if isinstance(candidate.media_path, str) and Path(candidate.media_path).is_file():
                if self.import_media(candidate.media_path):
                    self.config = candidate
                    self._set_config_vars(candidate)
                    stimulus_time = float(candidate.start_seconds) + max(0, last_time-float(candidate.warmup_seconds))
                    self.preview_var.set(stimulus_time)
                    self.preview_time.configure(text=f"{stimulus_time:.3f} s")
                    self._render_preview()
        self.time_var.set(f"{last_time:.5g} / {total:.5g} s" if total is not None else f"{last_time:.5g} s")
        self.progress["value"] = min(1, max(0, last_time/total)) if total is not None else 0

    def _choose_results(self):
        path = filedialog.askdirectory(title="载入运行结果目录")
        if path:
            self.load_results(path)

    def _update_ports(self, reset_cell=False):
        self.port_records = {}
        if self.store is not None:
            for component in self.store.manifest.get("components", []):
                if self.layer_var.get() != "全部层" and self.layer_var.get() not in component.get("layers", []):
                    continue
                for port, meta in component.get("ports", {}).items():
                    key = meta.get("key", f"{component['id']}.{port}")
                    self.port_records[key] = (component, meta)
        keys = list(self.port_records)
        self.port_combo.configure(values=keys)
        if self.port_var.get() not in self.port_records:
            self.port_var.set(keys[0] if keys else "")
            reset_cell = True
        self._port_changed(reset_cell=reset_cell)

    def _port_metadata(self):
        component, port = self.port_records.get(self.port_var.get(), ({}, {}))
        meta = {**component, **port}
        contact_frame = {
            "DoubleExponentialChemicalSynapse": "IPL",
            "RodContactTRPM1": "OPL",
        }.get(meta.get("mechanism"))
        meta["display_element"] = "接点" if contact_frame else "细胞"
        if contact_frame:
            # This is a display-only metadata view. Persisted source metadata
            # remains intact; contact locations are never labeled soma or RF.
            geometry = meta.get("geometry", {})
            meta["geometry"] = {contact_frame: geometry[contact_frame]} if contact_frame in geometry else {}
        return meta

    def _port_changed(self, reset_cell=True):
        if reset_cell:
            self.selected_cell = None
            self.cell_var.set("未选择细胞")
        meta = self._port_metadata()
        types = sorted({str(x) for x in meta.get("cell_types", [])})
        self.type_combo.configure(values=["全部类型"] + types)
        if self.type_var.get() not in ["全部类型"] + types:
            self.type_var.set("全部类型")
        available = list(meta.get("geometry", {}))
        self.coordinate_combo.configure(values=available)
        if self.coordinate_var.get() not in available:
            self.coordinate_var.set(available[0] if available else "")
        self._schedule_render()
        self._update_buttons()

    def _window_selected(self):
        self.live_var.set(False)
        self._window_cache = None
        self._schedule_render()

    def _follow_changed(self):
        if self.live_var.get() and self.store is not None:
            self.window_var.set(max(0, len(self.store.windows)-1))
            self._window_cache = None
            self._schedule_render()

    def _schedule_render(self):
        if not self._render_pending:
            self._render_pending = True
            self.root.after_idle(self._render_results)

    def _canvas_image(self, name, canvas, image):
        self._images[name] = image
        photo = ImageTk.PhotoImage(image)
        self._photos[name] = photo
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo, anchor="nw")

    def _render_results(self):
        self._render_pending = False
        scatter_size = (max(260, self.scatter_canvas.winfo_width()), max(170, self.scatter_canvas.winfo_height()))
        curve_size = (max(260, self.curve_canvas.winfo_width()), max(170, self.curve_canvas.winfo_height()))
        try:
            if self.store is None or not self.store.windows or not self.port_var.get():
                self.pointmap = []
                self._canvas_image("scatter", self.scatter_canvas, empty_plot(scatter_size, "运行或载入结果"))
                self._canvas_image("curve", self.curve_canvas, empty_plot(curve_size, "选择细胞查看记录"))
                return
            index = max(0, min(int(self.window_var.get()), len(self.store.windows)-1))
            if self._window_cache is None or self._current_index != index:
                self._window_cache = self.store.read_window(index)
                self._current_index = index
            record = self._window_cache
            key = self.port_var.get()
            signal = np.asarray(record["signals"][key])
            values = signal[:, -1] if signal.ndim == 2 else signal.reshape(-1)
            meta = self._port_metadata()
            geometry = meta.get("geometry", {}).get(self.coordinate_var.get(), {})
            xy = np.asarray(geometry.get("values", []), float)
            if len(xy) != len(values):
                raise ValueError(f"{key} 坐标数 {len(xy)} 与响应数 {len(values)} 不一致")
            indices = np.arange(len(values))
            owners=meta.get('is_owner',[])
            known_owner=len(owners)==len(values) and all(x is not None for x in owners)
            if self.scope_var.get()!='全部':
                if not known_owner:raise ValueError('此历史输出未保存owner/halo分类')
                mask=np.asarray(owners,bool)
                indices=indices[mask if self.scope_var.get()=='owner' else ~mask]
            if self.type_var.get() != "全部类型":
                types = np.asarray(meta.get("cell_types", []), str)
                if len(types) != len(values):
                    raise ValueError("细胞类型与输出数量不一致")
                indices = indices[types[indices] == self.type_var.get()]
            fixed = (float(self.min_var.get()), float(self.max_var.get())) if self.range_var.get() == "固定" else None
            times = np.asarray(record["time"])
            time = float(times[-1]) if len(times) else 0
            scope_text=(f"owner {sum(owners)} / halo {len(owners)-sum(owners)}" if known_owner else 'scope未分类')
            spatial, self.pointmap = scatter_plot(xy, values, indices, scatter_size, coordinate=self.coordinate_var.get(), coordinate_unit=geometry.get("unit", ""), value_unit=meta.get("unit", ""), selected=self.selected_cell, fixed_range=fixed, title=f"{key} · {scope_text} · {len(indices)} 个{meta['display_element']} · t={time:.5g} s")
            self._canvas_image("scatter", self.scatter_canvas, spatial)
            if self.selected_cell is None:
                curve = empty_plot(curve_size, "选择细胞查看记录")
            else:
                if self.selected_cell >= len(values):
                    self.selected_cell = None
                    curve = empty_plot(curve_size, "选择细胞查看记录")
                else:
                    # Read the selected recording window only. Full histories are
                    # exported on demand, rather than rescanned on every event.
                    if signal.ndim == 2:
                        t, y = times, signal[self.selected_cell]
                    else:
                        t, y = times[-1:], values[self.selected_cell:self.selected_cell+1]
                    ids = meta.get("cell_ids", [])
                    cell_id = ids[self.selected_cell] if self.selected_cell < len(ids) else self.selected_cell
                    self.cell_var.set(f"{meta['display_element']} {cell_id} · {meta.get('population', '')} · {key} / {meta.get('unit', '')}")
                    curve = curve_plot(t, y, curve_size, title=f"{cell_id} · {key} · 窗口 {index}", unit=meta.get("unit", ""), window_values=signal.ndim == 1)
            self._canvas_image("curve", self.curve_canvas, curve)
        except Exception as exc:
            self.pointmap = []
            self._canvas_image("scatter", self.scatter_canvas, empty_plot(scatter_size, str(exc)))
            self._error(exc)

    def _select_cell(self, event):
        if not self.pointmap:
            return
        px, py, idx = min(self.pointmap, key=lambda p: (p[0]-event.x)**2 + (p[1]-event.y)**2)
        if (px-event.x)**2 + (py-event.y)**2 <= 24**2:
            self.selected_cell = idx
            self._schedule_render()
            self._update_buttons()

    def export_curve(self, path):
        try:
            if self.store is None or self.selected_cell is None:
                raise ValueError("先选择细胞")
            self.store.export_curve(self.port_var.get(), self.selected_cell, path)
            self.status_var.set(f"曲线已导出：{path}")
            return True
        except Exception as exc:
            self._error(exc)
            return False

    def export_png(self, path):
        try:
            if Path(path).exists():
                raise FileExistsError("图像文件已存在，请选择新名称")
            self._render_results()
            spatial, curve = self._images["scatter"], self._images["curve"]
            image = Image.new("RGB", (max(spatial.width, curve.width), spatial.height+curve.height), "white")
            image.paste(spatial, (0, 0))
            image.paste(curve, (0, spatial.height))
            image.save(path, format="PNG")
            self.status_var.set(f"图已导出：{path}")
            return True
        except Exception as exc:
            self._error(exc)
            return False

    def export_bundle(self, destination):
        try:
            if self.store is None:
                raise ValueError("没有结果")
            self.store.export_bundle(destination)
            self.status_var.set(f"结果包已导出：{destination}")
            return True
        except Exception as exc:
            self._error(exc)
            return False

    def _choose_export_curve(self):
        path = filedialog.asksaveasfilename(title="导出曲线", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.export_curve(path)

    def _choose_export_png(self):
        path = filedialog.asksaveasfilename(title="导出图", defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.export_png(path)

    def _choose_export_bundle(self):
        path = filedialog.askdirectory(title="选择结果包保存的父目录")
        if path:
            self.export_bundle(Path(path) / self.store.path.name)

    def screenshot(self, path):
        """Capture the real, currently visible native desktop window."""
        self.root.deiconify()
        self.root.lift()
        self.root.update_idletasks()
        self.root.update()
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        image = ImageGrab.grab(bbox=(x, y, x+self.root.winfo_width(), y+self.root.winfo_height()))
        pixels = np.asarray(image.convert("RGB"), dtype=float)
        if pixels.size == 0 or float(pixels.mean()) < 2.0 or float(pixels.std()) < 2.0:
            raise RuntimeError("窗口截图为黑屏或近乎恒定图像；请保持窗口可见后重拍")
        image.save(path)
        return Path(path)

    def close(self):
        self._closing = True
        if self.session is not None and self.session.is_alive():
            self.stop()
        else:
            if self.media is not None:
                self.media.close()
            self.root.destroy()
