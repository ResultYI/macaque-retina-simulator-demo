"""Cell6 playback inside the existing desktop software; reads precomputed assets."""
import json
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .plotting import empty_plot, font, scatter_plot
from .replay_data import ReplayData
from .replay_aligned import AlignedProduct, discover
from .temporal_output import PRODUCTION, EXPERIMENTAL, NOTICE, replay_prediction

ALIGNED_MODE="全层严格对齐"


class PlaybackClock:
    def __init__(self, stop=7.1):
        self.stop, self.position, self.speed, self.playing = stop, 0., 1., False
        self.last = time.monotonic()

    def seek(self, seconds):
        self.position = min(self.stop, max(0., float(seconds)))
        self.last = time.monotonic()

    def play(self):
        if self.position >= self.stop:
            self.seek(0.)
        self.playing, self.last = True, time.monotonic()

    def pause(self):
        self.advance()
        self.playing = False

    def advance(self, now=None):
        now = time.monotonic() if now is None else now
        if self.playing:
            self.position = min(self.stop, self.position + max(0., now-self.last)*self.speed)
            if self.position >= self.stop:
                self.playing = False
        self.last = now
        return self.position


def trace_image(size, series, stop, title, unit, *, raster=None,empty_message="此 group 无已保存期望率；Cell6 实测仅对应红圈条件目标。"):
    im = empty_plot(size, "")
    draw = ImageDraw.Draw(im)
    w,h = im.size
    box = (65, 47, w-24, h-(69 if raster is not None else 32))
    left,top,right,bottom = box
    draw.text((15,8), title, fill="#243447", font=font(13))
    if not series:
        draw.text((20,55), empty_message, fill="#64748b", font=font(12))
        return im, box
    lo = min(float(np.min(y)) for _,y,_,_ in series)
    hi = max(float(np.max(y)) for _,y,_,_ in series)
    pad = max((hi-lo)*.07, abs(hi)*1e-5, 1e-6)
    lo,hi = (max(0.,lo-pad) if unit == "Hz" else lo-pad),hi+pad
    draw.rectangle(box, outline="#cbd5e1")
    axis_format='.6f' if unit=='mV' and hi-lo<.01 else '.4g'
    draw.text((4,top), format(hi,axis_format), fill="#64748b", font=font(10))
    draw.text((4,bottom-14), format(lo,axis_format), fill="#64748b", font=font(10))
    draw.text((4,top+23), unit, fill="#64748b", font=font(10))
    if stop>=1:
        xscore = [left+(right-left)*t/stop for t in (1,min(7,stop))]
        draw.rectangle((xscore[0],top,xscore[1],bottom), fill="#eff6ff")
        draw.text((xscore[0]+3,top+2), "score 1–7 s", fill="#94a3b8", font=font(10))
    for t,y,color,label in series:
        # Min/max envelope preserves brief voltage excursions in the raster width.
        px = np.clip(((np.asarray(t)/stop)*(right-left)).astype(int),0,right-left)
        lower = np.full(right-left+1,np.inf); upper = np.full(right-left+1,-np.inf)
        np.minimum.at(lower,px,y); np.maximum.at(upper,px,y)
        last = None
        for x in np.flatnonzero(np.isfinite(lower)):
            ya = bottom-(lower[x]-lo)/(hi-lo)*(bottom-top)
            yb = bottom-(upper[x]-lo)/(hi-lo)*(bottom-top)
            draw.line((left+x,ya,left+x,yb), fill=color)
            mid = (left+x,(ya+yb)/2)
            if last is not None: draw.line((last,mid),fill=color,width=1)
            last = mid
    legend_x = left
    for _,_,color,label in series:
        draw.text((legend_x,h-24), label, fill=color, font=font(11))
        legend_x += int(draw.textlength(label, font=font(11)))+18
    for t in ((0,1,3,5,7.1) if stop>=7.1 else np.linspace(0,stop,5)):
        x = left+(right-left)*t/stop
        draw.text((x-8,bottom+3),f"{t:.4g}",fill="#64748b",font=font(10))
    if raster is not None:
        for repeat,counts in enumerate(raster):
            y = bottom+24+repeat*7
            for j in np.flatnonzero(counts):
                x = left+(right-left)*(j*.001)/stop
                draw.line((x,y,x,y+4),fill="#334155")
        draw.text((4,bottom+23),"实测×3",fill="#64748b",font=font(10))
    return im, box


def component_plot(values,size,title,unit):
    im=empty_plot(size,"");draw=ImageDraw.Draw(im);w,h=im.size
    draw.text((14,10),title,fill="#243447",font=font(13))
    left,top,right,bottom=65,45,w-30,h-43
    lo,hi=float(np.min(values)),float(np.max(values));span=max(hi-lo,1e-30)
    points=[(left+(right-left)*i/max(len(values)-1,1),bottom-(float(v)-lo)/span*(bottom-top)) for i,v in enumerate(values)]
    draw.rectangle((left,top,right,bottom),outline="#cbd5e1")
    if len(points)>1:draw.line(points,fill="#2563eb",width=1)
    else:draw.ellipse((points[0][0]-2,points[0][1]-2,points[0][0]+2,points[0][1]+2),fill="#2563eb")
    draw.text((5,top),f"{hi:.4g}",fill="#64748b",font=font(10));draw.text((5,bottom-12),f"{lo:.4g}",fill="#64748b",font=font(10))
    draw.text((5,top+20),unit,fill="#64748b",font=font(10))
    draw.text((left,h-29),f"contact / component index 0–{len(values)-1}（无细胞空间坐标）",fill="#64748b",font=font(11))
    return im


class ReplayApp:
    def __init__(self, root, project_root, registry=None):
        self.root, self.project_root = root, Path(project_root).resolve()
        self.registry_path = registry
        self.data = None
        self.clock = PlaybackClock()
        self.selected = 0
        self._photos = {}
        self._plot_bases = None
        self._closed = False
        self._drawing = False
        self.pointmap = []
        self.sparse = None
        self.sparse_selected = 0
        self._sparse_cache = {}
        self._sparse_queue = queue.Queue()
        self._sparse_job = None
        self._sparse_error = ""
        self.aligned=None
        self.aligned_selected=0
        self._aligned_queue=queue.Queue()
        self._aligned_job=None
        self._aligned_field_job=None
        self._aligned_error=""
        self._aligned_paths={}
        self._aligned_cache={}
        self._aligned_ready=None
        self._aligned_desired=None
        self._aligned_labels={}
        self.output_mode_var=tk.StringVar(root,value=PRODUCTION)
        self.output_notice_var=tk.StringVar(root,value="默认保留原生产/既有读出；实验模式需要显式选择。")
        self.experimental_prediction=None
        self.experimental_metadata=None
        self._experimental_cache={}
        self.phase_var=tk.StringVar(root,value="primary")
        self.mode_var = tk.StringVar(root,value="全层稀疏快照")
        self.map_status_var = tk.StringVar(root)
        self.footer_var=tk.StringVar(root)
        self.case_var = tk.StringVar(root)
        self.layer_var = tk.StringVar(root,value="全部 RGC groups")
        self.field_var = tk.StringVar(root,value="V / mV")
        self.node_var = tk.StringVar(root)
        self.speed_var = tk.StringVar(root,value="1×")
        self.time_var = tk.DoubleVar(root,value=0)
        self.jump_var = tk.StringVar(root,value="0")
        self.status_var = tk.StringVar(root,value="读取完整缓存并核验 SHA256…")
        self.assessment_summary_var=tk.StringVar(root)
        self.evaluation_definition_var=tk.StringVar(root)
        self.detail_var = tk.StringVar(root)
        self.path_var = tk.StringVar(root)
        self.time_label = tk.StringVar(root,value="0.000 / 7.100 s · 已暂停")
        root.title("Cell6 · 局部网络预计算回放")
        root.geometry(f"{min(1460,root.winfo_screenwidth()-60)}x{min(950,root.winfo_screenheight()-90)}")
        root.minsize(1060,820)
        self._build()
        root.protocol("WM_DELETE_WINDOW",self.close)
        self.refresh()
        self._timer = root.after(60,self._tick)

    def _build(self):
        style=ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Replay.TLabel",font=("Microsoft YaHei UI",10))
        shell=ttk.Frame(self.root,padding=12)
        shell.pack(fill="both",expand=True)
        shell.columnconfigure(0,weight=1)
        for row,weight in ((3,3),(6,2)):
            shell.rowconfigure(row,weight=weight)
        ttk.Label(shell,text="Cell6  /  全场刺激与局部网络",font=("Microsoft YaHei UI",17,"bold")).grid(row=0,column=0,sticky="w")
        self.status_box=ttk.Frame(shell);self.status_box.grid(row=1,column=0,sticky="ew",pady=(5,6))
        self.status_label=ttk.Label(self.status_box,textvariable=self.status_var,foreground="#9a5a00",wraplength=1350,style="Replay.TLabel")
        self.status_label.pack(fill="x")
        self.assessment_summary_label=ttk.Label(self.status_box,textvariable=self.assessment_summary_var,wraplength=1350,font=("Microsoft YaHei UI",9))
        self.evaluation_definition_label=ttk.Label(self.status_box,textvariable=self.evaluation_definition_var,wraplength=1350,font=("Microsoft YaHei UI",9),foreground="#475569")
        outputs=ttk.Frame(self.status_box);outputs.pack(fill="x",pady=(3,0))
        ttk.Label(outputs,text="放电输出").pack(side="left")
        self.output_combo=ttk.Combobox(outputs,textvariable=self.output_mode_var,state="readonly",
            values=[PRODUCTION,EXPERIMENTAL],width=42)
        self.output_combo.pack(side="left",padx=6)
        self.output_combo.bind("<<ComboboxSelected>>",self.output_mode_changed)
        ttk.Label(outputs,textvariable=self.output_notice_var,foreground="#7e22ce",
            wraplength=830,font=("Microsoft YaHei UI",9)).pack(side="left",fill="x",expand=True)
        row=ttk.Frame(shell);row.grid(row=2,column=0,sticky="ew",pady=(0,7))
        ttk.Label(row,text="Case").pack(side="left")
        self.case_combo=ttk.Combobox(row,textvariable=self.case_var,state="readonly",width=13)
        self.case_combo.pack(side="left",padx=5)
        self.case_combo.bind("<<ComboboxSelected>>",lambda _:self.load_case())
        ttk.Button(row,text="刷新结果",command=self.refresh).pack(side="left",padx=4)
        ttk.Button(row,text="选择注册表…",command=self.choose_registry).pack(side="left",padx=4)
        ttk.Button(row,text="身份 / 拟合证据",command=self.show_evidence).pack(side="left",padx=4)
        ttk.Button(row,text="评价定义与指标",command=self.show_evaluation).pack(side="left",padx=4)
        ttk.Button(row,text="对齐产品…",command=self.choose_aligned).pack(side="left",padx=4)
        self.coverage_label=ttk.Label(row,text="");self.coverage_label.pack(side="right")
        upper=ttk.Frame(shell);upper.grid(row=3,column=0,sticky="nsew")
        upper.columnconfigure(0,weight=2);upper.columnconfigure(1,weight=4);upper.rowconfigure(0,weight=1)
        self.stim_canvas=tk.Canvas(upper,bg="#18212c",highlightthickness=0)
        self.stim_canvas.grid(row=0,column=0,sticky="nsew",padx=(0,10))
        map_box=ttk.Frame(upper);map_box.grid(row=0,column=1,sticky="nsew")
        self.map_canvas=tk.Canvas(map_box,bg="#f8fafc",highlightthickness=0)
        self.map_canvas.pack(fill="both",expand=True)
        ttk.Label(map_box,textvariable=self.map_status_var,wraplength=810,font=("Microsoft YaHei UI",9)).pack(fill="x",pady=(3,0))
        self.map_canvas.bind("<Button-1>",self.select_point)
        for canvas in (self.stim_canvas,self.map_canvas):
            canvas.bind("<Configure>",lambda _:self.render())
        filters=ttk.Frame(shell);filters.grid(row=4,column=0,sticky="ew",pady=6)
        mode=ttk.Combobox(filters,textvariable=self.mode_var,state="readonly",values=[ALIGNED_MODE,"全层稀疏快照","RGC 连续"],width=14)
        mode.pack(side="left",padx=(0,6));mode.bind("<<ComboboxSelected>>",self.map_mode_changed)
        self.phase_combo=ttk.Combobox(filters,textvariable=self.phase_var,state="disabled",values=["primary","post"],width=8)
        self.phase_combo.pack(side="left",padx=(0,6));self.phase_combo.bind("<<ComboboxSelected>>",self.aligned_phase_changed)
        self.layer_combo=ttk.Combobox(filters,textvariable=self.layer_var,state="readonly",width=24)
        self.layer_combo.pack(side="left")
        self.layer_combo.bind("<<ComboboxSelected>>",self.map_layer_changed)
        self.field_combo=ttk.Combobox(filters,textvariable=self.field_var,state="readonly",values=["V / mV","gE / nS","gI / nS"],width=28)
        self.field_combo.pack(side="left",padx=6);self.field_combo.bind("<<ComboboxSelected>>",self.map_field_changed)
        ttk.Label(filters,text="下方 RGC：").pack(side="left")
        self.node_combo=ttk.Combobox(filters,textvariable=self.node_var,state="readonly",width=62)
        self.node_combo.pack(side="left",fill="x",expand=True)
        self.node_combo.bind("<<ComboboxSelected>>",self.select_node)
        ttk.Label(shell,textvariable=self.detail_var,wraplength=1350,style="Replay.TLabel").grid(row=5,column=0,sticky="ew",pady=(0,6))
        charts=ttk.Frame(shell);charts.grid(row=6,column=0,sticky="nsew")
        charts.columnconfigure(0,weight=1);charts.columnconfigure(1,weight=1);charts.rowconfigure(0,weight=1)
        self.voltage_canvas=tk.Canvas(charts,bg="#f8fafc",highlightthickness=0)
        self.voltage_canvas.grid(row=0,column=0,sticky="nsew",padx=(0,8))
        self.rate_canvas=tk.Canvas(charts,bg="#f8fafc",highlightthickness=0)
        self.rate_canvas.grid(row=0,column=1,sticky="nsew")
        for c in (self.voltage_canvas,self.rate_canvas):
            c.bind("<Configure>",self.invalidate_plots)
        timeline=ttk.Frame(shell);timeline.grid(row=7,column=0,sticky="ew",pady=(8,3));timeline.columnconfigure(2,weight=1)
        self.play_button=ttk.Button(timeline,text="播放",command=self.toggle_play);self.play_button.grid(row=0,column=0,padx=(0,5))
        ttk.Button(timeline,text="回到开头",command=lambda:self.seek(0)).grid(row=0,column=1,padx=(0,5))
        self.slider=ttk.Scale(timeline,from_=0,to=7.1,variable=self.time_var,command=self.slider_changed)
        self.slider.grid(row=0,column=2,sticky="ew")
        ttk.Label(timeline,textvariable=self.time_label,width=32).grid(row=0,column=3,padx=6)
        speed=ttk.Combobox(timeline,textvariable=self.speed_var,values=["0.25×","0.5×","1×","2×","4×"],state="readonly",width=7)
        speed.grid(row=0,column=4);speed.bind("<<ComboboxSelected>>",self.change_speed)
        jump=ttk.Entry(timeline,textvariable=self.jump_var,width=7);jump.grid(row=0,column=5,padx=5);jump.bind("<Return>",lambda _:self.jump())
        ttk.Button(timeline,text="跳转 / s",command=self.jump).grid(row=0,column=6)
        ttk.Label(shell,textvariable=self.footer_var,wraplength=1350,style="Replay.TLabel").grid(row=8,column=0,sticky="ew",pady=(4,0))
        ttk.Label(shell,text="节点为模型 group/pool，非逐细胞独立解；AC pool 位置是工程 RF，不是 AC 解剖。模型 patch 中心 4.73°；Cell6 偏心度、空间配准、采集延迟和物理光度未核验。其他类型未拟合。",wraplength=1350,foreground="#64748b",style="Replay.TLabel").grid(row=9,column=0,sticky="ew",pady=(3,0))
        ttk.Label(shell,textvariable=self.path_var,wraplength=1350,foreground="#64748b",font=("Segoe UI",8)).grid(row=10,column=0,sticky="ew",pady=(3,0))

    def choose_registry(self):
        path=filedialog.askopenfilename(title="选择 Cell6 CACHE_REGISTRY.json",filetypes=[("JSON","*.json")])
        if path:
            self.registry_path=Path(path);self.refresh()

    def refresh(self):
        self.clock.pause()
        previous=self.case_var.get()
        try:
            data=ReplayData(self.project_root,self.registry_path)
            available=sorted(data.caches)
            if not available: raise ValueError("注册表中暂无完整缓存")
            self.data=data
            self.case_combo.configure(values=available)
            aligned_cases=[]
            if not previous:
                for case in available:
                    try:
                        if discover(self.project_root,case) is not None:aligned_cases.append(case)
                    except (ValueError,OSError):pass
            self.case_var.set(previous if previous in available else aligned_cases[0] if aligned_cases else available[0])
            self.layer_combo.configure(values=["全部 RGC groups"]+list(data.spec["recording"]))
            missing=sorted(set(data.rows)-set(available))
            self.coverage_label.configure(text="完整缓存："+", ".join(available)+("；待计算："+", ".join(missing) if missing else "；六组完整"))
            self.load_case()
        except Exception as exc:
            self.fail(exc)

    def load_case(self):
        self.clock.pause()
        try:
            self.data.load_case(self.case_var.get())
            self.sparse=None;self._sparse_error=""
            self.aligned=None;self._aligned_error="";self._aligned_ready=None
            self.clock=PlaybackClock(self.data.stop)
            self.clock.speed=float(self.speed_var.get().rstrip("×"))
            self.selected=self.data.target
            self.output_mode_changed(render=False)
            self.node_labels=[f"{n['population']} [{n['column']}] · pool {n['members']}" for n in self.data.nodes]
            self.node_combo.configure(values=self.node_labels)
            self.node_var.set(self.node_labels[self.selected])
            self.ranges={field:(min(float(np.min(self.data.arrays[p+':'+field])) for p in self.data.spec['recording'])*factor,
                                max(float(np.max(self.data.arrays[p+':'+field])) for p in self.data.spec['recording'])*factor)
                         for field,factor in [('V',1000),('gE',1e9),('gI',1e9)]}
            d=self.data
            self.update_status()
            self.path_var.set(str(d.cache_path)+"  |  stimulus SHA256 "+d.case['stimulus_sha256'][:16]+"…  |  payload / 51 source pins 已核验")
            self.slider.configure(to=d.stop,state="normal");self.play_button.configure(state="normal")
            self._plot_bases=None
            self.start_aligned()
            if self.aligned is None and self._aligned_job is None and self.mode_var.get()=="全层稀疏快照":self.start_sparse()
            self.map_mode_changed()
            self.render()
        except Exception as exc:
            self.fail(exc)

    def fail(self, exc):
        self.clock.pause();self.data=None;self._plot_bases=None
        self.sparse=None
        self.aligned=None;self._aligned_ready=None
        self.status_var.set("加载失败，回放已停止："+str(exc))
        self.path_var.set(str(self.registry_path or self.project_root/'reports/rgc_stage1_cpu_parallel_full_v1/CACHE_REGISTRY.json'))
        self.detail_var.set("修复资产或重新选择注册表后点击“刷新结果”；不会退回示例数据。")
        self.play_button.configure(state="disabled",text="播放");self.slider.configure(state="disabled")
        for c in (self.stim_canvas,self.map_canvas,self.voltage_canvas,self.rate_canvas):
            c.delete("all")
        self.map_canvas.create_text(25,30,text="无法读取已核验数据",anchor="nw",fill="#b45309",font=("Microsoft YaHei UI",14))

    def select_node(self, event=None):
        if not self.data: return
        self.selected=self.node_labels.index(self.node_var.get());self._plot_bases=None
        self.output_mode_changed(render=False)
        if self.mode_var.get()==ALIGNED_MODE:self.request_aligned_field()
        self.render()

    def output_mode_changed(self,event=None,*,render=True):
        self.experimental_prediction=None;self.experimental_metadata=None
        if self.output_mode_var.get()==EXPERIMENTAL and self.data is not None:
            try:
                node=self.data.nodes[self.selected]
                key=(self.data.case_id,node["id"],self.data.spec_sha,
                    self.data.caches[self.data.case_id]["cache_identity"],
                    self.data.caches[self.data.case_id]["payload_sha256"])
                if key not in self._experimental_cache:
                    self._experimental_cache[key]=replay_prediction(self.data,node)
                self.experimental_prediction,self.experimental_metadata=self._experimental_cache[key]
                self.output_notice_var.set(NOTICE)
            except (ValueError,OSError,KeyError) as exc:
                self.output_notice_var.set("实验输出已拒绝 / 不适用："+str(exc))
        else:
            self.output_notice_var.set("默认保留原生产/既有读出；实验模式需要显式选择。")
        self._plot_bases=None
        if render and self.data is not None:self.render()

    def select_point(self, event):
        if not self.data or not self.pointmap: return
        px,py,node=min(self.pointmap,key=lambda p:(p[0]-event.x)**2+(p[1]-event.y)**2)
        if (px-event.x)**2+(py-event.y)**2 < 225:
            if self.mode_var.get()==ALIGNED_MODE:
                self.aligned_selected=node;self.render()
            elif self.mode_var.get()=="全层稀疏快照":
                self.sparse_selected=node;self.render()
            else:
                self.node_var.set(self.node_labels[node]);self.select_node()

    def start_sparse(self):
        if not self.data:return
        case=self.data.case_id
        if case in self._sparse_cache:
            self.sparse=self._sparse_cache[case];return
        if self._sparse_job is not None:return
        self._sparse_job=case
        context=SimpleNamespace(**{k:getattr(self.data,k) for k in ("root","spec","source","cache_path","caches","case_id")})
        def read_snapshots():
            try:
                from .replay_snapshots import SparseSnapshots
                self._sparse_queue.put((case,SparseSnapshots(context),None))
            except Exception as exc:
                self._sparse_queue.put((case,None,str(exc)))
        threading.Thread(target=read_snapshots,daemon=True,name="Cell6-read-only-snapshots").start()

    def choose_aligned(self):
        if not self.data:return
        path=filedialog.askopenfilename(title="选择已完成的全层对齐 manifest.json",filetypes=[("JSON","*.json")])
        if path:
            self.clock.pause();self.aligned=None;self._aligned_error="";self._aligned_ready=None
            self._aligned_paths[self.data.case_id]=Path(path)
            self._aligned_cache.pop(self.data.case_id,None)
            self.mode_var.set(ALIGNED_MODE);self.start_aligned();self.map_mode_changed()

    def start_aligned(self):
        if not self.data:return
        case=self.data.case_id
        if case in self._aligned_cache:
            self.attach_aligned(self._aligned_cache[case]);return
        if self._aligned_job is not None:return
        try:path=self._aligned_paths.get(case) or discover(self.project_root,case)
        except (ValueError,OSError) as exc:
            self._aligned_error=str(exc);return
        if path is None:return
        self.mode_var.set(ALIGNED_MODE)
        self._aligned_job=case
        context=SimpleNamespace(**{k:getattr(self.data,k) for k in ("root","spec","spec_sha","case_id","case","caches","frames")})
        context.drive_origin_s=self.data.drive['model_origin_s']
        def read_product():
            try:self._aligned_queue.put(("product",case,AlignedProduct(path,context),None))
            except Exception as exc:self._aligned_queue.put(("product",case,None,str(exc)))
        threading.Thread(target=read_product,daemon=True,name="Cell6-aligned-index-reader").start()

    def attach_aligned(self,product):
        self.aligned=product;self._aligned_error="";self._aligned_ready=None
        self.phase_var.set("primary");self.mode_var.set(ALIGNED_MODE)
        self.clock.pause();self.clock.stop=product.stop;self.clock.seek(min(self.clock.position,product.stop))
        self.slider.configure(to=product.stop)
        self._plot_bases=None;self.map_mode_changed()

    def request_aligned_field(self):
        if self.aligned is None or self.field_var.get() not in self._aligned_labels:return
        n=self.data.nodes[self.selected]
        wanted=(self.data.case_id,self.phase_var.get(),self._aligned_labels[self.field_var.get()],n['population'],n['id'])
        self._aligned_desired=wanted
        if self._aligned_ready==wanted or self._aligned_field_job is not None:return
        self._aligned_ready=None;self._aligned_field_job=wanted;self._aligned_error=""
        self.clock.pause();self.play_button.configure(state="disabled");self.slider.configure(state="disabled")
        product=self.aligned
        def read_field():
            try:
                product.open_field(wanted[1],wanted[2])
                voltage=product.voltage_field(wanted[1],wanted[3],wanted[4])
                if voltage is not None:product.open_field(wanted[1],voltage[0])
                self._aligned_queue.put(("field",wanted,product,None))
            except Exception as exc:self._aligned_queue.put(("field",wanted,product,str(exc)))
        threading.Thread(target=read_field,daemon=True,name="Cell6-aligned-field-reader").start()

    def update_status(self):
        if not self.data:return
        d=self.data
        if self.mode_var.get()==ALIGNED_MODE:
            if self.aligned is not None and self.aligned.synthetic:
                text="合成微型 fixture · 非真实轨迹 · 仅同步/拖动/相位控件检查，未通过科学验收"
            elif self.aligned is not None:
                text=d.fit_label+f" | {d.case_id} · 全层共享 100 μs 索引/side/phase 已核验；数值字段按需核验"
            else:text=d.fit_label+" | 严格对齐产品不可用，未满足全层对齐交付"
            if self._aligned_error:text+=" | 拒绝显示："+self._aligned_error
        else:
            text=d.fit_label+f" | {d.case_id} · {d.case['split']} · score [1,7) s | 当前为稀疏预览或 RGC 单层连续回放"
        if d.fit_warning:text+=" | "+d.fit_warning
        self.status_var.set(text)
        synthetic=self.mode_var.get()==ALIGNED_MODE and self.aligned is not None and self.aligned.synthetic
        self.assessment_summary_var.set('' if synthetic else d.evaluation_summary)
        self.evaluation_definition_var.set('' if synthetic or d.fit is None else d.evaluation_definition)
        for widget,var in ((self.assessment_summary_label,self.assessment_summary_var),(self.evaluation_definition_label,self.evaluation_definition_var)):
            if var.get():widget.pack(fill='x',pady=(2,0))
            else:widget.pack_forget()
        inadequate=d.dynamic_output_insufficient and not synthetic
        self.status_label.configure(foreground='#b91c1c' if inadequate else '#9a5a00',font=('Microsoft YaHei UI',12,'bold') if inadequate else ('Microsoft YaHei UI',10))
        self.footer_var.set("预计算回放 · 绘制上限约16fps | 每帧所有层取同一100 μs源行/side/phase/event，UI不做时间插值；1 ms 放电独立。"
                            if self.mode_var.get()==ALIGNED_MODE else
                            "预计算回放 · 非实时模拟 | 稀疏模式保持实际旧快照而不插值；RGC 连续图红圈=Cell6 条件目标，橙框=选中节点。")
        if self.mode_var.get()==ALIGNED_MODE and self.aligned is not None:
            self.path_var.set(str(self.aligned.path)+(" | 合成 fixture，非真实轨迹或科学验收" if self.aligned.synthetic else " | 共享索引已核验；当前字段逐一校验后只读映射"))
        else:
            self.path_var.set(str(d.cache_path)+" | stimulus SHA256 "+d.case['stimulus_sha256'][:16]+"… | 已核验原始缓存")

    def map_mode_changed(self,event=None):
        self._plot_bases=None
        self.phase_combo.configure(state="readonly" if self.mode_var.get()==ALIGNED_MODE and self.aligned else "disabled")
        if self.data:
            self.slider.configure(state="normal");self.play_button.configure(state="normal")
            if self.mode_var.get()!=ALIGNED_MODE:
                self.clock.stop=self.data.stop;self.slider.configure(to=self.data.stop)
        if self.mode_var.get()==ALIGNED_MODE:
            if self.aligned is None:
                self.clock.pause();self.play_button.configure(state="disabled");self.slider.configure(state="disabled")
                self.layer_combo.configure(values=[]);self.field_combo.configure(values=[])
                self.layer_var.set("对齐产品不可用" if self._aligned_error else "等待完整对齐产品")
            else:
                self.clock.stop=self.aligned.stop;self.clock.position=min(self.clock.position,self.clock.stop)
                self.slider.configure(to=self.clock.stop)
                layers=list(dict.fromkeys(f["population"] for f in self.aligned.fields[self.phase_var.get()].values()))
                self.layer_combo.configure(values=layers)
                if self.layer_var.get() not in layers:
                    self.layer_var.set("cross_OFF_ON_parasol_AC_v1" if "cross_OFF_ON_parasol_AC_v1" in layers else layers[0])
                self.map_layer_changed()
            self.update_status()
        elif self.mode_var.get()=="RGC 连续":
            self.clock.stop=self.data.stop if self.data else 7.1
            self.slider.configure(to=self.clock.stop)
            self.layer_combo.configure(values=["全部 RGC groups"]+list(self.data.spec["recording"]) if self.data else [])
            self.layer_var.set("全部 RGC groups")
            self.field_combo.configure(values=["V / mV","gE / nS","gI / nS"]);self.field_var.set("V / mV")
        elif self.sparse is not None:
            layers=list(self.sparse.fields)
            self.layer_combo.configure(values=layers)
            if self.layer_var.get() not in layers:self.layer_var.set("cross_OFF_ON_parasol_AC_v1")
            self.map_layer_changed()
        else:
            if self._sparse_job is None and self.mode_var.get()=="全层稀疏快照":self.start_sparse()
            self.layer_combo.configure(values=[])
            self.layer_var.set("正在读取稀疏快照…" if not self._sparse_error else "稀疏快照不可用")
            self.field_combo.configure(values=[])
        self.update_status()
        self.render()

    def map_layer_changed(self,event=None):
        if self.mode_var.get()==ALIGNED_MODE and self.aligned is not None:
            fields=[f for f in self.aligned.fields[self.phase_var.get()].values() if f["population"]==self.layer_var.get()]
            self._aligned_labels={f"{f['field']} · {f['kind']} [{f['id']}] / {f['unit']}":f["id"] for f in fields}
            labels=list(self._aligned_labels)
            preferred=next((label for label,f in zip(labels,fields) if f["kind"]=="states" and f["field"] in ("V","Vis","Vh")),labels[0])
            self.field_combo.configure(values=labels);self.field_var.set(preferred)
            self.aligned_selected=0;self.request_aligned_field()
        elif self.mode_var.get()=="全层稀疏快照" and self.sparse is not None:
            labels=list(self.sparse.fields[self.layer_var.get()])
            preferred=next((label for label in labels if label.startswith(("V /","Vh /","Vis /","h /"))),labels[0])
            self.field_combo.configure(values=labels);self.field_var.set(preferred)
            self.sparse_selected=0
        self.render()

    def map_field_changed(self,event=None):
        if self.mode_var.get()==ALIGNED_MODE:self.request_aligned_field()
        self.render()

    def aligned_phase_changed(self,event=None):
        self.clock.pause();self._plot_bases=None;self._aligned_ready=None
        self.request_aligned_field();self.render()

    def invalidate_plots(self, event=None):
        self._plot_bases=None;self.render()

    def seek(self, seconds):
        self.clock.seek(seconds);self.render()

    def slider_changed(self, value):
        if not self._drawing: self.seek(float(value))

    def jump(self):
        try:
            value=float(self.jump_var.get())
            if not np.isfinite(value): raise ValueError("finite time required")
            self.seek(value)
        except ValueError:
            self.jump_var.set(f"{self.clock.position:.3f}")

    def change_speed(self, event=None):
        self.clock.advance();self.clock.speed=float(self.speed_var.get().rstrip("×"))

    def toggle_play(self):
        if not self.data: return
        self.clock.pause() if self.clock.playing else self.clock.play()
        self.render()

    def _tick(self):
        if self._closed: return
        try:
            kind,key,product,error=self._aligned_queue.get_nowait()
            if kind=="product":
                self._aligned_job=None
                if product is not None:self._aligned_cache[key]=product
                if self.data:
                    if key==self.data.case_id:
                        self._aligned_error=error or ""
                        if product is not None:self.attach_aligned(product)
                        else:self.map_mode_changed()
                    else:self.start_aligned()
            else:
                self._aligned_field_job=None
                if self.data and product is self.aligned and key==self._aligned_desired:
                    self._aligned_error=error or "";self._aligned_ready=key if not error else None
                    if error:self.clock.pause();self.play_button.configure(state="disabled");self.slider.configure(state="disabled")
                    else:
                        keep={(key[1],key[2])};v=product.voltage_field(key[1],key[3],key[4])
                        if v is not None:keep.add((key[1],v[0]))
                        product.release_fields(keep)
                        self.play_button.configure(state="normal");self.slider.configure(state="normal")
                    self._plot_bases=None;self.update_status();self.render()
                elif self.aligned is not None:self.request_aligned_field()
        except queue.Empty:pass
        try:
            case,sparse,error=self._sparse_queue.get_nowait()
            self._sparse_job=None
            if sparse is not None:self._sparse_cache[case]=sparse
            if self.data:
                if case==self.data.case_id:
                    self.sparse=sparse;self._sparse_error=error or "";self.map_mode_changed()
                else:self.start_sparse()
        except queue.Empty:pass
        if self.clock.playing:
            self.clock.advance();self.render()
        self._timer=self.root.after(60,self._tick)

    def show_image(self, canvas, im, key):
        self._photos[key]=ImageTk.PhotoImage(im,master=self.root)
        canvas.delete("all");canvas.create_image(0,0,image=self._photos[key],anchor="nw")

    def build_plots(self):
        d,n=self.data,self.data.nodes[self.selected]
        size=lambda c:(max(360,c.winfo_width()),max(190,c.winfo_height()))
        aligned=self.mode_var.get()==ALIGNED_MODE
        stop=self.clock.stop
        if aligned:
            series=[]
            if self.aligned is not None and self._aligned_ready==self._aligned_desired:
                found=self.aligned.voltage_field(self.phase_var.get(),n['population'],n['id'])
                if found is not None:
                    fid,column=found
                    if self.aligned.fields[self.phase_var.get()][fid]['unit']!="V":raise ValueError("Aligned voltage unit differs")
                    v=self.aligned.open_field(self.phase_var.get(),fid)[:,column]*1000
                    series=[(self.aligned.indices[self.phase_var.get()]['time_source_s'],v,"#2563eb",self.phase_var.get()+" · "+("合成测试 RGC 电位，非真实预测" if self.aligned.synthetic else "原始对齐膜电位"))]
            voltage=trace_image(size(self.voltage_canvas),series,stop,"选中 RGC · 与当前层同一 stream/row（无实测膜电位）","mV",empty_message="当前 RGC 字段尚未核验或身份不匹配；未混用旧曲线。")
        else:
            voltage=trace_image(size(self.voltage_canvas),[(d.time,d.trace(self.selected),"#2563eb","已缓存膜电位")],stop,"选中 group · 膜电位（无实测膜电位）","mV")
        series=[];raster=None
        bins=(np.arange(355)+.5)*.02
        if n['population']=='parasol_RGC_ON':
            baseline=d.arrays['expected_rate_base_Hz'][:,n['column']].reshape(355,20).mean(axis=1)
            series.append((bins,baseline,"#2563eb","pre-fit 基线 / 20 ms均值"))
        if self.selected==d.target:
            observed=d.counts.reshape(3,355,20).sum(axis=(0,2))/(3*.02)
            series.append((bins,observed,"#64748b","实测 PSTH / 20 ms"));raster=d.counts
            if d.prediction is not None:
                label=("拟合后(动态不足) / 20 ms" if d.dynamic_output_insufficient else "拟合后(动态待评) / 20 ms") if d.fit['accepted'] else "拒绝的候选 / 20 ms"
                series.append((bins,d.prediction.reshape(355,20).mean(axis=1),"#d97706" if d.fit['accepted'] else "#dc2626",label))
            if self.experimental_prediction is not None and self.output_mode_var.get()==EXPERIMENTAL:
                series.append((bins,self.experimental_prediction.reshape(355,20).mean(axis=1),
                    "#7e22ce","实验动态期望 / INCONCLUSIVE / 20 ms"))
        if aligned and self.aligned is not None and self.aligned.synthetic:series=[];raster=None
        rate=trace_image(size(self.rate_canvas),series,stop,"独立 1 ms 预事件率/实测；不充作 100 μs post 状态" if aligned else "期望率与 Cell6 实测（下方三次原始重复）","Hz",raster=raster)
        self._plot_bases=(voltage,rate)

    def render_aligned_map(self,size,t):
        if self.aligned is None:
            self.map_status_var.set("拒绝显示未完成/缺失产品；没有用稀疏快照或旧层数据替代。")
            return empty_plot(size,"严格对齐产品不可用："+self._aligned_error if self._aligned_error else "等待完整全层对齐产品；当前未满足严格对齐验收。")
        if self._aligned_ready!=self._aligned_desired:
            self.map_status_var.set("只读核验选中字段 SHA256 并创建内存映射；未读取整包。"+("拒绝："+self._aligned_error if self._aligned_error else ""))
            return empty_plot(size,"字段不可用："+self._aligned_error if self._aligned_error else "正在按需核验当前字段…")
        row=self.aligned.row_at(t);phase=self.phase_var.get();fid=self._aligned_labels[self.field_var.get()]
        sample=self.aligned.sample(phase,fid,row);p=sample['point'];f=sample['field']
        title=f"{f['population']} · {f['kind']}:{f['field']} · {phase}"
        baseline_history=f.get('mechanism')=='RGC_expected_history'
        if baseline_history:title+=" · pre-fit 历史状态"
        if f['axis']['axis_kind']=='contact_or_component_vector':
            im=component_plot(sample['values'],size,title,sample['unit'])
            identity="contact/component 向量；无细胞空间坐标"
        else:
            self.aligned_selected=min(self.aligned_selected,f['count']-1)
            role=f['coordinate_role'];coordinate="RF pool (engineering)" if f['cell_count_not_applicable'] else "Soma centroid" if role=='soma' else "RF centroid"
            im,self.pointmap=scatter_plot(f['centroid_mm'],sample['values'],np.arange(f['count']),size,
                coordinate=coordinate,coordinate_unit="mm",value_unit=sample['unit'] if len(sample['unit'])<9 else "native",
                fixed_range=sample['range'],title=title)
            for px,py,index in self.pointmap:
                if index==self.aligned_selected:ImageDraw.Draw(im).rectangle((px-7,py-7,px+7,py+7),outline="#f97316",width=2)
            i=self.aligned_selected
            identity=f"{f['group_ids'][i]} · "+("有效功能 pool，细胞数不适用" if f['cell_count_not_applicable'] else f"members={f['members'][i]}")+f" · {sample['values'][i]:.7g} {sample['unit']}"
        prefix="合成 fixture / 非真实轨迹 · " if self.aligned.synthetic else "原基线历史概率，非拟合后状态 · " if baseline_history else ""
        self.map_status_var.set(prefix+f"共享 row={row} · t={p['time_source_s']:.4f}s · {p['side']} / {p['phase']} · event={p['event_generation']} · frame={p['input_frame']} · rod/BC/RGC={p['clock_rod']}/{p['clock_bc']}/{p['clock_rgc']}\n"+identity+"；色标用最多128个共享行抽样，值未插值。")
        return im

    def render(self):
        if not self.data or self._drawing: return
        self._drawing=True
        try:
            d,t=self.data,self.clock.position
            aligned=self.mode_var.get()==ALIGNED_MODE
            point=None
            if aligned and self.aligned is not None:
                point=self.aligned.point(self.phase_var.get(),self.aligned.row_at(t));t=point['time_source_s']
            self.time_var.set(t)
            self.time_label.set(f"{t:.4f} / {self.clock.stop:.4f} s · "+("播放中" if self.clock.playing else "末尾 / 已暂停" if t>=self.clock.stop else "已暂停"))
            self.play_button.configure(text="暂停" if self.clock.playing else "从头播放" if t>=self.clock.stop else "播放")
            synthetic=aligned and self.aligned is not None and self.aligned.synthetic
            stim=None if synthetic else d.stimulus_at(t,frame=point['input_frame']) if point is not None else d.stimulus_at(t)
            c=self.stim_canvas;w,h=max(300,c.winfo_width()),max(190,c.winfo_height())
            im=Image.new("RGB",(w,h),"#18212c");draw=ImageDraw.Draw(im)
            draw.text((15,11),"合成 fixture 控件检查 · 非真实刺激" if synthetic else "对应全场刺激重建预览 · 150 Hz ZOH",font=font(15),fill="#f8fafc")
            gray=128 if synthetic else int(np.clip(round(128*stim['multiplier']),0,255))
            draw.rectangle((18,45,w-18,max(65,h-115)),fill=(gray,gray,gray))
            if synthetic:
                lines=["合成同步 fixture，不包含真实刺激、观测或拟合。",
                       f"共享 row={self.aligned.row_at(t)} · frame={point['input_frame']}",
                       f"{point['side']} / {point['phase']} · event={point['event_generation']}",
                       "恒定灰块仅是测试背景，不是刺激重建。"]
            else:
                v=stim['native_td']
                lines=[f"倍率 {stim['multiplier']:.6f}  ·  raw {stim['raw']:.6g}（源单位未核验）",
                       f"cone drive mean {v[1]:.5g} td  ·  frame {stim['frame']}",
                       f"源样本 j={stim['source_index']}  ·  t={t:.4f} s",
                       "灰度=128×输入倍率，仅为显示；非物理亮度或原视频。"]
            for j,line in enumerate(lines):draw.text((15,h-99+j*22),line,font=font(11),fill="#e2e8f0")
            self.show_image(c,im,"stimulus")
            size=(max(480,self.map_canvas.winfo_width()),max(190,self.map_canvas.winfo_height()))
            self.pointmap=[]
            if aligned:
                try:im=self.render_aligned_map(size,t)
                except Exception as exc:
                    self._aligned_error=str(exc);self._aligned_ready=None;self.clock.pause()
                    self.play_button.configure(state="disabled");self.slider.configure(state="disabled");self.update_status()
                    im=empty_plot(size,"对齐字段拒绝显示："+str(exc));self.map_status_var.set("未使用其他层/相位或旧数据替代。")
            elif self.mode_var.get()=="全层稀疏快照":
                if self.sparse is None:
                    im=empty_plot(size,"稀疏快照未核验："+self._sparse_error if self._sparse_error else "正在只读核验原始检查点…下方 RGC 可先播放。")
                    self.map_status_var.set("仅提取已保存的状态数组；不恢复或运行模型。")
                else:
                    pop=self.layer_var.get();sample=self.sparse.sample(pop,self.field_var.get(),t);g=self.sparse.geometry[pop]
                    self.sparse_selected=min(self.sparse_selected,len(g['ids'])-1)
                    im,self.pointmap=scatter_plot(g['xy'],sample['values'],np.arange(len(g['ids'])),size,
                        coordinate="RF pool (engineering)" if g['effective'] else "Soma centroid" if g['coordinate']=="soma" else "RF centroid",
                        coordinate_unit="mm",value_unit=sample['unit'] if len(sample['unit'])<9 else "native",fixed_range=sample['range'],
                        title=f"{pop} · {self.field_var.get()} · 实际稀疏快照")
                    for px,py,index in self.pointmap:
                        if index==self.sparse_selected:ImageDraw.Draw(im).rectangle((px-7,py-7,px+7,py+7),outline="#f97316",width=2)
                    node=self.sparse_selected
                    identity="有效功能 pool（无 AC 解剖位置/细胞数）" if g['effective'] else f"group members={g['members'][node]}"
                    self.map_status_var.set(f"快照 t={sample['stamp']:.7f} s；刺激 t={t:.4f} s；保持 {sample['age']:.4f} s，未插值。\n{g['ids'][node]} · {identity} · value={sample['values'][node]:.7g} {sample['unit']}")
            else:
                field=self.field_var.get().split()[0]
                ids=np.array([i for i,n in enumerate(d.nodes) if self.layer_var.get()=="全部 RGC groups" or n['population']==self.layer_var.get()])
                points=np.array([n['xy'] for n in d.nodes]);values=d.node_values(field,t)
                im,self.pointmap=scatter_plot(points,values,ids,size,
                    coordinate="RF group centroid",coordinate_unit="mm",value_unit="mV" if field=="V" else "nS",selected=d.target,
                    fixed_range=self.ranges[field],title=f"局部 RGC groups / pools · {field} · 颜色固定全程范围 · 点击可选择")
                if self.selected != d.target:
                    for px,py,index in self.pointmap:
                        if index == self.selected:ImageDraw.Draw(im).rectangle((px-9,py-9,px+9,py+9),outline="#f97316",width=2)
                self.map_status_var.set(f"RGC 源时间 {t:.4f} s；缓存采样 100 μs。红圈固定为条件目标 pool 8166（2 成员）。")
            self.show_image(self.map_canvas,im,"map")
            n=d.nodes[self.selected]
            self.detail_var.set(f"选中 {n['id']}  |  pool={n['members']} 个模型成员  |  centroid=({n['xy'][0]:.6f}, {n['xy'][1]:.6f}) mm"+
                                ("  |  Cell6 条件代表（非解剖同一细胞）" if self.selected==d.target else "  |  无对应 Cell6 实测身份"))
            if self.selected==d.target and d.goal_assessment is not None:
                row=next((r for r in d.goal_assessment.get('cache_dynamic_range_frozen_score_windows',[]) if r['case_id']==d.case_id),None)
                if row:self.detail_var.set(self.detail_var.get()+f" | 评分窗 V峰峰 {row['signals']['V']['peak_to_peak']*1e6:.3f} μV")
            if aligned and self.aligned is not None and self.aligned.synthetic:
                self.detail_var.set("合成微型 fixture：所有示例节点均与真实 Cell6 无关；实测、基线和拟合曲线已停用。仅验证共享索引与控件。")
            if self._plot_bases is None:self.build_plots()
            for (base,box),canvas,key in zip(self._plot_bases,(self.voltage_canvas,self.rate_canvas),("voltage","rate")):
                im=base.copy();draw=ImageDraw.Draw(im);x=box[0]+(box[2]-box[0])*t/self.clock.stop
                draw.line((x,box[1],x,box[3]),fill="#ef4444",width=2)
                self.show_image(canvas,im,key)
        finally:
            self._drawing=False

    def show_evidence(self):
        if not self.data:return
        window=tk.Toplevel(self.root);window.title("Cell6 只读身份与拟合证据");window.geometry("930x650")
        text=tk.Text(window,wrap="word",font=("Consolas",10));text.pack(fill="both",expand=True)
        evidence=self.data.evidence()
        evidence["output_selection"]=dict(mode=self.output_mode_var.get(),
            notice=self.output_notice_var.get(),experimental=self.experimental_metadata)
        if self.sparse is not None:evidence["sparse_snapshots"]=self.sparse.manifests
        if self.aligned is not None:
            evidence["aligned_product"]=dict(manifest=str(self.aligned.path),synthetic_fixture=self.aligned.synthetic,
                selected_phase=self.phase_var.get(),mapped_value_files=[list(k) for k in self.aligned._mapped],
                note="Shared indices checked; value hashes/memory maps opened only when selected. No whole-package read.")
        text.insert("1.0",json.dumps(evidence,ensure_ascii=False,indent=2));text.configure(state="disabled")

    def show_evaluation(self):
        if not self.data:return
        d=self.data
        window=tk.Toplevel(self.root);window.title('拟合评价：原NLL门与动态目标');window.geometry('1270x430')
        shell=ttk.Frame(window,padding=14);shell.pack(fill='both',expand=True)
        ttk.Label(shell,text=d.fit_label,foreground='#b91c1c',font=('Microsoft YaHei UI',13,'bold'),wraplength=1220).pack(fill='x')
        ttk.Label(shell,text=d.evaluation_definition,wraplength=1220,padding=(0,10)).pack(fill='x')
        columns=('case','old_nll','new_nll','improvement','corr','obs_mean','pred_mean','obs_sd','pred_sd','rmse')
        table=ttk.Treeview(shell,columns=columns,show='headings',height=3)
        for name,label in zip(columns,('验证case','pre-fit NLL/bin','拟合 NLL/bin','NLL改善','PSTH r','实测均值Hz','预测均值Hz','实测调制SD Hz','预测调制SD Hz','PSTH RMSE Hz')):
            table.heading(name,text=label);table.column(name,width=116,anchor='center')
        fmt=lambda value,digits=5:'—' if value is None else f'{value:.{digits}f}'
        for row in d.validation_metrics():
            p=row['PSTH'];old,new=row['baseline_NLL_per_bin'],row['candidate_NLL_per_bin']
            if not p:continue
            table.insert('', 'end',values=(row['case_id'],fmt(old,7),fmt(new,7),fmt((old-new)/old*100,2)+'%',
                fmt(p.get('PSTH_Pearson')),fmt(p.get('observed_rate_Hz'),2),fmt(p.get('predicted_rate_Hz'),2),
                fmt(p.get('observed_modulation_std_Hz')),fmt(p.get('predicted_modulation_std_Hz')),fmt(p.get('PSTH_RMSE_Hz'))))
        table.pack(fill='x',pady=8)
        text='原 accepted 字段仅对应冻结 NLL 门，未更改。动态目标未满足；这里没有新增事后通过阈值。\n评分窗内预测近常定约58.79 Hz，不能用 NLL 改善替代动态复现。\n蓝线 pre-fit，橙线真实拟合后自由历史期望，灰线/下方栅格为真实记录。曲线同一 Hz 轴，无幅度重标定。'
        ttk.Label(shell,text=text,wraplength=1220,foreground='#9a3412').pack(fill='x',pady=8)
        ttk.Label(shell,text='评价来源：fit/CELL6_RESULT.json 与 OUTPUT_GOAL_ASSESSMENT.json；完整身份与 SHA 见“身份 / 拟合证据”。',wraplength=1220).pack(fill='x')
        return window,table

    def close(self):
        self._closed=True;self.clock.pause()
        if hasattr(self,"_timer"):self.root.after_cancel(self._timer)
        # A daemon reader may still own a map; it exits with the application.
        if self._aligned_job is None and self._aligned_field_job is None:
            for product in {id(p):p for p in list(self._aligned_cache.values())+([self.aligned] if self.aligned else [])}.values():product.close()
        self.root.destroy()
