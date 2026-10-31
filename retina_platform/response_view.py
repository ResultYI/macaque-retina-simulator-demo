"""Layer-first response playback in the existing Tk desktop application."""
from pathlib import Path
import json
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
from .replay import PlaybackClock
from .response_bundle import open_result, export_package, sha
from .response_data import ResponseData, RGC, LAYERS, latest_index
from .response_drawing import spatial_image, trace_image

DEFAULT_RESULT = 'reports/external_video_two_positions_4s_20261029_v1'


class ResponseApp:
    def __init__(self, root, project_root, result=None):
        self.root, self.project_root = root, Path(project_root).resolve()
        self.source = self.data = None
        self.clock = PlaybackClock(4.)
        self.target = self.cell = None
        self.network, self.direct = {}, set()
        self.photos, self.maps, self.cards = {}, {}, []
        self.tasks = queue.Queue(); self.busy = self.closed = False
        self.curve_window = None; self.curve_key = tk.StringVar(root)
        self.position = tk.StringVar(root, value='near')
        self.rgc_type = tk.StringVar(root, value='全部类型')
        self.rgc = tk.StringVar(root)
        self.delta = tk.BooleanVar(root, value=True)
        self.all_patch = tk.BooleanVar(root, value=False)
        self.rate = tk.BooleanVar(root, value=False)
        self.speed = tk.StringVar(root, value='1×')
        self.timeline = tk.DoubleVar(root, value=0.)
        self.status = tk.StringVar(root, value='导入完整响应包或运行目录；只读取已保存数据，不启动计算。')
        self.time_label = tk.StringVar(root, value='0.000 / 4.000 s')
        self.detail = tk.StringVar(root, value='选择一个RGC组，再点击各层细胞查看身份与状态。')
        self.coverage = tk.StringVar(root)
        root.title('Macaque Retina · 完整响应分层回放')
        root.geometry(f'{min(1540,root.winfo_screenwidth()-55)}x{min(1000,root.winfo_screenheight()-90)}')
        root.minsize(1120, 760)
        self._build()
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.timer = root.after(50, self._tick)
        if result: root.after(100, lambda: self.load(result))

    def _build(self):
        style=ttk.Style(self.root);style.theme_use('clam')
        style.configure('Response.TLabelframe.Label', font=('Microsoft YaHei UI',10,'bold'))
        shell=ttk.Frame(self.root,padding=10);shell.pack(fill='both',expand=True)
        header=ttk.Frame(shell);header.pack(fill='x')
        ttk.Label(header,text='完整响应  /  逐层空间回放',font=('Microsoft YaHei UI',16,'bold')).pack(side='left')
        for text,command in [('导入结果ZIP…',self.choose_package),('导入响应目录…',self.choose_directory),
                             ('导入刺激视频…',self.choose_video),('导出结果ZIP…',self.export)]:
            ttk.Button(header,text=text,command=command).pack(side='left',padx=4)
        ttk.Button(header,text='已有4秒结果',command=lambda:self.load(self.project_root/DEFAULT_RESULT)).pack(side='right')
        ttk.Label(shell,textvariable=self.status,wraplength=1440,foreground='#226276').pack(fill='x',pady=(5,4))
        selection=ttk.Frame(shell);selection.pack(fill='x',pady=(0,5))
        ttk.Label(selection,text='位置').pack(side='left')
        self.position_combo=ttk.Combobox(selection,textvariable=self.position,state='readonly',width=8)
        self.position_combo.pack(side='left',padx=5);self.position_combo.bind('<<ComboboxSelected>>',self.change_position)
        self.type_combo=ttk.Combobox(selection,textvariable=self.rgc_type,state='readonly',width=23,values=['全部类型',*RGC])
        self.type_combo.pack(side='left');self.type_combo.bind('<<ComboboxSelected>>',lambda _:self.refresh_targets())
        self.rgc_combo=ttk.Combobox(selection,textvariable=self.rgc,state='readonly',width=66)
        self.rgc_combo.pack(side='left',padx=5,fill='x',expand=True);self.rgc_combo.bind('<<ComboboxSelected>>',self.choose_target)
        ttk.Checkbutton(selection,text='全部patch作灰色对照',variable=self.all_patch,command=self.render).pack(side='right')
        content=ttk.Panedwindow(shell,orient='horizontal');content.pack(fill='both',expand=True)
        sidepane=ttk.Frame(content,width=295);content.add(sidepane,weight=1)
        sidecanvas=tk.Canvas(sidepane,width=295,highlightthickness=0)
        scrollbar=ttk.Scrollbar(sidepane,orient='vertical',command=sidecanvas.yview)
        scrollbar.pack(side='right',fill='y');sidecanvas.pack(fill='both',expand=True)
        sidecanvas.configure(yscrollcommand=scrollbar.set)
        sidebar=ttk.Frame(sidecanvas)
        sidewindow=sidecanvas.create_window((0,0),window=sidebar,anchor='nw')
        sidebar.bind('<Configure>',lambda _:sidecanvas.configure(scrollregion=sidecanvas.bbox('all')))
        def resize_sidebar(event):
            sidecanvas.itemconfigure(sidewindow,width=event.width)
            for widget in sidebar.winfo_children():
                if isinstance(widget,ttk.Label):widget.configure(wraplength=max(180,event.width-12))
        sidecanvas.bind('<Configure>',resize_sidebar)
        self.video=tk.Canvas(sidebar,bg='#101b28',height=190,highlightthickness=0);self.video.pack(fill='x')
        self.video.bind('<Configure>',lambda _:self.render())
        ttk.Label(sidebar,text='同一物理时间轴 · 原始RGB帧',font=('Microsoft YaHei UI',9)).pack(anchor='w',pady=(3,8))
        ttk.Checkbutton(sidebar,text='Δ变化（相对各组t=0）',variable=self.delta,command=self.render).pack(anchor='w')
        ttk.Checkbutton(sidebar,text='RGC显示期望二元率 / Hz',variable=self.rate,command=self.render).pack(anchor='w')
        ttk.Label(sidebar,text='色标在整个片段内固定。\n白色圈：目标的直接输入\n黄色圈：选中的RGC\n灰色：当前RGC祖先之外\n颜色＝保存状态；不等同因果证明。',wraplength=280,foreground='#475569').pack(fill='x',pady=8)
        ttk.Label(sidebar,textvariable=self.detail,wraplength=285,justify='left',font=('Microsoft YaHei UI',9)).pack(fill='x',pady=4)
        buttons=ttk.Frame(sidebar);buttons.pack(fill='x',pady=6)
        ttk.Button(buttons,text='响应曲线…',command=self.show_curve).pack(side='left',padx=(0,5))
        ttk.Button(buttons,text='细胞 / 状态详情…',command=self.show_details).pack(side='left')
        ttk.Button(sidebar,text='参数、来源与缺失项…',command=self.show_provenance).pack(fill='x',pady=3)
        ttk.Button(sidebar,text='Cell6既有回放…',command=self.open_cell6).pack(fill='x',pady=3)
        ttk.Label(sidebar,textvariable=self.coverage,wraplength=280,foreground='#64748b',font=('Microsoft YaHei UI',9)).pack(fill='x',pady=8)
        ttk.Label(sidebar,text='FIT_READY / background_qualified /\nproduction_adopted 均保持 false。',foreground='#7e5b21',wraplength=280,font=('Microsoft YaHei UI',9)).pack(fill='x')
        grid=ttk.Frame(content);content.add(grid,weight=4)
        for i in range(3):grid.columnconfigure(i,weight=1,uniform='layer');grid.rowconfigure(i,weight=1,uniform='layer')
        for i,(title,populations) in enumerate(LAYERS):
            frame=ttk.LabelFrame(grid,text=title,padding=3,style='Response.TLabelframe')
            frame.grid(row=i//3,column=i%3,sticky='nsew',padx=3,pady=3)
            canvas=tk.Canvas(frame,bg='#101b28',highlightthickness=0);canvas.pack(fill='both',expand=True)
            caption=tk.StringVar(self.root,value='等待完整响应')
            label=ttk.Label(frame,textvariable=caption,font=('Microsoft YaHei UI',8),wraplength=250)
            label.pack(side='bottom',fill='x',before=canvas)
            canvas.bind('<Configure>',lambda event,label=label:(label.configure(wraplength=max(180,event.width)),self.render()))
            canvas.bind('<Button-1>',lambda event,c=canvas:self.click_cell(c,event))
            self.cards.append((canvas,caption,populations))
        controls=ttk.Frame(shell);controls.pack(side='bottom',fill='x',pady=(7,0),before=content);controls.columnconfigure(4,weight=1)
        self.play_button=ttk.Button(controls,text='播放',command=self.toggle_play);self.play_button.grid(row=0,column=0)
        ttk.Button(controls,text='◀ 一帧',command=lambda:self.step(-1)).grid(row=0,column=1,padx=3)
        ttk.Button(controls,text='一帧 ▶',command=lambda:self.step(1)).grid(row=0,column=2,padx=3)
        ttk.Button(controls,text='起点',command=lambda:self.seek(0)).grid(row=0,column=3,padx=3)
        self.slider=ttk.Scale(controls,from_=0,to=4,variable=self.timeline,command=self.slider_changed)
        self.slider.grid(row=0,column=4,sticky='ew',padx=5)
        ttk.Label(controls,textvariable=self.time_label,width=36).grid(row=0,column=5,padx=4)
        speed=ttk.Combobox(controls,textvariable=self.speed,state='readonly',values=['0.1×','0.25×','0.5×','1×','2×'],width=7)
        speed.grid(row=0,column=6);speed.bind('<<ComboboxSelected>>',lambda _:setattr(self.clock,'speed',float(self.speed.get()[:-1])))

    def task(self, work, done, message):
        if self.busy:
            self.status.set('当前文件任务尚未完成，请稍候。');return
        self.clock.pause();self.busy=True;self.status.set(message)
        def progress(text):self.tasks.put(('progress',text))
        def run():
            try:self.tasks.put(('done',done,work(progress)))
            except Exception as exc:self.tasks.put(('error',str(exc)))
        threading.Thread(target=run,daemon=True).start()

    def load(self, path):
        def work(progress):
            progress('读取包/真实连接元数据；不运行模型…')
            source=open_result(path,self.project_root)
            try:
                position=next(iter(source.manifest['positions']))
                return source,ResponseData(source,position,progress)
            except BaseException:source.close();raise
        def done(value):
            if self.source:self.source.close()
            self.source,self.data=value;self.position_combo.configure(values=list(self.source.manifest['positions']))
            self.position.set(self.data.position);self.loaded()
        self.task(work,done,'正在读取完整响应…')

    def loaded(self):
        if self.curve_window is not None and self.curve_window.winfo_exists():self.curve_window.destroy()
        self.curve_window=None
        self.clock=PlaybackClock(self.data.stop);self.clock.speed=float(self.speed.get()[:-1])
        self.slider.configure(to=self.data.stop);self.timeline.set(0)
        self.refresh_targets()
        n=sum(self.data.populations[p]['count'] for p in RGC if p in self.data.populations)
        self.status.set(f'已加载 {self.data.position} · {n}个真实RGC组 · {self.data.stop:g}s · 只读回放')
        self.coverage.set('真实连接祖先含反馈/电耦合闭包，范围可能较大。点为代表组/功能池；成员不是独立记录。固定S0外边界、Rod工程背景、Sbi高基线等限制保留。')
        self.render()

    def change_position(self, _=None):
        if not self.source:return
        position=self.position.get()
        def done(value):self.data=value;self.loaded()
        self.task(lambda progress:ResponseData(self.source,position,progress),done,'读取另一位置的原始响应…')

    def refresh_targets(self):
        if not self.data:return
        self.targets=[node for node in self.data.rgcs if self.rgc_type.get() in ('全部类型',node[0])]
        self.target_labels=[f"{p} [{i}] · {self.data.geometry[p]['ids'][i]} · {self.data.geometry[p]['members'][i]}成员" for p,i in self.targets]
        self.rgc_combo.configure(values=self.target_labels)
        if self.targets:
            i=self.targets.index(self.target) if self.target in self.targets else 0
            self.rgc.set(self.target_labels[i]);self.choose_target()

    def choose_target(self, _=None):
        if not self.data or self.rgc.get() not in self.target_labels:return
        self.target=self.targets[self.target_labels.index(self.rgc.get())];self.cell=self.target
        network,self.direct=self.data.network(self.target)
        self.network={name:set(rows.tolist()) for name,rows in network.items()}
        self.render()

    def render(self):
        if self.closed or not self.data or self.target is None:return
        seconds=self.clock.position;index=latest_index(self.data.time,seconds)
        frame=Image.fromarray(self.data.frame_at(seconds));size=(max(220,self.video.winfo_width()),max(140,self.video.winfo_height()))
        frame.thumbnail(size,Image.Resampling.LANCZOS)
        self.photos['video']=ImageTk.PhotoImage(frame,master=self.root)
        self.video.delete('all');self.video.create_image(size[0]/2,size[1]/2,image=self.photos['video'])
        for canvas,caption,populations in self.cards:
            image,points,label=spatial_image(self.data,populations,index,self.network,self.direct,self.target,self.cell,
                (canvas.winfo_width(),canvas.winfo_height()),self.delta.get(),self.all_patch.get(),self.rate.get() and populations==RGC)
            self.photos[canvas]=ImageTk.PhotoImage(image,master=self.root);self.maps[canvas]=points
            canvas.delete('all');canvas.create_image(0,0,image=self.photos[canvas],anchor='nw');caption.set(label)
        self.timeline.set(seconds)
        self.time_label.set(f'{seconds:.3f} / {self.data.stop:.3f} s · 样本 {self.data.time[index]:.3f}s')
        self.play_button.configure(text='暂停' if self.clock.playing else '播放')
        self.update_detail(index)
        self.render_curve()

    def update_detail(self,index):
        if self.cell is None:return
        p,row=self.cell;meta=self.data.populations[p];key=meta['primary']
        text=self.data.identity_text(p,row)
        if key in self.data.values:
            text+=f"\n状态：{key.split(':')[-1].split('__')[-1]} = {self.data.values[key][row,index]:.7g} {meta['fields'][key]}"
        text+=f'\n与目标：'+('真实连接祖先/目标' if row in self.network.get(p,set()) else 'patch对照，非所选祖先')
        self.detail.set(text)

    def click_cell(self,canvas,event):
        points=self.maps.get(canvas,[])
        if not points:return
        point=min(points,key=lambda item:(item[0]-event.x)**2+(item[1]-event.y)**2)
        if (point[0]-event.x)**2+(point[1]-event.y)**2>100:return
        self.cell=(point[2],point[3]);self.render()

    def toggle_play(self):
        if not self.data:return
        self.clock.pause() if self.clock.playing else self.clock.play();self.render()

    def seek(self,seconds):
        self.clock.pause();self.clock.seek(seconds);self.render()

    def step(self,direction):
        if self.data:self.seek(self.clock.position+direction/self.source.manifest['stimulus']['fps'])

    def slider_changed(self,value):
        if abs(float(value)-self.clock.position)>1e-7:self.seek(float(value))

    def _tick(self):
        if self.closed:return
        while True:
            try:item=self.tasks.get_nowait()
            except queue.Empty:break
            if item[0]=='progress':self.status.set(item[1])
            elif item[0]=='error':self.busy=False;self.status.set('加载/导出失败：'+item[1])
            else:
                self.busy=False
                try:item[1](item[2])
                except Exception as exc:self.status.set('界面绑定失败：'+str(exc))
        if self.clock.playing:self.clock.advance();self.render()
        self.timer=self.root.after(50,self._tick)

    def choose_package(self):
        path=filedialog.askopenfilename(parent=self.root,title='导入完整响应结果包或manifest',filetypes=[('响应ZIP','*.zip'),('JSON manifest','*.json')])
        if path:self.load(path)

    def choose_directory(self):
        path=filedialog.askdirectory(parent=self.root,title='选择包含PREPARATION.json的已完成响应目录')
        if path:self.load(path)

    def choose_video(self):
        path=filedialog.askopenfilename(parent=self.root,title='导入刺激视频（先匹配已生成响应）',filetypes=[('视频','*.avi *.mp4 *.mov *.mkv'),('全部','*.*')])
        if not path:return
        def work(progress):
            progress('按原视频SHA匹配结果，尚未启动计算…');digest=sha(path)
            from .response_bundle import read
            for package in (self.project_root/'reports/result_packages').glob('*.zip'):
                import zipfile
                with zipfile.ZipFile(package) as archive:
                    info=archive.getinfo('manifest.json')
                    if info.file_size>8*2**20:continue
                    if json.loads(archive.read(info)).get('stimulus',{}).get('original_sha256')==digest:return package
            optical=self.project_root/'reports/cone_video_113_4s_eccentric_aperture_0_6deg_20261029_v1/capture_plan.json'
            if optical.is_file() and read(optical).get('media_sha256')==digest and (self.project_root/DEFAULT_RESULT/'FINAL_RESULT.json').is_file():
                return self.project_root/DEFAULT_RESULT
            return None
        def done(result):
            if result:self.load(result)
            else:
                self.clock.pause()
                if self.source:self.source.close()
                self.source=self.data=None;self.target=self.cell=None
                for canvas,caption,_ in self.cards:
                    canvas.delete('all');canvas.create_text(12,30,text='无匹配响应 · 需要计算',fill='#cbd5e1',anchor='w');caption.set('未生成状态；不显示旧结果')
                self.video.delete('all');self.video.create_text(12,40,text=Path(path).name+'\n需计算完整响应',fill='#cbd5e1',anchor='w')
                self.detail.set('当前刺激没有完整响应。旧热图已清空。')
                self.status.set('刺激已导入，但没有匹配的完整响应：需要计算；未生成热图，未自动启动计算。')
                messagebox.showinfo('需要计算','该视频没有匹配的完整响应。\n本界面不会静默启动长计算。\n既有冻结113.AVI协议可由 tools/run_external_video_pipeline_v1.py 执行；其他刺激需先由现有计算入口生成相应响应。',parent=self.root)
        self.task(work,done,'匹配刺激视频…')

    def export(self):
        if not self.source:return
        path=filedialog.asksaveasfilename(parent=self.root,title='分类导出完整响应ZIP（保留原数据）',defaultextension='.zip',
            initialdir=self.project_root/'reports',initialfile='retina_113_4s_'+self.source.manifest['run_id'][:12]+'.zip',filetypes=[('响应包','*.zip')])
        if path:self.task(lambda progress:export_package(self.source,path,progress),lambda result:self.status.set('已导出：'+str(result)),'正在分类打包…')

    def show_curve(self):
        if not self.data or not self.cell:return
        if self.curve_window is not None and self.curve_window.winfo_exists():self.curve_window.destroy()
        window=tk.Toplevel(self.root);window.title('原始响应曲线 · 与主视图同一时间');window.geometry('850x420');self.curve_window=window
        p,row=self.cell;self.curve_cell=(p,row);self.curve_data=self.data
        keys=list(self.data.available_fields[p]);self.curve_key.set(self.data.populations[p]['primary'] or keys[0])
        combo=ttk.Combobox(window,textvariable=self.curve_key,values=keys,state='readonly',width=85);combo.pack(fill='x',padx=8,pady=5)
        self.curve_canvas=tk.Canvas(window,bg='#f8fafc');self.curve_canvas.pack(fill='both',expand=True)
        combo.bind('<<ComboboxSelected>>',lambda _:self.load_curve_field())
        self.curve_canvas.bind('<Configure>',lambda _:self.render_curve());self.render_curve()

    def load_curve_field(self):
        if self.curve_data is not self.data:return
        p,_=self.curve_cell;key=self.curve_key.get()
        self.task(lambda progress:self.data.field(p,key),lambda _:self.render_curve(),'读取已保存状态曲线…')

    def render_curve(self):
        if self.curve_window is None or not self.curve_window.winfo_exists() or self.curve_data is not self.data:return
        p,row=self.curve_cell;key=self.curve_key.get()
        if key not in self.data.values:return
        unit=self.data.populations[p]['fields'][key];factor=1000 if unit=='V' else 1
        events=[event['time_seconds'] for event in self.data.events if event['type']==p and event['cell_index']==row]
        image=trace_image(self.data.time,self.data.values[key][row]*factor,self.clock.position,
            (self.curve_canvas.winfo_width(),self.curve_canvas.winfo_height()),f'{p} [{row}] · {key}', 'mV' if unit=='V' else unit,events)
        self.photos['curve']=ImageTk.PhotoImage(image,master=self.root);self.curve_canvas.delete('all');self.curve_canvas.create_image(0,0,image=self.photos['curve'],anchor='nw')

    def text_window(self,title,text):
        window=tk.Toplevel(self.root);window.title(title);window.geometry('900x600')
        box=tk.Text(window,wrap='word',font=('Microsoft YaHei UI',10));box.pack(fill='both',expand=True)
        box.insert('1.0',text);box.configure(state='disabled')

    def show_details(self):
        if not self.data or not self.cell:return
        p,row=self.cell;index=latest_index(self.data.time,self.clock.position)
        lines=[self.data.identity_text(p,row),'',f'原保存样本：{self.data.time[index]:.9f} s','已加载状态：']
        for key,unit in self.data.available_fields[p].items():
            lines.append(f'{key}: '+(f'{self.data.values[key][row,index]:.10g} {unit}' if key in self.data.values else f'已保存，曲线面板可读取 / {unit}'))
        self.text_window('细胞组身份、单位和状态','\n'.join(lines))

    def show_provenance(self):
        if self.source:self.text_window('运行来源与适用范围',json.dumps({k:v for k,v in self.source.manifest.items() if k not in ('files','positions')},ensure_ascii=False,indent=2)+'\n\n实际连接是保留图中的非零路由祖先，含反馈与电耦合；不表示每条连接在此视频中都有可分离因果贡献。')

    def open_cell6(self):
        from .replay import ReplayApp
        window=tk.Toplevel(self.root);ReplayApp(window,self.project_root)

    def close(self):
        self.closed=True;self.clock.pause();self.root.after_cancel(self.timer)
        if self.source:self.source.close()
        self.root.destroy()


def open_response_window(root, project_root):
    window=tk.Toplevel(root)
    return ResponseApp(window,project_root)
