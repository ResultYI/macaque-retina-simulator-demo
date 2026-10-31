"""Exercise the actual Tk app with the real exported ZIP; no model execution."""
from pathlib import Path
import json
import sys
import tempfile
import time
import traceback
import tkinter as tk
import ctypes
from ctypes import wintypes
from PIL import ImageGrab

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_platform.response_view import ResponseApp
from retina_platform import response_view
from retina_platform.response_data import latest_index

OUT=ROOT/'reports/response_viewer_delivery_v1_20261030';OUT.mkdir(exist_ok=True)
PACKAGE=ROOT/'reports/result_packages/retina_113_4s_298fa2ef1524.response.zip'
errors=[];checks={};started=time.perf_counter()
root=tk.Tk()
root.report_callback_exception=lambda *args:errors.append(''.join(traceback.format_exception(*args)))
app=ResponseApp(root,ROOT,result=PACKAGE)

def wait_for(predicate,limit=160):
    deadline=time.monotonic()+limit;last=0
    while not predicate():
        root.update();time.sleep(.02)
        if time.monotonic()-last>5:
            print(app.status.get(),flush=True);last=time.monotonic()
        if errors:raise RuntimeError(errors[-1])
        if time.monotonic()>deadline:raise TimeoutError(app.status.get())
    root.update()

def capture(name,window=None):
    window=window or root
    window.attributes('-topmost',True);window.lift();root.update();time.sleep(.2);root.update()
    # Tk reports DPI-virtualized coordinates; DWM returns the actual physical
    # window rectangle used by ImageGrab. This changes no system/app setting.
    handle=ctypes.windll.user32.GetParent(window.winfo_id()) or window.winfo_id()
    rectangle=wintypes.RECT()
    result=ctypes.windll.dwmapi.DwmGetWindowAttribute(handle,9,ctypes.byref(rectangle),ctypes.sizeof(rectangle))
    if result:raise OSError('Cannot obtain own native window capture rectangle')
    ImageGrab.grab(bbox=(rectangle.left,rectangle.top,rectangle.right,rectangle.bottom)).save(OUT/name)
    window.attributes('-topmost',False)

try:
    wait_for(lambda:app.data is not None and not app.busy)
    assert app.status.get().startswith('已加载'),app.status.get()
    checks['real_zip_import']=True;checks['near_RGC_groups']=len(app.data.rgcs)
    assert len(app.data.rgcs)==55
    assert not any(str(path).startswith(str(ROOT)) for path in app.source.files.values())
    checks['playback_uses_extracted_package_only']=True
    assert all(canvas.winfo_width()>190 for canvas,_,_ in app.cards)
    assert app.play_button.winfo_ismapped() and app.slider.winfo_ismapped()
    assert app.play_button.winfo_rooty()+app.play_button.winfo_height()<=root.winfo_rooty()+root.winfo_height()
    app.seek(1.1);root.update()
    assert abs(app.clock.position-1.1)<1e-12 and app.data.frame_at(1.1).shape==(360,640,3)
    checks['shared_seek_and_original_frames']=True
    assert app.network[app.target[0]]=={app.target[1]}
    checks['target_specific_topology']=dict((name,len(rows)) for name,rows in app.network.items())
    capture('desktop_near_layers.png')
    app.rgc_type.set('small_bistratified_S_ON');app.refresh_targets();root.update()
    assert app.target[0]=='small_bistratified_S_ON'
    assert app.network['small_bistratified_S_ON']=={app.target[1]}
    assert not app.network['midget_RGC_ON']
    checks['changing_RGC_recomputes_real_ancestors']=True
    first=app.clock.position;app.step(1);root.update()
    assert abs(app.clock.position-first-1/30)<1e-10
    app.clock.speed=2.;app.toggle_play();end=time.monotonic()+.25
    while time.monotonic()<end:root.update();time.sleep(.01)
    app.toggle_play();assert app.clock.position>first+.2
    checks['play_pause_frame_step_speed']=True
    # Real cell click dispatch, including identity and units.
    canvas=app.cards[2][0];point=app.maps[canvas][0]
    from types import SimpleNamespace
    app.click_cell(canvas,SimpleNamespace(x=point[0],y=point[1]));root.update()
    assert app.cell==(point[2],point[3]) and '坐标' in app.detail.get()
    app.show_curve();root.update();app.render_curve();assert 'curve' in app.photos
    checks['cell_click_and_separate_curve']=True
    capture('desktop_cell_curve.png',app.curve_window)
    app.curve_window.destroy();app.curve_window=None
    app.delta.set(False);app.rate.set(True);app.all_patch.set(True);app.render();root.update()
    checks['absolute_rate_and_context_views']=True
    app.position.set('far');app.change_position()
    wait_for(lambda:not app.busy and app.data.position=='far')
    assert len(app.data.rgcs)==42;checks['far_RGC_groups']=42
    app.delta.set(True);app.rate.set(False);app.all_patch.set(False);app.seek(2.8);root.update()
    capture('desktop_far_layers.png')
    root.geometry('1160x820');root.update();app.render();root.update()
    assert min(canvas.winfo_width() for canvas,_,_ in app.cards)>190
    assert app.play_button.winfo_ismapped() and app.slider.winfo_ismapped()
    assert app.play_button.winfo_rooty()+app.play_button.winfo_height()<=root.winfo_rooty()+root.winfo_height()
    checks['timeline_controls_visible_both_sizes']=True
    capture('desktop_compact_layers.png');checks['compact_1160x820']=True
    # Unknown-video path is an explicitly synthetic UI fixture, never scientific data.
    with tempfile.TemporaryDirectory() as directory:
        unknown=Path(directory)/'SYNTHETIC_UNMATCHED_VIDEO_TEST.avi';unknown.write_bytes(b'UI_TEST_NO_RESPONSE')
        old_dialog=response_view.filedialog.askopenfilename;old_message=response_view.messagebox.showinfo
        response_view.filedialog.askopenfilename=lambda **kwargs:str(unknown)
        response_view.messagebox.showinfo=lambda *args,**kwargs:None
        try:
            app.choose_video();wait_for(lambda:not app.busy)
            assert app.data is None and '需要计算' in app.status.get()
            checks['unmatched_video_clears_old_heatmaps_no_compute']=True
        finally:
            response_view.filedialog.askopenfilename=old_dialog;response_view.messagebox.showinfo=old_message
    assert not errors,errors
    checks['Tk_callbacks_without_errors']=True
    report=dict(status='PASS',checks=checks,wall_seconds=time.perf_counter()-started,model_run=False,package=str(PACKAGE),screenshots=['desktop_near_layers.png','desktop_far_layers.png','desktop_compact_layers.png','desktop_cell_curve.png'],capture_rule='DWM physical window rectangle; no DPI/system setting changed')
    (OUT/'DESKTOP_TEST.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
except BaseException:
    (OUT/'DESKTOP_TEST_FAILED.txt').write_text(traceback.format_exc()+'\n'+app.status.get()+'\n'+'\n'.join(errors),encoding='utf8')
    raise
finally:app.close()
