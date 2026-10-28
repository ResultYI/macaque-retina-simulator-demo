"""Incremental cached-to-online/GUI acceptance; no training or retinal solve."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="1"
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
from retina_platform.replay_data import ReplayData
from retina_platform.temporal_output import (
    ConditionalTemporalOutput, EXPERIMENTAL, PRODUCTION, PHASE, MODEL_ID,
    binding_for_replay, sha)
from retina_sim.temporal_readout_v1 import load_model

HERE=ROOT/"reports/cell6_temporal_output_integration_20261028_v1/validation_r2"
FROZEN=ROOT/"reports/cell6_temporal_readout_v1_20261028"


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+".writing")
    t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,
        default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+"\n",encoding="utf8")
    t.replace(p)


def prepare():
    if (HERE/"CONTRACT.json").exists():raise ValueError("Integration contract already frozen")
    scientific=read(ROOT/"reports/rgc_stage1_complete_v1/CONTRACT.json")["code_pins"]
    protected={p:sha(ROOT/p) for p in scientific}
    paths=[FROZEN/name for name in ("MODEL.json","CONTRACT.json","FIT.json","NECESSARY_CHECKS.json",
        "FINAL_PARAMETER_ON_CHECK.json","RESULT.json","g3_CURVES.npz","REPORT.md")]
    paths += [ROOT/"tools/run_temporal_readout_v1.py",ROOT/"retina_sim/temporal_readout_v1.py",
        ROOT/"reports/cell6_replay/predictions/manifest.json",
        ROOT/"reports/cell6_replay/predictions/predictions.npz",
        ROOT/"reports/rgc_stage1_cpu_parallel_full_v1/CACHE_REGISTRY.json",
        ROOT/"run_retina_gui.cmd",ROOT/"reports/cell6_bc_time_calibration_20261028_v1/REQUEST_STOP"]
    protected.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    for p,expected in scientific.items():
        if protected[p]!=expected:raise ValueError("Original scientific source changed: "+p)
    files=["retina_platform/temporal_output.py","retina_platform/replay.py",
           "tools/export_temporal_output_v1.py","tools/check_temporal_output_v1.py",
           "configurations/calibration/temporal_output_adapter_v1.json"]
    contract=dict(schema="temporal-output-integration-1",frozen_utc=datetime.now(timezone.utc).isoformat(),
        model_id=MODEL_ID,protected=protected,implementation={p:sha(ROOT/p) for p in files},
        existing_checks_reused=str(FROZEN/"NECESSARY_CHECKS.json"),
        scope="g3 original cache -> causal online output, irregular chunks, new-process readout restore, actual GUI switch; no new prediction scoring",
        revision="r2 binds exact frozen cache payload SHA in addition to model/case identity; GUI waits for actual physical voltage field; r1 receipts preserved in parent directory",
        numerical_budget=dict(probability_max_abs=1e-12,expected_rate_Hz_max_abs=1e-9,
            final_state_max_abs=1e-12,source="existing frozen temporal trial chunk/restore checks; rate is p/.001"),
        discrete="bin indices/count/order and source time grid exact, no missing/duplicate bin at nonaligned restoration",
        phase=PHASE,
        source_semantics="initial right snapshot before initial_commit; subsequent authored100us snapshot before rod/BC/RGC commit, left at coincident input frame else continuous right; not postcommit history",
        output="1ms bin-start probabilities; terminal native7.1s has no following output bin",
        initial="frozen shared a0/pi; source first1s recorded context; no evaluation normalization or warmup re-estimate",
        GUI="real Tk __main__ dispatch, default original, target experimental overlay, non-target refusal, toggle back; screenshot and no simulation",
        cases_used=[3],new_training=0,new_retina_forwards=0,FIT_READY=False,
        qualification="INCONCLUSIVE preserved; explicit experimental use does not upgrade gates",
        production_default=PRODUCTION,experimental_opt_in=EXPERIMENTAL)
    write(HERE/"CONTRACT.json",contract)
    print(json.dumps(dict(stage="INTEGRATION_CONTRACT_FROZEN",protected_files=len(protected))),flush=True)


def check():
    c=read(HERE/"CONTRACT.json")
    for p,h in {**c["protected"],**c["implementation"]}.items():
        if sha(ROOT/p)!=h:raise ValueError("Frozen integration pin changed: "+p)
    if (ROOT/"reports/ac_vs_redesign_v1/ACTIVE_WORKER.json").exists():
        raise ValueError("Unexpected full-model worker; do not overlap")
    return c


def data_and_binding():
    d=ReplayData(ROOT).load_case("cell6_g3")
    return d,binding_for_replay(d,d.nodes[d.target])


def child():
    check();d,binding=data_and_binding();state=read(HERE/"STATE_AT_35437.json")
    online=ConditionalTemporalOutput(ROOT,binding,mode=EXPERIMENTAL,state=state)
    start=online.readout.native_index
    v=d.arrays["parasol_RGC_ON:V"][:-1,1]
    chunks=[];ids=[];times=[]
    while start<len(v):
        stop=min(start+977,len(v))
        out=online.push(v[start:stop],d.time[start:stop],phase=PHASE)
        chunks.append(out["probability"]);ids.append(out["bin_index"]);times.append(out["time_source_s"])
        start=stop
    with (HERE/"RESTORED_TAIL.npz").open("xb") as stream:
        np.savez_compressed(stream,probability=np.concatenate(chunks),bin_index=np.concatenate(ids),
                            time_source_s=np.concatenate(times))
    write(HERE/"RESTORED_FINAL_STATE.json",online.state())
    print(json.dumps(dict(stage="NEW_PROCESS_RESTORE_COMPLETE",pid=os.getpid(),native_stop=start)),flush=True)


def core_check():
    c=check();d,binding=data_and_binding()
    online=ConditionalTemporalOutput(ROOT,binding,mode=EXPERIMENTAL)
    v=d.arrays["parasol_RGC_ON:V"][:-1,1]
    with np.load(FROZEN/"g3_CURVES.npz",allow_pickle=False) as z:reference=z["T_probability"].copy()
    start=0;stop_at=35437;prob=[];ids=[];times=[];sizes=[1,9,13,37,1000,193]
    while start<stop_at:
        stop=min(start+sizes[len(prob)%len(sizes)],stop_at)
        out=online.push(v[start:stop],d.time[start:stop],phase=PHASE)
        prob.append(out["probability"]);ids.append(out["bin_index"]);times.append(out["time_source_s"])
        start=stop
    write(HERE/"STATE_AT_35437.json",online.state())
    began=time.perf_counter()
    run=subprocess.run([sys.executable,"-B","-X","utf8",str(Path(__file__)),"restore-child"],
        cwd=ROOT,capture_output=True,text=True)
    (HERE/"restore_child.log").write_text(run.stdout+run.stderr,encoding="utf8")
    if run.returncode:raise RuntimeError("Readout restore child failed; see restore_child.log")
    with np.load(HERE/"RESTORED_TAIL.npz",allow_pickle=False) as z:
        p=np.r_[np.concatenate(prob),z["probability"]]
        indices=np.r_[np.concatenate(ids),z["bin_index"]]
        t=np.r_[np.concatenate(times),z["time_source_s"]]
    batch=ConditionalTemporalOutput(ROOT,binding,mode=EXPERIMENTAL)
    whole=batch.push(v,d.time[:-1],phase=PHASE)
    state=read(HERE/"RESTORED_FINAL_STATE.json")
    end=batch.state()
    state_error=max(abs(end["readout"]["a"]-state["readout"]["a"]),
        float(np.max(abs(np.asarray(end["readout"]["history_probability"])-state["readout"]["history_probability"]))))
    p_error=float(np.max(abs(p-reference)))
    batch_error=float(np.max(abs(whole["probability"]-reference)))
    rejection={}
    for kind in ("target","stimulus","payload","phase","sampled"):
        b=dict(binding);kwargs={}
        if kind=="target":b["population"]="midget_RGC_OFF"
        if kind=="stimulus":b["stimulus_sha256"]="wrong"
        if kind=="payload":b["payload_sha256"]="wrong"
        if kind=="phase":b["phase"]="post_commit"
        if kind=="sampled":kwargs["output_kind"]="sampled_spikes"
        try:ConditionalTemporalOutput(ROOT,b,mode=EXPERIMENTAL,**kwargs)
        except ValueError:rejection[kind]=True
        else:rejection[kind]=False
    oldstate=json.dumps(online.state(),sort_keys=True)
    try:online.push(v[start:start+1],d.time[start:start+1],phase="post_commit")
    except ValueError:rejection["wrong_push_phase_no_mutation"]=json.dumps(online.state(),sort_keys=True)==oldstate
    else:rejection["wrong_push_phase_no_mutation"]=False
    try:
        corrupted=dict(state);corrupted["binding_id"]="wrong"
        ConditionalTemporalOutput(ROOT,binding,mode=EXPERIMENTAL,state=corrupted)
    except ValueError:rejection["wrong_restore_identity"]=True
    else:rejection["wrong_restore_identity"]=False
    # The underlying default loader must still refuse this inconclusive artifact.
    try:load_model(ROOT/"reports/cell6_temporal_readout_v1_20261028/MODEL.json")
    except ValueError:rejection["ordinary_loader_still_refuses_inconclusive"]=True
    else:rejection["ordinary_loader_still_refuses_inconclusive"]=False
    result=dict(passed=p_error<=1e-12 and batch_error<=1e-12 and state_error<=1e-12
        and np.array_equal(indices,np.arange(7100)) and np.array_equal(t,np.arange(7100)*.001)
        and all(rejection.values()),model_id=MODEL_ID,case="cell6_g3",restored_native_cursor=35437,
        probability_max_abs_error=p_error,batch_probability_max_abs_error=batch_error,
        expected_rate_Hz_max_abs_error=p_error/.001,
        final_state_max_abs_error=state_error,final_native_cursor=end["readout"]["native_index"],
        exact_bin_count_and_order=bool(np.array_equal(indices,np.arange(7100))),
        exact_bin_times=bool(np.array_equal(t,np.arange(7100)*.001)),rejections=rejection,
        existing_causality_history_initial_tests_reused=True,observed_history_supplied=False,
        source_voltage_not_modified=True,new_process_restore_wall_seconds=time.perf_counter()-began,
        metadata=batch.metadata(),retina_forwards=0,training_runs=0)
    write(HERE/"CONSISTENCY_RESULT.json",result)
    if not result["passed"]:raise RuntimeError("Online adapter failed original budget")
    print(json.dumps(dict(stage="CACHE_ONLINE_RESTORE_PASS",max_probability_error=p_error)),flush=True)


def gui_check():
    import tkinter as tk
    from PIL import ImageGrab
    from unittest.mock import patch
    from retina_platform.replay import ReplayApp
    from retina_platform import __main__ as entry
    captured={};real=ReplayApp
    def create(root,project_root,registry=None):
        app=real(root,project_root,registry);captured["app"]=app;return app
    def scenario(root):
        app=captured["app"]
        try:
            # Async aligned display is optional for this output-only check.
            root.update()
            if app.data is None:raise AssertionError(app.status_var.get())
            if app.data.case_id!="cell6_g3":
                app.case_var.set("cell6_g3");app.load_case()
            original_voltage=app.data.arrays["parasol_RGC_ON:V"].copy()
            original_prediction=app.data.prediction.copy()
            assert app.output_mode_var.get()==PRODUCTION and app.experimental_prediction is None
            app.output_combo.current(1);app.output_combo.event_generate("<<ComboboxSelected>>")
            root.update()
            assert app.output_mode_var.get()==EXPERIMENTAL and app.experimental_prediction is not None
            with np.load(FROZEN/"g3_CURVES.npz",allow_pickle=False) as z:
                error=float(np.max(abs(app.experimental_prediction-z["T_probability"]/.001)))
            assert error<=1e-9
            assert "INCONCLUSIVE" in app.output_notice_var.get()
            # Wait only for the existing UI's asynchronous read-only field loading.
            deadline=time.monotonic()+20
            while (app.aligned is None or app._aligned_ready is None
                    or app._aligned_ready!=app._aligned_desired) and time.monotonic()<deadline:
                root.update();time.sleep(.02)
            assert app.aligned is not None and app._aligned_ready==app._aligned_desired,"Aligned field still pending"
            app.seek(2.3);root.update();app._plot_bases=None;app.render();root.update()
            image=np.asarray(app._plot_bases[1][0])
            assert np.any(np.all(image==np.array([126,34,206]),axis=2)),"Experimental purple curve/legend absent"
            assert np.any(np.all(image==np.array([37,99,235]),axis=2)),"Original production blue curve absent"
            ImageGrab.grab(window=int(root.frame(),16)).save(HERE/"experimental_mode.png")
            other=next(i for i,n in enumerate(app.data.nodes) if n["population"]=="midget_RGC_OFF")
            app.node_var.set(app.node_labels[other]);app.select_node();root.update()
            assert app.experimental_prediction is None and "不适用" in app.output_notice_var.get()
            app.node_var.set(app.node_labels[app.data.target]);app.select_node();root.update()
            assert app.experimental_prediction is not None
            app.output_combo.current(0);app.output_combo.event_generate("<<ComboboxSelected>>");root.update()
            assert app.output_mode_var.get()==PRODUCTION and app.experimental_prediction is None
            assert np.array_equal(original_voltage,app.data.arrays["parasol_RGC_ON:V"])
            assert np.array_equal(original_prediction,app.data.prediction)
            result=dict(passed=True,actual_Tk_window=True,actual_main_entry_dispatch=True,
                default_mode=PRODUCTION,experimental_mode_explicit=True,
                displayed_qualification="INCONCLUSIVE",frozen_curve_max_Hz_error=error,
                non_target_explicitly_refused=True,original_voltage_and_prediction_unchanged=True,
                screenshot=str(HERE/"experimental_mode.png"),full_model_runs=0)
            write(HERE/"GUI_RESULT.json",result)
        finally:app.close()
    with patch.object(entry,"__name__",entry.__name__),patch("retina_platform.replay.ReplayApp",side_effect=create), \
         patch.object(tk.Tk,"mainloop",scenario), \
         patch.object(sys,"argv",["retina_platform","--project-root",str(ROOT),"--cell6-replay"]):
        entry.main()
    print(json.dumps(dict(stage="ACTUAL_GUI_PASS")),flush=True)


def finalize():
    c=check();r=read(HERE/"CONSISTENCY_RESULT.json");g=read(HERE/"GUI_RESULT.json")
    report=[
        "# temporal_readout_v1 实验输出接入验收","",
        "工程接入 PASS；候选科学结论仍为 INCONCLUSIVE，不升级 FIT_READY=false 或原 absolute monitor=false。","",
        "新增独立输出适配与GUI显式下拉选择。原生产/既有读出为每次启动默认；实验输出为单独紫色曲线，物理膜电压仍为原曲线。其他类型/节点、输入/模型/单位/时间/phase身份不匹配时明确拒绝，不静默套用。","",
        "冻结模型ID："+MODEL_ID+"；精确 tau_s="+str(r["metadata"]["tau_s"])+"，未取整、未训练、未重算标准化。","",
        "验收合同先冻结于 "+c["frozen_utc"]+"；只新增适配检查，复用原有因果/梯度/初态/分块检查，不重复生理审计。","",
        "g3原100μs膜缓存以不规则分块消费，在native索引35437（非1ms边界）保存后由新进程恢复；总7100个1ms bin索引/时刻逐值一致，无重复或漏bin。","",
        "与冻结离线预测比较，最大概率误差="+str(r["probability_max_abs_error"])+"，最大期望率误差="+str(r["expected_rate_Hz_max_abs_error"])+"Hz；整段适配误差="+str(r["batch_probability_max_abs_error"])+"；最终a/pi误差="+str(r["final_state_max_abs_error"])+"。原预算为概率/状态1e-12、期望率1e-9Hz，没有放宽。","",
        "相位：原compact cache在initial_commit前记录初始right节点，后续在rod/BC/RGC离散commit前记录，遇输入frame取left否则连续right。输出使用对应1ms bin-start；a尚未消费当前100μs x。不是100μs post-history状态。终端7.1s没有后续bin，不输入读出。","",
        "自由期望采用原八态算法，三个严格过去bin历史被边缘化，不提供真实spike参数。期望计数率=p/1ms，λ是危险率。当前适配不支持采样spike并明确拒绝；不创建RNG或伪造事件。","",
        "共享原灰背景初始a/pi与首1s真实记录上下文；不重新定背景，不将前1s声称恒定灰底，不从评估组估计mean/sd。保存状态仅为读出状态，不是全视网膜checkpoint。","",
        "实际Tk窗口、原__main__ --cell6-replay入口已运行：默认原模式、显式切换、紫色实验/蓝色原输出并存、其他类型拒绝、切回默认、原膜及原拟合曲线不变均通过。截图 experimental_mode.png。GUI可用不等同生理拟合通过。","",
        "运行方式：双击 run_retina_gui.cmd，在“放电输出”选择实验 temporal_readout_v1。选择Cell6目标parasol_RGC_ON列1；其他节点会提示不适用。操作详见 README.md。","",
        "全部原51科学pins、冻结模型/参数/合同/离线结果、原曲线文件、原STOP保持；没有全链前向、训练、历史test读取、Git、上传、UI自动采用或扩展其他偏心度。","",
        "完成后停止。INCONCLUSIVE的g5误差区间问题、生理标定/输入与边界未知均没有被本次工程接入消除。"]
    (HERE/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf8")
    write(HERE/"FINAL_RECEIPT.json",dict(status="PASS_AND_STOPPED",utc=datetime.now(timezone.utc).isoformat(),
        model_id=MODEL_ID,qualification="INCONCLUSIVE",consistency_passed=r["passed"],actual_GUI_passed=g["passed"],
        original_protected_files_unchanged=len(c["protected"]),new_retina_forwards=0,new_training=0,
        BC_STOP_retained=True,production_default_changed=False,FIT_READY=False,absolute_monitor_false_preserved=True,
        report=str(HERE/"REPORT.md"),report_sha256=sha(HERE/"REPORT.md"),
        wall_seconds=time.perf_counter()-BEGAN,active_worker_retained=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","run","restore-child"])
    args=parser.parse_args()
    if args.command=="prepare":prepare();return
    if args.command=="restore-child":child();return
    check()
    if (HERE/"FINAL_RECEIPT.json").exists():raise ValueError("Completed integration; no repeated acceptance")
    try:
        core_check();gui_check();finalize()
        print(json.dumps(dict(stage="INTEGRATION_COMPLETE",report=str(HERE/"REPORT.md"))),flush=True)
    except BaseException as exc:
        write(HERE/("FAILURE_"+datetime.now(timezone.utc).strftime("%H%M%S")+".json"),
              dict(reason=str(exc),traceback=traceback.format_exc(),retina_forwards=0,training_runs=0))
        raise


BEGAN=time.perf_counter()
if __name__=="__main__":main()
