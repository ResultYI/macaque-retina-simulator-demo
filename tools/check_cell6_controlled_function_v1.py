"""One frozen 627ms cone-domain functional check, no fitting or data survey."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
from retina_sim.contracts import canonical_hash
from retina_sim.group_time_storage import load_bundle, save_bundle
from retina_sim.group_time_runner import GroupTimeArm
from retina_sim.group_time_monitor import GroupTimeMonitor, frozen_budget
from retina_sim.background_checkpoint import MemoryGuard
from retina_sim.temporal_readout_v1 import TemporalReadout, load_model
from tools.validate_ac_vs_native_support import load as load_source
from tools.validate_ac_vs_equilibrium_start import initial_match
from tools.validate_ac_vs_timescale import calendar
from tools.validate_ac_vs import causal_hash

OUT=ROOT/"reports/cell6_controlled_function_v1_20261028/attempt_0002"
SPEC=ROOT/"reports/rgc_stage1_complete_v1/CONTRACT.json"
MODEL=ROOT/"reports/cell6_temporal_readout_v1_20261028/MODEL.json"
LEASE=ROOT/"reports/ac_vs_redesign_v1/ACTIVE_WORKER.json"
QUAL=dict(default_adopted=False,fit_eligible=False,physiological_calibration=False)
START,STOP=4.993,5.620
EDGES=[5.100,5.200,5.360,5.460]


def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".writing")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n",encoding="utf8");tmp.replace(p)


def protected():
    spec=read(SPEC);paths=[SPEC,MODEL,Path(__file__),
        ROOT/"reports/cell6_temporal_readout_v1_20261028/CONTRACT.json",
        ROOT/"retina_sim/temporal_readout_v1.py",
        ROOT/"reports/cell6_bc_time_calibration_20261028_v1/REQUEST_STOP"]
    pins={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for path,digest in spec["code_pins"].items():
        if sha(ROOT/path)!=digest:raise ValueError("Original scientific source changed: "+path)
        pins[path]=digest
    return pins


def verify():
    c=read(OUT/"CONTRACT.json")
    for p,h in c["pins"].items():
        if sha(ROOT/p)!=h:raise ValueError("Frozen functional contract source changed: "+p)
    return c


def prepare():
    if (OUT/"CONTRACT.json").exists():raise ValueError("Function protocol already frozen")
    if LEASE.exists():raise ValueError("Another model lease exists; do not overlap")
    if (ROOT/"REQUEST_STOP").exists():raise ValueError("Project STOP retained")
    began=time.perf_counter();spec=read(SPEC)
    if sha(ROOT/spec["prepared_model"]/"initial_seed/manifest.json")!=spec["prepared_manifest_sha256"]:
        raise ValueError("Frozen prepared model changed")
    assembly,seed,parent=load_source(ROOT/spec["prepared_model"])
    if canonical_hash(seed)!=spec["source_seed_sha256"]:raise ValueError("Original source seed changed")
    if seed["time_seconds"]!=START:raise ValueError("Source time changed")
    model=load_model(MODEL,allow_unqualified=True)
    target=spec["target"];rows=spec["recording"][target["population"]]["rows"]
    row=int(rows[target["column"]])
    if str(assembly.populations[target["population"]].group_ids[row])!=target["global_id"]:
        raise ValueError("Actual target group mismatch")
    if spec["recording"][target["population"]]["members"][target["column"]]!=2:
        raise ValueError("Original two-member group required")
    background=np.asarray(seed["input_reference"]["native_background_td"],dtype=np.float64)
    if np.any(background<0) or not np.isfinite(background).all():raise ValueError("Invalid cone background")
    rod=np.asarray(seed["rod_inputs"][min(seed["rod_inputs"])],dtype=np.float64)
    old=deepcopy(seed)
    frames=list(range(int(math.floor(START*150)),int(round(STOP*150))+1))
    plus_on,plus_off,minus_on,minus_off=[int(round(t*150)) for t in EDGES]
    multipliers={f:1.05 if plus_on<=f<plus_off else .95 if minus_on<=f<minus_off else 1. for f in frames}
    seed["native_inputs"]={f:background*multipliers[f] for f in frames}
    seed["rod_inputs"]={f:rod.copy() for f in frames}
    # All initial state/history/RNG/delay/parameters remain from the same source.
    retained=[k for k in old if k not in ("native_inputs","rod_inputs")]
    if any(canonical_hash(old[k])!=canonical_hash(seed[k]) for k in retained):
        raise ValueError("Only future inputs may change")
    protocol=deepcopy(parent["protocol"])
    protocol.update(start_seconds=START,stop_seconds=STOP,checkpoint_seconds=5.300,
        input_frame_indices=frames,controlled_function_version="cell6-cone-domain-627ms-v1")
    fields={name:f for name,f in parent["selection"]["fields"].items()
            if len(f["rows"]) and f["category"] in ("states","ports")}
    scalar_count=sum(len(f["rows"]) for f in fields.values())
    count=int(round((STOP-START)/.0001))+1
    memory=MemoryGuard(16*1024**3,4*1024**3)
    if not memory.check():raise ValueError("Memory guard admission failed: "+str(memory.last))
    # Old g3 timing is evidence only; no new benchmark or old response reuse.
    old_manifest=ROOT/"reports/rgc_stage1_cpu_parallel_full_v1/workers/cell6_g3/cache/cell6_g3/manifest.json"
    metrics=read(old_manifest)["metrics"]
    integration=(metrics["native_wall_seconds"]+metrics["inner_wall_seconds"])*(STOP-START)/7.1
    c=dict(schema="cell6-controlled-cone-function-1",frozen_utc=utc(),pins=protected(),
        model_id=model["model_id"],model_sha256=sha(MODEL),target=target,target_assembly_row=row,
        original_prepared_model=spec["prepared_model"],source_initial_seed_manifest_sha256=spec["prepared_manifest_sha256"],
        source_seed_sha256=spec["source_seed_sha256"],assembly_artifact=assembly.artifact_sha256,
        initial_condition="same conditional source construction seed at4.993s, all initial states/history/RNG/HC delay retained;107ms not claimed physiological steady state",
        start_seconds=START,stop_seconds=STOP,total_seconds=.627,
        sequence=[
            dict(kind="background",start=START,stop=5.100,duration_ms=107,multiplier=1.),
            dict(kind="bright",start=5.100,stop=5.200,duration_ms=100,multiplier=1.05),
            dict(kind="background",start=5.200,stop=5.360,duration_ms=160,multiplier=1.),
            dict(kind="dark",start=5.360,stop=5.460,duration_ms=100,multiplier=.95),
            dict(kind="background",start=5.460,stop=STOP,duration_ms=160,multiplier=1.)],
        input=dict(port="cone native retinal illuminance",unit="td",rule="B_i*fixed multiplier for ALL retained native0.6 cone groups; exterior reservoir remains fixed",
            cone_group_count=len(background),background_min_td=float(background.min()),
            background_mean_td=float(background.mean()),background_max_td=float(background.max()),
            background_sha256=canonical_hash(background),minimum_input_td=float((.95*background).min()),
            exact_frame_on_off=[plus_on,plus_off,minus_on,minus_off],input_fps=150.,
            contrast_source="predeclared engineering +/-5% within existing nonnegative native Td contract; not physical Cell6 optical calibration",
            rod_unit="R*/s",rod_min=float(rod.min()),rod_max=float(rod.max()),rod_sha256=canonical_hash(rod),
            rod_modulated=False,video_or_optical_pipeline=False,
            frame_hashes={str(f):canonical_hash(v) for f,v in seed["native_inputs"].items()},
            multiplier_by_frame={str(f):v for f,v in multipliers.items()}),
        source_scope="native0.6/inner0.3 fixed exterior conditional local group; no spatial expansion/no true whole-field retina qualification",
        numerical="same reference CPU GroupTimeArm/DirectACVSRHS, FP64/DOP853/rtol1e-6/atol1e-9/100us maxstep and original event/history clocks",
        temporal_readout="exact frozen tau/mean/sd/a0/pi, no renormalization; pure voltage observer at original precommit nodes; initial observed spikes not supplied",
        windows=dict(baseline_ms=20,transient_ms=20,sustained_ms=20,recovery_tail_ms=20,
            definitions="each pulse baseline is its preceding20ms; transient first20ms; sustained last20ms; recovery final20ms of160ms"),
        criteria=dict(polarity="bright transient mean>own prebackground, dark transient mean<own prebackground;1e-9Hz readout numeric floor, not biological significance or new solver-error guarantee",
            recovery="for each deterministic state/port and output: tail difference<=max(5% peak excursion from own prebackground, original frozen state budget); rate floor1e-9Hz",
            stochastic="sampled original spike history is not required to return to identical values; separate from deterministic expected output",
            drift="report first20 vs last20 of107ms background; no steady-state assertion",
            non_recovery="report not recovered within160ms only; never extend/retry or equate with instability",
            future="reuse frozen causal implementation proof, verify online readout equals causal postprocessing of only this run V",
            no_feedback="hash actual model states, native anchor, history/RNG, clocks, HCdelay and cursors before/after observer without invoking checkpoint API at a precommit time; output object takes copied scalarV only",
            numerical_domain="retain original physical/history monitor and finite/probability domain checks; original absolute monitor failure retained"),
        recording=dict(native_target_dt_s=.0001,state_core_dt_s=.001,native_nodes=count,bins=(count-1)//10,
            fields=fields,core_scalar_columns=scalar_count,
            estimated_numeric_trace_bytes=int(math.ceil(count/10))*scalar_count*8+count*3*8,
            representation="actual core groups/contact vectors with saved row registry; no member expansion"),
        checkpoints=[5.300,STOP],protocol=protocol,resources=memory.last,
        cost=dict(old_g3_manifest=str(old_manifest.relative_to(ROOT)),old_native_seconds=metrics["native_wall_seconds"],
            old_inner_seconds=metrics["inner_wall_seconds"],old_source_seconds=7.1,
            projected_integration_seconds=integration,
            limitation="linear projection only, excluding snapshot/assembly/IO, new stimulus adaptive cost and host load; no arbitrary hard wall budget"),
        no_fit=True,no_additional_conditions=True,no_independent_data_survey=True,
        qualification="INCONCLUSIVE retained; functional engineering only",FIT_READY=False,absolute_monitor_false_preserved=True)
    c["initial_seed_sha256"]=canonical_hash(seed)
    prior=read(OUT.parent/"CONTRACT.json")
    if c["initial_seed_sha256"]!=prior["initial_seed_sha256"] or c["input"]!=prior["input"] or c["sequence"]!=prior["sequence"]:
        raise ValueError("Engineering recovery may not change the stimulus/initial state")
    c["prior_attempt"]=dict(directory=str(OUT.parent),failure=read(OUT.parent/"FAILURE.json"),
        correction="only invalid precommit checkpoint invocation replaced with pure state/history hashing; no complete pulse was evaluated before failure",
        source_seed_and_stimulus_exactly_unchanged=True,response_based_adjustment=False)
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/"CONTRACT.json",c)
    identity=dict(contract_sha256=sha(OUT/"CONTRACT.json"),stimulus_sha256=canonical_hash(c["input"]),
                  assembly_artifact=assembly.artifact_sha256,model_id=model["model_id"])
    save_bundle(OUT/"initial_seed",kind="source_construction_seed",identity=identity,payload=seed,qualification=QUAL)
    write(OUT/"PREPARATION.json",dict(identity=identity,assembly_seconds=time.perf_counter()-began,
        real_model_advanced=False,estimated_integration_seconds=integration,
        estimated_trace_bytes=c["recording"]["estimated_numeric_trace_bytes"],resources=memory.last))
    print(json.dumps(dict(stage="FROZEN_READY",start=START,stop=STOP,duration=.627,
        background_td=[float(background.min()),float(background.mean()),float(background.max())],
        projected_integration_seconds=integration,preparation_seconds=time.perf_counter()-began,
        trace_bytes=c["recording"]["estimated_numeric_trace_bytes"],available_RAM_bytes=memory.last["available_physical_bytes"])),flush=True)


class Worker:
    def __enter__(self):
        self.owner=dict(pid=os.getpid(),scope="authorized-cell6-controlled627ms-v1",utc=utc(),
                        contract_sha256=sha(OUT/"CONTRACT.json"))
        self.guard=MemoryGuard(16*1024**3,4*1024**3);self.began=time.perf_counter()
        with LEASE.open("x") as stream:json.dump(self.owner,stream)
        try:self.check()
        except BaseException:
            if LEASE.exists() and read(LEASE)==self.owner:LEASE.unlink()
            raise
        return self
    def check(self):
        if ((ROOT/"REQUEST_STOP").exists() or (OUT/"REQUEST_STOP").exists()
                or read(LEASE)!=self.owner):raise RuntimeError("Functional worker STOP/ownership")
        if not self.guard.check():raise RuntimeError("Original memory protection: "+str(self.guard.last))
    def __exit__(self,*args):
        write(OUT/"RESOURCE_RECEIPT.json",dict(owner=self.owner,elapsed_seconds=time.perf_counter()-self.began,
            peak_RSS_bytes=self.guard.peak_rss_bytes,last_memory=self.guard.last,released=True,
            arbitrary_wall_budget_used=False))
        if LEASE.exists() and read(LEASE)==self.owner:LEASE.unlink()


def run():
    if (OUT/"RESULT.json").exists():raise ValueError("One stimulus only; completed result already exists")
    c=verify();prep=read(OUT/"PREPARATION.json")
    began=time.perf_counter()
    a,old,parent=load_source(ROOT/c["original_prepared_model"])
    seed=load_bundle(OUT/"initial_seed",expected_kind="source_construction_seed",
                     expected_identity=prep["identity"])["payload"]
    if canonical_hash(seed)!=c["initial_seed_sha256"]:raise ValueError("New stimulus seed changed")
    assembly_seconds=time.perf_counter()-began
    model=load_model(MODEL,allow_unqualified=True)
    expected=TemporalReadout(model)
    target=c["target"];row=c["target_assembly_row"];protocol=c["protocol"]
    core={k:[] for k in c["recording"]["fields"]};core_times=[]
    native=[];original=[];experimental=[];prob=[];outputs_time=[];proof=[]
    state_fields=c["recording"]["fields"];count=c["recording"]["native_nodes"]
    def save_record(path):
        arrays={"native_time_source_s":np.arange(len(native))*.0001,
            "target_V_gE_gI":np.asarray(native),"output_time_source_s":np.asarray(outputs_time),
            "original_expected_count_rate_Hz":np.asarray(original),
            "experimental_expected_count_rate_Hz":np.asarray(experimental),
            "experimental_probability":np.asarray(prob),"core_time_source_s":np.asarray(core_times)}
        arrays.update({k:np.stack(v) for k,v in core.items() if v})
        np.savez_compressed(path,**arrays)
    with Worker() as worker:
        arm=GroupTimeArm(a,seed,protocol,mode="reference",guard=worker.check)
        monitor=GroupTimeMonitor(a,seed["states"],parameters=seed["inner_parameters"])
        def original_hash(s):
            anchor=arm.native_state.anchor
            return canonical_hash(dict(states=s["states"],inner=arm.inner_state,native_y=arm.native_state.y,
                anchor=dict(time_seconds=anchor.time_seconds,R=anchor.R,E=anchor.E,light=anchor.light,
                    input_stop_seconds=anchor.input_stop_seconds,input_epoch=anchor.input_epoch),
                spike_states=arm.spike_states,clocks=arm.clock_indices,last_commit_times=arm.last_commit_times,
                HC_delay=arm.delay_payload(),event_generation=arm.event_generation,
                time_seconds=arm.time_seconds,inner_time=arm.inner_time,
                native_consumed=arm.native_consumed))
        def observe(t,side):
            s=arm.snapshot(t,side,queue_node=True)
            monitor.observe(t,side,s["states"],s["rhs"],transitions=s["transitions"],
                ports=s["ports"],aux=s["aux"],event_generation=s["event_generation"])
            if monitor.domain_failures:raise RuntimeError("Original physical/history/domain monitor failed")
            return s
        def record(s,j):
            if j!=len(native):raise RuntimeError("Native sample skipped/repeated")
            values=[float(s["ports"][target["population"],p][row]) for p in ("voltage","gE","gI")]
            if not np.isfinite(values).all() or min(values[1:])<0:raise RuntimeError("Invalid actual RGC input/membrane")
            if j==0 and values[0]!=model["core"]["initial_gray_V"]:
                raise RuntimeError("Frozen temporal source gray voltage differs")
            native.append(values)
            if j<count-1:
                witness=j in (0,1070,2070,3670,4670,count-2)
                before=original_hash(s) if witness else None
                result=expected.predict(np.array([values[0]],dtype=float))
                if witness:
                    after=original_hash(s)
                    proof.append(dict(native_index=j,unchanged=before==after,before=before,after=after))
                    if before!=after:raise RuntimeError("Output observer changed original causal state")
                if len(result["probability"]):
                    outputs_time.append(j*.0001)
                    prob.append(float(result["probability"][0]))
                    experimental.append(float(result["expected_count_rate_Hz"][0]))
                    original.append(float(s["ports"][target["population"],"expected_rate_Hz"][row]))
            if j%10==0:
                core_times.append(j*.0001)
                for name,f in state_fields.items():
                    core[name].append(np.asarray(s[f["category"]][tuple(f["key"])])[f["rows"]].copy())
        write(OUT/"LIVE_HANDOFF.json",dict(pid=os.getpid(),utc=utc(),stage="RUNNING",log=str(OUT/"run_v1.log"),
            contract=str(OUT/"CONTRACT.json"),total_source_seconds=.627,output_directory=str(OUT)))
        try:
            initial,_=initial_match(arm,seed);write(OUT/"INITIAL_MATCH.json",initial)
            first=observe(arm.start,"right");record(first,0)
            arm.initial_commit();observe(arm.start,"right");arm.ack(arm.start)
            for end,flags in calendar(protocol):
                if end<=arm.inner_time:continue
                start=arm.inner_time;arm.ensure_native(end,"left");arm.advance_inner(start,end)
                before=observe(end,"left" if flags["frame"] else "right")
                j=int(round((end-arm.start)/.0001))
                tick=flags["rod"] or end==arm.stop
                if tick and j==len(native):record(before,j)
                if flags["rod"] or flags["bc"]:
                    arm.commit(end,rod=flags["rod"],bc=flags["bc"],rgc=flags["bc"],
                        rod_sample_time=max(arm.start,end-.0001+1e-13) if flags["rod"] else None)
                last=observe(end,"right");arm.ack(end)
                if tick and (j in (3070,count-1)):
                    payload=arm.checkpoint_payload(monitor=monitor.payload(),recording=dict(native_records=len(native),core_records=len(core_times),
                        experimental_readout=expected.state()))
                    save_bundle(OUT/("checkpoint_"+str(j).zfill(6)),kind="complete_model_checkpoint",
                                identity=prep["identity"],payload=payload,qualification=QUAL)
                    save_record(OUT/("compact_"+str(j).zfill(6)+".npz"))
                    write(OUT/"LATEST_CHECKPOINT.json",dict(native_index=j,source_seconds=j*.0001,
                        path=str(OUT/("checkpoint_"+str(j).zfill(6)))))
                if tick and j%500==0:
                    progress=dict(pid=os.getpid(),utc=utc(),native_nodes=len(native),source_seconds=j*.0001,
                        total_source_seconds=.627,wall_seconds=time.perf_counter()-began,
                        checkpoint=read(OUT/"LATEST_CHECKPOINT.json") if (OUT/"LATEST_CHECKPOINT.json").exists() else None)
                    write(OUT/"PROGRESS.json",progress);print(json.dumps(dict(stage="PROGRESS",**progress)),flush=True)
            if len(native)!=count or len(prob)!=(count-1)//10:
                raise RuntimeError("Incomplete native/bin recording")
            save_record(OUT/"compact.npz");write(OUT/"EXPERIMENTAL_FINAL_STATE.json",expected.state())
            raw=monitor.report(STOP-START)
            metrics={k:v for k,v in arm.metrics.items() if not isinstance(v,list)}
            write(OUT/"RUN_RECEIPT.json",dict(status="FORWARD_COMPLETE",started_utc=worker.owner["utc"],ended_utc=utc(),
                assembly_seconds=assembly_seconds,run_wall_seconds=time.perf_counter()-worker.began,
                total_worker_seconds=time.perf_counter()-began,actual_simulated_seconds=arm.inner_time-arm.start,
                native_nodes=len(native),output_bins=len(prob),metrics=metrics,raw_absolute_monitor=raw,
                original_physical_history_domain_passed=not monitor.domain_failures,
                no_feedback_witnesses=proof,initial_match=initial["passed"],stimulus_identity=prep["identity"],
                model_id=model["model_id"],FIT_READY=False,absolute_monitor_false_preserved=True))
        except BaseException as exc:
            if native:save_record(OUT/"stopped_compact.npz")
            write(OUT/"FAILURE.json",dict(utc=utc(),reason=str(exc),traceback=traceback.format_exc(),
                source_seconds=arm.inner_time-arm.start,native_nodes=len(native),
                domain_failures=monitor.domain_failures,FIT_READY=False))
            try:
                save_bundle(OUT/"stopped_checkpoint",kind="complete_model_checkpoint",identity=prep["identity"],
                    payload=arm.checkpoint_payload(monitor=monitor.payload(),recording=dict(native_records=len(native),
                        experimental_readout=expected.state())),qualification=QUAL)
            except Exception as save_error:write(OUT/"CHECKPOINT_ERROR.json",dict(reason=str(save_error)))
            raise
        finally:arm.close()
    summarize()


def summarize():
    c=verify();receipt=read(OUT/"RUN_RECEIPT.json")
    with np.load(OUT/"compact.npz",allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    t=arrays["output_time_source_s"]+START;ct=arrays["core_time_source_s"]+START
    nt=arrays["native_time_source_s"]+START;v=arrays["target_V_gE_gI"][:,0]
    model=load_model(MODEL,allow_unqualified=True)
    frozen=TemporalReadout(model).predict(v[:-1])
    rate=arrays["experimental_expected_count_rate_Hz"]
    causal_error=float(np.max(abs(frozen["expected_count_rate_Hz"]-rate)))
    if causal_error>1e-9:raise RuntimeError("Functional online/offline readout exceeded frozen budget")
    rows=[]
    for label,on,off,end in (("bright",5.100,5.200,5.360),("dark",5.360,5.460,5.620)):
        base=(t>=on-.020-1e-12)&(t<on-1e-12)
        transient=(t>=on-1e-12)&(t<on+.020-1e-12)
        sustained=(t>=off-.020-1e-12)&(t<off-1e-12)
        tail=(t>=end-.020-1e-12)&(t<end-1e-12)
        response={}
        for name,values in (("experimental",rate),("original",arrays["original_expected_count_rate_Hz"])):
            baseline=float(values[base].mean());peak=float(np.max(abs(values[(t>=on-1e-12)&(t<end-1e-12)]-baseline)))
            delta=float(values[transient].mean()-baseline)
            recovery=float(values[tail].mean()-baseline)
            response[name]=dict(baseline_Hz=baseline,transient_delta_Hz=delta,
                sustained_delta_Hz=float(values[sustained].mean()-baseline),
                recovery_tail_delta_Hz=recovery,peak_excursion_Hz=peak,
                recovered_within160ms=bool(abs(recovery)<=max(.05*peak,1e-9)),
                polarity_point_passed=bool(delta>1e-9 if label=="bright" else delta<-1e-9),
                sign_is_descriptive_not_statistical_or_new_numerical_convergence=True)
        state_rows=[]
        for name,f in c["recording"]["fields"].items():
            if f["category"]!="states" or "RGC_expected_history" in f["key"]:continue
            values=arrays[name]
            cb=(ct>=on-.020-1e-12)&(ct<on-1e-12);cr=(ct>=end-.020-1e-12)&(ct<end-1e-12)
            cp=(ct>=on-1e-12)&(ct<end-1e-12)
            ref=values[cb].mean(axis=0);diff=abs(values[cr].mean(axis=0)-ref)
            peak=np.max(abs(values[cp]-ref),axis=0)
            budget=frozen_budget(ref,f["unit"])[2]
            allowance=np.maximum(.05*peak,budget)
            norm=diff/allowance
            state_rows.append(dict(field=name,unit=f["unit"],maximum_recovery_norm=float(norm.max()),
                peak_excursion=float(peak.max()),tail_difference=float(diff.max()),
                recovered_within160ms=bool(np.all(norm<=1)),scope="deterministic core states; not random spike history"))
        rows.append(dict(condition=label,on_model_seconds=on,off_model_seconds=off,recovery_end_model_seconds=end,
            response=response,state_recovery_all_passed=all(x["recovered_within160ms"] for x in state_rows),
            state_recovery=state_rows))
    pre=(t>=START)&(t<START+.020-1e-12);late=(t>=5.080-1e-12)&(t<5.100-1e-12)
    x=(v-model["core"]["mean_train_V"])/model["core"]["SD_train_V"]
    outcome="FUNCTIONAL_POINT_POLARITY_PASS" if all(r["response"]["experimental"]["polarity_point_passed"] for r in rows) else "FUNCTIONAL_POLARITY_NOT_CONFIRMED"
    full_recovery=all(r["response"]["experimental"]["recovered_within160ms"] and r["state_recovery_all_passed"] for r in rows)
    decision="FUNCTION_CHECK_COMPLETE_RECOVERY_WINDOW_INCOMPLETE" if not full_recovery else outcome
    result=dict(utc=utc(),decision=decision,polarity_summary=outcome,conditions=rows,
        initial_background_experimental_first20ms_Hz=float(rate[pre].mean()),
        initial_background_experimental_last20ms_Hz=float(rate[late].mean()),
        initial_background_drift_Hz=float(rate[late].mean()-rate[pre].mean()),
        standardized_voltage_min=float(x.min()),standardized_voltage_max=float(x.max()),
        training_envelope_known=False,training_scale="frozen g0-g2 SD only, no claimed training min/max envelope",
        finite_probability_domain=bool(np.isfinite(arrays["experimental_probability"]).all()
            and np.all((arrays["experimental_probability"]>=0)&(arrays["experimental_probability"]<=1))),
        online_vs_causal_postprocessing_max_Hz_error=causal_error,
        no_feedback_proof=all(x["unchanged"] for x in receipt["no_feedback_witnesses"]),
        actual_conditions_run=1,extra_conditions=0,parameter_fits=0,qualification="INCONCLUSIVE",
        FIT_READY=False,raw_absolute_monitor_unchanged_false=True,old_results_preserved=True)
    write(OUT/"RESULT.json",result)
    plot(arrays)
    lines=["# Cell6 627ms cone域工程功能检查","",
        "本轮完成并停止。决定："+decision+"；候选仍为 INCONCLUSIVE，FIT_READY=false，原 absolute monitor=false 保留。","",
        "唯一序列：原背景107ms → +5%亮100ms → 背景160ms → −5%暗100ms → 背景160ms。输入为全部保留native0.6 cone的原B_i(td)倍数，rod固定；外部reservoir不变。不是视频/光学端到端或独立生理验证，不推广其他细胞。","",
        "初态来自同一4.993s条件source seed，状态/历史/RNG/HC延迟逐项核验。107ms不是生理稳态证明。没有根据响应调幅度、延长、重跑、换符号、拟合或重新归一化。","",
        "|条件/输出|前背景Hz|前20ms变化Hz|末20ms变化Hz|恢复末20ms变化Hz|瞬态ON点判据|160ms期望恢复|",
        "|---|---:|---:|---:|---:|---|---|"]
    for row in rows:
        for name,d in row["response"].items():
            lines.append("|"+row["condition"]+"/"+name+"|"+
                "|".join(format(d[k],".8g") for k in ("baseline_Hz","transient_delta_Hz","sustained_delta_Hz","recovery_tail_delta_Hz"))+
                "|"+str(d["polarity_point_passed"])+"|"+str(d["recovered_within160ms"])+"|")
    lines+=["","确定性core状态在160ms内全部满足预声明恢复准则："+str(full_recovery)+
        "。逐字段见RESULT.json；未恢复只能解释为该160ms窗口内未恢复，不能据此宣称不稳定。准则为尾部相对各自前20ms背景差≤max(5%峰值偏离,原冻结状态数值预算)。随机spike history不要求逐值复原。","",
        "原背景首/末20ms实验率漂移="+str(result["initial_background_drift_Hz"])+"Hz；须与未充分适应、固定外边界和来源初态限制共同解读。","",
        "新V按冻结训练SD计范围=["+str(x.min())+","+str(x.max())+
        "]；冻结元数据无训练min/max，不能声称已证明处于训练包络。没有由此修改参数。","",
        "自由期望不喂真实spike，三个过去bin使用原八态算法；不新增采样spike。原RGC采样/history仍按原模型推进，原膜与上游未接受新读出反馈。原模型states/native anchor/history/RNG/clocks/HCdelay在观察器调用前后哈希一致；在线与仅本次V因果后处理最大误差="+str(causal_error)+"Hz。","",
        "有限/概率域="+str(result["finite_probability_domain"])+"；原物理/history域检查="+str(receipt["original_physical_history_domain_passed"])+
        "。原绝对monitor结果原样保留于RUN_RECEIPT.json；本轮未增加时间收敛或背景资格。瞬态符号是固定条件点检查，不是统计显著性或独立生理证据。","",
        "装配="+str(receipt["assembly_seconds"])+"s；模型运行="+str(receipt["run_wall_seconds"])+
        "s；实际模拟="+str(receipt["actual_simulated_seconds"])+"s。原627ms积分比例预计="+
        str(c["cost"]["projected_integration_seconds"])+"s；实际分项见RUN_RECEIPT.json，资源/释放见RESOURCE_RECEIPT.json。","",
        "本地functional_curves.png提供必要曲线；compact.npz含100μs真实RGC V/gE/gI、1ms原/实验期望及每1ms真实core状态/端口，字段单位/节点row见CONTRACT.json。没有28GB全层导出，不自动接入GUI新case或切换默认。","",
        "复现命令（当前结果存在时拒绝重复运行）：原Python执行 tools/check_cell6_controlled_function_v1.py prepare，然后run。所有输入帧hash和精确时刻已先冻结于CONTRACT.json；新刺激实际推进上游，未借用旧响应。","",
        "工程失败保留于上级目录：首轮在107ms处调用不满足公共watermark的checkpoint API，尚未完成亮阶跃；域检查无失败。修复只读观察器后用完全相同输入/种子/参数/判据重新完成，没有依响应改动刺激。没有可复用公共watermark断点，因此保留第一次不完整轨迹并从原seed启动。","",
        "本轮无其他ON-parasol盘点、跨细胞验证、BC扫描、训练、Git或上传。完成后停止，等待下一步明确授权。"]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    write(OUT/"FINAL_RECEIPT.json",dict(status="COMPLETE_AND_STOPPED",utc=utc(),decision=decision,
        qualification="INCONCLUSIVE",FIT_READY=False,BC_STOP_retained=True,worker_retained=False,
        report=str(OUT/"REPORT.md"),report_sha256=sha(OUT/"REPORT.md"),completed_retina_runs=1,incomplete_engineering_attempts=1,fit_count=0,
        new_data_survey=False,original_pins_verified=True))
    print(json.dumps(dict(stage="FUNCTION_COMPLETE",decision=decision,report=str(OUT/"REPORT.md"))),flush=True)


def plot(arrays):
    from PIL import Image
    from retina_platform.replay import trace_image
    t=arrays["output_time_source_s"];n=arrays["native_time_source_s"]
    multiplier=np.ones(len(n))
    multiplier[(n>=.107-1e-12)&(n<.207-1e-12)]=1.05
    multiplier[(n>=.367-1e-12)&(n<.467-1e-12)]=.95
    panels=[
        trace_image((1250,220),[(n,multiplier,"#64748b","cone B_i倍数；非视频/光学")],.627,"唯一冻结工程cone输入","factor")[0],
        trace_image((1250,240),[(n,arrays["target_V_gE_gI"][:,0]*1000,"#2563eb","原真实膜电位 / mV")],.627,"原模型目标group膜电位","mV")[0],
        trace_image((1250,260),[(t,arrays["original_expected_count_rate_Hz"],"#2563eb","原生产期望率"),
            (t,arrays["experimental_expected_count_rate_Hz"],"#7e22ce","实验期望率 / INCONCLUSIVE")],.627,"固定读出；无真实spike history输入","Hz")[0]]
    image=Image.new("RGB",(1250,sum(p.height for p in panels)),"white");top=0
    for p in panels:image.paste(p,(0,top));top+=p.height
    image.save(OUT/"functional_curves.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=["prepare","run"])
    args=parser.parse_args()
    try:prepare() if args.command=="prepare" else run()
    except BaseException as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        write(OUT/("ENTRY_FAILURE_"+datetime.now(timezone.utc).strftime("%H%M%S")+".json"),
              dict(reason=str(exc),traceback=traceback.format_exc(),stage=args.command,FIT_READY=False))
        raise


if __name__=="__main__":main()
