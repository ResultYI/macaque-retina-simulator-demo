"""Existing formal algorithms with explicit joint model dependencies; no module patching."""
from pathlib import Path
from types import FunctionType
import inspect
import textwrap
import json
import numpy as np
from retina_sim.contracts import canonical_hash
from retina_sim.spatial_group_graph import digest
from tools.gin69_joint_context import ROOT,OUT,FOLDER,load,protocol as original_protocol
from tools.durable_research_runtime_v1 import atomic_json

SOURCES=('tools/gin69_joint_context.py','tools/initialize_gin69_joint.py','tools/gin69_joint_execute.py',
         'retina_sim/gin69_outer_gaussian.py','retina_sim/gin69_frozen_aperture.py',
         'retina_sim/group_parameter_binding.py','retina_sim/group_inner_dynamics.py')


def protocol():
    p=original_protocol()
    p['joint_execution_sources']={name:digest(ROOT/name) for name in SOURCES}
    return p


def invoke(function,*args,**overrides):
    namespace=dict(function.__globals__)
    namespace.update(load=load,protocol=protocol,background_directory=lambda:FOLDER,BACKGROUND=FOLDER)
    namespace.update(overrides)
    fn=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    fn.__kwdefaults__=function.__kwdefaults__
    return fn(*args)


def execute(output,guard,stage):
    if stage=='joint_prepare_native':
        from tools.prepare_gin69_native_prefix_support_v2 import build
        build(output,guard)
    elif stage=='joint_short':
        from tools.gin_hc2_bc1_forward_support_v2 import trajectory
        from tools.durable_research_runtime_v1 import DurableRun
        journal=DurableRun(output/'runtime','GIN69_latest_joint_fullchain')
        folder=output/'forward';folder.mkdir()
        namespace=dict(trajectory.__globals__,load=load,protocol=protocol,BACKGROUND=FOLDER)
        source=textwrap.dedent(inspect.getsource(trajectory))
        old="drive=z['live_native_drive'].copy()"
        replacement="drive=np.stack([seed['input_reference']['native_background_td']*(1. if i<2 else 1.2 if i<4 else .8) for i in range(8)])"
        if source.count(old)!=1:raise ValueError('Existing short input adapter changed')
        source=source.replace(old,replacement)
        exec(compile(source,__file__+':joint_controlled_short','exec'),namespace)
        result=namespace['trajectory']('standard',guard,journal,folder)
        identity_file=folder/'standard/IDENTITY.json'
        evidence=json.loads(identity_file.read_bytes())
        evidence.update(source_frames_zero_based=None,role='Joint all-layer controlled native-input smoke test; original gray,1.2/0.8 factors; not movie prediction',
            controlled_input=dict(background_until=2/150,bright_until=4/150,then_dark_until=.05),
            source_transform=source)
        atomic_json(identity_file,evidence)
        atomic_json(output/'RESULT.json',dict(status='JOINT_FULLCHAIN_STANDARD_COMPLETED',
            result=result,scope='Existing 50ms real-movie full-chain standard and25ms exact checkpoint restore; no repeated HC/BC special tests',
            joint_source_hashes=protocol()['joint_execution_sources'],FIT_READY=False))
        journal.finish('PASS',phase='joint_short')
    elif stage=='joint_segment':
        from tools.gin69_formal_cache_support_v2 import run
        source=textwrap.dedent(inspect.getsource(run))
        old='rows=np.flatnonzero(pop.group_ids==7730)'
        if source.count(old)!=1:raise ValueError('Existing target index adapter changed')
        source=source.replace(old,'rows=np.flatnonzero(assembly.parent_population_rows[population]==7730)')
        namespace=dict(run.__globals__,load=load,protocol=protocol,background_directory=lambda:FOLDER)
        exec(compile(source,__file__+':verified_parent_row_target','exec'),namespace)
        namespace['run'](output,guard)
    elif stage in ('joint_train','joint_evaluate'):
        from tools.gin69_voltage_fit_support_v2 import execute as fit
        from tools.gin69_spatial_baselines_support_v2 import execute as baselines
        task='train' if stage=='joint_train' else 'evaluate'
        # Readout inputs and split remain unchanged. The formal cache identity is new.
        fit(output,guard,task);baselines(output,guard,task)
        if task=='evaluate':
            from tools.report_gin69_matched_support_v2 import build
            delivery=output/'delivery';delivery.mkdir();build(delivery,guard)
    else:raise ValueError(stage)
