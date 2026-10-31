"""Tiny non-model process used only to verify lifecycle records."""
import argparse,json
from pathlib import Path
import time
from tools.durable_research_runtime_v1 import DurableRun


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['normal','fail','hold'],default='normal');a=p.parse_args()
    a.output.mkdir(exist_ok=False);run=DurableRun(a.output/'runtime','dummy-no-model',period=.05)
    status='FAILED'
    try:
        run.heartbeat(force=True,status='RUNNING',phase='dummy_wait')
        until=time.perf_counter()+(6. if a.mode=='hold' else .3)
        while time.perf_counter()<until:run.heartbeat();time.sleep(.05)
        if a.mode=='fail':raise RuntimeError('Deliberate dummy failure')
        status='PASS'
    finally:
        (a.output/'old_style_finally.json').write_text(json.dumps(dict(status=status)))
        run.finish(status,phase='dummy_finished')
