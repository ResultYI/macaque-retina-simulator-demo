"""Single far integration worker; stages never overwrite previous results."""
from pathlib import Path
import argparse
import faulthandler
import sys
import time
import traceback

ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.integrated_far_context_v1 import BASE,prepare,write
from tools.durable_research_runtime_v1 import DurableRun
from retina_sim.background_checkpoint import MemoryGuard


def main(stage):
    started=time.perf_counter();directory=BASE/('prepare_execution' if stage=='prepare' else stage+'_execution')
    directory.mkdir(exist_ok=False)
    journal=DurableRun(directory/'runtime','single_far_integrated_'+stage)
    memory=MemoryGuard(24*1024**3,8*1024**3)
    def guard():
        if not memory.check():raise MemoryError(str((memory.stop_reason,memory.last)))
        if time.perf_counter()-started>(170 if stage=='prepare' else 820):raise TimeoutError('Original bounded worker guard')
        journal.heartbeat(phase=stage,memory=memory.last)
    status='FAILED'
    try:
        if stage=='prepare':prepare(guard)
        else:
            from tools.integrated_far_run_v1 import run
            run('tight' if stage=='tight_dense_v2' else stage,guard,journal,memory,output_name=stage)
        status='PASS';journal.finish(status,phase='complete')
    except BaseException as error:
        write(directory/'FAILURE.json',dict(error=str(error),traceback=traceback.format_exc()))
        journal.finish('FAILED',phase=stage);raise
    finally:
        destination=BASE/('background' if stage=='prepare' else stage)
        destination.mkdir(exist_ok=True)
        write(destination/'RESOURCE.json',dict(status=status,wall_seconds=time.perf_counter()-started,
            peak_rss_bytes=memory.peak_rss_bytes,memory=memory.last,guard_stop=memory.stop_reason,
            auto_retry=False,FIT_READY=False,background_qualified=False,production_adopted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['prepare','standard','tight','tight_dense_v2'],required=True)
    args=parser.parse_args()
    with (BASE/('native_'+args.stage+'.log')).open('x',encoding='utf-8') as stream:
        faulthandler.enable(file=stream,all_threads=True);main(args.stage)
