"""Owned single-process stages;24GiB/8GiB guard, no invented time deadline."""
from pathlib import Path
import argparse
import faulthandler
import sys
import time
import traceback

ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.cell6_halo_context_v1 import BASE,MODEL,prepare,write
from tools.durable_research_runtime_v1 import DurableRun
from retina_sim.background_checkpoint import MemoryGuard


def main(stage):
    directory=BASE/(stage+'_execution');directory.mkdir(exist_ok=False)
    journal=DurableRun(directory/'runtime','cell6_raw_'+stage);memory=MemoryGuard(24*1024**3,8*1024**3)
    started=time.perf_counter();status='FAILED'
    def guard():
        if (BASE/'REQUEST_STOP').exists() or (ROOT/'REQUEST_STOP').exists():raise RuntimeError('Explicit research STOP')
        if not memory.check():raise MemoryError(str((memory.stop_reason,memory.last)))
        journal.heartbeat(phase=stage,memory=memory.last)
    try:
        if stage in ('prepare','prepare2'):prepare(guard)
        elif stage=='fit':
            from tools.fit_cell6_raw_readout_v1 import main as fit
            fit(guard=guard)
        else:
            from tools.cell6_raw_cache_v1 import capture
            capture(int(stage[1:]),guard,journal,memory)
        status='PASS';journal.finish(status,phase='complete')
    except BaseException as error:
        write(directory/'FAILURE.json',dict(error=str(error),traceback=traceback.format_exc()))
        journal.finish('FAILED',phase=stage);raise
    finally:
        destination=MODEL if stage in ('prepare','prepare2') else BASE/'readout' if stage=='fit' else BASE/'cache'/stage;destination.mkdir(exist_ok=True)
        write(destination/'RESOURCE.json',dict(status=status,wall_seconds=time.perf_counter()-started,peak_rss_bytes=memory.peak_rss_bytes,
            memory=memory.last,guard_stop=memory.stop_reason,automatic_retry=False,wall_deadline_imposed=False,
            FIT_READY=False,background_qualified=False,production_adopted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=['prepare','prepare2','fit',*['g'+str(i) for i in range(6)]])
    args=parser.parse_args()
    with (BASE/('native_'+args.stage+'.log')).open('x',encoding='utf8') as stream:
        faulthandler.enable(file=stream,all_threads=True);main(args.stage)
