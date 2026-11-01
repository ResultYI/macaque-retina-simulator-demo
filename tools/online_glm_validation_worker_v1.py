"""Owned, resource-guarded array-only validation of the new stateful readout."""
from pathlib import Path
import faulthandler,json,sys,time,traceback,unittest
ROOT=Path(r'D:\macaque retinal simulator');sys.path.insert(0,str(ROOT))
from tools.run_cell6_matched_baselines_v1 import attachment_gate
from tools.durable_research_runtime_v1 import DurableRun,atomic_json
from retina_sim.background_checkpoint import MemoryGuard


def main(nonce,out):
    with (out/'NATIVE_FAULTHANDLER.log').open('x',encoding='utf-8') as log:
        faulthandler.enable(file=log,all_threads=True);attachment_gate(nonce,out)
        journal=DurableRun(out/'runtime','online_glm_array_equivalence');memory=MemoryGuard(24*1024**3,8*1024**3)
        began=time.perf_counter();status='FAILED'
        def guard():
            if (out/'STOP').exists():raise RuntimeError('Explicit STOP')
            if not memory.check():raise MemoryError(str(memory.last))
            journal.heartbeat(memory=memory.last)
        def phase(name,**fields):
            guard();journal.heartbeat(force=True,phase=name,**fields)
            print(json.dumps(dict(phase=name,**fields)),flush=True)
        try:
            phase('SOFTWARE_REGRESSION')
            suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_online_voltage_glm_v1.py')
            with (out/'UNIT_TESTS.log').open('x',encoding='utf-8') as stream:tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
            atomic_json(out/'UNIT_RESULT.json',dict(passed=tests.wasSuccessful(),tests=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),
                synthetic_software_fixtures_only=True))
            if not tests.wasSuccessful():raise RuntimeError('Focused regression failed')
            from tools.validate_online_voltage_glm_v1 import run
            result=run(out,guard,phase)
            if not result['passed']:raise RuntimeError('Required equivalence failed')
            status='PASS';journal.finish(status,phase='COMPLETE')
        except BaseException as error:
            atomic_json(out/'FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
            journal.finish('FAILED');raise
        finally:
            atomic_json(out/'RESOURCE.json',dict(status=status,wall_seconds=time.perf_counter()-began,peak_RSS_bytes=memory.peak_rss_bytes,
                guard_stop=memory.stop_reason,model_simulation_run=False,fit_run=False,memory_limit_GiB=24,system_reserve_GiB=8))


if __name__=='__main__':main(sys.argv[1],Path(sys.argv[2]))
