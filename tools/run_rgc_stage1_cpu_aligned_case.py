"""Observer facade around unchanged original compact capture; selected case only."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True

def main():
    from tools import run_rgc_stage1_cpu_full_case as original
    from retina_sim import rgc_stage1_cache as cache
    from retina_sim.rgc_stage1_aligned_recording import AlignedObserver,observing_arm,write,canonical_points
    from tools.validate_ac_vs_timescale import calendar
    jobpath=Path(sys.argv[sys.argv.index('--job')+1]);job=json.loads(jobpath.read_text());spec=json.loads((jobpath.parent/'aligned_record_spec.json').read_text())
    if job['case_id']!=spec['case_id'] or job['computation_identity']!=spec['provenance']['cache_identity']:raise RuntimeError('Aligned observer source identity differs')
    directory=Path(job['private_root'])/'aligned'/jobpath.parent.name;observers=[];base=cache.GroupTimeArm
    def factory(arm):
        try:
            observer=AlignedObserver(directory,spec,arm.assembly)
            observer.set_canonical_points(canonical_points(arm.start,arm.stop,spec['source_dt_s'],spec['source_nodes'],calendar(arm.protocol)))
        except Exception as e:
            reason=dict(status='FAILED_NOT_UI_READY',type=type(e).__name__,reason=str(e),original_capture_may_continue=True)
            write(jobpath.parent/'ALIGNED_FACTORY_FAILURE.json',reason)
            class Disabled:
                root=directory
                def observe(self,*a,**k):pass
                def checkpoint(self,*a,**k):pass
                def close(self,*a,**k):pass
                def resume_from(self,*a,**k):pass
                def fail(self,*a,**k):pass
                def publish(self,*a,**k):
                    directory.mkdir(parents=True,exist_ok=True);write(directory/'STATUS.json',reason);return False
            observer=Disabled()
        observers.append(observer);return observer
    cache.GroupTimeArm=observing_arm(base,factory)
    try:
        original.main()
        if len(observers)!=1:raise RuntimeError('Exactly one original model arm must own the recording')
        product=observers[0]
        try:product.publish(Path(job['private_root'])/'cache'/job['case_id'])
        except Exception as e:
            product.fail(e);write(directory/'STATUS.json',dict(status='FAILED_NOT_UI_READY',reason=str(e),original_cache_retained=True))
    finally:cache.GroupTimeArm=base

if __name__=='__main__':main()
