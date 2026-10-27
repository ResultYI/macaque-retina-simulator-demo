"""Run the same active project configuration used by the GUI; no profile switch."""
from pathlib import Path
import argparse,sys,queue,json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--duration',type=float)
    parser.add_argument('--background-checkpoint',type=Path,help='Resume/reuse an identity-bound complete background checkpoint')
    parser.add_argument('--prepare-background',action='store_true',help='Use unchanged bounded numerical preparation before the stimulus')
    args=parser.parse_args();root=args.project_root.resolve();sys.path.insert(0,str(root))
    from retina_platform.config import RunConfig
    from retina_platform.backend import SimulationSession
    config=RunConfig.project_default(root);config.media_path=str(args.media.resolve())
    if args.duration is not None:config.duration_seconds=args.duration
    if args.prepare_background or args.background_checkpoint:
        config.background_initialization_mode='bounded_convergence';config.background_require_converged=True
    if args.background_checkpoint:config.background_checkpoint_path=str(args.background_checkpoint.resolve())
    session=SimulationSession(config,root,args.output).start()
    while session.is_alive() or not session.events.empty():
        try:event=session.events.get(timeout=.5)
        except queue.Empty:continue
        print(json.dumps({k:event[k] for k in ('status','time','path','message')},ensure_ascii=False),flush=True)
    if session.status!='completed':raise RuntimeError('Run failed: '+str(session.path))


if __name__=='__main__':main()
