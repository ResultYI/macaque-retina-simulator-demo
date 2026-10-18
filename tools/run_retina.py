"""Run the GUI's RunConfig/SimulationSession without opening a window."""
from pathlib import Path
import argparse,json,queue,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_platform.config import RunConfig
from retina_platform.backend import SimulationSession

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,help='RunConfig JSON; otherwise current project default')
    parser.add_argument('--media',type=Path,help='Explicit media override')
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args()
    config=RunConfig.from_dict(json.loads(args.config.read_bytes())) if args.config else RunConfig.project_default(ROOT)
    if args.media:config.media_path=str(args.media.resolve())
    session=SimulationSession(config,ROOT,output_root=args.output_root).start()
    previous=None;printed=0.
    while session.is_alive() or not session.events.empty():
        try:event=session.events.get(timeout=.25)
        except queue.Empty:continue
        key=(event['kind'],event['status'],event.get('phase'),event.get('message'))
        if key!=previous or time.monotonic()-printed>=15:
            print(json.dumps(event,ensure_ascii=False),flush=True);previous=key;printed=time.monotonic()
    print(json.dumps({'status':session.status,'run_path':str(session.path or '')}),flush=True)
    return 0 if session.status=='completed' else 2

if __name__=='__main__':sys.exit(main())
