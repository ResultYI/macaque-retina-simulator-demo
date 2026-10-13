"""Run the same frozen local population configuration as the GUI.

No master generation or long background preparation. Default is one 1.2-ms
engineering media integration; physiological calibration is not claimed.
"""
from pathlib import Path
import argparse,json,sys,queue


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--media',type=Path)
    parser.add_argument('--config',type=Path,help='RunConfig JSON for the current population artifact')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--owner-x',type=float)
    parser.add_argument('--duration',type=float)
    parser.add_argument('--build-only',action='store_true')
    args=parser.parse_args();root=args.project_root.resolve();sys.path.insert(0,str(root))
    from retina_platform.config import RunConfig
    from retina_platform.backend import SimulationSession
    data=json.loads(args.config.read_bytes()) if args.config else RunConfig.project_default(root).to_dict()
    if data.get('execution_profile')!='atlas_population_revision_v1':parser.error('Config must select the current frozen population artifact')
    data['media_path']=str((args.media or Path(data.get('media_path') or root/'examples/platform_media/synthetic_rgb.png')).resolve())
    if args.owner_x is not None:data['owner_x_mm']=args.owner_x
    if args.duration is not None:data['duration_seconds']=args.duration
    config=RunConfig.from_dict(data);config.validate()
    if args.build_only:
        from retina_sim.atlas_population_network import atlas_population_network
        compiled,_,audit=atlas_population_network(root,duration=config.duration_seconds,light_context=config.light_context(),
            owner_bounds_mm=config.owner_bounds(),halo_mm=config.halo_mm,max_spatial_cells=config.max_spatial_cells,seed=config.seed)
        print(json.dumps({'status':'built','components':len(compiled.components),
            'counts':audit['atlas_runtime']['ancestor_population_counts'],'resource_estimate':audit['resource_estimate'],
            'unsupported_paths':audit['atlas_runtime']['unsupported_paths']},ensure_ascii=False));return
    session=SimulationSession(config,root,args.output).start()
    while session.is_alive() or not session.events.empty():
        try:event=session.events.get(timeout=.5)
        except queue.Empty:continue
        print(json.dumps({k:event[k] for k in ('status','time','path','message')},ensure_ascii=False),flush=True)
    if session.status!='completed':raise RuntimeError('Atlas Run failed: '+str(session.path))
    print(json.dumps({'status':session.status,'path':str(session.path),
        'summary':json.loads((session.path/'summary.json').read_bytes())},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
