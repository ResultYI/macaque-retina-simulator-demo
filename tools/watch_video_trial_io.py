"""Read-only checkpoint lifecycle observation for one running local video trial."""
from pathlib import Path
import argparse,json,time


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args()
    directory=args.directory.resolve();target=directory/'checkpoint_file_lifetimes.json'
    rows=[];seen=set()
    while True:
        for name in ('start_checkpoint.json','end_checkpoint.json','latest_checkpoint.json'):
            path=directory/name
            try:
                stat=path.stat();birth=getattr(stat,'st_birthtime_ns',stat.st_ctime_ns)
                identity=(name,birth,stat.st_mtime_ns,stat.st_size)
                if identity in seen:continue
                seen.add(identity)
                rows.append({'file':name,'bytes':stat.st_size,'creation_time_ns':birth,
                    'last_write_time_ns':stat.st_mtime_ns,
                    'file_lifetime_seconds':(stat.st_mtime_ns-birth)/1e9,
                    'observed_time_ns':time.time_ns()})
            except FileNotFoundError:pass
        report={'scope':'Local NTFS checkpoint file creation-to-last-write intervals; streamed writer creates a fresh temporary before hashing then atomically publishes it',
            'limitations':'Filesystem wall-clock intervals, not independent high-resolution CPU timers. Compare to tracked start/end/rolling writer timers. No checkpoint content or neural state is read or changed.',
            'published_files':rows}
        temporary=target.with_name(target.name+'.writing')
        temporary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');temporary.replace(target)
        try:status=json.loads((directory/'trial.json').read_bytes())['status']
        except (FileNotFoundError,json.JSONDecodeError):status=None
        if status in ('COMPLETED','FAILED_RECOVERABLE_ARTIFACTS_RETAINED','STOPPED_WITH_FULL_CHECKPOINT'):break
        time.sleep(2.)
    print(json.dumps({'status':'IO_OBSERVATION_ENDED','published_checkpoint_versions':len(rows)}),flush=True)

if __name__=='__main__':main()
