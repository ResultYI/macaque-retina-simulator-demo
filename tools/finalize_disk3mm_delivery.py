"""Finish the explicitly authorized source/evidence publication after acceptance.

One bounded continuation of this delivery, with no automatic reruns, generation,
force push, environment changes or scientific threshold changes. The request
freezes the complete source allowlist before waiting for the actual result.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'docs/pro_review_supplement'
REMOTE='https://github.com/ResultYI/macaque-retina-simulator-demo.git'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def local(value):
    path=(ROOT/value).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Publication path outside existing project')
    return path


def save(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temporary,path)


def command(args,*,timeout=180):
    result=subprocess.run(args,cwd=ROOT,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'),
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
    if result.returncode:raise RuntimeError('Command failed: '+args[0]+' '+str(args[1:3])+'; '+result.stderr[-1200:])
    return result.stdout.strip()


def identities(request,*,git_clean_index=True):
    if request.get('schema')!='authorized_disk3mm_publication_v1' or request.get('status')!='APPROVED':
        raise ValueError('Matching explicit publication authorization required')
    if Path(sys.executable).resolve()!=Path(request['python_executable']).resolve():raise ValueError('Approved environment differs')
    for relative,expected in request['source_allowlist'].items():
        if sha(local(relative))!=expected:raise ValueError('Parallel source edit preserved: '+relative)
    for relative,expected in request['helper_guards'].items():
        if sha(local(relative))!=expected:raise ValueError('Publication helper differs: '+relative)
    if command(['git','rev-parse','--abbrev-ref','HEAD'])!='main':raise ValueError('Existing branch changed; no branch switching')
    if command(['git','remote','get-url','--push','origin'])!=REMOTE:raise ValueError('Original authorized repository differs')
    if command(['git','rev-parse','HEAD'])!=request['original_git_head']:raise ValueError('Parallel commit preserved; publication needs reconciliation')
    if git_clean_index and command(['git','diff','--cached','--name-only']):raise ValueError('Parallel staged work preserved; index not changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--resume-wait',action='store_true',help='Explicit continuation after a saved stop before any Git/default changes')
    args=parser.parse_args()
    request_path=local(args.request);request=json.loads(request_path.read_bytes());build=request_path.parent
    identities(request)
    if args.preflight:
        print('PUBLICATION_ALLOWLIST_AND_ORIGINAL_GIT_STATE_VERIFIED; no staging, commit, push or evidence replacement',flush=True);return
    status=build/'publication_status.json'
    if status.exists():
        previous=json.loads(status.read_bytes())
        stopped=('STOPPED_BEFORE_GIT_FOR_EXACT_COMMIT_PIN','STOPPED_AFTER_VERIFIED_RECOVERY_FOR_EXPLICIT_CUDA_EXECUTION_REPAIR',
                 'RETRY_AUTHORIZED_AFTER_RECORDING_CLOCK_REPAIR')
        if not args.resume_wait or previous.get('status') not in stopped or previous.get('pushed') or previous.get('default_changed') or previous.get('commit'):
            raise FileExistsError('Previous publication retained; no automatic repeat')
    result={'status':'WAITING_FOR_ACTUAL_ACCEPTANCE','default_changed':False,'pushed':False,'started_UTC':datetime.now(timezone.utc).isoformat()}
    save(status,result);start=time.monotonic()
    try:
        final=build/'final_delivery.json'
        while True:
            if final.exists():
                delivery=json.loads(final.read_bytes())
                if delivery['status']!='RUNNING_MINIMUM_ACCEPTANCE':break
            supervisor=build/request.get('delivery_supervisor_status','delivery_supervisor_v2_status.json')
            if supervisor.exists() and json.loads(supervisor.read_bytes())['status'].startswith('BLOCKED'):
                raise RuntimeError('Actual transition/delivery is blocked; no publication or default-success claim')
            if time.monotonic()-start>request['maximum_wait_seconds']:raise TimeoutError('Bounded actual-delivery wait expired')
            time.sleep(30)
        if delivery.get('status')!='MINIMUM_ACCEPTANCE_COMPLETE_DEFAULT_ADOPTED' or delivery.get('default_changed') is not True:
            raise RuntimeError('Actual three-region acceptance/default adoption not established')
        result.update(status='PACKAGING_ACTUAL_ACCEPTED_EVIDENCE',default_changed=True);save(status,result)
        identities(request)
        acceptance_request=json.loads((build/'authorized_delivery_request.json').read_bytes())
        for relative,expected in acceptance_request['frozen_identity_guards'].items():
            if sha(local(relative))!=expected:raise ValueError('Actual acceptance source identity differs before publication: '+relative)
        if sha(ROOT/'configurations/retina_platform_v1.json')!=delivery['active_configuration_sha256'] or sha(ROOT/'docs/core_design.md')!=delivery['core_design_sha256']:
            raise ValueError('Parallel default/design edits preserved after actual adoption')
        previous=build/'previous_public_review_before_finalization'
        if PUBLIC.exists():
            if PUBLIC.resolve()!=PUBLIC or PUBLIC.parent!=ROOT/'docs' or previous.resolve().parent!=build.resolve() or not previous.resolve().is_relative_to(ROOT):
                raise ValueError('Evidence archive target validation failed')
            if previous.exists():raise FileExistsError('Prior public review archive retained')
            PUBLIC.rename(previous)
        command([sys.executable,'-B','-X','utf8','tools/package_pro_review.py','--build',build.relative_to(ROOT).as_posix(),
                 '--output','docs/pro_review_supplement'],timeout=600)
        index=json.loads((PUBLIC/'review_index.json').read_bytes())
        if index['delivery_status']!='MINIMUM_ACCEPTANCE_COMPLETE_DEFAULT_ADOPTED':raise ValueError('Public evidence does not describe actual accepted delivery')
        command([sys.executable,'-B','-X','utf8','tmp/public_payload_preflight.py'])
        scan=json.loads((ROOT/'tmp/public_payload_preflight.json').read_bytes())
        if scan['secret_pattern_findings']:raise ValueError('Public payload credential-pattern review blocked')
        allowed=set(request['source_allowlist'])|{'configurations/retina_platform_v1.json','docs/core_design.md'}
        current={p for p in scan['paths'] if not p.startswith('docs/pro_review_supplement/')}
        if current!=allowed:raise ValueError('Source allowlist changed during actual execution; parallel files preserved')
        if set(scan['home_path_files'])-{'README.md','docs/core_design.md'}:raise ValueError('Unexpected private home path in public payload')
        public_files={p.relative_to(ROOT).as_posix() for p in PUBLIC.rglob('*') if p.is_file()}
        paths=sorted(allowed|public_files)
        if scan['total_bytes']>64*1024**2:raise ValueError('Compact public payload exceeds explicit64MiB budget')
        identities(request)
        remote=command(['git','ls-remote','origin','refs/heads/main']).split()[0]
        if remote!=request['original_git_head']:raise ValueError('Remote main advanced; no force or unreviewed merge')
        expected_blobs={}
        for relative in paths:
            raw=local(relative).read_bytes()
            expected_blobs[relative]=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        save(build/'accepted_publication_payload.json',{'paths':paths,'git_blob_identities':expected_blobs,
            'source_sha256':{relative:sha(local(relative)) for relative in paths}})
        result['status']='STAGING_VERIFIED_ALLOWED_FILES';save(status,result)
        pathspec=build/'publication_pathspec.nul';pathspec.write_bytes(b''.join(p.encode('utf-8')+b'\0' for p in paths))
        command(['git','add','--pathspec-from-file='+str(pathspec),'--pathspec-file-nul'])
        staged=set(command(['git','diff','--cached','--name-only']).splitlines())
        if not staged or not staged<=set(paths):raise ValueError('Staged source scope differs; retained for review')
        # Preserve already-frozen source bytes, including one existing blank
        # EOF; retain the substantive trailing-whitespace checks.
        command(['git','-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check'])
        message=build/'publication_commit_message.txt'
        message.write_text('Use explicit finite-disk geometry and prepared representative responses\n\n'
            'Retain unwarped human-borrowed cone positions, source-informed project HC functional pooling, and explicit midget floor minima with original deficits. Preserve anatomical references and recovery identities.\n\n'
            'Archive the actual three-region ISET/background/matched20ms responses, executed parameter values, numerical checks and unrun physiological controls. Preserve existing source equations while reducing execution overhead.\n',encoding='utf-8')
        command(['git','commit','--file',str(message)],timeout=180)
        commit=command(['git','rev-parse','HEAD']);result.update(status='LOCAL_COMMIT_CREATED_PUSH_PENDING',commit=commit);save(status,result)
        if command(['git','rev-parse',commit+'^'])!=request['original_git_head']:
            raise ValueError('Concurrent commit detected; local work retained without publication')
        committed=set(command(['git','diff-tree','--no-commit-id','--name-only','-r',commit]).splitlines())
        if committed!=staged:raise ValueError('Commit hooks changed the staged scope; local commit retained without push')
        tree=subprocess.run(['git','ls-tree','-r','-z',commit],cwd=ROOT,capture_output=True,check=True).stdout
        actual={}
        for entry in tree.split(b'\0'):
            if entry:
                header,name=entry.split(b'\t',1);actual[name.decode('utf-8')]=header.split()[2].decode('ascii')
        if any(actual.get(relative)!=expected for relative,expected in expected_blobs.items()):
            raise ValueError('Committed bytes differ from the reviewed allowlist; no push')
        command(['git','push','origin',commit+':refs/heads/main'],timeout=180)
        remote=command(['git','ls-remote','origin','refs/heads/main']).split()[0]
        if remote!=commit:raise ValueError('Remote commit could not be verified')
        result.update(status='COMPLETED_ACTUAL_DELIVERY_AND_VERIFIED_PUSH',pushed=True,repository=REMOTE,branch='main',
            commit=commit,commit_URL=REMOTE.removesuffix('.git')+'/commit/'+commit,
            public_review_index_sha256=sha(PUBLIC/'review_index.json'),source_file_count=len(allowed),public_evidence_file_count=len(public_files),
            actual_delivery=final.relative_to(ROOT).as_posix(),physiological_validation=False)
    except Exception as error:
        result.update(status='BLOCKED_PUBLICATION',error=type(error).__name__+': '+str(error));raise
    finally:
        result['updated_UTC']=datetime.now(timezone.utc).isoformat();save(status,result)
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
