"""One explicit WIP snapshot publisher; preserve working HEAD/index/default.

An alternate index produces a normal Git tree/commit on a dedicated review
branch. The exact captured commit is pushed without force and verified remotely.
No final-acceptance status, local checkout or active default is changed.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/current_wip_review'
REMOTE='https://github.com/ResultYI/macaque-retina-simulator-demo.git'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    request=json.loads((REPORT/'request.json').read_bytes())
    if request.get('schema')!='authorized_WIP_review_push_v1' or request.get('status')!='APPROVED':
        raise ValueError('Explicit immediate WIP publication request required')
    branch=request['branch']
    if not re.fullmatch(r'review/physiology-wip-\d{8}T\d{4}Z',branch):raise ValueError('Unexpected review branch')
    environment=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    def git(args,*,data=None,index=False,timeout=120):
        env=dict(environment)
        if index:env['GIT_INDEX_FILE']=str(REPORT/'review.index')
        result=subprocess.run(['git',*args],cwd=ROOT,env=env,input=data,stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,timeout=timeout)
        if result.returncode:raise RuntimeError('git '+args[0]+' failed: '+(result.stderr+result.stdout).decode('utf-8',errors='replace')[:2500])
        return result.stdout
    if git(['remote','get-url','--push','origin']).decode().strip()!=REMOTE:raise ValueError('Authorized remote differs')
    base=git(['rev-parse','HEAD']).decode().strip();active=git(['rev-parse','--abbrev-ref','HEAD']).decode().strip()
    if active!='main':raise ValueError('Preserve the existing calling main checkout')
    index_path=Path(git(['rev-parse','--git-path','index']).decode().strip())
    if not index_path.is_absolute():index_path=ROOT/index_path
    original_index=sha(index_path) if index_path.exists() else None
    default=ROOT/'configurations/retina_platform_v1.json'
    if sha(default)!=request['default_configuration_sha256_before']:raise ValueError('Active default changed before snapshot')
    publisher=json.loads((ROOT/'reports/human_disk3mm_functional_hc_20261017/publication_status.json').read_bytes())
    if publisher['status']!='PAUSED_FOR_USER_REQUESTED_IMMEDIATE_WIP_REVIEW' or publisher.get('pushed'):
        raise ValueError('Old final publisher has not been disabled')
    state_path=REPORT/'push_status.json'
    if state_path.exists():raise FileExistsError('Prior WIP publication retained; inspect captured commit before retry')
    scan=json.loads((ROOT/'tmp/public_payload_preflight.json').read_bytes())
    if scan['secret_pattern_findings']:raise ValueError('Public credential-pattern scan failed')
    if set(scan['home_path_files'])-{'README.md','docs/core_design.md'}:raise ValueError('Unexpected private home path')
    paths=sorted(scan['paths'])
    if any(not (ROOT/p).resolve().is_relative_to(ROOT) for p in paths):raise ValueError('Snapshot path outside project')
    if any('benardete' in p.lower() or 'Pro审阅' in p for p in paths):raise ValueError('Unrelated/private research excluded')
    signatures={p:sha(ROOT/p) for p in paths}
    if sum((ROOT/p).stat().st_size for p in paths)>64*1024**2:raise ValueError('Compact source/evidence budget exceeded')
    remote_main=git(['ls-remote','origin','refs/heads/main']).decode().split()[0]
    if git(['ls-remote','origin','refs/heads/'+branch]).strip():raise ValueError('Review remote branch already exists; no overwrite')
    local_ref=subprocess.run(['git','show-ref','--verify','--quiet','refs/heads/'+branch],cwd=ROOT,env=environment)
    if local_ref.returncode==0:raise ValueError('Review local branch already exists; no overwrite')
    message=REPORT/'commit_message.txt'
    message.write_text('WIP: physiology contracts, functional HC alternative and execution review\n\n'
        'Snapshot current code/configuration and scoped real transition evidence for Pro review.\n'
        'Functional-HC three-ROI acceptance is incomplete; active default and main remain unchanged.\n'
        'Record the central empty-owner repair and AII positive-support blocker without fabricated input.\n'
        'Expose measured performance: most background wall time is outside integration and needs short profiling.\n\n'
        'Validation:10 focused CUDA/execution checks,2 recording-clock regressions,3 empty-owner regressions;\n'
        'real v7 transition background/20ms matched response passed. No full physiological validation claim.\n',encoding='utf-8')
    status={'status':'BUILDING_WIP_REVIEW_COMMIT','branch':branch,'base_commit':base,
        'remote_main_before':remote_main,'active_default_changed':False,'pushed':False,
        'source_and_evidence_sha256':signatures,'UTC':datetime.now(timezone.utc).isoformat()}
    def save():state_path.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    save()
    try:
        git(['read-tree',base],index=True)
        pathspec=REPORT/'snapshot_paths.nul';pathspec.write_bytes(b'\0'.join(p.encode('utf-8') for p in paths)+b'\0')
        git(['--literal-pathspecs','add','--pathspec-from-file='+str(pathspec),'--pathspec-file-nul'],index=True)
        # Preserve the upstream license verbatim, including its original spaces.
        git(['-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check',
             '--','.',':(exclude)docs/pro_review_supplement/SCIPY_LICENSE.txt'],index=True)
        changed=set(git(['diff','--cached','--name-only','-z'],index=True).decode('utf-8').rstrip('\0').split('\0'))
        if not changed or not changed<=set(paths):raise ValueError('Alternate index staged outside the exact snapshot')
        for p,expected in signatures.items():
            if sha(ROOT/p)!=expected:raise ValueError('Parallel source change preserved before commit: '+p)
        tree=git(['write-tree'],index=True).decode().strip()
        commit=git(['commit-tree',tree,'-p',base,'-F',str(message)]).decode().strip()
        for p in paths:
            raw=git(['cat-file','blob',commit+':'+p])
            if hashlib.sha256(raw).hexdigest()!=signatures[p]:raise ValueError('Committed blob differs from captured bytes: '+p)
        git(['update-ref','refs/heads/'+branch,commit,'0'*len(base)])
        status.update(status='LOCAL_WIP_REVIEW_COMMIT_CREATED',commit=commit,tree=tree,
            commit_url=REMOTE.removesuffix('.git')+'/commit/'+commit,
            tree_url=REMOTE.removesuffix('.git')+'/tree/'+commit);save()
        git(['push','origin',commit+':refs/heads/'+branch],timeout=300)
        remote=git(['ls-remote','origin','refs/heads/'+branch]).decode().split()[0]
        if remote!=commit:raise ValueError('Remote review commit verification failed')
        if git(['rev-parse','HEAD']).decode().strip()!=base or sha(index_path)!=original_index or sha(default)!=request['default_configuration_sha256_before']:
            raise ValueError('Working checkout/index/default changed during publication')
        status.update(status='WIP_REVIEW_PUSH_VERIFIED',pushed=True,remote_commit=remote,
            remote_main_after=git(['ls-remote','origin','refs/heads/main']).decode().split()[0],
            status_url=REMOTE.removesuffix('.git')+'/blob/'+commit+'/docs/pro_review_supplement/review_status.json',
            finished_UTC=datetime.now(timezone.utc).isoformat());save()
        print(json.dumps({k:status[k] for k in ('status','branch','commit','commit_url','status_url','remote_main_before','remote_main_after','active_default_changed')}),flush=True)
    except Exception as error:
        status.update(status='WIP_REVIEW_PUSH_BLOCKED',error=type(error).__name__+': '+str(error));save();raise
if __name__=='__main__':main()
