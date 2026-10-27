"""V2 reuses v1 safety gates and adds a strictly smaller worker wall cap."""
import importlib.util
from pathlib import Path
import time
import json
import os
import subprocess
import sys
from types import SimpleNamespace
import pytest

def load():
 p=Path(__file__).resolve().parents[1]/'tools/run_local_patch_v2.py'
 spec=importlib.util.spec_from_file_location('patch_v2_launch_test',p)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_worker_180_limit_precedes_resource_calls(tmp_path):
 m=load();g=m.Guard(tmp_path,dict(deadline_unix=time.time()+100,source_pins={}),worker_started=time.monotonic()-181)
 with pytest.raises(RuntimeError,match='180s'):g()

def test_stage_output_never_overwrites_old_stop(tmp_path):
 m=load();p=tmp_path/'old';p.mkdir();(p/'REQUEST_STOP').write_text('keep')
 with pytest.raises(FileExistsError):m.demo(SimpleNamespace(output=p,plan=tmp_path/'missing'))
 assert (p/'REQUEST_STOP').read_text()=='keep'


@pytest.mark.parametrize('change',['missing_role','wrong_role','wrong_cap'])
def test_child_rejects_role_cap_job_mutation(change):
 m=load();auth=dict(role_caps_seconds={'halo_030_prefix':55.})
 job=dict(role_name='halo_030_prefix',role_wall_cap_seconds=55.)
 assert m.validate_role(job,auth)=='halo_030_prefix'
 if change=='missing_role':del job['role_name']
 elif change=='wrong_role':job['role_name']='other'
 else:job['role_wall_cap_seconds']=70.
 with pytest.raises(RuntimeError,match='role cap'):m.validate_role(job,auth)


@pytest.mark.skipif(os.name!='nt',reason='Real Windows Job ownership test')
@pytest.mark.parametrize('gate',['halo_030_prefix','halo_030_restore','halo_060_complete','stop','parent','deadline'])
def test_real_child_guard_and_supervisor_ownership(tmp_path,gate):
 m=load();caps=dict(halo_030_prefix=55.,halo_030_restore=70.,halo_060_complete=70.)
 role=gate if gate in caps else 'halo_030_prefix'
 auth=dict(deadline_unix=time.time()+60,source_pins={},role_caps_seconds=caps,total_worker_limit_seconds=600.)
 m.write(tmp_path/'AUTHORIZATION.json',auth)
 m.write(tmp_path/'ledger.json',dict(used_worker_seconds=385.21899999998277))
 supervisor=m.process_identity(os.getpid())
 job=dict(supervisor_process=supervisor,token='guard-probe',role_name=role,role_wall_cap_seconds=caps[role],gate=gate,
          stage=str(tmp_path),root=str(Path(__file__).resolve().parents[1]),authorization_sha256=m.sha(tmp_path/'AUTHORIZATION.json'))
 m.write(tmp_path/'job.json',job)
 code='''import sys,json,time
from pathlib import Path
j=json.loads(Path(sys.argv[1]).read_text());sys.path.insert(0,j['root'])
from tools import run_local_patch_v2 as m
s=Path(j['stage']);a=m.read(s/'AUTHORIZATION.json')
assert m.sha(s/'AUTHORIZATION.json')==j['authorization_sha256']
role=m.validate_role(j,a)
h=m.original_worker_handshake(s,j,'patch')
started=time.monotonic()
if j['gate'] in a['role_caps_seconds']:started-=a['role_caps_seconds'][role]+.01
if j['gate']=='stop':(s/'REQUEST_STOP').write_text('probe')
if j['gate']=='deadline':a['deadline_unix']=time.time()-1
parent=dict(j['supervisor_process'])
if j['gate']=='parent':parent['creation_ticks']+=1
g=m.Guard(s,a,worker_started=started,parent=parent,role_name=role)
g.last=time.monotonic()
try:g()
except RuntimeError as e:m.write(s/'probe_result.json',dict(error=str(e),worker=h['worker'],role=role,cap=a['role_caps_seconds'][role]))
else:raise AssertionError('Guard did not reject')
'''
 owned=m.WindowsWorkerJob();process=None
 try:
  process=subprocess.Popen([sys.executable,'-I','-X','utf8','-B','-c',code,str(tmp_path/'job.json')],
      stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
  owned.bind_launcher(process);end=time.monotonic()+20
  while process.poll() is None:
   assert time.monotonic()<end,'Probe ownership timeout'
   hp=tmp_path/'patch_worker_handshake.json'
   if owned.worker is None and hp.exists():
    m.write(tmp_path/'patch_worker_ownership.json',owned.accept(m.read(hp),token=job['token'],label='patch',supervisor=supervisor))
   time.sleep(.02)
  out,err=process.communicate()
  assert process.returncode==0,(out,err)
  result=m.read(tmp_path/'probe_result.json')
  assert result['worker']==owned.worker
  assert result['cap']==caps[role]
  expected='role wall cap' if gate in caps else 'STOP' if gate=='stop' else 'Supervisor process identity' if gate=='parent' else 'deadline'
  assert expected in result['error']
 finally:
  if process is not None and process.poll() is None:owned.terminate();process.wait(timeout=15)
  owned.close()
