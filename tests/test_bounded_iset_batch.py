from pathlib import Path
import pytest
from retina_platform import bounded_iset_batch as batch


@pytest.mark.parametrize('mode',['raise','false'])
def test_boundary_failure_cleans_up_owned_process(tmp_path,monkeypatch,mode):
    class Process:
        returncode=None
        pid=123
        def poll(self):return self.returncode
    process=Process();stopped=[]
    monkeypatch.setattr(batch.subprocess,'Popen',lambda *args,**kwargs:process)
    def stop(value):
        assert value is process;stopped.append(value.pid);value.returncode=1
    monkeypatch.setattr(batch,'_terminate_owned',stop)
    def boundary():
        if mode=='raise':raise MemoryError('original reserve')
        return False
    error=MemoryError if mode=='raise' else batch.ContractError
    with pytest.raises(error):
        batch.run_batch(tmp_path,{'matlab_executable':'unchanged MATLAB','maximum_export_wall_seconds':300},
            [{'output':str(tmp_path/'capture.mat')}],tmp_path,None,boundary)
    assert stopped==[123]
