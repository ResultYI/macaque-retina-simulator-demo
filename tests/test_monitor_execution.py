from types import SimpleNamespace
import sys
import pytest
from retina_sim.contracts import ContractError
from retina_sim.monitor_execution import background_monitor_threads


def setup(monkeypatch,pool=4):
    mask=[1]
    module=SimpleNamespace(config=SimpleNamespace(NUMBA_NUM_THREADS=pool),
        get_num_threads=lambda:mask[0],set_num_threads=lambda n:mask.__setitem__(0,n))
    monkeypatch.setitem(sys.modules,'numba',module)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(key,'1')
    calls=[]
    def rhs(*args,**kwargs):calls.append((module.get_num_threads(),args,kwargs));return 'original'
    runner=SimpleNamespace(model=SimpleNamespace(rhs_flat=rhs,_compiled_rhs=SimpleNamespace(native_fused=True,native_threads=8)))
    return module,runner,calls,rhs


def test_reference_default_has_no_wrapper(monkeypatch):
    module,runner,calls,original=setup(monkeypatch)
    with background_monitor_threads(runner) as receipt:
        assert runner.model.rhs_flat is original
        assert runner.model.rhs_flat(7)=='original'
    assert receipt['restored'] and not receipt['applied'] and calls[0][0]==1


def test_explicit_threads_only_inside_rhs_and_restore(monkeypatch):
    module,runner,calls,original=setup(monkeypatch)
    with background_monitor_threads(runner,4) as receipt:
        assert module.get_num_threads()==1
        assert runner.model.rhs_flat(2,label='same')=='original'
        assert module.get_num_threads()==1
    assert calls==[(4,(2,),{'label':'same'})]
    assert runner.model.rhs_flat is original and receipt['restored']


def test_exception_keeps_original_failure_and_restores(monkeypatch):
    module,runner,calls,original=setup(monkeypatch)
    failure=ContractError('original failure')
    def bad():
        assert module.get_num_threads()==4
        raise failure
    runner.model.rhs_flat=bad
    with pytest.raises(ContractError) as caught:
        with background_monitor_threads(runner,4):runner.model.rhs_flat()
    assert caught.value is failure
    assert module.get_num_threads()==1 and runner.model.rhs_flat is bad


def test_pool_and_oversubscription_fail_before_rhs(monkeypatch):
    module,runner,calls,original=setup(monkeypatch,pool=1)
    with pytest.raises(ContractError,match='pool'):
        with background_monitor_threads(runner,4):pass
    assert not calls and runner.model.rhs_flat is original
    module.config.NUMBA_NUM_THREADS=4
    monkeypatch.setenv('OPENBLAS_NUM_THREADS','4')
    with pytest.raises(ContractError,match='BLAS'):
        with background_monitor_threads(runner,4):pass
    assert not calls and runner.model.rhs_flat is original


def test_unfused_candidate_refused_reference_remains(monkeypatch):
    module,runner,calls,original=setup(monkeypatch)
    runner.model._compiled_rhs.native_fused=False
    with pytest.raises(ContractError,match='fused'):
        with background_monitor_threads(runner,4):pass
    with background_monitor_threads(runner,1):assert runner.model.rhs_flat is original


@pytest.mark.parametrize('value',[True,0,8,1.0])
def test_invalid_requested_threads(value):
    with pytest.raises(ContractError):
        with background_monitor_threads(None,value):pass
