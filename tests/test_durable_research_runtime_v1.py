"""No model, MATLAB or large allocations; tests actual Windows process lifetime."""
import json,os
from pathlib import Path
import subprocess,sys,tempfile,time,unittest
from tools.durable_research_runtime_v1 import inspect_run,inspect_identity,process_identity
ROOT=Path(__file__).resolve().parents[1]


class DurableRuntimeTests(unittest.TestCase):
    def test_normal_and_exception_terminal_receipts(self):
        with tempfile.TemporaryDirectory(prefix='retina_lifecycle_') as t:
            root=Path(t).resolve();self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            for mode in ['normal','fail']:
                out=root/mode
                p=subprocess.run([sys.executable,'-m','tools.runtime_probe_v1','--output',str(out),'--mode',mode],cwd=ROOT,
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW)
                self.assertEqual(p.returncode,0 if mode=='normal' else 1)
                state=json.loads((out/'runtime/terminal.json').read_bytes())
                self.assertEqual(state['status'],'PASS' if mode=='normal' else 'FAILED')
                self.assertEqual(inspect_run(out/'runtime')['decision'],'TERMINAL_RECORDED')

    def test_forced_exit_preserves_start_but_never_claims_exact_cause(self):
        with tempfile.TemporaryDirectory(prefix='retina_lifecycle_') as t:
            root=Path(t).resolve();self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            out=root/'killed'
            p=subprocess.Popen([sys.executable,'-m','tools.runtime_probe_v1','--output',str(out),'--mode','hold'],cwd=ROOT,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline=time.monotonic()+10
                while not (out/'runtime/state.json').is_file():
                    if time.monotonic()>deadline:raise TimeoutError('Dummy did not start')
                    time.sleep(.02)
                self.assertEqual(inspect_run(out/'runtime')['decision'],'BLOCK_ACTIVE_OR_UNCERTAIN_PROCESS')
                p.terminate();p.wait(timeout=5)
                self.assertFalse((out/'old_style_finally.json').exists())
                self.assertFalse((out/'runtime/terminal.json').exists())
                observation=inspect_run(out/'runtime')
                self.assertEqual(observation['decision'],'INTERRUPTED_UNKNOWN_EXIT_NO_AUTOMATIC_RETRY')
                self.assertFalse(observation['original_exit_cause_confirmed'])
            finally:
                if p.poll() is None:p.kill();p.wait(timeout=5)

    def test_pid_reuse_cannot_match_just_by_number(self):
        identity=process_identity(os.getpid());identity['creation_100ns']='0'
        self.assertEqual(inspect_identity(identity)['state'],'PID_REUSED')


if __name__=='__main__':unittest.main()
