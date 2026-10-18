"""Windows long completion-marker paths must count as actual batch progress."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch
from retina_platform.bounded_iset_batch import run_batch


@unittest.skipUnless(os.name=='nt', 'Windows completion-marker regression')
class LongMarkerTests(unittest.TestCase):
    def test_committed_long_marker_prevents_false_stall_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();directory=root/('d'*100);directory.mkdir();(root/'batch').mkdir()
            filename='m'*(264-len(str(directory))-1-len('.mat.ready'))+'.mat.ready'
            marker=directory/filename;extended=Path('\\\\?\\'+str(marker))
            extended.touch()
            try:
                class Process:
                    returncode=0
                    calls=0
                    def poll(self):
                        self.calls+=1
                        return None if self.calls==1 else 0
                statuses=[]
                profile={'matlab_executable':'unused','maximum_export_wall_seconds':300.}
                job={'configuration':str(root/'unused.json'),'output':str(marker)[:-6]}
                with patch('retina_platform.bounded_iset_batch.subprocess.Popen',return_value=Process()), \
                     patch('retina_platform.bounded_iset_batch.time.monotonic',side_effect=[0.,400.,400.,400.,400.]), \
                     patch('retina_platform.bounded_iset_batch.time.sleep'):
                    run_batch(root,profile,[job],root/'batch',statuses.append,lambda:True)
                self.assertEqual(statuses,['ISET fixed global tiles 1/1'])
            finally:
                extended.unlink()


if __name__=='__main__':
    unittest.main()
