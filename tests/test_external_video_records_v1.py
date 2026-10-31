"""Time-chunk storage retains exact real arrays and both sides of a boundary."""
import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from tools.external_video_records_v1 import open_response


class SavedVideoRecordsTests(unittest.TestCase):
    def test_real_event_side_boundary_survives_chunking(self):
        root=Path(__file__).resolve().parents[1]
        source=root/'reports/external_first_frame_two_positions_20261029_v1/near_run_v2/RESPONSE.npz'
        with np.load(source,allow_pickle=False) as z:
            original={k:z[k].copy() for k in ['time_seconds','side','midget_RGC_ON__voltage']}
        cut=int(np.flatnonzero(np.diff(original['time_seconds'])==0)[0])+1
        with tempfile.TemporaryDirectory(prefix='retina_video_records_') as temporary:
            folder=Path(temporary).resolve()
            self.assertTrue(folder.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            chunks=[]
            for i,part in enumerate([slice(None,cut),slice(cut,None)]):
                path=folder/f'part{i}.npz'
                np.savez_compressed(path,**{k:v[part] if k in ('time_seconds','side') else v[:,part] for k,v in original.items()})
                chunks.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            (folder/'RESPONSE_INDEX.json').write_text(json.dumps(dict(chunks=chunks,records=len(original['time_seconds']))))
            with open_response(folder) as restored:
                for key,value in original.items():np.testing.assert_array_equal(restored[key],value)


if __name__=='__main__':unittest.main()
