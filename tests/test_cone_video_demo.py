import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tools.cone_video_demo import frame_timeline, main
from tools.cone_mosaic_demo import ROOT, load_central_geometry, load_geometry, load_saved_central_geometry
from retina_sim.isetbio_geometry import FixedISETConeMaster, cone_apertures
from tools.cone_central_video_demo import sample_cover


class ConeVideoTests(unittest.TestCase):
    def test_saved_geometry_rejects_changed_recipe_before_loading_payload(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as directory:
            path=Path(directory)
            (path/'geometry.json').write_text(json.dumps({'recipe_hashes':{}}))
            with self.assertRaisesRegex(ValueError,'recipe changed'):
                load_saved_central_geometry(ROOT,path)

    def test_cover_crops_sides_without_gray_or_geometric_stretch(self):
        image=np.zeros((5,9,3),dtype=np.uint8)
        image[:,:,:]=[51,102,153]
        image[:,0]=255;image[:,-1]=255
        x,y=np.meshgrid(np.linspace(-2,2,5),np.linspace(2,-2,5))
        result=sample_cover(image,[-2,2,-2,2],x,y,.73535698)
        np.testing.assert_allclose(result,np.broadcast_to([.2,.4,.6],result.shape),rtol=0,atol=1e-15)
        image[:,:,0]=np.arange(9)[None,:]*20
        image[:,:,1]=np.arange(5)[:,None]*20
        result=sample_cover(image,[-2,2,-2,2],x,y,.73535698)
        self.assertAlmostEqual(result[2,3,0]-result[2,2,0],result[1,2,1]-result[0,2,1])

    def test_central_selection_preserves_frozen_geometry_and_global_ids(self):
        _, old_ids, old_types, *_ = load_geometry(ROOT)
        old = dict(zip(old_ids,old_types))
        def labels(partition,table):
            return tuple(old.get(gid,'LMS'[i%3]) for i,gid in enumerate(table.global_cell_ids)), {'test_labels':True}
        with patch('retina_sim.atlas_s_partition.EffectiveMulattaSPartition.assign',labels):
            xy,ids,types,diameters,geometry=load_central_geometry(ROOT)
        master=FixedISETConeMaster(ROOT,geometry['master'])
        keep=np.linalg.norm(master.positions,axis=1)<=.446
        self.assertEqual(len(ids),45480)
        np.testing.assert_array_equal(xy,master.positions[keep])
        np.testing.assert_allclose(diameters,cone_apertures(master.positions)[keep],rtol=1e-12)
        self.assertTrue(geometry['previous_local_overlap_types_unchanged'])
        self.assertGreater(geometry['previous_local_overlap_cones'],0)
        self.assertLessEqual(geometry['eccentricity_range_deg'][1],2.)

    def test_exact_second_has_each_source_frame_once(self):
        indices, edges=frame_timeline(30.,447,0.)
        np.testing.assert_array_equal(indices,np.arange(30))
        self.assertEqual(edges[0],0.)
        self.assertEqual(edges[-1],1.)
        np.testing.assert_allclose(np.diff(edges),1/30,rtol=0,atol=1e-15)
        indices, shifted=frame_timeline(30.,447,2.)
        np.testing.assert_array_equal(indices,np.arange(60,90))
        np.testing.assert_array_equal(shifted,edges)

    def test_short_video_or_unsupported_timeline_is_not_padded(self):
        for fps,count,start in [(50.,6,0.),(29.97,500,0.),(30.,447,.01),(0.,30,0.),(30.,30,-1.)]:
            with self.subTest(fps=fps,count=count,start=start), self.assertRaises(ValueError):
                frame_timeline(fps,count,start)

    def test_four_seconds_preserve_the_one_second_prefix(self):
        indices,edges=frame_timeline(30.,447,0.,4.)
        one_indices,one_edges=frame_timeline(30.,447,0.)
        np.testing.assert_array_equal(indices,np.arange(120))
        np.testing.assert_array_equal(indices[:30],one_indices)
        np.testing.assert_array_equal(edges[:31],one_edges)
        self.assertEqual(edges[-1],4.)
        np.testing.assert_allclose(np.diff(edges),1/30,rtol=0,atol=5e-16)
        for duration in (0.,-4.,float('nan'),float('inf'),4.001,15.):
            with self.subTest(duration=duration),self.assertRaises(ValueError):
                frame_timeline(30.,447,0.,duration)

    def test_existing_output_is_rejected_before_loading_or_capture(self):
        with patch('sys.argv',['cone_video_demo.py','--media','unused.avi','--output',str(Path(__file__).parent)]), \
             patch('tools.cone_video_demo.source_signature') as signature:
            with self.assertRaisesRegex(ValueError,'Output exists'):
                main()
            signature.assert_not_called()
