import numpy as np
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.cone_mosaic_demo import aperture_support, sample_image, registration_check, main
from retina_sim.isetbio_geometry import cone_apertures


class ConeMosaicDemoTests(unittest.TestCase):
    def test_existing_output_is_rejected_before_model_loading(self):
        directory=str(Path(__file__).parent)
        with patch('sys.argv',['cone_mosaic_demo.py','--output',directory]), patch('tools.cone_mosaic_demo.load_geometry') as loader:
            with self.assertRaisesRegex(ValueError,'Output exists'):
                main()
            loader.assert_not_called()

    def test_reduced_support_preserves_full_master_apertures(self):
        x,y=np.meshgrid(np.arange(13)*.003,np.arange(11)*.003)
        full=np.column_stack((x.ravel(),y.ravel()))
        selected=np.array([70,71,83,84])
        support,active,diameters=aperture_support(full,selected)
        assert len(support)<len(full)
        np.testing.assert_array_equal(support[active],full[selected])
        np.testing.assert_allclose(cone_apertures(support)[active],cone_apertures(full)[selected],rtol=1e-12)
        with self.assertRaisesRegex(ValueError,'Unique'):
            aperture_support(full,[70,70])


    def test_image_top_maps_positive_y_without_stretching(self):
        image=np.zeros((2,4,3),dtype=np.uint8)
        image[0,:,0]=255;image[1,:,2]=255
        xx,yy=np.meshgrid([-1.5,-.5,.5,1.5],[.5,-.5,1.5])
        sampled=sample_image(image,[-2,2,-2,2],xx,yy,.5)
        np.testing.assert_allclose(sampled[0],np.tile([1,0,0],(4,1)))
        np.testing.assert_allclose(sampled[1],np.tile([0,0,1],(4,1)))
        np.testing.assert_allclose(sampled[2],.5)


    def test_axis_check_rejects_mirrored_response(self):
        x,y=np.meshgrid(np.linspace(-.45,.45,30),np.linspace(-.45,.45,30))
        xy=np.column_stack((x.ravel(),y.ravel()))
        types=np.tile(['L','M'],len(xy)//2)
        rates=2+xy[:,0]-xy[:,1]
        result=registration_check(xy,types,rates,np.ones(len(xy)),(100,100,3),[-.5,.5,-.5,.5])
        assert result['passed']
        with self.assertRaisesRegex(ValueError,'orientation'):
            registration_check(xy,types,4-rates,np.ones(len(xy)),(100,100,3),[-.5,.5,-.5,.5])
