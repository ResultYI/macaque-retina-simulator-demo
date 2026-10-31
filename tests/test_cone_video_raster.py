import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from tools.cone_mosaic_demo import ROOT
from tools.cone_video_raster import prepare_raster


class ConeRasterTests(unittest.TestCase):
    def test_more_than_uint16_cones_preserves_cell_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as directory:
            xy=np.full((65536,2),10.)
            xy[-1]=0.
            path=Path(directory)/'raster.mat'
            prepare_raster(path,xy,np.full(len(xy),1000.),[-1.,1.,-1.,1.],8,1)
            labels=loadmat(path)['cone_ids']
            self.assertEqual(labels.dtype,np.dtype('uint32'))
            self.assertEqual(int(labels.max()),65536)

    def test_physical_area_and_positive_y_orientation(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as directory:
            path=Path(directory)/'raster.mat'
            prepare_raster(path,np.array([[0.,.5]]),[500.],[-1.,1.,-1.,1.],64,4)
            labels=loadmat(path)['cone_ids']
            self.assertEqual(labels[64,128],1)
            self.assertEqual(labels[192,128],0)
            area=np.count_nonzero(labels)*4/256**2
            self.assertLess(abs(area/(np.pi*.25**2)-1),.02)
            with self.assertRaises(FileExistsError):
                prepare_raster(path,np.array([[0.,.5]]),[500.],[-1.,1.,-1.,1.])
