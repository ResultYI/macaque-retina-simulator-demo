"""Bounded grouped requests and exact capture views, without running MATLAB."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from scipy.io import loadmat, savemat
from retina_sim.contracts import ContractError
from retina_platform.grouped_disk_capture import SCHEMA, group_missing_jobs, read_group_frame


class GroupedCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.scope = 'a'*64
        self.jobs, self.labels, self.sources = [], {}, []
        for index in range(2):
            directory = self.root/('single_%d'%index);directory.mkdir()
            data = {'rgb_frames':np.full((2,2,3,1), .25+.1*index, dtype=np.float32),
                    'window_edges_seconds':np.array([0.,.0001]),'frame_indices':np.array([1.]),
                    'reference_rgb':np.ones(3)*.5,'is_mm':np.array([[0.,0.],[.01,0.]]),
                    'types':np.array([1.,2.]),'frozen_apertures_um':np.array([2.,3.]),
                    'image_center_deg':np.zeros(2)}
            source=directory/'input.mat';savemat(source,data)
            config=directory/'export.json';config.write_text(json.dumps({
                'profile':{},'global_reference':{'definition_sha256':self.scope},
                'background_signature':self.scope,'request_signature':'legacy_%d'%index,
                'patch_fov_deg':1.,'input_mat':str(source)}),encoding='utf8')
            self.jobs.append({'configuration':str(config),'output':str(directory/'unused.mat')})
            self.labels[str(config)]={'tile':{'tile_index':[0,0]},'exposure_index':index,
                                     'video_time_seconds':index*.0334,'decoded_frame_index':index,'gaze_deg':[0.,0.]}
            self.sources.append(data)

    def prepare(self, **kw):
        return group_missing_jobs(self.root,{},self.jobs,self.labels,self.scope,**kw)

    def test_pack_preserves_frozen_arrays_and_real_video_mapping(self):
        before=[Path(j['configuration']).with_name('input.mat').read_bytes() for j in self.jobs]
        jobs,views,summary=self.prepare()
        self.assertEqual(len(jobs),1)
        config=json.loads(Path(jobs[0]['configuration']).read_bytes())
        grouped=loadmat(config['input_mat'])
        self.assertEqual(config['grouped_capture']['schema'],SCHEMA)
        self.assertIn('grouped_tiles_v1',config['input_mat'])
        np.testing.assert_array_equal(grouped['rgb_frames'],np.concatenate([x['rgb_frames'] for x in self.sources],axis=3))
        np.testing.assert_array_equal(grouped['window_edges_seconds'].reshape(-1),[0.,.0001,.0002])
        for key in ('is_mm','types','frozen_apertures_um','reference_rgb','image_center_deg'):
            np.testing.assert_array_equal(grouped[key],loadmat(Path(self.jobs[0]['configuration']).with_name('input.mat'))[key])
        self.assertEqual([r['video_time_seconds'] for r in summary['groups'][0]['frames']],[0.,.0334])
        self.assertEqual([v['capture_frame_index'] for v in views.values()],[0,1])
        self.assertEqual(before,[Path(j['configuration']).with_name('input.mat').read_bytes() for j in self.jobs])

    def test_mismatched_geometry_cannot_be_grouped(self):
        self.sources[1]['is_mm'][0,0]=.1
        savemat(Path(self.jobs[1]['configuration']).with_name('input.mat'),self.sources[1])
        with self.assertRaisesRegex(ContractError,'frozen geometry'):
            self.prepare()

    def test_budget_split_does_not_drop_or_reorder_exposures(self):
        jobs,views,summary=self.prepare(max_rgb_bytes=self.sources[0]['rgb_frames'].nbytes)
        self.assertEqual(len(jobs),2)
        self.assertEqual([v['frame_source']['exposure_index'] for v in views.values()],[0,1])
        self.assertEqual(summary['missing_legacy_frame_requests'],2)
        self.assertTrue(all(v['capture_frame_index']==0 for v in views.values()))

    def test_frame_view_is_exact_and_does_not_impersonate_legacy_request(self):
        jobs,views,_=self.prepare();view=list(views.values())[1]
        config=json.loads(Path(jobs[0]['configuration']).read_bytes())
        meta={'schema':'approved_mixed_capture_v1','request_signature':view['request_signature'],
              'global_reference_definition_sha256':self.scope}
        savemat(Path(view['directory'])/'capture.mat',{
            'manifest_json':json.dumps(meta),'expected_counts':np.array([[1.,2.],[3.,4.]]),
            'background_counts':np.array([[5.,6.]]),'q_ref_LMS':np.array([[7.,8.,9.]]),
            'window_edges_seconds':np.array([0.,.0001,.0002]),'diameters_um':np.array([[2.,3.]]),
            'fixed_radiance_gain_numeric':np.array([[1.]])})
        raw,record=read_group_frame(view)
        np.testing.assert_array_equal(raw['expected_counts'],[[3.,4.]])
        np.testing.assert_array_equal(raw['window_edges_seconds'],[[0.,.0001]])
        self.assertNotEqual(record['request_signature'],'legacy_1')
        self.assertEqual(record['frame_source']['single_request_signature'],'legacy_1')
        self.assertEqual(record['capture_frame_index'],1)


if __name__=='__main__':
    unittest.main()
