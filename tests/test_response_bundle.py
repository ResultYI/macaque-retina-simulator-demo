"""Small synthetic package fixtures only; never presented as research results."""
from pathlib import Path
import io
import json
import tempfile
import unittest
from unittest.mock import Mock,patch
import zipfile
import numpy as np
from retina_platform.response_bundle import ResultSource,export_package,open_package,safe_name,sha
from retina_platform.response_data import ResponseData,ancestors,checked_npz,latest_index


def fixture(directory):
    files={}
    def array(name,**arrays):
        path=directory/(str(len(files))+'.npz');np.savez_compressed(path,**arrays);files[name]=path
    def document(name,value):
        path=directory/(str(len(files))+'.json');path.write_text(json.dumps(value),encoding='utf8');files[name]=path
    geometry={}
    populations={}
    for pop,n in [('cones',2),('HC',1),('midget_RGC_ON',2)]:
        geometry.update({pop+'__ids':np.array([pop+str(i) for i in range(n)]),pop+'__members':np.ones(n,int),
            pop+'__xy':np.column_stack([np.arange(n)*.1,np.zeros(n)]),pop+'__subtypes':np.full(n,pop)})
        primary=pop+'__voltage' if pop=='midget_RGC_ON' else 'ports__'+pop+':'+('release' if pop=='cones' else 'Vh')
        populations[pop]=dict(count=n,primary=primary,fields={primary:'V' if pop=='midget_RGC_ON' else 'dimensionless'},
                              coordinate_role='rf',representation='synthetic_test_group')
    array('cells/p/geometry.npz',**geometry)
    # RGC0 <- Cone0 <- HC0 <- Cone1; RGC1 is not an ancestor.
    routes=[dict(token='r0',name='c_r',pre='cones',post='midget_RGC_ON',kind='chemical',shape=[2,2]),
            dict(token='r1',name='h_c',pre='HC',post='cones',kind='native',shape=[2,1]),
            dict(token='r2',name='c_h',pre='cones',post='HC',kind='native',shape=[1,2])]
    array('cells/p/connections.npz',r0_indptr=np.array([0,1,2]),r0_indices=np.array([0,1]),r0_weights=np.ones(2),
        r1_indptr=np.array([0,1,1]),r1_indices=np.array([0]),r1_weights=np.ones(1),
        r2_indptr=np.array([0,1]),r2_indices=np.array([1]),r2_weights=np.ones(1))
    values={meta['primary']:np.tile(np.array([1.,2.,3.]),(meta['count'],1)) for meta in populations.values()}
    array('responses/p/part.npz',time_seconds=np.array([0.,.001,.002]),side=np.array(['right']*3),**values)
    document('responses/p/RESPONSE_INDEX.json',dict(chunks=[dict(file='part.npz',records=3,sha256=sha(files['responses/p/part.npz']))]))
    document('responses/p/IDENTITY.json',dict(types={'midget_RGC_ON':dict(group_ids=['midget_RGC_ON0','midget_RGC_ON1'],represented_members=[1,1])}))
    document('responses/p/GROUP_OBJECT_EVENTS.json',[])
    array('stimulus/frames.npz',rgb=np.zeros((2,4,4,3),np.uint8))
    manifest=dict(schema='retina-response-bundle-1',status='complete',run_id='SYNTHETIC_UI_TEST_ONLY',
        stimulus=dict(frames='stimulus/frames.npz',frame_count=2,fps=1000.,duration_seconds=.002),
        positions={'p':dict(populations=populations,routes=routes,geometry='cells/p/geometry.npz',connections='cells/p/connections.npz',response_prefix='responses/p/')})
    return ResultSource(manifest,files)


class Packages(unittest.TestCase):
    def test_portable_roundtrip_true_ancestors_and_causal_sample(self):
        with tempfile.TemporaryDirectory() as raw,tempfile.TemporaryDirectory() as elsewhere:
            source=fixture(Path(raw));target=Path(elsewhere)/'result.zip';export_package(source,target)
            # Read exported data after original assets have disappeared.
            for path in source.files.values():path.unlink()
            loaded=open_package(target)
            try:
                data=ResponseData(loaded,'p');network,direct=data.network(('midget_RGC_ON',0))
                self.assertEqual(network['cones'].tolist(),[0,1]);self.assertEqual(network['midget_RGC_ON'].tolist(),[0])
                self.assertIn(('cones',0),direct);self.assertNotIn(('HC',0),direct)
                self.assertEqual(latest_index(data.time,.0019),1)
                self.assertEqual(data.field('cones','ports__cones:release')[0,1],2.)
                self.assertEqual(data.frames.shape,(2,4,4,3))
            finally:loaded.close()

    def test_paths_symlink_duplicates_and_integrity(self):
        for path in ['../evil','/absolute','C:/drive','a\\b','a/../b','a/CON.txt','a/b.','a//b']:
            with self.assertRaises(ValueError):safe_name(path)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=fixture(root);good=root/'good.zip';export_package(source,good)
            bad=root/'bad.zip'
            with zipfile.ZipFile(good) as original,zipfile.ZipFile(bad,'w') as archive:
                for info in original.infolist():
                    payload=original.read(info)
                    if info.filename=='stimulus/frames.npz':payload=payload[:-1]+bytes([payload[-1]^1])
                    archive.writestr(info.filename,payload)
            with self.assertRaisesRegex(ValueError,'完整性'):open_package(bad)
            traversal=root/'slip.zip'
            with zipfile.ZipFile(traversal,'w') as archive:archive.writestr('../escape.txt','bad')
            with self.assertRaises(ValueError):open_package(traversal)
            self.assertFalse((root.parent/'escape.txt').exists())

    def test_partial_existing_target_and_object_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=fixture(root);target=root/'existing.zip';target.write_bytes(b'old')
            with self.assertRaises(FileExistsError):export_package(source,target)
            self.assertEqual(target.read_bytes(),b'old')
            source.manifest['status']='partial'
            with self.assertRaisesRegex(ValueError,'partial'):export_package(source,root/'partial.zip')
            self.assertFalse((root/'partial.zip').exists())
            array=root/'object.npz';np.savez(array,x=np.array([object()],dtype=object))
            with self.assertRaises(ValueError):checked_npz(array)
            bomb=root/'header_bomb.npz';header=io.BytesIO()
            np.lib.format.write_array_header_1_0(header,dict(descr='<f8',fortran_order=False,shape=(2**30,)))
            with zipfile.ZipFile(bomb,'w') as archive:archive.writestr('huge.npy',header.getvalue())
            with self.assertRaises(ValueError):checked_npz(bomb)


class AutomaticExport(unittest.TestCase):
    """Exercise pipeline dispatch with mocked compute; never simulate a model."""
    def test_complete_pipeline_dispatches_package_last(self):
        from tools import run_external_video_pipeline_v1 as pipeline
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_PIPELINE_TEST_') as directory:
            out=Path(directory);(out/'OPTICAL_RESOURCE.json').write_text('{"status":"PASS"}',encoding='utf8')
            stages=[]
            def execute(folder,name,command,limit,journal):
                stages.append((name,list(command)))
            with patch.object(pipeline,'execute',execute),patch.object(pipeline,'DurableRun',return_value=Mock()):
                pipeline.pipeline(out,True)
            self.assertEqual([name for name,_ in stages][-2:],['viewer','package'])
            self.assertIn(pipeline.ROOT/'tools/export_response_bundle.py',stages[-1][1])
            self.assertEqual(json.loads((out/'PIPELINE_STATUS.json').read_text())['status'],'COMPLETE')

    def test_failed_pipeline_marks_partial_without_packaging(self):
        from tools import run_external_video_pipeline_v1 as pipeline
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_PIPELINE_TEST_') as directory:
            out=Path(directory);(out/'OPTICAL_RESOURCE.json').write_text('{"status":"PASS"}',encoding='utf8')
            stages=[]
            def execute(folder,name,command,limit,journal):
                stages.append(name)
                if name=='near':raise RuntimeError('SYNTHETIC_COMPUTE_FAILURE_NO_MODEL')
            with patch.object(pipeline,'execute',execute),patch.object(pipeline,'DurableRun',return_value=Mock()):
                with self.assertRaises(RuntimeError):pipeline.pipeline(out,True)
            self.assertNotIn('package',stages)
            self.assertFalse(json.loads((out/'PARTIAL_RESPONSE_EXPORT.json').read_text())['complete_package'])


if __name__=='__main__':unittest.main()
