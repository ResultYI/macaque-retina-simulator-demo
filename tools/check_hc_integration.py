"""Small HC integration acceptance through real GUI parsing and backend export.

All footprint/sampling choices in this check are engineering test fixtures.
They are never written into the physiological default configuration.
"""
from pathlib import Path
import argparse
import hashlib
import json
import queue
import sys
import time
import tkinter as tk
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from retina_platform.app import RetinaApp
from retina_platform.backend import SimulationSession
from retina_platform.config import RunConfig
from retina_platform.media import MediaSource
from retina_sim.hc_mosaic import HCDensityProfile
from retina_sim.regional_support import load_regional_support
from retina_sim.vh2005 import NativeConeH1, SOURCE_VALUES
import examples.full_target_network as network


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-run',type=Path,
        help='Validate an existing completed backend run using a deterministic build; no replay')
    arguments=parser.parse_args()
    output=arguments.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    image=output/'engineering_uniform_rgb.png'
    Image.new('RGB',(16,16),(128,128,128)).save(image)
    geometry=output/'engineering_geometry_fixture.json'
    footprint={'kind':'ellipse','axes_mm':[.06,.05],'angle_rad':.25,
        'unit':'mm','coordinate_frame':'OPL',
        'source_profile_id':'HC_integration_engineering_fixture_only',
        'evidence_class':'project_approximation','approximation_id':'HC_user_finite_footprint_v1',
        'scope':'software integration test only; no physiological radius or morphology claim'}
    geometry.write_text(json.dumps({'profile_id':'hc_joint_synthetic_v1',
        'generator':{'sweeps':2},'footprints':{'H1':footprint,'H2':footprint}},indent=2)+'\n',encoding='utf-8')
    requested=RunConfig(media_path=str(image),duration_seconds=.0002,warmup_seconds=0.,
        window_seconds=.0002,sample_seconds=.0001,
        hc_geometry_profile='hc_joint_synthetic_v1',hc_geometry_path=str(geometry),
        hc_h1_radius_mm=.04,hc_h2_radius_mm=None)
    # Tcl StringVars and the actual GUI parser, without opening a desktop window.
    interpreter=tk.Tcl()
    app=RetinaApp.__new__(RetinaApp)
    app.config=requested
    app.regional_support=load_regional_support(ROOT)
    app.roi_labels={key:row['label'] for key,row in app.regional_support['roi_presets'].items()}
    app.params={name:tk.StringVar(master=interpreter) for name in
        [row[0] for row in RetinaApp.PARAMS]+['media_path','gaze_csv','hc_geometry_path']}
    app.gaze_label=SimpleNamespace(configure=lambda **kwargs:None)
    RetinaApp._set_config_vars(app,requested)
    with MediaSource(image) as media:
        app.media=media
        parsed=RetinaApp._read_config(app)
    assert parsed==requested,'GUI fields changed the supplied test configuration'
    (output/'gui_parsed_config.json').write_text(json.dumps(parsed.to_dict(),indent=2)+'\n',encoding='utf-8')
    captured={}
    original=network.full_target_network
    def build(*args,**kwargs):
        captured['kwargs']=kwargs
        result=original(*args,**kwargs)
        captured['compiled'],_,captured['audit']=result
        return result
    network.full_target_network=build
    session=SimulationSession(parsed,ROOT,output/'joint_run')
    started=time.perf_counter()
    try:
        if arguments.verify_run:
            kwargs={'effective_cones':int(parsed.effective_cones),'effective_rods':int(parsed.effective_rods),
                'seed':int(parsed.seed),'duration':parsed.duration_seconds+parsed.warmup_seconds,
                'adaptation_seconds':parsed.warmup_seconds,'light_context':parsed.light_context(),
                'runtime_profile':parsed.parameter_profile,'execution_profile':parsed.execution_profile,
                'owner_bounds_mm':parsed.owner_bounds(),'halo_mm':parsed.halo_mm,
                'max_spatial_cells':int(parsed.max_spatial_cells),
                **{key:getattr(parsed,key) for key in ('hc_geometry_profile','hc_geometry_path','hc_h1_radius_mm','hc_h2_radius_mm')}}
            build(ROOT,**kwargs)
            session=SimpleNamespace(path=arguments.verify_run.resolve(),status='completed')
        else:
            session.start()
            while session.is_alive() or not session.events.empty():
                if time.perf_counter()-started>180:
                    session.stop()
                    raise RuntimeError('Bounded integration check exceeded 180 seconds')
                try:event=session.events.get(timeout=.5)
                except queue.Empty:continue
                print(json.dumps({key:event[key] for key in ('kind','status','message','path')},ensure_ascii=False),flush=True)
        assert session.status=='completed', (session.path/'error.txt').read_text(encoding='utf-8') if session.path and (session.path/'error.txt').is_file() else session.status
    finally:
        network.full_target_network=original
    for key in ('hc_geometry_profile','hc_geometry_path','hc_h1_radius_mm','hc_h2_radius_mm'):
        assert captured['kwargs'][key]==getattr(parsed,key),key
    compiled=captured['compiled']
    run=session.path
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    assert summary['status']=='completed'
    assert compiled.assembly_hash==manifest['assembly_hash'],'Rebuild differs from the actual exported assembly'
    assert manifest['config']==parsed.to_dict()
    hc=json.loads((run/'hc_geometry.json').read_text(encoding='utf-8'))
    assert hc==manifest['spatial_profile']['HC_geometry']
    assert hc['configuration']['footprints']['H1']['kind']=='disk'
    assert hc['configuration']['footprints']['H1']['radius_mm']==.04
    assert hc['configuration']['footprints']['H2']['kind']=='ellipse'
    assert not summary['unimplemented_layers']
    native=compiled.components['outer_native'].native_model
    assert isinstance(native,NativeConeH1) and dict(native.p)==SOURCE_VALUES
    a=native.A.tocoo();bt=native.B.T.tocoo()
    assert set(zip(a.row,a.col))==set(zip(bt.row,bt.col))
    pairs=hc['contact_audit']['paired_cones']
    cones=compiled.populations['cones']
    assert [row['cone_id'] for row in pairs]==list(cones.cells.cell_id)
    np.testing.assert_array_equal([row['input_xy_mm'] for row in pairs],cones.receptive_field_positions.values)
    np.testing.assert_array_equal([row['pedicle_xy_mm'] for row in pairs],cones.pedicle_positions.values)
    instance=hc['canonical_instance']
    master={cell_id:(kind,xy) for cell_id,kind,xy in zip(instance['cell_ids'],instance['types'],instance['soma_xy_mm'])}
    assert len(master)==hc['master']['canonical_cell_count']
    counts=[instance['types'].count(kind) for kind in ('H1','H2')]
    assert counts==hc['master']['total_H1_H2_counts']
    assert np.all(np.abs(np.asarray(counts)-hc['master']['total_H1_H2_mass'])<1.)
    for partition in hc['master']['partitions']:
        assert np.all(np.abs(np.asarray(partition['H1_H2_integer_quota'])-partition['H1_H2_mass'])<1.)
        actual=[sum(cell_id.split(':')[-2]==partition['id'] and kind==wanted
            for cell_id,kind in zip(instance['cell_ids'],instance['types'])) for wanted in ('H1','H2')]
        assert actual==partition['H1_H2_integer_quota']
    dynamic=compiled.populations['HC']
    for cell_id,kind,xy in zip(dynamic.cells.cell_id,dynamic.cells.subtype,dynamic.anatomical_positions.values):
        expected_type,expected_xy=master[cell_id]
        assert kind==expected_type
        np.testing.assert_array_equal(xy,expected_xy)
    assert not hc['HC_is_displaced_by_cone_map']
    assert hc['runtime_view']['density_instance_thinned'] is False
    assert hc['gap_candidate_audit']['connected_to_dynamics'] is False
    profile=HCDensityProfile(ROOT,profile_data=hc['configuration'])
    density_bounds=profile.bounds_for_rectangle(hc['canonical_bounds_mm'])
    assert density_bounds['total_max']>density_bounds['total_min']>0
    assert max(np.linalg.norm(instance['soma_xy_mm'],axis=1))<=7.1
    windows=list((run/'windows').glob('window_*.npz'))
    assert len(windows)==1 and summary['windows']==1
    numeric=0
    for file in windows:
        with np.load(file,allow_pickle=False) as data:
            for key in data.files:
                value=data[key]
                if value.dtype.kind in 'biufc':
                    assert np.all(np.isfinite(value)),key
                    numeric+=value.size
    assert numeric>0
    assert (run/'initial_checkpoint.json').is_file() and (run/'final_checkpoint.json').is_file()
    assert json.loads((ROOT/'configurations/hc_joint_synthetic_v1.json').read_text())['footprints']=={'H1':None,'H2':None}
    result={'engineering_integration':'PASS','run':str(run),'GUI_configuration_path':'real Tcl variables -> RetinaApp._read_config -> SimulationSession -> full_target_network',
        'GUI_builder_fields_checked':['hc_geometry_profile','hc_geometry_path','hc_h1_radius_mm','hc_h2_radius_mm'],
        'simulation_seconds':summary['simulation_time_seconds'],'wall_seconds':time.perf_counter()-started,
        'actual_run_wall_seconds':summary['wall_seconds'],
        'verification_mode':'existing completed backend run; deterministic build, no replay' if arguments.verify_run else 'new backend run',
        'components':summary['components'],'target_layers':len(summary['required_layers']),
        'canonical_bounds_mm':hc['canonical_bounds_mm'],'canonical_H1_H2_counts':counts,
        'density_mass_H1_H2':hc['master']['total_H1_H2_mass'],'density_bounds_cells_mm2':density_bounds,
        'active_HC_count':hc['runtime_view']['generated_active_count'],'dynamic_HC_count':len(dynamic.cells.cell_id),
        'geometry_only_HC_count':len(hc['runtime_view']['geometry_only_cell_ids']),
        'finite_contact_edges':hc['contact_audit']['edge_count'],'cone_count':len(cones.cells.cell_id),
        'feedback_topology':'same finite contact pairs','native_equation_class':type(native).__name__,
        'native_source_parameters_unchanged':True,'HC_gap_candidates_connected_to_dynamics':False,
        'finite_recorded_numeric_values':numeric,'mosaic_sweeps':2,
        'mosaic_acceptance_fraction':hc['master']['acceptance_fraction'],
        'mosaic_mixing_certified':False,'physiological_or_central_arbor_validation':'NOT_RUN',
        'default_arbor_footprints_unchanged':True,
        'native_implementation_sha256':hashlib.sha256((ROOT/'retina_sim/vh2005.py').read_bytes()).hexdigest()}
    (output/'integration_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
