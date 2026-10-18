"""Prepare a tiny explicit ISET engineering fixture from frozen project cone IDs.

Invented optics/spectrum/apertures are only a dimension/API test, never defaults.
The JSON+v7 MAT pair is also the standalone exporter's executable field example.
"""
from pathlib import Path
import argparse,sys,json,hashlib
import numpy as np
from scipy.io import savemat

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from retina_platform.config import RunConfig
    from retina_sim.atlas_population_network import atlas_population_network
    from retina_sim.offline_input import load_offline_inputs
    from retina_sim.isetbio_input import project_rf_equals_is_binding
    cfg=RunConfig.project_default(root);light=cfg.light_context()
    compiled,_,audit=atlas_population_network(root,light_context=light,owner_bounds_mm=cfg.owner_bounds(),
        halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,seed=cfg.seed)
    ref=audit['atlas_runtime']['source_atlas']['reference_id']
    load_offline_inputs(args.template,compiled,light,reference_id=ref)
    with np.load(args.template,allow_pickle=False) as data:
        template=json.loads(data['manifest_json'].item());d=template['channels']['native_light'];edges=data[d['time']['axis_key']].copy()
    pop=compiled.populations[d['population_id']];rf=pop.receptive_field_positions.values
    binding=project_rf_equals_is_binding(pop,reference_id=ref);ispos=binding['positions_mm']
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    wave=np.arange(400.,701.,50.);t=len(edges)-1;center=np.mean(ispos,axis=0)
    absorbance=np.exp(-.5*((wave[:,None]-[560.,530.,440.])/40.)**2)
    scene=np.ones((128,128,len(wave),t))*1e15
    for k in range(t):scene[:,:,:,k]*=1+.1*k
    vendor=Path(r'D:\retipath\experiments\retipath_foveal_v1\exploratory_v2\vendor_pinned')
    base={'schema':'isetbio_capture_export_v1','scope':'engineering_fixture','source_id':'explicit_ISET_API_dimension_fixture_v1',
        'isetcam_root':str(vendor/'isetcam_17146a9f'),'isetbio_root':str(vendor/'isetbio_d2924c2f'),
        'scene':{'unit':'photons/(s m^2 sr nm)','axes':['row','column','wavelength','window'],
            'horizontal_fov_deg':1.,'distance_m':1e6,'source_id':'invented_uniform_spectral_radiance_fixture'},
        'optics':{'mode':'diffraction_limited_explicit','pupil_diameter_mm':3.,'focal_length_m':.223e-3/np.tan(np.pi/180),
            'offaxis_method':'skip','source_id':'invented_diffraction_limited_fixture',
            'model_assumption':'Software fixture only; focal length chosen to match existing project 223um/deg near-axis transfer'},
        'capture':{'source_id':'invented_spectral_aperture_fixture','blur_diameter_um':3.,'aperture_shape':'Pillbox',
            'optical_density_LMS':[.5,.5,.4],'peak_efficiency_LMS':[2/3]*3,'macular_density':0.,'pigment_diameter_m':2e-6},
        'coordinates':{'reference_id':ref,'registration':'frozen_project_RF_IS_verbatim; project 223um_per_deg transfer; ISET image rows internally flipped to y-up',
            'microns_per_degree':223.,'which_eye':'right eye','optical_image_position_deg':(center/.223).tolist(),
            'is_binding_id':binding['transfer_id'],'is_binding_source_sha256':binding['source_sha256'],
            'is_role':'inner_segment','is_frame':'anatomical_retina','rf_role':'rf','rf_frame':'functional_visual_field'},
        'time':{'source_origin_seconds':d['time']['source_origin_seconds'],'window_edges_seconds':edges.tolist(),
            'frame_semantics':'constant_radiance_within_each_window'}}
    for name,ids,types,positions,rfpos,diameters in [
        ('capture',list(pop.cells.cell_id),list(pop.cells.subtype),ispos,rf,np.full(len(rf),2.)),
        ('reference',['fixture_ref_L','fixture_ref_M','fixture_ref_S'],['L','M','S'],np.repeat(center[None,:],3,axis=0),np.repeat(center[None,:],3,axis=0),np.full(3,2.))]:
        if name=='capture':diameters[-1]=2.5 # Explicit area difference must survive the fixed class bridge.
        mat=out/(name+'_input.mat');config=json.loads(json.dumps(base))
        if mat.exists() or (out/(name+'_config.json')).exists():raise FileExistsError('Choose a fresh fixture output directory')
        savemat(mat,{'scene_photons':scene,'wave_nm':wave,'is_mm':positions,'rf_mm':rfpos,
            'diameters_um':diameters,'os_factors':np.ones(len(ids)),'absorbance_LMS':absorbance,
            'macular_unit_density':np.ones(len(wave)),'lens_transmittance':np.ones(len(wave))},do_compression=True)
        config.update(input_mat=str(mat),input_mat_sha256=hashlib.sha256(mat.read_bytes()).hexdigest(),
            cones={'cell_ids':ids,'types':types,'population_id':pop.population_id,'evidence_tag':d['evidence_tag']})
        (out/(name+'_config.json')).write_text(json.dumps(config,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'scope':'engineering_fixture','cones':len(rf),'windows':t,'output':str(out),'physiological_calibration':False}))

if __name__=='__main__':main()
