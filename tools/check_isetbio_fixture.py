"""Check an actual MATLAB export/conversion against frozen IDs and existing input contracts."""
from pathlib import Path
import argparse,sys,json,hashlib
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fixture',type=Path);p.add_argument('--template',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from retina_platform.config import RunConfig
    from retina_sim.atlas_population_network import atlas_population_network
    from retina_sim.offline_input import load_offline_inputs
    from retina_sim.isetbio_input import read_isetbio_capture,reference_equivalent_drive
    from retina_sim.photometry import photon_irradiance_to_energy
    from retina_sim.contracts import Quantity
    cfg=RunConfig.project_default(root);light=cfg.light_context()
    compiled,_,audit=atlas_population_network(root,light_context=light,owner_bounds_mm=cfg.owner_bounds(),
        halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,seed=cfg.seed)
    ref=audit['atlas_runtime']['source_atlas']['reference_id'];out=args.fixture
    old=load_offline_inputs(args.template,compiled,light,reference_id=ref)
    packet=load_offline_inputs(out/'input.npz',compiled,light,reference_id=ref)
    cap=read_isetbio_capture(out/'capture.mat');reference=read_isetbio_capture(out/'reference_fixed.mat')
    profile=json.loads((out/'reference_profile.json').read_bytes())
    assert profile['reference_capture_sha256']==reference['sha256']
    rates=cap['counts']/np.diff(cap['edges'])[:,None]
    drive,bridge=reference_equivalent_drive(rates,cap['manifest']['cones']['types'],profile,
        capture_scope_sha256=cap['manifest']['capture_scope_sha256'])
    for k,t in enumerate((cap['edges'][:-1]+cap['edges'][1:])/2):
        np.testing.assert_allclose(packet.channels['native_light'].at(float(t)).values,drive[k],rtol=2e-15,atol=0)
        for name in old.channels:
            if name!='native_light':np.testing.assert_array_equal(old.channels[name].at(float(t)).values,packet.channels[name].at(float(t)).values)
    np.testing.assert_allclose(rates*np.diff(cap['edges'])[:,None],cap['counts'],rtol=2e-15,atol=0)
    for kind in ('L','M','S'):
        idx=[i for i,x in enumerate(cap['manifest']['cones']['types']) if x==kind]
        np.testing.assert_allclose(drive[:,idx]*bridge['alpha_rstar_s_per_reference_td_LMS'][kind],rates[:,idx],rtol=2e-15)
    photons=np.moveaxis(cap['retinal_photons'],2,-1)
    energy=photon_irradiance_to_energy(Quantity(photons,'photons/(s m^2 nm)',
        'retinal_spectral_photon_irradiance','project_approximation'),
        Quantity(cap['wave_nm'],'nm','wavelength','project_approximation'))
    np.testing.assert_allclose(energy.values/(6.62607015e-34*299792458/(cap['wave_nm']*1e-9)),photons,rtol=2e-15,atol=0)
    matlab=json.loads((out/'matlab_checks.json').read_bytes())
    assert matlab['missing']['identifier']=='ISETExport:MissingFields' and matlab['coarse']['identifier']=='ISETExport:Resolution'
    protected=json.loads((root/'reports/offline_input_smoke_20261013_clockfix/check.json').read_bytes())['protected_files_unchanged']
    assert protected=={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in protected}
    result={'status':'PASS','actual_MATLAB_ISET_export':True,'cone_count':len(cap['manifest']['cones']['cell_ids']),
        'windows':len(cap['edges'])-1,'duration_seconds':float(cap['edges'][-1]),
        'capture_sha256':cap['sha256'],'reference_capture_sha256':reference['sha256'],
        'frozen_ids_types_IS_RF_preserved':True,'other_channels_background_context_preserved':True,
        'count_window_integral_and_photon_energy_preserved':True,'fixed_LMS_bridge':bridge,
        'model_configs_background_cache_unchanged':True,'new_background_preparation':False,
        'new_neural_simulation':False,'GUI_connected':False,'physiological_calibration':False,
        'remaining_calibration_fields':['absolute experimental scene/spectral radiance','macaque optical profile',
            'collecting apertures and spectral pigment/macular/OS capture profile','same-condition fixed LMS reference'],
        'legacy_toolbox_warning':'R2026b warns about hiddenHandle delete access; capture and cleanup checks pass; vendor unchanged'}
    with (out/'check.json').open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,allow_nan=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
