"""Generate one fixed source-bound cone master with the pinned MATLAB routine.

No neural simulation, input export, background preparation or human density.
"""
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
from scipy.io import savemat,loadmat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.source_atlas import SourceDensityAtlas
from retina_sim.contracts import Quantity,canonical_hash
from retina_sim.isetbio_geometry import COMMIT,SCHEMA,cone_apertures

VENDOR=Path('D:/retipath/experiments/retipath_foveal_v1/exploratory_v2/vendor_pinned/isetbio_d2924c2f')
MATLAB=Path('C:/Program Files/MATLAB/R2026b/bin/matlab.exe')
OUT=ROOT/'data/atlas/isetbio_macaque_cones_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def verify_methods():
    files=sorted((VENDOR/'ganglioncells/+retinalattice').rglob('*.m'))
    files.append(VENDOR/'ganglioncells/+RGCmodels/+Watson/+convert/spacingToDensityForHexGrid.m')
    methods={}
    for p in files:
        relative=p.relative_to(VENDOR).as_posix()
        original=subprocess.run(['git','-C',str(VENDOR),'show',COMMIT+':'+relative],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
        assert p.read_bytes().replace(b'\r\n',b'\n')==original.replace(b'\r\n',b'\n'),relative
        methods[relative]={'sha256':sha(p),'pinned_git_blob_sha256':hashlib.sha256(original).hexdigest()}
    return methods

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise RuntimeError('Frozen master already exists; never overwrite')
    methods=verify_methods();atlas=SourceDensityAtlas(ROOT,profile_id='retina_atlas_Schein_bridge_research_v1')
    request=OUT/'request.mat';raw=OUT/'generated.mat';density_file=OUT/'density_manifest.json'
    if not request.exists():
        grid=np.linspace(-1.1,1.1,513);rho=[]
        for i,x in enumerate(grid):
            q=Quantity(np.column_stack((np.full(len(grid),x),grid)),'mm','functional_position','project_approximation','functional_visual_field')
            rho.append(atlas.cones(q))
            if i%128==0:print('DENSITY_ROWS',i,'/513',flush=True)
        rho=np.asarray(rho);maximum=float(rho.max());minimum_spacing=np.sqrt(2e6/(np.sqrt(3)*maximum))
        # Generation domain contains the .032mm-tile aligned label support square.
        savemat(request,{'isetbio_root':str(VENDOR),'grid_x_mm':grid,'grid_y_mm':grid,'density_per_mm2':rho,
            'fov_degs':2*.8/(1.33*.223),'mm_per_deg':.223,'minimum_spacing_um':minimum_spacing,
            'seed':20261010.,'max_iterations':50.})
    if not density_file.exists():
        stored=loadmat(request,squeeze_me=True);grid=stored['grid_x_mm'];rho=stored['density_per_mm2']
        maximum=float(rho.max());minimum_spacing=float(stored['minimum_spacing_um'])
        sample=atlas.cones.sample(Quantity([[0,0],[.35,0],[0,.35]],'mm','functional_position','project_approximation','functional_visual_field'))
        save(density_file,{'provider':sample.provenance,'sample_positions_mm':[[0,0],[.35,0],[0,.35]],
            'sample_density_per_mm2':list(sample.supported_values),'grid_shape':list(rho.shape),'grid_domain_mm':[-1.1,1.1],
            'grid_rule':'Bilinear tabulation of unchanged approved Packer four-cardinal/project-central field',
            'grid_step_mm':float(grid[1]-grid[0]),'peak_grid_density_per_mm2':maximum,
            'minimum_spacing_um':minimum_spacing,'density_scale':1.,'request_sha256':sha(request)})
    if not raw.exists():
        expression="addpath('"+str(ROOT/'tools').replace('\\','/').replace("'","''")+"'); export_isetbio_geometry('"+str(request).replace('\\','/').replace("'","''")+"','"+str(raw).replace('\\','/').replace("'","''")+"');"
        print('GENERATING_PINNED_ISET_LATTICE',flush=True)
        with (OUT/'matlab.log').open('w',encoding='utf8') as log:
            subprocess.run([str(MATLAB),'-batch',expression],stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),check=True,timeout=240)
    result=loadmat(raw,squeeze_me=True);positions=np.asarray(result['positions_um'],dtype=float)/1000
    assert positions.ndim==2 and positions.shape[1]==2 and np.all(np.isfinite(positions)) and len(positions)<=150000
    keys=np.array(['iset_lattice_row:'+str(i) for i in range(len(positions))])
    diameters=cone_apertures(positions);payload=OUT/'positions.npz'
    if payload.exists():raise RuntimeError('Payload already exists without manifest; inspect instead of overwrite')
    np.savez_compressed(payload,positions_mm=positions,source_keys=keys,aperture_diameters_um=diameters)
    metadata=json.loads(str(result['metadata_json']))
    manifest={'schema':SCHEMA,'isetbio_commit':COMMIT,'population':'cones','density_species':'Macaca nemestrina',
        'human_density_used':False,'position_role':'rf_input_inner_segment_project_transfer',
        'axes':'+x nasal/+y superior; project RF equals IS registration retained',
        'complete_query_bounds_mm':[-.512,.512,-.512,.512],
        'generation_domain':'Upstream centered elliptical domain from nominal .8mm initial radius; retained boundary/relaxation conventions',
        'payload':{'path':payload.relative_to(ROOT).as_posix(),'sha256':sha(payload)},
        'count':len(positions),'source_methods':methods,'density':json.loads(density_file.read_bytes()),
        'generator_request_sha256':sha(request),'generator_output_sha256':sha(raw),
        'exporter_sha256':sha(ROOT/'tools/export_isetbio_geometry.m'),'matlab':metadata,
        'recipe':{'seed':20261010,'max_iterations':50,'mm_per_deg':.223,'use_parfor':False,
            'retinal_density_to_spacing':'sqrt(2/(sqrt(3)*rho_mm^-2))*1000 microns',
            'density_is_not_measurement':'Published/digitized Packer field with existing project bridges',
            'midget_density':'NOT_IMPORTED; human Watson midget model not enabled',
            'apertures':'0.79*median6NN on new complete fixed support; project estimate, not measured aperture'},
        'claims':{'measured_individual_mosaic':False,'physiological_validation':False,'global_0_to_3mm_coverage':False}}
    save(OUT/'manifest.json',manifest)
    config=dict(atlas.config);config.update(
        mosaic_rule='isetbio_fixed_macaque_cone_master_v1',midget_rf_rule='connected_cone_centroid_isetbio_v1',
        cone_master={'path':(OUT/'manifest.json').relative_to(ROOT).as_posix(),'sha256':sha(OUT/'manifest.json')},
        claim='Pinned ISET numerical cone lattice supplied with unchanged macaque density/project mappings; no human density or soma inference')
    save(ROOT/'configurations/retina_atlas_isetbio_macaque_v1.json',config)
    print('FROZEN_MASTER_READY',len(positions),sha(OUT/'manifest.json'),flush=True)

if __name__=='__main__':main()
