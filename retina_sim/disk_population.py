"""Fixed large-population arrays with global IDs generated only for queried rows.

Uses the existing equal-mass algorithm. Binary storage avoids materializing a
multi-million-row JSON CellTable merely to query a small recording region.
"""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
from .contracts import ContractError,Quantity,canonical_hash
from .geometry import RetinalDomain,MosaicBackend,integrate_density
from .atlas import CanonicalCellTable

SCHEMA='fixed_disk_population_arrays_v1'
REFERENCE='composite_macaque_nasal_positive_superior_positive_planar_v1'

class IntegratedDiskDomain(RetinalDomain):
    """A disk bound to one unchanged density field and its accurate integral."""
    def __init__(self,bounds,mask,label,field,mass,method):
        super().__init__(bounds,bounds,mask,label,'fixed_population_master:'+label)
        if not np.isfinite(mass) or mass<=0:raise ContractError('Positive disk density integral required')
        object.__setattr__(self,'_integral_field',field)
        object.__setattr__(self,'_integral_mass',float(mass))
        object.__setattr__(self,'_integral_method',str(method))
    def density_integral(self,field,relative_tolerance,*,allow_zero=False):
        if field is not self._integral_field:raise ContractError('Disk integral belongs to a different density field')
        return {'mass':self._integral_mass,'estimated_absolute_quadrature_error':0.,
            'rounding_target':int(np.floor(self._integral_mass+.5)),'relative_tolerance':relative_tolerance,
            'domain':self.label,'integral_method':self._integral_method}

def resolved_disk_quadrature(field,domain,mass,side,max_bytes):
    """Resolve the existing1e-4 mass audit before the costly unchanged sampler."""
    radius=(domain.bounds_mm[1]-domain.bounds_mm[0])/2;history=[]
    for attempt in range(5):
        estimate=side**2*80+int(np.floor(mass+.5))*96
        if estimate>max_bytes:raise ContractError('Disk quadrature exceeds resource budget; no density reduction')
        axis=-radius+(np.arange(side)+.5)*(2*radius/side);total=0.
        for start in range(0,side,64):
            xy=np.array(np.meshgrid(axis[start:start+64],axis,indexing='ij')).reshape(2,-1).T
            xy=xy[domain.contains(xy)];total+=float(np.sum(field(xy)))*(2*radius/side)**2
        residual=abs(total-mass)/mass;history.append({'side':side,'relative_mass_residual':residual})
        if residual<=1e-4:return side,history
        side=int(np.ceil(1.5*side))
    raise ContractError('Disk quadrature did not resolve existing1e-4 mass tolerance')

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def materialize_disk_population(root,directory,population,radius,field,seed,source,*,max_cells,max_bytes=6*1024**3,density_upper_bound=None,density_mass=None,integral_method=None):
    root=Path(root).resolve();directory=(root/directory).resolve()
    if not directory.is_relative_to(root):raise ContractError('Population asset must stay inside project')
    manifest=directory/'manifest.json'
    if manifest.exists():return FixedDiskPopulation(root,{'path':manifest.relative_to(root).as_posix(),'sha256':sha(manifest)})
    directory.mkdir(parents=True,exist_ok=True);radius=float(radius);bounds=(-radius,radius,-radius,radius)
    recipe={'schema':SCHEMA,'population':population,'radius_mm':radius,'canonical_bounds_mm':bounds,
        'seed':int(seed),'source':source,'generator':'greedy_capacitated_equal_rho_mass','iterations':1,
        'quota':'floor(integral rho dA+.5) on fixed complete disk only','density_scale':1,'boundary':'finite_open',
        'coordinate_registration':'existing nominal RF/physical chart; no new soma mapping','numpy_version':np.__version__,
        'quadrature_rule':{'minimum_points_per_target':8,'maximum_node_mass_fraction_of_capacity':.5,'density_upper_bound':density_upper_bound},
        'density_integral_method':integral_method}
    namespace=canonical_hash(recipe)
    if density_mass is None or not integral_method:raise ContractError('Fixed disk requires an explicit accurate source integral')
    domain=IntegratedDiskDomain(bounds,lambda xy:np.sum(xy**2,axis=1)<=radius**2,
        'fixed_global_disk:'+population+':'+namespace,field,density_mass,integral_method)
    started=time.monotonic();audit=integrate_density(field,domain);n=audit['rounding_target']
    if n<1 or n>max_cells:raise ContractError('Density-derived population exceeds declared budget; no thinning')
    side=max(64,int(np.ceil(np.sqrt(8*n))))
    if density_upper_bound is not None:
        if not np.isfinite(density_upper_bound) or density_upper_bound<=0:raise ContractError('Positive source-derived density bound required')
        capacity=audit['mass']/n;area=4*radius**2
        side=max(side,int(np.ceil(np.sqrt(2*area*density_upper_bound/capacity))))
    side,mesh_audit=resolved_disk_quadrature(field,domain,audit['mass'],side,max_bytes)
    progress={'status':'GENERATING','population':population,'density_integral':audit,'expected_count':n,
        'quadrature_resolution_audit':mesh_audit,
        'quadrature_side':side,'sampler_iterations':1,'max_bytes':max_bytes,'radius_mm':radius}
    (directory/'construction.json').write_text(json.dumps(progress,indent=2)+'\n',encoding='utf8')
    mosaic=MosaicBackend().generate('greedy_capacitated_equal_rho_mass',field,domain,int(seed),
        quadrature_shape=(side,side),iterations=1,max_bytes=max_bytes)
    if len(mosaic.cell_ids)!=n:raise ContractError('Sampler changed density-derived count')
    positions=directory/'positions_mm.npy';keys=directory/'source_keys.npy'
    if positions.exists() or keys.exists():raise FileExistsError('Partial population arrays exist; inspect before replacing')
    np.save(positions,mosaic.positions.values,allow_pickle=False)
    width=max(map(len,mosaic.cell_ids));target=np.lib.format.open_memmap(keys,mode='w+',dtype='S'+str(width),shape=(n,))
    for start in range(0,n,65536):target[start:start+65536]=np.asarray(mosaic.cell_ids[start:start+65536],dtype=target.dtype)
    target.flush();del target
    value={'schema':SCHEMA,'recipe':recipe,'namespace':namespace,'reference_id':REFERENCE,'count':n,
        'positions':{'path':positions.relative_to(root).as_posix(),'sha256':sha(positions),'unit':'mm','frame':'functional_visual_field'},
        'source_keys':{'path':keys.relative_to(root).as_posix(),'sha256':sha(keys)},'diagnostics':mosaic.diagnostics,
        'quadrature_resolution_audit':mesh_audit,
        'generation_wall_seconds':time.monotonic()-started,'regenerate_on_query':False,'physiological_validation':False}
    pending=manifest.with_suffix('.writing');pending.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');os.replace(pending,manifest)
    return FixedDiskPopulation(root,{'path':manifest.relative_to(root).as_posix(),'sha256':sha(manifest)})

class FixedDiskPopulation:
    def __init__(self,root,binding):
        self.root=Path(root).resolve();self.binding=dict(binding);path=(self.root/binding['path']).resolve()
        if not path.is_relative_to(self.root) or sha(path)!=binding['sha256']:raise ContractError('Fixed population manifest identity mismatch')
        self.record=json.loads(path.read_bytes());r=self.record
        if r['schema']!=SCHEMA or r['namespace']!=canonical_hash(r['recipe']) or r['regenerate_on_query'] is not False:
            raise ContractError('Unknown fixed population recipe/schema')
        arrays=[]
        for key in ['positions','source_keys']:
            p=(self.root/r[key]['path']).resolve()
            if not p.is_relative_to(self.root) or sha(p)!=r[key]['sha256']:raise ContractError('Fixed population array identity mismatch')
            arrays.append(np.load(p,mmap_mode='r',allow_pickle=False))
        self.positions,self.source_keys=arrays;self.namespace=r['namespace'];self.radius=float(r['recipe']['radius_mm'])
        if self.positions.shape!=(r['count'],2) or self.source_keys.shape!=(r['count'],):raise ContractError('Fixed population array dimensions differ')
    def rows(self,indices):
        indices=np.asarray(indices,dtype=np.int64);r=self.record
        return CanonicalCellTable('macaque_composite_source_atlas','1.0',self.namespace,r['recipe']['population'],
            tuple(x.decode('ascii') for x in self.source_keys[indices]),{'rf':Quantity(self.positions[indices],'mm',
            'functional_position','project_approximation','functional_visual_field')},r['reference_id'])
    def query(self,roi):
        if roi.reference_id!=self.record['reference_id']:raise ContractError('Population query reference differs')
        bounds=roi.query_bounds_mm;corner=max(np.hypot(x,y) for x in bounds[:2] for y in bounds[2:])
        extent=min(corner,getattr(roi,'query_radius_mm',corner))
        if extent>self.radius+1e-12:raise ContractError('Population query exceeds fixed disk; no extrapolation')
        return self.rows(np.flatnonzero(roi.query_mask(self.positions)))
