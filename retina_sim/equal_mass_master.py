"""Frozen local canonical populations from the unchanged equal-mass backend."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from .contracts import Quantity,ContractError,canonical_hash
from .geometry import RetinalDomain
from .spatial_profiles import generate_density
from .atlas import CanonicalCellTable,AtlasROI


def materialize_equal_mass_master(root,population,bounds,field,seed,source,max_cells=60000):
    root=Path(root);bounds=tuple(float(v) for v in bounds)
    recipe={'schema_version':'1.0','population':population,'canonical_bounds_mm':bounds,
        'seed':int(seed),'source':source,'generator':'greedy_capacitated_equal_rho_mass',
        'iterations':1,'quota':'floor(integral rho dA+.5) on fixed master only','boundary':'finite_open',
        'density_scale':1,'numpy_version':np.__version__}
    recipe['generation_code_sha256']={name:hashlib.sha256((root/'retina_sim'/name).read_bytes()).hexdigest()
        for name in ('geometry.py','spatial_profiles.py','equal_mass_master.py')}
    namespace=canonical_hash(recipe);label='fixed_population_master:'+namespace
    path=root/'data/atlas/population_masters'/(population+'_'+namespace[:20]+'.json')
    if path.exists():
        raw=path.read_bytes();payload=json.loads(raw)
        if canonical_hash(payload['recipe'])!=namespace:raise ContractError('Frozen population master recipe differs')
    else:
        domain=RetinalDomain(bounds,bounds,lambda xy:np.ones(len(xy),bool),source['provider_id'],label)
        start=time.perf_counter();mosaic=generate_density(field,domain,int(seed),max_cells)
        payload={'recipe':recipe,'source_keys':list(mosaic.cell_ids),'positions_mm':mosaic.positions.values.tolist(),
            'diagnostics':mosaic.diagnostics,'generation_wall_seconds':time.perf_counter()-start,
            'claims':{'physiological_validation':False,'global_0_to_3_coverage':False,'ROI_regeneration':False}}
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf8') as out:out.write(json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\n')
        raw=path.read_bytes()
    return FixedEqualMassMaster(payload,path,hashlib.sha256(raw).hexdigest())


class FixedEqualMassMaster:
    def __init__(self,payload,path,sha):
        self.payload=payload;self.path=Path(path);self.sha256=sha;self.recipe=payload['recipe']
        self.namespace=canonical_hash(self.recipe);self.bounds=tuple(self.recipe['canonical_bounds_mm'])
        self.table=CanonicalCellTable('macaque_composite_source_atlas','1.0',self.namespace,
            self.recipe['population'],tuple(payload['source_keys']),{'rf':Quantity(payload['positions_mm'],'mm',
            'functional_position','project_approximation','functional_visual_field')},
            'composite_macaque_nasal_positive_superior_positive_planar_v1')

    def crop(self,roi,max_cells=60000):
        b=roi.query_bounds_mm;x0,x1,y0,y1=self.bounds
        if b[0]<x0-1e-12 or b[1]>x1+1e-12 or b[2]<y0-1e-12 or b[3]>y1+1e-12:
            raise ContractError('FIXED_POPULATION_MASTER_UNSUPPORTED: query crosses frozen local master; no regeneration or density compensation')
        selected=self.table.query(roi)
        if len(selected.source_keys)>max_cells:raise ContractError('Fixed master crop exceeds cell budget; no thinning')
        return CanonicalCellTable(self.table.atlas_id,self.table.atlas_version,self.table.source_version,
            self.table.population,selected.source_keys,{'rf':selected.coordinate('rf')},self.table.reference_id,
        )

    def manifest(self):
        return {'path':str(self.path),'sha256':self.sha256,'namespace':self.namespace,'recipe':self.recipe,
            'diagnostics':self.payload['diagnostics'],'count':len(self.table.source_keys),
            'generation_wall_seconds':self.payload['generation_wall_seconds']}
