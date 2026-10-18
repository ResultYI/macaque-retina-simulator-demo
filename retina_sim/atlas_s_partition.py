"""An explicit whole legacy mulatta-effective S route, never a mixed new curve."""
from pathlib import Path
import json,hashlib
import numpy as np
from .contracts import ContractError,Quantity
from .geometry import SConeDensityField,Mosaic,RetinalDomain,assign_cone_types,keyed_rng

PROFILE='deMonasterio1985_mulatta_effective_S_v1'


class EffectiveMulattaSPartition:
    def __init__(self,atlas):
        self.atlas=atlas;root=atlas.root
        raw=(root/'configurations'/(PROFILE+'.json')).read_bytes();self.config=json.loads(raw)
        self.profile_sha256=hashlib.sha256(raw).hexdigest()
        source_raw=(root/self.config['source_path']).read_bytes()
        if hashlib.sha256(source_raw).hexdigest()!=self.config['expected_source_sha256']:
            raise ContractError('S effective route source differs from the pre-existing frozen baseline')
        source=json.loads(source_raw)
        if source['species']!='M. mulatta' or self.config['central_policy']!='assay_square_PCHIP':
            raise ContractError('Whole mulatta-effective rule required; no mixed S curve')
        self.field=SConeDensityField(source,self.config['mm_per_deg'],
            magnification_source_id=self.config['magnification_source_id'],central_policy=self.config['central_policy'])
        self.cache={}

    def _macro(self,tx,ty,max_cells):
        tx,ty=int(tx),int(ty)
        key=(tx,ty)
        if key in self.cache:return self.cache[key]
        size=self.config['label_macro_size_mm'];s=self.atlas.config['tile_size_mm']
        box=(tx*size,(tx+1)*size,ty*size,(ty+1)*size);tables=[];count=0
        for ix in range(int(np.floor(box[0]/s)),int(np.ceil(box[1]/s))):
            for iy in range(int(np.floor(box[2]/s)),int(np.ceil(box[3]/s))):
                table=self.atlas._tile('cones',ix,iy);count+=len(table.source_keys)
                if count>max_cells:raise ContractError('S canonical label-domain work budget exceeded; no density or subtype thinning')
                tables.append(table)
        points=np.concatenate([t.coordinate('rf').values for t in tables]);ids=tuple(g for t in tables for g in t.global_cell_ids)
        keep=(points[:,0]>=box[0])&(points[:,0]<box[1])&(points[:,1]>=box[2])&(points[:,1]<box[3])
        points=points[keep];ids=tuple(g for g,v in zip(ids,keep) if v)
        q=Quantity(points,'mm','functional_position','project_approximation','functional_visual_field')
        domain=RetinalDomain(box,box,lambda xy:np.ones(len(xy),bool),'effective_mulatta_label_domain','S_global_macro')
        mosaic=Mosaic(q,ids,np.ones(len(ids),bool),{})
        # Preserve the old balancing/conditional labelling operator; only its
        # immutable global domain and seed namespace replace ROI-local origins.
        seed=int(keyed_rng(self.atlas.config['seed'],PROFILE+'|'+self.profile_sha256+'|'+str(key)).integers(0,2**63))
        total=lambda xy:self.atlas.cones(Quantity(xy,'mm','functional_position','project_approximation','functional_visual_field'))
        labels,audit=assign_cone_types(mosaic,total,self.field,domain,self.config['LM_ratio'],
            'LM_equal_project_v1',seed,size)
        if self.atlas.config['mosaic_rule']=='isetbio_fixed_human_borrowed_cone_master_v1':
            audit.update(cone_geometry='official human-borrowed approximation',
                S_quota='unchanged rounded integral of existing macaque effective S field on the complete macro',
                selection_score='retained rhoS/rhoPacker score on actual human candidates, not measured human S probability',
                LM_rule='unchanged named1:1 conditional rule on actual remaining candidates')
        result=(dict(zip(ids,labels.tolist())),dict(audit,global_label_bounds_mm=list(box),macro_key=[tx,ty],
            density_generated_cell_count=count,global_macro_seed=seed))
        self.cache[key]=result;return result

    def assign(self,cones,*,max_cells=60000):
        points=cones.coordinate('rf').values;size=self.config['label_macro_size_mm'];labels=[];used={}
        for gid,xy in zip(cones.global_cell_ids,points):
            key=tuple(int(v) for v in np.floor(xy/size));mapping,audit=self._macro(*key,max_cells)
            if gid not in mapping:raise ContractError('S labels require the same immutable source-atlas cells')
            labels.append(mapping[gid]);used[str(key)]=audit
        return tuple(labels),{'profile_id':PROFILE,'profile_sha256':self.profile_sha256,'configuration':self.config,
            'source_species':'Macaca mulatta','target':'composite macaque atlas',
            'cross_species_transfer':True,'main_nemestrina_source_completed':False,
            'label_domains':used,'shared_S_kinetics':'existing uncalibrated project approximation remains separate',
            'Godat_curve_used':False,'Martin50um_mask_applied':False,
            'empty_subtype_samples_allowed':True,'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
