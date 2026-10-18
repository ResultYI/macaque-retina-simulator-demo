"""The approved joint HC generator, frozen once and cropped without resampling.

This limited physical master is not a full 0-3 mm atlas. Missing coverage never
falls back to independent Poisson sampling or changes density/footprints.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from .atlas import CanonicalCellTable
from .atlas_store import load_atlas_asset, save_atlas_asset
from .contracts import ContractError, Quantity, canonical_hash
from .geometry import RetinalDomain
from .hc_mosaic import JointHCMosaic, generate_joint_hc


CONFIG = 'configurations/atlas_hc_joint_master_v2.json'


class HCMasterCoverageError(ContractError):
    """The fixed master does not support this query and finite pair context."""


def _configuration(root, density, configuration_path=None):
    root = Path(root)
    config = json.loads((root/(configuration_path or CONFIG)).read_bytes())
    if (config['schema_version'] != '1.0' or
            config['mosaic_rule'] != 'generate_joint_hc_fixed_canonical_master_then_crop'):
        raise ContractError('Explicit restored joint HC master contract required')
    base = json.loads((root/config['density_profile_path']).read_bytes())
    research = json.loads((root/config['research_generator_path']).read_bytes())
    if density.config != base or research['profile_id'] != density.profile_id:
        raise ContractError('HC master must retain the approved density profile')
    options = dict(base['generator'], **research['generator'])
    return config, options


def materialize_hc_master(root, atlas_config, density):
    """One bounded geometry generation; exclusive asset write, no ODE run."""
    root = Path(root)
    configuration_path=atlas_config.get('HC_master_configuration',CONFIG)
    config, options = _configuration(root, density,configuration_path)
    path = root/config['asset_path']
    if path.exists() or config['expected_asset_sha256'] is not None:
        raise ContractError('Frozen joint HC master already registered; regeneration refused')
    bounds = tuple(config['canonical_bounds_mm'])
    disk=config.get('canonical_disk_radius_mm')
    mask=(lambda xy:np.linalg.norm(xy,axis=1)<=disk) if disk is not None else (lambda xy:np.ones(len(xy),bool))
    domain = RetinalDomain(bounds, bounds, mask,
        config['mask_source_id'], 'atlas_HC_fixed_central_master_v2')
    started = time.perf_counter()
    master = generate_joint_hc(density, domain, config['seed'], config['max_cells'], options=options)
    seconds = time.perf_counter()-started
    backend_sha = hashlib.sha256((root/'retina_sim/hc_mosaic.py').read_bytes()).hexdigest()
    namespace = canonical_hash({'canonical_instance':master.diagnostics['canonical_instance_sha256'],
        'backend_sha256':backend_sha, 'identity_policy':'restored_joint_HC_fixed_master_v2'})
    table = CanonicalCellTable(atlas_config['atlas_id'], atlas_config['atlas_version'], namespace,
        'HC', master.cell_ids, {'soma':master.positions}, atlas_config['reference_id'])
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = save_atlas_asset(path, table, asset_id=config['profile_id']+':'+namespace,
        asset_kind='canonical_atlas',
        source={'profile_id':density.profile_id,'profile_sha256':density.profile_hash,
            'density_source_asset_sha256':density.source_asset_hash,'backend_sha256':backend_sha,
            'generator_options':options,'seed':config['seed'],
            'construction_integral_cache':config.get('construction_integral_cache'),
            'mosaic_rule':config['mosaic_rule'],'types':master.types.tolist()},
        provenance={'joint_generation':master.diagnostics,'generation_wall_seconds':seconds,
            'scope':config['scope'],'old_Poisson_identity_reused':False,
            'physiological_validation':False},
        boundary={'canonical_bounds_mm':list(bounds),'canonical_disk_radius_mm':disk,'mask_source_id':config['mask_source_id'],
            'pair_support_context_mm':density.max_interaction_support_mm(),
            'boundary_mode':'finite_open_boundary','recurrent_closure':'NOT_CERTIFIED'})
    config['expected_asset_sha256'] = sha
    (root/configuration_path).write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'asset_path':str(path),'sha256':sha,'cell_count':len(master.cell_ids),
        'types':master.diagnostics['total_H1_H2_counts'],'generation_wall_seconds':seconds,
        'namespace':namespace,'canonical_bounds_mm':list(bounds),'scope':config['scope']}


class FixedJointHCMaster:
    def __init__(self, root, atlas_config, density):
        self.root = Path(root)
        self.config, options = _configuration(root, density,atlas_config.get('HC_master_configuration'))
        asset = load_atlas_asset(self.root/self.config['asset_path'],
            expected_sha256=self.config['expected_asset_sha256'])
        record = asset.metadata_record()
        source = record['source']
        if (source['profile_sha256'] != density.profile_hash or
                source['density_source_asset_sha256'] != density.source_asset_hash or
                source['generator_options'] != options or source['seed'] != self.config['seed'] or
                source['backend_sha256'] != hashlib.sha256((self.root/'retina_sim/hc_mosaic.py').read_bytes()).hexdigest()):
            raise ContractError('Frozen HC joint settings/source/backend differ; no implicit regeneration')
        self.table = asset.tables['HC']
        if (self.table.atlas_id,self.table.atlas_version,self.table.reference_id) != (
                atlas_config['atlas_id'],atlas_config['atlas_version'],atlas_config['reference_id']):
            raise ContractError('HC master must share the declared global atlas reference')
        self.bounds = tuple(record['boundary']['canonical_bounds_mm'])
        if self.bounds != tuple(self.config['canonical_bounds_mm']):
            raise ContractError('HC master domain differs from the frozen configuration')
        self.pair_halo = float(record['boundary']['pair_support_context_mm'])
        self.mask_source_id = record['boundary']['mask_source_id']
        self.types = tuple(source['types'])
        if len(self.types) != len(self.table.source_keys):
            raise ContractError('HC master types must align with every fixed cell')
        diagnostics = record['provenance']['joint_generation']
        self.master = JointHCMosaic(self.table.coordinate('soma'),np.asarray(self.types),
            self.table.source_keys,np.ones(len(self.types),dtype=bool),diagnostics)
        self.metadata = {**record,'canonical_rule':self.config['mosaic_rule'],
            'scope':self.config['scope'],'source_version':self.table.source_version,
            'scale':1.,'biological_spatial_statistics':'joint soft-core retained; mixing NOT_CERTIFIED',
            'Poisson_fallback':False,'new_namespace_replaces_Poisson_ids':True}

    def crop(self, bounds_mm, *, max_cells=60000):
        bounds = tuple(map(float,bounds_mm))
        if len(bounds)!=4 or not np.all(np.isfinite(bounds)) or bounds[0]>=bounds[1] or bounds[2]>=bounds[3]:
            raise ContractError('HC query requires finite increasing physical 2D bounds')
        if len(self.table.source_keys)>max_cells:
            raise ContractError('HC fixed master cell budget exceeded; no density scaling')
        context = (bounds[0]-self.pair_halo,bounds[1]+self.pair_halo,
            bounds[2]-self.pair_halo,bounds[3]+self.pair_halo)
        if not (self.bounds[0]<=context[0]<context[1]<=self.bounds[1] and
                self.bounds[2]<=context[2]<context[3]<=self.bounds[3]):
            raise HCMasterCoverageError('HC_FIXED_MASTER_UNSUPPORTED: query plus pair context '
                +str(context)+' outside fixed '+str(self.bounds)+'; no Poisson fallback')
        domain = RetinalDomain(bounds,bounds,lambda xy:np.ones(len(xy),bool),
            self.mask_source_id,'atlas_HC_crop')
        view = self.master.crop(domain)
        table = CanonicalCellTable(self.table.atlas_id,self.table.atlas_version,self.table.source_version,
            'HC',view.cell_ids,{'soma':view.positions},self.table.reference_id)
        return table,tuple(view.types),dict(self.metadata,query_bounds_mm=list(bounds),
            pair_context_bounds_mm=list(context),query_pair_context_complete=True,
            view_policy='JointHCMosaic.crop of one frozen master; no resampling',
            view_count=len(view.cell_ids))
