"""On-demand fixed global tiles; ROI requests never seed/regenerate overlap.

This is a source-informed geometry preview. Unknown S partition, HC arbor and
inner coordinates are kept unknown rather than borrowed from another source.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
from types import MappingProxyType
import numpy as np

from .atlas import AtlasROI, CanonicalCellTable
from .atlas_sources import PackerFourMeridianDensity
from .contracts import ContractError, Quantity, canonical_hash
from .coverage import load_atlas_source_domains
from .geometry import keyed_rng
from .henle import ScheinHenleDisplacement


@dataclass(frozen=True)
class AtlasPatch:
    selection: object
    mapped_table: CanonicalCellTable
    mapping_status: tuple
    manifest: dict


class SourceDensityAtlas:
    def __init__(self, root, *, profile_id='retina_atlas_v1'):
        self.root = Path(root)
        path = self.root/'configurations'/(profile_id+'.json')
        self.config = json.loads(path.read_bytes())
        if (self.config.get('schema_version') != '1.0' or
                self.config.get('mosaic_rule') not in ('piecewise_constant_poisson_tiles_v1','isetbio_fixed_macaque_cone_master_v1','isetbio_fixed_human_borrowed_cone_master_v1') or
                self.config.get('boundary_mode') != 'finite_open_boundary' or
                self.config.get('central_rod_rule') != 'missing'):
            raise ContractError('Unsupported source-atlas contract')
        size = self.config['tile_size_mm']
        if isinstance(size, bool) or not np.isfinite(size) or size <= 0:
            raise ContractError('Positive fixed atlas tile size required')
        self.profile_hash = canonical_hash(self.config)
        self.config = MappingProxyType({k:tuple(v) if isinstance(v,list) else v for k,v in self.config.items()})
        self.coverage_catalog = load_atlas_source_domains(root)
        self.cones = PackerFourMeridianDensity(root, 'all_cones',
            central_policy=self.config['central_cone_rule'], rf_transfer=self.config['source_registration'])
        self.rods = PackerFourMeridianDensity(root, 'rod', rf_transfer=self.config['source_registration'])
        henle_rule=self.config['Henle_rule']
        if henle_rule=='Schein1988_M1_bounded_bridge_v1':
            from .henle_bridge import ScheinBoundedBridge
            self.henle=ScheinBoundedBridge(root,rf_transfer='rf_equals_inner_segment_mm_v1')
        elif henle_rule=='Schein1988_digitized_cardinal_linear_project_transfer_v1':
            self.henle=ScheinHenleDisplacement(root,rf_transfer='rf_equals_inner_segment_mm_v1')
        else:raise ContractError('Unknown explicit Henle provider profile')
        self.generation_code_hashes = {name:hashlib.sha256((self.root/'retina_sim'/name).read_bytes()).hexdigest()
            for name in ('atlas.py','source_atlas.py','atlas_sources.py','henle.py','geometry.py')}
        if henle_rule=='Schein1988_M1_bounded_bridge_v1':
            self.generation_code_hashes['henle_bridge.py']=hashlib.sha256((self.root/'retina_sim/henle_bridge.py').read_bytes()).hexdigest()
        self.source_version = canonical_hash({'profile': self.profile_hash,
            'density_assets': self.cones.bundle.manifest()['source_assets'],
            'central_cone_code_asset':self.cones.central.table.asset_sha256,
            'generation_code_hashes':self.generation_code_hashes,'numpy_version':np.__version__,
            'Henle': self.henle.coverage(Quantity(np.empty((0,2)), 'mm', 'functional_position',
                 'project_approximation', 'functional_visual_field'))['provenance']})
        # Retinal sampling depends on density, never on a downstream coordinate
        # map or diagnostic code hash. Coordinate changes still invalidate the
        # full asset/cache identity above without resampling the RF population.
        self.mosaic_namespace = canonical_hash({'atlas_id':self.config['atlas_id'],
            'atlas_version':self.config['atlas_version'],'seed':self.config['seed'],
            'tile_size_mm':size,'mosaic_rule':self.config['mosaic_rule'],
            'density_rule':self.config['density_rule'],'central_cone_rule':self.config['central_cone_rule'],
            'registration':self.config['source_registration'],
            'Packer_fig_points':self.cones.bundle.manifest()['source_assets']['photoreceptor_points']['sha256'],
            'Packer_central_code_asset':self.cones.central.table.asset_sha256,'numpy_version':np.__version__})
        self._tiles = {}
        self.cone_master=None
        if self.config['mosaic_rule'] in ('isetbio_fixed_macaque_cone_master_v1','isetbio_fixed_human_borrowed_cone_master_v1'):
            from .isetbio_geometry import FixedISETConeMaster,HUMAN_SCHEMA
            self.cone_master=FixedISETConeMaster(self.root,self.config['cone_master'])
            human=self.config['mosaic_rule']=='isetbio_fixed_human_borrowed_cone_master_v1'
            if human != (self.cone_master.manifest['schema']==HUMAN_SCHEMA):
                raise ContractError('Cone profile species route differs from the frozen master; no implicit fallback')
            self.mosaic_namespace=canonical_hash({'legacy_density_namespace':self.mosaic_namespace,
                'cone_master':self.cone_master.identity,'position_rule':'fixed_global_master_index_no_ROI_regeneration'})
            importer_hash=hashlib.sha256((self.root/'retina_sim/isetbio_geometry.py').read_bytes()).hexdigest()
            self.generation_code_hashes['isetbio_geometry.py']=importer_hash
            self.source_version=canonical_hash({'source':self.source_version,'master':self.cone_master.identity,'importer':importer_hash})

    def _table(self, population, keys, positions, coordinates=None):
        rf = Quantity(np.asarray(positions).reshape((-1, 2)), 'mm', 'functional_position',
                      'source_informed_project_mosaic', 'functional_visual_field')
        return CanonicalCellTable(atlas_id=self.config['atlas_id'], atlas_version=self.config['atlas_version'],
            source_version=self.mosaic_namespace, population=population, reference_id=self.config['reference_id'],
            source_keys=tuple(keys), coordinates={'rf': rf, **(coordinates or {})})

    def _tile(self, population, tx, ty):
        key = (population, tx, ty)
        if key not in self._tiles:
            s = self.config['tile_size_mm']
            if population=='cones' and self.cone_master is not None:
                keys,xy=self.cone_master.tile(tx,ty,s)
                self._tiles[key]=self._table(population,keys,xy)
                return self._tiles[key]
            center = np.array([[(tx+.5)*s, (ty+.5)*s]])
            q = Quantity(center, 'mm', 'functional_position', 'project_approximation', 'functional_visual_field')
            provider = self.cones if population == 'cones' else self.rods
            corners = (np.array([[tx,ty],[tx+1,ty],[tx,ty+1],[tx+1,ty+1]])*s)
            provider(Quantity(corners,'mm','functional_position','project_approximation','functional_visual_field'))
            rho = float(provider(q)[0])
            rng = keyed_rng(self.config['seed'], self.mosaic_namespace+'|'+population+'|'+str(tx)+'|'+str(ty))
            count = int(rng.poisson(rho*s*s))
            xy = (np.array([tx,ty]) + rng.random((count,2)))*s
            # Sampling is explicitly piecewise constant per fixed tile. This is
            # not a measured cell mosaic or a calibrated exclusion process.
            keys = tuple(f'tile:{tx}:{ty}:cell:{i}' for i in range(count))
            self._tiles[key] = self._table(population, keys, xy)
        return self._tiles[key]

    def query(self, roi, *, population='cones', max_cells=60000, phase='phase1'):
        if not isinstance(roi, AtlasROI) or roi.reference_id != self.config['reference_id'] or roi.coordinate_role != 'rf':
            raise ContractError('Source atlas queries require the declared RF/mm global reference')
        if population not in ('cones', 'rods') or phase not in ('phase1', 'phase2'):
            raise ContractError('Unknown population or atlas phase')
        if not isinstance(max_cells, int) or isinstance(max_cells, bool) or max_cells < 1:
            raise ContractError('Positive integer cell budget required')
        x0,x1,y0,y1 = roi.owner_bounds_mm
        corners = np.array([[x0,y0],[x0,y1],[x1,y0],[x1,y1]])
        upper = self.config[phase+'_rf_radius_mm'][1]
        if np.any(np.linalg.norm(corners, axis=1) > upper):
            raise ContractError('Owner exceeds explicit atlas phase domain; query halo is a separate domain')
        s = self.config['tile_size_mm']
        qx0,qx1,qy0,qy1 = roi.query_bounds_mm
        ix = range(int(np.floor(qx0/s)), int(np.ceil(qx1/s)))
        iy = range(int(np.floor(qy0/s)), int(np.ceil(qy1/s)))
        # Early budget bound applies to tile work, never rescales biological density.
        if len(ix)*len(iy) > max_cells:
            raise ContractError('Atlas tile budget exceeded; choose a smaller ROI')
        tables, total = [], 0
        for tx in ix:
            for ty in iy:
                table = self._tile(population,tx,ty)
                total += len(table.source_keys)
                if total > max_cells:
                    raise ContractError('Atlas cell budget exceeded; density unchanged, choose a smaller ROI')
                tables.append(table)
        keys = tuple(k for t in tables for k in t.source_keys)
        xy = np.concatenate([t.coordinate('rf').values for t in tables]) if tables else np.empty((0,2))
        canonical = self._table(population, keys, xy)
        selection = canonical.query(roi)
        selected = selection.coordinate('rf')
        if population == 'cones':
            coverage = self.henle.coverage(selected)
            status = tuple(p['status'] for p in coverage['points'])
            keep = np.array([x in ('source_supported','project_approximation') for x in status], dtype=bool)
            known_rf = Quantity(selected.values[keep], 'mm','functional_position',
                                'project_approximation', 'functional_visual_field')
            known_opl = self.henle.apply(known_rf)
            mapped = self._table(population, tuple(k for k,ok in zip(selection.source_keys,keep) if ok),
                                 known_rf.values, {'opl_pedicle': known_opl})
        else:
            status = ('unsupported_rod_OPL_mapping',)*len(selected.values)
            mapped = self._table(population, (), np.empty((0,2)))
            coverage = {'provenance': {'rod_OPL_mapping': 'not supplied; cone Henle not applied to rods'}}
        audit = {'atlas_id': self.config['atlas_id'], 'atlas_version':self.config['atlas_version'],
            'reference_id':self.config['reference_id'], 'profile_sha256':self.profile_hash,
            'source_version':self.source_version, 'population':population,
            'generation_code_hashes':self.generation_code_hashes,'numpy_version':np.__version__,
            'mosaic_namespace':self.mosaic_namespace,
            'cell_source_version':self.mosaic_namespace,
            'source_version_meaning':'full geometry/provider/cache asset identity; cell IDs use the density/mosaic source version',
            'owner_domain':list(roi.owner_bounds_mm), 'query_support':list(roi.query_bounds_mm),
            'phase':phase, 'owner_status':selection.owner_status,
            'owner_count':int(selection.owner.sum()), 'query_count':len(selection.row_indices),
            'canonical_tile_count':len(tables), 'density_scale':1., 'max_cells':max_cells,
            'mosaic_rule':self.config['mosaic_rule'], 'fixed_tile_size_mm':s,
            'identity_rule':'fixed atlas/source/tile/cell namespace; no ROI seed',
            'boundary_mode':self.config['boundary_mode'], 'recurrent_closure':'NOT_CERTIFIED',
            'HC_dynamic_gap':'off', 'geometry_only':True, 'network_run_ready':False,
            'source_bundle':self.cones.bundle.manifest(), 'Henle':coverage['provenance'],
            'source_domain_catalog_sha256':self.coverage_catalog.sha256,
            'coordinate_roles_available':['rf','opl_pedicle'] if population=='cones' else ['rf'],
            'missing':['S full density partition','HC central/peripheral direct arbor',
                       'IPL preterminal','IPL postsynaptic arbor','soma'],
            'claim':self.config['claim']}
        if self.cone_master is not None:audit['cone_master']=dict(self.config['cone_master'])
        return AtlasPatch(selection,mapped,status,audit)
