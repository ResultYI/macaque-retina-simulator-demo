"""Versioned execution defaults and a shared, explicitly unit-labelled light context."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from .contracts import ContractError, canonical_hash
from .density import read_density_tabulation

# Frozen migrated extraction, tied to the already pinned source snapshot.
# The earlier f87e... delivery is a different artifact; it is not silently accepted.
PACKER_CSV_SHA256 = '897ab2056e5a92cd0180b09ee4b97fe01cf010cc17f269fb15d94e352515ba5c'
PACKER_SOURCE_SHA256 = '72408017cb5b6c4996b48c95050963057f61aab6e39154ea095062a8b5c35e9d'

def frozen_packer_table(root):
    root = Path(root)
    manifest = json.loads((root/'data/published_code/packer_cone_density.manifest.json').read_text(encoding='utf-8'))
    if manifest['csv_sha256'] != PACKER_CSV_SHA256 or manifest['source']['sha256'] != PACKER_SOURCE_SHA256:
        raise ContractError('Packer extraction identity differs from frozen migrated manifest')
    source = root/manifest['source']['snapshot']
    if hashlib.sha256(source.read_bytes()).hexdigest() != PACKER_SOURCE_SHA256:
        raise ContractError('Pinned Packer source identity mismatch')
    return read_density_tabulation(root/'data/published_code/packer_cone_density.csv', PACKER_CSV_SHA256)

@dataclass(frozen=True)
class LightContext:
    native_background_td: float = 100.
    cone_background_rstar_s: float = 1000.
    rod_background_rstar_s: float = 5.
    background_source: str = 'declared_project_background'
    input_class: str = 'synthetic_spatial_scene'
    calibration: str = 'uncalibrated_project_mapping'
    adaptation_policy: str = 'uniform_declared_background'
    input_domain: str = 'independent_consumer_backgrounds'
    adapter_id: str = 'direct_declared_rates'
    reference_linear_value: float | None = None
    scene_fov_deg: float | None = None
    native_background_values: tuple | None = None

    def __post_init__(self):
        for name in ('native_background_td', 'cone_background_rstar_s', 'rod_background_rstar_s'):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ContractError('Positive finite declared background required: '+name)
        if not self.background_source or not self.input_class or not self.calibration:
            raise ContractError('Explicit light source and calibration status required')
        if self.adaptation_policy not in ('uniform_declared_background', 'legacy_first_frame'):
            raise ContractError('Unknown adaptation policy')
        if self.input_domain not in ('independent_consumer_backgrounds','native_td','receptor_Rstar_per_s','relative_intensity'):
            raise ContractError('Unsupported tagged input domain; spectral irradiance needs a separate calibrated adapter')
        if self.native_background_values is not None:
            values=np.asarray(self.native_background_values,dtype=float)
            if values.ndim!=1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values<=0):
                raise ContractError('Explicit finite positive per-cone reference background required')
            object.__setattr__(self,'native_background_values',tuple(float(v) for v in values))

    @property
    def background_context_hash(self):return canonical_hash(asdict(self))

    def manifest(self, prerun_seconds):
        return dict(asdict(self),background_context_hash=self.background_context_hash,
            common_physical_scene_claim=False,units={'native_background_td': 'td', 'cone_background_rstar_s': 'R*/s',
            'rod_background_rstar_s': 'R*/s'}, prerun_seconds=prerun_seconds,
            outer_initialization='deterministic constant-background model equilibria',
            inner_initialization='local conditional balances, actual upstream reference histories, and declared unsolved coupled states; finite uniform-background prerun',
            adaptation_status='finite_prerun_NOT_steady_state_verified',
            inner_initialization_mode='transient_start',
            equilibrium_scope='source outer equilibria and local conditional balances; coupled AII/cone/gap system not solved',
            stochastic_spikes='seeded conditional process; no static spike-state equilibrium claim')

class RuntimeProfile:
    def __init__(self, root, profile_id='legacy_effective_v1'):
        self.path = Path(root)/'configurations'/ (profile_id+'.json')
        raw = self.path.read_bytes()
        self.data = json.loads(raw)
        if self.data.get('schema_version') != '1.0' or self.data.get('profile_id') != profile_id:
            raise ContractError('Unsupported runtime parameter profile identity/version')
        self.sha256 = hashlib.sha256(raw).hexdigest()
        for row in self.data.get('geometry_assets',{}).values():
            if hashlib.sha256((self.path.parent.parent/row['path']).read_bytes()).hexdigest()!=row['sha256']:
                raise ContractError('Geometry asset differs from expected local content baseline '+row['path'])

    def bind(self, name, entries):
        definitions = self.data['profiles'].get(name)
        if definitions is None:
            # Author reference profiles retain their independent frozen registries.
            if name not in self.data.get('external_parameter_profiles',[]):
                raise ContractError('Parameter family not declared in runtime profile '+name)
            return entries
        result = {}
        for key, (value, unit, meaning) in entries.items():
            if key not in definitions:
                raise ContractError('Runtime profile missing parameter '+name+'.'+key)
            row = definitions[key]
            if (unit, meaning) != (row['unit'], row['meaning']):
                raise ContractError('Runtime profile unit/type mismatch '+name+'.'+key)
            shape = np.asarray(value).shape
            if row['binding'] == 'broadcast':
                value = np.full(shape, row['value'])
            elif row['binding'] == 'literal':
                value = np.asarray(row['value'], dtype=float)
                if value.shape != shape:
                    raise ContractError('Runtime profile literal shape mismatch '+name+'.'+key)
            elif row['binding'] != 'derived':
                raise ContractError('Unknown runtime parameter binding')
            if not np.all(np.isfinite(value)):
                raise ContractError('Nonfinite runtime parameter '+name+'.'+key)
            result[key] = (value, unit, meaning)
        return result

    def source_identity(self,source_id):
        row=self.data['source_assets'][source_id]
        if hashlib.sha256((self.path.parent.parent/row['path']).read_bytes()).hexdigest()!=row['sha256']:
            raise ContractError('Runtime source asset differs from frozen identity '+source_id)
        return row['sha256']

    def manifest(self):
        return {'profile_id': self.data['profile_id'], 'schema_version': '1.0',
            'path': str(self.path), 'sha256': self.sha256, 'claim': self.data['claim'],
            'allowed_modes': self.data['allowed_modes'],'source_assets':self.data.get('source_assets',{}),
            'geometry_assets':self.data.get('geometry_assets',{})}

def execution_profile(root,name):
    ids={'source_constrained_approximation':'macaque_constrained_native_v1','engineering_preview':'engineering_preview_v1'}
    profile_id=ids.get(name,name)
    path=Path(root)/'configurations'/(profile_id+'.json')
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version')!='1.0' or data.get('profile_id')!=profile_id:raise ContractError('Invalid execution profile identity')
    return dict(data,profile_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
