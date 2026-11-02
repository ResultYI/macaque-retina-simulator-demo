"""Selected BC input-anchor policy and admission of its derived assets."""
from pathlib import Path
import hashlib
import json
from .contracts import ContractError, canonical_hash

CURRENT = 'configurations/bc_density_geometry_current.json'
PROFILE = 'macaque_bc_cone_pedicle_anchored_v1'
POPULATIONS = ('midget_BC_ON', 'midget_BC_OFF', 'diffuse_BC_ON', 'diffuse_BC_OFF', 'S_ON_BC')

def current_bc(root):
    root = Path(root).resolve()
    path = root / CURRENT
    if not path.exists():
        return None
    selected = json.loads(path.read_bytes())
    if selected.get('selected_profile_id') != PROFILE:
        raise ContractError('Unknown current BC density/contact policy')
    for binding in selected['bindings'].values():
        asset = (root / binding['path']).resolve()
        if not asset.is_relative_to(root) or hashlib.sha256(asset.read_bytes()).hexdigest() != binding['sha256']:
            raise ContractError('CURRENT_BC_IDENTITY_CHANGED: ' + binding['path'])
    selected['identity'] = canonical_hash(selected)
    return selected

def require_current_bc_asset(root, source):
    selected = current_bc(root)
    if selected is not None and source.get('bc_density_geometry_identity') != selected['identity']:
        raise ContractError('BC_ASSET_STALE_FOR_CURRENT_CONFIGURATION: keep historical graph/background/cache; explicitly rebuild with current registered-pedicle policy')

def require_current_bc_tables(root, tables):
    if current_bc(root) is None:
        return
    for name in POPULATIONS:
        source = tables.get(name)
        if source is None:
            raise ContractError('Current BC source table missing: ' + name)
        require_current_bc_asset(root, source.get('source_bindings', {}))
