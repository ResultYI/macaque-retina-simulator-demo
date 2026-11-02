"""Current HC density/geometry selection and stale-asset admission boundary."""
from pathlib import Path
import hashlib
import json
from .contracts import ContractError,canonical_hash

CURRENT='configurations/hc_density_geometry_current.json'
JOINT_ID='macaque_hc_density_rf_joint_v2'

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def current_hc(root):
    root=Path(root);path=root/CURRENT
    if not path.exists():return None
    value=json.loads(path.read_bytes())
    if value.get('selected_profile_id')!=JOINT_ID:raise ContractError('Unknown selected current HC profile')
    for binding in value['bindings'].values():
        asset=(root/binding['path']).resolve()
        if not asset.is_relative_to(root.resolve()) or file_hash(asset)!=binding['sha256']:
            raise ContractError('CURRENT_HC_IDENTITY_CHANGED: '+binding['path'])
    value['identity']=canonical_hash(value)
    return value

def require_current_hc_asset(root,source):
    selected=current_hc(root)
    if selected is None:return
    if source.get('hc_density_geometry_identity')!=selected['identity']:
        raise ContractError('HC_ASSET_STALE_FOR_CURRENT_CONFIGURATION: '+JOINT_ID+
            ' selected; old HC coordinates/graph/background/cache retained but not admitted as new HC. Rebuild explicitly; no automatic generation.')

def current_hc_configuration(root):
    selected=current_hc(root)
    if selected is None:return None
    return Path(root)/selected['bindings']['parameters']['path']
