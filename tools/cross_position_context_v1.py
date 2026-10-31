"""Explicit new-position bindings over frozen geometry and existing equations."""
from pathlib import Path
import json
import numpy as np
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.group_time_storage import load_bundle
from retina_sim.local_patch_graph import LocalPatchAssembly, _verified_base
from retina_sim.local_patch_boundary import BoundedInnerDynamics, BoundedNativePhysics
from retina_sim.local_patch_runner import prepare_seed
from retina_sim.spatial_group_model import GroupModelAssembly
from retina_sim.ac_vs_model import ACVSAssembly, load_config, redesign_seed
from retina_sim.h1_h2_effective import H1H2Assembly, verify_typed_source

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    from retina_sim.parameter_system import data
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(data(value), f, indent=2, allow_nan=False)


def patch(prepared):
    prepared = Path(prepared)
    folder = prepared / 'local_static'
    descriptor = read(folder / 'selection_identity.json')
    cfg = descriptor['configuration']
    base = _verified_base(ROOT, cfg)
    if base.artifact_sha256 != descriptor['parent_artifact']:
        raise ContractError('New-position parent graph differs')
    with np.load(folder / 'selection.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if canonical_hash(arrays) != descriptor['selection_sha256']:
        raise ContractError('New-position selection differs')
    rows = {k[12:]: v for k, v in arrays.items() if k.startswith('population__')}
    contacts = {k[9:]: v for k, v in arrays.items() if k.startswith('contact__')}
    core = {k[6:]: v for k, v in arrays.items() if k.startswith('core__')}
    g = read(prepared / 'GEOMETRY_SELECTION.json')
    result = object.__new__(LocalPatchAssembly)
    result._assemble(ROOT, folder, base, cfg, rows, contacts, core,
                     g['selected_parent_group_row'], canonical_hash(descriptor), descriptor)
    result.runtime_factories = dict(inner=BoundedInnerDynamics, native=BoundedNativePhysics)
    return base, result


def typed(prepared, *, guard=lambda: None):
    base, local = patch(prepared)
    seed = prepare_seed(ROOT, local, local.config, guard=guard)
    ac = ACVSAssembly(local, load_config(ROOT / 'configurations/ac_vs_functional_v1.json'))
    seed = redesign_seed(ac, seed)
    labels, evidence = verify_typed_source(base, ROOT)
    h = H1H2Assembly(ac, base, labels, evidence, engineering=True)
    return h, seed, labels


def load_background(prepared, directory):
    from retina_sim.hc_lateral_interface import context
    h, _, _ = typed(prepared)
    saved = load_bundle(Path(directory) / 'seed', expected_kind='source_construction_seed')
    seed = saved['payload']
    h.parent.patch.boundary_runtime = load_bundle(
        Path(directory) / 'inner_boundary', expected_kind='validation_evidence')['payload']
    h.native_boundary = seed['h1_h2_model']['boundary']
    cfg = read(Path(directory) / 'HC_POSITION_BINDING.json')
    a = context(h, seed, cfg, ROOT)
    a.parameter_binding_id = seed['lateral_hc_interface']['binding_id']
    return a, seed, saved['identity']
