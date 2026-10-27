"""Explicit finite-Halo forcing on the unchanged representative equations.

External inputs are a held, source-bound construction IVP. They are not an
adapted or physiologically qualified background and are valid only in a paired
experiment whose spatial error has been measured. Original CSR weights and
nonlinear native pooling order remain intact.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from .contracts import ContractError, canonical_hash
from .group_native_physics import GroupNativePhysics
from .group_inner_dynamics import GroupInnerDynamics, BC_MECH, MEMBRANE
from .group_time_recording import GroupRecording


SCHEMA = 'local-retinal-patch-held-exterior-boundary-2'


def _owned(value):
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ContractError('Finite source-bound exterior forcing required')
    array.flags.writeable = False
    return array


class AffineProduct:
    """Local linear operator plus the separately identified exterior product."""
    def __init__(self, matrix, offset):
        self.matrix = matrix
        self.offset = _owned(offset)
        if self.offset.shape != (matrix.shape[0],):
            raise ContractError('Exterior product shape differs')

    def __matmul__(self, value):
        return self.matrix @ value + self.offset

    def __getattr__(self, name):
        return getattr(self.matrix, name)

    def tocsr(self, *args, **kwargs):
        return self.matrix.tocsr(*args, **kwargs)


class BoundedNativePhysics(GroupNativePhysics):
    """The source provider with missing native pool terms inserted before power."""
    def __init__(self, assembly, *, torch=None, device=None, plans=None):
        if torch is not None or plans is not None:
            raise ContractError('Held-exterior v2 is explicitly CPU FP64 only')
        self.boundary = assembly.boundary_runtime
        super().__init__(assembly, torch=None)
        self.identity['bounded_exterior'] = self.boundary['identity']
        self.source_version = canonical_hash(self.identity)
        self._pool_argument = None

    def _coefficients(self, y, t, anchor):
        # The source equation pools Vbar before the adaptation power and pools
        # release separately. Identity tags the two operands without copying or
        # changing any source equation or arithmetic inside either pool.
        self._pool_argument = self.split(y)[4]
        try:
            return super()._coefficients(y, t, anchor)
        finally:
            self._pool_argument = None

    def _mv(self, matrix, value):
        result = super()._mv(matrix, value)
        if matrix is self.A:
            if self._pool_argument is None:
                raise ContractError('Native exterior pool used outside original coefficient evaluation')
            name = 'A_Vbar' if np.shares_memory(value, self._pool_argument) else 'A_release'
            return result + self.boundary[name]
        if matrix is self.B:
            return result + self.boundary['B_Vh']
        raise ContractError('Unexpected native exterior operator')


class BoundedInnerDynamics(GroupInnerDynamics):
    def __init__(self, assembly, parameters):
        super().__init__(assembly, parameters)
        self.gap_boundary_product = assembly.boundary_runtime['gap_product']
        self.gap = AffineProduct(self.gap, self.gap_boundary_product)
        self.identity = canonical_hash(dict(parent_inner=self.identity,
            exterior=assembly.boundary_runtime['identity']))

    def outputs(self, t, state, cone_release=None):
        result = super().outputs(t, state, cone_release)
        # The retained reciprocal gap graph exchanges charge with the held
        # exterior. Its opposite reservoir current completes the original
        # conservation predicate; it is observable and has no state equation.
        voltages = np.concatenate([self._mem(state, name, 'V') if name=='AII_AC'
            else self._bc(state, name, 'V') for name in self.gap_names])
        reservoir = (float(self.assembly.boundary_runtime['gap_missing_conductance_S'] @ voltages)
            + float(np.sum(self.gap_boundary_product)))
        result['local_patch_boundary', 'gap_current'] = np.array([reservoir], dtype=np.float64)
        return result


class BoundedRecording(GroupRecording):
    """Retain every actual field; empty, unrequested populations have no cells."""
    def _configure(self, assembly, artifactidentity):
        super()._configure(assembly, artifactidentity)
        omitted = [field.key for field in self.fields if not len(field.rows)]
        if not omitted:
            return
        self.fields = tuple(field for field in self.fields if len(field.rows))
        self.field_by_key = {field.key:field for field in self.fields}
        self.rows = {key:value for key,value in self.rows.items() if len(value)}
        self.components = [dict(component, ports={name:port for name,port in component['ports'].items()
            if port['recorded_count']}) for component in self.components]
        self.components = [component for component in self.components if component['ports']]
        self.layout_identity = canonical_hash(dict(parent_layout=self.layout_identity,
            empty_population_fields=omitted, components=self.components,
            rule='zero actual rows; no cell or state omitted from selected graph'))


def configure_boundary_runtime(assembly, parent, restricted):
    """Prepare constant missing-edge products once, never inside an RHS."""
    if not getattr(assembly, 'bounded_local', False):
        return None
    for route in assembly.routes.values():
        if route.kind == 'inner_chemical' and assembly.external_matrix('routes/'+route.name+'/mean_weights.npz').nnz:
            raise ContractError('Bounded chemical source closure is incomplete: '+route.name)
    B = assembly.external_matrix('routes/HC_to_cones/mean_weights.npz')
    if B.nnz:
        raise ContractError('Every retained Cone requires its actual HC feedback predecessors')
    original = GroupNativePhysics(assembly.base)
    frame = min(parent['native_inputs'])
    seed = SimpleNamespace(states=parent['states'], time_seconds=parent['time_seconds'])
    y, anchor = original.construction_point(seed, parent['native_inputs'][frame], (frame+1)/150., frame)
    output = original.evaluate(y, parent['time_seconds'], anchor).outputs
    A = assembly.external_matrix('routes/cones_to_HC/mean_weights.npz')
    gap_names = assembly.manifest['gap']['population_order']
    gap_values = np.concatenate([parent['states'][name,
        MEMBRANE if name=='AII_AC' else BC_MECH, 'V'] for name in gap_names])
    products = dict(A_Vbar=_owned(A@output['Vbar']), A_release=_owned(A@output['release']),
        B_Vh=_owned(B@output['Vh']),
        gap_product=_owned(assembly.external_matrix('gap/laplacian_S.npz')@gap_values),
        gap_missing_conductance_S=_owned(-np.asarray(
            assembly.external_matrix('gap/laplacian_S.npz').sum(axis=1)).ravel()))
    descriptor = dict(schema=SCHEMA, source_time_seconds=parent['time_seconds'],
        parent_states_sha256=canonical_hash(parent['states']),
        boundary_products_sha256=canonical_hash(products),
        classification='HELD_SOURCE_CONSTRUCTION_IVP_EXTERIOR_NOT_EQUILIBRATED_BACKGROUND',
        background_qualified=False, arbitrary_dynamic_exterior_input_supported=False,
        weights_renormalized=False,
        units=dict(A_Vbar='vH_native_voltage', A_release='dimensionless',
            B_Vh='vH_native_voltage', gap_product='A', gap_missing_conductance_S='S'),
        validity='Only the measured input/time/core/Halo comparison; no natural-video generalization')
    assembly.boundary_runtime = dict(products, identity=canonical_hash(descriptor))
    assembly.runtime_factories = dict(inner=BoundedInnerDynamics, native=BoundedNativePhysics)
    restricted['patch_boundary'] = dict(descriptor, identity=assembly.boundary_runtime['identity'])
    return restricted['patch_boundary']


def verify_initial_boundary(repo_root, assembly, seed):
    """One source-point algebra witness; caller must own the model-work guard.

    No integration, event commit, history change or RNG draw occurs here.
    This does not qualify the subsequent held-exterior approximation.
    """
    from pathlib import Path
    from .group_time_storage import load_bundle
    from .group_inner_compiled import CompiledGroupInnerRHS
    from .group_inner_dynamics import CONTACTS
    if not getattr(assembly, 'bounded_local', False):
        raise ContractError('Initial boundary witness requires the explicit bounded graph')
    binding = assembly.config['input_identity']
    parent = load_bundle((Path(repo_root)/binding['path']).parent,
        expected_kind='source_construction_seed')['payload']
    original_native = GroupNativePhysics(assembly.base)
    local_native = BoundedNativePhysics(assembly)
    frame = min(parent['native_inputs'])
    stop = (frame+1)/150.
    time = parent['time_seconds']
    full_y, full_anchor = original_native.construction_point(SimpleNamespace(states=parent['states'],
        time_seconds=time), parent['native_inputs'][frame], stop, frame)
    local_y, local_anchor = local_native.construction_point(SimpleNamespace(states=seed['states'],
        time_seconds=time), seed['native_inputs'][frame], stop, frame)
    full_eval = original_native.evaluate(full_y, time, full_anchor)
    local_eval = local_native.evaluate(local_y, time, local_anchor)
    native = {}
    for name, section in local_native.slices.items():
        rows = (assembly.parent_auxiliary_rows['cones'] if name=='z1_aux' else
            assembly.parent_population_rows['HC' if name in ('w','z2','Vh') else 'cones'])
        expected = full_eval.rhs[original_native.slices[name]][rows]
        actual = local_eval.rhs[section]
        native[name] = dict(count=len(actual), max_abs=float(np.max(np.abs(expected-actual), initial=0.)),
            reference_max_abs=float(np.max(np.abs(expected), initial=0.)))
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-10)
    full_inner = GroupInnerDynamics(assembly.base, parent['inner_parameters'])
    local_inner = BoundedInnerDynamics(assembly, seed['inner_parameters'])
    full_state = {key:parent['states'][key] for key in full_inner.state_keys}
    local_state = {key:seed['states'][key] for key in local_inner.state_keys}
    full_rhs = full_inner.rhs(time, full_state, full_eval.outputs['release'])
    local_rhs = local_inner.rhs(time, local_state, local_eval.outputs['release'])
    inner = {}
    for key, actual in local_rhs.items():
        rows = assembly.parent_contact_rows[key[1]] if key[1] in CONTACTS else assembly.parent_population_rows[key[0]]
        expected = full_rhs[key][rows]
        inner[':'.join(key)] = dict(count=len(actual), max_abs=float(np.max(np.abs(expected-actual), initial=0.)),
            reference_max_abs=float(np.max(np.abs(expected), initial=0.)))
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-10)
    compiled = CompiledGroupInnerRHS(local_inner, repo_root=repo_root)
    actual = compiled.rhs_flat(time, local_inner.pack(local_state), local_eval.outputs['release'])
    expected = local_inner.pack(local_rhs)
    compiled_error = float(np.max(np.abs(actual-expected), initial=0.))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-10)
    outputs = local_inner.outputs(time, local_state, local_eval.outputs['release'])
    reservoir = float(outputs['local_patch_boundary','gap_current'][0])
    dynamic_sum = sum(float(np.sum(outputs[pop,'gap_current'])) for pop in local_inner.gap_names)
    return dict(status='PASS_INITIAL_ALGEBRA_ONLY', no_forward_or_rng_draw=True,
        native=native, inner=inner, compiled_vs_python_max_abs=compiled_error,
        dynamic_gap_current_sum_A=dynamic_sum, exterior_reservoir_current_A=reservoir,
        total_gap_current_A=dynamic_sum+reservoir,
        numerical_tolerance=dict(rtol=2e-12, atol=1e-10),
        held_exterior_future_error_qualified=False, background_qualified=False)
