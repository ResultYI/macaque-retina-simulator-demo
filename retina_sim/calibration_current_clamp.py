"""Isolated current-injection boundary around the unchanged reference RGC factory.

This experimental boundary has a distinct mechanism identity and is deliberately
absent from compiled/GPU lowering. The default circuit keeps its reviewed source.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
from .assembly import PortSpec
from .contracts import Quantity, ContractError, canonical_hash

KEY = 'HistoryConductanceRGCCurrentClampBoundary'


def current_clamp_component(spec, populations, sources, mode):
    boundary = spec.options.get('module_boundary', {})
    if boundary.get('mode') != 'current_clamp':
        raise ContractError('Current-injection boundary requires an isolated current-clamp experiment')
    from .components import reference_component_registry
    base = reference_component_registry().instantiate(
        replace(spec, mechanism='HistoryConductanceRGC'), populations, sources, mode)
    inputs = dict(base.inputs)
    inputs['injected_current'] = PortSpec('injected_current', 'A', 'injected_current_positive_inward',
                                        base.outputs['voltage'].cell_ids, base.evidence_tag)

    def derivative(t, blocks, incoming):
        # Core membrane/receptor equations remain in inner.rgc_component.
        drive = {key: value for key, value in incoming.items() if key != 'injected_current'}
        result = base.derivative(t, blocks, drive)[0]
        current = incoming['injected_current'].require('A', 'injected_current_positive_inward', result.values.shape)
        return (Quantity(result.values + current / base.native_model.C,
                         result.unit, result.meaning, result.evidence_tag),)

    result = replace(base, mechanism=KEY, inputs=inputs, derivative=derivative,
        implementation_hash=canonical_hash(dict(base=base.implementation_hash,
            boundary=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
    result.base_implementation_hash = base.implementation_hash
    return result
