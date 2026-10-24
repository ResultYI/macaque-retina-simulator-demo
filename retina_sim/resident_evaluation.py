"""Partition-aware same-point evaluation; no full native CPU materialization.

The CPU inner observer/gather/gap order mirrors the pinned JointNetwork.
Native output values remain FP64 device tensors; only compact release consumers
cross the boundary. This implementation has not been executed or accepted.
"""
from dataclasses import dataclass
import numpy as np

from .contracts import ContractError, Quantity


@dataclass
class ResidentEvaluation:
    inner_y: object
    native_rhs: object
    inner_rhs: object
    inputs: dict
    outputs: dict
    native_outputs: dict
    local: dict

    def local_blocks(self, cid):
        if cid not in self.local:
            raise ContractError("Native blocks must not be materialized through CPU evaluation")
        return self.local[cid]


class ResidentEvaluator:
    def __init__(self, runner, compact_incoming):
        self.runner = runner
        self.incoming = compact_incoming
        native_id = runner.partition.cid
        if set(runner.model._context_providers) != {native_id}:
            raise ContractError("Resident evaluator requires the reviewed native-only context provider")
        self.inner_ids = tuple(cid for cid in runner.compiled.components if cid != native_id)

    def __call__(self, t, y, bundle, *, sampling=False, sampled_components=None,
                 require_rhs=False):
        runner = self.runner
        partition = runner.partition
        torch = runner.torch
        compiled = runner.compiled
        cid = partition.cid
        if (y.dtype != torch.float64 or y.shape != (runner.nstate,) or
                y.device != partition.gather.device):
            raise ContractError("Resident evaluation state layout differs")
        expected = {partition.native_external.channel} | {
            external.channel for external, _ in partition.external
        }
        if set(bundle.channels) != expected or bundle.delay_buffers:
            raise ContractError("Resident evaluation external/delay channel set differs")

        q_light = bundle.channels[partition.native_external.channel]
        partition.native_port.validate(q_light)
        light = partition._light(q_light)
        # Exact current state/input, never a prior endpoint's f_new.
        native_rhs, release = partition.native(y[:partition.offset], light)
        native = compiled.components[cid].native_model
        blocks = y[:partition.offset].split(partition.native.sizes)
        p = native.p
        # Public native readouts retain their declared expression, not an
        # algebraically reconstructed q=release*adaptation with extra rounding.
        q = 1. / (1. + torch.exp(-(
            (blocks[4] - torch.mv(partition.native.B, blocks[9]) - p["V_k"]) / p["V_n"]
        )))
        native_outputs = {
            spec.state_name: block for spec, block in zip(native.state_specs, blocks)
        }
        native_outputs.update(q=q, release=release, Ios=blocks[2] ** p["n_X"])
        if set(native_outputs) != set(compiled.components[cid].outputs):
            raise ContractError("Unreviewed native output registry")
        for name, value in native_outputs.items():
            port = compiled.components[cid].outputs[name]
            if (not port.unit or not port.meaning or
                    value.dtype != torch.float64 or value.shape != (len(port.cell_ids),) or
                    port.evidence_tag != native.tag):
                raise ContractError("Native device output metadata differs: " + name)
            if not torch.isfinite(value).all().item():
                raise ContractError("Nonfinite native device output: " + name)

        inner_y = y[partition.offset:].detach().cpu().numpy().copy()
        if not np.all(np.isfinite(inner_y)):
            raise ContractError("Nonfinite resident inner state")
        compact_release = torch.index_select(release, 0, partition.gather).cpu().numpy().copy()
        # This private Quantity has compact columns only. It is consumed solely
        # by remapped link matrices; never validated as the full public port.
        release_port = compiled.components[cid].outputs["release"]
        compact_q = Quantity(compact_release, release_port.unit,
                             release_port.meaning, release_port.evidence_tag)
        local = {
            key: tuple(inner_y[s.start-partition.offset:s.stop-partition.offset]
                       for s in runner.model._local_slices[key])
            for key in self.inner_ids
        }
        inputs = {key: {} for key in self.inner_ids}
        outputs = {}
        for external in compiled.spec.external_inputs:
            value = bundle.channels[external.channel]
            port = compiled.components[external.target_component].inputs[external.target_port]
            port.validate(value)
            if external.target_component != cid:
                inputs[external.target_component][external.target_port] = value
        gathered = {}

        def gather(target_cid, port_name):
            if port_name in inputs[target_cid]:
                return inputs[target_cid][port_name]
            key = (target_cid, port_name)
            if key in gathered:
                return gathered[key]
            target = compiled.components[target_cid].inputs[port_name]
            links = self.incoming.get(key, ())
            if not links:
                raise ContractError("Unbound resident runtime input")
            values = np.zeros(len(target.cell_ids))
            for link in links:
                spec = link.specification
                if link.delayed_operator is not None:
                    raise ContractError("Resident evaluator does not lower delayed links")
                source = compact_q if spec.source_component == cid else outputs.get(
                    (spec.source_component, spec.source_port))
                if source is None:
                    raise ContractError("Instantaneous source output unavailable")
                values += link.matrix @ source.values
            result = Quantity(values, target.unit, target.meaning, target.evidence_tag)
            gathered[key] = result
            return result

        for block in compiled.output_order:
            if block.cyclic:
                raise ContractError("Resident evaluator does not lower algebraic closures")
            owner, port_name = block.nodes[0]
            if owner == cid:
                continue
            component = compiled.components[owner]
            dependencies = {
                name: gather(owner, name)
                for name in component.output_dependencies[port_name]
            }
            dependencies["__sampling__"] = sampling and (
                sampled_components is None or owner in sampled_components)
            dependencies["__closure_trial__"] = False
            value = component.observe(port_name, t, local[owner], dependencies)
            component.outputs[port_name].validate(value)
            outputs[(owner, port_name)] = value
        for owner in self.inner_ids:
            component = compiled.components[owner]
            for name, port in component.inputs.items():
                value = gather(owner, name)
                port.validate(value)
                inputs[owner][name] = value
                if (value.meaning in ("effective_synaptic_drive", "effective_synaptic_activation",
                                      "excitatory_conductance", "inhibitory_conductance") and
                        np.any(value.values < 0)):
                    raise ContractError("Negative physical/effective input after closure")
        gap = {
            owner: np.zeros(len(compiled.components[owner].outputs["voltage"].cell_ids))
            for owner in self.inner_ids
            if hasattr(compiled.components[owner].native_model, "membrane_capacitance")
        }
        for left, right, ix, iy, conductance in runner.model._gap_plans:
            current = conductance * (outputs[(right, "voltage")].values[iy] -
                                     outputs[(left, "voltage")].values[ix])
            np.add.at(gap[left], ix, current)
            np.add.at(gap[right], iy, -current)
        for owner, current in gap.items():
            value = Quantity(current, "A", "electrical_gap_current",
                             compiled.components[owner].evidence_tag)
            inputs[owner]["__gap_current__"] = value
            outputs[(owner, "gap_current")] = value

        inner_rhs = None
        if require_rhs:
            values = [bundle.channels[external.channel].values
                      for external, _ in partition.external]
            values.append(compact_release)
            # Work belongs to this resident runner. Caller serializes access.
            inner_rhs = np.asarray(partition.fn(
                inner_y, np.concatenate(values), partition.work, partition.program,
                partition.weights, partition.indices, partition.pointers,
                partition.derivatives), dtype=np.float64).copy()
            if inner_rhs.shape != inner_y.shape or not np.all(np.isfinite(inner_rhs)):
                raise ContractError("Nonfinite or malformed resident inner RHS")
        inner_y.setflags(write=False)
        return ResidentEvaluation(inner_y, native_rhs if require_rhs else None,
                                  inner_rhs, inputs, outputs, native_outputs, local)
