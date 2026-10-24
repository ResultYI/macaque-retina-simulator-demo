"""True-equation reduced diagnostics plus the original CPU inner evaluator."""
import numpy as np

from dataclasses import dataclass
from .contracts import ContractError, Quantity
from .resident_evaluation import ResidentEvaluator, ResidentEvaluation
from .cone_hc_reduced_native import AnalyticREAnchor


class ReducedEvaluator(ResidentEvaluator):
    def __call__(self,t,point,bundle,*,sampling=False,sampled_components=None,require_rhs=False):
        runner = self.runner
        part = runner.partition
        point.native.check_owned()
        if t!=point.native.time_seconds or point.native.source_version!=runner.native.source_version:
            raise ContractError('Reduced evaluation point/time/source differs')
        expected = {part.native_external.channel}|{e.channel for e,_ in part.external}
        if set(bundle.channels)!=expected or bundle.delay_buffers:
            raise ContractError('Reduced external/delay input set differs')
        q = bundle.channels[part.native_external.channel]
        part.native_port.validate(q)
        R,E = point.native.anchor.at(t)
        # At an exact input edge the two observations use the same continuous
        # states and different light/R RHS. No time snapping or feedback delay.
        anchor = AnalyticREAnchor(t,R,E,part._light(q),np.nextafter(t,np.inf),
                                  point.native.anchor.input_epoch)
        native = runner.native.evaluate(point.native.y,t,anchor)
        outputs = dict(native.outputs)
        outputs['R_rhs'],outputs['E_rhs'] = native.analytic_rhs['R'],native.analytic_rhs['E']
        cursor = runner._active_cursor
        if point.compact_release is not None:
            compact = point.compact_release
        elif cursor is not None and any(p.start<=t<=p.stop for p in cursor.packets):
            compact = cursor.packet_at(t,left=bundle.delay_left_limit).release_at(t)
        else:
            # Migration/fully consumed endpoints only. Inner RK stages always
            # use the owned packet and never execute a per-stage D2H operation.
            compact = runner.torch.index_select(outputs['release'],0,part.gather).cpu().numpy().copy()
        return self.evaluate_inner(t,point.inner.copy(),bundle,compact,
            native_rhs=native.rhs if require_rhs else None,native_outputs=outputs,
            sampling=sampling,sampled_components=sampled_components,require_rhs=require_rhs)


    def evaluate_inner(self, t, inner_y, bundle, compact_release, *, native_rhs=None,
                       native_outputs=None, sampling=False, sampled_components=None,
                       require_rhs=False):
        """Original CPU gather/observe/gap/RHS order, also used by reduced native."""
        runner = self.runner
        partition = runner.partition
        compiled = runner.compiled
        cid = partition.cid
        if (inner_y.dtype != np.float64 or inner_y.shape != (partition.inner_size,)
                or not np.isfinite(inner_y).all()
                or compact_release.shape != (len(partition.used),)
                or not np.isfinite(compact_release).all()):
            raise ContractError('CPU inner evaluation layout/domain differs')
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


@dataclass(frozen=True)
class ReducedStateSpec:
    state_name: str
    cell_ids: tuple
    shape: tuple
    unit: str
    meaning: str
    initial_condition_source: str


def native_registry(runner):
    """State names, IDs and units for actual dynamic/analytic states only."""
    from dataclasses import replace
    original = runner.compiled.components[runner.partition.cid].native_model
    old = {s.state_name:s for s in original.state_specs}
    rows = []
    for name in ('X','C','Vis','g_i','Vbar','w','z2','Vh','z1_aux','R','E'):
        if name=='w':
            spec = ReducedStateSpec('w',old['z2'].cell_ids,old['z2'].shape,old['z2'].unit,
                                    'aggregated_transmitter_drive','Exact w=A z1 migration')
        elif name=='z1_aux':
            if not runner.native.aux_cell_ids:
                continue
            spec = replace(old['z1'],state_name=name,cell_ids=runner.native.aux_cell_ids,
                           shape=(len(runner.native.aux_cell_ids),))
        else:
            spec = old[name]
        rows.append(spec)
    return tuple(rows)


def state_arrays(runner,point):
    values = runner.native.blocks(point.native.y)
    values['R'],values['E'] = point.native.anchor.at(point.native.time_seconds)
    return values
