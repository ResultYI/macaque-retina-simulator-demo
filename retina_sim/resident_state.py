"""Explicit device-owned state for the opt-in, unvalidated resident backend.

No project model or CUDA runtime is initialized by this module. Construction,
upload and full export happen only when explicitly requested by a caller.
"""
from copy import deepcopy
from dataclasses import dataclass

from .contracts import ContractError, SimulationState


@dataclass
class ResidentStateHandle:
    """Internal execution state; never pretends to be a CPU RetinaState.

    The runner owns y. Observers must treat tensors as read-only. Commits replace
    y rather than overwrite earlier observer views; event patches copy on write.
    Explicit export is the only ordered_state_arrays conversion.
    """

    y: object
    time_seconds: float
    metadata: dict
    assembly_hash: str
    grid_origin_seconds: float
    sample_index: int
    spike_states: dict
    delay_states: dict
    input_position: dict
    link_delay_states: dict
    component_tick_indices: dict
    generation: int = 0

    @property
    def solver_config(self):
        return self.metadata["solver_config"]

    @property
    def model_hash(self):
        return self.metadata["model_hash"]

    @staticmethod
    def _validate_aux(state, runner):
        """Reject incomplete imported/exported clocks and stochastic owners."""
        import math
        from numbers import Integral

        spikes = {
            cid for cid, component in runner.compiled.components.items()
            if hasattr(component.native_model, "sample_spikes")
        }
        delays = {
            cid for cid, component in runner.compiled.components.items()
            if hasattr(component.native_model, "observation_interval_seconds")
        }
        if set(state.spike_states) != spikes:
            raise ContractError("Resident spike auxiliary component coverage differs")
        if set(state.component_tick_indices) != set(runner.periods):
            raise ContractError("Resident component clock coverage differs")
        if set(state.delay_states) != delays or delays - {runner.partition.cid}:
            raise ContractError("Resident observation delay coverage differs")
        indices = [state.sample_index, *state.component_tick_indices.values()]
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0
               for value in indices):
            raise ContractError("Resident sampling indices must be nonnegative integers")
        if (not math.isfinite(state.grid_origin_seconds) or
                state.grid_origin_seconds > state.time_seconds):
            raise ContractError("Resident grid origin is invalid")
        required = {"history", "last_bin_counts", "rng_state", "pending_events",
                    "conditional_rate_Hz"}
        for aux in state.spike_states.values():
            if not isinstance(aux, dict) or not required.issubset(aux):
                raise ContractError("Resident spike auxiliary payload is incomplete")
            rng = aux["rng_state"]
            if not isinstance(rng, dict) or rng.get("bit_generator") != "PCG64":
                raise ContractError("Resident spike auxiliary requires the original PCG64 state")

    @classmethod
    def upload(cls, snapshot, runner):
        """Upload an already identity-validated, complete CPU snapshot."""
        import numpy as np
        from .resident_delay import ResidentTimestampedDelay

        cls._validate_aux(snapshot, runner)
        if snapshot.link_delay_states:
            raise ContractError("Resident backend does not support explicit delayed links")
        torch = runner.torch
        flat = np.concatenate(snapshot.ordered_state_arrays)
        y = torch.as_tensor(flat, dtype=torch.float64,
                            device=runner.partition.gather.device).clone()
        fields = (
            "state_spec_hash", "graph_hash", "parameter_hash", "model_id",
            "model_version", "model_hash", "input_interpolation",
            "solver_config", "evidence_tag", "optional_rng_state",
        )
        metadata = {key: getattr(snapshot.dynamic, key) for key in fields}
        metadata["model_hash"] = runner.model.model_hash
        delays = {
            cid: ResidentTimestampedDelay.from_payload(data, torch, y.device)
            for cid, data in snapshot.delay_states.items()
        }
        return cls(
            y=y, time_seconds=float(snapshot.time_seconds), metadata=metadata,
            assembly_hash=snapshot.assembly_hash,
            grid_origin_seconds=snapshot.grid_origin_seconds,
            sample_index=snapshot.sample_index,
            spike_states=deepcopy(snapshot.spike_states), delay_states=delays,
            input_position=deepcopy(snapshot.input_position), link_delay_states={},
            component_tick_indices=dict(snapshot.component_tick_indices),
        )

    def fork(self):
        """Independent mutable aux; device rows and y are immutable until replaced."""
        return type(self)(
            y=self.y, time_seconds=self.time_seconds, metadata=dict(self.metadata),
            assembly_hash=self.assembly_hash,
            grid_origin_seconds=self.grid_origin_seconds, sample_index=self.sample_index,
            spike_states=deepcopy(self.spike_states),
            delay_states={cid: delay.fork() for cid, delay in self.delay_states.items()},
            input_position=deepcopy(self.input_position),
            link_delay_states=deepcopy(self.link_delay_states),
            component_tick_indices=dict(self.component_tick_indices),
            generation=self.generation,
        )

    def validate(self, runner):
        import math

        torch = runner.torch
        self._validate_aux(self, runner)
        expected = {
            "state_spec_hash": runner.model._state_spec_hash,
            "graph_hash": runner.model.graph.graph_hash,
            "parameter_hash": runner.model.parameters.parameter_hash,
            "model_hash": runner.model.model_hash,
            "model_id": runner.model.model_id,
            "model_version": runner.model.version,
            "evidence_tag": runner.model.evidence_tag,
        }
        if self.assembly_hash != runner.compiled.assembly_hash:
            raise ContractError("Resident state assembly identity differs")
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ContractError("Resident state identity differs: " + key)
        runner.validate_solver(self.solver_config)
        if (not math.isfinite(self.time_seconds) or
                self.y.dtype != torch.float64 or self.y.ndim != 1 or
                self.y.numel() != runner.nstate or
                self.y.device != runner.partition.gather.device):
            raise ContractError("Resident state time/device/layout differs")
        if not torch.isfinite(self.y).all().item():
            raise ContractError("Nonfinite resident state; no clipping")
        if self.link_delay_states:
            raise ContractError("Resident backend does not support explicit delayed links")

    def commit(self, y, time_seconds, runner):
        """Take ownership of a newly allocated interval result or event copy."""
        import math

        if (not math.isfinite(time_seconds) or time_seconds < self.time_seconds or
                y.shape != self.y.shape or y.dtype != self.y.dtype or
                y.device != self.y.device):
            raise ContractError("Invalid resident state commit")
        if not runner.torch.isfinite(y).all().item():
            raise ContractError("Nonfinite resident state commit")
        self.y = y
        self.time_seconds = float(time_seconds)
        self.generation += 1

    def export(self, runner):
        """Materialize one coherent full snapshot, including every delay and aux."""
        from .hybrid import RetinaState

        self.validate(runner)
        flat = self.y.detach().cpu().numpy().copy()
        dynamic = SimulationState(
            time_seconds=self.time_seconds,
            ordered_state_arrays=runner.model.unpack(flat), **self.metadata,
        )
        return RetinaState(
            dynamic, self.assembly_hash, self.grid_origin_seconds, self.sample_index,
            deepcopy(self.spike_states),
            {cid: delay.payload() for cid, delay in self.delay_states.items()},
            deepcopy(self.input_position), deepcopy(self.link_delay_states),
            dict(self.component_tick_indices),
        )
