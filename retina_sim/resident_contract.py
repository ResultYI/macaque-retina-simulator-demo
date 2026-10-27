"""Frozen prospective gate and bindings, not a numerical acceptance certificate."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import re

from .contracts import ContractError, canonical_hash


CONTRACT_SHA256 = "3cfab9768a856c13354ab38cc65b506da60661fe3e3b0f05e46c647c08981a77"
SOURCE_PINS_FILENAME = "resident_native_source_pins_v6.json"
SOURCE_PINS_SHA256 = "757112307dddb49e8c89ae49ed942d3eb6a2b33db6c59b90f3e337ef51024f3e"


def _channel_manifest(channels):
    """Metadata only; actual prepared content hashes are separately mandatory."""
    return {
        name: {
            "unit": channel.unit, "meaning": channel.meaning,
            "evidence_tag": channel.evidence_tag,
            "start_seconds": channel.start_seconds, "stop_seconds": channel.stop_seconds,
            "interpolation": channel.interpolation,
            "breakpoints_seconds": channel.breakpoints_seconds,
            "input_contract": channel.input_contract,
        }
        for name, channel in channels.items()
    }


class ResidentNativeContract:
    """Require the exact new scope; never load the old small-fixture v2 gate.

    The full runtime comparator is deliberately a separate, not-yet-run QA
    task. Loading this object establishes requirements, never a 'passed' result.
    """

    def __init__(self):
        directory = Path(__file__).resolve().parent.parent / "configurations" / "numerical_equivalence"
        data = (directory / "resident_native_fp64_v1.json").read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        if self.sha256 != CONTRACT_SHA256:
            raise ContractError("Frozen resident numerical contract changed")
        self.record = json.loads(data)
        # Prior manifests remain historical evidence. v6 records storage/lifetime
        # changes; no source manifest is numerical acceptance.
        raw_pins = (directory / SOURCE_PINS_FILENAME).read_bytes()
        self.source_pins_hash = hashlib.sha256(raw_pins).hexdigest()
        if self.source_pins_hash != SOURCE_PINS_SHA256:
            raise ContractError("Frozen resident reference source manifest changed")
        manifest = json.loads(raw_pins)
        self.source_pins = manifest["sources"]
        for name, expected in self.source_pins.items():
            if Path(name).name != name:
                raise ContractError("Resident reference source pin must be a local filename")
            actual = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            if actual != expected:
                raise ContractError("Unreviewed resident reference source change: " + name)
        # Prepared-array hashes alone do not bind the frame-selection function.
        # Pin the reviewed provider as well, without importing or evaluating it.
        self.input_provider_sources = manifest["input_provider_sources"]
        root = directory.parent.parent
        for name, expected in self.input_provider_sources.items():
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ContractError("Unreviewed resident input provider source change: " + name)

    def validate_solver(self, config):
        required = self.record["solver_required"]
        if config.dtype != "float64" or config.fixed_step_seconds is not None:
            raise ContractError("Resident scope requires original FP64 adaptive configuration")
        for name in ("method", "rtol", "atol", "max_step"):
            if getattr(config, name) != required[name]:
                raise ContractError("Resident full-graph solver contract differs: " + name)
        if config.input_interpolation not in ("analytic", "linear", "zero_order_hold"):
            raise ContractError("Unreviewed resident interpolation")

    def validate_reference(self, runner):
        identity = runner.model.compute_metadata.get("integration_backend", {})
        scope = self.record["scenario_required"]
        if identity.get("backend") != scope["reference_backend"]:
            raise ContractError("Resident QA reference backend differs")
        native = runner.partition.native
        counts = {
            "components": len(runner.compiled.components),
            "native_cones": native.nc, "native_HC": native.nh,
            "state_scalars": sum(spec.shape[0] for spec in runner.model.state_specs),
            "RGC_outputs": sum(
                len(component.state_specs[0].cell_ids)
                for component in runner.compiled.components.values()
                if hasattr(component.native_model, "sample_spikes")),
        }
        if any(counts[key] != scope[key] for key in counts):
            raise ContractError("Resident contract is bound to the current full graph, not a small fixture")

    def bind_inputs(self, reference, snapshot, reference_state, channels,
                    channel_data_hashes, monitor_policy):
        """Bind complete state/aux and caller-computed prepared-input digests.

        The future QA harness must compute digests from actual prepared stimulus
        content; metadata/filenames alone do not certify that content. No input
        evaluator is invoked here, and this function is not a QA pass.
        """
        self.validate_solver(snapshot.solver_config)
        expected = {external.channel for external in reference.compiled.spec.external_inputs}
        if set(channels) != expected or set(channel_data_hashes) != expected:
            raise ContractError("Resident QA requires one prepared-content digest per actual channel")
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in channel_data_hashes.values()):
            raise ContractError("Resident prepared input content must have lowercase SHA256 digests")
        record = {
            "contract_hash": self.sha256, "source_pins_hash": self.source_pins_hash,
            "reference_model_hash": reference.model.model_hash,
            "assembly_hash": reference.compiled.assembly_hash,
            "parameter_hash": reference.model.parameters.parameter_hash,
            "graph_hash": reference.model.graph.graph_hash,
            "state_spec_hash": reference.model._state_spec_hash,
            "initial_state_and_aux_hash": canonical_hash(snapshot),
            "monitor_reference_hash": canonical_hash(reference_state),
            "monitor_policy_hash": canonical_hash(asdict(monitor_policy)),
            "solver_hash": canonical_hash(asdict(snapshot.solver_config)),
            "channel_metadata_hash": canonical_hash(_channel_manifest(channels)),
            "channel_data_hashes": dict(channel_data_hashes),
        }
        if 'resident_background_origin_seconds' in snapshot.input_position:
            record['background_preparation_origin_seconds'] = snapshot.input_position['resident_background_origin_seconds']
        record["binding_hash"] = canonical_hash(record)
        return record

    def validate_channels(self, channels, binding):
        if canonical_hash(_channel_manifest(channels)) != binding["channel_metadata_hash"]:
            raise ContractError("Resident QA input channel metadata/breakpoints changed")
