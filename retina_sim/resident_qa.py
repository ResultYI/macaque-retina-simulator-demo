"""Prospective resident-native numerical QA, never an execution authorization.

This module imports only the standard library.  NumPy is imported lazily when
the explicitly called comparator receives already materialized CPU witnesses.
It never imports a model, constructs a graph, touches CUDA, runs a solver, or
consumes RNG.  A witness producer must observe the ORIGINAL operations; it must
not reconstruct missing random draws by advancing or replaying an RNG.

The stable entry point is ``ResidentQAComparator(plan, evidence_sink=...)``.
``plan`` is sealed BEFORE execution with ``seal_plan(plan)``.  It contains:

* contract_hash, binding (the complete ResidentNativeContract binding),
  arm_identity_hashes={reference: ..., candidate: ...}, unit_scales,
  monitor_relative_tolerance;
* states/ports: ordered rows {name, unit, meaning, cell_ids, initial}; state
  rows also have period (positive seconds for pure discrete transitions, else
  None). Initial vectors are immutable CPU float64 arrays, not Python lists;
* points: ordered {id, time_seconds, phase, context_hash}; intervals: ordered
  {id, start, stop, dense_times}; monitors: ordered {id, width, point_ids};
* draws: ordered {id, component, cell_id, bin_start_seconds, dt,
  voltage_state, cell_index, alpha, beta}; rgc_cells: ordered (component, id);
* checkpoint_ids, failure_cases ({id, label, barrier}), ownership_ids;
* scenario={start, stop, background_stop, frame_boundaries, rod_period,
  rgc_period, fit_eligible}.  This contract covers the existing eight cells,
  not a new requirement for every research RGC class.

Every compare_* method consumes one aligned pair and retains only small gate
summaries. Arrays are processed in bounded chunks. ``evidence_sink(record)``
must synchronously preserve the record (for example into a bounded disk spool)
and return normally; it must not retain all callback arrays in RAM. It receives
full raw monitor reports and every per-cell monitor statistic difference.
Missing evidence/sink/callbacks yields INCONCLUSIVE; numeric or exact-contract
violations yield FAIL. ``finish`` never grants production or fitting approval.

Point witnesses have the point fields plus state/rhs/ports/transitions mappings
of name -> one-dimensional float64 CPU arrays, delays={queue: {port, values}},
aux with EXACT_AUX_FIELDS, and checks with EXACT_CHECK_FIELDS. Delay values
include every retained row INCLUDING prehistory; timestamps, bracket choices,
prune/prehistory rules and ordering belong in aux.delay_structure. Conditional
rates and pending event times must not be hidden in the exact aux record: they
are supplied to compare_draw under the declared numerical budgets.

Monitor witnesses: {id,width,point_ids,raw_report,rows}, with ordered rows
{state,den,lo,hi,rhs,defect}. Raw report is the original _WindowMonitor.report,
without schema changes. Draw/solver/checkpoint/failure schemas are documented
on their comparison methods. No assertions of instrumentation provenance can
replace the complete raw witnesses required here.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from contextlib import nullcontext


CONTRACT_SHA256 = "3cfab9768a856c13354ab38cc65b506da60661fe3e3b0f05e46c647c08981a77"
EXACT_AUX_FIELDS = frozenset((
    "grid_origin_seconds", "component_tick_indices", "sample_index", "history",
    "last_bin_counts", "rng_state", "draw_order", "draw_count",
    "delay_structure", "pending_event_identity", "input_phase", "event_order",
    "generations"))
EXACT_CHECK_FIELDS = frozenset((
    "state_domain", "port_validation", "gap_decisions", "history_decisions",
    "failure_labels", "coverage", "failure_barrier"))
REQUIRED_BINDINGS = frozenset((
    "contract_hash", "source_pins_hash", "reference_model_hash", "assembly_hash",
    "parameter_hash", "graph_hash", "state_spec_hash", "initial_state_and_aux_hash",
    "monitor_reference_hash", "monitor_policy_hash", "solver_hash",
    "channel_metadata_hash", "channel_data_hashes", "binding_hash"))
RUNTIME_PAYLOAD_FIELDS = frozenset((
    "dynamic", "assembly_hash", "grid_origin_seconds", "sample_index", "spike_states",
    "delay_states", "input_position", "link_delay_states", "component_tick_indices"))


class QAIncomplete(ValueError):
    """A required/proven witness is absent; this is never a pass."""


def require_registered_observable_scales(state_specs, components, unit_scales, derived_ports=None):
    """Reject missing frozen output budgets before RHS, uploads or evidence I/O.

    A named physical unit is not itself a numerical allowance. In particular,
    ribbon release events/s is not the already-declared RGC conditional rate.
    This check must not invent a floor or substitute an unrelated Hz tolerance.
    """
    missing = []
    for spec in state_specs:
        if spec.unit not in unit_scales:
            missing.append('state:'+spec.state_name+' ['+spec.unit+']')
    for cid, component in components.items():
        for name, port in component.outputs.items():
            if port.unit not in unit_scales:
                descriptor = (derived_ports or {}).get(cid+'.'+name)
                if descriptor is None:
                    missing.append('port:'+cid+'.'+name+' ['+port.unit+']')
                else:
                    from .resident_derived_ports import validate_descriptor
                    from .contracts import canonical_hash
                    validate_descriptor(descriptor)
                    if (name!='event_rate' or descriptor['component']!=cid or
                            component.mechanism!='ReceptorConductanceMembrane' or
                            descriptor['unit']!=port.unit or descriptor['meaning']!=port.meaning or
                            tuple(descriptor['cell_ids'])!=tuple(port.cell_ids) or
                            descriptor['component_identity_hash']!=canonical_hash(component.identity) or
                            descriptor['parameter_hash']!=component.parameters.parameter_hash):
                        raise QAIncomplete('Derived port identity differs: '+cid+'.'+name)
    if missing:
        raise QAIncomplete('Frozen observable budget missing: '+', '.join(missing))


def _plain(value):
    """Strict JSON metadata only; no implicit tensor/array conversion."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise QAIncomplete("Nonfinite metadata")
        return value
    if isinstance(value, Mapping):
        if any(type(k) is not str for k in value):
            raise QAIncomplete("Metadata mapping keys must be strings")
        return {k: _plain(v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return [_plain(v) for v in value]
    raise QAIncomplete("Metadata must be plain CPU JSON values")


def _json(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _hash_metadata(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _array_digest(value, np, chunk=65536):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float64") or value.ndim != 1:
        raise QAIncomplete("Initial values must be 1-D CPU NumPy float64 arrays")
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    for start in range(0, value.size, chunk):
        part = value[start:start+chunk]
        if not np.isfinite(part).all():
            raise QAIncomplete("Nonfinite frozen initial state/output")
        digest.update(part.astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def plan_digest(plan):
    """Hash complete metadata and actual initial vectors, without running a model."""
    import numpy as np
    if plan.get('plan_digest_version',1) not in (1,2):
        raise QAIncomplete('Unreviewed prospective plan digest version')

    metadata = {k: v for k, v in plan.items() if k not in ("plan_sha256", "states", "ports")}
    cell_orders = {}
    for key in ("states", "ports"):
        metadata[key] = []
        for row in plan[key]:
            item = {k: v for k, v in row.items() if k not in ("initial", "monitor_initial")}
            if plan.get('plan_digest_version',1)==2:
                order=row['cell_ids'];cached=cell_orders.get(id(order))
                if cached is None or cached[0] is not order:
                    cached=(order,_hash_metadata(order));cell_orders[id(order)]=cached
                item.pop('cell_ids')
                item['cell_ids_sha256']=cached[1]
            item["initial_sha256"] = _array_digest(row["initial"], np)
            if "monitor_initial" in row:
                item["monitor_initial_sha256"] = _array_digest(row["monitor_initial"], np)
            metadata[key].append(item)
    return _hash_metadata(metadata)


def seal_plan(plan):
    """Call before either arm runs. Own immutable initial arrays, then bind them.

    This is runtime QA preparation, prohibited during the user's gaming hold.
    It creates no graph and does not authorize subsequent simulation.
    """
    import numpy as np

    result = deepcopy({k: v for k, v in plan.items() if k not in ("states", "ports", "plan_sha256")})
    result['plan_digest_version']=2
    for key in ("states", "ports"):
        result[key] = []
        for row in plan[key]:
            item = deepcopy({k: v for k, v in row.items() if k not in ('initial', 'monitor_initial')})
            for name in ('initial', 'monitor_initial'):
                if name not in row:
                    continue
                _array_digest(row[name], np)
                initial = row[name].copy()
                initial.flags.writeable = False
                item[name] = initial
            result[key].append(item)
    result["plan_sha256"] = plan_digest(result)
    return result


@dataclass(frozen=True)
class GateIssue:
    status: str
    gate: str
    detail: str


@dataclass(frozen=True)
class QAResult:
    status: str
    issues: tuple
    compared: dict
    observed_spike_cells: tuple
    missing_spike_cells: tuple
    contract_hash: str
    plan_hash: str
    numerical_only: bool = True
    fit_eligible: bool = False
    performance_validated: bool = False


class QARecorder:
    """Bounded streaming witness transport; does not accumulate observations.

    The producer owns each array until sink returns. Arrays must be detached CPU
    witnesses of the original operations. Capacity rejection occurs before the
    sink is called. A spool sink must additionally enforce its own TOTAL disk
    quota and propagate errors rather than silently dropping records.
    """
    def __init__(self, sink, *, max_record_bytes=256*1024**2):
        if not callable(sink) or type(max_record_bytes) is not int or max_record_bytes <= 0:
            raise QAIncomplete("A synchronous sink and positive record cap are required")
        self.sink = sink
        self.max_record_bytes = max_record_bytes
        self.records = 0

    def emit(self, kind, witness):
        import numpy as np

        def size(value):
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise QAIncomplete("Object arrays are not QA numeric payloads")
                return value.nbytes
            if isinstance(value, Mapping):
                return sum(len(str(k).encode("utf-8")) + size(v) for k, v in value.items())
            if type(value) in (tuple, list):
                return sum(size(v) for v in value)
            return len(_json(value))
        if size(witness) > self.max_record_bytes:
            raise QAIncomplete("QA record exceeds the prospectively selected storage cap")
        self.sink({"kind": str(kind), "witness": witness})
        self.records += 1


class ResidentQAComparator:
    """Streaming, full-coverage comparison under the unchanged frozen budget."""
    def __init__(self, plan, *, evidence_sink=None, max_working_bytes=64*1024**2,
                 max_issues=256):
        import numpy as np

        self.np = np
        self.plan = plan
        self.sink = evidence_sink
        self.issues = []
        self.max_issues = max_issues
        self.issue_overflow = False
        self.worst_status = "PASS"
        self.closed = False
        if type(max_working_bytes) is not int or max_working_bytes < 4096:
            raise QAIncomplete("A positive bounded CPU comparison workspace is required")
        # At most 24 simultaneous FP64 scratch vectors per chunk, conservatively.
        self.chunk = min(65536, max_working_bytes // (24*8))
        self.seen = {key: [] for key in ("points", "monitors", "draws", "intervals", "checkpoint_ids", "failure_cases", "ownership_ids")}
        self.spike_cells = set()
        self.contract = json.loads((Path(__file__).resolve().parent.parent /
            "configurations/numerical_equivalence/resident_native_fp64_v1.json").read_bytes())
        raw = (Path(__file__).resolve().parent.parent /
            "configurations/numerical_equivalence/resident_native_fp64_v1.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != CONTRACT_SHA256 or plan.get("contract_hash") != CONTRACT_SHA256:
            raise QAIncomplete("Frozen resident numerical contract identity differs")
        if not _sha(plan.get("plan_sha256")) or plan_digest(plan) != plan["plan_sha256"]:
            raise QAIncomplete("The complete QA plan was not sealed before execution")
        self.states = self._registry("states")
        self.ports = self._registry("ports")
        self._validate_plan()
        if not callable(evidence_sink):
            self._issue("INCONCLUSIVE", "evidence", "No sink preserves complete monitor reports/per-cell differences")

    def _issue(self, status, gate, detail):
        rank = {"PASS": 0, "INCONCLUSIVE": 1, "FAIL": 2}
        if rank[status] > rank[self.worst_status]:
            self.worst_status = status
        if len(self.issues) < self.max_issues:
            self.issues.append(GateIssue(status, str(gate), str(detail)))
        else:
            self.issue_overflow = True

    def _emit(self, kind, payload):
        if self.sink is None:
            return
        try:
            self.sink({"kind": kind, "payload": payload})
        except Exception as exc:
            self._issue("INCONCLUSIVE", "evidence", "Evidence sink failed: " + type(exc).__name__)
            self.sink = None

    def _registry(self, key):
        rows = {}
        for row in self.plan[key]:
            name = row["name"]
            if name in rows or not name or not row["meaning"]:
                raise QAIncomplete("Duplicate/invalid typed observable")
            initial = row["initial"]
            _array_digest(initial, self.np)
            if initial.flags.writeable or len(row["cell_ids"]) != len(initial):
                raise QAIncomplete("Initial vectors must be owned immutable arrays with complete cell ordering")
            if key == 'states':
                _array_digest(row.get('monitor_initial'), self.np)
                if row['monitor_initial'].flags.writeable or row['monitor_initial'].shape != initial.shape:
                    raise QAIncomplete('Complete immutable original monitor reference is required')
            if row["unit"] not in self.plan["unit_scales"]:
                from .resident_derived_ports import ADDENDUM_SHA256, BUDGET_KIND, validate_descriptor, load_addendum
                load_addendum()
                if (key!='ports' or row.get('budget_kind')!=BUDGET_KIND or
                        self.plan.get('derived_port_addendum_sha256')!=ADDENDUM_SHA256):
                    raise QAIncomplete("Unregistered typed observable: " + name)
                descriptor=row['derived_descriptor'];validate_descriptor(descriptor)
                if (name!=descriptor['component']+'.event_rate' or row['unit']!=descriptor['unit'] or
                        row['meaning']!=descriptor['meaning'] or list(row['cell_ids'])!=descriptor['cell_ids']):
                    raise QAIncomplete('Derived observable identity differs: '+name)
            if key == "states" and row.get("period") is not None:
                if not math.isfinite(row["period"]) or row["period"] <= 0:
                    raise QAIncomplete("Positive original discrete period required")
            rows[name] = row
        if not rows:
            raise QAIncomplete("Empty full-coverage observable registry")
        return rows

    def _validate_plan(self):
        p = self.plan
        binding = p["binding"]
        if not REQUIRED_BINDINGS.issubset(binding) or binding["contract_hash"] != CONTRACT_SHA256:
            raise QAIncomplete("Incomplete prospective source/graph/input/reference binding")
        for key in REQUIRED_BINDINGS - {"channel_data_hashes"}:
            if not _sha(binding[key]):
                raise QAIncomplete("Invalid binding digest: " + key)
        if not binding["channel_data_hashes"] or any(not _sha(h) for h in binding["channel_data_hashes"].values()):
            raise QAIncomplete("Actual prepared input hashes are mandatory")
        if set(p["arm_identity_hashes"]) != {"reference", "candidate"} or any(
                not _sha(h) for h in p["arm_identity_hashes"].values()):
            raise QAIncomplete("Both implementation identities must be prospectively fixed")
        if sum(len(row["initial"]) for row in self.states.values()) != self.contract["scenario_required"]["state_scalars"]:
            raise QAIncomplete("State registry is not the frozen complete real graph")
        if len(p["rgc_cells"]) != 8 or len({tuple(c) for c in p["rgc_cells"]}) != 8:
            raise QAIncomplete("Current eight-cell RGC output cohort is required")
        if not math.isfinite(p["monitor_relative_tolerance"]) or p["monitor_relative_tolerance"] < 0:
            raise QAIncomplete("Original background relative tolerance required")
        for entry in p["unit_scales"].values():
            if any(not math.isfinite(entry[k]) or entry[k] <= 0 for k in ("atol", "scale_floor")):
                raise QAIncomplete("Invalid registered numerical unit scale")
        scenario = p["scenario"]
        start, stop = scenario["start"], scenario["stop"]
        if not (.008 <= stop-start <= .010):
            raise QAIncomplete("Frozen QA scenario requires 8-10 ms")
        if not start < scenario["background_stop"] < stop or not any(
                start < t < stop for t in scenario["frame_boundaries"]):
            raise QAIncomplete("Background and actual frame-boundary coverage required")
        if scenario["rod_period"] != .0001 or scenario["rgc_period"] != .001 or scenario["fit_eligible"] is not False:
            raise QAIncomplete("Original clocks and diagnostic non-qualification required")
        for category in self.seen:
            rows = p.get(category)
            if not rows:
                raise QAIncomplete("Missing prospective witness schedule: " + category)
            ids = [r if isinstance(r, str) else r["id"] for r in rows]
            if len(set(ids)) != len(ids):
                raise QAIncomplete("Duplicate prospective witness identifier: " + category)
        if not any(row["dense_times"] for row in p["intervals"]):
            raise QAIncomplete("At least one actual dense request required")
        for row in p["draws"]:
            if (tuple((row["component"], row["cell_id"])) not in {tuple(c) for c in p["rgc_cells"]}
                    or row["voltage_state"] not in self.states or row["alpha"] <= 0 or row["beta"] <= 0
                    or row["dt"] != .001):
                raise QAIncomplete("Draw schedule/voltage parameter binding differs")

    def _expected(self, category, identifier):
        if self.closed:
            raise QAIncomplete("Comparator is finalized")
        index = len(self.seen[category])
        rows = self.plan[category]
        if index >= len(rows):
            self._issue("FAIL", category, "Unexpected additional witness: " + str(identifier))
            return None
        expected = rows[index]
        key = expected if isinstance(expected, str) else expected["id"]
        if identifier != key:
            self._issue("FAIL", category, "Witness traversal/order differs: expected " + str(key))
            return None
        self.seen[category].append(identifier)
        return expected

    def _exact(self, reference, candidate, gate):
        try:
            if _json(reference) != _json(candidate):
                self._issue("FAIL", gate, "Exact identity/decision differs")
                return False
        except (TypeError, ValueError) as exc:
            self._issue("INCONCLUSIVE", gate, str(exc))
            return False
        return True

    def _vector(self, value, size, gate):
        np = self.np
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float64") or value.shape != (size,):
            self._issue("INCONCLUSIVE", gate, "Complete CPU float64 vector missing or wrong shape")
            return None
        return value

    def _budget(self, spec, section):
        np = self.np
        entry = self.plan["unit_scales"][spec["unit"]]
        scale = np.maximum(np.abs(spec["initial"][section]), entry["scale_floor"])
        solver = self.contract["solver_required"]
        budget = self.contract["state_budget_fraction"] * np.minimum(
            solver["atol"] + solver["rtol"]*scale,
            entry["atol"] + self.plan["monitor_relative_tolerance"]*scale)
        return scale, budget

    def _numeric(self, reference, candidate, spec, gate, *, kind="state", width=None,
                 den=None, preserve=False, exact=False):
        np = self.np
        n = len(spec["initial"])
        ref = self._vector(reference, n, gate)
        cand = self._vector(candidate, n, gate)
        if ref is None or cand is None:
            return
        if den is not None and self._vector(den, n, gate + ".den") is None:
            return
        eps = self.contract["roundoff_epsilon_multiplier"] * sys.float_info.epsilon
        h = self.contract["solver_required"]["max_step"]
        for first in range(0, n, self.chunk):
            sl = slice(first, first+self.chunk)
            a, b = ref[sl], cand[sl]
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                self._issue("FAIL", gate, "Nonfinite numeric witness; no clipping")
                return
            scale, budget = self._budget(spec, sl)
            if exact:
                allowed = np.zeros_like(budget)
            elif kind == "rhs":
                allowed = np.minimum(np.minimum(eps*scale/h, budget/h) +
                    eps*np.maximum(np.abs(a), np.abs(b)), budget/h)
            elif kind == "rhs_max":
                allowed = np.minimum(np.minimum(eps*scale/h, budget/h) +
                    eps*np.maximum(np.abs(a), np.abs(b)), budget/h)
            elif kind == "defect":
                period = spec.get("period")
                allowed = 2*budget/period if period is not None else np.zeros_like(budget)
            elif kind == "excursion_norm":
                allowed = 2*budget/den[sl]
            elif kind == "rhs_norm":
                allowed = width*(budget/h)/den[sl]
            elif kind == "defect_norm":
                period = spec.get("period")
                allowed = width*(2*budget/period)/den[sl] if period is not None else np.zeros_like(budget)
            elif kind == "state":
                allowed = budget
            else:
                self._issue("INCONCLUSIVE", gate, "Unsupported typed propagation")
                return
            delta = np.abs(b-a)
            bad = np.flatnonzero(delta > allowed)
            if len(bad):
                i = int(bad[0])
                self._issue("FAIL", gate, "cell_index=%d difference=%.17g allowance=%.17g" %
                            (first+i, delta[i], allowed[i]))
            if preserve:
                self._emit("per_cell", {"gate": gate, "first": first,
                    "reference": a, "candidate": b, "difference": b-a, "allowance": allowed})

    def compare_derived_port(self, reference, candidate, spec, gate, dependency_states, *, exact=False):
        from .resident_derived_ports import compare_vectors
        try:
            a,b=dependency_states
            descriptor=spec['derived_descriptor']
            v,x=descriptor['voltage_state'],descriptor['resource_state']
            _,bv=self._budget(self.states[v],slice(None))
            _,bx=self._budget(self.states[x],slice(None))
            result=compare_vectors(descriptor,reference,candidate,a[v],b[v],a[x],b[x],bv,bx,exact=exact)
            self._emit('derived_port',dict(gate=gate,reference=reference,candidate=candidate,**result))
            if not result['passed']:
                self._issue('FAIL',gate,'Ribbon port or exact V/x prerequisite exceeds derived bound')
        except (KeyError,TypeError,ValueError) as exc:
            self._issue('INCONCLUSIVE',gate,'Incomplete derived port witness: '+str(exc))

    def _mapping_numeric(self, reference, candidate, registry, gate, dependency_states=None, **kwargs):
        if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
            self._issue("INCONCLUSIVE", gate, "Numeric mapping missing")
            return
        if set(reference) != set(registry) or set(candidate) != set(registry):
            self._issue("INCONCLUSIVE", gate, "Full observable coverage differs")
            return
        for name, spec in registry.items():
            if spec.get('budget_kind'):
                self.compare_derived_port(reference[name],candidate[name],spec,gate+'.'+name,
                                          dependency_states,exact=kwargs.get('exact',False))
            else:
                self._numeric(reference[name], candidate[name], spec, gate + "." + name, **kwargs)

    def _point(self, reference, candidate, expected, *, exact_values=False):
        gate = "point." + str(expected["id"])
        for field in ("id", "time_seconds", "phase", "context_hash"):
            self._exact(reference.get(field), expected[field], gate+".reference."+field)
            self._exact(candidate.get(field), expected[field], gate+".candidate."+field)
        self._mapping_numeric(reference.get("state"), candidate.get("state"), self.states, gate+".state", exact=exact_values)
        self._mapping_numeric(reference.get("rhs"), candidate.get("rhs"), self.states, gate+".rhs", kind="rhs", exact=exact_values)
        self._mapping_numeric(reference.get("ports"), candidate.get("ports"), self.ports, gate+".ports",
                              dependency_states=(reference.get('state'),candidate.get('state')),exact=exact_values)
        transitions = {name: spec for name, spec in self.states.items() if spec.get("period") is not None}
        self._mapping_numeric(reference.get("transitions"), candidate.get("transitions"), transitions,
                              gate+".transition", exact=exact_values)
        for field, keys in (("aux", EXACT_AUX_FIELDS), ("checks", EXACT_CHECK_FIELDS)):
            a, b = reference.get(field), candidate.get(field)
            if not isinstance(a, Mapping) or not isinstance(b, Mapping) or set(a) != keys or set(b) != keys:
                self._issue("INCONCLUSIVE", gate+"."+field, "Missing/unsupported exact witness fields")
            else:
                self._exact(a, b, gate+"."+field)
        a, b = reference.get("delays"), candidate.get("delays")
        if not isinstance(a, Mapping) or not isinstance(b, Mapping) or set(a) != set(b):
            self._issue("INCONCLUSIVE", gate+".delays", "Full queue payload coverage missing")
            return
        for queue in a:
            if a[queue].get("port") != b[queue].get("port") or a[queue].get("port") not in self.ports:
                self._issue("INCONCLUSIVE", gate+".delays", "Queue port unit/identity missing")
                continue
            spec = self.ports[a[queue]["port"]]
            x, y = a[queue].get("values"), b[queue].get("values")
            if (not isinstance(x, self.np.ndarray) or not isinstance(y, self.np.ndarray)
                    or x.ndim != 2 or y.shape != x.shape or x.shape[0] < 1 or x.shape[1] != len(spec["initial"])):
                self._issue("INCONCLUSIVE", gate+".delays", "Queue must include prehistory and every retained full row")
                continue
            for i in range(x.shape[0]):
                self._numeric(x[i], y[i], spec, gate+".delay."+queue+"."+str(i), exact=exact_values)

    def compare_point(self, reference, candidate):
        """Compare one FULL original callback, including direct pure transitions."""
        expected = self._expected("points", reference.get("id"))
        if expected is not None:
            self._point(reference, candidate, expected)

    def compare_solver(self, reference, candidate):
        """Interval witness: id,start,stop,dense_times,initial_rhs_calls,
        full_state_dimension,attempts,accepted_steps,rejected_attempts,nfev,
        dense_polynomials,endpoint_choice,controller_restart,coefficient_hash,
        expression_hash. attempts is the ORDERED full sequence of
        {t,h,t_new,accepted}; totals alone cannot prove the schedule.
        """
        expected = self._expected("intervals", reference.get("id"))
        if expected is None:
            return
        required = {"id", "start", "stop", "dense_times", "initial_rhs_calls", "rhs_inputs", "full_state_dimension",
                    "attempts", "accepted_steps", "rejected_attempts", "nfev", "dense_polynomials",
                    "endpoint_choice", "controller_restart", "coefficient_hash", "expression_hash"}
        if set(reference) != required or set(candidate) != required:
            self._issue("INCONCLUSIVE", "solver", "Full per-attempt schedule/arithmetic evidence missing")
            return
        self._exact(reference, candidate, "solver.schedule")
        for key in ("id", "start", "stop", "dense_times"):
            self._exact(reference[key], expected[key], "solver."+key)
        if reference["initial_rhs_calls"] != 2 or reference["full_state_dimension"] != 5096368 or reference["controller_restart"] is not True:
            self._issue("FAIL", "solver", "Original restart/RHS count/full norm dimension changed")
        attempts = reference["attempts"]
        if not attempts or any(set(a) != {"t", "h", "t_new", "accepted"} for a in attempts):
            self._issue("INCONCLUSIVE", "solver", "Per-attempt acceptance evidence missing")
            return
        accepted = sum(a["accepted"] is True for a in attempts)
        if any(type(a["accepted"]) is not bool for a in attempts) or accepted != reference["accepted_steps"] or len(attempts)-accepted != reference["rejected_attempts"]:
            self._issue("FAIL", "solver", "Attempt totals disagree")
        if reference["nfev"] != 2 + 12*len(attempts) + 3*reference["dense_polynomials"]:
            self._issue("FAIL", "solver", "DOP853 original evaluation schedule differs")
        if (len(reference['rhs_inputs']) != reference['nfev'] or any(
                set(row) != {'time_seconds', 'input_hash', 'delay_left_limit'} or
                not _sha(row['input_hash']) or not math.isfinite(row['time_seconds']) or
                type(row['delay_left_limit']) is not bool for row in reference['rhs_inputs'])):
            self._issue('INCONCLUSIVE', 'solver.inputs', 'Actual per-RHS input content/time/left-limit evidence missing')
        endpoint = "dense_sample" if expected["dense_times"] else "accepted_endpoint"
        if reference["endpoint_choice"] != endpoint:
            self._issue("FAIL", "solver", "Dense endpoint semantics differ")
        for key in ("coefficient_hash", "expression_hash"):
            if not _sha(reference[key]):
                self._issue("INCONCLUSIVE", "solver", "Original arithmetic identity missing")

    def compare_draw(self, reference, candidate):
        """One ORIGINAL draw, no replay: plan draw fields plus u,rate_Hz,
        probability,event_decision,event_time_seconds (None without an event),
        history_hash,rng_before_hash,rng_after_hash,draw_index. Uniform equality,
        Bernoulli uncertainty margins and event-time intervals are mandatory.
        """
        expected = self._expected("draws", reference.get("id"))
        if expected is None:
            return
        gate = "draw."+expected["id"]
        required = set(expected) | {"u", "rate_Hz", "probability", "event_decision", "event_time_seconds",
                                    "history_hash", "rng_before_hash", "rng_after_hash", "draw_index"}
        if set(reference) != required or set(candidate) != required:
            self._issue("INCONCLUSIVE", gate, "Original RNG/rate/probability/event witness incomplete")
            return
        for key in expected:
            self._exact(reference[key], expected[key], gate+".reference."+key)
            self._exact(candidate[key], expected[key], gate+".candidate."+key)
        for key in ("u", "history_hash", "rng_before_hash", "rng_after_hash", "draw_index", "event_decision"):
            self._exact(reference[key], candidate[key], gate+"."+key)
        if any(not _sha(reference[k]) for k in ("history_hash", "rng_before_hash", "rng_after_hash")):
            self._issue("INCONCLUSIVE", gate, "Strict-past/RNG identities are missing")
            return
        numeric = [arm[k] for arm in (reference, candidate) for k in ("u", "rate_Hz", "probability")]
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in numeric):
            self._issue("FAIL", gate, "Nonfinite/invalid probability witness")
            return
        u, rate, other = reference["u"], reference["rate_Hz"], candidate["rate_Hz"]
        if not 0 <= u < 1 or min(rate, other) < 0 or any(not 0 <= a["probability"] <= 1 for a in (reference, candidate)):
            self._issue("FAIL", gate, "Uniform/rate/probability outside original domain")
            return
        spec = self.states[expected["voltage_state"]]
        index = expected["cell_index"]
        if type(index) is not int or not 0 <= index < len(spec["initial"]) or spec["unit"] != "V" or spec["cell_ids"][index] != expected["cell_id"]:
            self._issue("INCONCLUSIVE", gate, "Typed RGC voltage identity differs")
            return
        _, budget = self._budget(spec, slice(index, index+1))
        eps = self.contract["roundoff_epsilon_multiplier"]*sys.float_info.epsilon
        rate_allowance = expected["alpha"]/expected["beta"]*float(budget[0]) + eps*max(abs(rate), 1.)
        probability_allowance = expected["dt"]*expected["alpha"]/expected["beta"]*float(budget[0]) + eps
        if abs(other-rate) > rate_allowance or abs(candidate["probability"]-reference["probability"]) > probability_allowance:
            self._issue("FAIL", gate, "Rate/probability exceeds frozen propagated allowance")
        margin = abs(u-reference["probability"])
        if margin <= probability_allowance:
            self._issue("INCONCLUSIVE", gate, "Uniform lies inside the probability uncertainty interval")
        for arm in (reference, candidate):
            if type(arm["event_decision"]) is not bool or arm["event_decision"] != (arm["u"] < arm["probability"]):
                self._issue("FAIL", gate, "Original Bernoulli decision differs")
            # NumPy is already explicitly active on CPU. Reuse original formulas.
            computed = float(-self.np.expm1(-arm["rate_Hz"]*expected["dt"]))
            if arm["probability"] != computed:
                self._issue("INCONCLUSIVE", gate, "Original probability expression witness cannot be reproduced exactly")
        if reference["event_decision"] is True:
            self.spike_cells.add((expected["component"], expected["cell_id"]))
            lower, upper = rate-rate_allowance, rate+rate_allowance
            if lower <= 0:
                self._issue("INCONCLUSIVE", gate, "Actual event has nonpositive lower allowed rate")
            else:
                offset = float(-self.np.log1p(-u))
                earliest = expected["bin_start_seconds"]+offset/upper
                latest = expected["bin_start_seconds"]+offset/lower
                for arm in (reference, candidate):
                    time = arm["event_time_seconds"]
                    if type(time) not in (int, float) or not math.isfinite(time) or not earliest <= time <= latest:
                        self._issue("FAIL", gate, "Event time outside the declared original-formula interval")
                    elif not expected["bin_start_seconds"] <= time <= expected["bin_start_seconds"]+expected["dt"]:
                        self._issue("FAIL", gate, "Event moved out of its original bin")
        elif reference["event_time_seconds"] is not None or candidate["event_time_seconds"] is not None:
            self._issue("FAIL", gate, "Non-event has an event timestamp")
        self._emit("draw", {"reference": reference, "candidate": candidate,
            "rate_allowance": rate_allowance, "probability_allowance": probability_allowance, "uniform_margin": margin})

    def compare_monitor(self, reference, candidate):
        """Complete raw reports plus full per-cell lo/hi/rhs/defect and den.

        Every monitored point must already have a direct transition/state
        comparison. Derived budgets are only the explicitly frozen formulas.
        Original per-cell normalized criteria are formed BEFORE any maximum.
        """
        expected = self._expected("monitors", reference.get("id"))
        if expected is None:
            return
        gate = "monitor."+expected["id"]
        for field in ("id", "width", "point_ids"):
            self._exact(reference.get(field), expected[field], gate+".reference."+field)
            self._exact(candidate.get(field), expected[field], gate+".candidate."+field)
        if expected["width"] <= 0 or any(i not in self.seen["points"] for i in expected["point_ids"]):
            self._issue("INCONCLUSIVE", gate, "Positive window and every original callback witness required")
            return
        if len(reference.get("rows", ())) != len(self.states) or len(candidate.get("rows", ())) != len(self.states):
            self._issue("INCONCLUSIVE", gate, "Complete per-cell monitor rows missing")
            return
        raw_a, raw_b = reference.get("raw_report"), candidate.get("raw_report")
        required = {"states", "maximum_norm", "dominant", "observations", "domain_failures", "passed"}
        if not isinstance(raw_a, Mapping) or not isinstance(raw_b, Mapping) or set(raw_a) != required or set(raw_b) != required:
            self._issue("INCONCLUSIVE", gate, "Original complete report schema missing or unsupported")
            return
        self._emit("raw_monitor_reports", {"id": expected["id"], "reference": raw_a, "candidate": raw_b})
        for field in ("observations", "domain_failures", "passed"):
            self._exact(raw_a[field], raw_b[field], gate+"."+field)
        if raw_a["observations"] != len(expected["point_ids"]):
            self._issue("FAIL", gate, "Observation count differs from complete bound callback window")
        computed = [{"states": [], "maximum_norm": 0., "dominant": None} for _ in range(2)]
        winner_bounds = [0., 0.]
        max_bound = 0.
        labels = ("excursion", "original_rhs", "discrete_one_step_defect")
        report_fields = ("maximum_observed_excursion_norm", "original_rhs_norm", "discrete_defect_norm")
        for (name, spec), a, b in zip(self.states.items(), reference["rows"], candidate["rows"]):
            if a.get("state") != name or b.get("state") != name or set(a) != {"state", "den", "lo", "hi", "rhs", "defect"} or set(b) != set(a):
                self._issue("INCONCLUSIVE", gate, "Monitor state traversal/schema differs")
                return
            n = len(spec["initial"])
            if any(self._vector(row[key], n, gate+"."+name+"."+key) is None for row in (a, b) for key in ("den", "lo", "hi", "rhs", "defect")):
                return
            maxima = [[0., 0., 0.], [0., 0., 0.]]
            argmax = [[0, 0, 0], [0, 0, 0]]
            arg_bounds = [[0., 0., 0.], [0., 0., 0.]]
            for first in range(0, n, self.chunk):
                sl = slice(first, first+self.chunk)
                scale, budget = self._budget(spec, sl)
                if 'monitor_initial' not in spec:
                    self._issue("INCONCLUSIVE", gate, "Frozen original monitor reference values are missing")
                    return
                monitor_scale = self.np.maximum(self.np.abs(spec['monitor_initial'][sl]),
                    self.plan['unit_scales'][spec['unit']]['scale_floor'])
                den = self.plan["unit_scales"][spec["unit"]]["atol"] + self.plan["monitor_relative_tolerance"]*monitor_scale
                if not self.np.array_equal(a["den"][sl], den) or not self.np.array_equal(b["den"][sl], den):
                    self._issue("FAIL", gate+"."+name, "Immutable original per-cell denominator changed")
                    return
                period = spec.get("period")
                h = self.contract['solver_required']['max_step']
                eps = self.contract['roundoff_epsilon_multiplier'] * sys.float_info.epsilon
                rhs_allowance = self.np.minimum(self.np.minimum(eps*scale/h, budget/h) +
                    eps*self.np.maximum(self.np.abs(a['rhs'][sl]), self.np.abs(b['rhs'][sl])), budget/h)
                bounds = (2*budget/den,
                          expected["width"]*rhs_allowance/den,
                          expected["width"]*2*budget/period/den if period else self.np.zeros_like(budget))
                values = []
                for row in (a, b):
                    if any(not self.np.isfinite(row[key][sl]).all() for key in ("lo", "hi", "rhs", "defect")):
                        self._issue("FAIL", gate, "Nonfinite monitor statistic")
                        return
                    if self.np.any(row["hi"][sl] < row["lo"][sl]) or self.np.any(row["rhs"][sl] < 0) or self.np.any(row["defect"][sl] < 0):
                        self._issue("FAIL", gate, "Invalid per-cell extrema/nonnegative statistic")
                        return
                    values.append(((row["hi"][sl]-row["lo"][sl])/den,
                                   expected["width"]*row["rhs"][sl]/den,
                                   expected["width"]*row["defect"][sl]/den))
                for k, bound in enumerate(bounds):
                    max_bound = max(max_bound, float(self.np.max(bound)) if len(bound) else 0.)
                    if self.np.any(self.np.abs(values[1][k]-values[0][k]) > bound):
                        self._issue("FAIL", gate+"."+name+"."+labels[k], "Per-cell normalized statistic exceeds proved propagation")
                    self._emit("monitor_norm_cells", {"monitor": expected["id"], "state": name,
                        "criterion": labels[k], "first": first, "reference": values[0][k],
                        "candidate": values[1][k], "difference": values[1][k]-values[0][k], "allowance": bound})
                    for arm in range(2):
                        index = int(self.np.argmax(values[arm][k])) if len(bound) else 0
                        value = float(values[arm][k][index]) if len(bound) else 0.
                        if value > maxima[arm][k]:
                            maxima[arm][k], argmax[arm][k], arg_bounds[arm][k] = value, first+index, float(bound[index])
            for key, kind in (("lo", "state"), ("hi", "state"), ("rhs", "rhs_max"), ("defect", "defect")):
                self._numeric(a[key], b[key], spec, gate+"."+name+"."+key, kind=kind, preserve=True)
            for arm in range(2):
                computed[arm]["states"].append(dict(state=name, unit=spec["unit"],
                    **dict(zip(report_fields, maxima[arm]))))
                for k, value in enumerate(maxima[arm]):
                    if value > computed[arm]["maximum_norm"]:
                        computed[arm]["maximum_norm"] = value
                        computed[arm]["dominant"] = {"state": name, "component": name.split(".", 1)[0],
                            "cell_id": spec["cell_ids"][argmax[arm][k]], "criterion": labels[k], "norm": value}
                        winner_bounds[arm] = arg_bounds[arm][k]
        for own, raw in zip(computed, (raw_a, raw_b)):
            for key in ("states", "maximum_norm", "dominant"):
                if _json(own[key]) != _json(raw[key]):
                    self._issue("INCONCLUSIVE", gate, "Original report/first-argmax cannot be reproduced from per-cell witness: "+key)
            if raw["passed"] != (own["maximum_norm"] <= 1 and not raw["domain_failures"]):
                self._issue("FAIL", gate, "Original report decision differs from complete original criteria")
        value = computed[0]["maximum_norm"]
        if not raw_a["domain_failures"] and value-max_bound <= 1 < value+max_bound:
            self._issue("INCONCLUSIVE", gate, "Declared bounds cannot guarantee the original threshold decision")
        # Maxima of bounded vectors are bounded by the maximum per-cell bound.
        if abs(computed[1]["maximum_norm"]-value) > max_bound:
            self._issue("FAIL", gate, "Dominant maximum exceeds the proved per-cell bound")
        self._emit("dominant_witnesses", {"id": expected["id"], "reference": computed[0]["dominant"],
            "candidate": computed[1]["dominant"], "winner_allowances": winner_bounds, "maximum_allowance": max_bound})

    def compare_checkpoint(self, reference, candidate):
        """Runtime roundtrip/continuation only; no qualified-background claim.

        Each arm: id,runtime_identity_hash,restored_identity_hash,payload_hash,
        verified_payload_hash,layout_hash,payload_fields,fit_eligible,
        supplied_state_hash_before,supplied_state_hash_after,export,restored,
        uninterrupted,continued,uninterrupted_schedule,continued_schedule.
        The four state witnesses use the complete point schema. Export/restored
        is EXACT within an arm. Continuation uses frozen numeric budgets and
        exact events/clocks/schedule; then both arms' continuations are compared.
        """
        identifier = reference.get("id")
        if self._expected("checkpoint_ids", identifier) is None:
            return
        gate = "checkpoint."+identifier
        required = {"id", "runtime_identity_hash", "restored_identity_hash", "payload_hash", "verified_payload_hash", "restored_payload_hash",
                    "layout_hash", "payload_fields", "fit_eligible", "supplied_state_hash_before", "supplied_state_hash_after",
                    "export", "restored", "uninterrupted", "continued", "uninterrupted_schedule", "continued_schedule"}
        if set(reference) != required or set(candidate) != required:
            self._issue("INCONCLUSIVE", gate, "Complete runtime export/restore/continuation evidence missing")
            return
        self._exact(identifier, candidate["id"], gate+".id")
        self._exact(reference["layout_hash"], candidate["layout_hash"], gate+".layout")
        for arm_name, arm in (("reference", reference), ("candidate", candidate)):
            if arm["runtime_identity_hash"] != self.plan["arm_identity_hashes"][arm_name] or arm["restored_identity_hash"] != arm["runtime_identity_hash"]:
                self._issue("FAIL", gate, "Runtime backend/restore identity differs")
            if any(not _sha(arm[k]) for k in ("payload_hash", "verified_payload_hash", "layout_hash", "supplied_state_hash_before", "supplied_state_hash_after")):
                self._issue("INCONCLUSIVE", gate, "Checkpoint integrity/ownership digest missing")
            if (arm["payload_hash"] != arm["verified_payload_hash"] or
                    arm['payload_hash'] != arm['restored_payload_hash'] or
                    arm["supplied_state_hash_before"] != arm["supplied_state_hash_after"]):
                self._issue("FAIL", gate, "Checkpoint integrity failed or functional input was mutated")
            if set(arm["payload_fields"]) != RUNTIME_PAYLOAD_FIELDS or arm["fit_eligible"] is not False:
                self._issue("FAIL", gate, "Incomplete runtime payload or unsupported background qualification")
            self._exact(arm["uninterrupted_schedule"], arm["continued_schedule"], gate+".continuation_schedule")
            loader=getattr(self,'point_loader',nullcontext)
            with loader(arm['export']) as exported, loader(arm['restored']) as restored:
                start = {k: exported[k] for k in ("id", "time_seconds", "phase", "context_hash")}
                self._point(exported, restored, start, exact_values=True)
            with loader(arm['uninterrupted']) as uninterrupted, loader(arm['continued']) as continued:
                end = {k: uninterrupted[k] for k in ("id", "time_seconds", "phase", "context_hash")}
                self._point(uninterrupted, continued, end)
        with loader(reference['continued']) as a, loader(candidate['continued']) as b:
            end = {k: a[k] for k in ("id", "time_seconds", "phase", "context_hash")}
            self._point(a, b, end)

    def compare_failure(self, reference, candidate):
        """Each witness: id,label,barrier,time_seconds,stage,exception_type,
        checks_before,post_failure_updates. Original barrier/stage/time must
        match; post_failure_updates MUST be zero (no deferred rejection).
        """
        expected = self._expected("failure_cases", reference.get("id"))
        if expected is None:
            return
        required = {"id", "label", "barrier", "time_seconds", "stage", "exception_type", "checks_before", "post_failure_updates"}
        if set(reference) != required or set(candidate) != required:
            self._issue("INCONCLUSIVE", "failure_timing", "Complete original rejection barrier evidence missing")
            return
        self._exact(reference, candidate, "failure_timing")
        for key in ("id", "label", "barrier"):
            self._exact(reference[key], expected[key], "failure_timing."+key)
        if reference["post_failure_updates"] != 0 or candidate["post_failure_updates"] != 0:
            self._issue("FAIL", "failure_timing", "Rejection was delayed past its original failure barrier")

    def compare_ownership(self, reference, candidate):
        """Each witness: id,snapshot_hash_at_return,snapshot_hash_after_reuse,
        supplied_state_hash_before,supplied_state_hash_after. Same-arm equality
        proves sampled/final snapshots survived RK reuse and functional advance.
        """
        if self._expected("ownership_ids", reference.get("id")) is None:
            return
        required = {"id", "snapshot_hash_at_return", "snapshot_hash_after_reuse", "supplied_state_hash_before", "supplied_state_hash_after"}
        if set(reference) != required or set(candidate) != required:
            self._issue("INCONCLUSIVE", "ownership", "Snapshot/workspace alias witness missing")
            return
        self._exact(reference["id"], candidate["id"], "ownership.id")
        for arm in (reference, candidate):
            if any(not _sha(arm[k]) for k in required-{"id"}):
                self._issue("INCONCLUSIVE", "ownership", "Missing ownership digests")
            if arm["snapshot_hash_at_return"] != arm["snapshot_hash_after_reuse"] or arm["supplied_state_hash_before"] != arm["supplied_state_hash_after"]:
                self._issue("FAIL", "ownership", "Reusable storage mutated an owned snapshot or functional input")

    def finish(self):
        """Finalize only after all original gates; never translates missing to pass."""
        if self.closed:
            raise QAIncomplete("Comparator already finalized")
        for category, observed in self.seen.items():
            expected = [r if isinstance(r, str) else r["id"] for r in self.plan[category]]
            if observed != expected:
                self._issue("INCONCLUSIVE", category, "Missing aligned witnesses: %d of %d" % (len(observed), len(expected)))
        missing = tuple(tuple(cell) for cell in self.plan["rgc_cells"] if tuple(cell) not in self.spike_cells)
        if missing:
            self._issue("INCONCLUSIVE", "spike_coverage", "No actual event for %d of the current eight output cells; no manufactured spikes or old-fixture credit" % len(missing))
        if plan_digest(self.plan) != self.plan["plan_sha256"]:
            self._issue("FAIL", "identity", "Frozen QA plan/initial scale changed during execution")
        if self.issue_overflow:
            self._emit("issue_overflow", {"retained": len(self.issues), "status": self.worst_status})
        self.closed = True
        return QAResult(self.worst_status, tuple(self.issues), {k: len(v) for k, v in self.seen.items()},
                        tuple(sorted(self.spike_cells)), missing, CONTRACT_SHA256, self.plan["plan_sha256"])
