"""Exact current-profile adapters for parameter-system v1.

Only the requested local/group snapshot is expanded. Existing graph/CSR,
equations, source profiles, qualification and legacy state initialization stay
authoritative. No physiological values are supplied by this migration.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np
from scipy import sparse
from .contracts import ContractError, canonical_hash
from .parameter_system import (
    SCHEMA, RunContext, PopulationIndex, Evidence, Applicability, AggregationRule,
    ParameterRecord, IntrinsicParameterRegistry, ParameterResolver,
    ResolvedParameterBundle, RouteRecord, RouteRegistry, StateBundle, FrozenList, data, identity, fp64)

MECHANISMS = {
    "rods": "Chen2024PrimateRodCurrent+rod_membrane",
    "rod_BC": "ReceptorConductanceMembrane+TRPM1+release_pool",
    "AII_AC": "ReceptorConductanceMembrane+resource+AMPA/glycine+gap",
    "cones": "native_cone", "HC": "native_H1",
}
UNIT = {"C": "F", "gL": "S", "EL": "V", "EE": "V", "EI": "V", "V0": "V",
        "mu": "V", "beta": "V", "alpha": "Hz", "kernel": "dimensionless",
        "spike_history_kernel": "V", "baseline": "dimensionless",
        "tau_a_seconds": "s", "adaptation_enabled": "dimensionless",
        "adaptation_scale": "dimensionless", "native_reference": "vH_native_voltage",
        "current_anchor": "A", "generator_reference": "dimensionless"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clone(value):
    if isinstance(value, np.ndarray): return value.copy()
    if isinstance(value, Mapping): return {k: clone(v) for k, v in value.items()}
    if isinstance(value, FrozenList): return [clone(v) for v in value]
    if isinstance(value, tuple): return tuple(clone(v) for v in value)
    if isinstance(value, list): return [clone(v) for v in value]
    return deepcopy(value)


def _numeric(value):
    try: return np.asarray(value).dtype.kind in "bifu" and np.isfinite(value).all()
    except (TypeError, ValueError): return False


def _mechanism(pop):
    if pop in MECHANISMS: return MECHANISMS[pop]
    if "_BC_" in pop or pop == "S_ON_BC": return "HistoryBipolarTerminal"
    if "RGC" in pop or pop == "small_bistratified_S_ON": return "RGC_spike_history"
    if pop.endswith("_AC_v1"): return "FunctionalAmacrineVoltageGate_v1"
    return "DynamicAmacrineInhibition"


def context_from_seed(assembly, seed, protocol=None):
    protocol = {} if protocol is None else dict(protocol)
    reference = seed.get("input_reference", {})
    background_id = identity(dict(reference=reference, boundary=seed.get("patch_boundary")))
    geometry = getattr(assembly, "ac_config", {}).get("geometry", {})
    scale = geometry.get("mm_per_degree", assembly.manifest.get("configuration", {}).get("mm_per_degree"))
    units = dict(voltage="V", conductance="S", current="A", capacitance="F",
                 cone_input="td", rod_input="R*/s", dtype="float64")
    if scale is not None: units["mm_per_degree"] = scale
    # Unknown anatomical species/eye/optics are retained as unknown, never inferred.
    return RunContext("macaque_effective_source", "unknown",
        dict(mode="cone_native_domain" if "native_inputs" in seed else "source_construction",
             physical_optical_calibration=False),
        dict(policy_id=background_id, native_background_td=reference.get("native_background_td"),
             rod_background_rstar_s=reference.get("rod_background_rstar_s")),
        units, dict(identity=canonical_hash(dict(native_inputs=seed.get("native_inputs"),
            rod_inputs=seed.get("rod_inputs"), input_reference=reference)),
            source_stimulus_identity=seed.get("ac_vs_lineage", {}).get("stimulus_identity"),
            units="explicit_native_td_and_rod_Rstar_s", calibrated=False),
        dict(rod_step_seconds=protocol.get("rod_step_seconds", .0001),
             BC_RGC_period_seconds=protocol.get("BC_RGC_period_seconds", .001),
             native_delay_seconds=protocol.get("native_observation_delay_seconds", .00282),
             complete_RNG_state_identity=canonical_hash(seed.get("spike_states", {})),
             event_semantics="existing pre/post and left/right contract, unchanged"),
        dict(artifact=assembly.artifact_sha256, scope="current_local_group_only",
             existing_scope=getattr(assembly, "config", assembly.manifest.get("configuration", {})),
             runtime_goal_0_6_deg_is_not_validation=True),
        seed.get("rng_seed"), engineering_preview=True)


def population_indices(assembly, context):
    result = {}
    for name, p in assembly.populations.items():
        n = p.group_count
        ids = tuple(str(x) for x in getattr(p, "group_ids", tuple(name+":"+str(i) for i in range(n))))
        xy = getattr(p, "centroid_mm", None)
        mesh = getattr(p, "mesh_geometry", {})
        degxy = mesh.get("centroid_deg") if isinstance(mesh, dict) else None
        if degxy is not None:
            e = np.linalg.norm(np.asarray(degxy), axis=1)
            az = np.arctan2(np.asarray(degxy)[:, 1], np.asarray(degxy)[:, 0])
        elif xy is not None and context.units.get("mm_per_degree") is not None:
            e = np.linalg.norm(xy, axis=1)/context.units["mm_per_degree"]
            az = np.arctan2(np.asarray(xy)[:, 1], np.asarray(xy)[:, 0])
        else: e = az = None
        result[name] = PopulationIndex(name, ids, (name,)*n, np.asarray(p.counts),
            "coarse_group", xy, "mm" if xy is not None else "unknown", e, az,
            (None,)*n, ("effective_pool" if name.endswith("_AC_v1") else "source_model",)*n,
            assembly.artifact_sha256,coordinate_frame=getattr(p,"coordinate_role","source_declared_unknown"))
    return result


def source_catalog(root):
    """Import every existing effective profile including unexecuted/derived rows."""
    path = Path(root)/"configurations/effective_dynamics_v1.json"
    raw = read(path); records = []; coverage = {}
    for profile, definitions in raw["profiles"].items():
        coverage[profile] = {}
        for quantity, row in definitions.items():
            if not isinstance(row, dict): continue
            status = "proposed" if row.get("binding") != "derived" else "unsupported"
            if not row.get("unit"):
                raise ContractError("Existing profile has no explicit unit: "+profile+"/"+quantity)
            value = row.get("value")
            rule = dict(rule="type_constant", value=value) if value is not None and _numeric(value) else dict(
                rule="derived", dependencies=(), operation="unbound_existing_derivation")
            record = ParameterRecord("profile/"+profile+"/"+quantity, "profile:"+profile, (profile,), quantity,
                rule, row["unit"], Applicability(),
                Evidence(row.get("evidence_class", "project_approximation"), row.get("source_id", profile),
                    str(path.relative_to(root))+"#/profiles/"+profile+"/"+quantity, sha(path),
                    limitations=("Existing profile evidence is not upgraded by import",
                                 "uncalibrated_for_eccentricity")),
                "profile:"+profile, status=status)
            records.append(record); coverage[profile][quantity] = dict(status=status,
                binding=row.get("binding"), unit=row["unit"], derivation=row.get("derivation"))
    return records, coverage, raw


def route_registry(assembly):
    records = []; matrices = {}
    for name, route in assembly.routes.items():
        operators = {}; units = {}
        kinds = [("mean_weights", "dimensionless"), ("sum_weights", "dimensionless")]
        if getattr(route, "contact_component", None): kinds.append(("contact_coefficient_S", "S"))
        for kind, unit in kinds:
            path = "routes/"+name+"/"+kind+".npz"
            declared = assembly.manifest.get("files")
            generated = getattr(assembly, "_new_matrices", {})
            if declared is not None and path not in declared and path not in generated:
                continue  # No fabricated sum operator for a mean-only effective pool.
            try: matrix = assembly.matrix(path)
            except (KeyError, FileNotFoundError): continue
            # Do not clone, densify, renormalize, or mutate frozen CSR objects.
            if not sparse.isspmatrix_csr(matrix): raise ContractError("Frozen route is not CSR: "+path)
            operators[kind] = dict(identity=canonical_hash(dict(shape=matrix.shape, data=matrix.data,
                indices=matrix.indices, indptr=matrix.indptr)), shape=matrix.shape, nnz=matrix.nnz, path=path)
            units[kind] = unit; matrices[name, kind] = matrix
        records.append(RouteRecord(name, route.pre, route.post, operators, assembly.artifact_sha256,
            "N_post*permember_gain*(W_mean@activation); contact coefficient already summed; gap inS; no crop renormalization",
            units))
    if "gap" in assembly.manifest:
        matrix = assembly.matrix("gap/laplacian_S.npz")
        record = RouteRecord("gap_laplacian", "source_gap_population_order", "source_gap_population_order",
            dict(laplacian=dict(identity=canonical_hash(dict(shape=matrix.shape,data=matrix.data,
                indices=matrix.indices,indptr=matrix.indptr)),shape=matrix.shape,nnz=matrix.nnz)),
            assembly.artifact_sha256,"already aggregated original gap Laplacian, no N factor",dict(laplacian="S"))
        records.append(record);matrices["gap_laplacian","laplacian"]=matrix
    return RouteRegistry(records, matrices)


def resolve_group_snapshot(assembly, seed, root, *, context=None, protocol=None):
    """Resolve the final selected/derived snapshot, not stale base recipes.

    The selected B5/current RGC recipes, V+s pools, per-contact values, native
    references and qualified limitations come from the actual bound seed.
    """
    root = Path(root).resolve()
    context = context or context_from_seed(assembly, seed, protocol)
    indices = population_indices(assembly, context)
    records, catalog, raw = source_catalog(root)
    params = seed["inner_parameters"]
    registry = IntrinsicParameterRegistry(records)
    descriptors = {}
    numeric = {}; refs = {}
    source = "current-source-snapshot:"+canonical_hash(params)
    source_profiles = raw["profiles"]
    ac_params = getattr(assembly, "ac_config", {}).get("parameters_per_effective_pool_SI", {})
    coverage = dict(schema=SCHEMA, raw_source_profiles=catalog, active_parameters={},
        unsupported=dict(H2="not_implemented_in_current_native_H1_group",
            per_cell_mode="separate representation; not synthesized from group states",
            cone_two_compartment="only resolved if present in source profile/component, not added to nativeH1"),
        eccentricity_rule="existing spatial operators retained; no guessed C/gL/beta/history/AC/BC curves",
        input_optics=dict(context.optics), qualifiers=dict(FIT_READY=False, physiological_calibration=False),
        initial_state_recomputed=False)

    def add(path, value, unit, subject, mechanism, source_row=None, aggregation=None):
        if not _numeric(value): return
        row = {} if source_row is None else source_row
        status = "conditional" if context.conditional else "diagnostic"
        aggregation = aggregation or AggregationRule()
        n = len(indices[subject].ids)
        shape = np.shape(value); q = path.split("/")[-1]
        layout = ("broadcast" if not shape or q in ("kernel","spike_history_kernel")
                  else "per_group" if shape[:1] == (n,) else "component_vector")
        record = ParameterRecord(path, mechanism, (subject,), q,
            dict(rule="type_constant", value=value,layout=layout), unit, Applicability(),
            Evidence(row.get("evidence_class", "project_approximation"), row.get("source_id", source),
                row.get("locator", "current seed inner_parameters/"+path),
                limitations=("Exact existing value, not a new physiological estimate",
                             "uncalibrated_for_eccentricity")),
            subject, aggregation, status=status)
        registry.register(record); descriptors[path] = (subject, record.quantity, unit, mechanism)
        coverage["active_parameters"][path] = dict(unit=unit, mechanism=mechanism,
            source_class=record.evidence.source_class, aggregation=data(aggregation), status=status,
            uncalibrated_for_eccentricity=True)

    for pop, values in params["populations"].items():
        definitions = source_profiles.get("effective_"+pop, {})
        for key, value in values.items():
            row = definitions.get(key, {})
            unit = row.get("unit", UNIT.get(key))
            if pop.endswith("_AC_v1"):
                unit = dict(C="F", gL="S", EL="V", EE="V", theta="V", k="V",
                            alpha="s^-1", beta="s^-1", gE_base="S").get(key, unit)
            if unit is None:
                unit = "source_native:"+key  # opaque native units, never silently SI
            agg = AggregationRule("extensive") if key in ("C", "gL") and unit in ("F", "S") and not pop.endswith("_AC_v1") else AggregationRule()
            add("populations/"+pop+"/"+key, value, unit, pop, _mechanism(pop), row, agg)
    # Contact axes are distinct from population axes and may have different lengths.
    for component, values in params["contacts"].items():
        definitions = source_profiles.get("effective_"+component, {})
        # Scalar/component parameters are not expanded over contact count here.
        subject = "contact:"+component
        indices[subject] = PopulationIndex(subject, (subject,), (subject,), np.ones(1), "per_cell",
            None, "unknown", None, None, (None,), ("contact_component",), assembly.artifact_sha256)
        for key, value in values.items():
            row = definitions.get(key, {})
            add("contacts/"+component+"/"+key, value, row.get("unit", "source_native:"+key),
                subject, component, row, AggregationRule("none", "contact_component"))
    for route, gain in params["route_gains"].items():
        subject = "route:"+route
        indices[subject] = PopulationIndex(subject, (subject,), (subject,), np.ones(1), "per_cell",
            None, "unknown", None, None, (None,), ("route",), assembly.artifact_sha256)
        add("route_gains/"+route, gain, "S", subject, "RGC_receptor_route",
            aggregation=AggregationRule("none", "per_post_member"))
    subject = "rod_source"
    indices[subject] = PopulationIndex(subject, (subject,), (subject,), np.ones(1), "per_cell",
        None, "unknown", None, None, (None,), ("source",), assembly.artifact_sha256)
    for key, value in params["rod_phototransduction"].items():
        add("rod_phototransduction/"+key, value, "Chen_native:"+key, subject, "Chen2024PrimateRodCurrent")
    # Native source constants remain in their original units, explicitly opaque
    # where the existing snapshot offers no registered SI contract.
    from .vh2005 import SOURCE_VALUES
    subject = "cone_H1_native"
    indices[subject] = PopulationIndex(subject, (subject,), (subject,), np.ones(1), "per_cell",
        None, "unknown", None, None, (None,), ("native_source",), assembly.artifact_sha256)
    for key, value in SOURCE_VALUES.items():
        add("cone_H1_native/"+key, value, "vH_native:"+key, subject, "VanHateren2005NativeConeH1",
            dict(source_id="vh2005.SOURCE_VALUES", locator="retina_sim/vh2005.py::SOURCE_VALUES"))
    resolver = ParameterResolver(registry, context, indices)
    for path, (subject, quantity, unit, mechanism) in descriptors.items():
        numeric[path], refs[path] = resolver.resolve(subject, quantity, unit, mechanism=mechanism)
    # Additional totals are validation products, never supplied as per-member C/gL.
    for pop, values in params["populations"].items():
        if pop.endswith("_AC_v1"): continue
        for key in ("C", "gL"):
            path = "populations/"+pop+"/"+key
            if path in numeric:
                numeric[path+"_group_total"] = fp64(numeric[path]*indices[pop].member_count)
                refs[path+"_group_total"] = dict(from_parameter=path, aggregation="N_once",
                                               consumed_as="validation_only_not_legacy_permember")
    routes = route_registry(assembly)
    bundle = ResolvedParameterBundle(context, indices, numeric, refs, registry, routes, params, coverage)
    # Fail closed if any resolved active value deviates from the snapshot.
    legacy = compatible_inner_parameters(bundle)
    if canonical_hash(legacy) != canonical_hash(params):
        raise ContractError("Migration changed current parameter snapshot")
    return bundle


def compatible_inner_parameters(bundle):
    """Restore exact original scalar/array/boolean shapes and legacy identity."""
    original = clone(bundle.legacy_parameters)
    for path, value in bundle.values.items():
        keys = path.split("/")
        if keys[0] not in ("populations", "contacts", "route_gains", "rod_phototransduction"):
            continue
        if path.endswith("_group_total"): continue
        node = original
        for k in keys[:-1]: node = node[k]
        old = node[keys[-1]]
        if np.shape(old) != value.shape:
            if value.shape[1:] == np.shape(old) and np.array_equal(value,np.broadcast_to(value[0],value.shape)):
                value = value[0]
            else: raise ContractError("Resolved compatibility shape differs: "+path)
        if isinstance(old, np.ndarray): new = np.array(value, dtype=old.dtype, copy=True)
        elif isinstance(old, bool): new = bool(value.item())
        elif isinstance(old, int): new = int(value.item())
        elif isinstance(old, float): new = float(value.item())
        elif isinstance(old, (tuple, list)): new = type(old)(value.tolist())
        else: raise ContractError("Unsupported legacy numeric type")
        node[keys[-1]] = new
    return original


def state_bundle(seed, bundle):
    return StateBundle(seed["states"], {k: seed[k] for k in
        ("spike_states", "observation_prehistory") if k in seed},
        float(seed["time_seconds"]), bundle.identity, False)


def load_initial_state(path,bundle):
    """Distinct initial-state schema inside the unchanged generic evidence IO.

    This is deliberately not a complete model checkpoint: runtime controller,
    dense intervals and causal queues remain owned by the existing Runner.
    """
    from .group_time_storage import load_bundle
    expected=dict(schema="parameter-system-initial-state-1",static_parameter_bundle_id=bundle.identity)
    saved=load_bundle(path,expected_kind="validation_evidence",expected_identity=expected)["payload"]
    if saved.get("state_bundle_schema")!="parameter-system-initial-state-1":
        raise ContractError("StateBundle schema mismatch")
    return StateBundle(saved["states"],saved["history_rng_delay"],saved["time_seconds"],bundle.identity,False)


def export_bundle(bundle, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    keys = sorted(bundle.values)
    np.savez_compressed(destination/"resolved_fp64.npz",
                        **{"p"+str(i): bundle.values[k] for i, k in enumerate(keys)})
    manifest = bundle.manifest()
    bundle.registry.write_json(destination/"registry.json")
    manifest["array_keys"] = {k: "p"+str(i) for i, k in enumerate(keys)}
    manifest["array_sha256"] = sha(destination/"resolved_fp64.npz")
    manifest["registry_file_sha256"] = sha(destination/"registry.json")
    (destination/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
        allow_nan=False)+"\n", encoding="utf8")
    return manifest


def bind_constructed_group(assembly, result, root):
    """Actual legacy construction entry now returns resolver-consumed parameters."""
    seed = dict(inner_parameters=result.inner_parameters, states=result.states,
        spike_states=result.spike_states, observation_prehistory=result.observation_prehistory,
        input_reference=result.input_reference, time_seconds=result.time_seconds,
        rng_seed=result.identity["rng_seed"])
    bundle = resolve_group_snapshot(assembly, seed, root)
    from dataclasses import replace
    new_identity = dict(result.identity, parameter_binding_schema=SCHEMA,
                        resolved_parameter_bundle_id=bundle.identity)
    return replace(result, inner_parameters=compatible_inner_parameters(bundle),
                   identity=new_identity, resolved_parameters=bundle,
                   separate_state_bundle=state_bundle(seed, bundle))


def attach_compiled_parameters(compiled, root):
    """Builder integration uses actual component values, including chosen recipes.

    This component path records validated native matrices/ragged kernels without
    flattening them or applying group aggregation. Group execution uses the exact
    adapter above, including its real member counts and CSR operators.
    """
    snapshot = {}; indices = {}; records = []; resolved = {}; refs = {}
    for name,pop in compiled.populations.items():
        cells=pop.cells
        q=pop.receptive_field_positions or pop.anatomical_positions
        xy=None if q is None else q.values
        unit="unknown" if q is None else q.unit
        e=cells.eccentricity.values if cells.eccentricity.unit=="deg" else None
        az=None if xy is None else np.arctan2(xy[:,1],xy[:,0])
        indices[name]=PopulationIndex(name,tuple(cells.cell_id),tuple(cells.cell_type),
            np.ones(len(cells.cell_id)),"per_cell",xy,unit,e,az,(None,)*len(cells.cell_id),
            ("source_component",)*len(cells.cell_id),pop._population_hash,
            coordinate_frame=q.coordinate_frame if q is not None else "source_declared_unknown")
    for cid, component in compiled.components.items():
        source_records = getattr(component.parameters, "records", ())
        snapshot[cid] = {}
        subject="component:"+cid
        indices[subject]=PopulationIndex(subject,(subject,),(subject,),np.ones(1),"per_cell",
            None,"unknown",None,None,(None,),("native_component",),compiled.assembly_hash)
        for r in source_records:
            raw = getattr(r, "value", None)
            value = getattr(raw, "values", raw)
            unit = getattr(raw, "unit", None) or getattr(r, "unit", None)
            if value is not None and _numeric(value):
                if not unit: raise ContractError("Compiled parameter lost unit: "+cid)
                snapshot[cid][r.parameter_id] = dict(value=fp64(value), unit=unit,
                    mechanism=component.mechanism, source_id=getattr(r, "source_id", "existing_component"),
                    locator=getattr(r, "source_locator", "actual component parameter record"))
                key=subject+"/"+r.parameter_id
                evidence_class=getattr(r,"evidence_class",getattr(r,"evidence_kind","project_approximation"))
                source_id=getattr(r,"source_id",None) or (
                    "software_fixture:"+cid if evidence_class in ("software_fixture","synthetic_fixture")
                    else "existing_component_derivation:"+cid)
                evidence=Evidence(evidence_class,source_id,
                    getattr(r,"source_locator",None) or "actual compiled component/"+cid+"/"+r.parameter_id,
                    limitations=("Imported current component, qualification unchanged",))
                records.append(ParameterRecord(key,component.mechanism,(subject,),r.parameter_id,
                    dict(rule="type_constant",value=value),unit,Applicability(),evidence,subject,
                    AggregationRule("none","actual_component_parameter"),status="diagnostic"))
                resolved[key]=fp64(value);refs[key]=key
    registry=IntrinsicParameterRegistry(records)
    context=RunContext("existing_component_species_profiles","unknown",
        dict(source="existing compiled ports/photometry contracts",calibrated_by_this_migration=False),
        dict(policy_id="existing_component_background"),
        dict(source_units="per_parameter_exact",dtype="float64"),
        dict(model_id=compiled.spec.model_id,source="existing_external_input_contracts"),
        compiled.clock,dict(scope=compiled.spec.scope,compiled_assembly=compiled.assembly_hash),None)
    route_records=[];matrices={}
    for i,link in enumerate(compiled.links):
        name="component_link:"+str(i);m=link.matrix
        if not sparse.isspmatrix_csr(m):m=sparse.csr_matrix(m)
        matrices[name,"existing_operator"]=m
        route_records.append(RouteRecord(name,link.specification.source_component,
            link.specification.target_component,dict(operator=dict(shape=m.shape,nnz=m.nnz,
                sha=canonical_hash(dict(data=m.data,indices=m.indices,indptr=m.indptr)))),
            compiled.assembly_hash,"existing component CSR, no aggregation/renormalization",
            dict(existing_operator=tuple(sorted(set(link.specification.connections.weight_unit))))))
    routes=RouteRegistry(route_records,matrices)
    bundle=ResolvedParameterBundle(context,indices,resolved,refs,registry,routes,
        dict(components=snapshot),dict(scope="already_instantiated_component_arrays_only",
            no_global_preexpansion=True,source_qualification_unchanged=True))
    compiled.parameter_system_schema = SCHEMA
    compiled.parameter_system_snapshot = snapshot
    compiled.resolved_parameters=bundle
    compiled.parameter_system_binding_id=bundle.identity
    return compiled
