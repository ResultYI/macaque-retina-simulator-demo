"""Inspect a retained module, or run an explicitly supplied clamped experiment.

Default is inspection only. This tool never fits parameters or creates stimuli.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retina_platform.config import RunConfig
from retina_sim.atlas_population_network import atlas_population_network
from retina_sim.calibration_modules import (MECHANISMS, MODULE_MODES, build_module, gap_boundary_id,
    PortTrace, parameter_package, save_parameter_package)
from retina_sim.contracts import Quantity, SolverConfig, ContractError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', required=True)
    parser.add_argument('--kind', required=True, choices=MECHANISMS)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true', help='Explicitly advance only this module')
    parser.add_argument('--equation-route', choices=('retained','bc_ac_receptor_candidate_v1','bc_ac_contact_first_candidate_v2'), default='retained')
    parser.add_argument('--candidate-profile', type=Path)
    args = parser.parse_args()
    candidate = None
    if args.equation_route != 'retained':
        from retina_sim.bc_ac_candidate import CandidateProfile
        if args.kind not in ('bc_clamped','generic_ac_clamped'):
            raise ContractError('BC/AC candidate selection applies only to BC or generic AC modules')
        candidate=CandidateProfile(args.candidate_profile or ROOT/('configurations/'+args.equation_route+'.json'))
        if candidate.route!=args.equation_route:raise ContractError('Selected route differs from candidate profile identity')
        missing=candidate.missing((args.component,))
        if missing and not args.run:
            print(json.dumps(dict(status='BC_AC_CANDIDATE_UNBOUND_NO_RUN', missing=missing,
                equation_route=args.equation_route, candidate_profile=str(candidate.path),
                pooling_order=candidate.data['pooling_order'], default_enabled=False,
                pooling_approval_status='CONTACT_FIRST_SHARED_TAU_APPROVED_STRUCTURE' if candidate.contact_first else 'UNRESOLVED_SYNTHETIC_AGGREGATE_EXAMPLE_ONLY',
                production_parameter_binding=False,
                glutamate_concentration_claim=False),ensure_ascii=False,indent=2))
            return
        candidate.require_ready((args.component,))
    elif args.candidate_profile:
        raise ContractError('--candidate-profile requires explicit --equation-route selection')
    run = RunConfig.project_default(ROOT)
    parent, _, audit = atlas_population_network(ROOT, duration=.0001,
        execution_profile=run.execution_profile, configuration_path=run.circuit_configuration_path)
    if candidate is not None:
        from retina_sim.bc_ac_candidate import migrate_bc_ac
        background=None
        if candidate.data.get('hypothesis_baseline'):
            from retina_sim.bc_ac_hypothesis import constant_background_channels
            from retina_sim.runtime_profiles import LightContext
            spec=candidate.data['background_spec']
            light=LightContext(**{k:v for k,v in spec.items() if k!='common_physical_scene_claim'})
            bg=constant_background_channels(parent,light,stop=.0001)
            background={k:v.at(0.) for k,v in bg.items()}
        parent, audit['registry'], _ = migrate_bc_ac(parent,audit['registry'],candidate,targets=(args.component,),background_inputs=background)
    if args.component not in parent.components:
        raise ContractError('Component absent from the shared default circuit')
    component = parent.components[args.component]
    if component.mechanism not in MECHANISMS[args.kind] or args.mode not in MODULE_MODES[args.kind]:
        raise ContractError('Inspection/run kind and mode must match the retained factory')
    gaps = [gap_boundary_id(g) for g in parent.spec.gap_junctions if
            args.component in (g.left_component, g.right_component)]
    if not args.run:
        from retina_sim.calibration_current_clamp import KEY as CURRENT_CLAMP_KEY
        ports = {name: dict(unit=port.unit, meaning=port.meaning, cell_ids=list(port.cell_ids))
                 for name, port in component.inputs.items()}
        if args.mode == 'current_clamp':
            ports['injected_current'] = dict(unit='A', meaning='injected_current_positive_inward',
                cell_ids=list(component.outputs['voltage'].cell_ids))
        print(json.dumps(dict(status='INSPECTION_ONLY_NO_NEURAL_ADVANCE', component=args.component,
            parent_assembly_hash=parent.assembly_hash,
            equation_revision=parent.modeling_scope['equation_revision'] if hasattr(parent,'modeling_scope') else 'historical',
            mechanism=CURRENT_CLAMP_KEY if args.mode=='current_clamp' else component.mechanism,
            base_mechanism=component.mechanism, base_implementation_hash=component.implementation_hash,
            compute_policy='python_reference_only' if args.mode=='current_clamp' or candidate is not None else 'existing_reviewed_backend_selection',
            mode=args.mode, input_ports=ports, required_open_gap_boundaries=gaps,
            baseline_policy='Provide per-cell typed baselines from the experimental contract',
            conditions_required=['source_id', 'species', 'retinal_region', 'baseline_source',
                                 'time_origin', 'adaptation_status'],
            trace_format='Explicit bin edges_seconds, bin-by-cell values, ordered cell_ids; no resampling',
            calibration_status='unfitted; no automatic optimizer'), ensure_ascii=False, indent=2))
        return
    if args.contract is None or args.output is None:
        parser.error('--run requires --contract and a new --output directory')
    contract = json.loads(args.contract.read_text(encoding='utf-8'))
    baseline = {name: Quantity(row['baseline'], row['unit'], row['meaning'], component.evidence_tag)
                for name, row in contract['inputs'].items()}
    module = build_module(parent, audit['registry'], args.component, kind=args.kind, mode=args.mode,
        conditions=contract['conditions'], baselines=baseline,
        open_gap_boundaries=contract.get('open_gap_boundaries', ()))
    traces = {}
    for name, row in contract['inputs'].items():
        if 'npz_file' in row:
            with np.load(args.contract.parent / row['npz_file'], allow_pickle=False) as archive:
                edges = archive[row['edges_key']].copy(); values = archive[row['values_key']].copy()
        else:
            edges = row['edges_seconds']; values = row['values']
        traces[name] = PortTrace(edges, values, row['unit'], row['meaning'], tuple(row['cell_ids']),
            row['baseline'], row['recording_kind'], row['source_id'], contract['conditions']['time_origin'])
    channels = module.inputs(traces)
    start = next(iter(channels.values())).start_seconds
    stop = next(iter(channels.values())).stop_seconds
    solver = SolverConfig(**contract['solver'])
    if solver.input_interpolation != 'zero_order_hold':
        raise ContractError('This explicit bin interface requires zero_order_hold')
    package = parameter_package(module, version=contract['parameter_snapshot_version'],
        source_id=contract['conditions']['source_id'], applicability=contract['conditions'])
    # Fail unsupported parameter types before running or creating an output folder.
    json.dumps(package, ensure_ascii=False, allow_nan=False)
    # Exclusive output creation prevents overwriting any prior experiment.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'module.json').write_text(json.dumps(module.manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    state = module.compiled.initialize(solver, time_seconds=start)
    samples = sorted({float(t) for trace in traces.values() for t in trace.edges_seconds})
    state, result = module.compiled.advance(state, channels, stop, solver, sample_times=samples)
    np.savez_compressed(args.output/'outputs.npz', **{
        name: q.values for name, q in result.outputs_with_units.items()})
    summary = dict(status='ISOLATED_MODULE_RUN_COMPLETE_NOT_CALIBRATED', time_seconds=state.time_seconds,
        output_units={name: dict(unit=q.unit, meaning=q.meaning) for name, q in result.outputs_with_units.items()},
        input_contract=result.input_contract, module=module.manifest)
    save_parameter_package(package, args.output/'parameters.json')
    summary['completed_files'] = ['module.json', 'outputs.npz', 'parameters.json']
    if any(not (args.output/name).is_file() for name in summary['completed_files']):
        raise ContractError('Module output files incomplete; completion was not published')
    # A final success receipt is published only after every deliverable exists.
    pending = args.output/'result.pending.json'
    pending.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(args.output/'result.json')
    print(json.dumps(dict(status=summary['status'], output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    from retina_sim.bc_ac_candidate import CandidateParametersMissing,CandidatePoolingUnresolved
    try:
        main()
    except CandidateParametersMissing as error:
        print(json.dumps(dict(status='BC_AC_CANDIDATE_UNBOUND_NO_RUN',missing=error.missing,
            message=str(error)),ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
    except CandidatePoolingUnresolved as error:
        print(json.dumps(dict(status='CANDIDATE_POOLING_UNRESOLVED_NO_RUN',
            message=str(error),production_parameter_binding=False),ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
