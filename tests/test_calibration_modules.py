"""Small isolated experiments; synthetic clamps exercise units, not animal fits."""
from pathlib import Path
from dataclasses import replace
import json
import sys
import numpy as np
import pytest
from retina_sim.contracts import Quantity, ContractError, SolverConfig
from retina_sim.calibration_modules import (build_module, PortTrace, affected_closure,
    gap_boundary_id, voltage_clamp_conductance, parameter_package, merge_parameter_package)
from examples.full_target_network import full_target_network

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = dict(source_id='synthetic_unit_test_only', species='macaque_project_fixture',
    retinal_region='engineering_preview_strip', baseline_source='explicit_test_clamps',
    time_origin='experiment_start_seconds', adaptation_status='factory_state_not_adapted')


@pytest.fixture(scope='module')
def circuit():
    return full_target_network(ROOT, effective_cones=64, effective_rods=96,
                               duration=.0002, runtime_profile='effective_dynamics_v1')


def baselines(component):
    return {name: Quantity(np.zeros(len(port.cell_ids)), port.unit, port.meaning, port.evidence_tag)
            for name, port in component.inputs.items()}


def make_rgc(circuit, mode='conductance_clamp'):
    parent, _, audit = circuit
    cid = 'midget_RGC_ON'
    baseline = baselines(parent.components[cid])
    if mode == 'current_clamp':
        n = len(parent.components[cid].outputs['voltage'].cell_ids)
        baseline['injected_current'] = Quantity(np.zeros(n), 'A', 'injected_current_positive_inward',
                                                parent.components[cid].evidence_tag)
    return build_module(parent, audit['registry'], cid, kind='rgc', mode=mode,
                        conditions=CONDITIONS, baselines=baseline)


def traces(module):
    cid = module.manifest['component_id']
    return {name: PortTrace([0., .0001, .0002], np.tile(module.baselines[name].values, (2, 1)),
        port.unit, port.meaning, port.cell_ids, module.baselines[name].values,
        'injected_current' if name == 'injected_current' else 'synaptic_conductance',
        'synthetic_unit_test_only', CONDITIONS['time_origin'])
        for name, port in module.compiled.components[cid].inputs.items()}


def test_rgc_uses_same_factory_and_runs_two_clamped_bins(circuit):
    parent, _, _ = circuit
    module = make_rgc(circuit)
    cid = module.manifest['component_id']
    original = parent.components[cid]
    isolated = module.compiled.components[cid]
    assert type(original.native_model) is type(isolated.native_model)
    assert original.implementation_hash == isolated.implementation_hash
    incoming = module.baselines
    blocks = tuple(q.values for q in isolated.initial())
    a = original.derivative(0, blocks, incoming)[0].values
    b = isolated.derivative(0, blocks, incoming)[0].values
    assert np.array_equal(a, b)
    config = SolverConfig(method='DOP853', rtol=1e-8, atol=1e-11,
                          max_step=.0001, input_interpolation='zero_order_hold')
    state = module.compiled.initialize(config)
    state, result = module.compiled.advance(state, module.inputs(traces(module)), .0002, config)
    assert state.time_seconds == .0002
    assert not module.manifest['automatic_fitting_enabled']


def test_injected_current_has_explicit_port_and_correct_units(circuit):
    module = make_rgc(circuit, 'current_clamp')
    component = module.compiled.components[module.manifest['component_id']]
    blocks = tuple(q.values for q in component.initial())
    baseline = dict(module.baselines)
    zero = component.derivative(0, blocks, baseline)[0].values
    n = len(baseline['injected_current'].values)
    baseline['injected_current'] = Quantity(np.full(n, 1e-12), 'A', 'injected_current_positive_inward', component.evidence_tag)
    injected = component.derivative(0, blocks, baseline)[0].values
    assert np.allclose(injected-zero, 1e-12/component.native_model.C, atol=1e-14)
    wrong = traces(module)
    wrong['injected_current'] = replace(wrong['injected_current'], recording_kind='voltage_clamp_recording')
    with pytest.raises(ContractError, match='Recording boundary type mismatch'):
        module.inputs(wrong)


def test_current_clamp_is_reference_only_and_original_compiler_pins_survive(circuit):
    import hashlib, ast
    from retina_sim.compiled_kernel import IMPLEMENTATIONS, SUPPORTED
    from retina_sim.network import JointNetwork, InputBundle
    from retina_sim.calibration_current_clamp import KEY
    for name, expected in IMPLEMENTATIONS.items():
        assert hashlib.sha256((ROOT/'retina_sim'/name).read_bytes()).hexdigest() == expected
    # Inspect the GPU source guards without initializing or running a GPU job.
    tree = ast.parse((ROOT/'retina_sim/gpu_joint_validation.py').read_text(encoding='utf8'))
    gpu_pins = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'PINS' for target in node.targets))
    for name, expected in gpu_pins.items():
        assert hashlib.sha256((ROOT/'retina_sim'/name).read_bytes()).hexdigest() == expected
    assert KEY not in SUPPORTED
    module = make_rgc(circuit, 'current_clamp')
    assert module.manifest['compute_policy'] == 'python_reference_only'
    auto = JointNetwork(module.compiled, compute_backend='auto')
    assert auto.compute_metadata['backend'] == 'python_reference'
    assert KEY in auto.compute_metadata['fallback_reason']
    with pytest.raises(ContractError, match='Unsupported mechanisms'):
        JointNetwork(module.compiled, compute_backend='numba')
    # The unchanged conductance-driven RGC still takes the original compiled path.
    normal = make_rgc(circuit)
    jit = JointNetwork(normal.compiled, compute_backend='numba')
    py = JointNetwork(normal.compiled, compute_backend='python')
    assert jit.compute_metadata['backend'] != 'python_reference'
    y = np.concatenate([q.values for q in py.initial_conditions()])
    bundle = InputBundle({key: value.at(0.) for key, value in normal.inputs(traces(normal)).items()})
    np.testing.assert_allclose(jit.rhs_flat(0., y, bundle, jit.parameters, jit.graph),
        py.rhs_flat(0., y, bundle, py.parameters, py.graph), rtol=0, atol=2e-10)
    current_package = parameter_package(module, version='current-fixture-v1', source_id='unit_test', applicability=CONDITIONS)
    parent, _, audit = circuit
    assert merge_parameter_package(parent, audit['registry'], current_package,
        applicability=CONDITIONS).assembly_hash == parent.assembly_hash


def test_conductance_trace_rejects_units_negative_sign_and_timing(circuit):
    module = make_rgc(circuit)
    data = traces(module)
    for change in ({'unit': 'nS'}, {'values': -np.ones_like(data['excitation'].values)},
                   {'edges_seconds': [0, .0002, .0001]}, {'time_origin': 'unknown_start'}):
        wrong = dict(data); wrong['excitation'] = replace(data['excitation'], **change)
        with pytest.raises(ContractError):
            module.inputs(wrong)


def test_voltage_clamp_conversion_does_not_guess_or_rectify():
    current = lambda values: Quantity(values, 'A', 'voltage_clamp_current_positive_outward', 'synthetic_fixture')
    q, metadata = voltage_clamp_conductance(current([[-60e-12]]), holding_V=-.06,
        reversal_V=0., receptor='excitatory', leak_subtracted=True,
        isolated_receptor_current=True, source_id='synthetic_unit_test_only')
    assert np.allclose(q.values, 1e-9) and q.unit == 'S'
    assert metadata['input_current_sign'] == 'positive_outward'
    with pytest.raises(ContractError, match='negative conductance'):
        voltage_clamp_conductance(current([60e-12]), holding_V=-.06, reversal_V=0.,
            receptor='excitatory', leak_subtracted=True, isolated_receptor_current=True, source_id='fixture')
    with pytest.raises(ContractError, match='nonzero driving force'):
        voltage_clamp_conductance(current([0.]), holding_V=0., reversal_V=0., receptor='excitatory',
            leak_subtracted=True, isolated_receptor_current=True, source_id='fixture')


@pytest.mark.parametrize('cid,kind', [('local_crossover_AC', 'generic_ac_clamped'),
                                    ('midget_BC_ON', 'bc_clamped')])
def test_bc_ac_boundary_and_derived_baseline_are_explicit(circuit, cid, kind):
    parent, _, audit = circuit
    component = parent.components[cid]
    baseline = baselines(component)
    profile = component.parameters.profile
    reference = next(r.value for r in profile.records if r.parameter_id == 'baseline')
    baseline['drive'] = Quantity(reference.values, component.inputs['drive'].unit,
                                 component.inputs['drive'].meaning, component.evidence_tag)
    gaps = [gap_boundary_id(g) for g in parent.spec.gap_junctions if cid in (g.left_component, g.right_component)]
    if gaps:
        with pytest.raises(ContractError, match='every opened electrical gap'):
            build_module(parent, audit['registry'], cid, kind=kind, mode='declared_input_clamps',
                         conditions=CONDITIONS, baselines=baseline)
    module = build_module(parent, audit['registry'], cid, kind=kind, mode='declared_input_clamps',
                         conditions=CONDITIONS, baselines=baseline, open_gap_boundaries=gaps)
    assert not module.compiled.spec.links and not module.compiled.spec.gap_junctions
    assert module.manifest['free_fit_parameters'] == []
    assert module.manifest['initial_state'].endswith('not verified')


def test_cone_hc_stays_one_joint_feedback_factory(circuit):
    parent, _, audit = circuit
    cid = next(cid for cid, c in parent.components.items() if c.mechanism == 'VanHateren2005NativeConeH1')
    component = parent.components[cid]
    baseline = {name: Quantity(np.full(len(port.cell_ids), 100.), port.unit, port.meaning, port.evidence_tag)
                for name, port in component.inputs.items()}
    module = build_module(parent, audit['registry'], cid, kind='cone_hc_joint', mode='native_light_drive',
                          conditions=CONDITIONS, baselines=baseline)
    assert module.compiled.components[cid].native_model.__class__ is component.native_model.__class__
    assert set(module.compiled.spec.components[0].population_roles) >= {'cone', 'h1'}
    hc_id = module.compiled.spec.components[0].population_roles['h1']
    assert set(module.compiled.populations[hc_id].cells.subtype) == {'H1', 'H2'}
    assert module.manifest['feedback'] == 'joint internal Cone-HC retained'


def test_parameter_snapshot_roundtrip_and_feedback_invalidation(circuit):
    parent, _, audit = circuit
    module = make_rgc(circuit)
    applicability = dict(CONDITIONS)
    package = parameter_package(module, version='fixture-v1', source_id='unit_test', applicability=applicability)
    merged = merge_parameter_package(parent, audit['registry'], package, applicability=applicability)
    assert merged.assembly_hash == parent.assembly_hash
    assert not merged.parameter_package_merge['initialization_transferred']
    assert package['status'] == 'unfitted_snapshot'
    assert package['schema_version']=='module-parameters-1.1'
    context=package['system_compatibility']
    assert set(context['component_identities'])==set(parent.components)
    assert context['assembled_RGC_targets']==[cid for cid,c in parent.components.items() if c.mechanism=='HistoryConductanceRGC']
    assert not merged.parameter_package_merge['system_acceptance']
    assert not context['integration_policy']['arbitrary_other_layer_compensation_allowed']
    assert 'all_assembled_RGC_targets' in context['integration_policy']['required_reassembly_checks']
    # Rehashing cannot make a package with altered surrounding bindings valid.
    import copy
    from retina_sim.contracts import canonical_hash
    changed=copy.deepcopy(package)
    changed['system_compatibility']['connectivity_hash']='changed_feedback'
    changed['content_hash']=canonical_hash({k:v for k,v in changed.items() if k!='content_hash'})
    with pytest.raises(ContractError,match='Full-system'):
        merge_parameter_package(parent,audit['registry'],changed,applicability=applicability)
    # Historical identity-only snapshots stay readable, with no new validation claim.
    legacy=copy.deepcopy(package);legacy['schema_version']='module-parameters-1.0'
    legacy.pop('system_compatibility')
    legacy['content_hash']=canonical_hash({k:v for k,v in legacy.items() if k!='content_hash'})
    restored=merge_parameter_package(parent,audit['registry'],legacy,applicability=applicability)
    assert restored.parameter_package_merge['integration_contract_status']=='legacy_identity_snapshot_no_system_acceptance'
    closure = affected_closure(parent.spec, ['midget_BC_ON'])
    assert {'midget_BC_ON', 'AII_AC', 'diffuse_BC_ON', 'midget_BC_OFF'} <= set(closure)
    assert affected_closure(parent.spec, ['midget_RGC_ON']) == ['midget_RGC_ON']
    with pytest.raises(ContractError, match='applicability mismatch'):
        merge_parameter_package(parent, audit['registry'], package, applicability={'region': 'other'})


@pytest.mark.parametrize('kind', ['rgc', 'cone_hc_joint'])
def test_cli_runs_only_the_supplied_module_contract(circuit, tmp_path, monkeypatch, kind):
    from tools import run_calibration_module as cli
    monkeypatch.setattr(cli, 'atlas_population_network', lambda *a, **k: circuit)
    if kind == 'rgc':
        module = make_rgc(circuit)
    else:
        parent, _, audit = circuit
        cid = next(cid for cid, c in parent.components.items() if c.mechanism == 'VanHateren2005NativeConeH1')
        component = parent.components[cid]
        baseline = {name: Quantity(np.full(len(port.cell_ids), 100.), port.unit, port.meaning, port.evidence_tag)
                    for name, port in component.inputs.items()}
        module = build_module(parent, audit['registry'], cid, kind=kind, mode='native_light_drive',
                              conditions=CONDITIONS, baselines=baseline)
    cid = module.manifest['component_id']
    data = {name: PortTrace([0., .0001, .0002], np.tile(module.baselines[name].values, (2, 1)),
        port.unit, port.meaning, port.cell_ids, module.baselines[name].values,
        'native_light' if kind == 'cone_hc_joint' else 'synaptic_conductance',
        'synthetic_unit_test_only', CONDITIONS['time_origin'])
        for name, port in module.compiled.components[cid].inputs.items()}
    contract = dict(conditions=CONDITIONS, parameter_snapshot_version='synthetic-cli-v1',
        solver=dict(method='DOP853', rtol=1e-8, atol=1e-11, max_step=.0001,
                    input_interpolation='zero_order_hold'),
        inputs={name: dict(edges_seconds=list(trace.edges_seconds), values=trace.values.tolist(),
            unit=trace.unit, meaning=trace.meaning, cell_ids=list(trace.cell_ids),
            baseline=trace.baseline.tolist(), recording_kind=trace.recording_kind, source_id=trace.source_id)
            for name, trace in data.items()})
    path=tmp_path/'contract.json'; path.write_text(json.dumps(contract), encoding='utf8')
    output=tmp_path/'experiment'
    monkeypatch.setattr(sys, 'argv', ['run_calibration_module.py', '--component', cid,
        '--kind', kind, '--mode', module.manifest['mode'], '--contract', str(path), '--output', str(output), '--run'])
    cli.main()
    result=json.loads((output/'result.json').read_bytes())
    assert result['status'] == 'ISOLATED_MODULE_RUN_COMPLETE_NOT_CALIBRATED'
    assert result['time_seconds'] == .0002
    assert (output/'outputs.npz').is_file()
    package = json.loads((output/'parameters.json').read_bytes())
    assert package['status'] == 'unfitted_snapshot'
    assert result['completed_files'] == ['module.json', 'outputs.npz', 'parameters.json']
    if kind == 'cone_hc_joint':
        assert package['parameter_record_format'] == 'native_ParameterSet'
        assert package['native_profile_metadata']['profile_id'] == module.compiled.components[cid].parameters.profile.profile_id
        assert package['records']['VH2005_tau_R']['unit'] == 'ms'
        parent, _, audit = circuit
        merged = merge_parameter_package(parent, audit['registry'], package, applicability=CONDITIONS)
        assert merged.assembly_hash == parent.assembly_hash
    else:
        # A late persistence failure must never leave a completed result receipt.
        def fail_save(*args, **kwargs):
            raise OSError('synthetic parameter persistence failure')
        monkeypatch.setattr(cli, 'save_parameter_package', fail_save)
        failed = tmp_path/'failed_experiment'
        monkeypatch.setattr(sys, 'argv', ['run_calibration_module.py', '--component', cid,
            '--kind', kind, '--mode', module.manifest['mode'], '--contract', str(path), '--output', str(failed), '--run'])
        with pytest.raises(OSError, match='synthetic parameter persistence failure'):
            cli.main()
        assert not (failed/'result.json').exists()
