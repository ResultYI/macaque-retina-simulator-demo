"""Named independent Chen/effective-cable reference using existing closure values."""
from dataclasses import replace
from pathlib import Path
from retina_sim.assembly import RetinaBuilder
from retina_sim.contracts import RunMode
from retina_sim.runtime_profiles import execution_profile
from examples.full_target_network import full_target_network

def physical_effective_network(root,**options):
    root=Path(root)
    profile=execution_profile(root,'macaque_physical_effective_v1')
    options.update(execution_profile='source_constrained_approximation',runtime_profile=profile['parameter_profile'])
    _,channels,audit=full_target_network(root,**options)
    selected={'physical_OS','physical_current_SI','physical_membrane','physical_terminal','physical_HC'}
    original=audit['spec']
    components=tuple(c for c in original.components if c.component_id in selected)
    populations={pid for c in components for pid in c.population_roles.values()}
    spec=replace(original,model_id='macaque_physical_effective_v1',
        components=components,populations=tuple(p for p in original.populations if p.population_id in populations),
        links=tuple(l for l in original.links if l.source_component in selected and l.target_component in selected),
        external_inputs=tuple(e for e in original.external_inputs if e.target_component in selected),
        gap_junctions=(),required_mechanisms=('cone_outer_segment','cone_passive_cable','terminal_activation_proxy','physical_effective_HC_feedback'),
        scope=profile['claim'])
    compiled=RetinaBuilder(audit['registry'],max_compile_bytes=profile['budgets']['max_compile_bytes']).build(spec,RunMode.REFERENCE)
    audit.update(spec=spec,resolved_execution_profile=profile,cone_route='Chen_peripheral_LM_effective_reference')
    return compiled,{'physical_cone_light':channels['physical_cone_light']},audit
