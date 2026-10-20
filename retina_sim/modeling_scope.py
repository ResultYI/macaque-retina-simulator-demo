"""Executable scope for the retained effective circuit, without a dynamics refit."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import numpy as np
from .contracts import ContractError, canonical_hash


class ModelingScope:
    def __init__(self, root, binding):
        root = Path(root).resolve()
        path = (root / binding['path']).resolve()
        if not path.is_relative_to(root):
            raise ContractError('Modeling scope must stay inside project')
        raw = path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.data = json.loads(raw)
        if (self.sha256 != binding['sha256'] or
                self.data.get('schema_version') != 'modeling-scope-1.0'):
            raise ContractError('Modeling scope identity/version mismatch')
        self.path = str(path)

    def validate_parameters(self, profiles):
        for name, expected in self.data['fixed_parameter_signatures'].items():
            rows = profiles.get(name)
            if rows is None:
                raise ContractError('Scoped parameter family missing: ' + name)
            # Evidence prose can be corrected independently. These fields actually
            # determine numbers, units, broadcasting and allowed derived balances.
            signature = {key: {field: row[field] for field in
                         ('binding', 'unit', 'meaning', 'value') if field in row}
                         for key, row in rows.items()}
            if canonical_hash(signature) != expected:
                raise ContractError('Fixed effective parameters changed: ' + name +
                                    '; revise the versioned scope before calibration')

    def validate_bound_parameters(self, registry, definitions):
        for name in self.data['fixed_parameter_signatures']:
            profile = registry.get(name)
            if profile is None:
                continue
            records = {row.parameter_id: row for row in profile.records}
            if set(records) != set(definitions[name]):
                raise ContractError('Scoped runtime parameter set changed: ' + name)
            for key, row in definitions[name].items():
                record = records[key]
                if (record.value.unit, record.value.meaning) != (row['unit'], row['meaning']):
                    raise ContractError('Scoped runtime parameter units changed: ' + name + '.' + key)
                if row['binding'] == 'derived':
                    continue  # Actual upstream baseline / conditional initial V0.
                value = np.asarray(record.value.values)
                expected = np.asarray(row['value'])
                if row['binding'] == 'broadcast':
                    matches = np.all(value == expected)
                else:
                    matches = value.shape == expected.shape and np.array_equal(value, expected)
                if not matches:
                    raise ContractError('Fixed runtime parameter changed: ' + name + '.' + key)

    def annotate(self, spec):
        acs = self.data['generic_ac_units']
        targets = self.data['rgc_targets']
        components = {c.component_id: c for c in spec.components}
        annotation = {}
        isolated = spec.scope == 'isolated_module_clamped_v1'
        if isolated and (len(spec.components) != 1 or spec.links or spec.gap_junctions):
            raise ContractError('Isolated module must expose all cut boundaries explicitly')
        for cid, c in components.items():
            boundary = c.options.get('module_boundary') if isolated else None
            if isolated and (not boundary or not boundary.get('parent_assembly_hash') or
                    set(boundary['ports']) != {e.target_port for e in spec.external_inputs}):
                raise ContractError('Isolated module boundary provenance/ports missing')
            if c.mechanism == 'ResidualAC' or cid in acs:
                candidate=c.options.get('candidate_dynamics',{})
                candidate_ac=((c.mechanism,candidate.get('route')) in (
                    ('CandidateGenericAmacrineReceptorMembrane','bc_ac_receptor_candidate_v1'),
                    ('ContactFirstGenericAmacrineReceptorMembraneV2','bc_ac_contact_first_candidate_v2')) and
                    candidate.get('parameter_profile_id')==c.parameter_profile_id and
                    candidate.get('glutamate_concentration_claim') is False and candidate.get('profile_sha256'))
                if cid not in acs or (c.mechanism != 'ResidualAC' and not candidate_ac):
                    raise ContractError('Unapproved generic AC identity/mechanism: ' + cid)
                if not candidate_ac and c.parameter_profile_id != 'effective_' + cid:
                    raise ContractError('Generic AC parameter profile changed: ' + cid)
                drivers = [(l.source_component, l.source_port) for l in spec.links
                           if l.target_component == cid and l.target_port == 'drive']
                if candidate_ac and candidate.get('route')=='bc_ac_contact_first_candidate_v2' and not isolated:
                    traced=[]
                    for aid,port in drivers:
                        a=components[aid]
                        if (a.mechanism!='ContactFirstStaticOccupancyV2' or port!='occupied' or
                                a.options.get('biological_owner')!=cid):
                            raise ContractError('Generic AC contact transform identity mismatch: '+cid)
                        traced.extend((l.source_component,l.source_port) for l in spec.links
                            if l.target_component==aid and l.target_port=='drive')
                    drivers=traced
                if not isolated and drivers != [(acs[cid]['driver'], 'activation')]:
                    raise ContractError('Generic AC direction changed: ' + cid)
                annotation[cid] = dict(acs[cid], biological_subtype_identified=False,
                    validation_status='not_independently_validated', fit_enabled=False)
                if candidate_ac:annotation[cid]['candidate_equations']=candidate
            if c.mechanism == 'HistoryConductanceRGC' or cid in targets:
                current_boundary = (isolated and boundary.get('mode') == 'current_clamp' and
                                    c.mechanism == 'HistoryConductanceRGCCurrentClampBoundary')
                if cid not in targets or (c.mechanism != 'HistoryConductanceRGC' and not current_boundary):
                    raise ContractError('Unapproved RGC target/mechanism: ' + cid)
                revision=getattr(self,'parameter_proposal_revisions',{}).get(cid,{})
                expected_profile=revision.get('new_profile_id','effective_'+cid)
                if c.parameter_profile_id != expected_profile:
                    raise ContractError('RGC parameter profile changed: ' + cid)
                # Follow the actual conductance adapters instead of inferring
                # synaptic direction from the word "crossover" in a name.
                adapters = [l.source_component for l in spec.links if
                            l.target_component == cid and l.target_port == 'inhibition']
                drivers = []
                for adapter in adapters:
                    if components[adapter].mechanism != 'SynapticConductance':
                        raise ContractError('Unapproved RGC inhibition adapter: ' + adapter)
                    sources = [l.source_component for l in spec.links if
                               l.target_component == adapter and l.target_port == 'activation']
                    if len(sources) != 1 or components[adapter].parameter_profile_id != (
                            'effective_' + cid + '_inhibitory_' + sources[0]):
                        raise ContractError('RGC inhibitory parameter binding changed: ' + cid)
                    drivers.extend(sources)
                expected = [name for name, row in acs.items() if cid in row['targets']]
                if not isolated and sorted(drivers) != sorted(expected):
                    raise ContractError('Generic AC target route changed: ' + cid)
                annotation[cid] = dict(targets[cid], validation_status='pending_data_comparison',
                    internal_mechanisms_validated_by_output=False, fit_enabled=False)
                if c.options.get('current_clamp_boundary') and not isolated:
                    raise ContractError('Current-clamp input is restricted to the isolated RGC experiment')
            if cid == 'AII_AC':
                annotation[cid] = dict(identity='separate_AII_pathway_retained',
                    validation_status='topology_source_informed_dynamics_uncalibrated', fit_enabled=False)
        # Generic AC units may only deliver their retained inhibition through
        # the existing adapters; an extra BC/AC target requires explicit revision.
        for link in spec.links:
            if link.source_component not in acs:
                continue
            target = components[link.target_component]
            downstream = [l.target_component for l in spec.links if
                          l.source_component == target.component_id and l.target_port == 'inhibition']
            if (link.source_port != 'activation' or link.target_port != 'activation' or
                    target.mechanism != 'SynapticConductance' or not downstream or
                    any(cid not in acs[link.source_component]['targets'] for cid in downstream)):
                raise ContractError('Unapproved generic AC output route: ' + link.source_component)
        return replace(spec, components=tuple(replace(c, options=dict(c.options,
            modeling_scope=annotation[c.component_id])) if c.component_id in annotation else c
            for c in spec.components))

    def manifest(self, spec=None):
        result = dict(self.data, path=self.path, sha256=self.sha256,
                      physiology_validated=False, automatic_fitting_enabled=False)
        if getattr(self,'parameter_proposal_revisions',None):
            result.update(parameter_proposal_revisions=self.parameter_proposal_revisions,
                base_scope_sha256=self.base_scope_sha256,system_acceptance=False,production_default=False)
        if spec is not None:
            candidates={c.component_id:c.options['candidate_dynamics'] for c in spec.components
                        if 'candidate_dynamics' in c.options}
            if candidates:
                result.update(default_equation_revision=result['equation_revision'],
                    equation_revision=next(iter(candidates.values()))['route'], candidate_targets=candidates,
                    candidate_only=True, core_equations_replaced=True, default_route_changed=False,
                    replacement_scope='selected cone BC/generic AC only; coneHC/rod/RBC/AII/RGC retained')
            present = {c.component_id for c in spec.components}
            result['assembled_rgc_targets'] = {
                name: dict(row, assembled=name in present, validation_status='pending_data_comparison')
                for name, row in self.data['rgc_targets'].items()}
            result['generic_ac_aii_overlap_review'] = self.overlap_review(spec)
        return result

    def overlap_review(self, spec):
        links = {(l.source_component, l.source_port, l.target_component, l.target_port) for l in spec.links}
        gaps = sorted({tuple(sorted((g.left_component, g.right_component))) for g in spec.gap_junctions})
        # Shared ON drive is an overlap of influence, not proof that two modeled
        # synapses implement the same biological effect with the same magnitude.
        aii_targets = sorted({post for syn in ('AII_FMB_glycine', 'AII_DB1_glycine')
                              for pre, port, post, target_port in links
                              if pre == syn and port == 'conductance' and target_port == 'inhibition'})
        return dict(decision='retain_all_routes_no_proven_duplicate',
            generic_OFF_to_ON=self.data['generic_ac_units']['local_crossover_AC'],
            AII_inhibitory_targets=aii_targets, reciprocal_gap_pairs=[list(p) for p in gaps if 'AII_AC' in p],
            unresolved='ON-driven local/widefield residual inhibition and AII-mediated OFF suppression may overlap in influence; contributions have not been separated by data',
            quantitative_attribution='not_measured_no_ablation_run', removed_routes=[])
