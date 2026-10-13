"""Source-aware assembly contracts, component registration and sparse compilation."""
from dataclasses import dataclass,field
from pathlib import Path
import inspect,json,hashlib
import numpy as np
from scipy import sparse
from .contracts import ContractError,RunMode,Quantity,StateSpec,canonical_hash
from .graph import ConnectionTable
from .provenance import Registry,require_runnable
from .spatial import Population

class AssemblyError(ContractError):
    def __init__(self,issues):
        self.issues=tuple(issues)
        super().__init__('; '.join(self.issues))

@dataclass(frozen=True)
class PortSpec:
    name: str
    unit: str | None
    meaning: str | None
    cell_ids: tuple
    evidence_tag: str
    sign_convention: str = 'explicit_quantity_convention'
    absolute_or_deviation: str = 'declared_by_meaning'
    reference_state_id: str | None = None
    support: dict = field(default_factory=dict)
    def validate(self,q):
        if not self.unit or not self.meaning: raise ContractError('Unbound signal contract: '+self.name)
        q.require(self.unit,self.meaning,(len(self.cell_ids),))
        if q.evidence_tag!=self.evidence_tag: raise ContractError('Port evidence mismatch: '+self.name)

@dataclass(frozen=True)
class TimeContract:
    kind: str
    fixed_step_seconds: float | None = None
    update_rule: str = 'joint RHS at current time/state'
    def __post_init__(self):
        if self.kind not in ('continuous','author_discrete','causal_history'): raise ContractError('Unknown time contract')
        if self.kind=='continuous' and self.fixed_step_seconds is not None: raise ContractError('Continuous interface cannot hide a fixed step')
        if self.kind in ('author_discrete','causal_history') and (self.fixed_step_seconds is None or self.fixed_step_seconds<=0 or not np.isfinite(self.fixed_step_seconds)):
            raise ContractError('Author discrete step must be explicit')

@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    mechanism: str | None
    population_roles: dict
    parameter_profile_id: str | None
    options: dict = field(default_factory=dict)
    required_layer: str | None = None
    required_ports: dict = field(default_factory=dict)

@dataclass(frozen=True)
class PortLink:
    source_component: str
    source_port: str
    target_component: str
    target_port: str
    connections: ConnectionTable
    geometry_frame: str
    delay_prehistory: Quantity | None = None
    delay_interpolation: str | None = None
    delay_sampling_seconds: float | None = None
    delay_evidence: object | None = None

@dataclass(frozen=True)
class ExternalInput:
    channel: str
    target_component: str
    target_port: str

@dataclass(frozen=True)
class GapJunction:
    left_component: str
    left_cell_ids: tuple
    right_component: str
    right_cell_ids: tuple
    conductance_evidence: object
    geometry_frame: str = 'IPL'

    def validate(self,components):
        n=len(self.left_cell_ids)
        if not n or len(self.right_cell_ids)!=n or self.geometry_frame not in ('IPL','OPL'):raise ContractError('Explicit IPL/OPL electrical contacts required')
        r=self.conductance_evidence
        g=r.value.require('S','gap_junction_conductance',(n,))
        if np.any(g<=0):raise ContractError('Gap conductance must be positive')
        for cid,ids in ((self.left_component,self.left_cell_ids),(self.right_component,self.right_cell_ids)):
            if cid not in components:raise ContractError('Unknown gap component')
            c=components[cid]
            if not hasattr(c.native_model,'membrane_capacitance') or c.outputs.get('voltage') is None:raise ContractError('Electrical coupling requires actual membrane states/C')
            if not set(ids)<=set(c.outputs['voltage'].cell_ids):raise ContractError('Electrical contact cell IDs mismatch')
        a=components[self.left_component];b=components[self.right_component]
        if a.evidence_tag!=b.evidence_tag or r.value.evidence_tag!=a.evidence_tag:raise ContractError('Gap evidence mismatch')
        if r.evidence_class=='software_fixture' and a.parameters.evidence_class!='software_fixture':raise ContractError('Engineering gap cannot satisfy biological binding')

@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    populations: tuple
    components: tuple
    links: tuple = ()
    external_inputs: tuple = ()
    required_mechanisms: tuple = ()
    intended_regions: tuple = ('center','transition','periphery')
    scope: str = 'explicit_component_assembly'
    algebraic_closure: dict | None = None
    clock_policy: dict | None = None
    gap_junctions: tuple = ()

    @classmethod
    def read_target(cls,path):
        d=json.loads(Path(path).read_text(encoding='utf-8'))
        if d['schema_version']!='architecture-1.0': raise ContractError('Unknown architecture schema')
        return cls(d['model_id'],(),tuple(ComponentSpec(**c) for c in d['components']),required_mechanisms=tuple(d['required_mechanisms']),intended_regions=tuple(d['intended_regions']),scope=d['scope'],clock_policy=d.get('clock_policy'))

@dataclass
class RuntimeComponent:
    component_id: str
    mechanism: str
    parameters: object
    native_model: object
    inputs: dict
    outputs: dict
    state_specs: tuple
    initial: object
    derivative: object
    observe: object
    output_dependencies: dict
    time_contract: TimeContract
    layers: tuple
    populations: tuple
    evidence_tag: str
    source_ids: tuple
    implementation_hash: str
    internal_dependencies: tuple = ()
    binding_options_hash: str | None = None

    @property
    def identity(self):
        return canonical_hash({'id':self.component_id,'mechanism':self.mechanism,'parameters':self.parameters.parameter_hash,'binding_options_hash':self.binding_options_hash,'native':getattr(self.native_model,'model_hash',None),'inputs':self.inputs,'outputs':self.outputs,'states':self.state_specs,'dependencies':self.output_dependencies,'time':self.time_contract,'population_hashes':[p._population_hash for p in self.populations],'source_ids':self.source_ids,'implementation':self.implementation_hash})

class ComponentRegistry:
    def __init__(self): self._factories={}
    def register(self,name,factory):
        if not name or not callable(factory) or name in self._factories: raise ContractError('Invalid/duplicate component registration')
        self._factories[name]=factory
    def instantiate(self,spec,populations,sources,mode):
        if spec.mechanism not in self._factories: raise AssemblyError([f'{spec.component_id}: unbound mechanism {spec.mechanism}'])
        c=self._factories[spec.mechanism](spec,populations,sources,mode)
        def options_record(value):
            if sparse.issparse(value):
                m=sparse.csr_matrix(value);return {'sparse_CSR':True,'shape':m.shape,'data':m.data,'indices':m.indices,'indptr':m.indptr}
            if isinstance(value,dict):return {k:options_record(v) for k,v in value.items()}
            if isinstance(value,(tuple,list)):return [options_record(x) for x in value]
            return value
        c.binding_options_hash=canonical_hash(options_record(spec.options))
        if spec.required_layer and spec.required_layer not in c.layers:
            phenotype_layers={
                'ResidualBC_ON':{'midget_BC_ON','diffuse_BC_ON','S_ON_BC'},
                'ResidualBC_OFF':{'midget_BC_OFF','diffuse_BC_OFF'},
                'ResidualAC':{'local_feedforward_AC','local_crossover_AC','widefield_AC'},
                'HistoryConductanceRGC':{'midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON'}}
            if spec.required_layer not in phenotype_layers.get(spec.mechanism,set()):raise ContractError('Requested target layer is not supported by this executable mechanism')
            # An explicit phenotype label binds configured geometry/profiles; its
            # physiological correctness is independently unvalidated, never inferred.
            c.layers=c.layers+(spec.required_layer,)
        if c.component_id!=spec.component_id or c.mechanism!=spec.mechanism: raise ContractError('Factory identity mismatch')
        if not callable(c.initial) or (c.time_contract.kind!='author_discrete' and not callable(c.observe)): raise ContractError('Missing executable component contract')
        if c.time_contract.kind!='author_discrete' and not callable(c.derivative): raise ContractError('Component missing inter-sample derivative')
        if set(c.output_dependencies)!=set(c.outputs): raise ContractError('Output dependency declarations incomplete')
        for name,dependencies in c.output_dependencies.items():
            if any(x not in c.inputs for x in dependencies): raise ContractError('Unknown algebraic input dependency')
        for p in tuple(c.inputs.values())+tuple(c.outputs.values()):
            if not p.unit or not p.meaning or not p.cell_ids: raise ContractError('Executable ports require units, meaning and actual cells')
        for source_id in c.source_ids:
            if source_id not in sources.sources: raise ContractError('Component source not registered: '+source_id)
        return c

@dataclass(frozen=True)
class CompiledLink:
    specification: PortLink
    matrix: object
    delayed_operator: object | None = None

@dataclass(frozen=True)
class OutputBlock:
    nodes: tuple
    cyclic: bool

@dataclass
class CompiledRetina:
    spec: ModelSpec
    components: dict
    populations: dict
    links: tuple
    output_order: tuple
    assembly_hash: str
    run_mode: RunMode
    clock: dict
    def runner(self):
        from .network import NetworkRunner
        return NetworkRunner(self)
    def initialize(self,solver_config=None,time_seconds=0): return self.runner().initialize(solver_config,time_seconds)
    def advance(self,state,channels,stop,solver_config=None,sample_times=None,recording_plan=None): return self.runner().advance(state,channels,stop,solver_config,sample_times,recording_plan)
    def step(self,channels,state,dt_seconds,**kwargs): return self.advance(state,channels,state.time_seconds+dt_seconds,**kwargs)
    def stream(self,state,channels,stop,chunk_seconds,recorder,**kwargs): return self.runner().stream(state,channels,stop,chunk_seconds,recorder,**kwargs)
    def checkpoint(self,state): return self.runner().checkpoint(state)
    def restore(self,payload): return self.runner().restore(payload)

class RetinaBuilder:
    def __init__(self,sources,components=None,max_compile_bytes=512*1024*1024):
        self.sources=sources
        self.max_compile_bytes=max_compile_bytes
        if components is None:
            from .components import reference_component_registry
            components=reference_component_registry()
        self.registry=components

    def estimate(self,spec,sample_count=2):
        population_count=sum(len(p.cells.cell_id) for p in spec.populations)
        edges=sum(len(x.connections.edge_id) for x in spec.links)+sum(len(x.left_cell_ids) for x in spec.gap_junctions); states=0; history=0; scratch=0
        pops={p.population_id:p for p in spec.populations}
        for c in spec.components:
            sizes={r:len(pops[p].cells.cell_id) for r,p in c.population_roles.items() if p in pops}
            if c.mechanism=='SomaratnaFreeman2025GaussianHCReference':
                n=sizes.get('cone',0);m=sizes.get('g_off',0);edges+=2*n*n+2*n*m;scratch+=64*n*n;states+=4*n+2*m
            elif c.mechanism=='Chen2024PeripheralLMCurrent': states+=5*sizes.get('cone',0)
            elif c.mechanism in ('VanHateren2005NativeConeH1','VanHateren2005FiniteContactZeroFeedbackApproximation'):
                states+=8*sizes.get('cone',0)+2*sizes.get('h1',0)
                for key in ('A','B'):
                    if key in c.options: edges+=sparse.csr_matrix(c.options[key]).nnz
                if 'A' in c.options and 'B' in c.options:
                    a=sparse.csc_matrix(c.options['A']);b=sparse.csr_matrix(c.options['B'])
                    if a.shape[1]!=b.shape[0]:raise AssemblyError(['Native A/B cone dimension mismatch'])
                    jacobian_edge_bound=int(np.dot(np.diff(a.indptr).astype(np.int64),np.diff(b.indptr).astype(np.int64)))+a.shape[0]
                    scratch+=64*jacobian_edge_bound
            elif c.mechanism in ('ResidualBC_ON','ResidualBC_OFF','ResidualAC','HistoryConductanceRGC'):
                profiles=getattr(self.sources,'mechanism_profiles',{})
                if c.parameter_profile_id in profiles:
                    p=profiles[c.parameter_profile_id]
                    length=sum(np.asarray(r.value.values).size for r in p.records if r.parameter_id in ('kernel','spike_history_kernel'))
                    history+=sizes.get('cells',0)*length;states+=sizes.get('cells',0)*(length+1)
            else: states+=sum(sizes.values())
        delay_steps=[x.delay_sampling_seconds for x in spec.links if x.delay_sampling_seconds is not None]
        for component in spec.components:
            if component.mechanism in ('VanHateren2005NativeConeH1','VanHateren2005FiniteContactZeroFeedbackApproximation') and 'observation_interval_seconds' in component.options:delay_steps.append(component.options['observation_interval_seconds'])
            profile=getattr(self.sources,'mechanism_profiles',{}).get(component.parameter_profile_id)
            if profile is not None:delay_steps.extend(float(r.value.values) for r in profile.records if r.parameter_id in ('sample_interval','spike_interval'))
        delay_steps.extend(float(d) for link in spec.links for d in link.connections.delay_seconds if d is not None and d>0)
        delay_bytes=0
        for link in spec.links:
            ds=[d for d in link.connections.delay_seconds if d is not None and d>0]
            if ds and delay_steps:
                source=next((c for c in spec.components if c.component_id==link.source_component),None)
                count=max((len(pops[pid].cells.cell_id) for pid in source.population_roles.values() if pid in pops),default=0) if source else 0
                samples=int(np.ceil(max(ds)/min(delay_steps)))*(4+2*len(set(ds)))+16
                delay_bytes+=samples*(count*16+32)
        scratch+=delay_bytes
        minimum=population_count*1024+edges*128+states*32+history*32+scratch
        return {'explicit_cells':population_count,'estimated_edges':edges,'estimated_state_scalars':states,'estimated_history_scalars':history,'estimated_compile_bytes':minimum,'estimated_full_recording_bytes':states*8*sample_count,'meaning':'Conservative engineering estimate; unknown bindings remain unestimated; no automatic density reduction'}

    def build(self,spec,mode=RunMode.REFERENCE): return self.compile(spec,mode)

    def describe(self,spec):
        from .plugins import BOUNDARIES,boundary_status
        population_ids={p.population_id for p in spec.populations}
        bound=[c.component_id for c in spec.components if c.mechanism in self.registry._factories and c.parameter_profile_id in self.sources.profiles and all(x in population_ids for x in c.population_roles.values()) and (c.mechanism not in BOUNDARIES or (c.mechanism,c.options.get('backend_id')) in self.registry._physiology_plugins)]
        missing=[{'component_id':c.component_id,'layer':c.required_layer,'mechanism':c.mechanism,'required_ports':c.required_ports} for c in spec.components if c.component_id not in bound]
        return {'model_id':spec.model_id,'declared_components':len(spec.components),'registered_operator_names':sorted(x for x in self.registry._factories if x not in BOUNDARIES),'plugin_boundary_status':boundary_status(self.registry),'bound_component_ids':bound,'unbound':missing,'executed_component_ids':[],'physiological_completion':False,'intended_regions':list(spec.intended_regions)}

    def compile(self,spec,mode=RunMode.REFERENCE):
        mode=RunMode(mode); issues=[]
        from .mechanism_contracts import validate_mechanism_evidence
        for profile in getattr(self.sources,'mechanism_profiles',{}).values():
            validate_mechanism_evidence(self.sources,profile)
        if spec.clock_policy is not None and (set(spec.clock_policy)!={'kind','base_step_seconds','sampled_output_hold'} or spec.clock_policy['base_step_seconds']<=0):raise AssemblyError(['Explicit shared clock policy invalid'])
        estimate=self.estimate(spec)
        if estimate['estimated_compile_bytes']>self.max_compile_bytes: raise AssemblyError([f'Resource estimate {estimate} exceeds compile budget {self.max_compile_bytes}; declare ROI/halo or raise budget, never silently thin cells'])
        if mode==RunMode.FULL_PHYSIOLOGY:
            path=self.sources.root/'requirements.json'
            require_runnable({'mode':mode.value},json.loads(path.read_text(encoding='utf-8')))
        populations={p.population_id:p for p in spec.populations}
        if len(populations)!=len(spec.populations): issues.append('Duplicate population ID')
        cells=[x for p in spec.populations for x in p.cells.cell_id]
        if len(cells)!=len(set(cells)): issues.append('Cell IDs must be globally unique across populations')
        names=[c.component_id for c in spec.components]
        if not names or len(set(names))!=len(names): issues.append('Components must be nonempty and uniquely named')
        for s in spec.components:
            if s.mechanism not in self.registry._factories: issues.append(f'{s.component_id}: mechanism unbound')
            if not s.parameter_profile_id: issues.append(f'{s.component_id}: parameter profile unbound')
            for role,pid in s.population_roles.items():
                if pid not in populations: issues.append(f'{s.component_id}.{role}: unknown population {pid}')
        if issues: raise AssemblyError(issues)
        from .mechanism_contracts import validate_population_evidence_domain
        for s in spec.components:
            profile=getattr(self.sources,'mechanism_profiles',{}).get(s.parameter_profile_id)
            if profile is not None:validate_population_evidence_domain(profile,[populations[pid] for pid in s.population_roles.values()])
        components={s.component_id:self.registry.instantiate(s,populations,self.sources,mode) for s in spec.components}
        for required in spec.required_mechanisms:
            if not any(required in c.layers for c in components.values()): issues.append('Required layer unbound: '+required)
        kinds={c.time_contract.kind for c in components.values()}
        intervals={c.time_contract.fixed_step_seconds for c in components.values() if c.time_contract.kind=='causal_history'}
        intervals.update(c.native_model.dt for c in components.values() if hasattr(c.native_model,'sample_spikes'))
        if len(intervals)>1 and spec.clock_policy is None: issues.append('Multi-rate histories require explicit shared clock/ZOH policy')
        if 'author_discrete' in kinds and len(components)!=1 and spec.clock_policy is None:
            issues.append('Author-discrete isolated execution retained unless mixed assembly supplies explicit clock/ZOH contract')
        for gap in spec.gap_junctions:gap.validate(components)
        contacts=set()
        for gap in spec.gap_junctions:
            for left,right in zip(gap.left_cell_ids,gap.right_cell_ids):
                key=tuple(sorted(((gap.left_component,left),(gap.right_component,right))))
                if key[0]==key[1] or key in contacts:raise ContractError('Duplicate/self electrical contact would double conductance')
                contacts.add(key)
        delay_steps=[x.delay_sampling_seconds for x in spec.links if x.delay_sampling_seconds is not None]
        delay_steps.extend(intervals)
        delay_steps.extend(c.native_model.observation_interval_seconds for c in components.values() if hasattr(c.native_model,'observation_interval_seconds'))
        if spec.clock_policy is not None:delay_steps.append(spec.clock_policy['base_step_seconds'])
        input_sources={}; compiled=[]; cell_frames={}
        for population in populations.values():
            frames={q.coordinate_frame for q in (population.cells.rf_position,population.cells.soma_position,population.anatomical_positions,population.receptive_field_positions,population.pedicle_positions,population.terminal_positions,population.dendritic_positions) if q is not None}
            for cell in population.cells.cell_id: cell_frames[cell]=frames
        for link in spec.links:
            if link.source_component not in components or link.target_component not in components: raise AssemblyError(['Link references unknown component'])
            pre=components[link.source_component].outputs.get(link.source_port)
            post=components[link.target_component].inputs.get(link.target_port)
            if pre is None or post is None: raise AssemblyError(['Link port does not exist'])
            if (pre.unit,pre.meaning)!=(post.unit,post.meaning): raise AssemblyError([f'Incompatible ports {pre.unit}/{pre.meaning} -> {post.unit}/{post.meaning}; explicit physiological transducer required'])
            if pre.evidence_tag!=post.evidence_tag: raise AssemblyError(['Connected ports require compatible explicit evidence class'])
            table=link.connections
            pi={v:i for i,v in enumerate(pre.cell_ids)}; po={v:i for i,v in enumerate(post.cell_ids)}
            if any(x not in pi for x in table.pre_cell_id) or any(x not in po for x in table.post_cell_id): raise AssemblyError(['Directed sparse edge endpoints do not match pre/output and post/input cells'])
            if len(set(zip(table.pre_cell_id,table.post_cell_id)))!=len(table.edge_id): raise AssemblyError(['Duplicate cell pair requires separate synapse/port operators'])
            if not link.geometry_frame: raise AssemblyError(['Every edge operator requires its coordinate frame'])
            for port in (pre,post):
                for cell in port.cell_ids:
                    if link.geometry_frame not in cell_frames.get(cell,set()): raise AssemblyError(['Edge frame unavailable for '+cell+': '+link.geometry_frame])
            if any(x<0 for x in table.weight): raise AssemblyError(['Pooling/contact weights must be nonnegative; receptor/current mechanisms own signs'])
            if any(x!='dimensionless' for x in table.weight_unit): raise AssemblyError(['Explicit coefficient operator supports dimensionless weights only; no unit conversion'])
            if any(x is None or not np.isfinite(x) or x<0 for x in table.delay_seconds): raise AssemblyError(['Every edge needs an explicit nonnegative delay in seconds'])
            delayed=any(x>0 for x in table.delay_seconds)
            if delayed:
                if link.delay_prehistory is None or link.delay_interpolation not in ('linear','zero_order_hold') or link.delay_sampling_seconds is None or not np.isfinite(link.delay_sampling_seconds) or link.delay_sampling_seconds<=0:raise AssemblyError(['Positive delays require explicit full-source prehistory, sampling step and interpolation'])
                pre.validate(link.delay_prehistory)
                from .mechanism_contracts import EvidenceValue
                evidence=link.delay_evidence
                if not isinstance(evidence,EvidenceValue):raise AssemblyError(['Nonzero connection delays require their own unit-bearing evidence entry'])
                literal=evidence.value.require('s','connection_delay',(len(table.edge_id),))
                if not np.array_equal(literal,np.array(table.delay_seconds)):raise AssemblyError(['Delay evidence values do not match the directed edge table'])
                if evidence.evidence_class=='software_fixture' and mode!=RunMode.NUMERIC_FIXTURE:raise AssemblyError(['Engineering delay values are synthetic-only'])
                if evidence.source_id is not None:
                    if evidence.source_id not in self.sources.sources:raise AssemblyError(['Delay evidence source is not registered'])
                    self.sources.sources[evidence.source_id].verify(self.sources.root)
            if any(x not in self.sources.profiles for x in table.source_profile_id): raise AssemblyError(['Edge profile not registered'])
            if mode==RunMode.NUMERIC_FIXTURE and any(x!='synthetic_fixture' for x in table.evidence_tags): raise AssemblyError(['Numeric edge evidence must be synthetic_fixture'])
            matrix=sparse.coo_matrix((table.weight,([po[x] for x in table.post_cell_id],[pi[x] for x in table.pre_cell_id])),shape=(len(po),len(pi))).tocsr()
            for a in (matrix.data,matrix.indices,matrix.indptr):a.setflags(write=False)
            operator=None
            if delayed:
                from .delays import DelayedEdgeOperator
                er=np.array([po[x] for x in table.post_cell_id]);ec=np.array([pi[x] for x in table.pre_cell_id]);ew=np.array(table.weight);ed=np.array(table.delay_seconds)
                for array in (er,ec,ew,ed):array.setflags(write=False)
                step=min([link.delay_sampling_seconds]+[float(x) for x in table.delay_seconds if x>0]+[float(x) for x in delay_steps]) if delay_steps else min(link.delay_sampling_seconds,float(ed[ed>0].min()))
                capacity=int(np.ceil(float(ed.max())/step))*(4+2*len(np.unique(ed[ed>0])))+16
                operator=DelayedEdgeOperator('link:'+str(len(compiled)),er,ec,ew,ed,matrix.shape,link.delay_sampling_seconds,link.delay_interpolation,link.delay_prehistory,capacity)
            compiled.append(CompiledLink(link,matrix,operator))
            input_sources.setdefault((link.target_component,link.target_port),[]).append(('delayed' if operator is not None and not operator.has_instantaneous_edges else 'link',(link.source_component,link.source_port)))
        channels=set()
        for external in spec.external_inputs:
            key=(external.target_component,external.target_port)
            if external.target_component not in components or external.target_port not in components[external.target_component].inputs: raise AssemblyError(['External input target missing'])
            if external.channel in channels: raise AssemblyError(['External channel must have one explicit target'])
            channels.add(external.channel)
            if key in input_sources: raise AssemblyError(['Cannot implicitly combine external and connected input'])
            input_sources[key]=[('external',external.channel)]
        for cid,c in components.items():
            for port in c.inputs:
                if (cid,port) not in input_sources: issues.append(f'{cid}.{port}: no bound input')
                elif all(kind in ('link','delayed') for kind,source in input_sources[(cid,port)]):
                    incoming=[x for x in compiled if x.specification.target_component==cid and x.specification.target_port==port]
                    totals=sum((np.asarray(x.matrix.sum(axis=1)).ravel() for x in incoming),np.zeros(len(c.inputs[port].cell_ids)))
                    if np.any(totals<=0):issues.append(f'{cid}.{port}: explicit positive contact/input support missing for some cells; no implicit zero drive')
        # Output feedthrough graph, distinct from state feedback in the simultaneous RHS.
        dependencies={}
        for cid,c in components.items():
            for port,inputs in c.output_dependencies.items():
                dependencies[(cid,port)]={src for inp in inputs for kind,src in input_sources.get((cid,inp),[]) if kind=='link'}
        from scipy.sparse.csgraph import connected_components
        nodes=list(dependencies); idx={x:i for i,x in enumerate(nodes)}
        rows=[idx[dst] for dst,deps in dependencies.items() for src in deps]; cols=[idx[src] for deps in dependencies.values() for src in deps]
        graph=sparse.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(nodes),len(nodes))).tocsr()
        _,labels=connected_components(graph,directed=True,connection='strong')
        groups={int(i):tuple(x for x in nodes if labels[idx[x]]==i) for i in set(labels)}
        group_deps={i:{int(labels[idx[src]]) for dst in group for src in dependencies[dst] if labels[idx[src]]!=i} for i,group in groups.items()}
        order=[]; remaining={k:set(v) for k,v in group_deps.items()}
        while remaining:
            ready=sorted(k for k,v in remaining.items() if not v)
            if not ready: raise ContractError('Condensed dependency graph inconsistent')
            for k in ready:
                cyclic=len(groups[k])>1 or any(x in dependencies[x] for x in groups[k])
                if cyclic and spec.algebraic_closure is None: issues.append('Instantaneous algebraic cycle requires explicit joint closure; no sample-delay workaround')
                order.append(OutputBlock(groups[k],cyclic)); del remaining[k]
            for v in remaining.values(): v.difference_update(ready)
        if spec.algebraic_closure is not None:
            closure=spec.algebraic_closure
            if set(closure)!={'method','atol','rtol','max_evaluations'} or closure['method']!='hybr' or closure['atol']<=0 or closure['rtol']<=0 or closure['max_evaluations']<1: issues.append('Explicit closure method/tolerances/budget required')
        if issues: raise AssemblyError(issues)
        from .delays import clock_contract
        clock=clock_contract(components,compiled,spec.clock_policy)
        identity=canonical_hash({'model_id':spec.model_id,'components':{k:c.identity for k,c in components.items()},'populations':[p.record() for p in populations.values()],'links':spec.links,'external':spec.external_inputs,'order':order,'closure':spec.algebraic_closure,'gap_junctions':spec.gap_junctions,'clock':clock,'mode':mode,'required':spec.required_mechanisms,'scope':spec.scope,'compiler_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        return CompiledRetina(spec,components,populations,tuple(compiled),tuple(order),identity,mode,clock)
