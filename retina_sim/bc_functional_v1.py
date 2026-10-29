"""Research-only typed BC-layer interface over the unchanged source dynamics.

ConeBC terminal output and RodBC expected vesicle output have different units.
This facade keeps that distinction, exact group capacities, all coupling, and
the original source clocks; it is not a production factory or calibration.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from .contracts import ContractError,canonical_hash
from .group_time_storage import load_bundle
from .hc_lateral_interface import context
from .group_time_monitor import frozen_budget
from .spatial_group_model import BC

def _read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def _matrix_id(m):
    a=m.tocsr();return canonical_hash(dict(shape=a.shape,data=a.data,indices=a.indices,indptr=a.indptr))

class BCFunctionalV1:
    """All five retained coneBC types plus the existing RodBC source pathway."""
    def __init__(self,root,configuration):
        from tools.validate_h1_h2_background import assembly_from_saved
        self.root=Path(root);self.configuration=_read(self.root/configuration) if isinstance(configuration,(str,Path)) else dict(configuration)
        cfg=self.configuration
        if cfg['schema']!='BC-functional-effective-layer-1' or cfg['S_ON_input']!='retained_current' or cfg['availability_D_enabled'] is not False:
            raise ContractError('Explicit retained BCfunctionalv1 identity required; no silent hypothesis adoption')
        self.seed=load_bundle(self.root/cfg['upstream_seed'],expected_kind='source_construction_seed')['payload']
        if self.seed['joint_background_initialization']['initialized_model_id']!=cfg['upstream_initialized_model']:raise ContractError('New ConeHC input identity differs')
        if canonical_hash(self.seed['inner_parameters'])!=cfg['frozen_inner_parameters_sha256']:raise ContractError('Retained BC or associated parameters changed')
        base_seed=load_bundle(self.root/'reports/h1_h2_joint_background_init_20261028_v1/validation/seed',expected_kind='source_construction_seed')['payload']
        base,_=assembly_from_saved(base_seed);HC=_read(self.root/'configurations/research/HC_homotypic_lateral_only_v2.json')
        self.assembly=context(base,self.seed,HC,self.root);self.assembly.parameter_binding_id=self.seed['lateral_hc_interface']['binding_id']
        self.model=self.assembly.runtime_factories['inner'](self.assembly,self.seed['inner_parameters'])
        if _matrix_id(self.model.mean['cones_to_S_ON_BC'])!=cfg['retained_S_ON_input_sha256']:raise ContractError('Experimental75/25 operator must remain disabled')
        self.background=np.load(self.root/cfg['upstream_release_background'])['cone_release'].copy()
        self.rod_background=float(self.seed['input_reference']['rod_background_rstar_s'])
        self.protocol=_read(self.root/cfg['validation_protocol']);self.origin=float(self.seed['time_seconds'])
        self.populations=tuple(BC)+('rod_BC',);self.fields={};self.types={}
        def field(name,pop,unit,meaning):self.fields[name]=dict(population=pop,unit=unit,meaning=meaning,layout='actual_group_rows',count=self.model.counts[pop])
        for pop in BC:
            for name,unit,meaning in [('input_native_release','dimensionless','native Cone release convex mean, not micromolar glutamate'),
                ('generator_activation','dimensionless','actual causal FIR/softplus generator'),('voltage','V','physical terminal voltage'),
                ('unadapted_release','dimensionless','unadapted sigmoid terminal state'),('adaptation_activity','dimensionless','existing release-domain adaptation state A'),
                ('effective_release','dimensionless','terminal output after original gain control')]:field(pop+'__'+name,pop,unit,meaning)
        for name,unit,meaning in [('input_rod_release_native_mean','dimensionless','contact-coefficient-weighted presynaptic rod release proxy'),
            ('input_conductance_total','S','actual grouped mGluR6/TRPM1 receptor conductance, applied once'),
            ('receptor_gate_mean','dimensionless','contact-coefficient-weighted original receptor gate'),('voltage','V','physical RodBC membrane voltage'),
            ('membrane_release_proxy','dimensionless','source sigmoid release state; not a coneBC generator'),
            ('vesicle_resource_x','dimensionless','original releasable-resource fraction, distinct from coneBC adaptation A'),
            ('effective_output_event_rate','events/s','original expected ribbon/vesicle rate for existing RBC-AII AMPA interface')]:field('rod_BC__'+name,'rod_BC',unit,meaning)
        field('AII_interface__voltage','AII_AC','V','unchanged coupled-interface voltage, not a new AC model')
        field('AII_interface__gap_current','AII_AC','A','unchanged electrical interface current including original held exterior')
        c=self.model.contacts['individual_rod_RBC_contacts'];self.rod_contacts=c
        self.rod_weight=np.bincount(c['post'],weights=c['coefficient'],minlength=self.model.counts['rod_BC'])
        self.rod_input_coverage=self.rod_weight>0
        for pop in self.populations:
            p=self.assembly.populations[pop]
            self.types[pop]=dict(group_ids=list(map(str,p.group_ids)),represented_members=p.counts.tolist(),
                groups=p.group_count,total_represented_members=int(p.counts.sum()),coordinate_role=p.coordinate_role,
                group_centroids_mm=p.centroid_mm.tolist(),fine_cell_identity_reconstructed=False)
        self.identity=canonical_hash(dict(configuration=cfg,upstream=self.seed['joint_background_initialization']['initialized_model_id'],
            source_inner_identity=self.model.identity,fields=self.fields,types=self.types,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

    def initial(self):return {k:np.array(self.seed['states'][k],copy=True) for k in self.model.state_keys}

    def assay_inputs(self,index):
        cone_factor=rod_factor=1.
        for e in self.protocol['cone_release_epochs']:
            if round(e['start_seconds']/.0001)<=index<round(e['stop_seconds']/.0001):cone_factor=float(e['factor'])
        for e in self.protocol['rod_light_epochs']:
            if round(e['start_seconds']/.0001)<=index<round(e['stop_seconds']/.0001):rod_factor=float(e['factor'])
        return self.background*cone_factor,np.full(self.model.counts['rods'],self.rod_background*rod_factor),cone_factor,rod_factor

    def commit(self,index,state,cone_release,rod_light):
        bc=index%10==0;before=state
        nxt,_,events=self.model.commit_ticks(self.origin+index*.0001,before,
            cone_release=cone_release if bc else None,rod_light=rod_light,rod_tick=True,bc_tick=bc,rgc_tick=False)
        if events:raise ContractError('BCfunctionalv1 does not sample RGC spike events')
        if bc:nxt.update(self.model._probability_tick_updates(before))
        return nxt

    def rhs(self,t,y):return self.model.pack(self.model.rhs(t,self.model.unpack(y,copy=False)))

    def observe(self,t,state,cone_release):
        out=self.model.outputs(t,state,cone_release);result={}
        for pop in BC:
            for name,port in [('input_native_release','drive'),('generator_activation','generator_activation'),('voltage','voltage'),
                ('unadapted_release','unadapted_release'),('adaptation_activity','adaptation_activity'),('effective_release','effective_release')]:result[pop+'__'+name]=np.array(out[pop,port],copy=True)
        c=self.rod_contacts;weight=self.rod_weight;count=self.model.counts['rod_BC']
        def mean(v):
            total=np.bincount(c['post'],weights=c['coefficient']*v,minlength=count)
            return np.divide(total,weight,out=np.zeros(count),where=self.rod_input_coverage)
        result['rod_BC__input_rod_release_native_mean']=mean(out['rods','release'][c['pre']])
        result['rod_BC__input_conductance_total']=np.array(out['individual_rod_RBC_contacts','conductance'],copy=True)
        result['rod_BC__receptor_gate_mean']=mean(state['rods','individual_rod_RBC_contacts','m'])
        result['rod_BC__voltage']=np.array(out['rod_BC','voltage'],copy=True)
        result['rod_BC__membrane_release_proxy']=np.array(out['rod_BC','release'],copy=True)
        result['rod_BC__vesicle_resource_x']=np.array(state['rod_BC','ReceptorConductanceMembrane','x'],copy=True)
        result['rod_BC__effective_output_event_rate']=np.array(out['rod_BC','event_rate'],copy=True)
        result['AII_interface__voltage']=np.array(out['AII_AC','voltage'],copy=True)
        result['AII_interface__gap_current']=np.array(out['AII_AC','gap_current'],copy=True)
        for name,value in result.items():
            if value.shape!=(self.fields[name]['count'],) or not np.isfinite(value).all():raise ContractError('Finite complete typed BC output required: '+name)
        return result

    def initial_checks(self):
        state=self.initial();rhs=self.model.rhs(self.origin,state,self.background);specs={(b.population,b.mechanism,b.name):b for b in self.model.blocks};norms=[]
        for key,value in rhs.items():
            budget=frozen_budget(state[key],specs[key].unit)[2]
            norms.append(float(np.max(abs(value)/(budget/.0001),initial=0)))
        nxt,_,_=self.model.commit_ticks(self.origin,state,cone_release=self.background,bc_tick=True)
        changes={':'.join(k):float(np.max(abs(nxt[k]-state[k]),initial=0)) for k in state if k[1]=='HistoryBipolarTerminal' and (k[2].startswith('history_') or k[2]=='activation')}
        return dict(max_initial_frozen_RHS_norm=max(norms),first_BC_history_changes=changes,
            passed=max(norms)<=1 and not any(changes.values()),baseline_fixed=True,parameters_sha256=canonical_hash(self.seed['inner_parameters']),
            gap_matrix_sha256=_matrix_id(self.model.gap),gap_offset_sha256=canonical_hash(self.model.gap_boundary_product),
            conditional_initial_point_only=True,background_qualified=False,FIT_READY=False)
