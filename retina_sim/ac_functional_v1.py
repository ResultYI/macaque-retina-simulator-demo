"""Research-only AC facade; retain the original conditional upstream and physics.

The ordinary feedforward block can be factored under an explicit BC replay.
AII is always observed in the complete original bidirectionally coupled domain.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.special import expit
from .bc_functional_v1 import BCFunctionalV1
from .contracts import ContractError,canonical_hash
from .group_inner_dynamics import key,AC_MECH,MEMBRANE,BC_MECH
from .ac_vs_model import MECHANISM,LEGACY_S

class ACFunctionalV1:
    def __init__(self,root,configuration):
        self.root=Path(root);self.configuration=json.loads((self.root/configuration).read_text(encoding='utf8')) if isinstance(configuration,(str,Path)) else dict(configuration)
        cfg=self.configuration
        if cfg['schema']!='AC-functional-effective-layer-1' or cfg['new_kinetic_mechanism'] is not False:raise ContractError('Explicit retained AC functional identity required')
        self.bc=BCFunctionalV1(self.root,cfg['BC_configuration']);self.model=self.bc.model;self.assembly=self.bc.assembly
        if self.bc.identity!=cfg['BC_layer_identity']:raise ContractError('Frozen BC interface changed')
        if canonical_hash(self.bc.seed['inner_parameters'])!=cfg['frozen_inner_parameters_sha256']:raise ContractError('Frozen upstream/inner scientific parameters changed')
        self.channels=self.model.channels;self.ordinary=tuple(c['pool'] for c in self.channels);self.pools=(*self.ordinary,LEGACY_S)
        if set(self.model.ac_populations)!=set(self.pools):raise ContractError('Actual existing AC coverage differs')
        # Factoring is permitted only for these existing feedforward state blocks.
        ordinary_set=set(self.pools)
        if any(r.pre in ordinary_set and r.post not in self.model.counts for r in self.model.routes):raise ContractError('Unknown ordinary AC output')
        if any(r.pre in ordinary_set and r.post in (*self.bc.populations,'AII_AC',*ordinary_set) for r in self.model.routes):raise ContractError('Ordinary AC block has upstream/recurrent feedback; cannot factor replay')
        self.state_keys=tuple(k for k in self.model.state_keys if k[0] in ordinary_set);self.slices={};at=0
        for k in self.state_keys:self.slices[k]=slice(at,at+self.model.sizes[k]);at+=self.model.sizes[k]
        self.nordinary=at;self.fields={}
        def field(name,pop,unit,meaning,count):self.fields[name]=dict(population=pop,unit=unit,meaning=meaning,count=count)
        for c in self.channels:
            p=c['pool'];n=self.model.counts[p]
            for port,u,meaning in [('input_effective_release','dimensionless','Original member-weighted BC fractional mean'),('input_conductance','S','Original BC-to-effective-pool excitation'),('voltage','V','Existing effective AC V'),('gate_probability','dimensionless','Original sigmoid(V)'),('inhibition_gate','dimensionless','Existing s, not vesicle release or intrinsic adaptation')]:field(p+'__'+port,p,u,meaning,n)
        for port,u,meaning in [('input_effective_release','dimensionless','Original diffuse ON BC mean'),('target_gate','dimensionless','Original bounded target'),('inhibition_gate','dimensionless','Existing h'),('output_activity','dimensionless','Original output_scale times h')]:field(LEGACY_S+'__'+port,LEGACY_S,u,meaning,self.model.counts[LEGACY_S])
        for pop in ('AII_AC','rod_BC'):
            for port,u,meaning in [('voltage','V','Existing membrane voltage'),('release_proxy','dimensionless','Original source release proxy'),('vesicle_resource_x','dimensionless','Existing ribbon resource'),('expected_event_rate','events/s','Original expected output events'),('gap_current','A','Original gap plus original held exterior')]:
                if pop=='rod_BC' and port=='gap_current':continue
                field(pop+'__'+port,pop,u,meaning,self.model.counts[pop])
        for component in ('RBC_AII_AMPA','AII_FMB_glycine','AII_DB1_glycine'):
            c=self.model.contacts[component];post=c['route'].post;n=self.model.counts[post]
            field(component+'__conductance',post,'S','Original grouped physical conductance, summed once',n)
            field(component+'__current',post,'A','This route only: g*(original reversal-actual target V)',n)
        for target in (*[c['target'] for c in self.channels],'small_bistratified_S_ON'):
            field(target+'__ordinary_inhibitory_conductance',target,'S','Selected actual functional channel/legacy route only',self.model.counts[target])
            field(target+'__ordinary_inhibitory_current',target,'A','This route only, with actual existing RGC voltage; no RGC model change',self.model.counts[target])
        for pop in self.model.gap_names:
            if pop=='AII_AC':continue
            field('AII_gap_to_'+pop+'__current',pop,'A','Offdiagonal AII-to-ON contribution: sum g_ij*(V_AII-V_BC); no held offset',self.model.counts[pop])
        self.identity=canonical_hash(dict(configuration=cfg,BC_identity=self.bc.identity,source_inner=self.model.identity,fields=self.fields,ordinary_state_keys=self.state_keys,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

    def ordinary_initial(self):return np.concatenate([self.bc.initial()[k] for k in self.state_keys])
    def ordinary_unpack(self,y):return {k:y[s] for k,s in self.slices.items()}
    def ordinary_outputs(self,y,bc_release):
        state=self.ordinary_unpack(y);m=self.model;out={}
        for c in self.channels:
            p=c['pool'];r=m.mean[c['pre']+'_to_'+p]@bc_release[c['pre']];gE=m.par(p,'gE_base')+c['G_BA_S']*r
            V=state[key(p,MECHANISM,'V')];s=state[key(p,MECHANISM,'s')]
            out[p+'__input_effective_release']=r;out[p+'__input_conductance']=gE;out[p+'__voltage']=V;out[p+'__gate_probability']=expit((V-m.par(p,'theta'))/m.par(p,'k'));out[p+'__inhibition_gate']=s
        pop=LEGACY_S;h=state[key(pop,AC_MECH,'h')]
        drive=sum((m.mean[r.name]@bc_release[r.pre] for r in m.routes if r.post==pop),start=np.zeros(m.counts[pop]))
        out[pop+'__input_effective_release']=drive;out[pop+'__target_gate']=expit(m.par(pop,'input_slope')*(drive-m.par(pop,'reference_input')));out[pop+'__inhibition_gate']=h;out[pop+'__output_activity']=m.par(pop,'output_scale')*h
        for c in self.channels:
            target=c['target'];route=c['pool']+'_to_'+target
            out[target+'__ordinary_inhibitory_conductance']=m.members[target]*m.parameters['route_gains'][route]*(m.mean[route]@out[c['pool']+'__inhibition_gate'])
        target='small_bistratified_S_ON';route=LEGACY_S+'_to_'+target
        out[target+'__ordinary_inhibitory_conductance']=m.members[target]*m.parameters['route_gains'][route]*(m.mean[route]@out[LEGACY_S+'__output_activity'])
        return out

    def ordinary_rhs(self,y,bc_release):
        state=self.ordinary_unpack(y);out=self.ordinary_outputs(y,bc_release);m=self.model;d={}
        for c in self.channels:
            pop=c['pool'];V=out[pop+'__voltage'];s=out[pop+'__inhibition_gate'];p=out[pop+'__gate_probability'];gE=out[pop+'__input_conductance']
            d[key(pop,MECHANISM,'V')]=(m.gL[pop]*(m.par(pop,'EL')-V)+gE*(m.par(pop,'EE')-V))/m.C[pop]
            d[key(pop,MECHANISM,'s')]=m.par(pop,'alpha')*p*(1-s)-m.par(pop,'beta')*s
        pop=LEGACY_S;d[key(pop,AC_MECH,'h')]=(out[pop+'__target_gate']-out[pop+'__inhibition_gate'])/m.par(pop,'tau_A_seconds')
        return np.concatenate([d[k] for k in self.state_keys])

    def observe_coupled(self,t,state,cone_release):
        m=self.model;out=m.outputs(t,state,cone_release);y=np.concatenate([state[k] for k in self.state_keys]);r={p:out[p,'effective_release'] for p in self.bc.populations if p!='rod_BC'};result=self.ordinary_outputs(y,r)
        for pop in ('AII_AC','rod_BC'):
            for port,value in [('voltage',out[pop,'voltage']),('release_proxy',out[pop,'release']),('vesicle_resource_x',state[key(pop,MEMBRANE,'x')]),('expected_event_rate',out[pop,'event_rate'])]:result[pop+'__'+port]=np.array(value,copy=True)
            if pop=='AII_AC':result[pop+'__gap_current']=np.array(out[pop,'gap_current'],copy=True)
        for component in ('RBC_AII_AMPA','AII_FMB_glycine','AII_DB1_glycine'):
            c=m.contacts[component];post=c['route'].post;g=out[component,'conductance'];V=out[post,'voltage'];E=m.par(post,'EE' if component=='RBC_AII_AMPA' else 'EI')
            result[component+'__conductance']=g.copy();result[component+'__current']=g*(E-V)
        for target in (*[c['target'] for c in self.channels],'small_bistratified_S_ON'):
            result[target+'__ordinary_inhibitory_current']=result[target+'__ordinary_inhibitory_conductance']*(m.par(target,'EI')-out[target,'voltage'])
        G=m.gap.tocsr();ai=m.gap_slices['AII_AC'];VA=out['AII_AC','voltage']
        for pop in m.gap_names:
            if pop=='AII_AC':continue
            K=-G[m.gap_slices[pop],ai];VB=out[pop,'voltage'];result['AII_gap_to_'+pop+'__current']=K@VA-np.asarray(K.sum(axis=1)).ravel()*VB
        if set(result)!=set(self.fields) or any(not np.isfinite(v).all() for v in result.values()):raise ContractError('AC declared readouts missing/nonfinite')
        return result
