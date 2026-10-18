"""Continuous equation transcription of the complete Fig6A/7 native-scale profile.

Not Fortran ARMA numerical reproduction, physical mV/pA, or H2/S calibration.
The spatial pooling extension is explicit and reduces to the scalar source loop.
"""
from pathlib import Path
from dataclasses import dataclass
import hashlib
from types import MappingProxyType
import numpy as np
from scipy import sparse
from scipy.optimize import brentq,least_squares
from scipy.special import expit
from .contracts import ContractError,Quantity,canonical_hash
from .mechanism_contracts import MechanismStateSpec
from .assembly import RuntimeComponent,PortSpec,TimeContract

KEY='VanHateren2005NativeConeH1'
PROFILE='VH2005_Fig6A7_native_source'
SOURCE='VH2005_source_supplement'
PARAMETER_UNITS={'tau_R':'ms','tau_E':'ms','c_beta':'ms^-1','k_beta':'ms^-1/td','n_X':'dimensionless','n_C':'dimensionless','tau_C':'ms','a_C':'vH_native_current^-1','tau_m':'ms','gamma':'dimensionless','tau_is':'ms','a_is':'vH_native_conductance/vH_native_voltage^gamma','g_t':'vH_native_voltage','V_k':'vH_native_voltage','V_n':'vH_native_voltage','tau_1':'ms','tau_2':'ms','tau_h':'ms','V_I':'vH_native_voltage','mu':'dimensionless','tau_a':'ms','observation_delay':'ms'}
SOURCE_VALUES={'tau_R':.49,'tau_E':16.8,'c_beta':2.80e-3,'k_beta':1.63e-4,'n_X':1.,'n_C':4.,'tau_C':2.89,'a_C':9.08e-2,'tau_m':4.,'gamma':.678,'tau_is':56.9,'a_is':7.09e-2,'g_t':151.1,'V_k':-10.,'V_n':3.,'tau_1':4.,'tau_2':4.,'tau_h':20.,'V_I':19.7,'mu':.733,'tau_a':250.,'observation_delay':2.82}

@dataclass(frozen=True)
class _NativeSignalContext:
    owner: object
    blocks: tuple
    signals: tuple

class NativeConeH1:
    def __init__(self,parameters,cones,hcs,A,B,background,*,pooling_policy,geometry_frame,unsupported_type_policy='reject',observation_interval_seconds=None):
        if parameters.profile.profile_id!=PROFILE: raise ContractError('Exact vH native profile required')
        self.parameters=parameters; self.cones=cones; self.hcs=hcs
        if set(cones.cells.cell_type)!={'cone'} or set(hcs.cells.cell_type)!={'HC'}:raise ContractError('Native route requires explicit cone and HC cell types')
        if observation_interval_seconds is None or not np.isfinite(observation_interval_seconds) or observation_interval_seconds<=0:raise ContractError('Explicit numerical observation sampling interval required for timestamped source delay')
        self.observation_interval_seconds=float(observation_interval_seconds)
        self.tag=cones.cells.evidence_tags[0]
        unsupported=set(cones.cells.subtype)-{'L','M'} or set(hcs.cells.subtype)-{'H1'}
        if unsupported and not ((unsupported_type_policy=='software_fixture_shared_H1_kinetics' and self.tag=='synthetic_fixture' and parameters.run_mode.value=='numeric_fixture') or (unsupported_type_policy=='project_approximation_shared_H1_kinetics' and parameters.run_mode.value=='component_reference')): raise ContractError('H1/L-M source profile does not calibrate S/H2 kinetics; an explicit software-only extension is required')
        self.approximations=[{'id':'shared_H1_kinetics_project_approximation' if unsupported_type_policy=='project_approximation_shared_H1_kinetics' else 'shared_H1_kinetics_software_fixture_only','affected_cone_ids':[i for i,t in zip(cones.cells.cell_id,cones.cells.subtype) if t=='S'],'affected_hc_ids':[i for i,t in zip(hcs.cells.cell_id,hcs.cells.subtype) if t!='H1'],'why':'No independent S/H2 kinetic calibration. Explicit approved shared-H1 project approximation, source values untouched; synthetic mode remains software only. Not macaque S/H2 measurement.','alternatives':['Missing-binding strict biological configuration remains blocked','Separate S/H2 calibration required']} ] if unsupported else []
        if set(cones.cells.evidence_tags)|set(hcs.cells.evidence_tags)!={self.tag}: raise ContractError('Native evidence class mismatch')
        self.p={r.parameter_id.removeprefix('VH2005_'):float(r.value) for r in parameters.records}
        if self.p!=SOURCE_VALUES: raise ContractError('Native full profile must not mix/tune source values')
        self.p=MappingProxyType(self.p)
        self.nc=len(cones.cells.cell_id); self.nh=len(hcs.cells.cell_id)
        self.A=sparse.csr_matrix(A,dtype=float,copy=True); self.B=sparse.csr_matrix(B,dtype=float,copy=True)
        for matrix,shape in ((self.A,(self.nh,self.nc)),(self.B,(self.nc,self.nh))):
            if matrix.shape!=shape or np.any(matrix.data<0) or not np.all(np.isfinite(matrix.data)) or not np.allclose(np.asarray(matrix.sum(axis=1)).ravel(),1,rtol=0,atol=1e-12): raise ContractError('Native spatial extension needs nonnegative unit-DC A/B with explicit rows=post, columns=pre')
        for matrix in (self.A,self.B):
            for a in (matrix.data,matrix.indices,matrix.indptr):a.setflags(write=False)
        if pooling_policy!='pool_Vbar_before_aI' or not geometry_frame: raise ContractError('Explicit native spatial approximation and frame required')
        self.background=background.require('td','native_retinal_illuminance',(self.nc,))
        if np.any(self.background<0) or background.evidence_tag!=self.tag: raise ContractError('Native background/calibration mismatch')
        self.pooling_policy=pooling_policy; self.geometry_frame=geometry_frame
        definitions=[('R',cones,'td','native_illuminance'),('E',cones,'td','native_illuminance'),('X',cones,'vH_native_precursor','native_precursor'),('C',cones,'vH_native_current','native_os_current'),('Vis',cones,'vH_native_voltage','native_relative_voltage'),('g_i',cones,'vH_native_conductance','native_conductance'),('Vbar',cones,'vH_native_voltage','native_relative_voltage'),('z1',cones,'vH_native_voltage','effective_transmitter_drive'),('z2',hcs,'vH_native_voltage','effective_transmitter_drive'),('Vh',hcs,'vH_native_voltage','native_relative_voltage')]
        self.state_specs=tuple(MechanismStateSpec(name,pop.cells.cell_id,(len(pop.cells.cell_id),),unit,meaning,'Source equilibrium equations; complete Fig6A7 profile') for name,pop,unit,meaning in definitions)
        self._initial=self.equilibrium(background)
    @property
    def model_hash(self):
        def csr(m): return {'data':m.data,'indices':m.indices,'indptr':m.indptr,'shape':m.shape}
        return canonical_hash({'source':SOURCE,'profile':self.parameters.parameter_hash,'A':csr(self.A),'B':csr(self.B),'pooling':self.pooling_policy,'approximations':self.approximations,'observation_interval_seconds':self.observation_interval_seconds,'frame':self.geometry_frame,'background':self.background,'implementation':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'integration':'continuous RHS in seconds; source ms derivatives multiplied by1000','observation_delay_role':'readout only, not feedback edge'})
    def equilibrium(self,background):
        L=background.require('td','native_retinal_illuminance',(self.nc,)); p=self.p
        beta=p['c_beta']+p['k_beta']*L
        current=np.array([brentq(lambda c:c-(1/(b*(1+(p['a_C']*c)**p['n_C'])))**p['n_X'],0,(1/b)**p['n_X'],xtol=1e-12) for b in beta])
        X=current**(1/p['n_X']); Vis=(current/p['a_is'])**(1/(1+p['gamma'])); gi=p['a_is']*Vis**p['gamma']; aI=(Vis/p['V_I'])**p['mu']
        def drive(h): return p['g_t']/aI*expit((Vis-self.B@h-p['V_k'])/p['V_n'])
        upper=np.asarray(self.A@(p['g_t']/aI)).ravel()
        if np.all(Vis==Vis[0]) and np.all(aI==aI[0]):
            scalar=brentq(lambda h:h-p['g_t']/aI[0]*expit((Vis[0]-h-p['V_k'])/p['V_n']),0,float(upper.max()),xtol=1e-12)
            h=np.full(self.nh,scalar)
        else:
            def jacobian(h):
                q=expit((Vis-self.B@h-p['V_k'])/p['V_n']);scale=p['g_t']/aI*q*(1-q)/p['V_n']
                return sparse.eye(self.nh,format='csr')+self.A@sparse.diags(scale)@self.B
            solution=least_squares(lambda h:h-self.A@drive(h),np.minimum(upper*.5,np.max(Vis)),jac=jacobian,tr_solver='lsmr',bounds=(np.zeros(self.nh),upper),xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=2000)
            if not solution.success:raise ContractError('Sparse native background equilibrium solver failed')
            h=solution.x
        if np.max(np.abs(h-self.A@drive(h)))>1e-8: raise ContractError('Native background equilibrium residual failed; no silent 100td fallback')
        z1=drive(h); z2=self.A@z1
        arrays=(L,L,X,current,Vis,gi,Vis,z1,z2,h)
        return tuple(Quantity(a,s.unit,s.meaning,self.tag) for a,s in zip(arrays,self.state_specs))
    def initial_conditions(self): return self._initial
    def evaluation_context(self,blocks):
        # Owned by one JointNetwork evaluation; never stored on the model.
        return _NativeSignalContext(self,blocks,self.signals(blocks))
    def signals(self,blocks,context=None):
        if isinstance(context,_NativeSignalContext) and context.owner is self and context.blocks is blocks:
            return context.signals
        R,E,X,C,Vis,gi,Vbar,z1,z2,h=blocks; p=self.p
        if np.any(gi<=0) or np.any(Vbar<=0) or np.any(Vis<0): raise ContractError('Native positive conductance/adaptation domain violated')
        aI=(Vbar/p['V_I'])**p['mu']; q=expit((Vis-self.B@h-p['V_k'])/p['V_n'])
        return X**p['n_X'],aI,q,q/aI
    def derivative(self,t,blocks,incoming):
        L=incoming['light'].require('td','native_retinal_illuminance',(self.nc,))
        if np.any(L<0): raise ContractError('Native illuminance must be nonnegative; no implicit contrast calibration')
        R,E,X,C,Vis,gi,Vbar,z1,z2,h=blocks; p=self.p
        if np.any(X<0) or np.any(C<0): raise ContractError('Native OS state outside nonnegative domain')
        I,aI,q,r=self.signals(blocks,incoming.get('__evaluation_context__')); pooled_aI=((self.A@Vbar)/p['V_I'])**p['mu']
        alpha=1/(1+(p['a_C']*C)**p['n_C']); beta=p['c_beta']+p['k_beta']*E
        d=((L-R)/p['tau_R'],(R-E)/p['tau_E'],alpha-beta*X,(I-C)/p['tau_C'],(I/gi-Vis)/p['tau_m'],(p['a_is']*Vis**p['gamma']-gi)/p['tau_is'],(Vis-Vbar)/p['tau_a'],(p['g_t']*r-z1)/p['tau_1'],(self.A@z1-z2)/(p['tau_2']*pooled_aI),(z2-h)/(p['tau_h']*pooled_aI))
        return tuple(Quantity(1000*v,s.unit+'/s',s.meaning+'_derivative',self.tag) for v,s in zip(d,self.state_specs))
    def observe(self,port,t,blocks,incoming):
        if port in ('q','release'):
            I,aI,q,r=self.signals(blocks,incoming.get('__evaluation_context__'))
            return Quantity(q if port=='q' else r,'dimensionless','sigmoid_activation' if port=='q' else 'effective_synaptic_drive',self.tag)
        if port=='Ios': return Quantity(self.signals(blocks,incoming.get('__evaluation_context__'))[0],'vH_native_current','native_outer_segment_current',self.tag)
        s=next(s for s in self.state_specs if s.state_name==port)
        return Quantity(blocks[self.state_specs.index(s)],s.unit,s.meaning,self.tag)

def vh_component(spec,populations,sources,mode):
    from .components import _binding,_implementation
    p,pops,tag=_binding(spec,populations,sources,mode,('cone','h1'))
    required=('A','B','background_td','pooling_policy','geometry_frame','observation_interval_seconds')
    if any(k not in spec.options for k in required): raise ContractError('Native route requires explicit sparse A/B, declared background, pooling policy and geometry frame')
    model=NativeConeH1(p,pops['cone'],pops['h1'],**{k:spec.options[k] for k in required if k not in ('background_td',)},background=spec.options['background_td'],unsupported_type_policy=spec.options.get('unsupported_type_policy','reject'))
    outputs={s.state_name:PortSpec(s.state_name,s.unit,s.meaning,s.cell_ids,tag,absolute_or_deviation='native effective source scale') for s in model.state_specs}
    ids=pops['cone'].cells.cell_id
    outputs.update({'q':PortSpec('q','dimensionless','sigmoid_activation',ids,tag),'release':PortSpec('release','dimensionless','effective_synaptic_drive',ids,tag,sign_convention='increases with native synaptic depolarization; q/aI may exceed1'),'Ios':PortSpec('Ios','vH_native_current','native_outer_segment_current',ids,tag)})
    return RuntimeComponent(spec.component_id,spec.mechanism,p,model,{'light':PortSpec('light','td','native_retinal_illuminance',ids,tag)},outputs,model.state_specs,model.initial_conditions,model.derivative,model.observe,{k:() for k in outputs},TimeContract('continuous'),('cone_outer_segment','cone_native_synapse','HC_H1')+(('HC_H2',) if set(pops['h1'].cells.subtype)-{'H1'} else ()),tuple(pops.values()),tag,(SOURCE,),hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),(('cone.Vis','cone.z1'),('cone.z1','h1.z2'),('h1.Vh','cone.release')))
