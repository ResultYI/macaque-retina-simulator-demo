"""Named finite-instance zero-contact extension; original VH source is untouched.

Only the admissible feedback topology and its background solver differ. Signals,
continuous RHS, outputs and all source parameters are inherited unchanged.
Empty feedback rows denote an explicitly certified empty finite partner sum,
not a measured absence of HC innervation or an exterior background closure.
"""
from pathlib import Path
from types import MappingProxyType
import hashlib
import numpy as np
from scipy import sparse
from scipy.optimize import brentq,least_squares
from scipy.special import expit
from .contracts import ContractError,Quantity,canonical_hash
from .mechanism_contracts import MechanismStateSpec
from .assembly import RuntimeComponent,PortSpec,TimeContract
from .vh2005 import NativeConeH1,PROFILE,SOURCE,SOURCE_VALUES

KEY='VanHateren2005FiniteContactZeroFeedbackApproximation'
POLICY='finite_geometry_zero_contact_execution_v1'
# Reviewed 2026-10-15: per-evaluation context only; source math/constructor unchanged.
SOURCE_SHA256='d4f867c75e3c9cdb44c0073caad2de61b2bf383883a59d6e9c5c34b4dfddfa4a'

def graph_hash(cone_ids,hc_ids,A,B):
    def csr(m):
        m=sparse.csr_matrix(m);return {'shape':m.shape,'data':m.data,'indices':m.indices,'indptr':m.indptr}
    return canonical_hash({'cone_ids':tuple(cone_ids),'HC_ids':tuple(hc_ids),'A':csr(A),'B':csr(B)})

def certify_zero_contacts(cones,hc,A,B,contacts,configuration,active_bounds):
    """Independently recheck every empty column against full fixed-disk support."""
    from .spatial_profiles import finite_support_coverage
    A=sparse.csr_matrix(A);B=sparse.csr_matrix(B)
    if ((A.T!=0)!=(B!=0)).nnz:raise ContractError('ZERO_CONTACT_GRAPH_RECIPROCITY_FAILED')
    p=cones.pedicle_positions
    p.require('mm','pedicle_position',frame='OPL')
    h=hc.dendritic_positions
    if h is None or h.coordinate_frame!='OPL':raise ContractError('Zero-contact check requires physical OPL HC anchors')
    if configuration.get('arbor_profile_id')!='W2000_equal_area_bounded_edges_047_104':
        raise ContractError('Zero-contact execution requires the bounded fixed-disk contract')
    band=configuration['anchor_applicability_band_mm'];rows=[]
    missing=np.flatnonzero(np.diff(B.indptr)==0);ht=np.asarray(hc.cells.subtype)
    for i in missing:
        point=p.values[i,:2];ctype=cones.cells.subtype[i];checks={}
        for subtype in ('H1','H2'):
            if subtype=='H1' and ctype=='S':continue
            footprint=configuration['footprints'][subtype]
            if footprint['kind']!='disk':raise ContractError('Zero-contact support certificate only implements the named fixed disk')
            reach=min(float(footprint['radius_mm']),float(configuration['final_contact_distance_mm']))
            complete=bool(finite_support_coverage(point[None],reach,[active_bounds])[0])
            r=float(np.linalg.norm(point));applicability=band[0]<=r-reach and r+reach<=band[1]
            delta=np.linalg.norm(h.values[ht==subtype,:2]-point,axis=1)
            legal=int(np.sum(delta<=reach+1e-12))
            if not complete or not applicability:raise ContractError('ZERO_CONTACT_SUPPORT_INCOMPLETE: '+cones.cells.cell_id[i])
            if legal or A[:,i].nnz:raise ContractError('ZERO_CONTACT_QUERY_DISAGREEMENT: '+cones.cells.cell_id[i])
            checks[subtype]={'query_radius_mm':reach,'anchor_candidate_domain_complete':complete,
                'anchor_applicability_complete':applicability,'legal_partner_count_independent_recheck':legal,
                'nearest_eligible_anchor_distance_mm':float(delta.min()) if len(delta) else None}
        rows.append({'cone_index':int(i),'cone_id':cones.cells.cell_id[i],'cone_type':ctype,
            'OPL_pedicle_mm':point.tolist(),'is_owner':bool(contacts['paired_cones'][i]['input_owner']),
            'support_complete':True,'forward_column_degree':int(A[:,i].nnz),'feedback_row_degree':int(B[i].nnz),
            'cause':'zero legal finite disk/type partners in complete generated anchor query; not a missing domain',
            'independent_type_support_checks':checks,'feedback_contribution':'exact empty sum 0 in native scale',
            'physiological_zero_innervation_claim':False})
    return {'policy_id':POLICY,'support_complete':True,'rows':rows,
        'zero_cone_indices':missing.tolist(),'zero_cone_ids':[r['cone_id'] for r in rows],
        'full_geometry_graph_hash':graph_hash(cones.cells.cell_id,hc.cells.cell_id,A,B),
        'forward_feedback_topology_reciprocal':True,'cone_population_pruned':False,
        'evidence_class':'project_approximation','source_RHS_changes':False,'RNG_calls':0,
        'boundary':'existing finite instance; recurrent exterior closure remains uncertified'}


class FiniteContactConeH1(NativeConeH1):
    def __init__(self,parameters,cones,hcs,A,B,background,*,pooling_policy,geometry_frame,
                 zero_contact_certificate,unsupported_type_policy='reject',observation_interval_seconds=None):
        # Source-constructor checks are copied explicitly, not bypassed by fake
        # unit rows or mutations of an already-initialized source instance.
        if hashlib.sha256(Path(__file__).with_name('vh2005.py').read_bytes()).hexdigest()!=SOURCE_SHA256:
            raise ContractError('Finite adapter must be reviewed against changed VH source implementation')
        if parameters.profile.profile_id!=PROFILE:raise ContractError('Exact vH native profile required')
        self.parameters=parameters;self.cones=cones;self.hcs=hcs
        if set(cones.cells.cell_type)!={'cone'} or set(hcs.cells.cell_type)!={'HC'}:raise ContractError('Explicit cone/HC types required')
        if observation_interval_seconds is None or not np.isfinite(observation_interval_seconds) or observation_interval_seconds<=0:raise ContractError('Explicit positive observation interval required')
        self.observation_interval_seconds=float(observation_interval_seconds);self.tag=cones.cells.evidence_tags[0]
        unsupported=set(cones.cells.subtype)-{'L','M'} or set(hcs.cells.subtype)-{'H1'}
        if unsupported and not ((unsupported_type_policy=='software_fixture_shared_H1_kinetics' and self.tag=='synthetic_fixture' and parameters.run_mode.value=='numeric_fixture') or
            (unsupported_type_policy=='project_approximation_shared_H1_kinetics' and parameters.run_mode.value=='component_reference')):
            raise ContractError('S/H2 kinetics require explicit uncalibrated shared-H1 extension')
        if set(cones.cells.evidence_tags)|set(hcs.cells.evidence_tags)!={self.tag}:raise ContractError('Native evidence class mismatch')
        values={r.parameter_id.removeprefix('VH2005_'):float(r.value) for r in parameters.records}
        if values!=SOURCE_VALUES:raise ContractError('Native full profile must not mix/tune source values')
        self.p=MappingProxyType(values);self.nc=len(cones.cells.cell_id);self.nh=len(hcs.cells.cell_id)
        self.A=sparse.csr_matrix(A,dtype=float,copy=True);self.B=sparse.csr_matrix(B,dtype=float,copy=True)
        for matrix,shape in ((self.A,(self.nh,self.nc)),(self.B,(self.nc,self.nh))):
            if matrix.shape!=shape or np.any(matrix.data<0) or not np.all(np.isfinite(matrix.data)):raise ContractError('Finite nonnegative native matrices with explicit dimensions required')
        asum=np.asarray(self.A.sum(1)).ravel();bsum=np.asarray(self.B.sum(1)).ravel()
        if not np.allclose(asum,1.,rtol=0,atol=1e-12):raise ContractError('Native HC forward rows must remain unit-DC')
        self.zero_feedback_indices=np.flatnonzero(np.diff(self.B.indptr)==0)
        nonzero=np.diff(self.B.indptr)>0
        if not np.allclose(bsum[nonzero],1.,rtol=0,atol=1e-12):raise ContractError('Nonempty native feedback rows must remain unit-DC')
        if ((self.A.T!=0)!=(self.B!=0)).nnz:raise ContractError('Finite native forward/feedback partner graph mismatch')
        c=zero_contact_certificate
        if (c.get('policy_id')!=POLICY or c.get('support_complete') is not True or
            c.get('zero_cone_indices')!=self.zero_feedback_indices.tolist() or
            c.get('zero_cone_ids')!=[cones.cells.cell_id[i] for i in self.zero_feedback_indices] or
            c.get('runtime_graph_hash')!=graph_hash(cones.cells.cell_id,hcs.cells.cell_id,self.A,self.B)):
            raise ContractError('Exact complete-support zero-contact certificate required')
        self.zero_contact_certificate=c
        self.approximations=[{'id':POLICY,'affected_cone_ids':c['zero_cone_ids'],
            'why':'Explicit authorized finite-instance empty partner sum; not biological absence, exterior feedback or measured morphology',
            'source_parameters_changed':False,'RNG_calls':0}]
        if unsupported:self.approximations.append({'id':'shared_H1_kinetics_project_approximation',
            'affected_cone_ids':[i for i,t in zip(cones.cells.cell_id,cones.cells.subtype) if t=='S'],
            'affected_hc_ids':[i for i,t in zip(hcs.cells.cell_id,hcs.cells.subtype) if t!='H1'],
            'why':'Source H1 parameters unchanged; S/H2 uncalibrated'})
        for matrix in (self.A,self.B):
            for a in (matrix.data,matrix.indices,matrix.indptr):a.setflags(write=False)
        if pooling_policy!='pool_Vbar_before_aI' or geometry_frame!='OPL':raise ContractError('Explicit source pooling and OPL frame required')
        self.pooling_policy=pooling_policy;self.geometry_frame=geometry_frame
        self.background=background.require('td','native_retinal_illuminance',(self.nc,))
        if np.any(self.background<0) or background.evidence_tag!=self.tag:raise ContractError('Native background/calibration mismatch')
        defs=[('R',cones,'td','native_illuminance'),('E',cones,'td','native_illuminance'),('X',cones,'vH_native_precursor','native_precursor'),('C',cones,'vH_native_current','native_os_current'),('Vis',cones,'vH_native_voltage','native_relative_voltage'),('g_i',cones,'vH_native_conductance','native_conductance'),('Vbar',cones,'vH_native_voltage','native_relative_voltage'),('z1',cones,'vH_native_voltage','effective_transmitter_drive'),('z2',hcs,'vH_native_voltage','effective_transmitter_drive'),('Vh',hcs,'vH_native_voltage','native_relative_voltage')]
        self.state_specs=tuple(MechanismStateSpec(name,pop.cells.cell_id,(len(pop.cells.cell_id),),unit,meaning,'Background equilibrium of exact finite graph; certified empty feedback sum where declared') for name,pop,unit,meaning in defs)
        self._initial=self.equilibrium(background)

    @property
    def model_hash(self):
        return canonical_hash({'original_source_model_contract':super().model_hash,
            'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'zero_contact_certificate':self.zero_contact_certificate})

    def equilibrium(self,background):
        if not len(self.zero_feedback_indices):
            values=super().equilibrium(background)
            self.background_equilibrium_audit={'method':'unchanged original native equilibrium for nonempty graph',
                'maximum_residual_native':float(np.max(np.abs(values[-1].values-self.A@values[-3].values))),
                'nfev':None,'whole_inner_network_steady_claim':False}
            return values
        L=background.require('td','native_retinal_illuminance',(self.nc,));p=self.p
        beta=p['c_beta']+p['k_beta']*L
        current=np.array([brentq(lambda c:c-(1/(b*(1+(p['a_C']*c)**p['n_C'])))**p['n_X'],0,(1/b)**p['n_X'],xtol=1e-12) for b in beta])
        X=current**(1/p['n_X']);Vis=(current/p['a_is'])**(1/(1+p['gamma']));gi=p['a_is']*Vis**p['gamma'];aI=(Vis/p['V_I'])**p['mu']
        def drive(h):return p['g_t']/aI*expit((Vis-self.B@h-p['V_k'])/p['V_n'])
        upper=np.asarray(self.A@(p['g_t']/aI)).ravel()
        def jacobian(h):
            q=expit((Vis-self.B@h-p['V_k'])/p['V_n']);scale=p['g_t']/aI*q*(1-q)/p['V_n']
            return sparse.eye(self.nh,format='csr')+self.A@sparse.diags(scale)@self.B
        if self.nh==1:
            # scipy's sparse two-dimensional trust-region subproblem cannot
            # handle a one-state fixture; solve the exact same scalar residual.
            scalar,info=brentq(lambda x:float(x-(self.A@drive(np.array([x])))[0]),0.,float(upper[0]),xtol=1e-12,full_output=True)
            h=np.array([scalar]);success=info.converged;nfev=info.function_calls
        else:
            sol=least_squares(lambda h:h-self.A@drive(h),np.minimum(upper*.5,np.max(Vis)),jac=jacobian,
                tr_solver='lsmr',bounds=(np.zeros(self.nh),upper),xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=2000)
            h=sol.x;success=sol.success;nfev=sol.nfev
        residual=float(np.max(np.abs(h-self.A@drive(h))))
        if not success or residual>1e-8:raise ContractError('Finite zero-feedback background equilibrium residual failed')
        self.background_equilibrium_audit={'method':'same sparse nonlinear feedback equilibrium; uniform scalar shortcut disabled for zero rows',
            'maximum_residual_native':residual,'nfev':int(nfev),'zero_feedback_release':'same expit((Vis-V_k)/V_n) divided by actual aI',
            'whole_inner_network_steady_claim':False}
        z1=drive(h);z2=self.A@z1
        return tuple(Quantity(a,s.unit,s.meaning,self.tag) for a,s in zip((L,L,X,current,Vis,gi,Vis,z1,z2,h),self.state_specs))


def finite_feedback_component(spec,populations,sources,mode):
    from .components import _binding
    p,pops,tag=_binding(spec,populations,sources,mode,('cone','h1'))
    keys=('A','B','pooling_policy','geometry_frame','observation_interval_seconds','zero_contact_certificate')
    model=FiniteContactConeH1(p,pops['cone'],pops['h1'],background=spec.options['background_td'],
        **{k:spec.options[k] for k in keys},unsupported_type_policy=spec.options.get('unsupported_type_policy','reject'))
    outputs={s.state_name:PortSpec(s.state_name,s.unit,s.meaning,s.cell_ids,tag,absolute_or_deviation='native effective source scale') for s in model.state_specs}
    ids=pops['cone'].cells.cell_id
    outputs.update({'q':PortSpec('q','dimensionless','sigmoid_activation',ids,tag),
        'release':PortSpec('release','dimensionless','effective_synaptic_drive',ids,tag),
        'Ios':PortSpec('Ios','vH_native_current','native_outer_segment_current',ids,tag)})
    return RuntimeComponent(spec.component_id,spec.mechanism,p,model,{'light':PortSpec('light','td','native_retinal_illuminance',ids,tag)},
        outputs,model.state_specs,model.initial_conditions,model.derivative,model.observe,{k:() for k in outputs},TimeContract('continuous'),
        ('cone_outer_segment','cone_native_synapse','HC_H1')+(('HC_H2',) if set(pops['h1'].cells.subtype)-{'H1'} else ()),
        tuple(pops.values()),tag,(SOURCE,),hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        (('cone.Vis','cone.z1'),('cone.z1','h1.z2'),('h1.Vh','cone.release')))
