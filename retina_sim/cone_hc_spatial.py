"""Configured source-bound Cone/H1/H2 research build and native CPU execution.

The first supported asset is the existing fine-cell local diagnostic. Broader
group parent construction remains separate; no scheduler or tissue is rebuilt.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time
import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import expit
from .contracts import ContractError, Quantity, canonical_hash
from .disk_tables import FrozenDiskTable
from .h1_h2_effective import H1H2Native, cone_fixed_point, matrix_id
from .hc_research_native import joint_equilibrium
from .hc_minimal_diagnostic import bind_diagnostic, require_diagnostic_native
from .cone_hc_reduced_native import AnalyticREAnchor
from .hc_expansion_geometry import native_td_step
from .vh2005 import SOURCE_VALUES

FUNCTIONAL_KERNEL = 'functional_exponential_shared_finite_support'
LEGACY_CHECKPOINT_IMPLEMENTATION = '1aaeb8e5338a935802e9e859cad28e4094c70a10268503612cd38bcee96e2261'


def select_spatial_A(uniform, biexponential, hc_types, mode):
    """Select H1 rows only; H2 retains its independent original uniform row."""
    if mode not in ('uniform','biexponential'): raise ContractError('Unknown Cone-HC spatial mode')
    u,b=sparse.csr_matrix(uniform),sparse.csr_matrix(biexponential)
    types=np.asarray(hc_types)
    if u.shape!=b.shape or types.shape!=(u.shape[0],) or not set(types)<={'H1','H2'}:
        raise ContractError('Explicit spatial operator dimensions and HC types required')
    if mode=='uniform': return u.copy()
    return (sparse.diags((types=='H1').astype(float))@b+sparse.diags((types=='H2').astype(float))@u).tocsr()


class ConeHCSpatialModel:
    def __init__(self, repo_root, configuration):
        self.repo_root=Path(repo_root).resolve()
        path=Path(configuration)
        if not path.is_absolute():path=self.repo_root/path
        self.config=json.loads(path.read_bytes());cfg=self.config
        if (cfg.get('schema')!='cone_hc_spatial_research' or cfg.get('research_only') is not True
                or cfg.get('production_qualified') is not False or cfg.get('H2_mode')!='uniform'
                or cfg.get('boundary_mode')!='held_background'):
            raise ContractError('Explicit research-only spatial configuration required')
        if set(cfg['H1_parameters'])!={'lambda_s_mm','lambda_l_mm','w_s'} or cfg['H1_parameters']!=dict(lambda_s_mm=.020,lambda_l_mm=.300,w_s=.15):
            raise ContractError('Frozen spatial parameters differ from supported source asset')
        self.kernel_definition=cfg.get('kernel_definition','legacy_contact_mask')
        if (self.kernel_definition not in ('legacy_contact_mask',FUNCTIONAL_KERNEL)
                or cfg.get('execution_mode','local_cpu')!='local_cpu'
                or (self.kernel_definition==FUNCTIONAL_KERNEL and cfg['spatial_mode']!='biexponential')):
            raise ContractError('Unsupported spatial kernel/execution mode; no uniform fallback')
        assets={}
        for name,binding in cfg['assets'].items():
            p=(self.repo_root/binding['path']).resolve()
            if not p.is_relative_to(self.repo_root) or hashlib.sha256(p.read_bytes()).hexdigest()!=binding['sha256']:
                raise ContractError('Spatial source asset changed: '+name)
            assets[name]=p
        with np.load(assets['geometry'],allow_pickle=False) as z:self.geometry={k:z[k].copy() for k in z.files}
        g=self.geometry;ci,hj=g['cone_parent_rows'],g['hc_parent_rows']
        receipt=json.loads(assets['source_receipt'].read_bytes())
        cones,hcs=FrozenDiskTable(self.repo_root,receipt['cone_table']),FrozenDiskTable(self.repo_root,receipt['hc_table'])
        for key,actual in [('cone_keys',cones.keys[ci]),('hc_keys',hcs.keys[hj]),
                           ('cone_pedicle_mm',cones.coordinates['opl_pedicle'][ci]),('hc_anchor_mm',hcs.coordinates['soma'][hj]),
                           ('cone_types',cones.extras['subtype'][ci].astype('U1'))]:
            if not np.array_equal(g[key],actual):raise ContractError('Fine source/member identity differs: '+key)
        self.cone_types=cones.extras['subtype'][ci].astype('U1');self.hc_types=hcs.extras['subtype'][hj].astype('U2')
        uniform=sparse.load_npz(assets['A_uniform']);biexp=sparse.load_npz(assets['A_biexponential'])
        self.A=select_spatial_A(uniform,biexp,self.hc_types,cfg['spatial_mode'])
        self.B=sparse.load_npz(assets['B_local']).tocsr()
        if self.A.shape!=(len(hj),len(ci)) or self.B.shape!=self.A.shape[::-1]:raise ContractError('Fine A/B layout differs')
        if (self.A[self.hc_types=='H1'][:,self.cone_types=='S'].nnz
                or self.B[self.cone_types=='S'][:,self.hc_types=='H1'].nnz
                or self.B[self.cone_types!='S'][:,self.hc_types=='H2'].nnz):raise ContractError('Typed spatial eligibility differs')
        with np.load(assets['held_mass']) as z:self.held_mass=z[cfg['spatial_mode']].copy()
        self.spatial_receipt=None
        if self.kernel_definition==FUNCTIONAL_KERNEL:
            if 'spatial_receipt' not in assets:
                raise ContractError('Functional matrix requires its precomputation receipt')
            self.spatial_receipt=json.loads(assets['spatial_receipt'].read_bytes())
            receipt=self.spatial_receipt
            if (receipt.get('kernel_definition')!=FUNCTIONAL_KERNEL or receipt.get('status')!='PRECOMPUTED'
                    or receipt.get('parameters')!=cfg['H1_parameters']
                    or receipt.get('A_identity')!=matrix_id(self.A)
                    or receipt.get('B_identity')!=matrix_id(self.B)
                    or receipt.get('crop_renormalized') is not False
                    or receipt.get('H2_mode')!='uniform'):
                raise ContractError('Functional matrix/source receipt differs')
        light=np.full(len(ci),cfg['input']['background_td']);cone0=cone_fixed_point(np.array([cfg['input']['background_td']]))
        p=SOURCE_VALUES;v0=float(cone0['Vis'][0]);ai0=(v0/p['V_I'])**p['mu']
        h0=brentq(lambda h:h-p['g_t']*expit((v0-h-p['V_k'])/p['V_n'])/ai0,0.,p['g_t']/ai0,xtol=1e-12)
        identity_record=dict(configuration=cfg,A=matrix_id(self.A),B=matrix_id(self.B),
            fine_cones=g['cone_keys'].astype('U'),fine_HC=g['hc_keys'].astype('U'),parameters=p,
            implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            native_implementation=hashlib.sha256(Path(__file__).with_name('hc_research_native.py').read_bytes()).hexdigest())
        self.identity=canonical_hash(identity_record)
        self.binding=bind_diagnostic(self.A,self.B,cone_parent_rows=ci,
            registered_unsupported_rows=np.asarray(cfg['registered_unsupported_cone_rows'],dtype=int),
            A_held_mass=self.held_mass,B_held_mass=np.zeros(len(ci)),exterior_release=h0/p['g_t'],
            exterior_Vbar=v0,exterior_h=h0,source_identity=self.identity)
        pop=lambda ids:SimpleNamespace(group_count=len(ids),group_ids=ids)
        cp=pop(g['cone_keys'].astype('U'));cp.recording_auxiliary_group_rows=np.array([],int);cp.recording_auxiliary_fine_rows=np.array([],int)
        self.h1_rows=np.flatnonzero(self.hc_types=='H1');self.h2_rows=np.flatnonzero(self.hc_types=='H2')
        pars=dict(tau_seconds=p['tau_h']*.001,input_scale=p['g_t'],feedback_scale=1.)
        assembly=SimpleNamespace(populations=dict(cones=cp,HC=pop(g['hc_keys'].astype('U'))),
            matrix=lambda name:self.A if 'cones_to_HC' in name else self.B,artifact_sha256=self.identity,
            h1_rows=self.h1_rows,h2_rows=self.h2_rows,h2_parameters=pars,
            layout=dict(scope='configured_source_bound_spatial_research',fine_cone_rows=ci,fine_HC_rows=hj),
            connection_identity=dict(identity=self.identity,spatial_mode=cfg['spatial_mode']))
        self.native=H1H2Native(assembly);self.native.boundary=self.binding['boundary']
        self.checkpoint_bindings={(self.identity,self.native.source_version)}
        if self.kernel_definition=='legacy_contact_mask':
            # Accept the unchanged frozen operator's previous code identity, never a different A.
            legacy=canonical_hash(dict(identity_record,implementation=LEGACY_CHECKPOINT_IMPLEMENTATION))
            native_identity=dict(self.native.identity,artifact_sha256=legacy,
                                 connections=dict(identity=legacy,spatial_mode=cfg['spatial_mode']))
            self.checkpoint_bindings.add((legacy,canonical_hash(native_identity)))
        eq=joint_equilibrium(self.A,self.B,self.h1_rows,self.h2_rows,light,pars,boundary=self.native.boundary)
        self.y0=np.concatenate([eq['cone'][k] for k in ('X','C','Vis','g_i','Vbar')]+[eq['w'],eq['z2'],eq['h'],np.array([])])
        self.light=light;self.start_anchor=AnalyticREAnchor(0.,light,light,light,cfg['input']['onset_seconds'],0)
        self.background_rhs=float(np.max(abs(self.native.rhs(self.y0,0.,self.start_anchor))))
        self.permission=require_diagnostic_native(self.binding,input_available=np.ones(len(ci),bool),
            feedback_geometry_complete=np.ones(len(ci),bool),background_max_rhs=self.background_rhs)
        self.state_layout=dict(names=list(self.native.slices),sizes=list(self.native.sizes),
            cone_voltage_unit='vH_native_voltage',HC_voltage_unit='vH_native_voltage',
            release_unit='dimensionless',release_meaning='effective_synaptic_drive')

    def runner(self, *, execution_mode='local_cpu'):
        if execution_mode!='local_cpu':
            raise ContractError('This spatial asset supports local_cpu only; no uniform fallback')
        return ConeHCSpatialRunner(self)


class ConeHCSpatialRunner:
    """CPU research runner using the unchanged native RHS and analytic input epochs."""
    def __init__(self,model):
        self.model=model;self.time=0.;self.y=model.y0.copy();self.anchor=model.start_anchor
        self.metrics=dict(wall_seconds=0.,nfev=0);self.segments=[]

    def advance(self, stop, sample_times):
        m=self.model;cfg=m.config;times=np.asarray(sample_times,float)
        if (not self.time<stop<=cfg['input']['offset_seconds'] or times.ndim!=1 or not len(times)
                or times[0]<self.time or times[-1]>stop or np.any(np.diff(times)<=0)):
            raise ContractError('Explicit increasing sample interval inside configured input required')
        packed=[];stamps=[];clock=time.perf_counter();onset=cfg['input']['onset_seconds']
        while self.time<stop:
            if self.time==onset and self.anchor.input_epoch==0:
                spec=dict(cfg['input']);bg=spec.pop('background_td')
                light=native_td_step(m.geometry['cone_IS_mm'],bg,time_seconds=self.time,**spec)
                R,E=self.anchor.at(self.time)
                self.anchor=AnalyticREAnchor(self.time,R,E,light,cfg['input']['offset_seconds'],1)
            end=min(stop,self.anchor.input_stop_seconds)
            sol=solve_ivp(lambda t,y:m.native.rhs(y,t,self.anchor),(self.time,end),self.y,
                method=cfg['solver']['method'],rtol=cfg['solver']['rtol'],atol=cfg['solver']['atol'],
                max_step=cfg['solver']['max_step_seconds'],dense_output=True)
            if not sol.success:raise ContractError(sol.message)
            self.segments.append((self.time,end,sol.sol,self.anchor))
            mask=(times>=self.time)&(times<=end)
            if stamps:mask&=times>stamps[-1]
            for t in times[mask]:
                value=sol.sol(t);m.native._validate(value,self.anchor,float(t));packed.append(value);stamps.append(float(t))
            self.y=sol.y[:,-1];self.time=float(end);self.metrics['nfev']+=sol.nfev
        self.metrics['wall_seconds']+=time.perf_counter()-clock
        y=np.column_stack(packed);h=y[m.native.slices['Vh']];feedback=m.B@h+m.native.boundary['B_h'][:,None]
        p=SOURCE_VALUES;release=expit((y[m.native.slices['Vis']]-feedback-p['V_k'])/p['V_n'])/(y[m.native.slices['Vbar']]/p['V_I'])**p['mu']
        core=m.geometry['core_local'];target=int(m.geometry['target_hc_local'])
        pooled_Vbar=m.A@y[m.native.slices['Vbar']]+m.native.boundary['A_Vbar'][:,None]
        return dict(t_seconds=np.asarray(stamps),y=y,H1_target=h[target],release_core=release[core],
            release_all=release,H1_all=h,feedback_core=feedback[core],
            Vbar_core=y[m.native.slices['Vbar']][core],
            aI_core=(y[m.native.slices['Vbar']][core]/p['V_I'])**p['mu'],
            pooled_Vbar_target=pooled_Vbar[target],pooled_aI_target=(pooled_Vbar[target]/p['V_I'])**p['mu'],
            pooled_release_target=(m.A@release+m.native.boundary['A_release'][:,None])[target])

    def release(self):
        values=self.model.native.evaluate(self.y,self.time,self.anchor).outputs['release']
        return Quantity(values,'dimensionless','effective_synaptic_drive','source_bound_research')

    def release_at(self,t):
        if t==self.time:return self.release()
        segment=next((s for s in self.segments if s[0]<=t<=s[1]),None)
        if segment is None:raise ContractError('Release query outside accepted native intervals')
        _,_,curve,anchor=segment
        values=self.model.native.evaluate(curve(t),t,anchor).outputs['release']
        return Quantity(values,'dimensionless','effective_synaptic_drive','source_bound_research')

    def checkpoint_payload(self):
        R,E=self.anchor.at(self.time)
        return dict(schema='cone_hc_spatial_research_checkpoint',model_identity=self.model.identity,
            native_identity=self.model.native.source_version,time=self.time,y=self.y.copy(),R=R,E=E,
            light=self.anchor.light.copy(),input_stop=self.anchor.input_stop_seconds,epoch=self.anchor.input_epoch,
            state_layout=self.model.state_layout)

    @classmethod
    def from_checkpoint(cls,model,payload):
        if (payload.get('schema')!='cone_hc_spatial_research_checkpoint'
                or (payload.get('model_identity'),payload.get('native_identity')) not in model.checkpoint_bindings
                or payload.get('state_layout')!=model.state_layout):
            raise ContractError('Spatial checkpoint mode, source, parameters or state layout differ')
        r=cls(model);r.time=float(payload['time']);r.y=np.asarray(payload['y'],float).copy()
        if not 0<=r.time<payload['input_stop']<=model.config['input']['offset_seconds']:
            raise ContractError('Checkpoint time outside registered input epoch')
        r.anchor=AnalyticREAnchor(r.time,payload['R'],payload['E'],payload['light'],payload['input_stop'],int(payload['epoch']))
        model.native._validate(r.y,r.anchor,r.time);return r


class FineMemberReleaseProjection:
    """Existing all-member mean for a subset's release perturbation at a group port."""
    def __init__(self,model,group_population,group_reference,*,target_identity):
        rows=model.geometry['cone_parent_rows'];mapping=np.asarray(group_population.fine_to_group)[rows]
        valid=mapping>=0;counts=np.asarray(group_population.counts)
        if (not valid.any() or np.any(mapping[valid]>=len(counts)) or np.any(counts<=0)
                or np.asarray(group_reference).shape!=(len(counts),)):
            raise ContractError('Explicit existing fine-member group mapping required')
        self.matrix=sparse.csr_matrix((1./counts[mapping[valid]],(mapping[valid],np.flatnonzero(valid))),
                                      shape=(len(counts),len(rows)))
        self.reference=np.asarray(group_reference,float).copy()
        self.native_reference=model.native.evaluate(model.y0,0.,model.start_anchor).outputs['release'].copy()
        self.mapped_fine_count=int(valid.sum())
        self.identity=canonical_hash(dict(source=model.identity,target=target_identity,weights=matrix_id(self.matrix),
            reference=self.reference,native_reference=self.native_reference,normalization='all original group members'))

    def apply(self,release):
        value=release.require('dimensionless','effective_synaptic_drive',(self.matrix.shape[1],))
        result=self.reference+self.matrix@(value-self.native_reference)
        if not np.isfinite(result).all() or np.any(result<0):raise ContractError('Invalid projected Cone release')
        return Quantity(result,'dimensionless','effective_synaptic_drive','source_bound_research')


from .h1_h2_effective import H1H2GroupRunner
from .group_time_runner import GroupTimeArm


class ConeHCGroupPortRunner(H1H2GroupRunner):
    """Existing full inner scheduler consuming the registered research release port.

    This is a one-way downstream interface run. The parent's outer states are
    retained for its other interfaces; source Cone/HC readouts belong to the
    separately identified source runner, never relabelled as parent HC states.
    """
    def __init__(self,assembly,seed,protocol,*,source_runner,projection,source_start_seconds=.02,**kwargs):
        self.release_source=source_runner;self.release_projection=projection
        self.release_source_start=float(source_start_seconds);self.group_source_start=protocol['start_seconds']
        self.release_port_identity=canonical_hash(dict(source=source_runner.checkpoint_payload(),
            projection=projection.identity,source_start=self.release_source_start,group_start=self.group_source_start))
        if (seed.get('h1_h2_model',{}).get('artifact')!=assembly.artifact_sha256
                or kwargs.get('mode','reference')!='reference' or kwargs.get('torch') is not None):
            raise ContractError('Pinned source assembly and CPU reference group port required')
        # Consume the existing assembly's verified native factory, including its
        # independently bound HC variant, without relabelling the source model.
        kwargs.setdefault('mode','reference')
        GroupTimeArm.__init__(self,assembly,seed,protocol,**kwargs)
        self.identity=canonical_hash(dict(parent=self.identity,release_port=self.release_port_identity))

    def _cone_release(self,t,side,native_outputs=None):
        source_time=self.release_source_start+(t-self.group_source_start)
        return self.release_projection.apply(self.release_source.release_at(source_time)).values.copy()

    def snapshot(self,t,side='right',**kwargs):
        snapshot=super().snapshot(t,side,**kwargs)
        # The published port must be the same release consumed at the BC event.
        snapshot['ports']['cones','release']=self._cone_release(t,side)
        snapshot['research_release_source']=dict(identity=self.release_port_identity,
            source_model=self.release_source.model.identity,
            parent_auxiliary_outer_states_retained=True)
        return snapshot

    def checkpoint_payload(self,**kwargs):
        payload=GroupTimeArm.checkpoint_payload(self,**kwargs)
        payload['controller']['research_cone_release_port']=self.release_port_identity
        payload['controller']['source_HC_model']=self.seed['h1_h2_model']['model']
        return payload

    @classmethod
    def from_checkpoint(cls,assembly,seed,protocol,payload,**kwargs):
        result=GroupTimeArm.from_checkpoint.__func__(cls,assembly,seed,protocol,payload,**kwargs)
        if (payload['controller'].get('research_cone_release_port')!=result.release_port_identity
                or payload['controller'].get('source_HC_model')!=seed['h1_h2_model']['model']):
            result.close();raise ContractError('Checkpoint research release source or projection changed')
        return result
