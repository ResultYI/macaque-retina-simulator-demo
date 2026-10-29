"""Opt-in H1/H2 effective network on the existing immutable joint HC graph.

H2 uses a causal first-order state in borrowed source-native feedback units.
This module does not change the legacy provider, its source pins or defaults.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
import json

import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import expit

from .contracts import ContractError, Quantity, canonical_hash
from .spatial_group_graph import digest, verified_path
from .spatial_group_model import StateBlock
from .group_native_physics import GroupNativePhysics
from .cone_hc_reduced_native import AnalyticREAnchor, STATE_NAMES
from .group_time_runner import GroupTimeArm
from .vh2005 import SOURCE_VALUES

LEGACY = 'legacy_native_shared_H1_kinetics'
MODEL = 'h1_h2_effective_v1'
MECHANISM = 'h1_h2_effective_v1'
QUALIFICATION = 'engineering_borrowed_uncalibrated'
UNIT = 'vH_native_voltage'  # Opaque effective feedback scale; no mV/SI mapping.


def owned(value):
    a = np.array(value, copy=True)
    a.flags.writeable = False
    return a


def csr(value):
    a = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    a.eliminate_zeros(); a.sort_indices()
    for v in (a.data, a.indices, a.indptr): v.flags.writeable = False
    return a


def matrix_id(a):
    return canonical_hash(dict(shape=a.shape, data=a.data, indices=a.indices, indptr=a.indptr))


def verify_typed_source(full, repo_root):
    """Verify original IDs/types and group membership; never infer a type by position."""
    root_path = Path(repo_root)
    result, evidence = {}, {}
    for name, allowed in (('cones', ('L','M','S')), ('HC', ('H1','H2'))):
        p = full.populations[name]; meta = full.manifest['populations'][name]
        path = verified_path(root_path, meta['source_table'])
        source = json.loads(path.read_bytes())
        arrays = {}
        for key, binding in dict(ids=source['source_keys'], subtype=source['extras']['subtype'],
                                 xy=source['coordinates'][p.coordinate_role]).items():
            arrays[key] = np.load(verified_path(root_path, binding), allow_pickle=False)
        if source['coordinates'][p.coordinate_role]['unit'] != 'mm':
            raise ContractError('HC/Cone source coordinate unit must remain mm')
        labels = tuple(meta['subtype_labels'])
        keys = full.array('populations/'+name+'/keys.npy')
        column = meta['key_columns'].index('subtype')
        codes = keys[:, column]
        if set(labels) != set(allowed) or np.any(codes < 0) or np.any(codes >= len(labels)):
            raise ContractError('Unknown or missing authoritative spectral/HC subtype')
        types = np.asarray(labels)[codes]
        members, offsets = p.member_global_rows, p.member_offsets
        if (len(np.unique(arrays['ids'])) != source['count'] or offsets[-1] != len(members)
                or len(np.unique(members)) != len(members)
                or not np.array_equal(np.diff(offsets), p.counts)):
            raise ContractError('Original ID/member layout differs')
        expected = np.repeat(types, p.counts)
        actual = arrays['subtype'][members].astype('U')
        if not np.array_equal(expected, actual):
            raise ContractError('Group subtype differs from actual fine members')
        if not np.array_equal(p.fine_to_group[members], np.repeat(np.arange(p.group_count), p.counts)):
            raise ContractError('Fine-to-group and ragged member maps disagree')
        evidence[name] = dict(source_table=meta['source_table'], source_keys=source['source_keys'],
            subtype=source['extras']['subtype'], coordinate=source['coordinates'][p.coordinate_role],
            source_bindings=source.get('source_bindings'),
            fine_counts={t:int(np.count_nonzero(actual == t)) for t in allowed},
            group_counts={t:int(np.count_nonzero(types == t)) for t in allowed},
            per_member_type_exact=True)
        result[name] = dict(types=owned(types), fine_ids=arrays['ids'].astype('U'), fine_xy=arrays['xy'], source=source)
    hc = result['HC']; master_sha = hc['source']['source_bindings']['master_sha256']
    master_path = verified_path(root_path, dict(path='data/atlas/hc_disk3mm_master_v1.json',sha256=master_sha))
    master = json.loads(master_path.read_bytes())
    table = next(t for t in master['tables'] if t['population'] == 'HC')
    lookup = {key:i for i,key in enumerate(table['source_keys'])}
    try: rows = np.array([lookup[str(key)] for key in hc['fine_ids']], dtype=np.int64)
    except KeyError as exc: raise ContractError('Production HC is absent from its joint master') from exc
    if (not np.array_equal(np.asarray(table['coordinates']['soma']['values'])[rows], hc['fine_xy'])
            or not np.array_equal(np.asarray(master['source']['types'])[rows],
                                  np.load(verified_path(root_path,hc['source']['extras']['subtype'])).astype('U'))):
        raise ContractError('Production HC identity/coordinate/type differs from its joint master')
    evidence['joint_master'] = dict(path=master_path.relative_to(root_path).as_posix(),sha256=master_sha,
        original_ID_coordinate_type_exact=True,regenerated=False,
        boundary=master['boundary'], density_source_sha256=master['source'].get('density_source_asset_sha256'),
        backend_sha256=master['source']['backend_sha256'])
    return result, evidence


def global_operators(full, types):
    """Type rules on full-global normalized operators, before any patch crop.

    A retains every original association and weight, including nonzero LM->H2.
    H2 feedback is enabled for S only. The new B is normalized on the COMPLETE
    global graph after this rule; local submatrices are never renormalized.
    """
    cone, hc = types['cones'], types['HC']
    if set(cone)-{'L','M','S'} or set(hc) != {'H1','H2'}:
        raise ContractError('Both verified HC types and known cone labels required')
    A = csr(full.matrix('routes/cones_to_HC/mean_weights.npz'))
    old_B = full.matrix('routes/HC_to_cones/mean_weights.npz').tocoo()
    keep = ((hc[old_B.col] == 'H1') & (cone[old_B.row] != 'S') |
            (hc[old_B.col] == 'H2') & (cone[old_B.row] == 'S'))
    B = sparse.coo_matrix((old_B.data[keep],(old_B.row[keep],old_B.col[keep])),shape=old_B.shape).tocsr()
    dc = np.asarray(B.sum(1)).ravel()
    if np.any(dc <= 0): raise ContractError('New complete-global feedback has an unsupported cone')
    B = csr(sparse.diags(1/dc)@B)
    h1, h2 = np.flatnonzero(hc == 'H1'), np.flatnonzero(hc == 'H2')
    if A[h1][:,cone == 'S'].nnz:
        raise ContractError('Source H1 input unexpectedly contains S cones')
    if not A[h2][:,cone == 'S'].nnz or not A[h2][:,cone != 'S'].nnz or not B[cone == 'S'][:,h2].nnz:
        raise ContractError('H2 requires actual S and LM inputs and S feedback')
    return A, B, owned(h1), owned(h2)


def cone_fixed_point(light):
    """Unchanged scalar cone equations from NativeConeH1.equilibrium, in Td."""
    light = np.asarray(light, dtype=np.float64); p = SOURCE_VALUES
    if light.ndim != 1 or not np.isfinite(light).all() or np.any(light < 0):
        raise ContractError('Finite nonnegative native Td background required')
    unique, inverse = np.unique(light, return_inverse=True)
    current = np.array([brentq(lambda c:c-(1/((p['c_beta']+p['k_beta']*L)*
        (1+(p['a_C']*c)**p['n_C'])))**p['n_X'], 0.,
        (1/(p['c_beta']+p['k_beta']*L))**p['n_X'], xtol=1e-12) for L in unique])[inverse]
    X = current**(1/p['n_X'])
    V = (current/p['a_is'])**(1/(1+p['gamma']))
    return dict(R=light.copy(),E=light.copy(),X=X,C=current,Vis=V,
                g_i=p['a_is']*V**p['gamma'],Vbar=V.copy())


def joint_equilibrium(A, B, h1_rows, h2_rows, light, parameters, *, guard=lambda:None):
    """All H1 and H2 feedback enters one nonlinear fixed-point residual."""
    cone = cone_fixed_point(light); p = SOURCE_VALUES
    scale = (cone['Vbar']/p['V_I'])**p['mu']
    gains = np.full(A.shape[0],p['g_t']); gains[h2_rows] = parameters['input_scale']
    feedback = B.copy().tocsc(); feedback[:,h2_rows] *= parameters['feedback_scale']; feedback = feedback.tocsr()
    def release(h): return expit((cone['Vis']-feedback@h-p['V_k'])/p['V_n'])/scale
    def residual(h):
        guard(); return h-gains*(A@release(h))
    # Analytic Jv avoids finite-difference cancellation in the full coupled graph.
    # Linear budgets are the existing group_native_radau 1e-10/1e-14 values.
    h = gains*(A@release(np.zeros(A.shape[0])))*.5
    history = []; nfev = 0
    for iteration in range(200):
        f = residual(h); nfev += 1; error = float(np.max(np.abs(f)))
        history.append(error)
        if error <= 1e-10: break
        q = expit((cone['Vis']-feedback@h-p['V_k'])/p['V_n'])
        derivative = q*(1-q)/(p['V_n']*scale)
        jac = LinearOperator((len(h),len(h)),matvec=lambda v:v+gains*(A@(derivative*(feedback@v))),dtype=np.float64)
        delta,status = gmres(jac,-f,rtol=1e-10,atol=1e-14,restart=32,maxiter=200)
        if status: raise ContractError('Joint analytic Newton linear solve failed: '+str((status,history)))
        for power in range(32):
            proposed = h+delta*(.5**power)
            if np.any(proposed < 0): continue
            candidate = residual(proposed); nfev += 1
            if np.max(np.abs(candidate)) < error:
                h = proposed; break
        else: raise ContractError('Joint Newton line search failed without clipping: '+str(history))
    else: raise ContractError('Joint fixed point failed to converge: '+str(history))
    if error > 1e-8 or not np.isfinite(h).all() or np.any(h < 0):
        raise ContractError('H1+H2 joint fixed point domain/residual failed: '+str(history))
    r = release(h); w = p['g_t']*(A[h1_rows]@r)
    return dict(cone=cone,h=owned(h),release=owned(r),w=owned(w),z2=owned(w),
        residual=error,nfev=nfev,newton_residual_history=history,feedback=csr(feedback))


class H1H2Assembly:
    """Independent local artifact over actual parent rows and held exterior."""
    def __init__(self, parent, full, typed, evidence, *, engineering=False, feedback_scale=1.):
        if not engineering or feedback_scale not in (0.,1.):
            raise ContractError('H2 engineering opt-in required; only declared feedback ablation is allowed')
        self.parent = parent
        self.global_types = {n:owned(typed[n]['types']) for n in ('cones','HC')}
        self.global_A, self.global_B, self.global_h1, self.global_h2 = global_operators(full,self.global_types)
        self.native_rows = {n:owned(parent.parent_population_rows[n]) for n in ('cones','HC')}
        self.types = {n:owned(self.global_types[n][self.native_rows[n]]) for n in ('cones','HC')}
        self.h1_rows = owned(np.flatnonzero(self.types['HC'] == 'H1'))
        self.h2_rows = owned(np.flatnonzero(self.types['HC'] == 'H2'))
        if not len(self.h1_rows) or not len(self.h2_rows):
            raise ContractError('h1_h2_effective_v1 requires both retained HC subtypes; no fallback')
        self.h2_parameters = dict(tau_seconds=SOURCE_VALUES['tau_h']/1000.,
            input_scale=SOURCE_VALUES['g_t'], feedback_scale=float(feedback_scale))
        self.global_light = owned(full.array('input/group_background_native_td.npy'))
        self._new_matrices = {}
        c, h = self.native_rows['cones'],self.native_rows['HC']
        A = csr(self.global_A[h][:,c]); B = csr(self.global_B[c][:,h])
        self._new_matrices['routes/cones_to_HC/mean_weights.npz'] = A
        # Bridge feedback scale must match the RHS, including the ablation.
        scaled = B.copy().tocsc(); scaled[:,self.h2_rows] *= feedback_scale
        B = csr(scaled)
        self._new_matrices['routes/HC_to_cones/mean_weights.npz'] = B
        self.operators = dict(A1=csr(A[self.h1_rows]),A2=csr(A[self.h2_rows]),
            B1=csr(B[:,self.h1_rows]),B2=csr(B[:,self.h2_rows]))
        for name,a in self.operators.items(): self._new_matrices['h1_h2/'+name+'.npz'] = a
        self.manifest = deepcopy(parent.manifest)
        self.connection_identity = dict(model=MODEL, qualification=QUALIFICATION,types=evidence,
            parent_artifact=parent.artifact_sha256, full_parent=full.artifact_sha256,
            global_A=matrix_id(self.global_A),global_B=matrix_id(self.global_B),
            local_operators={n:dict(shape=a.shape,nnz=a.nnz,sha256=matrix_id(a),unit='dimensionless') for n,a in self.operators.items()},
            row_identity={n:parent.populations[n].group_ids for n in ('cones','HC')},
            local_parent_rows=self.native_rows,cone_types=self.types['cones'],hc_types=self.types['HC'],
            input_weights='existing complete-global per-association unit-DC A, no spectral gain multiplier',
            feedback_rule='H1 to LM; H2 to S; complete-global B normalized once before crop',
            crop_normalized=False,member_multiplier=1,explicit_gap_coupling=False,
            boundary='held jointly solved full-parent background; no stimulus-synchronous exterior cells',
            source_sha256=digest(Path(__file__)))
        self.artifact_sha256 = canonical_hash(dict(connection=self.connection_identity,parameters=self.h2_parameters))
        blocks = [b for b in parent.state_blocks if b.population != 'HC']
        for name,rows in (('w',self.h1_rows),('z2',self.h1_rows),('h1',self.h1_rows),('h2',self.h2_rows)):
            blocks.append(StateBlock('HC',MECHANISM,name,UNIT,
                'nonnegative_effective_HC_feedback_state_no_mV_mapping',len(rows),0))
        at = 0; self.state_blocks = []
        for b in blocks:
            self.state_blocks.append(replace(b,offset=at)); at += b.count
        self.state_blocks = tuple(self.state_blocks); self.state_scalar_count = at
        self.runtime_factories = dict(parent.runtime_factories,native=H1H2Native)
        self.recording_fields = tuple(replace(f,rows=owned(np.intersect1d(f.rows,self.h1_rows)))
            if f.population == 'HC' and f.name in ('w','z2') else f for f in parent.recording_fields)
        self.layout = dict(schema='h1-h2-effective-state-layout-1',model=MODEL,
            state_blocks=[vars(b) for b in self.state_blocks],h1_group_rows=self.h1_rows,h2_group_rows=self.h2_rows,
            transport_carrier='Vh[original HC row]: H1=h1, H2=h2; opaque source-native feedback units',
            packed_order=STATE_NAMES,packed_sizes=(len(c),)*5+(len(self.h1_rows),)*2+(len(h),len(parent.populations['cones'].recording_auxiliary_group_rows)))
        self.time_partition = dict(schema='h1-h2-effective-time-partition-1',
            native='joint Cone+H1+H2 continuous RHS',native_max_step_seconds=.0001,
            analytic_RE='unchanged source adaptation with native Td inputs',
            rod_step_seconds=.0001,BC_RGC_period_seconds=.001,
            unsupported_backends=['structured_native_radau_v1','native_forced_waveform_candidate_v1','CUDA_v1'])

    def __getattr__(self,name): return getattr(self.parent,name)

    def matrix(self,relative):
        return self._new_matrices[relative] if relative in self._new_matrices else self.parent.matrix(relative)

    def seed(self, old_seed, *, guard=lambda:None):
        eq = joint_equilibrium(self.global_A,self.global_B,self.global_h1,self.global_h2,
                              self.global_light,self.h2_parameters,guard=guard)
        c,h = self.native_rows['cones'],self.native_rows['HC']
        seed = deepcopy(old_seed)
        seed['states'] = {k:v for k,v in seed['states'].items() if k[0] not in ('cones','HC')}
        for name,v in eq['cone'].items():
            seed['states']['cones','analytic_RE' if name in ('R','E') else 'reduced_native',name] = v[c].copy()
        A = self.matrix('routes/cones_to_HC/mean_weights.npz')
        B = self.matrix('routes/HC_to_cones/mean_weights.npz')
        self.native_boundary = dict(A_release=owned(self.global_A[h]@eq['release']-A@eq['release'][c]),
            A_Vbar=owned(self.global_A[h]@eq['cone']['Vbar']-A@eq['cone']['Vbar'][c]),
            B_h=owned(eq['feedback'][c]@eq['h']-B@eq['h'][h]),
            mode='held_background_reservoir',equilibrium_residual=eq['residual'])
        local_h = eq['h'][h]; w = SOURCE_VALUES['g_t']*(self.global_A[h[self.h1_rows]]@eq['release'])
        for name,value in (('w',w),('z2',w),('h1',local_h[self.h1_rows]),('h2',local_h[self.h2_rows])):
            seed['states']['HC',MECHANISM,name] = value.copy()
        aux = self.populations['cones'].recording_auxiliary_group_rows
        seed['states']['cones','selected_original_z1_auxiliary','z1_aux'] = SOURCE_VALUES['g_t']*eq['release'][c][aux]
        observation = seed['observation_prehistory']
        observation.update(history_times=np.array([seed['time_seconds']]),history_values=local_h[None,:].copy(),
                           prehistory=local_h.copy(),origin_seconds=seed['time_seconds'])
        seed['h1_h2_model'] = dict(model=MODEL,qualification=QUALIFICATION,artifact=self.artifact_sha256,
            layout_sha256=canonical_hash(self.layout),boundary=deepcopy(self.native_boundary),
            parameters=self.h2_parameters,full_parent_equilibrium=dict(residual=eq['residual'],nfev=eq['nfev'],
                newton_residual_history=eq['newton_residual_history']),
            coupled_outer_background_qualified=True,whole_retina_background_qualified=False,
            physiological_calibration=False,FIT_READY=False)
        seed['inner_parameters']['identity'] = canonical_hash(dict(parent=seed['inner_parameters'].get('identity'),
            new_outer_seed=seed['h1_h2_model']))
        return seed


class H1H2Native(GroupNativePhysics):
    """Compact w/z2 for H1; independent h2 occupies its own typed carrier rows."""
    def __init__(self,assembly,*,torch=None,device=None,plans=None):
        if torch is not None or plans is not None:
            raise ContractError('H1/H2 effective v1 currently requires explicit CPU joint-reference backend')
        super().__init__(assembly,torch=None)
        self.assembly = assembly; self.h1_rows = assembly.h1_rows; self.h2_rows = assembly.h2_rows
        self.parameters = MappingProxyType(dict(assembly.h2_parameters))
        if (set(self.parameters) != {'tau_seconds','input_scale','feedback_scale'}
                or not all(np.isfinite(v) for v in self.parameters.values())
                or self.parameters['tau_seconds'] <= 0 or self.parameters['input_scale'] <= 0
                or self.parameters['feedback_scale'] not in (0.,1.)):
            raise ContractError('H2 requires positive tau/input scale and explicit feedback-on/off domain')
        self.sizes = (self.nc,)*5+(len(self.h1_rows),)*2+(self.nh,len(self.aux_indices))
        self.slices = {}; at = 0
        for name,size in zip(STATE_NAMES,self.sizes): self.slices[name] = slice(at,at+size); at += size
        self.nstate = at
        self.identity.update(model=MODEL,qualification=QUALIFICATION,layout=assembly.layout,
            h2_parameters=dict(self.parameters),connections=assembly.connection_identity,
            implementation_sha256=digest(Path(__file__)))
        self.source_version = canonical_hash(self.identity)
        self.boundary = None

    @property
    def boundary(self): return self._boundary

    @boundary.setter
    def boundary(self,value):
        if getattr(self,'_boundary',None) is not None:
            raise ContractError('Held H1/H2 exterior cannot refresh with stimulus')
        if value is None:
            self._boundary = None; return
        required = dict(A_release=self.nh,A_Vbar=self.nh,B_h=self.nc)
        arrays = {}
        for name,n in required.items():
            a = np.asarray(value[name])
            if a.dtype != np.float64 or a.shape != (n,) or not np.isfinite(a).all():
                raise ContractError('Explicit held native boundary dtype/shape/domain differs')
            arrays[name] = owned(a)
        self._boundary = MappingProxyType(arrays)

    def _validate(self,y,anchor,t):
        result = super()._validate(y,anchor,t)
        if np.any(y[self.slices['Vh']] < 0):
            raise ContractError('H1/H2 effective state left its nonnegative invariant domain; no clipping')
        return result

    def construction_point(self,seed,light,input_stop_seconds,input_epoch):
        meta = seed.states
        carrier = np.empty(self.nh)
        for name,rows in (('h1',self.h1_rows),('h2',self.h2_rows)):
            carrier[rows] = meta['HC',MECHANISM,name]
        values = [meta['cones','reduced_native',n] for n in ('X','C','Vis','g_i','Vbar')]
        values += [meta['HC',MECHANISM,n] for n in ('w','z2')]
        values += [carrier,meta['cones','selected_original_z1_auxiliary','z1_aux']]
        y = np.concatenate(values)
        anchor = AnalyticREAnchor(seed.time_seconds,meta['cones','analytic_RE','R'],
            meta['cones','analytic_RE','E'],light,input_stop_seconds,input_epoch)
        self._validate(y,anchor,seed.time_seconds)
        return y,anchor

    def migrate(self,*args,**kwargs):
        raise ContractError('Legacy H1/shared-H2 states cannot migrate into a qualified H1+H2 seed')

    def _coefficients(self,y,t,anchor):
        R,E = self._validate(y,anchor,t)
        X,C,V,gi,Vbar,w,z2,h,aux = self.split(y); p = self.p
        if self.boundary is None: raise ContractError('Explicit H1/H2 boundary binding required')
        bh = self.B@h+self.boundary['B_h']
        q = expit((V-bh-p['V_k'])/p['V_n']); aI = (Vbar/p['V_I'])**p['mu']; release = q/aI
        pooled = ((self.A@Vbar+self.boundary['A_Vbar'])[self.h1_rows]/p['V_I'])**p['mu']
        ar = self.A@release+self.boundary['A_release']
        ah,bh_decay = np.empty(self.nh),np.empty(self.nh)
        ah[self.h1_rows] = 1000.*z2/(p['tau_h']*pooled)
        bh_decay[self.h1_rows] = 1000./(p['tau_h']*pooled)
        ah[self.h2_rows] = self.parameters['input_scale']*ar[self.h2_rows]/self.parameters['tau_seconds']
        bh_decay[self.h2_rows] = 1/self.parameters['tau_seconds']
        a = (1000./(1+(p['a_C']*C)**p['n_C']),1000.*X/p['tau_C'],1000.*X/(gi*p['tau_m']),
            1000.*p['a_is']*V**p['gamma']/p['tau_is'],1000.*V/p['tau_a'],
            1000.*p['g_t']*ar[self.h1_rows]/p['tau_1'],1000.*w/(p['tau_2']*pooled),
            ah,1000.*p['g_t']*release[self._aux]/p['tau_1'])
        b = (1000.*(p['c_beta']+p['k_beta']*E),1000./p['tau_C'],1000./p['tau_m'],1000./p['tau_is'],
            1000./p['tau_a'],1000./p['tau_1'],1000./(p['tau_2']*pooled),bh_decay,1000./p['tau_1'])
        # Sparse readout carrier keeps the existing exact native->BC bridge API.
        full_w,full_z2,full_pool = np.zeros(self.nh),np.zeros(self.nh),np.ones(self.nh)
        full_w[self.h1_rows]=w;full_z2[self.h1_rows]=z2;full_pool[self.h1_rows]=pooled
        outputs = dict(zip(STATE_NAMES,(X,C,V,gi,Vbar,full_w,full_z2,h,aux)))
        outputs.update(R=R,E=E,q=q,release=release,Ios=X,Bh=bh,aI=aI,pooled_aI=full_pool)
        self.statistics['coefficient_evaluations'] += 1
        return a,b,outputs


class H1H2GroupRunner(GroupTimeArm):
    """Actual original full-network runner; explicitly reference-only for v1."""
    def __init__(self,assembly,seed,protocol,*,mode='reference',**kwargs):
        if mode != 'reference' or kwargs.get('torch') is not None or kwargs.get('shared_plans') is not None:
            raise ContractError('H1/H2 supports CPU joint DOP853 100us only; unsupported backend cannot bypass H2')
        meta = seed.get('h1_h2_model',{})
        if meta.get('model') != MODEL or meta.get('artifact') != assembly.artifact_sha256:
            raise ContractError('New independently equilibrated H1/H2 seed required')
        if (meta.get('layout_sha256') != canonical_hash(assembly.layout)
                or meta.get('qualification') != QUALIFICATION
                or meta.get('parameters') != dict(assembly.h2_parameters)):
            raise ContractError('H1/H2 seed layout/engineering qualification/parameters differ')
        if not meta.get('parameter_binding_id') or meta.get('parameter_binding_id') != getattr(assembly,'parameter_binding_id',None):
            raise ContractError('H2 must consume its separately resolved ParameterRecord binding')
        assembly.native_boundary = deepcopy(meta['boundary'])
        # The constructor validates the supplied joint seed without evaluating RHS.
        super().__init__(assembly,seed,protocol,mode=mode,**kwargs)
        self.native_physics.boundary = assembly.native_boundary
        self.identity = canonical_hash(dict(parent_execution=self.identity,model=MODEL,
            layout=assembly.layout,boundary=meta['boundary'],implementation=digest(Path(__file__))))

    def _cone_release(self,t,side,native_outputs=None):
        # The bridge's affine boundary is an explicit held background product.
        # Interior dense release uses exactly the same current H1+H2 feedback RHS.
        if native_outputs is None: native_outputs = self._native_evaluation(t,side)[0].outputs
        return np.array(native_outputs['release'],copy=True)

    def snapshot(self,*args,**kwargs):
        snapshot = super().snapshot(*args,**kwargs)
        for category in ('states','rhs'):
            values = snapshot[category]
            for name in ('w','z2'):
                values['HC',MECHANISM,name] = values.pop(('HC','reduced_native',name))
            carrier = values.pop(('HC','reduced_native','Vh'))
            values['HC',MECHANISM,'h1'] = carrier[self.assembly.h1_rows].copy()
            values['HC',MECHANISM,'h2'] = carrier[self.assembly.h2_rows].copy()
        for name in ('h1','h2'):
            snapshot['ports']['HC',name] = snapshot['states']['HC',MECHANISM,name].copy()
            snapshot['port_units']['HC',name] = UNIT
        return snapshot

    def checkpoint_payload(self,**kwargs):
        payload = super().checkpoint_payload(**kwargs)
        payload['controller']['hc_model'] = dict(model=MODEL,artifact=self.assembly.artifact_sha256,
            layout_sha256=canonical_hash(self.assembly.layout),parameter_binding_id=self.assembly.parameter_binding_id)
        h = self.native_state.y[self.native_physics.slices['Vh']]
        payload['native']['h1_h2_semantic_frontier'] = dict(h1=h[self.assembly.h1_rows].copy(),
                                                         h2=h[self.assembly.h2_rows].copy())
        return payload

    @classmethod
    def from_checkpoint(cls,assembly,seed,protocol,payload,**kwargs):
        expected = dict(model=MODEL,artifact=assembly.artifact_sha256,
            layout_sha256=canonical_hash(assembly.layout),parameter_binding_id=assembly.parameter_binding_id)
        if payload.get('controller',{}).get('hc_model') != expected:
            raise ContractError('H1/H2 checkpoint model/layout/parameter identity differs; legacy is incompatible')
        result = super().from_checkpoint(assembly,seed,protocol,payload,**kwargs)
        h = result.native_state.y[result.native_physics.slices['Vh']]
        semantic = payload['native'].get('h1_h2_semantic_frontier',{})
        if set(semantic) != {'h1','h2'} or any(not np.array_equal(semantic[n],h[rows])
                for n,rows in (('h1',assembly.h1_rows),('h2',assembly.h2_rows))):
            result.close()
            raise ContractError('Checkpoint lost or changed the explicit H1/H2 frontier states')
        return result


def bind_parameters(assembly,seed,repo_root,protocol):
    """Extend the existing v1 registry/resolver; all H2 values are conditional."""
    from .parameter_system import (PopulationIndex,ParameterRecord,IntrinsicParameterRegistry,
        ParameterResolver,Evidence,Applicability,AggregationRule,ResolvedParameterBundle,
        RouteRecord,RouteRegistry,data)
    from .assembly import RetinaBuilder
    base = RetinaBuilder.bind_group_snapshot(assembly,seed,repo_root,protocol=protocol)
    conditional = dict(hc_model=MODEL,h2_qualification=QUALIFICATION)
    context = replace(base.context,engineering_preview=True,conditional=conditional)
    populations = dict(base.populations)
    for name in ('cones','HC'):
        populations[name] = replace(populations[name],cell_types=tuple(assembly.types[name]))
    for subtype,rows in (('H1',assembly.h1_rows),('H2',assembly.h2_rows)):
        p = assembly.populations['HC']
        populations[subtype] = PopulationIndex(subtype,tuple(p.group_ids[rows]),(subtype,)*len(rows),
            p.counts[rows],'coarse_group',p.centroid_mm[rows],'mm',None,None,(None,)*len(rows),
            ('effective_feedback_pool',)*len(rows),assembly.artifact_sha256,coordinate_frame=p.coordinate_role)
    registry = IntrinsicParameterRegistry(base.registry.records)
    units = dict(tau_seconds='s',input_scale=UNIT,feedback_scale='dimensionless')
    borrowed = dict(tau_seconds='tau_h (base20ms only; H1 adaptation multiplier is not borrowed)',
        input_scale='g_t (H1 effective release-to-feedback scale)',
        feedback_scale='unit conversion identity in borrowed source-native feedback scale;0 is explicit ablation')
    for name,value in assembly.h2_parameters.items():
        registry.register(ParameterRecord('H2/'+name,MODEL,('H2',),name,
            dict(rule='conditional_override',value=value,layout='broadcast'),units[name],
            Applicability(conditional=conditional),
            Evidence('engineering_cross_type_borrow','vh2005.SOURCE_VALUES',
                'retina_sim/vh2005.py::SOURCE_VALUES; '+borrowed[name],
                digest(Path(repo_root)/'retina_sim/vh2005.py'),limitations=(
                    'Not measured or fitted H2 physiology','No RGC/PSTH-selected value',
                    'H2 dynamics and effective gain uncalibrated; engineering opt-in only')),
            'H2',AggregationRule(),calibration_permissions=(),
            domain=dict(minimum=0.,minimum_inclusive=name=='feedback_scale'),status='conditional'))
    for subtype,quantity,value,source in (
            ('H1','cone_input_type_mask',[1.,1.,0.],'Dacey1996 doi:10.1126/science.271.5249.656'),
            ('H2','cone_input_type_mask',[1.,1.,1.],'Dacey1996 doi:10.1126/science.271.5249.656'),
            ('H2','cone_feedback_type_mask',[0.,0.,1.],'Packer2010 doi:10.1523/JNEUROSCI.4738-09.2010')):
        registry.register(ParameterRecord(subtype+'/'+quantity,MODEL,(subtype,),quantity,
            dict(rule='conditional_override',value=value,layout='broadcast'),'dimensionless',
            Applicability(conditional=conditional),Evidence('source_informed_engineering_rule',source,
                'L,M,S order; enable existing associations only; no inferred anatomical synapses',
                limitations=('H2->LM feedback excluded in this evidence-conservative version',)),
            subtype,AggregationRule(),status='conditional'))
    resolver = ParameterResolver(registry,context,populations)
    values,refs = dict(base.values),dict(base.provenance_refs)
    for record in registry.records[len(base.registry.records):]:
        values[record.parameter_id],refs[record.parameter_id] = resolver.resolve(
            record.binding_subject,record.quantity,record.unit,mechanism=MODEL)
    # The runner consumes the resolved values. No H2 parameter broadcasts to H1.
    assembly.h2_parameters = {n:float(values['H2/'+n][0]) for n in units}
    route_records = list(base.routes.records)
    matrices = dict(base.routes._matrices)
    for name,a in assembly.operators.items():
        pre,post = ('cones','H1' if name=='A1' else 'H2') if name[0]=='A' else ('H1' if name=='B1' else 'H2','cones')
        route_records.append(RouteRecord(name,pre,post,dict(mean_weights=dict(identity=matrix_id(a),
            shape=a.shape,nnz=a.nnz,path='h1_h2/'+name+'.npz')),
            assembly.artifact_sha256,'global source normalization before crop; no member multiplier',
            dict(mean_weights='dimensionless')))
        matrices[name,'mean_weights'] = a
    coverage = data(base.coverage)
    coverage['unsupported'].pop('H2',None)
    coverage.update(hc_model=MODEL,H2=dict(status=QUALIFICATION,physiological_calibration=False,
        spectral_gain_calibration=False,input_weights='unchanged existing associations',
        full_parent_joint_equilibrium=True,calibration_permissions=[]),initial_state_recomputed=True)
    bundle = ResolvedParameterBundle(context,populations,values,refs,registry,
        RouteRegistry(route_records,matrices),base.legacy_parameters,coverage)
    assembly.parameter_binding_id = bundle.identity
    seed['h1_h2_model']['parameter_binding_id'] = bundle.identity
    return bundle


def runner(model,assembly,seed,protocol,**kwargs):
    if model == LEGACY:
        if isinstance(assembly,H1H2Assembly) or 'h1_h2_model' in seed:
            raise ContractError('H1/H2 assembly or seed cannot carry a legacy identity')
        return GroupTimeArm(assembly,seed,protocol,**kwargs)
    if model == MODEL: return H1H2GroupRunner(assembly,seed,protocol,**kwargs)
    raise ContractError('Explicit recognized HC model identity required')


def save_artifact(assembly,typed,destination):
    """Traceable typed local matrices; never overwrites a legacy group artifact."""
    from .parameter_system import data
    path = Path(destination); path.mkdir(parents=True,exist_ok=False)
    files = {}
    for pop in ('cones','HC'):
        p = assembly.populations[pop]
        rows = p.member_global_rows
        np.savez_compressed(path/(pop+'.npz'),group_ids=p.group_ids,group_types=assembly.types[pop],
            parent_group_rows=assembly.native_rows[pop],centroid_mm=p.centroid_mm,member_count=p.counts,
            member_offsets=p.member_offsets,member_global_rows=rows,
            original_member_ids=typed[pop]['fine_ids'][rows],original_member_xy_mm=typed[pop]['fine_xy'][rows])
    for name,a in assembly.operators.items():
        pre,post = ('cones','HC') if name[0]=='A' else ('HC','cones')
        subtype = 'H1' if name[-1]=='1' else 'H2'
        selected = assembly.h1_rows if subtype=='H1' else assembly.h2_rows
        pre_rows = np.arange(assembly.populations[pre].group_count) if pre=='cones' else selected
        post_rows = np.arange(assembly.populations[post].group_count) if post=='cones' else selected
        coo = a.tocoo()
        np.savez_compressed(path/(name+'.npz'),data=a.data,indices=a.indices,indptr=a.indptr,shape=a.shape,
            pre_ids=assembly.populations[pre].group_ids[pre_rows],post_ids=assembly.populations[post].group_ids[post_rows],
            pre_types=assembly.types[pre][pre_rows],post_types=assembly.types[post][post_rows],
            edge_pre_group_rows=pre_rows[coo.col],edge_post_group_rows=post_rows[coo.row],edge_weights=coo.data,
            unit=np.array('dimensionless'),representation=np.array('source_effective_functional_association'))
    for p in path.iterdir(): files[p.name] = dict(sha256=digest(p),bytes=p.stat().st_size)
    manifest = dict(schema='h1-h2-effective-local-artifact-1',artifact_sha256=assembly.artifact_sha256,
        model=MODEL,connection_contract=assembly.connection_identity,state_layout=assembly.layout,
        time_partition=assembly.time_partition,files=files,FIT_READY=False,physiological_calibration=False)
    (path/'manifest.json').write_text(json.dumps(data(manifest),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    return manifest
