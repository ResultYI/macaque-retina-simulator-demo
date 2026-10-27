"""Opt-in functional V+s AC model; unchanged upstream and S-ON equations.

Full-parent input kernels are built before selecting a local graph. Missing
chemical ancestors are rejected, never hidden by crop normalization.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import json

import numpy as np
from scipy import sparse
from scipy.integrate import DOP853
from scipy.spatial import cKDTree
from scipy.special import expit

from .contracts import ContractError, canonical_hash
from .group_inner_dynamics import GroupInnerDynamics, InnerAdvance, AC_MECH, RGC_MECH, key
from .local_patch_boundary import BoundedInnerDynamics
from .spatial_group_model import AC, RGC, StateBlock, RouteLayout
from .spatial_group_graph import digest
from .continuous_state import Dop853Dense
from .rgc_expectation import STATES
from .local_patch_graph import LocalPatchAssembly,_select_bounded,_verified_base

SCHEMA = 'ordinary-ac-voltage-gate-functional-1'
MECHANISM = 'FunctionalAmacrineVoltageGate_v1'
LEGACY_S = 'local_feedforward_AC'


def functional_patch(repo_root,config,directory,guard):
    """Union the original exact chemical closures of all five requested cores."""
    root=Path(repo_root);directory=Path(directory)
    base=_verified_base(root,config)
    selections=[]
    for target in RGC:
        request=deepcopy(config);request['target_rgc_type']=target
        if target!=config['target_rgc_type']:request['target_rgc_id']=None
        selections.append(_select_bounded(base,request,guard))
    rows={name:np.unique(np.concatenate([s[0][name] for s in selections])) for name in selections[0][0]}
    contacts={name:np.unique(np.concatenate([s[1][name] for s in selections])) for name in selections[0][1]}
    core={name:np.unique(np.concatenate([s[2][name] for s in selections])) for name in selections[0][2]}
    target_row=selections[RGC.index(config['target_rgc_type'])][3]
    descriptor=dict(schema='ac-vs-multiple-target-source-closure-1',configuration=config,
        base_artifact=base.artifact_sha256,selection_hash=canonical_hash(dict(rows=rows,contacts=contacts,core=core)),
        original_closure_audits=[s[4] for s in selections],target_parent_row=int(target_row),
        implementation=digest(Path(__file__)),weights_renormalized=False)
    artifact=canonical_hash(descriptor)
    if directory.exists():
        stored=json.loads((directory/'selection_identity.json').read_text(encoding='utf8'))
        if stored!=descriptor:raise ContractError('Frozen new multiple-target graph selection changed')
    else:
        directory.mkdir()
        (directory/'selection_identity.json').write_text(json.dumps(descriptor,indent=2)+'\n',encoding='utf8')
        np.savez_compressed(directory/'selection.npz',**{'population__'+n:v for n,v in rows.items()},
            **{'contact__'+n:v for n,v in contacts.items()},**{'core__'+n:v for n,v in core.items()})
    patch=object.__new__(LocalPatchAssembly)
    patch._assemble(root,directory,base,config,rows,contacts,core,target_row,artifact,descriptor)
    return patch


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf8'))
    if config.get('schema') != SCHEMA or config.get('calibrated') is not False:
        raise ContractError('Explicit uncalibrated AC V+s configuration required')
    p = config['parameters_per_effective_pool_SI']
    if set(p) != {'C','gL','EL','EE','theta','k','alpha','beta','gE_base'}:
        raise ContractError('Complete SI membrane/gate parameter binding required')
    if any(not np.isfinite(v) for v in p.values()) or any(p[n] <= 0 for n in ('C','gL','k','alpha','beta')) or p['gE_base'] < 0:
        raise ContractError('AC SI parameter domain differs')
    expected = [('midget_BC_ON','midget_RGC_ON','same_sign'), ('midget_BC_OFF','midget_RGC_OFF','same_sign'),
                ('diffuse_BC_OFF','parasol_RGC_ON','crossover'), ('diffuse_BC_ON','parasol_RGC_OFF','crossover')]
    if [(c['pre'],c['target'],c['relation']) for c in config['channels']] != expected or len({c['pool'] for c in config['channels']}) != 4:
        raise ContractError('Four separate priority functional channels required')
    if any(not np.isfinite(c[g]) or c[g] < 0 for c in config['channels'] for g in ('G_BA_S','G_AR_S')):
        raise ContractError('Absolute route conductances must be nonnegative')
    g = config['geometry']
    if not 0 < g['input_sigma_deg'] <= g['input_support_deg'] or g['crop_renormalization'] is not False or g['output_support_deg'] != 0:
        raise ContractError('Explicit finite full-parent input and delta output geometry required')
    return config


def equilibrium(gE, p):
    gE = np.asarray(gE, dtype=np.float64)
    if not np.isfinite(gE).all() or np.any(gE < 0):
        raise ContractError('Nonnegative AC excitation required')
    V = (p['gL']*p['EL']+gE*p['EE'])/(p['gL']+gE)
    probability = expit((V-p['theta'])/p['k'])
    gate = p['alpha']*probability/(p['alpha']*probability+p['beta'])
    return V, probability, gate


def full_input_kernel(pre, post, geometry):
    scale = geometry['mm_per_degree']
    tree = cKDTree(pre.centroid_mm/scale)
    neighbors = tree.query_ball_point(post.centroid_mm/scale, geometry['input_support_deg'])
    rows, columns, data = [], [], []
    empty = []
    for i, indices in enumerate(neighbors):
        indices = np.asarray(sorted(indices), dtype=np.int64)
        if not len(indices):
            empty.append(i)
            continue
        distance2 = np.sum(((pre.centroid_mm[indices]-post.centroid_mm[i])/scale)**2, axis=1)
        weights = pre.counts[indices]*np.exp(-distance2/(2*geometry['input_sigma_deg']**2))
        weights /= weights.sum()  # Full parent only, before any local crop.
        rows.extend([i]*len(indices)); columns.extend(indices.tolist()); data.extend(weights.tolist())
    matrix = sparse.csr_matrix((data,(rows,columns)), shape=(post.group_count,pre.group_count))
    return matrix, empty


class ACVSAssembly:
    def __init__(self, patch, config, *, max_step=.0001):
        if max_step not in (.0001,.00005):
            raise ContractError('Frozen base/half inner steps required')
        self.patch, self.ac_config, self.max_step = patch, deepcopy(config), max_step
        self.populations = dict(patch.populations)
        self.ac_populations = (LEGACY_S, *(c['pool'] for c in config['channels']))
        self.routes = {n:r for n,r in patch.routes.items() if
            (r.pre not in AC and r.post not in AC) or
            (r.post == LEGACY_S) or (r.pre == LEGACY_S and r.post == 'small_bistratified_S_ON')}
        self._new_matrices = {}
        self.route_ledger = []
        self.core_population_rows = dict(patch.core_population_rows)
        center = np.asarray(patch.config.get('rf_center_deg', [patch.config['eccentricity_deg'],0.]))
        geometry = config['geometry']
        if geometry['mm_per_degree'] != patch.manifest['configuration']['mm_per_degree']:
            raise ContractError('Existing engineering degree mapping differs')
        for channel in config['channels']:
            pool, pre, post = channel['pool'], channel['pre'], channel['target']
            post_rows = patch.parent_population_rows[post]
            pre_rows = patch.parent_population_rows[pre]
            target = patch.populations[post]
            n = target.group_count
            self.populations[pool] = SimpleNamespace(group_count=n, counts=np.ones(n,dtype=np.int64),
                group_ids=np.asarray([pool+':'+str(i) for i in target.group_ids]), centroid_mm=target.centroid_mm,
                coordinate_role='effective_functional_RF_pool_not_anatomical_AC')
            full, empty = full_input_kernel(patch.base.populations[pre], patch.base.populations[post], geometry)
            original = full[post_rows]
            local = original[:,pre_rows].tocsr()
            missing = original.nnz-local.nnz
            empty_selected = np.intersect1d(post_rows, empty)
            if missing or len(empty_selected):
                raise ContractError(f'New AC route lacks full-parent BC support: {pool}; missing edges={missing}; empty rows={len(empty_selected)}')
            for a,b,matrix in ((pre,pool,local),(pool,post,sparse.eye(n,format='csr'))):
                name = a+'_to_'+b
                route = RouteLayout(name,a,b,'inner_chemical',matrix.shape,matrix.nnz,None,0,0,None,
                    'full-parent fixed functional weights; target members applied once at receptor',
                    'full-parent normalized BC fractional mean' if a==pre else 'owning functional pool delta')
                self.routes[name] = route
                self._new_matrices['routes/'+name+'/mean_weights.npz'] = matrix
            self.core_population_rows[post] = np.flatnonzero(np.linalg.norm(target.centroid_mm/geometry['mm_per_degree']-center,axis=1)<=patch.config['patch_geometry']['core_radius_deg'])
            self.core_population_rows[pool] = self.core_population_rows[post].copy()
            self.route_ledger.append(dict(channel, input_route=pre+'_to_'+pool, output_route=pool+'_to_'+post,
                full_parent_kernel_sha256=canonical_hash(dict(data=full.data,indices=full.indices,indptr=full.indptr,shape=full.shape)),
                local_kernel_sha256=canonical_hash(dict(data=local.data,indices=local.indices,indptr=local.indptr,shape=local.shape)),
                full_parent_empty_rows=len(empty), selected_empty_rows=0, missing_edges=0,
                full_parent_outer_boundary_row_indices=[int(i) for i in post_rows if np.linalg.norm(patch.base.populations[post].centroid_mm[i]/geometry['mm_per_degree'])+geometry['input_support_deg']>patch.base.manifest['configuration']['output_radius_deg']],
                effective_pool_count=n, member_semantics='one functional state per target group; input BC counts in full kernel once; output target members once',
                crop_renormalized=False))
        blocks = [b for b in patch.state_blocks if b.population not in ('local_crossover_AC','widefield_AC')]
        for c in config['channels']:
            n = self.populations[c['pool']].group_count
            blocks += [StateBlock(c['pool'],MECHANISM,'V','V','effective_functional_membrane_voltage',n,0),
                       StateBlock(c['pool'],MECHANISM,'s','dimensionless','effective_inhibitory_activity',n,0)]
        self.state_blocks = []
        offset = 0
        for b in blocks:
            self.state_blocks.append(replace(b,offset=offset)); offset += b.count
        self.state_blocks = tuple(self.state_blocks)
        self.state_scalar_count = offset
        self.manifest = deepcopy(patch.manifest)
        self.manifest['ac_vs_model'] = dict(schema=SCHEMA,config=config,routes=self.route_ledger)
        self.artifact_sha256 = canonical_hash(dict(schema=SCHEMA,parent_patch=patch.artifact_sha256,
            config=config,routes=self.route_ledger,states=[vars(b) for b in self.state_blocks],
            implementation=digest(Path(__file__))))
        self.runtime_factories = dict(patch.runtime_factories,inner=ACVSInner,compiled=DirectACVSRHS)

    def __getattr__(self, name):
        return getattr(self.patch,name)

    def matrix(self, relative):
        return self._new_matrices[relative] if relative in self._new_matrices else self.patch.matrix(relative)


def redesign_seed(assembly, old_seed):
    seed = deepcopy(old_seed)
    seed['states'] = {k:v for k,v in seed['states'].items() if k[0] not in ('local_crossover_AC','widefield_AC')}
    parameters = seed['inner_parameters']
    gains = parameters['route_gains']
    for name in list(gains):
        if name not in assembly.routes: del gains[name]
    for c in assembly.ac_config['channels']:
        pop = c['pool']
        p = dict(assembly.ac_config['parameters_per_effective_pool_SI'])
        parameters['populations'][pop] = p
        pre_p = parameters['populations'][c['pre']]
        release = seed['states'][c['pre'],'HistoryBipolarTerminal','terminal_release']
        adaptation = seed['states'][c['pre'],'HistoryBipolarTerminal','A']
        r = release/(1+adaptation**6) if pre_p['adaptation_enabled'] else release
        gE = p['gE_base']+c['G_BA_S']*(assembly.matrix('routes/'+c['pre']+'_to_'+pop+'/mean_weights.npz')@r)
        V,_,s = equilibrium(gE,p)
        seed['states'][pop,MECHANISM,'V'] = V
        seed['states'][pop,MECHANISM,'s'] = s
        gains[pop+'_to_'+c['target']] = c['G_AR_S']
    parameters['identity'] = canonical_hash(dict(schema=SCHEMA,parent=parameters['identity'],config=assembly.ac_config))
    parameters['provenance'] = dict(parent=parameters.get('provenance'),ac_config=assembly.ac_config,
        release_boundary='dimensionless terminal_release/(1+A**6); conversion 1 fraction/fraction; no second sigmoid',
        initialization='new AC conditional equilibrium only; all other source IVP and S-ON states retained; no whole-network background qualification')
    seed['ac_vs_lineage'] = dict(schema=SCHEMA,artifact=assembly.artifact_sha256,parameter_identity=parameters['identity'],
        old_cache_reused=False,old_background_reused=False,source_IVP_restricted=True,coupled_background_qualified=False)
    return seed


class ACVSInner(BoundedInnerDynamics):
    port_unit_overrides = {'inhibitory_driving_force':'V', 'current':'A',
        'I_total':'A', 'expected_rate_Hz':'Hz', 'expected_hazard_Hz':'Hz',
        **{'gI_'+n:'S' for n in ('same_sign','crossover','AII_direct','legacy_S_ON')},
        **{'I_'+n:'A' for n in ('same_sign','crossover','AII_direct','legacy_S_ON')}}
    def __init__(self, assembly, parameters):
        self.channels = assembly.ac_config['channels']
        super().__init__(assembly,parameters)
        for c in self.channels:
            pop = c['pool']
            self.C[pop] = self.par(pop,'C',positive=True)  # One effective pool, not anatomical density.
            self.gL[pop] = self.par(pop,'gL',positive=True)
        self.identity = canonical_hash(dict(parent=self.identity,schema=SCHEMA,implementation=digest(Path(__file__))))

    def ac_outputs(self, state, out):
        pop = LEGACY_S
        h = state[key(pop,AC_MECH,'h')]
        drive = sum((self.mean[r.name]@out[r.pre,'effective_release'] for r in self.routes if r.post==pop),start=np.zeros(self.counts[pop]))
        out[pop,'drive'] = drive
        out[pop,'target_inhibition'] = expit(self.par(pop,'input_slope',positive=True)*(drive-self.par(pop,'reference_input')))
        out[pop,'inhibition_state'] = h
        out[pop,'activation'] = self.par(pop,'output_scale',positive=True)*h
        for c in self.channels:
            pop = c['pool']; r = out[c['pre'],'effective_release']
            if np.any(r<0) or np.any(r>1+1e-10):
                raise ContractError('BC effective_release must be a nonnegative dimensionless fraction')
            gE = self.par(pop,'gE_base',nonnegative=True)+c['G_BA_S']*(self.mean[c['pre']+'_to_'+pop]@r)
            V,s = state[key(pop,MECHANISM,'V')],state[key(pop,MECHANISM,'s')]
            if np.any(s < -1e-10) or np.any(s > 1+1e-10) or np.any(gE<0):
                raise ContractError('AC gate/conductance physical domain violated; no clipping')
            p = expit((V-self.par(pop,'theta'))/self.par(pop,'k',positive=True))
            out[pop,'gE'],out[pop,'voltage'],out[pop,'p'],out[pop,'s'] = gE,V,p,s
            out[pop,'activation'] = out[pop,'inhibition_state'] = s

    def ac_rhs(self, state, out, result):
        pop = LEGACY_S
        h = state[key(pop,AC_MECH,'h')]
        if np.any(h < -1e-10) or np.any(h > 1+1e-10):
            raise ContractError('Preserved S-ON AC gate domain violated')
        result[key(pop,AC_MECH,'h')] = (out[pop,'target_inhibition']-h)/self.par(pop,'tau_A_seconds',positive=True)
        for c in self.channels:
            pop = c['pool']; V,p,s,gE = (out[pop,n] for n in ('voltage','p','s','gE'))
            result[key(pop,MECHANISM,'V')] = (self.gL[pop]*(self.par(pop,'EL')-V)+gE*(self.par(pop,'EE')-V))/self.C[pop]
            result[key(pop,MECHANISM,'s')] = self.par(pop,'alpha',positive=True)*p*(1-s)-self.par(pop,'beta',positive=True)*s

    def outputs(self,t,state,cone_release=None):
        out = super().outputs(t,state,cone_release)
        for target in RGC:
            V = out[target,'voltage']; force = self.par(target,'EI')-V
            out[target,'inhibitory_driving_force'] = force
            out[target,'I_total'] = out[target,'gI']*force
            pi=np.column_stack([state[key(target,'RGC_expected_history','history_probability_'+str(i))] for i in range(8)])
            hazard=self.par(target,'alpha',positive=True)[:,None]*np.logaddexp(0.,
                (V[:,None]+(STATES@np.asarray(self.p[target]['spike_history_kernel']))[None,:]-self.par(target,'mu')[:,None])/self.par(target,'beta',positive=True)[:,None])
            out[target,'expected_hazard_Hz']=(pi*hazard).sum(axis=1)
            out[target,'expected_rate_Hz']=(pi*(-np.expm1(-.001*hazard))).sum(axis=1)/.001
            for relation in ('same_sign','crossover','AII_direct','legacy_S_ON'):
                g = np.zeros(self.counts[target])
                for c in self.channels:
                    if c['target']==target and c['relation']==relation:
                        route = c['pool']+'_to_'+target
                        g += self.members[target]*self.parameters['route_gains'][route]*(self.mean[route]@out[c['pool'],'s'])
                if relation=='legacy_S_ON' and target=='small_bistratified_S_ON': g = out[target,'gI'].copy()
                out[target,'gI_'+relation] = g
                out[target,'I_'+relation] = g*force
        for component in ('AII_FMB_glycine','AII_DB1_glycine'):
            target = self.contacts[component]['route'].post
            out[component,'current'] = out[component,'conductance']*(self.par(target,'EI')-out[target,'voltage'])
        return out

    def advance_reference(self,state,t0,t1,*,rtol,atol):
        started = perf_counter()
        solver = DOP853(lambda t,y:self.compiled_rhs(t,y),t0,self.pack(state),t1,
                        rtol=rtol,atol=atol,max_step=self.assembly.max_step)
        dense = []
        while solver.status=='running':
            message=solver.step()
            if solver.status=='failed': raise ContractError('AC V+s DOP853 failed: '+str(message))
            dense.append(Dop853Dense.from_scipy(solver.dense_output(),np.arange(self.nstate)))
        def curve(t,side):
            return next(d for d in dense if d.t0<=t<=d.t1 and (t!=d.t1 or side=='left' or t==t1))
        result = InnerAdvance(self,t0,t1,self.unpack(solver.y),lambda t,s:self.unpack(curve(t,s).evaluate(t,s)),{},
            dict(method='new_AC_Vs_CPU_DOP853',nfev=solver.nfev,wall_seconds=perf_counter()-started,max_step_seconds=self.assembly.max_step),
            accepted_times=tuple(sorted({t0,t1,*(d.t1 for d in dense)})),dense_segments=tuple(dense))
        result.representation_derivative = lambda t,side='right': self.unpack(curve(t,side).representation_derivative(t,side))
        return result

    def advance_candidate(self,*args,**kwargs):
        raise ContractError('New AC uses CPU joint DOP853 reference; legacy semianalytic/GPU candidate is incompatible')


class DirectACVSRHS:
    """CPU adapter for the existing runner; no old compiled AC opcode reuse."""
    def __init__(self,model,*,repo_root=None):
        self.model=model
        self.identity=canonical_hash(dict(schema=SCHEMA,model=model.identity,execution='direct_numpy_cpu_fp64',max_step=model.assembly.max_step))

    def rhs_flat(self,t,y,cone_release=None,rod_current=None):
        return self.model.pack(self.model.rhs(t,self.model.unpack(y,copy=False),cone_release,rod_current))

    def __call__(self,t,y):
        return self.rhs_flat(t,y)
