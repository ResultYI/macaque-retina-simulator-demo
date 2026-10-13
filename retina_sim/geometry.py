"""Source-bounded density, capacity-constrained mosaics, labels and displacement."""
from dataclasses import dataclass
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.interpolate import PchipInterpolator,CubicHermiteSpline
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from .contracts import ContractError,Quantity
from .density import read_density_tabulation

def keyed_rng(seed,key):
    digest=hashlib.sha256((str(seed)+'|'+str(key)).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8],'little')))

class PackerDensityField:
    def __init__(self,table,*,central_policy,angular_policy):
        if central_policy!='central_constant_cap' or angular_policy!='log_NT_cosine': raise ContractError('Named central/angular density approximations required')
        self.table=table;self.profiles={};self.knots={};self.policy={'interpolation':'log PCHIP project numerical choice, not author Rc interpolation','central_policy':central_policy,'angular_policy':angular_policy,'vertical_symmetry':'project_approximation','evidence_representation':'published_code_tabulation','asset_sha256':table.asset_sha256}
        for meridian in ('nasal','temporal'):
            rows=[r for r in table.rows if r['meridian']==meridian]
            x=np.array([float(r['eccentricity_mm']) for r in rows]);y=np.array([float(r['cone_density_cones_per_mm2']) for r in rows])
            if np.any(np.diff(x)<=0): raise ContractError('Conflicting duplicate density eccentricities; no averaging')
            self.profiles[meridian]=PchipInterpolator(x,np.log(y),extrapolate=False);self.knots[meridian]=x
        self.minimum=self.knots['nasal'][0]
    def radial(self,r,theta):
        r=np.asarray(r);theta=np.asarray(theta);w=(1+np.cos(theta))/2
        # Only contributing meridians constrain the query; no hidden 10mm engine limit.
        if np.any((w>1e-14)&(r>self.knots['nasal'][-1])) or np.any((w<1-1e-14)&(r>self.knots['temporal'][-1])): raise ContractError('Density query beyond applicable source support')
        n=self.profiles['nasal'](np.minimum(np.maximum(r,self.minimum),self.knots['nasal'][-1]));t=self.profiles['temporal'](np.minimum(np.maximum(r,self.minimum),self.knots['temporal'][-1]))
        return np.exp(w*n+(1-w)*t)
    def __call__(self,xy):
        a=np.asarray(xy);return self.radial(np.linalg.norm(a,axis=-1),np.arctan2(a[...,1],a[...,0]))

@dataclass(frozen=True)
class RetinalDomain:
    bounds_mm: tuple
    owner_bounds_mm: tuple
    mask: object
    mask_source_id: str
    label: str
    def __post_init__(self):
        x0,x1,y0,y1=self.bounds_mm
        if x1<=x0 or y1<=y0 or not callable(self.mask) or not self.mask_source_id or not self.label: raise ContractError('Explicit ROI/halo/mask domain required')
        a,b,c,d=self.owner_bounds_mm
        if not x0<=a<b<=x1 or not y0<=c<d<=y1: raise ContractError('Owner region must lie inside halo domain')
    def contains(self,xy,owner=False):
        a=np.asarray(xy);x0,x1,y0,y1=self.owner_bounds_mm if owner else self.bounds_mm
        return (a[...,0]>=x0)&(a[...,0]<x1)&(a[...,1]>=y0)&(a[...,1]<y1)&np.asarray(self.mask(a),dtype=bool)

def integrate_density(field,domain,relative_tolerance=1e-4,*,allow_zero=False):
    x0,x1,y0,y1=domain.bounds_mm
    def column(x):
        value,error=quad(lambda y:float(field(np.array([[x,y]]))[0]) if bool(domain.mask(np.array([[x,y]]))[0]) else 0,y0,y1,epsabs=relative_tolerance*.1,epsrel=relative_tolerance*.1,limit=200)
        return value
    mass,error=quad(column,x0,x1,epsabs=relative_tolerance*.1,epsrel=relative_tolerance*.1,limit=200)
    if not np.isfinite(mass) or mass<0 or (mass==0 and not allow_zero): raise ContractError('Domain has no finite positive density mass')
    return {'mass':float(mass),'estimated_absolute_quadrature_error':float(error),'rounding_target':int(np.floor(mass+.5)),'relative_tolerance':relative_tolerance,'domain':domain.label}

def balanced_round(weights,total,capacities=None):
    w=np.asarray(weights,dtype=float)
    if np.any(w<0) or not np.all(np.isfinite(w)) or total<0 or (w.sum()<=0 and total>0): raise ContractError('Invalid quota mass')
    capacities=np.full(len(w),total,dtype=int) if capacities is None else np.asarray(capacities,dtype=int)
    if capacities.shape!=w.shape or np.any(capacities<0) or total>capacities.sum(): raise ContractError('Quota exceeds eligible capacity')
    allocation=np.zeros(len(w),dtype=int);remaining=int(total)
    while remaining:
        eligible=allocation<capacities
        if not np.any(w[eligible]>0): raise ContractError('Cannot meet quota without violating spatial eligibility')
        expected=remaining*w[eligible]/w[eligible].sum();indices=np.flatnonzero(eligible)
        base=np.minimum(np.floor(expected).astype(int),capacities[indices]-allocation[indices]);allocation[indices]+=base;remaining-=int(base.sum())
        if remaining:
            order=indices[np.argsort(-(expected-np.floor(expected)),kind='stable')]
            for i in order:
                if remaining and allocation[i]<capacities[i] and w[i]>0:allocation[i]+=1;remaining-=1
    return allocation

@dataclass(frozen=True)
class Mosaic:
    positions: Quantity
    cell_ids: tuple
    owner: np.ndarray
    diagnostics: dict

def partition_mosaic(master,domain):
    """Views share master coordinates/IDs; overlapping halos never regenerate cells.

    Ownership is half-open. Partitioning changes exported/active view, not density.
    Callers must build a source-covered master domain before adjoining partitions.
    Independent ROI mosaics are not advertised as seam-consistent global anatomy.
    """
    xy=master.positions.values;selected=domain.contains(xy);points=xy[selected]
    ids=tuple(i for i,keep in zip(master.cell_ids,selected) if keep)
    return Mosaic(Quantity(points,master.positions.unit,master.positions.meaning,master.positions.evidence_tag,master.positions.coordinate_frame),ids,domain.contains(points,owner=True),dict(master.diagnostics,partition_label=domain.label,master_cell_count=len(master.cell_ids),partition_count=len(ids),partition_policy='shared immutable master geometry and cell IDs; no resampling',owner_count=int(domain.contains(points,owner=True).sum())))

def capacity_constrained_mosaic(field,domain,seed,*,quadrature_shape,iterations,max_bytes=512*1024*1024):
    """Sparse greedy capacitated transport approximation, NOT exact power-diagram OT.

    Every point owns equal quadrature rho-mass; ordinary weighted Lloyd is absent.
    Exact Laguerre dual optimization would require a separate scalable solver.
    This declared approximation preserves capacities and audits geometry/counts.
    """
    audit=integrate_density(field,domain);n=audit['rounding_target']
    if iterations<1:raise ContractError('Capacity relaxation requires at least one declared iteration')
    nx,ny=quadrature_shape;qn=nx*ny
    estimate=qn*80+n*96
    if n<1 or qn<4*n or estimate>max_bytes: raise ContractError(f'Mosaic resource/quadrature requirement n={n}, quadrature={qn}, estimate={estimate}; enlarge budget/declare ROI, no count reduction')
    x0,x1,y0,y1=domain.bounds_mm
    x=x0+(np.arange(nx)+.5)*(x1-x0)/nx;y=y0+(np.arange(ny)+.5)*(y1-y0)/ny
    points=np.array(np.meshgrid(x,y,indexing='ij')).reshape(2,-1).T;points=points[domain.contains(points)]
    mass=np.asarray(field(points))*((x1-x0)/nx)*((y1-y0)/ny)
    relative_error=abs(float(mass.sum())-audit['mass'])/audit['mass']
    if relative_error>audit['relative_tolerance']: raise ContractError(f'Mosaic quadrature mass residual {relative_error} exceeds declared1e-4; increase quadrature')
    mass*=audit['mass']/mass.sum();capacity=audit['mass']/n
    # Equal-mass stratified initialization with keyed geometry disorder, not a PPP.
    cumulative=np.cumsum(mass);targets=(np.arange(n)+.5)*capacity
    locations=points[np.minimum(np.searchsorted(cumulative,targets),len(points)-1)].copy()
    rng=keyed_rng(seed,'geometry:'+domain.label)
    locations+=(rng.random(locations.shape)-.5)*np.array([(x1-x0)/nx,(y1-y0)/ny])*.1
    worst=0
    for iteration in range(iterations):
        tree=cKDTree(locations);remaining=np.full(n,capacity);moments=np.zeros((n,2));assigned=np.zeros(n)
        for j in rng.permutation(len(points)):
            amount=mass[j];k=min(8,n)
            while amount>capacity*1e-12:
                distances,candidates=tree.query(points[j],k=k);candidates=np.atleast_1d(candidates)
                usable=[i for i in candidates if remaining[i]>capacity*1e-12]
                if not usable:
                    if k==n: raise ContractError('Capacity allocation exhausted numerically')
                    k=min(n,k*2);continue
                for i in usable:
                    take=min(amount,remaining[i]);moments[i]+=take*points[j];assigned[i]+=take;remaining[i]-=take;amount-=take
                    if amount<=capacity*1e-12:break
        worst=float(np.max(np.abs(assigned-capacity))/capacity)
        if worst>1e-8: raise ContractError('Equal-mass transport capacity residual')
        locations=moments/assigned[:,None]
    if not np.all(domain.contains(locations)): raise ContractError('Centroid escaped mask/ROI; subdivide nonconvex domain')
    order=np.lexsort((locations[:,1],locations[:,0]));locations=locations[order]
    nearest=cKDTree(locations).query(locations,k=min(2,n))[0]
    nn=nearest[:,1] if n>1 else np.array([np.nan]);scaled=nn*np.sqrt(field(locations))
    if n>1 and np.any(nn<=0): raise ContractError('Duplicate cone points after capacity relaxation')
    ids=tuple(f'{domain.label}:cone:{i:09d}' for i in range(n));owner=domain.contains(locations,owner=True)
    diagnostics=dict(audit,algorithm='greedy_capacitated_transport_equal_rho_mass',exact_power_diagram=False,count_minus_density_mass=float(n-audit['mass']),maximum_integer_rounding_error=.5,algorithm_evidence='project_approximation; sparse neighborhood search, not exact Laguerre power cells',why='Available SciPy/NumPy implementation preserves target capacities without an unbounded dense LP; exact scalable transport solver not installed; ordinary rho-weighted Lloyd rejected',quadrature_mass_relative_residual=relative_error,maximum_capacity_relative_residual=worst,seed=seed,iterations=iterations,owner_count=int(owner.sum()),halo_count=int((~owner).sum()),minimum_nn_mm=float(np.nanmin(nn)) if n>1 else None,scaled_nn_cv=float(np.std(scaled)/np.mean(scaled)) if n>1 else None,axes_std_mm=np.std(locations,axis=0).tolist(),resource_estimate_bytes=estimate)
    return Mosaic(Quantity(locations,'mm','functional_position','project_approximation','functional_visual_field'),ids,owner,diagnostics)

class MosaicBackend:
    """Replaceable equal-mass sampler ABI, retaining count/owner/evidence audits."""
    def __init__(self):
        self.backends={'greedy_capacitated_equal_rho_mass':(capacity_constrained_mosaic,False,'project_approximation')}
    def register(self,name,implementation,*,exact_power_diagram,evidence_class):
        if not name or name in self.backends or not callable(implementation) or evidence_class not in ('project_approximation','source_verified_algorithm'):raise ContractError('Explicit executable mosaic backend and algorithm classification required')
        self.backends[name]=(implementation,bool(exact_power_diagram),evidence_class)
    def generate(self,name,field,domain,seed,**options):
        if name not in self.backends:raise ContractError('Mosaic backend is not implemented/registered')
        callback,exact,evidence=self.backends[name];mosaic=callback(field,domain,seed,**options);audit=integrate_density(field,domain)
        if not isinstance(mosaic,Mosaic) or len(mosaic.cell_ids)!=audit['rounding_target'] or len(set(mosaic.cell_ids))!=len(mosaic.cell_ids):raise ContractError('Sampler changed density-derived count or stable unique IDs')
        mosaic.positions.require('mm','functional_position',(len(mosaic.cell_ids),2),'functional_visual_field')
        if mosaic.owner.shape!=(len(mosaic.cell_ids),) or not np.all(domain.contains(mosaic.positions.values)):raise ContractError('Sampler violated declared ROI/owner shape')
        return Mosaic(mosaic.positions,mosaic.cell_ids,mosaic.owner,dict(mosaic.diagnostics,backend_name=name,exact_power_diagram=exact,algorithm_evidence_class=evidence,count_minus_density_mass=float(len(mosaic.cell_ids)-audit['mass']),maximum_integer_rounding_error=.5))

class SConeDensityField:
    def __init__(self,source,mm_per_deg,*,magnification_source_id,central_policy):
        if mm_per_deg<=0 or not magnification_source_id or central_policy not in ('assay_square_PCHIP','no_hard_mask'): raise ContractError('Explicit S assay/magnification/profile required')
        self.source=source;self.scale=mm_per_deg;self.magnification_source_id=magnification_source_id;self.policy=central_policy
    def _direction(self,e,theta,name):
        a,b,c,d=self.source[name];fit=lambda r:a*np.exp(b*r)+c*np.exp(d*r)
        if self.policy=='no_hard_mask':
            if np.any(e<2): raise ContractError('No-mask source fit does not define central extrapolation')
            return fit(e)
        boundary=.075/np.maximum(np.abs(np.cos(theta)),np.abs(np.sin(theta)))
        peak,density,sd=self.source['central_peaks'][name]
        result=np.empty_like(e,dtype=float)
        for index in np.ndindex(e.shape):
            if e[index]<=boundary[index]:result[index]=0
            elif e[index]<2:result[index]=PchipInterpolator([boundary[index],peak,2],[0,density,fit(2)])(e[index])
            else:result[index]=fit(e[index])
        return result
    def __call__(self,xy):
        a=np.asarray(xy);e=np.linalg.norm(a,axis=-1)/self.scale;theta=np.arctan2(a[...,1],a[...,0])
        if np.any(e>60): raise ContractError('S source fit exceeds60deg support')
        w=(1+np.cos(theta))/2;n=self._direction(e,theta,'nasal');t=self._direction(e,theta,'temporal')
        return (w*n+(1-w)*t)/self.scale**2

def assign_cone_types(mosaic,total_density,s_density,domain,LM_ratio,ratio_source_id,seed,tile_size_mm):
    if LM_ratio<=0 or not ratio_source_id or tile_size_mm<=0: raise ContractError('Named L/M ratio and label tiling required')
    xy=mosaic.positions.values;prob=np.asarray(s_density(xy))/np.asarray(total_density(xy))
    if np.any(prob<0) or np.any(prob>1): raise ContractError('S density exceeds total density; no clipping')
    audit=integrate_density(s_density,domain,allow_zero=True);ns=int(np.floor(audit['mass']+.5))
    keys=np.floor((xy-np.array(domain.bounds_mm)[[0,2]])/tile_size_mm).astype(int)
    groups={tuple(k):np.flatnonzero(np.all(keys==k,axis=1)) for k in np.unique(keys,axis=0)}
    quotas=balanced_round([prob[v].sum() for v in groups.values()],ns,[int(np.sum(prob[v]>0)) for v in groups.values()])
    selected=[]
    for (key,indices),quota in zip(groups.items(),quotas):
        candidates=indices[prob[indices]>0];rng=keyed_rng(seed,'S:'+str(key));score=-np.log(np.maximum(rng.random(len(candidates)),np.finfo(float).tiny))/prob[candidates]
        selected.extend(candidates[np.argsort(score)[:quota]].tolist())
    labels=np.full(len(xy),'M',dtype='<U1');labels[selected]='S';nonS=np.flatnonzero(labels!='S');nl=int(np.floor(len(nonS)*LM_ratio/(1+LM_ratio)+.5))
    quotasL=balanced_round([np.sum(labels[v]!='S') for v in groups.values()],nl,[np.sum(labels[v]!='S') for v in groups.values()])
    for (key,indices),quota in zip(groups.items(),quotasL):
        candidates=indices[labels[indices]!='S'];rng=keyed_rng(seed,'LM:'+str(key));labels[rng.permutation(candidates)[:quota]]='L'
    return labels,{'Ns':ns,'NL':nl,'NM':int(np.sum(labels=='M')),'pS':prob.tolist(),'pL_given_nonS':LM_ratio/(1+LM_ratio),'ratio_source_id':ratio_source_id,'algorithm':'balanced fixed-tile quotas with keyed conditional labels','evidence':'S assay + named L/M record; spatial label independence is project approximation'}

class McGregorDisplacement:
    cardinal={'nasal':0.,'superior':np.pi/2,'temporal':np.pi,'inferior':3*np.pi/2}
    def __init__(self,nodes,density,*,profile_name,join_fraction=.01):
        if profile_name!='published_rounded_anchor_composite' or not 0<join_fraction<.1: raise ContractError('Explicit rounded-anchor map and smoothing policy required')
        self.nodes=nodes;self.density=density;self.curves={};self.diagnostics={};self.join_fraction=join_fraction
        self.P=lambda u:18825*u*u/2+8420*u*u*u/3
        for name,theta in self.cardinal.items():
            node=nodes['production_nodes'][name]
            if node['anchor_profile']!='published-rounded-f40': raise ContractError('Do not mix refitted/published-rounded anchors')
            central=PchipInterpolator(node['r_mm'],node['soma_radius_mm']);ra=.04;pa=ra+.85*(float(central(ra))-ra)
            knots=sorted(set(np.concatenate(list(density.knots.values()))))
            def mass(r):return quad(lambda s:float(density.radial(s,theta))*s,ra,r,points=[x for x in knots if ra<x<r],epsabs=1e-5,epsrel=1e-7,limit=200)[0]
            crossing=brentq(lambda r:self.P(r)-self.P(pa)-mass(r),.5,min(4.,min(x[-1] for x in density.knots.values())),xtol=1e-10)
            outer_r=np.linspace(ra,crossing,256);outer_mass=np.array([mass(r) for r in outer_r])
            outer_p=np.array([brentq(lambda u:self.P(u)-self.P(pa)-m,pa,4.,xtol=1e-11) for m in outer_mass])
            outer_g=outer_r+(outer_p-outer_r)/.85
            # PCHIP retains every source-derived central node and has continuous slope.
            r=np.r_[node['r_mm'],outer_r[1:]];g=np.r_[node['soma_radius_mm'],outer_g[1:]]
            curve=PchipInterpolator(r,g);lo=crossing*(1-join_fraction)
            join=CubicHermiteSpline([lo,crossing],[curve(lo),crossing],[curve.derivative()(lo),1.])
            self.curves[name]=(curve,join,lo,crossing)
            grid=np.linspace(1e-6,crossing,1000);der=np.where(grid<lo,curve.derivative()(grid),join.derivative()(grid))
            if np.min(der)<=0: raise ContractError('Displacement fold after finite join')
            smooth_g=np.where(outer_r<lo,curve(outer_r),join(outer_r));smooth_p=outer_r+.85*(smooth_g-outer_r)
            count_error=self.P(smooth_p)-self.P(outer_p)
            self.diagnostics[name]={'catchup_mm':crossing,'minimum_dR_dr':float(der.min()),'pedicle_anchor_mm':pa,'source_anchors_preserved':True,'join_start_mm':lo,'join_evidence':'project finite C1 join; not measured','maximum_join_count_error_on_source_grid':float(np.max(np.abs(count_error))),'maximum_relative_join_count_error_on_source_grid':float(np.max(np.abs(count_error)/np.maximum(self.P(outer_p),1))),'profile':profile_name}
    def _cardinal(self,name,r,derivative=False):
        curve,join,lo,catch=self.curves[name];f=curve.derivative() if derivative else curve;j=join.derivative() if derivative else join
        return np.where(r>=catch,1. if derivative else r,np.where(r>=lo,j(r),f(r)))
    def radial(self,r,theta,derivative=False):
        theta=np.mod(theta,2*np.pi);index=np.floor(theta/(np.pi/2)).astype(int);fraction=(theta-index*np.pi/2)/(np.pi/2);names=('nasal','superior','temporal','inferior');out=np.zeros(np.broadcast(r,theta).shape)
        for i in range(4):
            selected=index==i
            out=np.where(selected,(1-fraction)*self._cardinal(names[i],r,derivative)+fraction*self._cardinal(names[(i+1)%4],r,derivative),out)
        return out
    def apply(self,positions,origin_seed=None):
        xy=positions.require('mm','functional_position',frame='functional_visual_field');r=np.linalg.norm(xy,axis=1);theta=np.arctan2(xy[:,1],xy[:,0])
        if np.any(r==0):
            if origin_seed is None: raise ContractError('Exact origin has no unique displaced angle; explicit single-point convention required')
            theta=np.where(r==0,keyed_rng(origin_seed,'origin_displacement').uniform(0,2*np.pi,len(r)),theta)
        R=self.radial(r,theta);P=r+.85*(R-r);unit=np.stack((np.cos(theta),np.sin(theta)),axis=1)
        J=np.full(len(r),np.nan);nz=r>0;J[nz]=R[nz]/r[nz]*self.radial(r[nz],theta[nz],True)
        if np.any(J[nz]<=0): raise ContractError('Nonpositive displacement area Jacobian')
        return {'soma':Quantity(unit*R[:,None],'mm','anatomical_position','project_approximation_source_derived','anatomical_retina'),'pedicle':Quantity(unit*P[:,None],'mm','pedicle_position','project_approximation_source_derived','OPL'),'jacobian':J,'input_cell_count':len(r),'output_cell_count':len(r),'evidence':'McGregor processed anchors + Packer cross-species/source equal-count extension;15% author estimated postreceptoral share','diagnostics':self.diagnostics,'origin_singular':bool(np.any(~nz))}
