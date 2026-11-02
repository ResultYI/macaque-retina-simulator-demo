"""User-selected v2 HC soma density; engineering extent is not validation."""
from pathlib import Path
import json
import numpy as np
from .contracts import ContractError,canonical_hash
from .hc_mosaic import HCDensityProfile,_radial_rectangle_range
from .hc_current import JOINT_ID,current_hc,current_hc_configuration,file_hash

class JointV2HCDensityProfile(HCDensityProfile):
    def __init__(self,root,profile_path=None,profile_data=None):
        root=Path(root)
        # Retain only the existing mosaic pair/generator conventions and source records.
        super().__init__(root,profile_path=root/'configurations/hc_joint_synthetic_v1.json')
        self.profile_path=Path(profile_path) if profile_path is not None else current_hc_configuration(root)
        if self.profile_path is None:raise ContractError('Current HC v2 configuration missing')
        if not self.profile_path.is_absolute():self.profile_path=root/self.profile_path
        self.config=json.loads(self.profile_path.read_bytes()) if profile_data is None else json.loads(json.dumps(profile_data,allow_nan=False))
        if self.config.get('profile_id')!=JOINT_ID:raise ContractError('HC v2 explicit profile identity required')
        self.coefficients=np.array([v['c'] for v in self.config['density']['coefficients']],dtype=np.float64)
        self.lambdas=np.array([v['lambda_per_mm'] for v in self.config['density']['coefficients']],dtype=np.float64)
        if self.coefficients.shape!=(3,) or not np.isfinite(self.coefficients).all() or not np.isfinite(self.lambdas).all():raise ContractError('Three finite frozen HC coefficients required')
        self.profile_id=JOINT_ID;self.support_radius_mm=float(self.config['split']['application_scope_mm'][1]);self.peak_radius_mm=.6
        self.central_fraction=float(self.config['split']['rho_H1_fraction']);self.h2_fraction=float(self.config['split']['rho_H2_fraction'])
        if self.central_fraction+self.h2_fraction!=1.:raise ContractError('Frozen HC v2 subtype split required')
        extension=self.config.get('owner_support')
        if self.support_radius_mm!=1.338 and (not extension or
                extension.get('authorization')!='Sentinel_5a05022065f081919c81f4f9db2d6697' or
                extension.get('owner_RF_radius_mm')!=1.338 or extension.get('HC_support_radius_mm')!=self.support_radius_mm):
            raise ContractError('Explicit owner/support engineering extension required')
        self.nodes_mm=np.array([.6,self.support_radius_mm]);self.source_path=self.profile_path
        self.source_asset_hash=file_hash(self.profile_path);self.raw_profile_asset_hash=self.source_asset_hash
        self.current_selection=current_hc(root)
        self.profile_hash=canonical_hash(dict(configuration=self.config,current_selection=None if self.current_selection is None else self.current_selection['identity']))
        self.center_total=float(self.radial(0.));self.peak_total=float(self.radial(.6))
        self.metadata=dict(profile_id=self.profile_id,profile_sha256=self.profile_hash,source_asset_sha256=self.source_asset_hash,
            current_hc_identity=None if self.current_selection is None else self.current_selection['identity'],
            unit='cells/mm^2',coordinate_frame='anatomical_retina',plane='HC soma physical projection',radial_support_mm=[0.,self.support_radius_mm],
            split_status='0.8/0.2 engineering extension; not subtype density fit or physiological validation',owner_support=extension,
            source=self.config['source'],source_risks=self.config['source_risks'],density_family='Three signed exponentials with supplied v2 coefficients; no refit',
            contact_join=self.config['geometry']['central_join'],
            origin='Nonzero radial derivative retained; no smoothing at origin',physiological_validation=False,FIT_READY=False,background_qualified=False,
            generator_policy='Existing joint mosaic/pair mechanics retained; new coordinates/IDs require explicit construction')

    def radial(self,radius_mm):
        r=self._radius(radius_mm)
        result=np.exp(r[...,None]*self.lambdas)@self.coefficients
        if not np.isfinite(result).all() or np.any(result<=0):raise ContractError('HC v2 density nonpositive/nonfinite; no clipping or floor')
        return result

    def derivative(self,radius_mm):
        r=self._radius(radius_mm);return np.exp(r[...,None]*self.lambdas)@(self.coefficients*self.lambdas)

    def density_at_soma_eccentricity_deg(self,eccentricity_deg):
        e=np.asarray(eccentricity_deg,dtype=np.float64)
        maximum=self.support_radius_mm/.223
        if not np.isfinite(e).all() or np.any(e<0) or np.any(e>maximum):raise ContractError('HC soma-equivalent angular radius outside explicitly bound physical support; not source RF owner radius')
        r=e*.223
        # Only the already-validated exact angular endpoint may round one ULP.
        return self.radial(np.where(e==maximum,self.support_radius_mm,r))

    def radial_fraction(self,radius_mm):
        return np.full_like(self._radius(radius_mm),self.central_fraction)

    def type_density(self,xy,cell_type):
        if cell_type not in ('H1','H2'):raise ContractError('HC subtype must be H1/H2')
        total=self.total(xy);h1=total*self.central_fraction
        return h1 if cell_type=='H1' else total-h1

    def bounds_for_rectangle(self,bounds):
        lower,upper=_radial_rectangle_range(bounds);self._radius(np.array([lower,upper]))
        nodes=[lower,upper]+([.6] if lower<=.6<=upper else []);values=self.radial(nodes)
        return dict(total_min=float(values.min()),total_max=float(values.max()),
            H1_min=float(values.min()*.8),H1_max=float(values.max()*.8),H2_min=float(values.min()*.2),H2_max=float(values.max()*.2))

    def _outer_contact(self,r,cell_type):
        density=self.radial(r);derivative=self.derivative(r);fraction=.8 if cell_type=='H1' else .2
        divergence=3.7+(7.5-3.7)*(r-.6)/(5.5-.6) if cell_type=='H1' else np.full_like(r,2.2)
        d_divergence=(7.5-3.7)/(5.5-.6) if cell_type=='H1' else 0.
        radius=np.sqrt(divergence/(np.pi*fraction*density))
        return radius,.5*radius*(d_divergence/divergence-derivative/density)

    def _contact_join(self,r,cell_type,*,derivative=False):
        if cell_type not in ('H1','H2'):raise ContractError('HC subtype must be H1/H2')
        r=self._radius(r);a,b=self.config['geometry']['central_join']['interval_mm'];width=b-a
        central=float(self.config['geometry']['central_radii_mm'][cell_type])
        result=np.full_like(r,0. if derivative else central);take=r>=b
        if np.any(take):
            values,slopes=self._outer_contact(r[take],cell_type);result[take]=slopes if derivative else values
        inside=(r>a)&(r<b)
        if np.any(inside):
            endpoint,slope=self._outer_contact(np.asarray(b),cell_type);t=(r[inside]-a)/width
            if derivative:
                result[inside]=((6*t*t-6*t)*central+(-6*t*t+6*t)*endpoint+(3*t*t-2*t)*width*slope)/width
            else:
                result[inside]=(2*t**3-3*t*t+1)*central+(-2*t**3+3*t*t)*endpoint+(t**3-t*t)*width*slope
        if not np.isfinite(result).all() or (not derivative and np.any(result<=0)):
            raise ContractError('HC contact join nonfinite/nonpositive; no floor or parameter adjustment')
        return result

    def contact_radius_mm(self,radius_mm,cell_type):
        return self._contact_join(radius_mm,cell_type)

    def contact_radius_derivative_mm(self,radius_mm,cell_type):
        return self._contact_join(radius_mm,cell_type,derivative=True)

    def expected_disk_count(self,radius_mm):
        R=float(self._radius(radius_mm));k=-self.lambdas;x=k*R
        factor=np.where(abs(x)<1e-3,R*R*(.5-x/3+x*x/8-x**3/30+x**4/144),(-np.expm1(-x)-x*np.exp(-x))/(k*k))
        return float(2*np.pi*(factor@self.coefficients))
