"""User-approved rod central completion and independent parasol RF population.

Density evidence, project completion and engineering position sampling are
separate. Neither field changes arbors, conductance gains or dynamics.
"""
from types import MappingProxyType
import numpy as np
from scipy.interpolate import PchipInterpolator,PPoly
from .atlas_sources import AtlasSourceBundle,MERIDIANS
from .contracts import Quantity,ContractError,canonical_hash

APPROVAL='Sentinel_28318027a0ec8191a8ead57b6439c9c1'
VANNI_COMMIT='08aea38003db41c4753990038ed8ac9a75b2706a'


class ApprovedPackerRodDensity:
    def __init__(self,root):
        self.bundle=AtlasSourceBundle(root)
        self.rows=MappingProxyType({m:tuple(sorted((dict(r) for r in self.bundle.data['photoreceptor_points']
            if r['cell_class']=='rod' and r['meridian']==m),key=lambda r:r['eccentricity_mm'])) for m in MERIDIANS})
        self.curves={m:PchipInterpolator([r['eccentricity_mm'] for r in rows],
            [r['density_per_mm2'] for r in rows],extrapolate=False) for m,rows in self.rows.items()}
        self.metadata={'provider_id':'Packer1989_approved_rod_quadratic_PCHIP_cos2_v1','approval':APPROVAL,
            'species':'Macaca nemestrina','source':'Packer1989 Fig9 author-map digitization, not raw counting data',
            'source_asset':self.bundle.manifest()['source_assets']['photoreceptor_points'],
            'central':'rho1*(r/r1)^2 for each cardinal below its actual first node; origin0; no finite hard hole',
            'outer':'non-log PCHIP of unchanged density nodes; no extrapolation or added peaks',
            'angular':'adjacent cardinal cos²(phi)/sin²(phi) convex blend; +x nasal/+y superior',
            'join':'C0 only; first-point and digitization uncertainty retained; no tuned smoothing',
            'application_frame':'physical retinal mm -> nominal planar RF/physical chart project registration',
            'first_nodes':{m:rows[0] for m,rows in self.rows.items()},'physiological_validation':False}
        self.profile_hash=canonical_hash(self.metadata)

    def __call__(self,positions):
        if isinstance(positions,Quantity):
            if positions.coordinate_frame=='functional_visual_field':positions.require('mm','functional_position',frame='functional_visual_field')
            else:positions.require('mm','anatomical_position',frame='anatomical_retina')
            xy=positions.values
        else:xy=np.asarray(positions,dtype=float)
        if xy.ndim!=2 or xy.shape[1]!=2 or not np.all(np.isfinite(xy)):raise ContractError('Rod density needs finite Nx2 mm coordinates')
        radius=np.linalg.norm(xy,axis=1);theta=np.mod(np.arctan2(xy[:,1],xy[:,0]),2*np.pi)
        sector=np.floor(theta/(np.pi/2)).astype(int)%4;phi=theta-sector*(np.pi/2)
        result=np.zeros(len(xy))
        for side,weight in ((0,np.cos(phi)**2),(1,np.sin(phi)**2)):
            for k,m in enumerate(MERIDIANS):
                mask=((sector+side)%4==k)&(weight>1e-14)&(radius>0)
                if not np.any(mask):continue
                rr=radius[mask];rows=self.rows[m];r1=rows[0]['eccentricity_mm'];rho1=rows[0]['density_per_mm2']
                if np.any(rr>rows[-1]['eccentricity_mm']):raise ContractError('Packer rod outer source domain unsupported; no extrapolation')
                values=np.empty(len(rr));central=rr<r1
                values[central]=rho1*(rr[central]/r1)**2
                values[~central]=self.curves[m](rr[~central]);result[mask]+=weight[mask]*values
        if np.any(result<0) or not np.all(np.isfinite(result)):raise ContractError('Approved rod field became invalid')
        return result

    def disk_mass(self,radius):
        """Exact angular integral and polynomial radial integral of this field."""
        radius=float(radius)
        if not np.isfinite(radius) or radius<=0:raise ContractError('Positive disk radius required')
        total=0.
        for m,rows in self.rows.items():
            first=rows[0]['eccentricity_mm'];rho=rows[0]['density_per_mm2']
            if radius>rows[-1]['eccentricity_mm']:raise ContractError('Rod disk exceeds source domain')
            total+=rho*min(radius,first)**4/(4*first**2)
            if radius>first:
                c=self.curves[m].c;x=self.curves[m].x[:-1]
                # Multiply each local cubic in (r-x) by r=(r-x)+x.
                weighted=np.vstack((c[0],c[1]+x*c[0],c[2]+x*c[1],c[3]+x*c[2],x*c[3]))
                total+=float(PPoly(weighted,self.curves[m].x,extrapolate=False).integrate(first,radius))
        # Each cardinal contributes pi/2 across its two adjacent quadrants.
        return float(np.pi/2*total)


class ApprovedParasolRFDensity:
    def __init__(self,polarity):
        if polarity not in ('ON','OFF'):raise ContractError('Explicit parasol polarity required')
        self.polarity=polarity
        self.metadata={'provider_id':'Vanni2026_parasol_RF_sampling_approved_v1','polarity':polarity,
            'DOI':'10.64898/2026.03.09.710551','author_commit':VANNI_COMMIT,'approval':APPROVAL,
            'D':'580000*exp(-4.2*r)+56000*exp(-.56*r)','per_polarity_density':'0.04*D',
            'combined_density':'0.08*D; not 8 percent of every project RGC population',
            'frame':'nominal RF-plane mm; cone-derived author RF sampling model, not soma density',
            'source_level':'author preprint/model; no independent physiological validation',
            'project_transfers':['ON/OFF equal split','temporal model radial transfer to four directions'],
            'supported_radius_mm':[0,3],'position_sampler':'existing fixed-domain equal-mass engineering algorithm',
            'physiological_validation':False}
        self.profile_hash=canonical_hash(self.metadata)

    def __call__(self,positions):
        if isinstance(positions,Quantity):xy=positions.require('mm','functional_position',frame='functional_visual_field')
        else:xy=np.asarray(positions,dtype=float)
        if xy.ndim!=2 or xy.shape[1]!=2 or not np.all(np.isfinite(xy)):raise ContractError('Parasol RF density needs finite Nx2 mm coordinates')
        r=np.linalg.norm(xy,axis=1)
        if np.any(r>3):raise ContractError('Parasol approved radial application is limited to0..3mm')
        return .04*(580000*np.exp(-4.2*r)+56000*np.exp(-.56*r))

    def disk_mass(self,radius):
        from scipy.integrate import quad
        radius=float(radius)
        if not 0<radius<=3:raise ContractError('Parasol disk must remain inside approved0..3mm')
        value,error=quad(lambda r:r*float(self(np.array([[r,0.]]))[0]),0,radius,epsabs=1e-9,epsrel=1e-12)
        return float(2*np.pi*value)
