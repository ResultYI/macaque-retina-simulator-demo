"""Approved C0 linear-P endpoint bridges around the unchanged Schein source.

Only positive-radius 2D inputs have a direction. The exact origin is a radial
limit, never an implicitly chosen nasal point or a measured individual cone.
"""
from pathlib import Path
import hashlib,json
from types import MappingProxyType
import numpy as np

from .contracts import ContractError,Quantity
from .henle import ScheinHenleDisplacement

PROFILE='Schein1988_M1_bounded_bridge_v1'


class ScheinBoundedBridge(ScheinHenleDisplacement):
    def __init__(self,root,**kwargs):
        super().__init__(root,**kwargs)
        path=Path(root)/'configurations'/(PROFILE+'.json')
        raw=path.read_bytes();self.bridge=json.loads(raw);self.bridge_sha256=hashlib.sha256(raw).hexdigest()
        if (self.bridge['profile_id']!=PROFILE or self.bridge['source_curve_sha256']!=self.provenance['verified_asset_hashes']['schein1988_fig15_digitized_curves.json']):
            raise ContractError('Named Schein bridge must retain the frozen original curve')
        self.limit=float(self.bridge['central_radial_limit_P_mm']);self.maximum=float(self.bridge['application_max_r_mm'])
        nodes={}
        for m,c in self.curves.items():
            zero=float(self.bridge['zero_displacement_r_mm'][m])
            r=np.r_[0.,c.inner_segment_radius_mm,zero,self.maximum]
            p=np.r_[self.limit,c.inner_segment_radius_mm+c.displacement_mm,zero,self.maximum]
            if c.domain_mm[0]!=self.bridge['central_join_r_mm'] or np.any(np.diff(r)<=0) or np.any(np.diff(p)<=0) or np.any(p<r):
                raise ContractError('Schein bridge must be continuous, strictly increasing P>=r; no slope clipping')
            r.setflags(write=False);p.setflags(write=False);nodes[m]=(r,p)
        self.nodes=MappingProxyType(nodes)

    @property
    def provenance(self):
        source=super().provenance
        if not hasattr(self,'bridge'):return source
        return dict(source,profile_id=PROFILE,source_version='Schein1988_M1_bounded_bridge_v1',
            bridge_configuration_sha256=self.bridge_sha256,bridge_contract=self.bridge,
            application_domain='0<r<=6mm; r=0 radial limit with undefined direction',
            continuity='C0 piecewise linear P; slope jumps retained, not C1',
            bridge_evidence_status='project_approximation',strict_source_retained=True,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def _radial(self,r,weights):
        return sum(w*np.interp(r,*self.nodes[m]) for m,w in weights)

    def _inverse_radial(self,p,weights):
        r=np.unique(np.concatenate([self.nodes[m][0] for m,_ in weights]))
        mapped=self._radial(r,weights)
        if np.any(np.asarray(p)<self.limit-1e-12) or np.any(np.asarray(p)>self.maximum+1e-12):
            raise ContractError('Schein bridge inverse outside [.15,6]mm mapped radial domain')
        return np.interp(p,mapped,r)

    def radial_limit(self):
        return {'pedicle_radius_mm':self.limit,'status':'DIRECTION_UNDEFINED',
            'evidence_status':'project_approximation','individual_cone_measurement':False}

    def coverage(self,positions):
        xy,transfer=self._input_xy(positions);points=[]
        for i,point in enumerate(xy):
            r=float(np.linalg.norm(point));weights=self._weights(point)
            row={'index':i,'inner_segment_radius_mm':r,'meridian_weights':dict(weights),'rf_transfer':transfer}
            if r==0:row.update(status='DIRECTION_UNDEFINED',displacement_mm=None,radial_pedicle_limit_mm=self.limit,
                reason='Exact origin has no 2D direction; no implicit angle')
            elif r>self.maximum+1e-12:row.update(status='unsupported',displacement_mm=None,reason='bridge application exceeds6mm')
            else:
                segments={m:('central_linear_P_bridge' if r<c.domain_mm[0] else 'source_digitized' if r<=c.domain_mm[1]
                    else 'tail_linear_P_bridge' if r<self.bridge['zero_displacement_r_mm'][m] else 'outer_identity_project')
                    for m,_ in weights for c in [self.curves[m]]}
                exact=len(weights)==1 and all(s=='source_digitized' for s in segments.values()) and any(
                    np.isclose(self.curves[weights[0][0]].inner_segment_radius_mm,r,rtol=0,atol=1e-12))
                p=float(self._radial(r,weights));row.update(status='source_supported' if exact and transfer is None else 'project_approximation',
                    pedicle_radius_mm=p,displacement_mm=p-r,segments=segments,
                    evidence_status='author_derived_then_digitized' if all(s=='source_digitized' for s in segments.values()) else 'project_approximation',
                    reason='unchanged source numerical rule' if all(s=='source_digitized' for s in segments.values()) else 'approved linear-P bridge / outer identity')
            points.append(row)
        return {'provenance':self.provenance,'points':points,'supported':all(p['status'] in ('source_supported','project_approximation') for p in points),
            'input_cell_count':len(xy),'query_frame':positions.coordinate_frame,'query_coordinate':positions.meaning}

    def apply(self,positions,*,origin_direction_policy=None,origin_direction_radians=None):
        xy,_=self._input_xy(positions);report=self.coverage(positions)
        if any(p['status']=='unsupported' for p in report['points']):raise ContractError('Schein bridge outside approved application domain')
        mapped=np.empty_like(xy)
        for i,(point,row) in enumerate(zip(xy,report['points'])):
            r=row['inner_segment_radius_mm']
            if r==0:
                if origin_direction_policy!='fixed_angle_project_origin_v1' or origin_direction_radians is None or not np.isfinite(origin_direction_radians):
                    raise ContractError('DIRECTION_UNDEFINED: exact origin requires explicit named 2D angle policy')
                mapped[i]=self.limit*np.array([np.cos(origin_direction_radians),np.sin(origin_direction_radians)])
            else:mapped[i]=point*(row['pedicle_radius_mm']/r)
        return Quantity(mapped,'mm','pedicle_position','Schein1988_M1_bounded_bridge_project_approximation','OPL')

    def inverse(self,positions):
        xy=positions.require('mm','pedicle_position',frame='OPL');mapped=np.empty_like(xy)
        if xy.ndim!=2 or xy.shape[1]!=2:raise ContractError('Schein inverse requires shape(n,2)')
        for i,point in enumerate(xy):
            p=float(np.linalg.norm(point));r=float(self._inverse_radial(p,self._weights(point)))
            mapped[i]=point*(r/p)
        return Quantity(mapped,'mm','cone_inner_segment_position','Schein1988_M1_bridge_project_inverse','anatomical_retina')

    def inverse_support_bounds(self,bounds_mm):
        """Conservative interval enclosure of an OPL rectangle's supported RF.

        For any angle the inverse lies between the cardinal inverses. Each
        inverse is affine between P nodes; r/P extrema occur at the endpoints.
        The P<.15 model hole is excluded explicitly, never mapped to fake cones.
        """
        x0,x1,y0,y1=map(float,bounds_mm)
        if not np.all(np.isfinite((x0,x1,y0,y1))) or x0>=x1 or y0>=y1:raise ContractError('Increasing finite OPL support box required')
        near=np.array([0 if x0<=0<=x1 else min(abs(x0),abs(x1)),0 if y0<=0<=y1 else min(abs(y0),abs(y1))])
        low=max(self.limit,float(np.linalg.norm(near)));high=max(np.hypot(x,y) for x in (x0,x1) for y in (y0,y1))
        if high<self.limit:return {'bounds_RF_mm':None,'model_pedicle_free':True,'complete':True}
        if high>self.maximum+1e-12:raise ContractError('Finite OPL support exceeds approved inverse application')
        ratios=[]
        for m,(r,p) in self.nodes.items():
            samples=np.r_[low,p[(p>low)&(p<high)],high]
            ratios.extend(np.interp(samples,p,r)/samples)
        lo,hi=float(min(ratios)),float(max(ratios))
        xx=[a*f for a in (x0,x1) for f in (lo,hi)];yy=[a*f for a in (y0,y1) for f in (lo,hi)]
        return {'bounds_RF_mm':[min(xx),max(xx),min(yy),max(yy)],'complete':True,
            'method':'cardinal_inverse_piecewise_affine_ratio_interval_enclosure_v1',
            'radial_ratio_interval':[lo,hi],'OPL_bounds_mm':[x0,x1,y0,y1],
            'pedicle_free_region_mm':[0,self.limit],'boundary_mode':'finite_open; recurrent closure not certified'}
