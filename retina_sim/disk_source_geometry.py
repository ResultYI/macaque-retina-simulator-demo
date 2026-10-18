"""Array evaluation of the unchanged approved Schein and Wool adapters."""
from pathlib import Path
import json
import numpy as np
from .contracts import ContractError,Quantity
from .disk_graph_storage import file_sha,save_json
from .disk_tables import write_table

def schein_array(henle,points):
    """Same adjacent-cardinal linear weights, axis snap and linear-P nodes."""
    if not hasattr(henle,'nodes'):raise ContractError('Approved bounded Schein node profile required')
    xy=np.asarray(points,dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.all(np.isfinite(xy)):
        raise ContractError('Finite Nx2 cone coordinates required')
    r=np.linalg.norm(xy,axis=1)
    if np.any(r==0):
        raise ContractError('DIRECTION_UNDEFINED: finite positive-radius2D cone coordinates required')
    if np.any(r>henle.maximum+1e-12):raise ContractError('Schein array outside approved application domain')
    quadrant=np.mod(np.arctan2(xy[:,1],xy[:,0]),2*np.pi)/(np.pi/2)
    nearest=np.rint(quadrant).astype(int);snap=np.abs(quadrant-nearest)<=1e-12
    lower=np.floor(quadrant).astype(int);fraction=quadrant-lower
    lower[snap]=nearest[snap]%4;fraction[snap]=0.
    names=('nasal','superior','temporal','inferior');mapped=np.zeros(len(xy))
    for side,weights in ((0,1-fraction),(1,fraction)):
        for k,name in enumerate(names):
            mask=((lower+side)%4==k)&(weights>0)
            mapped[mask]+=weights[mask]*np.interp(r[mask],*henle.nodes[name])
    return xy*(mapped/r)[:,None]

def wool_temporal_array(field,points):
    """Solve the existing temporal iso-density equation without an approximate LUT.

Outside the source bracket remains NaN. The central constant plateau resolves
to the same minimum-radius root used by the original brentq adapter.
"""
    xy=np.asarray(points,dtype=float);target=field(xy);low=np.full(len(xy),field.minimum);high=np.full(len(xy),10.)
    temporal=lambda r:field.radial(r,np.full(len(r),np.pi))
    a=temporal(low);b=temporal(high);valid=(target<=a)&(target>=b)
    at_low=valid&(target==a);at_high=valid&(target==b);interior=valid&~at_low&~at_high
    for _ in range(44):
        midpoint=(low+high)/2;value=temporal(midpoint)
        upper=value<target;high=np.where(interior&upper,midpoint,high);low=np.where(interior&~upper,midpoint,low)
    result=np.where(valid,(low+high)/2,np.nan);result[at_low]=field.minimum;result[at_high]=10.
    return result

def freeze_outer_tables(root,atlas_profile_id,support,output):
    """One global fixed cone/HC cohort; no ROI-specific sampling or normalization."""
    from .source_atlas import SourceDensityAtlas
    from .atlas_s_partition import EffectiveMulattaSPartition
    from .atlas_hc import AtlasHCProvider
    from .atlas_hc_master import FixedJointHCMaster
    root=Path(root).resolve();output=Path(output);atlas=SourceDensityAtlas(root,profile_id=atlas_profile_id)
    if atlas.cone_master is None:raise ContractError('Fixed source-stopped cone master required')
    master=atlas.cone_master;limits=support['limits'];radius=limits['cone_RF_query_radius_mm']
    rows=np.flatnonzero(np.sum(master.positions**2,axis=1)<=radius**2);rf=master.positions[rows]
    payload=root/master.manifest['payload']['path']
    with np.load(payload,allow_pickle=False) as data:apertures=np.asarray(data['aperture_diameters_um'])[rows]
    if np.any(np.linalg.norm(rf,axis=1)==0):raise ContractError('Exact origin cone has no approved individual direction')
    opl=schein_array(atlas.henle,rf);partition=EffectiveMulattaSPartition(atlas);labels=np.empty(len(rows),dtype='S1')
    metadata={'atlas_id':atlas.config['atlas_id'],'atlas_version':atlas.config['atlas_version'],
        'source_version':atlas.mosaic_namespace,'population':'cones','reference_id':atlas.config['reference_id']}
    for start in range(0,len(rows),4096):
        stop=min(len(rows),start+4096)
        table=atlas._table('cones',tuple(master.source_keys[i] for i in rows[start:stop]),rf[start:stop])
        assigned,_=partition.assign(table,max_cells=1000000);labels[start:stop]=assigned
    cone=write_table(root,output/'cones',metadata,len(rows),lambda a,b:tuple(master.source_keys[i] for i in rows[a:b]),
        {'rf':Quantity(rf,'mm','functional_position','source_informed_project_mosaic','functional_visual_field'),
         'opl_pedicle':Quantity(opl,'mm','pedicle_position','Schein1988_M1_bounded_bridge_project_approximation','OPL')},
        extras={'subtype':labels,'original_cone_master_row':rows,'aperture_diameter_um':apertures},source_bindings={
            'cone_master':dict(atlas.config['cone_master']),'atlas_profile':atlas_profile_id,'profile_hash':atlas.profile_hash,
            'mosaic_namespace':atlas.mosaic_namespace,'S_profile_sha256':partition.profile_sha256,
             'cone_geometry_route':master.manifest.get('geometry_route','custom_macaque_density_lattice'),
             'density_species':master.manifest['density_species'],
             'S_rule':'unchanged macaque absolute macro quotas and Packer selection scores; conditional project labels on actual candidates',
            'Schein':atlas.henle.provenance,'fixed_RF_query_radius_mm':radius,
            'implementation_sha256':file_sha(Path(__file__))})
    # The HC master is a rectangular physical enclosure. Only the fixed circular
    # cohort derived from inner OPL support participates in the global circuit.
    hc=AtlasHCProvider(root);fixed=FixedJointHCMaster(root,atlas.config,hc.density)
    cohort=limits['inner_OPL_support_radius_mm']+hc.maximum_envelope_mm
    table,types,record=fixed.crop((-cohort,cohort,-cohort,cohort),max_cells=1000000)
    xy=table.coordinate('soma').values;keep=np.flatnonzero(np.sum(xy**2,axis=1)<=cohort**2);xy=xy[keep]
    metadata={key:getattr(table,key) for key in ('atlas_id','atlas_version','source_version','population','reference_id')}
    horizontal=write_table(root,output/'HC',metadata,len(keep),lambda a,b:tuple(table.source_keys[i] for i in keep[a:b]),
        {'soma':Quantity(xy,'mm','anatomical_position',table.coordinate('soma').evidence_tag,'anatomical_retina')},
        extras={'subtype':np.asarray(types,dtype='S2')[keep]},source_bindings={
            'master_sha256':fixed.config['expected_asset_sha256'],'OPL_anchor_transfer':'HC_soma_to_OPL_colocation_project_v1',
            'fixed_cohort_OPL_radius_mm':cohort,'pair_context_in_master':True,
            'construction':'crop of one frozen joint master; no exterior HC feedback'})
    result={'cones':cone.binding,'HC':horizontal.binding,'cone_count':cone.count,'HC_count':horizontal.count,
        'reference_id':atlas.config['reference_id'],'status':'GLOBAL_OUTER_TYPED_TABLES_FROZEN_PENDING_CONNECTIONS',
        'population_geometry_regenerated_on_ROI':False,'physiological_validation':False}
    save_json(root/output/'outer_tables.json',result);return result
