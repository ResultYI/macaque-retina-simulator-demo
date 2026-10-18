"""Finite HC/inner connection previews in a shared, source-bounded atlas.

Unknown cone subtypes remain unknown. A geometric candidate is not an enabled
synapse, and an incomplete envelope cannot certify zero physiological input.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
import json

from .atlas import AtlasROI,CanonicalCellTable
from .contracts import Quantity,ContractError,canonical_hash
from .geometry import keyed_rng,PackerDensityField
from .runtime_profiles import frozen_packer_table,RuntimeProfile
from .source_atlas import SourceDensityAtlas
from .hc_arbor import validate_footprints,_candidate_contacts,_support_extent
from .ipl_geometry import read_ipl_geometry,RESEARCH


@dataclass(frozen=True)
class HCPatch:
    table: CanonicalCellTable
    types: tuple
    anchor: Quantity
    bounds_mm: tuple
    metadata: dict


class ConnectionAtlas:
    def __init__(self,root,*,hc_mosaic_mode='fixed_joint_master_v2',atlas_profile_id='retina_atlas_v1',
                 s_partition_profile='nemestrina_strict_pending_v1'):
        from .atlas_hc import AtlasHCProvider
        from .atlas_hc_master import FixedJointHCMaster
        if hc_mosaic_mode not in ('fixed_joint_master_v2','historical_marked_Poisson_preview_v1'):
            raise ContractError('Explicit HC joint master or historical preview mode required')
        self.atlas=SourceDensityAtlas(root,profile_id=atlas_profile_id)
        self.s_partition_profile=s_partition_profile
        if s_partition_profile=='deMonasterio1985_mulatta_effective_S_v1':
            from .atlas_s_partition import EffectiveMulattaSPartition
            self.s_partition=EffectiveMulattaSPartition(self.atlas)
        elif s_partition_profile=='nemestrina_strict_pending_v1':self.s_partition=None
        else:raise ContractError('Unknown explicit S partition profile')
        self.hc=AtlasHCProvider(root)
        self.inner_config=read_ipl_geometry(root,RESEARCH)
        self.wool_density=PackerDensityField(frozen_packer_table(root),
            central_policy='central_constant_cap',angular_policy='log_NT_cosine')
        self._hc_tiles={}
        self.hc_mosaic_mode=hc_mosaic_mode
        self.hc_master=FixedJointHCMaster(root,self.atlas.config,self.hc.density) if hc_mosaic_mode=='fixed_joint_master_v2' else None

    def hc_patch(self,bounds_mm,*,max_cells=60000):
        if self.hc_master is None:
            return self._historical_poisson_hc_patch(bounds_mm,max_cells=max_cells)
        table,types,metadata=self.hc_master.crop(bounds_mm,max_cells=max_cells)
        anchor=Quantity(table.coordinate('soma').values,'mm','arbor_anchor_position',
            'HC_soma_to_OPL_colocation_project_v1','OPL')
        metadata.update(density=self.hc.density.metadata,
            OPL_transfer='HC_soma_to_OPL_colocation_project_v1',
            original_verified_table_rows=self.hc.verified_density_rows)
        return HCPatch(table,types,anchor,tuple(bounds_mm),metadata)

    def _historical_poisson_hc_patch(self,bounds_mm,*,max_cells=60000):
        """Fixed total+mark tiles in physical soma space, explicit OPL projection."""
        profile=self.hc.density
        size=self.atlas.config['tile_size_mm'];seed=self.atlas.config['seed']
        namespace=canonical_hash({'profile':profile.profile_hash,'source':profile.source_asset_hash,
            'tile_size_mm':size,'seed':seed,'rule':'fixed_marked_Poisson_HC_preview_v1',
            'reference_id':self.atlas.config['reference_id'],'numpy_version':np.__version__})
        x0,x1,y0,y1=map(float,bounds_mm)
        if not (np.all(np.isfinite(bounds_mm)) and x0<x1 and y0<y1):
            raise ContractError('HC query requires finite physical 2D bounds')
        ix=range(int(np.floor(x0/size)),int(np.ceil(x1/size)))
        iy=range(int(np.floor(y0/size)),int(np.ceil(y1/size)))
        if len(ix)*len(iy)>max_cells:raise ContractError('HC tile budget exceeded; no density scaling')
        keys=[];positions=[];types=[]
        for tx in ix:
            for ty in iy:
                tile_key=(namespace,tx,ty)
                if tile_key not in self._hc_tiles:
                    center=np.array([[(tx+.5)*size,(ty+.5)*size]])
                    profile.total(np.array([[tx,ty],[tx+1,ty],[tx,ty+1],[tx+1,ty+1]])*size)
                    rng=keyed_rng(seed,namespace+'|'+str(tx)+'|'+str(ty))
                    count=int(rng.poisson(float(profile.total(center)[0])*size*size))
                    xy=(np.array([tx,ty])+rng.random((count,2)))*size
                    marks=tuple(np.where(rng.random(count)<profile.fraction(xy),'H1','H2'))
                    self._hc_tiles[tile_key]=(tuple(f'tile:{tx}:{ty}:HC:{i}' for i in range(count)),xy,marks)
                kk,xy,tt=self._hc_tiles[tile_key]
                keep=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
                keys.extend(k for k,v in zip(kk,keep) if v)
                positions.extend(xy[keep]);types.extend(t for t,v in zip(tt,keep) if v)
                if len(keys)>max_cells:raise ContractError('HC cell budget exceeded; no density scaling')
        points=np.asarray(positions,dtype=float).reshape((-1,2))
        soma=Quantity(points,'mm','anatomical_position','project_marked_HC_mosaic','anatomical_retina')
        table=CanonicalCellTable(atlas_id=self.atlas.config['atlas_id'],atlas_version=self.atlas.config['atlas_version'],
            source_version=namespace,population='HC',source_keys=tuple(keys),coordinates={'soma':soma},
            reference_id=self.atlas.config['reference_id'])
        anchor=Quantity(points,'mm','arbor_anchor_position','HC_soma_to_OPL_colocation_project_v1','OPL')
        return HCPatch(table,tuple(types),anchor,tuple(bounds_mm),{'density':profile.metadata,
            'canonical_rule':'fixed_marked_Poisson_HC_preview_v1','scale':1.,
            'OPL_transfer':'HC_soma_to_OPL_colocation_project_v1','biological_spatial_statistics':'NOT_VALIDATED',
            'role':'historical independent Poisson preview; not the approved joint HC distribution',
            'original_verified_table_rows':self.hc.verified_density_rows,'namespace':namespace})

    def _types(self,cones,*,max_cells=60000):
        if self.s_partition is not None:return self.s_partition.assign(cones,max_cells=max_cells)
        radius=np.linalg.norm(cones.coordinate('rf').values,axis=1)
        types=[]
        for gid,r in zip(cones.global_cell_ids,radius):
            # Explicit main-profile radial convention, not a universal circular
            # absence measurement. Outside it no nonzero S curve is invented.
            types.append(('L' if keyed_rng(self.atlas.config['seed'],gid+'|LM_equal_project_v1').random()<.5 else 'M')
                         if r<.05 else 'UNKNOWN')
        return tuple(types),{'source':'MartinGrunert1999_abstract_constraints',
            'S_absence_quote_scope_mm':[0,.05],'mask_application':'source_informed_radial_convention_project_v1',
            'shape_and_specimen_boundary':'unverified; counterprofile retained',
            'LM_split':'LM_equal_project_v1','outside_mask':'UNKNOWN; no deMonasterio substitution',
            'Bumsted_counterprofile':'BumstedHendrickson1999_abstract_constraints'}

    def _temporal_equivalent(self,rf):
        values=[]
        for point in rf:
            target=float(self.wool_density(point[None,:])[0])
            f=lambda r:float(self.wool_density.radial(np.array([r]),np.array([np.pi]))[0])-target
            try:values.append(brentq(f,self.wool_density.minimum,10.))
            except ValueError:values.append(None)
        return values

    def _soma_views(self,inner):
        """Use only existing central RF->midget soma nodes, independently of OPL."""
        relative='sources/spatial/mcgregor2018_production_map_nodes.json'
        runtime=RuntimeProfile(self.atlas.root,'effective_dynamics_v1')
        asset=next(row for row in runtime.data['geometry_assets'].values() if row['path']==relative)
        nodes=json.loads((self.atlas.root/relative).read_bytes())['production_nodes']
        curves={m:PchipInterpolator(n['r_mm'],n['soma_radius_mm'],extrapolate=False) for m,n in nodes.items()}
        names=('nasal','superior','temporal','inferior');views={};missing={}
        for name,table in inner.tables.items():
            if not name.startswith('midget_RGC'):continue
            rf=table.coordinate('rf').values;r=np.linalg.norm(rf,axis=1)
            keep=(r>0)&(r<=.04)
            xy=rf[keep];theta=np.mod(np.arctan2(xy[:,1],xy[:,0]),2*np.pi)/(np.pi/2)
            radius=np.zeros(len(xy))
            for i,(rr,a) in enumerate(zip(r[keep],theta)):
                k=int(np.floor(a))%4;f=a-np.floor(a)
                radius[i]=(1-f)*float(curves[names[k]](rr))+f*float(curves[names[(k+1)%4]](rr))
            positions=xy*(radius/r[keep])[:,None]
            coords={role:Quantity(q.values[keep],q.unit,q.meaning,q.evidence_tag,q.coordinate_frame)
                    for role,q in table.coordinates.items()}
            coords['soma']=Quantity(positions,'mm','anatomical_position',
                'McGregor2018_central_RF_to_soma_source_derived_project_interpolation','anatomical_retina')
            view=CanonicalCellTable(table.atlas_id,table.atlas_version,table.source_version,table.population,
                tuple(k for k,v in zip(table.source_keys,keep) if v),coords,table.reference_id)
            views[name]=view.record();missing[name]=int((~keep).sum())
        return views,{'source_asset':relative,'sha256':asset['sha256'],'source_RF_domain_mm':[0,.04],
            'application':'midget soma only, independent central-node radial/angle interpolation',
            'Schein_OPL_used_as_soma':False,'old_15_percent_split_used':False,
            'outside_domain':'ABSENT','exact_origin':'direction undefined; no invented convention',
            'missing_counts':missing,'physiological_validation':False}

    def _dependency_patch(self,roi,*,max_cells,phase,enable_central_candidate):
        """One upstream expansion for a fixed requested HC cohort, not recursion."""
        from .atlas_hc_master import HCMasterCoverageError
        patch=self.atlas.query(roi,max_cells=max_cells,phase=phase)
        audit={'requested_RF_query_mm':list(roi.query_bounds_mm),'expanded_RF_query_mm':list(roi.query_bounds_mm),
            'status':'strict_source_sampled_only','active_HC_cohort':'from original declared query; held fixed during upstream expansion',
            'feedback_of_support_only_cones':'finite induced boundary; exterior/recurrent closure not certified'}
        fixed_bounds=None
        if not hasattr(self.atlas.henle,'inverse_support_bounds') or not len(patch.mapped_table.source_keys):
            return patch,roi,fixed_bounds,audit
        xy=patch.mapped_table.coordinate('opl_pedicle').values;pad=self.hc.maximum_envelope_mm
        lo=xy.min(0)-pad;hi=xy.max(0)+pad;fixed_bounds=(float(lo[0]),float(hi[0]),float(lo[1]),float(hi[1]))
        try:hc=self.hc_patch(fixed_bounds,max_cells=max_cells)
        except HCMasterCoverageError:
            audit['status']='HC_FIXED_MASTER_UNSUPPORTED';return patch,roi,fixed_bounds,audit
        report=self.hc.footprints(hc.table.global_cell_ids,hc.types,hc.anchor,
            mode='connection_preview',enable_central_candidate=enable_central_candidate)
        configured=validate_footprints(report['footprints'],require_configured=False)
        bounds=list(roi.query_bounds_mm);rows=[]
        for j,(gid,subtype) in enumerate(zip(hc.table.global_cell_ids,hc.types)):
            fp=configured['by_hc_id'].get(gid,configured[subtype])
            if fp is None:rows.append({'hc_id':gid,'status':'ARBOUR_UNSUPPORTED'});continue
            _,box=_support_extent(hc.anchor.values[j],fp)
            envelope=self.atlas.henle.inverse_support_bounds(box);rows.append(dict(envelope,hc_id=gid))
            b=envelope['bounds_RF_mm']
            if b is not None:
                bounds=[min(bounds[0],b[0]),max(bounds[1],b[1]),min(bounds[2],b[2]),max(bounds[3],b[3])]
        expanded=AtlasROI(roi.owner_bounds_mm,tuple(bounds),roi.reference_id)
        patch=self.atlas.query(expanded,max_cells=max_cells,phase=phase)
        audit.update(status='finite_upstream_interval_enclosure',expanded_RF_query_mm=bounds,
            fixed_HC_query_OPL_mm=list(fixed_bounds),HC_envelope_rows=rows,
            geometry_regenerated=False,query_changes_cell_IDs=False,
            model_pedicle_free_OPL_mm=[0,self.atlas.henle.limit],
            source_or_query_missing_is_orphan=False)
        return patch,expanded,fixed_bounds,audit

    def preview(self,roi,*,enable_central_candidate=True,max_cells=60000,phase='phase1',hc_reference_bounds_mm=None):
        from .atlas_inner import build_effective_inner
        from .atlas_rod_inner import build_effective_rod_inner
        patch,roi,fixed_hc_bounds,dependency_audit=self._dependency_patch(roi,max_cells=max_cells,
            phase=phase,enable_central_candidate=enable_central_candidate)
        cones=patch.mapped_table
        if not len(cones.source_keys):
            cones=CanonicalCellTable(cones.atlas_id,cones.atlas_version,cones.source_version,cones.population,
                cones.source_keys,dict(cones.coordinates,opl_pedicle=Quantity(np.empty((0,2)),'mm','pedicle_position',
                'empty_supported_subset_no_coordinate_claim','OPL')),cones.reference_id)
        all_owner={g for g,v in zip(patch.selection.global_cell_ids,patch.selection.owner) if v}
        owner=np.array([g in all_owner for g in cones.global_cell_ids],dtype=bool)
        missing_owner=len(all_owner)-int(owner.sum())
        types,type_provenance=self._types(cones,max_cells=max_cells)
        blockers=[]
        if missing_owner:
            r=np.linalg.norm(patch.selection.coordinate('rf').values[patch.selection.owner],axis=1)
            blockers.append({'frame':'RF -> OPL','type':'cone','radial_range_mm':[float(r.min()),float(r.max())],
                'missing_count':missing_owner,'quantity':'IS-to-pedicle displacement',
                'reason':'Schein cardinal domains missing; no central bridge or tail fill','needs_new_assumption':True})
        input_radius=np.linalg.norm(patch.selection.coordinate('rf').values,axis=1)
        unknown=input_radius>=.05 if self.s_partition is None else np.zeros(len(input_radius),dtype=bool)
        if np.any(unknown):
            r=input_radius[unknown]
            blockers.append({'frame':'RF/OPL','type':'L/M/S partition','radial_range_mm':[float(r.min()),float(r.max())],
                'missing_count':int(unknown.sum()),'quantity':'main-source nonzero S density/partition',
                'reason':'MartinGrunert full curve missing; not replaced by old cross-species fit','needs_new_assumption':True})
        equivalent=self._temporal_equivalent(cones.coordinate('rf').values)
        # Globalize the existing engineering one-in-four effective sampling;
        # this is a named computational scaffold, never a BC subtype density.
        diffuse_seeds=tuple(g for g in cones.global_cell_ids if
            int(hashlib.sha256((g+'|effective_LM_stride4_v1').encode()).hexdigest()[:16],16)%4==0)
        inner_kwargs={'cone_subtypes':types,'configuration':self.inner_config,
            'diffuse_seed_cone_ids':diffuse_seeds,'include_widefield_ac':True,'max_effective_cells':max_cells}
        inner_kwargs['midget_rf_rule']=self.atlas.config.get('midget_rf_rule','parent_seed_legacy')
        inner_kwargs.update(temporal_equivalent_mm=np.asarray([np.nan if v is None else v for v in equivalent]),
            temporal_equivalence_source_id='midget_Wool_iso_density_adapter_v1_frozen_Packer_published_code')
        inner=build_effective_inner(cones,**inner_kwargs)
        inner_record=inner.record()
        rod_inner=build_effective_rod_inner(self.atlas,roi,max_cells=max_cells)
        soma_views,soma_provenance=self._soma_views(inner)
        hc=None;candidate_edges=[];hc_rows=[];footprint_report=None;hc_query_unsupported=False
        def query_hc(bounds):
            from .atlas_hc_master import HCMasterCoverageError
            nonlocal hc_query_unsupported
            try:return self.hc_patch(bounds,max_cells=max_cells)
            except HCMasterCoverageError as error:
                hc_query_unsupported=True
                blockers.append({'frame':'anatomical HC soma / OPL anchor','type':'HC fixed master coverage',
                    'quantity':'materialized joint master including finite pair context',
                    'query_bounds_mm':list(bounds),'canonical_bounds_mm':list(self.hc_master.bounds),
                    'reason':str(error),'status':'HC_FIXED_MASTER_UNSUPPORTED',
                    'needs_new_assumption':False,'remedy':'explicit new fixed joint master asset/version; no Poisson fallback',
                    'not_biological_absence':True})
                return None
        if len(cones.source_keys):
            p=cones.coordinate('opl_pedicle').values
            pad=self.hc.maximum_envelope_mm
            lo=p.min(0)-pad;hi=p.max(0)+pad
            hc=query_hc(fixed_hc_bounds or (float(lo[0]),float(hi[0]),float(lo[1]),float(hi[1])))
            if hc is not None:
                footprint_report=self.hc.footprints(hc.table.global_cell_ids,hc.types,hc.anchor,
                    mode='connection_preview',enable_central_candidate=enable_central_candidate)
                configured=validate_footprints(footprint_report['footprints'],require_configured=False)
                tree=cKDTree(p)
                for j,(gid,subtype) in enumerate(zip(hc.table.global_cell_ids,hc.types)):
                    fp=configured['by_hc_id'].get(gid,configured[subtype])
                    if fp is None:
                        hc_rows.append({'hc_id':gid,'type':subtype,'status':'ARBOUR_UNSUPPORTED','candidate_count':0});continue
                    radial,box=_support_extent(hc.anchor.values[j],fp)
                    indices,distances,nodes,extensions=_candidate_contacts(tree,p,hc.anchor.values[j],fp,
                        cones.global_cell_ids,None,np.ones(len(p),dtype=bool))
                    # Test the finite physical envelope in the named inverse chart;
                    # this numerical sampling is an engineering diagnostic, not a
                    # theorem that all upstream/recurrent dependencies are closed.
                    x0,x1,y0,y1=box
                    ticks=np.linspace(0,1,17)
                    border=np.r_[np.c_[x0+(x1-x0)*ticks,np.full(17,y0)],
                        np.c_[x0+(x1-x0)*ticks,np.full(17,y1)],
                        np.c_[np.full(17,x0),y0+(y1-y0)*ticks],np.c_[np.full(17,x1),y0+(y1-y0)*ticks]]
                    try:
                        b=roi.query_bounds_mm
                        if hasattr(self.atlas.henle,'inverse_support_bounds'):
                            enclosure=self.atlas.henle.inverse_support_bounds(box);e=enclosure['bounds_RF_mm']
                            complete=e is None or bool(b[0]-1e-12<=e[0]<=e[1]<=b[1]+1e-12 and b[2]-1e-12<=e[2]<=e[3]<=b[3]+1e-12)
                            query_reason='approved_pedicle_free_model_region' if e is None else 'inverse_interval_enclosure_contained' if complete else 'inverse_interval_enclosure_exceeds_RF_query'
                        else:
                            inverse=self.atlas.henle.inverse(Quantity(border,'mm','pedicle_position','project_preview','OPL')).values
                            complete=bool(np.all((inverse[:,0]>=b[0])&(inverse[:,0]<=b[1])&
                                                 (inverse[:,1]>=b[2])&(inverse[:,1]<=b[3])))
                            query_reason='inverse_envelope_sample_contained' if complete else 'inverse_envelope_exceeds_RF_query'
                    except ContractError as error:
                        complete=False;query_reason=str(error)
                    known=0;conditional=0
                    for i,d,node,extension in zip(indices,distances,nodes,extensions):
                        i=int(i);eligible=types[i]!='S' if subtype=='H1' and types[i]!='UNKNOWN' else True if subtype=='H2' else None
                        if eligible is False:continue
                        status='conditional_subtype_unknown' if eligible is None else 'finite_geometric_candidate'
                        known+=eligible is True;conditional+=eligible is None
                        candidate_edges.append({'edge_id':'atlas_HC:'+canonical_hash([gid,cones.global_cell_ids[i]]),
                            'hc_id':gid,'hc_type':subtype,'cone_id':cones.global_cell_ids[i],
                            'cone_type':types[i],'cone_owner':bool(owner[i]),'distance_mm':float(d),
                            'frame':'OPL','eligibility':status,'raw_geometry_weight':1.,
                            'feedforward_weight':None,'feedback_weight':None,
                            'morphology_node_index':None if node<0 else int(node),'extension_mm':float(extension),
                            'source_refs':{'arbor':'#/HC_footprints/footprints/by_hc_id/'+gid,
                                'cone_coordinates':'#/atlas/Henle','type_partition':'#/cone_type_provenance'},
                            'dynamic_connection_enabled':False})
                    hc_rows.append({'hc_id':gid,'type':subtype,'anchor_OPL_mm':hc.anchor.values[j].tolist(),
                        'finite_envelope_radial_mm':list(radial),'arbor_radius_mm':fp.get('radius_mm'),
                        'raw_geometry_count':len(indices),'candidate_count':int(known+conditional),'known_eligible_count':int(known),'conditional_count':int(conditional),
                        'query_complete_sampled':complete,'query_reason':query_reason,
                        'status':'CANDIDATES' if known+conditional else 'ZERO_CANDIDATES_NOT_CLOSURE_PROOF',
                        'direct_input_status':'INPUT_ENVELOPE_INCOMPLETE_NOT_ORPHAN' if not complete else
                            'SUBTYPE_UNRESOLVED' if conditional else 'COMPLETE_EMPTY_DIRECT_INPUT' if not known else 'COMPLETE_DIRECT_INPUT',
                        'zero_cone_dynamic_status':'NATIVE_ADAPTATION_UNDEFINED_IF_GAP_CONNECTED' if not known+conditional else 'native source input available after type/query completion'})
                blockers.extend(footprint_report['blocked_records'])
                required_ids={e['hc_id'] for e in candidate_edges if e['cone_owner']}
                incomplete=[row for row in hc_rows if row['hc_id'] in required_ids and not row.get('query_complete_sampled',False)]
                if incomplete:
                    rr=[np.linalg.norm(row['anchor_OPL_mm']) for row in incomplete if 'anchor_OPL_mm' in row]
                    blockers.append({'frame':'OPL -> RF query support','type':'owner-required H1/H2',
                        'radial_range_mm':[float(min(rr)),float(max(rr))] if rr else None,
                        'missing_count':len(incomplete),'quantity':'complete finite cone input envelope',
                        'reason':'finite inverse-envelope sample exceeds requested RF query or source mapping domain',
                        'needs_new_assumption':False,'remedy':'larger supported input ROI; never normalize truncated contacts as complete'})
        elif hc_reference_bounds_mm is not None:
            # This is an independent physical-domain query, never an RF->HC
            # coordinate fallback. It permits inspecting valid HC candidates
            # even where the photoreceptor-to-OPL source route is missing.
            hc=query_hc(hc_reference_bounds_mm)
            if hc is not None:
                footprint_report=self.hc.footprints(hc.table.global_cell_ids,hc.types,hc.anchor,
                    mode='connection_preview',enable_central_candidate=enable_central_candidate)
                blockers.extend(footprint_report['blocked_records'])
        # Only mapped owners can have a numerical geometric degree. Missing
        # source coordinates are not zero partner rows or biological absence.
        counts={g:0 for g,v in zip(cones.global_cell_ids,owner) if v} if hc is not None else {}
        for edge in candidate_edges:
            if edge['cone_id'] in counts:counts[edge['cone_id']]+=1
        zero=[g for g,n in counts.items() if n==0]
        if zero:
            blockers.append({'frame':'OPL','type':'cone -> H1/H2 feedback','missing_count':len(zero),
                'quantity':'supported finite HC partners','reason':'owner zero partners in selected finite '+self.hc_mosaic_mode+' instance; not biological/source failure; no nearest fill',
                'status':'JOINT_PROFILE_UNSUPPORTED' if len(zero)==len(counts) else 'OWNER_ZERO_FEEDBACK_REQUIRES_REVIEW',
                'needs_new_assumption':True})
        by_reason={}
        for item in inner.blockers:
            key=(item.get('route',item.get('population','inner sampling')),item['reason'])
            group=by_reason.setdefault(key,{'frame':'OPL' if key[0].startswith('cones->') else 'IPL',
                'type':key[0],'reason':key[1],'missing_count':0,'example_ids':[],
                'RF_query_bounds_mm':list(roi.query_bounds_mm),'needs_new_assumption':False})
            group['missing_count']+=1
            if item.get('post_id') and len(group['example_ids'])<3:group['example_ids'].append(item['post_id'])
        blockers.extend(by_reason.values())
        rod_reasons={}
        for item in rod_inner['blockers']:
            route=item.get('route','rods/rod_BC source density')
            key=(route,item['reason'])
            group=rod_reasons.setdefault(key,{'frame':'OPL' if route=='rods->rod_BC' else
                'IPL' if route=='rod_BC->AII_AC' else 'RF -> physical rod chart',
                'type':route,'reason':item['reason'],'missing_count':0,'owner_missing_count':0,
                'quantity':'finite effective input envelope' if 'route' in item else 'supported rod density curve',
                'RF_query_bounds_mm':list(roi.query_bounds_mm),'needs_new_assumption':'route' not in item,
                'not_biological_absence':True})
            group['missing_count']+=1
            group['owner_missing_count']+=bool(item.get('is_owner',False))
        blockers.extend(rod_reasons.values())
        if any(soma_provenance['missing_counts'].values()):
            blockers.append({'frame':'RF -> anatomical soma','type':'midget RGC',
                'source_RF_domain_mm':[0,.04],'missing_counts':soma_provenance['missing_counts'],
                'quantity':'independent RF-to-midget-soma displacement','reason':'outside directly source-derived central nodes',
                'role':'coordinate/visualization limit; current effective IPL dynamics can operate without a soma coordinate',
                'needs_new_assumption':True})
        return {'atlas':patch.manifest,'cones':cones.record(),'cone_subtypes':list(types),
            'cone_type_provenance':type_provenance,'HC':None if hc is None else
                {'table':hc.table.record(),'types':list(hc.types),'OPL_anchor':{'values':hc.anchor.values.tolist(),
                 'unit':'mm','meaning':hc.anchor.meaning,'frame':'OPL','evidence_tag':hc.anchor.evidence_tag},'metadata':hc.metadata},
            'HC_footprints':footprint_report,'HC_rows':hc_rows,'HC_candidate_edges':candidate_edges,
            'inner_tables':{name:t.record() for name,t in inner.tables.items()},
            'inner_parents':inner_record['parent_ids'],'inner_contacts':inner_record['contacts'],'inner_metadata':inner.metadata,
            'inner_soma_views':soma_views,'inner_soma_provenance':soma_provenance,
            'rod_inner':rod_inner,
            'dependency_query':dependency_audit,
            'inner_sampling':{'diffuse_rule':'effective_LM_stride4_global_ID_hash_v1',
                 'is_biological_density':False,'unknown_subtypes_are_structural_only':True,
                 'widefield_AC':'existing global40um/3 typed computational grid; no RF/soma inverse'},
            'HC_reference_query':None if hc_reference_bounds_mm is None else
                {'bounds_OPL_mm':list(hc_reference_bounds_mm),'role':'independent physical query, not inferred from RF'},
            'closure_blockers':blockers,'gap':{'enabled':False,'effective_rate_ms_inverse':0,
                 'positive_rate_or_physical_parameters':'explicit named uncalibrated caller values required',
                 'geometry_only_HC_dynamic_membership':'include all HC in enabled coupling components; empty-source adaptation currently undefined'},
            'summary':{'owner_cones':len(all_owner),'mapped_owner_cones':int(owner.sum()),
                 'owner_coordinate_unsupported':missing_owner,
                 'HC_mosaic_mode':self.hc_mosaic_mode,
                 'S_partition_profile':self.s_partition_profile,
                 'cone_subtype_counts':{t:types.count(t) for t in ('L','M','S','UNKNOWN')},
                 'HC_query_status':'HC_FIXED_MASTER_UNSUPPORTED' if hc_query_unsupported else 'supported' if hc is not None else 'not_requested',
                 'mapped_query_cones':len(cones.source_keys),'HC_cells':0 if hc is None else len(hc.table.source_keys),
                 'HC_candidate_edges':len(candidate_edges),'HC_conditional_subtype_edges':sum(e['eligibility']=='conditional_subtype_unknown' for e in candidate_edges),
                 'zero_owner_candidate_rows':len(zero),'inner_population_counts':{k:len(t.source_keys) for k,t in inner.tables.items()},
                 'inner_weighted_edges':sum(len(c['edges']) for c in inner_record['contacts'].values()),
                 'rod_inner_population_counts':{k:len(t['source_keys']) for k,t in rod_inner['tables'].items()},
                 'rod_inner_weighted_edges':sum(len(c['edges']) for c in rod_inner['contacts'].values()),
                 'rod_inner_source_status':rod_inner['metadata']['source_status'],
                 'network_run_ready':False,'source_or_joint_full_coverage_claim':False}}
