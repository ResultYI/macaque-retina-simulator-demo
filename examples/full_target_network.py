"""All-target continuous center/periphery assembly with a named effective profile.

This is a configurable physiological mechanism implementation, not animal data.
The default explicitly resolves an effective mosaic over one continuous retinal
corridor. Density scale and counts are declared computing resolution, not anatomy.
Every numerical approximation is exported and replaceable; no plugin transport.
"""
from pathlib import Path
from dataclasses import replace
import json,hashlib
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from retina_sim.contracts import Quantity,RunMode,CellTable,StimulusSegment,ContractError
from retina_sim.spatial import Population,GazeTrajectory,gaze_driven_channel,InputChannel
from retina_sim.assembly import ModelSpec,ComponentSpec,ExternalInput,PortLink,GapJunction,RetinaBuilder
from retina_sim.graph import ConnectionTable
from retina_sim.core_sources import load_core_registry
from retina_sim.mechanism_contracts import EvidenceValue,MechanismProfile,register_mechanism_profile
from retina_sim.vh2005 import KEY,PROFILE
from retina_sim.chen2024 import PROFILE as CHEN_PROFILE,SPECIES as CHEN_SPECIES
from retina_sim.rod_pathway import rod_source_values
from retina_sim.parameter_contracts import CHEN_PARAMETER_SPECS
from retina_sim.geometry import PackerDensityField,RetinalDomain,MosaicBackend,integrate_density,assign_cone_types,SConeDensityField,McGregorDisplacement
from retina_sim.density import read_density_tabulation
from retina_sim.runtime_profiles import LightContext, RuntimeProfile, frozen_packer_table, execution_profile as read_execution_profile
from retina_sim.initialization import initial_event_rate,initial_release
from retina_sim.regional_support import navigation_regions,regional_binding_audit
from examples.full_network_fixture import population

APPROVAL='User delegation 2026-10-10: complete actual mechanisms using explicitly retractable project_approximation; never label as macaque measurement'
PROJECT_SOURCE='PRIMARY_PATHWAY_project_profile'

def q(v,u,m,frame=None):return Quantity(v,u,m,'synthetic_fixture',frame)

def register_profile(registry,name,n,entries,*,source_id=PROJECT_SOURCE,evidence='project_approximation',reason='Effective reduced circuit profile; not measured macaque coefficients'):
    if hasattr(registry,'runtime_profile'):entries=registry.runtime_profile.bind(name,entries)
    records=[]
    for key,(value,unit,meaning) in entries.items():
        records.append(EvidenceValue(key,q(value,unit,meaning),evidence,source_id,reason+'; parameter '+key,{'effective_profile':name,'cells':n,'domain':'Explicit synthetic stimulus or supplied calibrated input; no macaque validity claimed','provenance_tag':('AUTHOR_MODEL_PRIMATE' if evidence=='published_fit' else 'PROJECT_EFFECTIVE_TEMP')},'Uncalibrated/retractable' if evidence=='project_approximation' else 'Source fit; no independent validation'))
    p=MechanismProfile(name,'Macaque_effective_project_profile',{'implementation':'mechanism-complete approximation','not_animal_measurement':True},tuple(records),(source_id,),evidence,APPROVAL)
    register_mechanism_profile(registry,p);return p

def contact_positions(pop,frame):
    candidates=(pop.pedicle_positions,pop.dendritic_positions) if frame=='OPL' else (pop.terminal_positions,pop.dendritic_positions)
    for value in candidates:
        if value is not None and value.coordinate_frame==frame:return value.values
    if pop.anatomical_positions is None:raise ValueError('Missing explicit synaptic-frame point morphology')
    return pop.anatomical_positions.values

def local_matrix(pre,post,k=4,radius=None,frame='IPL'):
    xy=contact_positions(pre,frame);target=contact_positions(post,frame);k=min(k,len(xy))
    d,idx=cKDTree(xy).query(target,k=k);d=np.asarray(d).reshape(len(target),k);idx=np.asarray(idx).reshape(len(target),k)
    if radius is None:w=np.ones_like(d)
    else:
        logw=-(d/np.asarray(radius)[:,None])**2
        w=np.exp(logw-logw.max(axis=1,keepdims=True))
    w/=w.sum(axis=1,keepdims=True)
    return sparse.csr_matrix((w.ravel(),(np.repeat(np.arange(len(target)),k),idx.ravel())),shape=(len(target),len(xy)))

def connection(pre,port,post,target,W,populations,*,synapse='explicit_receptor',frame='IPL',profile=PROJECT_SOURCE):
    a=populations[pre];b=populations[post];W=sparse.coo_matrix(W);n=W.nnz
    t=ConnectionTable(tuple(f'{pre}.{port}>{post}.{target}:{i}' for i in range(n)),tuple(a.cells.cell_id[i] for i in W.col),tuple(b.cells.cell_id[i] for i in W.row),(synapse,)*n,tuple(W.data),('dimensionless',)*n,('nonnegative_contact_weight; receptor owns sign',)*n,(0.,)*n,(profile,)*n,('explicit local contact row gain',)*n,('synthetic_fixture',)*n)
    return PortLink(pre,port,post,target,t,frame)

def membrane_entries(n,kind,*,subtype=None):
    fill=lambda v:np.full(n,v)
    entries={'C':(fill(5e-12 if kind=='rod_membrane' else 20e-12),'F','membrane_capacitance'),'gL':(fill(1e-9),'S','leak_conductance'),'EL':(fill(-.07723875 if kind=='rod_membrane' else -.060),'V','leak_reversal'),'V0':(fill(-.040 if kind=='rod_membrane' else -.055),'V','initial_membrane_voltage'),'release_half':(fill(-.040 if kind=='rod_membrane' else -.050),'V','activation_half_voltage'),'release_slope':(fill(.002 if kind=='rod_membrane' else .005),'V','activation_voltage_slope'),'tau_release':(fill(.002 if kind=='rod_membrane' else .003),'s','release_time_constant')}
    if kind=='rod_membrane':entries['current_anchor']=(fill(0.),'A','absolute_outer_segment_current')
    else:
        entries.update({'K':(fill(.25),'dimensionless','receptor_half_drive'),'hill':(fill(2.),'dimensionless','receptor_hill_exponent'),'tau_e':(fill(.012 if kind=='rod_BC_ON' else .004),'s','excitatory_receptor_time_constant'),'gE':(fill(.5e-9),'S','maximum_excitatory_conductance'),'EE':(fill(0.),'V','excitatory_reversal'),'e0':(fill(.1),'dimensionless','initial_receptor_gate')})
        if kind=='cone_BC_OFF':
            # Distinct FMB-like and diffuse/DB1-like effective profiles. These are
            # not transplanted macaque measured kinetics and not generic OFF biology.
            tau=.005 if subtype=='midget_OFF' else .015
            gain=.6e-9 if subtype=='midget_OFF' else .25e-9
            entries.update({'tau_i':(fill(tau),'s','glycine_receptor_time_constant'),'gI':(fill(gain),'S','maximum_inhibitory_conductance'),'EI':(fill(-.080),'V','inhibitory_reversal'),'i0':(fill(0.),'dimensionless','initial_receptor_gate')})
        if kind=='AII':entries.update({'gNa':(fill(.4e-9),'S','maximum_sodium_conductance'),'ENa':(fill(.055),'V','sodium_reversal'),'na_half':(fill(-.045),'V','sodium_activation_half_voltage'),'na_slope':(fill(.005),'V','sodium_activation_slope'),'gK':(fill(.5e-9),'S','maximum_potassium_conductance'),'EK':(fill(-.085),'V','potassium_reversal'),'k_half':(fill(-.050),'V','potassium_activation_half_voltage'),'k_slope':(fill(.008),'V','potassium_activation_slope'),'tau_k':(fill(.020),'s','potassium_activation_time_constant')})
    if kind in ('rod_BC_ON','AII'):
        entries.update({'rmax':(fill(100.),'events/s','release_attempt_rate'),'Npool':(fill(20. if kind=='rod_BC_ON' else 5.),'dimensionless','release_pool_size'),'U':(fill(.2),'dimensionless','release_probability'),'tau_recovery':(fill(.1),'s','release_recovery_time_constant'),'r_half':(fill(-.045),'V','release_half_voltage'),'r_slope':(fill(.004),'V','release_voltage_slope')})
    if kind=='cone_BC_OFF':entries['EI']=(fill(-.076 if subtype=='midget_OFF' else -.073),'V','inhibitory_reversal')
    if kind in ('rod_BC_ON','AII'):
        for key in ('K','hill','tau_e','gE','e0','gNa','ENa','na_half','na_slope','gK','EK','k_half','k_slope','tau_k'):entries.pop(key,None)
    if kind=='cone_BC_OFF':
        for key in ('tau_i','gI','i0'):entries.pop(key,None)
    return entries

def full_target_network(root,*,effective_cones=128,effective_rods=384,seed=20261010,duration=.12,gaze=None,adaptation_seconds=0.,ablations=(),secondary_gap_scale=0.,bounds_mm=(.004,7.2,-.04,.04),cone_density_scale=None,rod_density_scale=None,light_context=None,runtime_profile=None,execution_profile='engineering_preview',cone_route=None,owner_bounds_mm=None,halo_mm=.04,max_spatial_cells=8192,hc_geometry_profile='source_patch_v1',hc_geometry_path='',hc_h1_radius_mm=None,hc_h2_radius_mm=None,hc_arbor_profile='explicit_footprints',ipl_geometry_profile='legacy_point_ipl_v1',requested_owner_outputs=(),parasol_rf_seeds=None,bc_output_gain_control_enabled=None,ac_inhibition_enabled=None):
    root=Path(root);registry=load_core_registry(root)
    light_context=light_context or LightContext()
    selected_profile=read_execution_profile(root,execution_profile)
    registry.runtime_profile=RuntimeProfile(root,runtime_profile or selected_profile['parameter_profile'])
    if selected_profile['execution_mode'] not in ('engineering_preview','source_constrained_approximation'):raise ValueError('Unknown execution profile')
    source_spatial=selected_profile['execution_mode']=='source_constrained_approximation'
    from retina_sim.ipl_geometry import read_ipl_geometry,TYPED_PROFILES,RESEARCH,typed_ipl_points,in_bounds
    ipl_configuration=read_ipl_geometry(root,ipl_geometry_profile);typed_ipl=ipl_geometry_profile in TYPED_PROFILES
    if parasol_rf_seeds is not None:
        from retina_sim.ipl_geometry import ParasolRFSeedInput
        if not typed_ipl or not isinstance(parasol_rf_seeds,ParasolRFSeedInput):
            raise ContractError('Explicit parasol RF seed input requires the typed IPL planner')
    research_contacts=ipl_geometry_profile==RESEARCH
    if typed_ipl and not source_spatial:raise ContractError('Typed P-chart IPL requires an unscaled source ROI; engineering legacy preview remains separate')
    if typed_ipl and not requested_owner_outputs:raise ContractError('Typed IPL requires an explicit nonempty requested owner-output set')
    ipl_audit={'configuration':ipl_configuration,'populations':{},'anchors':{},'contact_rules':[],
        'boundary_status':'existing finite induced graph; recurrent closure NOT_ESTABLISHED; no new boundary enabled'}
    def ipl_points(pop,role):
        return typed_ipl_points(pop,role) if typed_ipl else contact_positions(pop,'IPL')
    from retina_sim.hc_geometry import read_hc_geometry
    hc_configuration=read_hc_geometry(root,hc_geometry_profile,config_path=hc_geometry_path,
        h1_radius_mm=hc_h1_radius_mm,h2_radius_mm=hc_h2_radius_mm,arbor_profile_id=hc_arbor_profile)
    if (selected_profile['profile_id']==RESEARCH or research_contacts) and not (
        selected_profile['profile_id']==RESEARCH and research_contacts and hc_geometry_profile=='hc_joint_synthetic_v1'
        and hc_arbor_profile=='W2000_equal_area_bounded_edges_047_104'):
        raise ContractError('Research prototype requires its three named HC/anchor/S-ON contracts together; select a legacy profile for other combinations')
    if not source_spatial and hc_geometry_profile!='source_patch_v1':
        raise ValueError('Joint HC geometry requires the source-constrained mode, not scaled engineering preview')
    if hc_geometry_profile=='hc_joint_synthetic_v1':
        selected_profile=dict(selected_profile,HC_geometry_profile=hc_configuration['profile_id'],
            HC_geometry_configuration_sha256=hc_configuration['resolved_configuration_sha256'],
            allowed_approximation_ids=list(dict.fromkeys(selected_profile['allowed_approximation_ids']+hc_configuration['allowed_approximation_ids'])),
            region=dict(selected_profile['region'],HC_interval_policy='joint synthetic source support0..7.1mm; separate named profile, explicit finite arbor configuration'))
    if source_spatial:
        needed={'finite_induced_graph_boundary_v1','finite_background_prerun_v1','shared_H1_S_H2_kinetics_v1',
            'effective_BC_AC_RGC_dynamics_v1','native_to_voltage_adapter_v1','RBC_central_degree_geometry_transfer_v1',
            'AII_density_anchor_geometry_transfer_v1','midget_Wool_iso_density_adapter_v1','glycine_unverified_primary_quotation_v1',
            'McGregor_RF_OPL_rectangular_box_adapter_v1','McLaughlin_peripheral_slice_to_ROI_transfer_v1'}
        missing=needed-set(selected_profile['allowed_approximation_ids'])
        if missing:raise ValueError('Execution profile does not permit required named approximations: '+','.join(sorted(missing)))
    cone_route=cone_route or selected_profile['cone_route']
    hard_support_mm=selected_profile['geometry'].get('hard_support_mm')
    zero_contact_route=cone_route=='native_finite_geometry_zero_contact_approximation_v1'
    if cone_route not in ('native_shared_H1_approximation','legacy_native_Chen_blend_approximation','native_finite_geometry_zero_contact_approximation_v1'):raise ValueError('Unknown named cone route')
    if zero_contact_route and not research_contacts:raise ContractError('Zero-contact execution requires the certified research finite geometry profile')
    wiring_audits=[]
    legacy_matrix=globals()['local_matrix']
    def local_matrix(pre,post,k=4,radius=None,frame='IPL'):
        if not source_spatial:return legacy_matrix(pre,post,k,radius,frame)
        from retina_sim.spatial_profiles import bounded_matrix
        pre_points=ipl_points(pre,'presynaptic') if frame=='IPL' else contact_positions(pre,frame)
        post_points=ipl_points(post,'postsynaptic') if frame=='IPL' else contact_positions(post,frame)
        W,rows=bounded_matrix(pre_points,post_points,k,
            support_mm=hard_support_mm,gaussian_scale=radius)
        nearest,nearest_indices=cKDTree(pre_points).query(post_points,k=1)
        for j,(row,distance,index) in enumerate(zip(rows,nearest,nearest_indices)):
            row['nearest_available_source_distance_mm']=float(distance)
            if row['missing_degree']:
                row.update(post_cell_id=post.cells.cell_id[j],post_contact_xy_mm=post_points[j].tolist(),
                    nearest_source_cell_id=pre.cells.cell_id[index],nearest_source_contact_xy_mm=pre_points[index].tolist(),
                    contact_coordinate_frame=frame)
        wiring_audits.append({'pre':pre.population_id,'post':post.population_id,'frame':frame,
            'support_evidence':'project finite support 40 um; no measured arbor-radius claim','rows':rows})
        return W
    if secondary_gap_scale<0:raise ValueError('Secondary electrical strength must be nonnegative')
    ablations=set(ablations)
    if not ablations<= {'ampa','gap','glycine'}:raise ValueError('Unknown primary-pathway ablation')
    if not 0<=adaptation_seconds<duration:raise ValueError('Adaptation interval must lie within supplied input')
    from retina_sim.provenance import SourceRecord
    asset=root/'sources/primary_pathway_effective_profile.json';raw=asset.read_bytes()
    registry.register_source(SourceRecord(PROJECT_SOURCE,'Explicit retractable primary rod/cone circuit project profile',('Project implementation',),2026,'https://pmc.ncbi.nlm.nih.gov/articles/PMC6218188/','supplement',None,str(asset.relative_to(root)),'Equation/parameter classification ledger; approximation is not source measurement',registry.runtime_profile.source_identity(PROJECT_SOURCE),None,'Project original configuration; source code MIT attribution retained','identity_verified','Qualitative Grimes primary routing and named conductance equations. All unmeasured numbers explicitly project_approximation.', '2026-10-10',{}))
    anchor_asset=root/'sources/mclaughlin2021_verified_anchors.json';anchor_raw=anchor_asset.read_bytes()
    registry.register_source(SourceRecord('MCLAUGHLIN2021_glycine_anchors','Macaque bipolar glycinergic spontaneous-event anchors',('McLaughlin et al.',),2021,'https://www.eneuro.org/content/8/1/ENEURO.0432-20.2020','data_table',None,str(anchor_asset.relative_to(root)),'Parent source-verification extract: spontaneous event peaks and reversals; not original experimental dataset',registry.runtime_profile.source_identity('MCLAUGHLIN2021_glycine_anchors'),None,'Source citation and factual parameter extract','identity_verified','Supplied source quotation; factual extraction stored separately; primary verification is pending.', '2026-10-10',{}))
    register_profile(registry,PROJECT_SOURCE,1,{},reason='Explicit geometry/contact approximation provenance profile')
    if not source_spatial and (effective_cones<64 or effective_rods<64):raise ValueError('Declare at least64 preview effective cells to cover region/types; no implicit downsampling')
    raw_density=root/'data/published_code/packer_cone_density.csv'
    density=PackerDensityField(frozen_packer_table(root),central_policy='central_constant_cap',angular_policy='log_NT_cosine')
    disp=McGregorDisplacement(json.loads((root/'sources/spatial/mcgregor2018_production_map_nodes.json').read_text()),density,profile_name='published_rounded_anchor_composite')
    domain=RetinalDomain(tuple(bounds_mm),tuple(bounds_mm),lambda xy:np.ones(len(xy),dtype=bool),PROJECT_SOURCE,'continuous_center_periphery_corridor')
    spatial_audit={'profile_id':'engineering_scaled_corridor','density_counts_are_anatomical':False}
    if source_spatial:
        from retina_sim.spatial_profiles import source_roi
        if cone_density_scale is not None or rod_density_scale is not None:raise ValueError('Source-constrained mode forbids density scale overrides')
        generation_bounds=None
        requested_owner=owner_bounds_mm or (4.59,4.61,-.01,.01)
        core_bounds=(requested_owner[0]-halo_mm,requested_owner[1]+halo_mm,requested_owner[2]-halo_mm,requested_owner[3]+halo_mm)
        output_domain=RetinalDomain(core_bounds,tuple(requested_owner),lambda xy:np.ones(len(xy),bool),PROJECT_SOURCE,'frozen_RF_output_cohort')
        if typed_ipl:
            from retina_sim.regional_support import opl_roi_domain
            from retina_sim.ipl_geometry import plan_feedforward_guard
            core_opl,_=opl_roi_domain(disp,output_domain,seed)
            generation_bounds,guard=plan_feedforward_guard(disp,core_bounds,core_opl.bounds_mm)
            ipl_audit['generation_plan']=guard
        try:
            domain,mosaic,hc_mosaics,spatial_audit=source_roi(root,density,seed,
                owner_bounds=requested_owner,halo_mm=halo_mm,max_cells=max_spatial_cells,displacement=disp,hc_configuration=hc_configuration,
                generation_bounds_mm=generation_bounds)
        except ContractError as error:
            error.partial_build_audit={'phase':'source_ROI_and_HC_generation','IPL_geometry':ipl_audit,
                'owner_bounds_mm':list(requested_owner),'max_spatial_cells':max_spatial_cells,
                'HC_configuration':hc_configuration,'compiled_support_errors':[str(error)],
                'actual_requested_owner_counts':'NOT_CONSTRUCTED; upstream generation failed'}
            raise
        bounds_mm=domain.bounds_mm
    mass=integrate_density(density,domain)['mass'];scale=1. if source_spatial else effective_cones/mass if cone_density_scale is None else float(cone_density_scale)
    if scale<=0:raise ValueError('Positive explicit cone density scale required')
    if not source_spatial:mosaic=MosaicBackend().generate('greedy_capacitated_equal_rho_mass',lambda xy:scale*density(xy),domain,seed,quadrature_shape=(max(8192,int(np.ceil(scale*mass))),8),iterations=1)
    s_source=json.loads((root/'sources/spatial/demonasterio1985_Scone_reference.json').read_text())
    # This source stores the analytic fit under the source-correct field inspected below.
    s_field=SConeDensityField(s_source['fit_parameters'] if 'fit_parameters' in s_source else s_source,.223,magnification_source_id=PROJECT_SOURCE,central_policy='assay_square_PCHIP')
    types,type_audit=assign_cone_types(mosaic,lambda xy:scale*density(xy),lambda xy:scale*s_field(xy),domain,1.,PROJECT_SOURCE,seed,.5)
    if not np.any(types=='S'):raise ValueError('Declared effective resolution has no S cone; raise it explicitly')
    xy=mosaic.positions.values;n=len(xy);ecc=np.linalg.norm(xy,axis=1)
    region=navigation_regions
    mapped=disp.apply(q(xy,'mm','functional_position','functional_visual_field'))
    if source_spatial:
        hbounds=spatial_audit['HC_frame_policy']['bounds_mm'];p=mapped['pedicle'].values
        if np.any((p[:,0]<hbounds[0])|(p[:,0]>hbounds[1])|(p[:,1]<hbounds[2])|(p[:,1]>hbounds[3])):
            raise ValueError('Mapped cone pedicle outside explicit HC OPL bounding domain; no silent frame extent repair')
    cones=population('cones','cone',types,xy,region(xy),PROFILE,'macaque_H1_source_unspecified',xy,mapped['pedicle'].values)
    if typed_ipl:
        instance=hashlib.sha256(json.dumps({'bounds':bounds_mm,'seed':seed,'xy':xy.tolist()},sort_keys=True).encode()).hexdigest()[:20]
        cone_ids=tuple(f'cone_instance:{instance}:{cell_id}' for cell_id in mosaic.cell_ids)
        cones=replace(cones,cells=replace(cones.cells,cell_id=cone_ids))
        ipl_audit['cone_canonical_instance_id']=instance
        ipl_audit['DB_sampling']='LM[::4] in frozen complete canonical cone ordering, before output-domain crop; independent domains create different instances'
    pops={'cones':cones};components=[];links=[];gaps=[];externals=[];by={}
    def add_population(pid,kind,positions,profile,subtypes=None,anatomy=None,cell_ids=None):
        pop=population(pid,kind,subtypes or (pid,)*len(positions),positions,region(positions),profile,'Macaque_effective_project_profile',anatomy)
        if cell_ids is not None:pop=replace(pop,cells=replace(pop.cells,cell_id=tuple(cell_ids)))
        if kind=='rod':pop=replace(pop,pedicle_positions=q(positions,'mm','pedicle_position','OPL'),terminal_positions=q(positions,'mm','terminal_position','OPL'),dendritic_positions=q(positions,'mm','dendritic_position','OPL'))
        pops[pid]=pop;return pop
    hidx=np.arange(0,n,max(1,n//16));hxy=xy[hidx];hpos=np.repeat(hxy,2,axis=0)
    hc=population('HC','HC',('H1','H2')*len(hxy),hpos,region(hpos),PROFILE,'macaque_H1_source_unspecified',np.repeat(mapped['pedicle'].values[hidx],2,axis=0));pops['HC']=hc
    if source_spatial:
        hpos=np.concatenate([m.positions.values for _,m in hc_mosaics]);hsub=tuple(t for t,m in hc_mosaics for _ in m.cell_ids)
        if not len(hpos):raise ValueError('Actual HC count is zero; native HC assembly requires cells. No minimum count is inserted; explicitly enlarge the canonical domain.')
        hc_runtime_owner=np.concatenate([m.owner for _,m in hc_mosaics])
        # HC coordinates are constructed directly in physical OPL, independently of cone RF density.
        hc=population('HC','HC',hsub,hpos,region(hpos),PROFILE,'macaque_H1_source_unspecified',hpos,hpos);pops['HC']=hc
        hc=replace(hc,cells=replace(hc.cells,rf_position=None,
            soma_position=q(hpos,'mm','anatomical_position','anatomical_retina'),coordinate_frame='anatomical_retina',
            eccentricity=q(np.linalg.norm(hpos,axis=1),'mm','global_eccentricity','anatomical_retina')),
            receptive_field_positions=None)
        if hc_geometry_profile=='hc_joint_synthetic_v1':
            stable_ids=tuple(cell_id for _,m in hc_mosaics for cell_id in m.cell_ids)
            hc=replace(hc,cells=replace(hc.cells,cell_id=stable_ids))
            if not stable_ids:raise ValueError('Actual joint HC count is zero; native HC assembly requires cells. No minimum count is inserted; explicitly enlarge the canonical domain.')
        pops['HC']=hc
    if source_spatial:
        if hc_geometry_profile=='hc_joint_synthetic_v1':
            from retina_sim.hc_arbor import build_hc_arbor_contacts,homotypic_gap_candidates
            try:
                A,B,contacts=build_hc_arbor_contacts(cones.cells.cell_id,cones.cells.rf_position,cones.pedicle_positions,
                    hc.cells.cell_id,hc.cells.subtype,hc.anatomical_positions,
                    q(hpos,'mm','arbor_anchor_position','OPL'),mosaic.owner,hc_configuration['footprints'],
                    cone_types=types,source_radial_support_mm=hc_configuration.get('contact_source_radial_support_mm',(0.,hc_configuration['support_radius_mm'])),
                    cone_pedicle_generation_bounds_mm=spatial_audit['HC_frame_policy']['mapped_input_bounds_mm'],
                    hc_generation_bounds_mm=spatial_audit['HC_geometry']['active_bounds_mm'],
                    divergence_profile=hc_configuration.get('divergence_profile'),
                    require_owner_quota=hc_configuration.get('require_owner_quota',False),
                    anchor_applicability_band_mm=hc_configuration.get('anchor_applicability_band_mm'),
                    source_anchor_band_mm=hc_configuration.get('source_anchor_band_mm'),
                    final_contact_distance_mm=hc_configuration.get('final_contact_distance_mm'),
                    contact_policy=hc_configuration.get('contact_policy'),owner_hc_mask=hc_runtime_owner)
            except ContractError as error:
                hc_r=np.linalg.norm(hpos,axis=1);cone_r=np.linalg.norm(cones.pedicle_positions.values,axis=1)
                error.partial_build_audit={'phase':'HC_finite_arbor_contacts','IPL_geometry':ipl_audit,
                    'HC_geometry':spatial_audit['HC_geometry'],'compiled_support_errors':[str(error)],
                    'source_contact_radial_support_mm':hc_configuration.get('contact_source_radial_support_mm'),
                    'actual_HC_radial_range_mm':[float(hc_r.min()),float(hc_r.max())],
                    'actual_cone_pedicle_radial_range_mm':[float(cone_r.min()),float(cone_r.max())],
                    'actual_owner_cone_counts':{t:int(np.sum((types==t)&mosaic.owner)) for t in ('L','M','S')},
                    'actual_requested_owner_counts':'NOT_CONSTRUCTED; upstream contact assembly failed'}
                raise
            spatial_audit['HC_geometry']['contact_audit']=contacts
            if research_contacts:
                contacts['required_pedicle_query_domain']['source_provider']={
                    'provider':'frozen Packer RF density + named McGregor P-chart map; published-code tabulation and project adapters',
                    'RF_materialized_bounds_mm':list(domain.bounds_mm),
                    'tabulated_density_minimum_mm':float(density.minimum),
                    'tabulated_density_maximum_by_meridian_mm':{m:float(k[-1]) for m,k in density.knots.items()},
                    'central_extension':'named central_constant_cap below tabulated minimum, project approximation',
                    'OPL_materialized_bounds_mm':spatial_audit['HC_frame_policy']['mapped_input_bounds_mm'],
                    'owner_required_query_truncated':False,
                    'outer_halo_footprints_truncated':contacts['required_pedicle_query_domain']['halo_incomplete_count'],
                    'coverage_basis':'declared numerical provider bounding domain; source-to-OPL map and finite recurrent closure remain project approximations'}
            spatial_audit['HC_geometry']['gap_candidate_audit']=homotypic_gap_candidates(hc.cells.cell_id,hc.cells.subtype,
                hc.anatomical_positions,hc_configuration.get('gap_candidate_max_distance_mm'),
                source_profile_id='user_configured_HC_gap_candidate_geometry_v1')
            from retina_sim.hc_geometry import hc_contact_runtime_view
            spatial_audit['HC_geometry']['generated_active_population']=hc.record()
            zero_certificate=None
            if zero_contact_route:
                from retina_sim.finite_feedback_native import certify_zero_contacts
                zero_certificate=certify_zero_contacts(cones,hc,A,B,contacts,hc_configuration,
                    spatial_audit['HC_geometry']['active_bounds_mm'])
            try:
                hc,A,B,hc_participants,runtime_view=hc_contact_runtime_view(hc,A,B,zero_contact_certificate=zero_certificate)
            except ContractError as error:
                missing=np.flatnonzero(np.asarray(B.sum(1)).ravel()==0)
                error.partial_build_audit={'phase':'HC_native_runtime_binding','IPL_geometry':ipl_audit,
                    'HC_geometry':spatial_audit['HC_geometry'],'compiled_support_errors':[str(error)],
                    'uncovered_cones':[{'cone_id':cones.cells.cell_id[i],'type':str(types[i]),
                        'is_owner':bool(mosaic.owner[i]),'RF_mm':xy[i].tolist(),
                        'pedicle_OPL_mm':cones.pedicle_positions.values[i].tolist()} for i in missing]}
                raise
            hc_runtime_owner=hc_runtime_owner[hc_participants]
            hpos=hc.anatomical_positions.values;pops['HC']=hc
            spatial_audit['HC_geometry']['runtime_view']=runtime_view
            if zero_contact_route:
                from retina_sim.finite_feedback_native import graph_hash
                zero_certificate['runtime_graph_hash']=graph_hash(cones.cells.cell_id,hc.cells.cell_id,A,B)
                spatial_audit['HC_geometry']['zero_contact_execution']=zero_certificate
        else:
            from retina_sim.spatial_profiles import hc_contact_matrices
            A,B,contacts=hc_contact_matrices(contact_positions(cones,'OPL'),types,contact_positions(hc,'OPL'),hc.cells.subtype,mosaic.owner,support_mm=hard_support_mm)
        wiring_audits.append({'route':'cone_HC_reciprocal_contacts','frame':'OPL','contacts':contacts})
    else:
        A=local_matrix(cones,hc,8,radius=.08+.06*np.linalg.norm(hpos,axis=1),frame='OPL').tolil();B=local_matrix(hc,cones,4,radius=.08+.06*ecc,frame='OPL')
        for j,sub in enumerate(hc.cells.subtype):
            if sub=='H1':A[j,np.flatnonzero(types=='S')]=0
        A=sparse.csr_matrix(A);sums=np.asarray(A.sum(axis=1)).ravel()
        A=sparse.diags(np.divide(1.,sums,out=np.zeros_like(sums),where=sums>0))@A
    outer=ComponentSpec('outer_native',KEY,{'cone':'cones','h1':'HC'},PROFILE,{'explicit_opt_in':True,'A':A,'B':B,'background_td':q(np.full(n,light_context.native_background_td),'td','native_retinal_illuminance'),'pooling_policy':'pool_Vbar_before_aI','geometry_frame':'OPL','observation_interval_seconds':.0001,'unsupported_type_policy':'project_approximation_shared_H1_kinetics'},required_layer='cone_native_synapse')
    if zero_contact_route:
        from retina_sim.finite_feedback_native import KEY as zero_key
        outer=replace(outer,mechanism=zero_key,options=dict(outer.options,zero_contact_certificate=zero_certificate))
    components.append(outer);by['outer_native']=cones;externals.append(ExternalInput('native_light','outer_native','light'))
    # Physical peripheral cone route, explicit same-RF alternative representations.
    physical_indices=np.flatnonzero((ecc/.223>20)&(types!='S'));pxy=xy[physical_indices]
    if len(pxy)==0 and cone_route=='legacy_native_Chen_blend_approximation':raise ValueError('Legacy blend needs its peripheral source representation')
    pn=len(pxy);physical=None
    if pn:
        physical=population('peripheral_physical_cones','cone',types[physical_indices],pxy,region(pxy),CHEN_PROFILE,CHEN_SPECIES,pxy,mapped['pedicle'].values[physical_indices])
        physical=replace(physical,cells=replace(physical.cells,eccentricity=q(ecc[physical_indices]/.223,'deg','global_eccentricity','functional_visual_field')))
        pops[physical.population_id]=physical;by['physical_OS']=physical
        components.append(ComponentSpec('physical_OS','Chen2024PeripheralLMCurrent',{'cone':physical.population_id},CHEN_PROFILE,{'explicit_opt_in':True,'initial_background_rate':q(np.full(len(pxy),light_context.cone_background_rstar_s),'R*/s','photoisomerization_rate')},'cone_outer_segment'));externals.append(ExternalInput('physical_cone_light','physical_OS','light'))
        pn=len(pxy)
        physical_native=RetinaBuilder(registry).registry.instantiate(components[-1],{physical.population_id:physical},registry,RunMode.REFERENCE)
        adapted_source_current=physical_native.observe('current',0,tuple(v.values for v in physical_native.initial()),{}).values*1e-12
        pid='effective_physical_cone';entries={'C':(np.array([5e-12,10e-12]),'F','membrane_capacitance'),'gL':(np.array([.25e-9,.5e-9]),'S','leak_conductance'),'axial':(np.array([[0,2e-9],[2e-9,0]]),'S','axial_conductance'),'injection':(np.vstack((np.ones(pn),np.zeros(pn))),'dimensionless','current_injection_distribution'),'current_anchor':(adapted_source_current,'A','absolute_outer_segment_current'),'initial_deviation':(np.zeros((2,pn)),'V','physical_voltage_deviation'),'voltage_anchor':(np.full((2,pn),-.055),'V','physical_membrane_voltage')}
        register_profile(registry,pid,pn,entries,reason='Two-compartment physical cone reduction; C/gL/axial/V anchor uncalibrated; current anchor derived from source equilibrium at1000R*/s; voltage anchor remains independent project approximation')
        physical_mem=add_population('cone_compartments','cone',pxy,pid,tuple(types[physical_indices]),pxy);by['physical_membrane']=physical_mem
        components.append(ComponentSpec('physical_membrane','PassiveConeCable',{'cells':'cone_compartments'},pid,{'absolute_voltage_anchor':True,'terminal_compartment':1},'cone_passive_cable'))
        pidunit='physical_unit_adapter';register_profile(registry,pidunit,pn,{});by['physical_current_SI']=physical_mem
        components.append(ComponentSpec('physical_current_SI','ExplicitPhysicalUnitConversion',{'cells':'cone_compartments'},pidunit,{'source_unit':'pA','target_unit':'A','meaning':'absolute_outer_segment_current'}))
        # Unit adapter uses same target population but no physiological parameter defaults.
        components[-1]=replace(components[-1],options=dict(components[-1].options,target_population_profile_id='effective_physical_cone'))
        termpid='effective_physical_terminal';register_profile(registry,termpid,pn,{'Vhalf':(np.full(pn,-.036),'V','activation_half_voltage'),'slope':(np.full(pn,.004),'V','activation_voltage_slope')})
        terminal=add_population('cone_terminals','cone',pxy,termpid,tuple(types[physical_indices]),pxy);by['physical_terminal']=terminal
        components.append(ComponentSpec('physical_terminal','QuasiSteadyTerminalActivation',{'cells':'cone_terminals'},termpid,{'approximation':'quasi_steady_nonloading'},'terminal_activation_proxy'))
        hcpid='effective_physical_HC';register_profile(registry,hcpid,pn,{'tau':(np.full(pn,.020),'s','effective_HC_time_constant'),'K':(np.full(pn,.25),'dimensionless','HC_half_drive'),'n':(np.full(pn,2.),'dimensionless','HC_hill_exponent'),'kappa':(np.full(pn,.002),'V','HC_threshold_shift_gain'),'H_anchor':(np.full(pn,.001),'dimensionless','effective_HC_anchor'),'H_initial':(np.full(pn,.001),'dimensionless','effective_HC_activation')},reason='Physical branch effective HC threshold feedback; uncalibrated reduction, no invented pA/native mapping')
        if source_spatial:
            from scipy.optimize import brentq
            from scipy.special import expit
            hp=registry.mechanism_profiles[hcpid];tp=registry.mechanism_profiles[termpid]
            voltage0=registry.mechanism_profiles['effective_physical_cone'].quantity('voltage_anchor','V','physical_membrane_voltage').values[1]+registry.mechanism_profiles['effective_physical_cone'].quantity('initial_deviation','V','physical_voltage_deviation').values[1]
            half=tp.quantity('Vhalf','V','activation_half_voltage').values;slope=tp.quantity('slope','V','activation_voltage_slope').values
            K=hp.quantity('K','dimensionless','HC_half_drive').values;exponent=hp.quantity('n','dimensionless','HC_hill_exponent').values
            gain=hp.quantity('kappa','V','HC_threshold_shift_gain').values;anchor=hp.quantity('H_anchor','dimensionless','effective_HC_anchor').values
            def residual(h,i):
                release=expit((voltage0[i]-half[i]-gain[i]*(h-anchor[i]))/slope[i])
                drive=(release/K[i])**exponent[i]
                return h-drive/(1+drive)
            h0=np.array([brentq(lambda h:residual(h,i),0.,1.,xtol=1e-14) for i in range(pn)])
            records=tuple(replace(r,value=q(h0,'dimensionless','effective_HC_activation'),locator='Scalar equilibrium of declared physical terminal/HC feedback at cable background reference; effective closure, not calibrated release') if r.parameter_id=='H_initial' else r for r in hp.records)
            hp=replace(hp,records=records);registry.mechanism_profiles[hcpid]=hp;registry.profiles[hcpid]=hp
        hcpop=add_population('physical_HC','HC',pxy,hcpid);by['physical_HC']=hcpop
        components.append(ComponentSpec('physical_HC','EffectiveHCThresholdFeedback',{'cells':'physical_HC'},hcpid))
        links.extend([connection('physical_terminal','release','physical_HC','drive',sparse.eye(pn),by,frame='OPL'),connection('physical_HC','shift','physical_terminal','shift',sparse.eye(pn),by,frame='OPL')])
        links.extend([connection('physical_OS','current','physical_current_SI','input',sparse.eye(pn),by,frame='OPL'),connection('physical_current_SI','output','physical_membrane','current',sparse.eye(pn),by,frame='OPL'),connection('physical_membrane','terminal_voltage','physical_terminal','voltage',sparse.eye(pn),by,frame='OPL')])
    lm=np.flatnonzero(types!='S');si=np.flatnonzero(types=='S');didx=lm[::4]
    native_outer=RetinaBuilder(registry).registry.instantiate(outer,pops,registry,RunMode.REFERENCE)
    if zero_contact_route:
        spatial_audit['HC_geometry']['zero_contact_background_equilibrium']=native_outer.native_model.background_equilibrium_audit
    native_baseline=native_outer.observe('release',0,tuple(v.values for v in native_outer.initial()),{}).values
    definitions=[('midget_BC_ON','cone_BC_ON',lm),('midget_BC_OFF','cone_BC_OFF',lm),('diffuse_BC_ON','cone_BC_ON',didx),('diffuse_BC_OFF','cone_BC_OFF',didx),('S_ON_BC','cone_BC_ON',si)]
    for cid,kind,idx in definitions:
        if source_spatial and cid.startswith('midget') and np.any(np.linalg.norm(mapped['pedicle'].values[idx],axis=1)>10.):
            raise ValueError('Midget BC private-line source limit <=10 mm anatomical/OPL; no unnamed regional extension')
        pid='effective_'+cid;nn=len(idx);fill=lambda v:np.full(nn,v)
        entries={'kernel':([.5,.25,-.1],'dimensionless','causal_residual_kernel'),'sample_interval':(.001,'s','history_sampling_interval'),'bias':(fill(0.),'dimensionless','activation_bias'),'baseline':(native_baseline[idx],'dimensionless','input_baseline'),'initial_history':(np.zeros((3,nn)),'dimensionless','causal_input_history'),'C':(fill(20e-12),'F','membrane_capacitance'),'gL':(fill(1e-9),'S','leak_conductance'),'Vref':(fill(-.050),'V','terminal_voltage_reference'),'native_voltage_gain':(fill(.010),'V','voltage_per_native_activation'),'native_reference':(fill(np.log(2.)),'dimensionless','native_activation_reference'),'V0':(fill(-.050),'V','initial_membrane_voltage'),'release_half':(fill(-.050),'V','activation_half_voltage'),'release_slope':(fill(.005),'V','activation_voltage_slope'),'tau_release':(fill(.003),'s','release_time_constant')}
        if kind=='cone_BC_OFF':entries['EI']=(fill(-.076 if cid=='midget_BC_OFF' else -.073),'V','inhibitory_reversal')
        register_profile(registry,pid,nn,entries,reason='Original finite BC history retained with separately explicit physical terminal/native gain; project reduced profile not macaque fit')
        if kind=='cone_BC_OFF':
            p=registry.mechanism_profiles[pid]
            records=tuple(replace(r,evidence_class='project_approximation',source_id='MCLAUGHLIN2021_glycine_anchors',locator='Unverified primary-source quotation: macaque glycine event reversal for FMB or DB1; extraction identity alone does not verify measurement',supported_domain={'type':'FMB' if cid=='midget_BC_OFF' else 'DB1','provenance_tag':'SOURCE_QUOTATION_PRIMARY_VERIFICATION_PENDING'}) if r.parameter_id=='EI' else r for r in p.records)
            p=replace(p,records=records,source_ids=p.source_ids+('MCLAUGHLIN2021_glycine_anchors',));registry.mechanism_profiles[pid]=p;registry.profiles[pid]=p
        pop=add_population(cid,'BC',xy[idx],pid,anatomy=mapped['pedicle'].values[idx],
            cell_ids=tuple(cid+':'+cones.cells.cell_id[i] for i in idx) if typed_ipl else None);by[cid]=pop
        options={'polarity':'ON' if kind=='cone_BC_ON' else 'OFF'}
        if kind=='cone_BC_OFF':options['receptor_scope']='Effective residual implementation, not explicit AMPA/KAR states; macaque OFF BC source supports KAR, receptor/desensitization generative fit remains unbound'
        if cid.startswith('midget'):options['parent_cone_ids']=tuple(cones.cells.cell_id[i] for i in idx)
        components.append(ComponentSpec(cid,'HistoryBipolarTerminal',{'cells':cid},pid,options,cid))
        W=sparse.eye(n,format='csr')[idx] if cid.startswith('midget') or cid=='S_ON_BC' else local_matrix(cones,pop,8,radius=.08+.04*ecc[idx],frame='OPL')
        nativeW=W.copy().tolil();blend_u=np.minimum(1.,np.maximum(0.,(ecc[physical_indices]/.223-20.)/5.));blend=blend_u**2*(3-2*blend_u)
        if cone_route in ('native_shared_H1_approximation','native_finite_geometry_zero_contact_approximation_v1'):blend=np.zeros_like(blend)
        if pn:
            nativeW[:,physical_indices]=W[:,physical_indices]@sparse.diags(1-blend);nativeW=nativeW.tocsr();physW=W[:,physical_indices]@sparse.diags(blend)
        else:nativeW=W.tocsr();physW=sparse.csr_matrix((nn,0))
        # Baseline follows the actual selected native/physical route, not an arbitrary dark value.
        baseline=np.asarray(nativeW@native_baseline).ravel()+np.asarray(physW@np.full(pn,1/(1+np.exp(-(-.055+.036)/.004)))).ravel()
        profile=registry.mechanism_profiles[pid]
        records=tuple(replace(r,value=q(baseline,'dimensionless','input_baseline'),locator='Derived input baseline from declared native equilibrium and selected physical terminal reference') if r.parameter_id=='baseline' else r for r in profile.records)
        profile=replace(profile,records=records);registry.mechanism_profiles[pid]=profile;registry.profiles[pid]=profile
        if nativeW.nnz:links.append(connection('outer_native','release',cid,'drive',nativeW,by,frame='OPL'))
        if physW.nnz:links.append(connection('physical_terminal','release',cid,'drive',physW,by,frame='OPL'))
    # Prepare displaced inner-retinal support before generating its upstream
    # tissue. Enlarging only AII leaves most of that physical patch without RBCs.
    rod_domain=rbc_domain=domain
    if source_spatial:
        from retina_sim.spatial_profiles import aii_ipl_support_domain,presynaptic_support_domain
        aii_domain,aii_domain_audit=aii_ipl_support_domain(domain,
            spatial_audit['HC_frame_policy']['owner_bounds_mm'],support_mm=hard_support_mm,
            bc_target_positions=np.concatenate([contact_positions(by[cid],'IPL') for cid,_,_ in definitions]))
        if aii_domain_audit['domain_extended']:
            rbc_domain=presynaptic_support_domain(aii_domain,hard_support_mm,domain_name='RBC_actual_AII_IPL_support')
            rod_domain=presynaptic_support_domain(rbc_domain,hard_support_mm,domain_name='rod_actual_RBC_OPL_support')
    # Explicit effective rod density: continuous annulus, no rods in declared foveal center.
    def rod_density(points):
        r=np.linalg.norm(points,axis=1);theta=np.arctan2(points[:,1],points[:,0]);u=np.maximum(0,(r-.020)/(4.-.020))
        peak=120000+np.where(np.sin(theta)>=0,57000,30000)*np.abs(np.sin(theta))
        return np.where(r>.020,peak*u*np.exp(1-u),0.)
    rod_mass=integrate_density(rod_density,rod_domain)['mass'];rod_scale=1. if source_spatial else effective_rods/rod_mass if rod_density_scale is None else float(rod_density_scale)
    if rod_scale<=0:raise ValueError('Positive explicit rod density scale required')
    if source_spatial:
        from retina_sim.spatial_profiles import generate_density
        rmosaic=generate_density(rod_density,rod_domain,seed+1,max_spatial_cells)
    else:rmosaic=MosaicBackend().generate('greedy_capacitated_equal_rho_mass',lambda p:rod_scale*rod_density(p),domain,seed+1,quadrature_shape=(max(8192,int(np.ceil(rod_scale*rod_mass))),8),iterations=1)
    rxy=rmosaic.positions.values
    rp='Chen2024_primateRod_source';vals=rod_source_values(root)
    rod_entries={k:(v,('Chen_scaled_concentration' if k in ('gdark','cdark','kGC') else 'pA/Chen_scaled_concentration^3' if k=='k' else CHEN_PARAMETER_SPECS['CHEN2024_'+k][0]), 'source_coefficient_'+k) for k,v in vals.items()}
    register_profile(registry,rp,len(rxy),rod_entries,source_id='CHEN2024_initPhotoreceptorParams_m',evidence='published_fit',reason='Pinned author initPhotoreceptorParams.m primateRod branch; source effective concentration normalization')
    rod=add_population('rod','rod',rxy,rp,('rod',)*len(rxy));by['rod']=rod
    components.append(ComponentSpec('rod','Chen2024PrimateRodCurrent',{'cells':'rod'},rp,{'initial_background_rate':q(np.full(len(rxy),light_context.rod_background_rstar_s),'R*/s','photoisomerization_rate')},required_layer='rod'));externals.append(ExternalInput('rod_light','rod','light'))
    rod_runtime=RetinaBuilder(registry).registry.instantiate(components[-1],{'rod':rod},registry,RunMode.REFERENCE)
    rod_background_current=rod_runtime.observe('current',0,tuple(v.values for v in rod_runtime.initial()),{}).values*1e-12
    rmemid='effective_rod_membrane';rod_membrane_entries=membrane_entries(len(rxy),'rod_membrane')
    rod_membrane_entries=registry.runtime_profile.bind(rmemid,rod_membrane_entries)
    rod_membrane_entries['current_anchor']=(rod_background_current+rod_membrane_entries['gL'][0]*(rod_membrane_entries['V0'][0]-rod_membrane_entries['EL'][0]),'A','absolute_outer_segment_current')
    register_profile(registry,rmemid,len(rxy),rod_membrane_entries,reason='Declared voltage-reference leak balance at LightContext rod background; current anchor derived from source equilibrium, V0/leak are uncalibrated project choices')
    rodmem=add_population('rod_membrane','rod',rxy,rmemid,('rod',)*len(rxy));by['rod_membrane']=rodmem
    components.append(ComponentSpec('rod_membrane','ReceptorConductanceMembrane',{'cells':'rod_membrane'},rmemid,{'cell_kind':'rod_membrane'}))
    unitpid='rod_current_adapter';register_profile(registry,unitpid,len(rxy),{});by['rod_current_SI']=rodmem
    components.append(ComponentSpec('rod_current_SI','ExplicitPhysicalUnitConversion',{'cells':'rod_membrane'},unitpid,{'source_unit':'pA','target_unit':'A','meaning':'absolute_outer_segment_current','target_population_profile_id':rmemid}))
    links.extend([connection('rod','current','rod_current_SI','input',sparse.eye(len(rxy)),by,frame='OPL'),connection('rod_current_SI','output','rod_membrane','current',sparse.eye(len(rxy)),by,frame='OPL')])
    rbxy=rxy[::12];npop=len(rbxy)
    rod_inner_geometry={'basis':'effective_sampling','RBC_stride':12,'AII_layout':'legacy_y0'}
    if source_spatial:
        rbc_field=lambda p:selected_profile['geometry']['RBC_to_rod_density_ratio']*rod_density(p)
        rbc_mosaic=generate_density(rbc_field,rbc_domain,seed+2,max_spatial_cells)
        # The effective AC/parasol RF scaffold was tied to the original RBC
        # sample. Keep that cohort, count and RF IDs separate from physical
        # upstream RBC tissue added to cover the displaced AII calculation.
        scaffold_domain=output_domain if typed_ipl else domain
        inner_rf_scaffold=(None if parasol_rf_seeds is not None else
            rbc_mosaic if rbc_domain is scaffold_domain else generate_density(rbc_field,scaffold_domain,seed+2,max_spatial_cells))
        aii_mosaic=generate_density(lambda p:np.full(len(p),selected_profile['geometry']['AII_density_cells_per_mm2']),aii_domain,seed+3,max_spatial_cells)
        if aii_domain_audit['domain_extended']:
            from retina_sim.spatial_profiles import finite_support_coverage
            aii_owner=finite_support_coverage(aii_mosaic.positions.values,0.,
                [domain.owner_bounds_mm,spatial_audit['HC_frame_policy']['owner_bounds_mm']])
            aii_mosaic=replace(aii_mosaic,owner=aii_owner,diagnostics=dict(aii_mosaic.diagnostics,
                owner_count=int(aii_owner.sum()),halo_count=int((~aii_owner).sum()),
                coordinate_frame='IPL',owner_policy='union of actual rod/RBC and mapped BC owner patches'))
        rbxy=rbc_mosaic.positions.values;npop=len(rbxy)
        rod_inner_geometry={'profile_id':'macaque_rod_aii_geometry_effective_v1',
            'basis':'source_anchored_approx_density','RBC_formula':'2*rho_rod/36',
            'RBC_scope':'rounded central convergence assumption transferred to selected peripheral ROI; not measured regional density',
            'RBC_mosaic':rbc_mosaic.diagnostics,'AII_cells_per_mm2':5000.,
            'AII_scope':'1.5 mm Mills anchor transferred with constant density to selected ROI; project approximation',
            'AII_mosaic':aii_mosaic.diagnostics,'AII_generation_domain':aii_domain_audit,
            'RBC_generation_bounds_mm':list(rbc_domain.bounds_mm),'rod_generation_bounds_mm':list(rod_domain.bounds_mm),
            'upstream_generation_policy':'physical AII support + unchanged RBC/rod density fields and one hard-support halo per upstream layer; no cone displacement applied to rods/RBCs',
            'AC_RGC_RF_scaffold':({'basis':'explicit supplied RF seed geometry; origin retained',
                'count':len(parasol_rf_seeds.site_ids),'seed_input':parasol_rf_seeds.record()}
                if parasol_rf_seeds is not None else
                {'basis':'existing effective RBC sample on original RF ROI; no measured AC/RGC density claim',
                'count':len(inner_rf_scaffold.cell_ids),'mosaic':inner_rf_scaffold.diagnostics}),
            'AC_contact_frame_policy':{'RF_sites_preserved':True,'mapping':'existing McGregor RF->cone pedicle map',
                'IPL_anchor':'effective BC-path co-located point at mapped pedicle; same point approximation already used for cone BC terminals',
                'AC_soma_or_arbor_measured':False,'RBC_physical_sites_are_not_AC_RF_sites':True,
                'evidence_class':'project_approximation','source_profile':'published_rounded_anchor_composite + existing effective scaffold'},
            'source_primary_verification':'pending; quotations supplied in delegated contract'}
    # AII is independently placed down to inner-retinal 0.14mm; it remains cone-connected where rods/RBCs are absent.
    ax=np.r_[.15,.3,rbxy[:,0]];aiixy=np.c_[np.unique(ax),np.zeros(len(np.unique(ax)))];aiin=len(aiixy)
    if source_spatial:aiixy=aii_mosaic.positions.values;aiin=len(aiixy)
    for cid,kind,typ in [('rod_BC','rod_BC_ON','BC'),('AII_AC','AII','AC')]:
        positions=aiixy if cid=='AII_AC' else rbxy;size=len(positions)
        pid='effective_'+cid;register_profile(registry,pid,size,membrane_entries(size,kind))
        pop=add_population(cid,typ,positions,pid);by[cid]=pop;components.append(ComponentSpec(cid,'ReceptorConductanceMembrane',{'cells':cid},pid,{'cell_kind':kind},cid))
    contactW=local_matrix(rodmem,by['rod_BC'],40,frame='OPL').tocoo();cpid='effective_individual_TRPM1'
    contact_xy=(rxy[contactW.col]+rbxy[contactW.row])/2;cn=contactW.nnz
    rod_initial_release=initial_release(registry.mechanism_profiles[rmemid])
    register_profile(registry,cpid,cn,{'K':(np.full(cn,.25),'dimensionless','receptor_half_drive'),'hill':(np.full(cn,4.),'dimensionless','receptor_hill_exponent'),'tau':(np.full(cn,.010),'s','receptor_time_constant'),'gain':(2e-9*contactW.data,'S','TRPM1_maximum_contact_conductance'),'drive0':(rod_initial_release[contactW.col],'dimensionless','initial_release_drive')},reason='Each rod contact nonlinear before pooling; initial drive follows actual declared rod terminal; Hill/gain/10ms gate temporary project approximation')
    syn=add_population('individual_rod_RBC_contacts','BC',contact_xy,cpid);by['individual_rod_RBC_contacts']=syn
    components.append(ComponentSpec('individual_rod_RBC_contacts','RodContactTRPM1',{'cells':syn.population_id},cpid))
    take=sparse.csr_matrix((np.ones(cn),(np.arange(cn),contactW.col)),shape=(cn,len(rxy)))
    pool=sparse.csr_matrix((np.ones(cn),(contactW.row,np.arange(cn))),shape=(npop,cn))
    if source_spatial:
        contact_profile=registry.mechanism_profiles[cpid]
        drive0=contact_profile.quantity('drive0','dimensionless','initial_release_drive').values
        K=contact_profile.quantity('K','dimensionless','receptor_half_drive').values
        hill=contact_profile.quantity('hill','dimensionless','receptor_hill_exponent').values
        gate=1/(1+(drive0/K)**hill)
        excitation=np.asarray(pool@(contact_profile.quantity('gain','S','TRPM1_maximum_contact_conductance').values*gate)).ravel()
        p=registry.mechanism_profiles['effective_rod_BC']
        gL=p.quantity('gL','S','leak_conductance').values
        V0=(gL*p.quantity('EL','V','leak_reversal').values+excitation*p.quantity('EE','V','excitatory_reversal').values)/(gL+excitation)
        records=tuple(replace(r,value=q(V0,'V','initial_membrane_voltage'),locator='Analytical RBC membrane balance at actual constant rod/TRPM1 conductance; no incoming gap or chemical inhibition in selected primary graph') if r.parameter_id=='V0' else r for r in p.records)
        p=replace(p,records=records);registry.mechanism_profiles[p.profile_id]=p;registry.profiles[p.profile_id]=p
    links.extend([connection('rod_membrane','release','individual_rod_RBC_contacts','release',take,by,synapse='one_rod_per_contact',frame='OPL'),connection('individual_rod_RBC_contacts','conductance','rod_BC','excitation',pool,by,synapse='TRPM1_contact_conductance',frame='OPL')])
    chemical_audits={}
    def chemical(name,pre,post,receptor,k,peak,rise,decay,source_id=PROJECT_SOURCE,peak_evidence='project_approximation'):
        W=local_matrix(by[pre],by[post],k).tocoo();nn=W.nnz
        coverage=None
        if source_spatial:
            from retina_sim.spatial_profiles import finite_support_coverage
            rectangles=(aii_domain_audit['generation_rectangles_mm'] if pre=='AII_AC' else [rbc_domain.bounds_mm])
            coverage=finite_support_coverage(contact_positions(by[post],'IPL'),hard_support_mm,rectangles)
        if nn==0:
            if coverage is not None and not np.all(coverage):
                raise ContractError(f'{name}: IPL_SUPPORT_DOMAIN_UNGENERATED; zero candidate edges are not biological zero connectivity; {int((~coverage).sum())} target supports cross the generated presynaptic domain')
            # Empty sums are zero. Reuse the real target population/typed ports;
            # do not invent a contact cell merely to keep the operator nonempty.
            profile='effective_'+name;register_profile(registry,profile,0,{})
            by[name]=by[post]
            components.append(ComponentSpec(name,'DoubleExponentialChemicalSynapse',{'cells':by[post].population_id},profile,
                {'receptor':receptor,'empty_contact_operator':True,'empty_support_verified':True,
                 'target_population_profile_id':by[post].cells.parameter_profile_id[0]}))
            links.append(connection(name,'conductance',post,'excitation' if receptor=='excitatory' else 'inhibition',sparse.eye(len(by[post].cells.cell_id)),by))
            chemical_audits[name]={'per_edge_states':0,'edge_count':0,'empty_contact_operator':True,
                'status':'LEGAL_EMPTY_FINITE_GRAPH_ZERO_CONTRIBUTION','full_target_support_covered':True,
                'no_dummy_contacts':True,'zero_is_not_a_measured_connectivity_claim':True,'support_mm':hard_support_mm}
            return
        distances=np.linalg.norm(contact_positions(by[pre],'IPL')[W.col]-contact_positions(by[post],'IPL')[W.row],axis=1)
        effective_radius=.03 if source_spatial else max(.03,float(distances.max())*1.05)
        overlap=np.exp(-(distances/effective_radius)**2)
        positions=(contact_positions(by[pre],'IPL')[W.col]+contact_positions(by[post],'IPL')[W.row])/2
        profile='effective_'+name
        pre_profile=registry.mechanism_profiles[by[pre].cells.parameter_profile_id[0]]
        rates0=initial_event_rate(pre_profile)[W.col]
        register_profile(registry,profile,nn,{'tau_rise':(np.full(nn,rise),'s','synaptic_rise_time_constant'),'tau_decay':(np.full(nn,decay),'s','synaptic_decay_time_constant'),'g_peak':(np.full(nn,peak),'S','single_event_peak_conductance'),'ncontact':(overlap,'dimensionless','effective_release_site_multiplier'),'event_rate0':(rates0,'events/s','expected_release_event_rate')},reason='Peak-normalized double exponential expected-event conductance; times/site-overlap explicitly effective')
        if receptor=='inhibitory':
            low,high=(.0021,.0040) if post=='midget_BC_OFF' else (.0027,.0053)
            source_conditions={'recorded_cell_type':'FMB' if post=='midget_BC_OFF' else 'DB1',
                'recording_region':'>8 mm nasal superior/inferior retinal slice','species':'adult M. mulatta',
                'temperature_C':[31,33],'slice_thickness_um':300,'illumination':'low photopic',
                'liquid_junction_correction_mV':-10,'filter_Hz':2000,'sample_Hz':10000,
                'source_locator':'McLaughlin2021 Fig4/results; PMC7920536',
                'applied_policy':'McLaughlin_peripheral_slice_to_ROI_transfer_v1',
                'transfer_status':'project approximation; current ROI/input is not the original recording condition'}
            target_profile=registry.mechanism_profiles[by[post].cells.parameter_profile_id[0]]
            voltage0=target_profile.quantity('V0','V','initial_membrane_voltage').values[W.row]
            effective_decay=low+(high-low)*np.minimum(1.,np.maximum(0.,(voltage0+.1)/.1))
            p=registry.mechanism_profiles[profile]
            records=tuple(replace(r,value=q(effective_decay,'s','synaptic_decay_time_constant'),
                supported_domain=dict(r.supported_domain,source_conditions=source_conditions),
                locator='McLaughlin2021 Fig4 clamp endpoints; continuous voltage interpolation and ROI transfer are project choices') if r.parameter_id=='tau_decay' else r for r in p.records)
            records+=tuple(EvidenceValue(key,q(np.full(nn,v),'s','synaptic_decay_time_constant'),'project_approximation','MCLAUGHLIN2021_glycine_anchors','Unverified primary-source quotation: voltage-specific event decay endpoint; interpolation is project approximation',{'voltage_V':voltage,'source_conditions':source_conditions,'provenance_tag':'SOURCE_QUOTATION_PRIMARY_VERIFICATION_PENDING'},'Extracted experimental event kinetics; primary verification not established by asset identity') for key,v,voltage in [('decay_minus100',low,-.1),('decay_0',high,0.)])
            p=replace(p,records=records);registry.mechanism_profiles[profile]=p;registry.profiles[profile]=p
        # The measured event anchor is separate from the effective compound profile.
        if peak_evidence=='measurement':
            p=registry.mechanism_profiles[profile]
            records=tuple(replace(r,evidence_class='project_approximation',source_id=source_id,
                supported_domain=dict(r.supported_domain,source_conditions=source_conditions),
                locator='Unverified primary-source quotation: macaque spontaneous glycinergic event peak; not total maximal conductance',uncertainty='Extraction identity verified; primary experimental quotation verification pending') if r.parameter_id=='g_peak' else r for r in p.records)
            p=replace(p,records=records,source_ids=p.source_ids+(source_id,));registry.mechanism_profiles[profile]=p;registry.profiles[profile]=p
        pp=add_population(name,'AC',positions,profile);by[name]=pp
        components.append(ComponentSpec(name,'DoubleExponentialChemicalSynapse',{'cells':name},profile,{'receptor':receptor,'voltage_dependent_decay':receptor=='inhibitory','enabled':('ampa' not in ablations if receptor=='excitatory' else 'glycine' not in ablations)}))
        src=sparse.csr_matrix((np.ones(nn),(np.arange(nn),W.col)),shape=(nn,len(by[pre].cells.cell_id)))
        dst=sparse.csr_matrix((np.ones(nn),(W.row,np.arange(nn))),shape=(len(by[post].cells.cell_id),nn))
        links.extend([connection(pre,'event_rate',name,'event_rate',src,by,synapse='presynaptic_expected_vesicle_events'),connection(name,'conductance',post,'excitation' if receptor=='excitatory' else 'inhibition',dst,by,synapse='AMPA' if receptor=='excitatory' else 'glycine_type_specific')])
        if receptor=='inhibitory':
            backward=sparse.csr_matrix((np.ones(nn),(np.arange(nn),W.row)),shape=(nn,len(by[post].cells.cell_id)))
            links.append(connection(post,'voltage',name,'postsynaptic_voltage',backward,by,synapse='voltage_conditioned_glycine_decay'))
        chemical_audits[name]={'per_edge_states':nn,'event_peak_S':peak,'effective_overlap_multiplier':overlap.tolist(),'edge_lengths_mm':distances.tolist(),'effective_radius_mm':effective_radius,'radius_is_measured':False,'row_normalized_contact_multiplier':False,'rise_seconds':rise,'decay_seconds':decay,'times_are_measured_10_90_rise':False,
            'target_support_complete':coverage.tolist() if coverage is not None else None,
            'zero_row_classification':'no candidates in covered finite support vs boundary-unavailable support; neither is measured animal zero connectivity'}
    chemical('RBC_AII_AMPA','rod_BC','AII_AC','excitatory',3,.2e-9,.0002,.002)

    def gap(left,right,pairs,gain,label):
        pairs=np.asarray(pairs,int);count=len(pairs)
        distance=np.linalg.norm(contact_positions(by[left],'IPL')[pairs[:,0]]-contact_positions(by[right],'IPL')[pairs[:,1]],axis=1)
        if source_spatial:
            keep=distance<=hard_support_mm
            wiring_audits.append({'route':label,'frame':'IPL','hard_support_mm':.04,'candidate_edges':count,'omitted_outside_support':int((~keep).sum()),'boundary_policy':'finite_induced_graph'})
            pairs,distance=pairs[keep],distance[keep];count=len(pairs)
            if not count:return
        effective_radius=.03 if source_spatial else max(.03,float(distance.max())*1.05);overlap=np.exp(-(distance/effective_radius)**2)
        record=EvidenceValue(label,q(gain*overlap,'S','gap_junction_conductance'),'project_approximation',PROJECT_SOURCE,'Conservative reciprocal ohmic Cx36-like junction; shared gmax times symmetric overlap; effective conductance/footprints, not measured macaque values',{'route':label,'geometry':'IPL','gmax_S':gain,'edge_lengths_mm':distance.tolist(),'effective_overlap_radius_mm':effective_radius,'effective_radius_factor_relative_30um':effective_radius/.03,'radius_expanded_for_effective_resolution':effective_radius>.03,'no_row_normalization':True},'Uncalibrated geometry/nS; replaceable')
        gaps.append(GapJunction(left,tuple(by[left].cells.cell_id[i] for i in pairs[:,0]),right,tuple(by[right].cells.cell_id[i] for i in pairs[:,1]),record))
    gap('AII_AC','AII_AC',[(i,i+1) for i in range(aiin-1)],.2e-9,'AII_AII_homologous')
    for cid in ('midget_BC_ON','diffuse_BC_ON','S_ON_BC'):
        pairs=[(int(cKDTree(aiixy).query(p)[1]),i) for i,p in enumerate(contact_positions(by[cid],'IPL'))]
        gap('AII_AC',cid,pairs,.15e-9,'AII_'+cid+'_bidirectional')
    chemical('AII_FMB_glycine','AII_AC','midget_BC_OFF','inhibitory',3,.97e-9,.00008,.003,source_id='MCLAUGHLIN2021_glycine_anchors',peak_evidence='measurement')
    chemical('AII_DB1_glycine','AII_AC','diffuse_BC_OFF','inhibitory',3,.80e-9,.00011,.004,source_id='MCLAUGHLIN2021_glycine_anchors',peak_evidence='measurement')
    # Local/widefield AC residual physiology remains an explicitly named approximation.
    inner_rf_xy=(parasol_rf_seeds.positions.values if parasol_rf_seeds is not None else
        inner_rf_scaffold.positions.values if source_spatial else rbxy)
    ac_n=len(inner_rf_xy)
    ac_ipl_xy=(disp.apply(q(inner_rf_xy,'mm','functional_position','functional_visual_field'))['pedicle'].values
        if source_spatial else None)
    routes=[('midget_RGC_ON','midget_BC_ON','local_crossover_AC'),('midget_RGC_OFF','midget_BC_OFF','local_feedforward_AC'),('parasol_RGC_ON','diffuse_BC_ON','widefield_AC'),('parasol_RGC_OFF','diffuse_BC_OFF','widefield_AC'),('small_bistratified_S_ON','S_ON_BC','local_feedforward_AC')]
    rgc_layouts={};ac_layouts={};degree_issues=[]
    if typed_ipl:
        from retina_sim.ipl_geometry import typed_layouts
        rgc_layouts,ac_layouts,planned=typed_layouts(by,cones.cells.cell_id,xy,mapped['pedicle'].values,
            inner_rf_scaffold,output_domain,disp,ipl_configuration,instance,max_spatial_cells,
            parasol_rf_seeds=parasol_rf_seeds)
        ipl_audit.update(planned)
    ac_defs=[('local_feedforward_AC','diffuse_BC_ON'),('local_crossover_AC','diffuse_BC_OFF'),('widefield_AC','diffuse_BC_ON')]
    for cid,source in ac_defs:
        if typed_ipl:
            ac_ids,ac_rf,ac_points=ac_layouts[cid];ac_n=len(ac_ids)
        else:ac_ids=None;ac_rf=inner_rf_xy;ac_points=ac_ipl_xy
        source_profile=registry.mechanism_profiles[by[source].cells.parameter_profile_id[0]]
        upstream_initial=initial_release(source_profile)
        pid='effective_'+cid;register_profile(registry,pid,ac_n,{'kernel':([.5,.25,-.1],'dimensionless','causal_residual_kernel'),'sample_interval':(.00073 if cid=='widefield_AC' else .001,'s','history_sampling_interval'),'bias':(np.full(ac_n,-.3),'dimensionless','activation_bias'),'baseline':(np.full(ac_n,.3),'dimensionless','input_baseline'),'initial_history':(np.zeros((3,ac_n)),'dimensionless','causal_input_history')},reason='Finite causal AC residual; effective kernel and sampling, not fitted physiology')
        pop=add_population(cid,'AC',ac_rf,pid,anatomy=ac_points,cell_ids=ac_ids);by[cid]=pop;components.append(ComponentSpec(cid,'ResidualAC',{'cells':cid},pid,{'polarity_representation':'stored_signed'},cid))
        if typed_ipl:
            input_region='ON' if source.endswith('_ON') else 'OFF'
            if input_region not in ipl_configuration['strata'][cid]['input']:
                raise ContractError(f'IPL_STRATUM_MISMATCH: {source}->{cid} input {input_region}')
            ipl_audit['contact_rules'].append({'pre':source,'post':cid,'postsynaptic_input_region':input_region,
                'frame':'IPL','metric':'point_proxy','support_mm':hard_support_mm,
                'degree':16 if cid=='widefield_AC' else 4,'weights':'existing normalized activation drive retained'})
        W=local_matrix(by[source],pop,16 if cid=='widefield_AC' else 4)
        if source_spatial:
            baseline=np.asarray(W@upstream_initial).ravel()
            p=registry.mechanism_profiles[pid]
            records=tuple(replace(r,value=q(baseline,'dimensionless','input_baseline'),locator='Derived from actual upstream BC terminal initial release and actual sparse W; zero residual background prehistory') if r.parameter_id=='baseline' else r for r in p.records)
            p=replace(p,records=records);registry.mechanism_profiles[pid]=p;registry.profiles[pid]=p
        links.append(connection(source,'activation',cid,'drive',W,by))
    secondary_pairs=[]
    for j,p in enumerate(pxy):
        candidates=cKDTree(rxy).query_ball_point(p,.010)
        secondary_pairs.extend((i,j) for i in candidates)
    if secondary_gap_scale and secondary_pairs:
        pairs=np.array(secondary_pairs);distances=np.linalg.norm(rxy[pairs[:,0]]-pxy[pairs[:,1]],axis=1)
        record=EvidenceValue('optional_rod_cone_gap',q(secondary_gap_scale*.05e-9*np.exp(-distances**2/(2*.005**2)),'S','gap_junction_conductance'),'project_approximation',PROJECT_SOURCE,'Optional reciprocal rod/cone terminal conductance; not6directedges inferred from tracer pooling',{'frame':'OPL','radius_mm':.010,'sigma_mm':.005},'Uncalibrated project secondary route, separately ablatable')
        gaps.append(GapJunction('rod_membrane',tuple(rodmem.cells.cell_id[i] for i in pairs[:,0]),'physical_membrane',tuple(physical_mem.cells.cell_id[j] for j in pairs[:,1]),record,'OPL'))
    if 'gap' in ablations:gaps=[]
    for ordinal,(cid,source,inhib) in enumerate(routes):
        positions=(rgc_layouts[cid]['RF'] if typed_ipl else
            by[source].cells.rf_position.values if cid.startswith('midget') or cid.startswith('small') else inner_rf_xy)
        gn=len(positions);pid='effective_'+cid;fill=lambda v:np.full(gn,v)
        register_profile(registry,pid,gn,{'C':(fill(60e-12),'F','membrane_capacitance'),'gL':(fill(2e-9),'S','leak_conductance'),'EL':(fill(-.065),'V','leak_reversal'),'EE':(fill(0.),'V','excitatory_reversal'),'EI':(fill(-.080),'V','inhibitory_reversal'),'V0':(fill(-.065),'V','initial_membrane_voltage'),'alpha':(fill(150.),'Hz','spike_intensity_scale'),'mu':(fill(-.070),'V','spike_voltage_offset'),'beta':(fill(.005),'V','spike_voltage_scale'),'spike_history_kernel':([-.012,-.008,-.004],'V','strict_past_spike_history_kernel'),'initial_spike_history':(np.zeros((3,gn)),'dimensionless','spike_count_history'),'spike_interval':(.001,'s','spike_sampling_interval')},reason='History-conductance RGC effective profile; each phenotype separate, no animal fit')
        rgc_map=disp.apply(q(positions,'mm','functional_position','functional_visual_field'))
        rgc_ids=(tuple(cid+':'+hashlib.sha256(site.encode()).hexdigest()[:24] for site in rgc_layouts[cid]['site_ids']) if typed_ipl else None)
        pop=add_population(cid,'RGC',positions,pid,anatomy=rgc_map['soma'].values,cell_ids=rgc_ids);by[cid]=pop
        options={'rng_seed':seed+100+ordinal}
        if typed_ipl:
            options.update(ipl_geometry_profile_sha256=ipl_configuration['profile_sha256'],
                postsynaptic_regions=ipl_configuration['strata'][cid],
                frozen_anchor_records=ipl_audit['anchors'][cid]['records'])
        if cid.startswith('midget'):options['parent_bc_ids']=rgc_layouts[cid]['parents'] if typed_ipl else tuple(by[source].cells.cell_id)
        if typed_ipl and not cid.startswith('midget'):
            pop=replace(pop,dendritic_positions=q(rgc_layouts[cid]['P'],'mm','dendritic_position','IPL'))
            pops[cid]=pop;by[cid]=pop
            options.update(point_morphology_policy='approved frozen upstream P-chart anchor; legacy G visualization only',
                ipl_geometry_profile_sha256=ipl_configuration['profile_sha256'],
                frozen_anchor_records=ipl_audit['anchors'][cid]['records'],
                postsynaptic_regions=ipl_configuration['strata'][cid])
        if source_spatial and cid.startswith('midget'):
            from retina_sim.wiring import wool_midget_profile
            from scipy.optimize import brentq
            temporal_equivalent=np.array([brentq(lambda r:float(density.radial(np.array([r]),np.array([np.pi]))[0])-float(density(point[None,:])[0]),density.minimum,10.) for point in positions])
            options['temporal_equivalent_mm']=temporal_equivalent.tolist()
            private=wool_midget_profile(temporal_equivalent)['effective_contact_target']<=1
            dendrites=pop.dendritic_positions.values.copy()
            parent_index={cell_id:i for i,cell_id in enumerate(by[source].cells.cell_id)}
            for j in np.flatnonzero(private):dendrites[j]=contact_positions(by[source],'IPL')[parent_index[options['parent_bc_ids'][j]]]
            pop=replace(pop,dendritic_positions=q(dendrites,'mm','dendritic_position','IPL'))
            pops[cid]=pop;by[cid]=pop
            options['point_morphology_policy']='Private-line IPL contact at qualified parent BC terminal; anatomical soma remains separately mapped, not reused as synaptic point'
        components.append(ComponentSpec(cid,'HistoryConductanceRGC',{'cells':cid},pid,options,cid))
        initial_conductances={'excitatory':np.zeros(gn),'inhibitory':np.zeros(gn)}
        for receptor,src in [('excitatory',source),('inhibitory',inhib)]+([('excitatory','diffuse_BC_OFF')] if cid=='small_bistratified_S_ON' else []):
            name=cid+'_'+receptor+'_'+src;gainpid='effective_'+name
            register_profile(registry,gainpid,gn,{'gain':(fill(1e-9 if receptor=='excitatory' else .4e-9),'S','synaptic_conductance_scale')})
            by[name]=pop;components.append(ComponentSpec(name,'SynapticConductance',{'cells':cid},gainpid,{'receptor':receptor,'target_population_profile_id':pid}))
            # Eccentricity-dependent midget convergence and distinct diffuse fields.
            W=local_matrix(by[src],pop,1 if cid.startswith('midget') else 8,radius=.02+.06*np.linalg.norm(positions,axis=1))
            if typed_ipl:
                post_region=('OFF' if src=='diffuse_BC_OFF' or receptor=='inhibitory' and cid.startswith('small') else
                    'ON' if cid.endswith('ON') or cid.startswith('small') else 'OFF')
                allowed=('ON' if src.endswith('_ON') or src=='S_ON_BC' else 'OFF' if src.endswith('_OFF') else
                    ipl_configuration['strata'][src]['output'])
                allowed=[allowed] if isinstance(allowed,str) else allowed
                if post_region not in allowed:raise ContractError(f'IPL_STRATUM_MISMATCH: {src}->{cid} {post_region}')
                ipl_audit['contact_rules'].append({'pre':src,'post':cid,'postsynaptic_region':post_region,
                    'presynaptic_allowed_regions':allowed,'frame':'IPL','metric':'point_proxy','support_mm':hard_support_mm,
                    'degree':('source Wool per-target private/convergence' if cid.startswith('midget') and receptor=='excitatory' else 1 if cid.startswith('midget') else 8),'weights':'existing normalized activation W and total gain retained'})
                if not (cid.startswith('midget') and receptor=='excitatory'):
                    relay=research_contacts and cid=='small_bistratified_S_ON' and src=='S_ON_BC'
                    degree=np.diff(W.indptr);desired=1 if cid.startswith('midget') or relay else 8
                    if relay:
                        ipl_audit['contact_rules'][-1].update(contract_id='S_ON_effective_positive_input_legacy_selection_v1',
                            exact_target_enforcement=False,legacy_selection_budget=8,minimum_legal_selected_inputs=1,
                            actual_degree=degree.tolist(),calibration_status='effective_uncalibrated',
                            interpretation='effective relay partners; not ribbon count or anatomical BC convergence')
                        ipl_audit['contact_rules'][-1]['legacy_exact_degree_mismatch']=[{
                            'post_id':pop.cells.cell_id[j],'actual_degree':int(degree[j]),'legacy_target':8,
                            'is_owner':bool(domain.contains(positions[j:j+1],owner=True)[0]),
                            'finite_query_coverage':'source generation guard covers final 40um query',
                            'zero_input':bool(degree[j]==0),'status':'MANDATORY_INPUT_FAILURE' if degree[j]==0 else 'effective_uncalibrated'}
                            for j in np.flatnonzero(degree!=8)]
                    for j in np.flatnonzero(degree<desired):
                        row=wiring_audits[-1]['rows'][j]
                        degree_issues.append({'cause':'contact_capacity_conflict','pre':src,'post':cid,'post_id':pop.cells.cell_id[j],
                            'is_owner':bool(domain.contains(positions[j:j+1],owner=True)[0]),'target_degree':desired,'realized_degree':int(degree[j]),
                            'available_degree':row['available_candidate_degree'],'support_mm':hard_support_mm,'frame':'IPL',
                            'post_anchor_mm':ipl_points(pop,'postsynaptic')[j].tolist(),'nearest_source_distance_mm':row['nearest_available_source_distance_mm'],
                            'coverage':'generation guard planned by finite feedforward support; continuous coverage not independently certified',
                            'affected_scope':'owner' if domain.contains(positions[j:j+1],owner=True)[0] else 'diagnostic_output_only',
                            'evidence':'project_approximation','approval_state':'anchor and AC sampler approved; degree/support unchanged'})
            if cid.startswith('midget'):
                if source_spatial and receptor=='excitatory':
                    from retina_sim.wiring import wool_midget_profile
                    from scipy.optimize import brentq
                    # Named iso-density adapter to temporal-equivalent radius; no retinal-x substitution.
                    temporal_equivalent=np.asarray(options['temporal_equivalent_mm'])
                    convergence=wool_midget_profile(temporal_equivalent)
                    from retina_sim.spatial_profiles import bounded_matrix
                    radii=np.where(convergence['private_line_branch'],.001,convergence['center_radius_mm'])
                    pre_positions=contact_positions(by[src],'IPL');post_positions=contact_positions(pop,'IPL')
                    support=np.minimum(hard_support_mm,radii*np.sqrt(np.log(100.)))
                    W,rows=bounded_matrix(pre_positions,post_positions,convergence['effective_contact_target'],support_mm=support,gaussian_scale=radii)
                    # Same explicit cone -> BC -> RGC parent in the central private-line branch.
                    from retina_sim.wiring import enforce_midget_parent_matrix
                    W=enforce_midget_parent_matrix(W,by[src].cells.cell_id,options['parent_bc_ids'],convergence['effective_contact_target'],pre_positions=pre_positions,post_positions=post_positions,support_mm=support)
                    parent_index={cell_id:i for i,cell_id in enumerate(by[src].cells.cell_id)}
                    for j in np.flatnonzero(convergence['effective_contact_target']<=1):
                        distance=float(np.linalg.norm(pre_positions[parent_index[options['parent_bc_ids'][j]]]-post_positions[j]))
                        rows[j].update(actual_degree=1,missing_degree=0,maximum_distance_mm=distance,
                            normalization_input_sum=1.,edge_count=1,sum_weights_before_normalization=1.,sum_weights_after_normalization=1.,N_eff=1.,
                            parent_bc_id=options['parent_bc_ids'][j],selection_rule='qualified_private_parent')
                    wiring_audits.append({'route':cid,'receptor':receptor,'profile':'Wool_temporal_equivalent_iso_density_project_adapter','rows':rows,'source_fit':convergence['evidence'],'adapter':'iso-density inverse is project approximation'})
                elif not source_spatial:W=local_matrix(by[src],pop,min(12,len(by[src].cells.cell_id)),radius=.005+.04*np.linalg.norm(positions,axis=1))
            links.extend([connection(src,'activation',name,'activation',W,by),connection(name,'conductance',cid,'excitation' if receptor=='excitatory' else 'inhibition',sparse.eye(gn),by)])
            if source_spatial:
                upstream=registry.mechanism_profiles[by[src].cells.parameter_profile_id[0]]
                if 'BC' in src:activation0=initial_release(upstream)
                else:
                    bias=upstream.quantity('bias','dimensionless','activation_bias').values
                    kernel=upstream.quantity('kernel','dimensionless','causal_residual_kernel').values
                    history=upstream.quantity('initial_history','dimensionless','causal_input_history').values
                    activation0=np.logaddexp(0.,bias+kernel@history)
                gain=registry.mechanism_profiles[gainpid].quantity('gain','S','synaptic_conductance_scale').values
                initial_conductances[receptor]+=gain*np.asarray(W@activation0).ravel()
        if source_spatial:
            p=registry.mechanism_profiles[pid]
            gL=p.quantity('gL','S','leak_conductance').values
            gE,gI=initial_conductances['excitatory'],initial_conductances['inhibitory']
            V0=(gL*p.quantity('EL','V','leak_reversal').values+gE*p.quantity('EE','V','excitatory_reversal').values+gI*p.quantity('EI','V','inhibitory_reversal').values)/(gL+gE+gI)
            records=tuple(replace(r,value=q(V0,'V','initial_membrane_voltage'),locator='Analytical RGC membrane balance conditional on actual declared BC/AC initial conductances; zero seeded spike prehistory remains explicit') if r.parameter_id=='V0' else r for r in p.records)
            p=replace(p,records=records);registry.mechanism_profiles[pid]=p;registry.profiles[pid]=p
    geometry_entries={'corridor_bounds':(list(bounds_mm),'mm','retinal_domain_bounds'),'cone_density_scale':(scale,'dimensionless','effective_sampling_density_scale'),'rod_density_scale':(rod_scale,'dimensionless','effective_sampling_density_scale'),'magnification':(.223,'mm/deg','retinal_magnification'),'LM_ratio':(1.,'dimensionless','L_to_M_ratio'),'rod_free_radius':(.020,'mm','rod_free_region_radius'),'rod_ring_peak_radius':(4.,'mm','effective_rod_ring_peak_radius'),'rod_horizontal_peak':(120000.,'cells/mm^2','rod_density_landmark'),'rod_superior_peak':(177000.,'cells/mm^2','rod_density_landmark'),'rod_inferior_peak':(150000.,'cells/mm^2','rod_density_landmark'),'HC_pool_radius_intercept':(.08,'mm','effective_contact_radius'),'HC_pool_radius_eccentricity_slope':(.06,'dimensionless','effective_radius_eccentricity_gain'),'diffuse_BC_pool_radius_intercept':(.08,'mm','effective_contact_radius'),'diffuse_BC_radius_eccentricity_slope':(.04,'dimensionless','effective_radius_eccentricity_gain'),'midget_RGC_radius_intercept':(.005,'mm','effective_contact_radius'),'midget_RGC_radius_eccentricity_slope':(.04,'dimensionless','effective_radius_eccentricity_gain'),'physical_blend_start':(20.,'deg','source_route_blend_start'),'physical_blend_extent':(5.,'deg','source_route_blend_extent'),'rod_RBC_target_degree':(40.,'dimensionless','effective_rod_RBC_degree'),'effective_RBC_stride':(12.,'dimensionless','effective_resolution_population_stride'),'AII_center_sites':([.15,.3],'mm','effective_AII_inner_retinal_sites')}
    # Every spatial coefficient is classified rather than hidden in a fitted density claim.
    if source_spatial:
        geometry_entries.pop('effective_RBC_stride');geometry_entries.pop('AII_center_sites')
        geometry_entries.pop('physical_blend_start');geometry_entries.pop('physical_blend_extent')
        if hc_geometry_profile=='hc_joint_synthetic_v1':
            geometry_entries.pop('HC_pool_radius_intercept');geometry_entries.pop('HC_pool_radius_eccentricity_slope')
        geometry_entries.update({'owner_bounds':(list(domain.owner_bounds_mm),'mm','retinal_owner_bounds'),
            'halo_mm':(halo_mm,'mm','finite_computation_halo'),
            'hard_support':(hard_support_mm,'mm','finite_candidate_support'),
            'RBC_rod_density_ratio':(selected_profile['geometry']['RBC_to_rod_density_ratio'],'dimensionless','source_anchor_geometry_transfer_ratio'),
            'AII_density':(selected_profile['geometry']['AII_density_cells_per_mm2'],'cells/mm^2','source_anchor_geometry_transfer_density')})
    old=registry.mechanism_profiles[PROJECT_SOURCE]
    records=tuple(EvidenceValue(k,q(v,u,m),'project_approximation',PROJECT_SOURCE,'Declared effective geometry; source landmarks cited in source ledger, interpolation/count policy unmeasured',{'provenance_tag':'PROJECT_EFFECTIVE_TEMP','source_landmarks_are_not_a_digital_map':True},'Replaceable geometry/resolution; not animal cell count') for k,(v,u,m) in geometry_entries.items())
    gp=replace(old,records=records);registry.mechanism_profiles[PROJECT_SOURCE]=gp;registry.profiles[PROJECT_SOURCE]=gp
    target=ModelSpec.read_target(root/'configurations/full_target.json')
    omitted_layers={row.required_layer for row in target.components if pn==0 and row.component_id in
        ('physical_OS','physical_membrane','physical_terminal','physical_HC') and row.required_layer}
    # Native cone/H1/H2 already binds cone_outer_segment in central ROIs. Keep
    # every architecture target; missing optional Chen representations do not
    # permit silently removing an independently bound native target layer.
    required_layers=target.required_mechanisms
    spec=ModelSpec('MacaqueContinuousCenterPeriphery_effective_profile',tuple(pops.values()),tuple(components),tuple(links),tuple(externals),required_layers,scope='all_applicable_target_mechanisms_explicit_project_approximation',clock_policy={'kind':'explicit_event_clock','base_step_seconds':.0001,'sampled_output_hold':'zero_order_hold'},gap_junctions=tuple(gaps))
    mapping={row.component_id:('outer_native' if row.component_id=='cone_S_H2' else row.component_id) for row in target.components}
    for row in target.components:
        match=next((s for s in components if s.component_id==mapping[row.component_id]),None)
        if pn==0 and row.component_id in ('physical_OS','physical_membrane','physical_terminal','physical_HC'):
            mapping[row.component_id]='outside_source_domain_not_instantiated';continue
        expected=zero_key if zero_contact_route and mapping[row.component_id]=='outer_native' else row.mechanism
        if match is None or match.mechanism!=expected:raise ValueError('Declared full_target component has no matching actual runtime mechanism: '+row.component_id)
    requested_counts={}
    if typed_ipl:
        for cid in requested_owner_outputs:
            if cid not in rgc_layouts:raise ContractError('Unknown requested owner output '+cid)
            requested_counts[cid]=int(domain.contains(rgc_layouts[cid]['RF'],owner=True).sum())
        ipl_audit['requested_owner_counts']=requested_counts
        ipl_audit['degree_deficits']=degree_issues
        ipl_audit['all_dynamic_output_cohort_nodes_retained']=True
    try:
        if typed_ipl and any(count==0 for count in requested_counts.values()):
            raise ContractError('EMPTY_TARGET_OWNER: '+','.join(cid for cid,count in requested_counts.items() if count==0))
        if degree_issues:raise ContractError(f'IPL_CONTACT_CAPACITY_CONFLICT: {len(degree_issues)} rows fail declared mandatory degree inside40um; no nearest fallback')
        compiled=RetinaBuilder(registry,max_compile_bytes=selected_profile['budgets']['max_compile_bytes']).build(spec,RunMode.REFERENCE)
    except ContractError as error:
        # Preserve the actual attempted geometry/rows for a small build diagnosis;
        # do not convert missing mandatory inputs into hidden zero drives.
        error.partial_build_audit={'rod_inner_geometry':rod_inner_geometry,'chemical_synapses':chemical_audits,'IPL_geometry':ipl_audit,
            'connectivity_support':wiring_audits,'HC_geometry':spatial_audit.get('HC_geometry'),
            'populations':{pid:pop.record() for pid,pop in pops.items() if pid in
                ('rod_BC','AII_AC','midget_BC_ON','midget_BC_OFF','diffuse_BC_ON','diffuse_BC_OFF','S_ON_BC',
                 'local_feedforward_AC','local_crossover_AC','widefield_AC','midget_RGC_ON','midget_RGC_OFF',
                 'parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON')},
            'compiled_support_errors':list(getattr(error,'issues',(str(error),)))}
        raise
    gaze=gaze or GazeTrajectory(q([0,duration],'s','absolute_time'),q([[0,0],[.01,-.01]],'deg','prescribed_gaze_position','world_visual_field'),'linear','synthetic_prescribed_gaze')
    def make_channel(pop,unit,meaning,base,amp,dt=None):
        def evaluate(t):
            sample_t=(np.floor(t/dt+1e-9)*dt) if dt else t
            g=gaze.at(float(sample_t));pos=pop.cells.rf_position.values/.223+g.values
            subtype=np.array(pop.cells.subtype);phase=np.where(subtype=='S',1.2,np.where(subtype=='M',.6,0))
            return q(np.full(len(pos),base) if t<adaptation_seconds else base+amp*np.sin(3*pos[:,0]+40*(sample_t-adaptation_seconds)+phase),unit,meaning)
        info={'synthetic':True,'source':'synthetic_spatial_spectral_scene','given_gaze':True,'includes_background':True,'no_optical_or_light_unit_conversion_inferred':True,'background_context_hash':light_context.background_context_hash}
        if dt:info.update(grid_step_seconds=dt,interval_count=int(np.ceil(duration/dt)))
        return InputChannel(0,float(duration),evaluate,unit,meaning,'zero_order_hold' if dt else 'analytic','synthetic_fixture',info,tuple(sorted(set((np.arange(dt,duration,dt).tolist() if dt else [])+([adaptation_seconds] if adaptation_seconds>0 else [])))))
    channels={'native_light':make_channel(cones,'td','native_retinal_illuminance',light_context.native_background_td,.2*light_context.native_background_td),'rod_light':make_channel(rod,'R*/s','photoisomerization_rate',light_context.rod_background_rstar_s,.4*light_context.rod_background_rstar_s,.0001)}
    if physical is not None:channels['physical_cone_light']=make_channel(physical,'R*/s','photoisomerization_rate',light_context.cone_background_rstar_s,.2*light_context.cone_background_rstar_s,.0001)
    audit={'claim':'All target mechanisms have executable connected paths; named project approximation. No calibrated macaque physiology claim.','cone_density_source_mass':mass,'effective_density_scale':scale,'effective_cones':n,'effective_rods':len(rxy),'mosaic':mosaic.diagnostics,'cone_types':type_audit,'rod_geometry':{'class':'project_approximation','formula':'rho_peak(theta)*u*exp(1-u), u=(r-.02)/(4-.02), r>.02; sector peaks source landmarks; radial shape unmeasured','source_density_mass':rod_mass,'effective_density_scale':rod_scale,'mosaic':rmosaic.diagnostics,'AII_placement':'independent inner retinal sites including0.15/0.3mm, preserved without rods; effective locations not measured'},'representation_policy':'The named cone_route selects BC drive. The peripheral physical reference is instantiated only for source-supported LM cells; native and physical compartments are alternate representation layers, not additional animal cells.','displacement':mapped['diagnostics'],'target_component_mapping':mapping,'bounds_mm':list(bounds_mm),'geometry_parameter_evidence':records,'physical_route_blend':'C1 smoothstep 20..25deg, source support respected, explicitly project approximation','secondary_gap_scale':secondary_gap_scale,'secondary_candidate_contacts':len(secondary_pairs),'secondary_scope':'Implemented reciprocal physical terminal coupling; default primary-only strength0, no automatic pathway switch','ablations':sorted(ablations),'author_chemistry_initialization':'Derived constant-background source equilibrium, rod5R*/s and cone1000R*/s; inner states then declared warmup','chemical_synapses':chemical_audits,'first_synapse_contacts':cn,'rod_RBC_connection_lengths_mm':np.linalg.norm(rxy[contactW.col]-rbxy[contactW.row],axis=1).tolist(),'registry':registry,'spec':spec}
    audit['light_context']=light_context.manifest(adaptation_seconds)
    audit['runtime_profile']=registry.runtime_profile.manifest()
    audit['regional_support']=regional_binding_audit(root,compiled,selected_profile,light_context)
    audit['glycine_source_condition']={'source':'McLaughlin2021 Fig4/results',
        'canonical_url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC7920536/','species':'adult M. mulatta',
        'temperature_C':[31,33],'slice_thickness_um':300,'liquid_junction_correction_mV':-10,
        'filter_Hz':2000,'sample_Hz':10000,
        'source_region':'>8 mm nasal superior/inferior peripheral retinal slice','illumination':'low photopic; no calibrated transfer to current input',
        'FMB_peak_nS':.97,'DB1_peak_nS':.80,'FMB_Erev_mV':-76,'DB1_Erev_mV':-73,
        'FMB_decay_ms_at_minus100_0':[2.1,4.0],'DB1_decay_ms_at_minus100_0':[2.7,5.3],
        'FMB_decay_SEM_ms':[.2,.2],'DB1_decay_SEM_ms':[.2,.3],'cell_n':{'FMB':20,'DB1':7},
        'measured_rise_10_90_us':{'FMB':[123,245],'DB1':[206,278]},
        'tau_rise':'independent effective project coefficient, not the reported decay or measured 10..90 rise',
        'applied_policy':'McLaughlin_peripheral_slice_to_ROI_transfer_v1; explicit project approximation outside original region/condition',
        'voltage_interpolation':'modeled voltage interpolation of clamp endpoints is project approximation, not measured continuous kinetics',
        'primary_verification':'complete local source locator/material binding pending'}
    audit['execution_profile']=execution_profile
    audit['spatial_profile']=spatial_audit
    audit['HC_geometry_profile']=hc_geometry_profile
    audit['cone_route']=cone_route
    audit['physical_route_blend']=cone_route
    audit['physical_reference_applicability']={'included':bool(pn),'count':pn,'source_domain':'LM >20 deg only','omission_reason':None if pn else 'No source-supported peripheral LM cells; no foveal/S transplant',
        'omitted_optional_component_layers':sorted(omitted_layers),'omitted_required_layers':[],
        'architecture_target_layers':list(target.required_mechanisms)}
    audit['connectivity_support']=wiring_audits
    audit['IPL_geometry']=ipl_audit
    audit['population_owner_mask']={pid:(hc_runtime_owner.tolist() if source_spatial and pid=='HC' else
        aii_mosaic.owner.tolist() if source_spatial and pid=='AII_AC' else
        domain.contains(pop.cells.rf_position.values,owner=True).tolist()) for pid,pop in pops.items()}
    audit['resource_estimate']=RetinaBuilder(registry).estimate(spec)
    audit['rod_inner_geometry']=rod_inner_geometry
    audit['resolved_execution_profile']=selected_profile
    audit['density_integrals']={'generated_bounds_mm':list(domain.bounds_mm),'generated_area_mm2':(bounds_mm[1]-bounds_mm[0])*(bounds_mm[3]-bounds_mm[2]),
        'owner_bounds_mm':list(domain.owner_bounds_mm),'owner_area_mm2':(domain.owner_bounds_mm[1]-domain.owner_bounds_mm[0])*(domain.owner_bounds_mm[3]-domain.owner_bounds_mm[2]),
        'cone':mosaic.diagnostics,'rod':rmosaic.diagnostics,
        'HC':{subtype:m.diagnostics for subtype,m in hc_mosaics} if source_spatial else {'basis':'legacy_effective_sampling'}}
    if source_spatial:
        frame=spatial_audit['HC_frame_policy'];hc_bounds=frame['bounds_mm'];hc_owner=frame['owner_bounds_mm']
        audit['density_integrals'].update(HC_coordinate_frame='OPL',HC_generated_bounds_mm=hc_bounds,
            HC_generated_area_mm2=(hc_bounds[1]-hc_bounds[0])*(hc_bounds[3]-hc_bounds[2]),
            HC_owner_bounds_mm=hc_owner,HC_owner_area_mm2=(hc_owner[1]-hc_owner[0])*(hc_owner[3]-hc_owner[2]))
        audit['density_integrals'].update(rod_coordinate_frame='existing co-located physical OPL rod field',
            rod_generated_bounds_mm=list(rod_domain.bounds_mm),RBC_generated_bounds_mm=list(rbc_domain.bounds_mm),
            rod_density_scale_changed=False,RBC_density_ratio_changed=False)
        audit['rod_geometry']['AII_placement']='independent 2D density-anchor mosaic; no rod-count dependence, source-anchor regional transfer'
    contact_names={'individual_rod_RBC_contacts',*chemical_audits}
    representation_names={'peripheral_physical_cones','cone_compartments','cone_terminals','physical_HC'}
    audit['population_records']={pid:{'count':len(pop.cells.cell_id),
        'owner_count':sum(audit['population_owner_mask'][pid]),
        'owner_frame':'OPL mapped owner bounding box' if source_spatial and pid=='HC' else 'IPL physical owner patch union' if source_spatial and pid=='AII_AC' else 'functional_visual_field',
        'entity_kind':'contact' if pid in contact_names else 'representation_layer' if pid in representation_names else
            'cell' if pid in ('cones','rod','HC','rod_BC','AII_AC') else 'effective_unit',
        'count_basis':('source_density' if pid=='cones' else 'source_anchored_approx_density' if pid in ('rod','HC','rod_BC','AII_AC') else 'effective_sampling') if source_spatial else 'effective_sampling',
        'evidence_class':'project_approximation','representation':'source_derived_model_geometry; not measured individual cell locations'} for pid,pop in pops.items()}
    if hc_geometry_profile=='hc_joint_synthetic_v1':
        audit['population_records']['HC'].update(count_basis='contact participant view of unthinned source-constrained synthetic instance',
            generated_active_count=spatial_audit['HC_geometry']['runtime_view']['generated_active_count'],
            geometry_only_count=len(spatial_audit['HC_geometry']['runtime_view']['geometry_only_cell_ids']))
    audit['author_chemistry_initialization']='Declared LightContext background equilibria; inner states are declared finite-prerun states, not full-network steady state'
    from retina_sim.bc_gain_control import apply_bc_gain_control
    result=apply_bc_gain_control((compiled,channels,audit),root,bc_output_gain_control_enabled)
    from retina_sim.ac_inhibition import apply_ac_inhibition
    return apply_ac_inhibition(result,root,ac_inhibition_enabled)
