"""One bounded central typed-IPL build/short step through parsed GUI settings."""
from pathlib import Path
import os,sys,json,time,argparse
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datetime import datetime,timezone
import tkinter as tk
import numpy as np
from retina_platform.config import RunConfig
from retina_platform.app import RetinaApp
from retina_sim.ipl_geometry import TYPED
from retina_sim.contracts import SolverConfig
from examples.full_target_network import full_target_network


def source_ipl_geometry(cfg,out):
    """Shared real-layout planner on unscaled source cones; no HC bypass in runtime."""
    import hashlib
    from dataclasses import replace
    from retina_sim.geometry import (PackerDensityField,McGregorDisplacement,RetinalDomain,
        SConeDensityField,assign_cone_types)
    from retina_sim.runtime_profiles import frozen_packer_table,execution_profile
    from retina_sim.regional_support import opl_roi_domain
    from retina_sim.spatial_profiles import generate_density,bounded_matrix
    from retina_sim.ipl_geometry import read_ipl_geometry,plan_feedforward_guard,typed_layouts,IPL_ROUTES
    from examples.full_network_fixture import population,q
    from examples.full_target_network import PROJECT_SOURCE
    density=PackerDensityField(frozen_packer_table(ROOT),central_policy='central_constant_cap',angular_policy='log_NT_cosine')
    disp=McGregorDisplacement(json.loads((ROOT/'sources/spatial/mcgregor2018_production_map_nodes.json').read_text(encoding='utf-8')),
        density,profile_name='published_rounded_anchor_composite')
    owner=cfg.owner_bounds();h=cfg.halo_mm;core=(owner[0]-h,owner[1]+h,owner[2]-h,owner[3]+h)
    output=RetinalDomain(core,owner,lambda xy:np.ones(len(xy),bool),'project','frozen_RF_output_cohort')
    opl,_=opl_roi_domain(disp,output,cfg.seed);bounds,guard=plan_feedforward_guard(disp,core,opl.bounds_mm)
    domain=RetinalDomain(bounds,owner,output.mask,'ISETMacaque_PackerConeRc','source_limited_peripheral_patch')
    mosaic=generate_density(density,domain,cfg.seed,cfg.max_spatial_cells)
    sdata=json.loads((ROOT/'sources/spatial/demonasterio1985_Scone_reference.json').read_text(encoding='utf-8'))
    sfield=SConeDensityField(sdata,.223,magnification_source_id=PROJECT_SOURCE,central_policy='assay_square_PCHIP')
    types,type_audit=assign_cone_types(mosaic,density,sfield,domain,1.,PROJECT_SOURCE,cfg.seed,.5)
    xy=mosaic.positions.values;mapped=disp.apply(q(xy,'mm','functional_position','functional_visual_field'))
    instance=hashlib.sha256(json.dumps({'bounds':bounds,'seed':cfg.seed,'xy':xy.tolist()},sort_keys=True).encode()).hexdigest()[:20]
    cone_ids=tuple(f'cone_instance:{instance}:{cell_id}' for cell_id in mosaic.cell_ids)
    lm=np.flatnonzero(types!='S');ss=np.flatnonzero(types=='S');db=lm[::4];bcs={}
    for cid,idx in (('midget_BC_ON',lm),('midget_BC_OFF',lm),('diffuse_BC_ON',db),('diffuse_BC_OFF',db),('S_ON_BC',ss)):
        pop=population(cid,'BC',(cid,)*len(idx),xy[idx],('center',)*len(idx),'geometry_only','Macaque_effective_project_profile',mapped['pedicle'].values[idx])
        bcs[cid]=replace(pop,cells=replace(pop.cells,cell_id=tuple(cid+':'+cone_ids[i] for i in idx)))
    ratio=execution_profile(ROOT,cfg.execution_profile)['geometry']['RBC_to_rod_density_ratio']
    def rbc_field(p):
        r=np.linalg.norm(p,axis=1);theta=np.arctan2(p[:,1],p[:,0]);u=np.maximum(0,(r-.020)/(4.-.020))
        peak=120000+np.where(np.sin(theta)>=0,57000,30000)*np.abs(np.sin(theta))
        return ratio*np.where(r>.020,peak*u*np.exp(1-u),0.)
    scaffold=generate_density(rbc_field,output,cfg.seed+2,cfg.max_spatial_cells)
    conf=read_ipl_geometry(ROOT,cfg.ipl_geometry_profile)
    rgcs,acs,audit=typed_layouts(bcs,cone_ids,xy,mapped['pedicle'].values,scaffold,output,disp,conf,instance,cfg.max_spatial_cells)
    counts={cid:int(domain.contains(layout['RF'],owner=True).sum()) for cid,layout in rgcs.items()}
    assert counts[cfg.requested_owner_output]>0,'Requested geometry owner must be nonempty'
    rows=[];deficits=[];legacy_mismatch=[]
    for cid,exc,inh in IPL_ROUTES:
        for pre in [exc,inh]+(['diffuse_BC_OFF'] if cid=='small_bistratified_S_ON' else []):
            points=bcs[pre].terminal_positions.values if pre in bcs else acs[pre][2]
            degree=1 if cid.startswith('midget') else 8
            W,details=bounded_matrix(points,rgcs[cid]['P'],degree,support_mm=.04)
            relay=cfg.ipl_geometry_profile=='retina_contact_geometry_effective_v1' and pre=='S_ON_BC' and cid=='small_bistratified_S_ON'
            for row in details:
                j=row['post_index'];row['is_owner']=bool(domain.contains(rgcs[cid]['RF'][j:j+1],owner=True)[0])
                if row['missing_degree'] and relay:legacy_mismatch.append(dict(row,pre=pre,post=cid,status='MANDATORY_INPUT_FAILURE' if row['actual_degree']==0 else 'effective_uncalibrated'))
                if row['actual_degree']<(1 if relay else degree):deficits.append(dict(row,pre=pre,post=cid))
            rows.append({'pre':pre,'post':cid,'rows':details,'zero_rows':sum(r['actual_degree']==0 for r in details),
                'legacy_exact_degree_mismatch_rows':sum(r['missing_degree']>0 for r in details),
                'unfilled_required_degree_rows':sum(r['actual_degree']<(1 if relay else degree) for r in details),
                'owner_degree_deficits':sum(r['actual_degree']<(1 if relay else degree) and r['is_owner'] for r in details),
                'finite_query_coverage':'planned full source guard and correct typed stratum; no radius enlargement',
                'effective_relay_minimum':1 if relay else None,'selection_budget':degree})
    # Central private-line point check. A peripheral diagnostic must use the
    # builder's Wool branch/legacy non-private midget point separately.
    assert all(np.max(np.linalg.norm(rgcs[cid]['RF'],axis=1))<.5 for cid in ('midget_RGC_ON','midget_RGC_OFF'))
    audit.update(generation_plan=guard,requested_owner_counts={cfg.requested_owner_output:counts[cfg.requested_owner_output]},
        owner_counts=counts,cone_density_scale=1.,cone_count=len(cone_ids),cone_types=type_audit,
        RGC_geometry={cid:{'RF_xy_mm':layout['RF'].tolist(),'IPL_dendrite_xy_mm':layout['P'].tolist(),
            'Soma_visualization_xy_mm':disp.apply(q(layout['RF'],'mm','functional_position','functional_visual_field'))['soma'].values.tolist()}
            for cid,layout in rgcs.items()},
        source_cone_mosaic=mosaic.diagnostics,contact_rows=rows,degree_deficits=deficits,legacy_exact_degree_mismatch=legacy_mismatch,
        full_native_assembly='NOT_RUN; geometry diagnostic does not replace failed HC contacts or their drives',
        recurrence='NOT_ESTABLISHED; no new boundary operator')
    (out/'ipl_geometry.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    return {'engineering_build':'NOT_RUN_GEOMETRY_ONLY','geometry_planner':'PASS','owner_counts':counts,
        'cone_count':len(cone_ids),'degree_deficits_count':len(deficits),
        'RGC_route_rows':[{k:v for k,v in row.items() if k!='rows'} for row in rows]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--roi',choices=('central','peripheral'),default='central')
    parser.add_argument('--geometry-only',action='store_true')
    parser.add_argument('--hc-max-depth',type=int,default=20)
    parser.add_argument('--hc-max-partitions',type=int,default=8192)
    parser.add_argument('--hc-quadrature-order',type=int,default=12)
    args=parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    out=ROOT/'reports/central_build_acceptance'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    geometry=out/'central_geometry_fixture.json'
    geometry.write_text(json.dumps({'profile_id':'hc_joint_synthetic_v1','arbor_profile_id':'central_OPL_equal_area_W2000',
        'generator':{'sweeps':2,'max_depth':args.hc_max_depth,'max_partitions':args.hc_max_partitions,
            'quadrature_order':args.hc_quadrature_order,'integration_relative_tolerance':.0001,
            'max_quadrature_points':8000000}},indent=2)+'\n',encoding='utf-8')
    requested=RunConfig(media_path=str(ROOT/'reports/hc_acceptance/20261011T100207621130Z/engineering_uniform_rgb.png'),
        duration_seconds=.0002,warmup_seconds=0.,window_seconds=.0002,sample_seconds=.0001,
        owner_x_mm=.35,owner_y_mm=0.,owner_width_mm=.02,owner_height_mm=.02,halo_mm=.04,
        roi_preset='central_OPL_support',max_spatial_cells=20000,
        hc_geometry_profile='hc_joint_synthetic_v1',hc_arbor_profile='central_OPL_equal_area_W2000',
        hc_geometry_path=str(geometry),ipl_geometry_profile=TYPED,requested_owner_output='midget_RGC_ON')
    if args.roi=='peripheral':
        requested.owner_x_mm=4.6;requested.roi_preset='peripheral_anchor'
        requested.hc_geometry_profile='source_patch_v1';requested.hc_arbor_profile='explicit_footprints';requested.hc_geometry_path=''
    # Exercise the real form parser; no display, synthetic Tk replacements or
    # changes to the GUI's selected values are needed.
    app=RetinaApp.__new__(RetinaApp);app.config=requested;app.media=None
    app.regional_support=json.loads((ROOT/'configurations/regional_support_v1.json').read_text(encoding='utf-8'))
    app.roi_labels={key:value['label'] for key,value in app.regional_support['roi_presets'].items()}
    tcl=tk.Tcl();app.params={name:tk.StringVar(master=tcl) for name in
        [row[0] for row in RetinaApp.PARAMS]+['media_path','gaze_csv','hc_geometry_path']}
    class Label:
        def configure(self,**kw):pass
    app.gaze_label=Label();app._set_config_vars(requested);cfg=app._read_config()
    (out/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2)+'\n',encoding='utf-8')
    start=time.monotonic()
    result={'configuration_path':str(out/'config.json'),'profile':TYPED,'ROI':args.roi,'owner_rectangle_mm':list(cfg.owner_bounds()),
        'baseline_comparison':'reports/central_build_acceptance/20261011T115108685443Z/acceptance.json',
        'numerical_HC_fixture':{'sweeps':2,'max_depth':args.hc_max_depth,'max_partitions':args.hc_max_partitions,
            'quadrature_order':args.hc_quadrature_order,'integration_relative_tolerance':.0001,
            'max_quadrature_points':8000000,'mixing_certified':False,'applied':args.roi=='central' and not args.geometry_only,
            'change_scope':'declared sampler numerical budget only; density/arbor/contact degree unchanged'},
        'time_integration':'NOT_RUN','physiological_validation':'NOT_RUN',
        'new_boundary_operator':'NOT_ENABLED','recurrent_closure':'NOT_ESTABLISHED'}
    if args.geometry_only:
        if args.roi!='central':raise ValueError('This source-only diagnostic is declared for the central private-line patch')
        result.update(source_ipl_geometry(cfg,out));result['wall_seconds']=time.monotonic()-start
        result['full_native_build_attempts']=[{'directory':str(ROOT/'reports/central_build_acceptance'/stamp),
            'error':json.loads((ROOT/'reports/central_build_acceptance'/stamp/'acceptance.json').read_text(encoding='utf-8'))['error']}
            for stamp in ('20261011T145038434526Z','20261011T145916294574Z')]
        (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        (out.parent/'latest.json').write_text(json.dumps({'acceptance_directory':str(out),
            'scope':'source geometry only; full native build blocked','full_native_build_attempts':result['full_native_build_attempts']},indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(result,output=str(out)),ensure_ascii=False));return
    try:
        compiled,channels,audit=full_target_network(ROOT,duration=.0002,adaptation_seconds=0.,seed=cfg.seed,
            light_context=cfg.light_context(),runtime_profile=cfg.parameter_profile,execution_profile=cfg.execution_profile,
            owner_bounds_mm=cfg.owner_bounds(),halo_mm=cfg.halo_mm,max_spatial_cells=cfg.max_spatial_cells,
            hc_geometry_profile=cfg.hc_geometry_profile,hc_geometry_path=cfg.hc_geometry_path,hc_arbor_profile=cfg.hc_arbor_profile,
            ipl_geometry_profile=cfg.ipl_geometry_profile,requested_owner_outputs=(cfg.requested_owner_output,))
    except Exception as error:
        audit=getattr(error,'partial_build_audit',{})
        result.update(engineering_build='BLOCKED',error=str(error),phase=audit.get('phase','IPL_or_compile'),
            requested_owner_counts=audit.get('IPL_geometry',{}).get('requested_owner_counts','NOT_CONSTRUCTED'),
            degree_deficits=audit.get('IPL_geometry',{}).get('degree_deficits',[]))
        if 'populations' in audit:
            owner=np.asarray(cfg.owner_bounds())
            counts={}
            for cid,pop in audit['populations'].items():
                if 'RGC' not in cid and cid!='small_bistratified_S_ON':continue
                rf=np.asarray(pop['receptive_field_positions']['values'])
                mask=(rf[:,0]>=owner[0])&(rf[:,0]<owner[1])&(rf[:,1]>=owner[2])&(rf[:,1]<owner[3])
                counts[cid]=int(mask.sum())
            result['owner_counts']=counts
            result['RGC_route_rows']=[{'pre':row['pre'],'post':row['post'],
                'zero_rows':sum(r['actual_degree']==0 for r in row['rows']),
                'unfilled_required_degree_rows':sum(r['missing_degree']>0 for r in row['rows'])}
                for row in audit['connectivity_support'] if row.get('post') in counts]
        (out/'partial_build_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    else:
        rgcs=('midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON')
        result.update(engineering_build='PASS',requested_owner_counts=audit['IPL_geometry']['requested_owner_counts'],
            owner_counts={cid:int(np.sum(audit['population_owner_mask'][cid])) for cid in rgcs},
            all_dynamic_counts={cid:len(compiled.populations[cid].cells.cell_id) for cid in rgcs},
            components=len(compiled.components),layers=len(compiled.spec.required_mechanisms),
            routes=[{'pre':row['pre'],'post':row['post'],'zero_rows':sum(r['actual_degree']==0 for r in row['rows']),
                'unfilled_required_degree_rows':sum(r['missing_degree']>0 for r in row['rows'])}
                for row in audit['connectivity_support'] if row.get('post') in rgcs])
        (out/'ipl_geometry.json').write_text(json.dumps(audit['IPL_geometry'],indent=2)+'\n',encoding='utf-8')
        (out/'connectivity_support.json').write_text(json.dumps(audit['connectivity_support'],indent=2)+'\n',encoding='utf-8')
        (out/'population_owner_mask.json').write_text(json.dumps(audit['population_owner_mask'],indent=2)+'\n',encoding='utf-8')
        runner=compiled.runner();state=runner.initialize(SolverConfig(**audit['resolved_execution_profile']['solver']))
        state,trace=runner.advance(state,channels,.0002)
        assert all(np.all(np.isfinite(a)) for a in state.ordered_state_arrays)
        result.update(time_integration='PASS_0.2ms',state_scalars=sum(a.size for a in state.ordered_state_arrays),
            assembly_hash=compiled.assembly_hash,model_hash=runner.model.model_hash)
        (out/'final_checkpoint.json').write_text(runner.checkpoint(state),encoding='utf-8')
    result['wall_seconds']=time.monotonic()-start
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (out.parent/'latest.json').write_text(json.dumps({'acceptance_directory':str(out)},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(result,degree_deficits_count=len(result.get('degree_deficits',[])),degree_deficits=None,output=str(out)),ensure_ascii=False))


if __name__=='__main__':main()
