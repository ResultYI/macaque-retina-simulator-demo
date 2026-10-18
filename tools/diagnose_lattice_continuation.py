"""Read-only trend and fresh residual checks for a saved lattice segment."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_sim.lattice_relaxation import SourceSpringField,project_ellipse,ellipse_distance,quality_state,quality_widths,width_condition

def diagnose(directory):
    g=json.loads((directory/'generation.json').read_bytes());trace=g['trace']
    windows=[]
    for start in range(0,len(trace),100):
        rows=trace[start:start+100]
        stats={}
        for key in ('q_0_8','actual_max_movement_um','actual_p30_movement_um'):
            values=np.array([row[key] for row in rows])
            stats[key]={'first':float(values[0]),'last':float(values[-1]),'min':float(values.min()),
                'max':float(values.max()),'median':float(np.median(values)),
                'slope_per_iteration':float(np.polyfit(np.arange(len(values)),values,1)[0]) if len(values)>1 else 0.}
        windows.append({'first_iteration':rows[0]['iteration'],'last_iteration':rows[-1]['iteration'],
            'statistics':stats,'backtracks':sum(row['backtracks'] for row in rows),
            'topology_changes':sum(row['topology_changed'] for row in rows),
            'hull_events':sum(row['legal_hull_orientation_events'] for row in rows)})
    v=loadmat(ROOT/'data/atlas/isetbio_macaque_cones_coveragefix_v1_20261015/initialization.mat',squeeze_me=True)
    field=SourceSpringField(v['lookup_xy_um'],v['lookup_spacing_um'],v['radius_um'])
    checks={}
    with np.load(directory/'coordinates.npz',allow_pickle=False) as z:
        for label,key in [('selected','selected_raw_um'),('final','final_raw_um')]:
            x=z[key];tri,springs=field.topology(x);q=quality_state(x,tri)
            force,desired,length=field.force(x,springs);trial,_=project_ellipse(x+.2*force,field.radius)
            movement=np.linalg.norm(trial-x,axis=1);at=int(movement.argmax())
            interior=ellipse_distance(x,field.radius)<-field.border_tolerance
            widths=[quality_widths(row) for row in trace]
            # The selected state is evaluated with history only through its own iteration.
            if label=='selected':widths=widths[:g['selected_iteration']]
            qc=q['q_0_8']>g['thresholds']['quality'];wc=width_condition(widths)
            checks[label]={'quality':q,'saved_quality_matches':q==g[label+'_quality'],
                'quality_threshold_pass':qc,'source_width_condition':wc,'quality_plus_width_pass':qc and wc,
                'fresh_nominal_step_max_residual_um':float(movement.max()),
                'movement_threshold_um':g['thresholds']['d_tolerance_um'],
                'fresh_maximum_movement_pass':bool(movement.max()<g['thresholds']['d_tolerance_um']),
                'fresh_interior_p30_displacement_um':float(np.percentile(movement[interior],30,method='hazen')),
                'max_residual_row':at,'max_residual_raw_xy_um':x[at].tolist(),
                'max_residual_implicit_boundary_distance_um':float(ellipse_distance(x[[at]],field.radius)[0]),
                'fresh_interior_max_residual_um':float(movement[interior].max()),
                'point_count':len(x),'nearest_spring_um':float(length.min()),
                'ellipse_max_excess_um':float(ellipse_distance(x,field.radius).max())}
    topology=[row['topology_diagnostic'] for row in trace if 'topology_diagnostic' in row]
    topo_summary=None
    if topology:
        cosines=[row['max_mover_previous_direction_cosine'] for row in topology
            if row.get('max_mover_previous_direction_cosine') is not None]
        topo_summary={'diagnosed_steps':len(topology),'immediate_edge_reversals':sum(row['edges_immediately_reversed'] for row in topology),
            'removed_repulsive_edges':sum(row['removed_edges_with_repulsion'] for row in topology),
            'max_mover_direction_cosine_min':min(cosines) if cosines else None,
            'max_mover_direction_cosine_median':float(np.median(cosines)) if cosines else None,
            'max_mover_direction_reversed_steps':sum(value<0 for value in cosines)}
    return {'directory':directory.relative_to(ROOT).as_posix(),'stop_reason':g['stop_reason'],
        'accepted_iterations':g['accepted_iterations'],'selected_iteration':g['selected_iteration'],
        'criterion_definitions':{'quality':'MATLAB-style Hazen 0.8th percentile of q=2r/R >0.84 AND pinned histogram width-history condition',
            'movement':'Maximum over ALL points of projected displacement x -> project(x+0.2*F(current Delaunay)) <0.000215 um; source p30 shown only as diagnostic',
            'independence':'quality stopping is an engineering early-stop; neither condition is biological validation'},
        'windows':windows,'topology_diagnostic_summary':topo_summary,
        'saved_state_checks':checks,'physiological_validation':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output');args=p.parse_args()
    directory=(ROOT/args.directory).resolve();assert directory.is_relative_to(ROOT)
    result=diagnose(directory);s=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:
        output=(ROOT/args.output).resolve();assert output.is_relative_to(ROOT);output.write_text(s,encoding='utf-8')
    print(s)
