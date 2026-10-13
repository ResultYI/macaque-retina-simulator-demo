"""Complete missing existing S relay contacts on a frozen actual atlas graph.

Existing 40um support, degree8, Gaussian scale and typed strata are retained.
No rod field, RGC density, tissue sample or passive drive is invented.
"""
import numpy as np
import hashlib
from pathlib import Path
from scipy import sparse
from .contracts import Quantity,ContractError,canonical_hash
from .spatial_profiles import bounded_matrix


def complete_typed_contacts(record,atlas):
    result=dict(record,inner_tables=dict(record['inner_tables']),inner_contacts=dict(record['inner_contacts']))
    tables=result['inner_tables'];contacts=result['inner_contacts'];added=[]
    repaired=[]
    for pre,post,budget in (('diffuse_BC_ON','local_feedforward_AC',4),('diffuse_BC_OFF','local_crossover_AC',4),('diffuse_BC_ON','widefield_AC',16)):
        key=pre+'->'+post;old=contacts[key];upstream=contacts['cones->'+pre]['row_status']
        eligible=np.array([i for i,s in enumerate(upstream) if s!='BLOCKED_DIFFUSE_SEED_IS_S'],dtype=int)
        W,rows=bounded_matrix(np.asarray(old['pre_positions']['values'])[eligible],np.asarray(old['post_positions']['values']),
            budget,support_mm=.04,gaussian_scale=None)
        coo=W.tocoo();W=sparse.csr_matrix((coo.data,(coo.row,eligible[coo.col])),shape=(len(old['post_ids']),len(old['pre_ids'])))
        edges=[];statuses=[]
        for j,row in enumerate(rows):
            indices=W.getrow(j).indices
            status='READY_PROJECT_FINITE_POOL' if len(indices) else 'INCOMPLETE_NO_INPUT_WITHIN_40UM'
            if any(not upstream[i].startswith('READY') for i in indices):status='BLOCKED_UPSTREAM_TYPED_DRIVE'
            statuses.append(status)
            if status.startswith('READY'):
                edges.extend({'pre_id':old['pre_ids'][i],'post_id':old['post_ids'][j],'weight':float(W[j,i])} for i in indices)
            row.update(selection_budget=budget,exact_budget_enforcement=budget==1,
                source_rule='known S-seeded diffuse placeholder excluded before unchanged finite selection')
        contacts[key]=dict(old,row_status=statuses,rows=rows,edges=edges)
        repaired.append(key)
    name='small_bistratified_S_ON'
    if name in tables:
        target=tables[name];post=target['coordinates']['ipl_post_arbor'];rf=np.asarray(target['coordinates']['rf']['values'])
        for pre,region in (('local_feedforward_AC','OFF'),('diffuse_BC_OFF','OFF')):
            if pre=='local_feedforward_AC' and region not in atlas.inner_config['strata'][pre]['output']:
                raise ContractError('S inhibitory typed stratum differs from existing route')
            source=tables[pre];points=source['coordinates']['ipl_preterminal']
            matrix,rows=bounded_matrix(np.asarray(points['values']),np.asarray(post['values']),8,
                support_mm=.04,gaussian_scale=.02+.06*np.linalg.norm(rf,axis=1))
            upstream=contacts[('diffuse_BC_ON' if pre=='local_feedforward_AC' else 'cones')+'->'+pre]['row_status']
            edge=[];statuses=[]
            for j,row in enumerate(rows):
                indices=matrix.getrow(j).indices;status='READY_PROJECT_FINITE_POOL' if len(indices) else 'INCOMPLETE_NO_INPUT_WITHIN_40UM'
                if any(not upstream[i].startswith('READY') for i in indices):status='BLOCKED_UPSTREAM_TYPED_DRIVE'
                statuses.append(status)
                if status.startswith('READY'):
                    edge.extend({'pre_id':source['global_cell_ids'][i],'post_id':target['global_cell_ids'][j],
                        'weight':float(matrix[j,i])} for i in indices)
                row.update(selection_budget=8,minimum_legal_selected_inputs=1,
                    source_rule='existing full_target S route and weight law; finite deficits retained')
            key=pre+'->'+name
            contacts[key]={'pre_ids':source['global_cell_ids'],'post_ids':target['global_cell_ids'],'frame':'IPL',
                'row_status':statuses,'rows':rows,'edges':edge,'pre_positions':points,'post_positions':post,
                'evidence_class':'project_approximation','boundary_mode':'finite_supplied_candidates',
                'query_support_complete':'NOT_ESTABLISHED_BY_THIS_ADAPTER'}
            added.append(key)
    # Wide AC canonical grid has physical IPL anchors and no measured RF. The
    # inverse only supplies a named registered-coordinate view for owner selection.
    wide=tables.get('widefield_AC')
    if wide is not None and 'rf' not in wide['coordinates']:
        p=wide['coordinates']['ipl_post_arbor'];inverse=atlas.atlas.henle.inverse(
            Quantity(p['values'],'mm','pedicle_position','project_approximation','OPL'))
        tables['widefield_AC']=dict(wide,coordinates=dict(wide['coordinates'],rf={
            'values':inverse.values.tolist(),'unit':'mm','meaning':'functional_position','frame':'functional_visual_field',
            'evidence_tag':'IPL_point_OPL_colocation_Schein_inverse_registered_RF_project_v1'}))
    result['typed_completion']={'added_routes':added,'repaired_routes':repaired,'resampled_cells':False,
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'changed_existing_edges':'only known ineligible S-seeded diffuse placeholders removed before existing finite selection',
        'graph_origin':record.get('frozen_graph_origin'),
        'wide_AC_RF':'derived view from existing IPL point via approved Schein inverse; no measured soma/RF claim',
        'parasol_blocker':{'kind':'data_and_unapproved_source_transfer','reason':
            'Explicit ParasolRFSeedInput interface is available without a rod provider. New atlas has no approved supplied RF seed source; old rod-derived regression geometry is not adopted as physiological density or a default population.'}}
    result['typed_completion']['completion_hash']=canonical_hash({'metadata':result['typed_completion'],
        'contact_edges':{k:contacts[k]['edges'] for k in added+repaired}})
    return result
