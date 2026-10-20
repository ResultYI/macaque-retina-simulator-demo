"""Attach existing physical rod/RBC/AII/BC terminal factories to one atlas graph."""
from dataclasses import replace
import hashlib,json
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from .atlas_runtime import q,_population,_link,bind_runtime_profile,_population_owner_mask
from .contracts import RunMode,ContractError
from .assembly import ComponentSpec,ExternalInput,GapJunction,RetinaBuilder
from .mechanism_contracts import EvidenceValue
from .provenance import SourceRecord
from .rod_pathway import rod_source_values
from .parameter_contracts import CHEN_PARAMETER_SPECS
from .initialization import initial_release,initial_event_rate
from .spatial_profiles import bounded_matrix,finite_support_coverage
from examples.full_target_network import register_profile,PROJECT_SOURCE


def attach_primary_rod(root,compiled,channels,audit,record,light,*,spatial_asset=None,rod_reference_policy=None):
    from .circuit_geometry import csr_record,plain
    recipe={'pools':{},'contact_populations':{}}
    registry=audit['registry'];pops=dict(compiled.populations);specs=list(compiled.spec.components);links=list(compiled.spec.links);gaps=[]
    sid='MCLAUGHLIN2021_glycine_anchors'
    if sid not in registry.sources:
        registry.register_source(SourceRecord(sid,'Macaque bipolar glycinergic event quotation',('McLaughlin et al.',),2021,
            'https://pmc.ncbi.nlm.nih.gov/articles/PMC7920536/','data_table',None,'sources/mclaughlin2021_verified_anchors.json',
            'Existing event peak/reversal quotation; no whole circuit calibration',registry.runtime_profile.source_identity(sid),None,
            'Existing source quotation','identity_verified','Primary quotation verification pending; peripheral slice transfer stays project approximation','2026-10-12',{}))
    builder=RetinaBuilder(registry,max_compile_bytes=getattr(spatial_asset,'max_compile_bytes',536870912))
    bind=lambda name,n,derived=None:bind_runtime_profile(registry,name,n,derived)
    # Restore the original physical BC terminal; retain all original coefficients.
    for i,spec in enumerate(specs):
        if spec.mechanism in ('ResidualBC_ON','ResidualBC_OFF'):
            options=dict(spec.options,polarity='ON' if spec.mechanism.endswith('ON') else 'OFF')
            options['scope']='Original physical BC terminal restored; actual AII glycine and reciprocal gaps bound'
            specs[i]=replace(spec,mechanism='HistoryBipolarTerminal',options=options)
    def get_record(name,pid,kind,runtime_name=None,prefix=None,opl_pre=None,opl_post=None):
        data=dict(record['rod_inner']['tables'][name]);data['population']=runtime_name or name
        if prefix:data=dict(data,global_cell_ids=[prefix+g for g in data['global_cell_ids']])
        pop=_population(data,kind,pid,'Macaque_effective_project_profile',opl_pre=opl_pre,opl_post=opl_post)
        pops[pop.population_id]=pop;return pop
    xy=np.asarray(record['rod_inner']['tables']['rods']['coordinates']['rf']['values'])
    n=len(xy);rp='Chen2024_primateRod_source';values=rod_source_values(root)
    entries={k:(v,'Chen_scaled_concentration' if k in ('gdark','cdark','kGC') else 'pA/Chen_scaled_concentration^3' if k=='k' else CHEN_PARAMETER_SPECS['CHEN2024_'+k][0],
        'source_coefficient_'+k) for k,v in values.items()}
    register_profile(registry,rp,n,entries,source_id='CHEN2024_initPhotoreceptorParams_m',evidence='published_fit',reason='Unchanged pinned primateRod branch; geometry updated by approved population package')
    rod=get_record('rods',rp,'rod','rod',opl_pre=xy)
    rod_spec=ComponentSpec('rod','Chen2024PrimateRodCurrent',{'cells':'rod'},rp,
        {'initial_background_rate':q(np.full(n,light.rod_background_rstar_s),'R*/s','photoisomerization_rate')},'rod')
    specs.append(rod_spec);runtime=builder.registry.instantiate(rod_spec,pops,registry,RunMode.REFERENCE)
    current0=runtime.observe('current',0,tuple(v.values for v in runtime.initial()),{}).values*1e-12
    defs=registry.runtime_profile.data['profiles']['effective_rod_membrane'];gL=defs['gL']['value'];V0=defs['V0']['value'];EL=defs['EL']['value']
    bind('effective_rod_membrane',n,{'current_anchor':current0+gL*(V0-EL)})
    membrane_pid='effective_rod_membrane'
    if rod_reference_policy is not None:
        from .mechanism_contracts import register_mechanism_profile
        policy=rod_reference_policy
        if (policy.get('schema_version')!='fixed-rod-dark-reference-1.0' or policy.get('current_anchor_A')!=0. or
                policy.get('dark_voltage_V')!=-.04 or policy.get('physiological_calibration') is not False):
            raise ContractError('Explicit approved fixed rod dark reference required')
        dark_current=-runtime.native_model.p['k']*runtime.native_model.p['gdark']**runtime.native_model.p['n']*1e-12
        expected=dark_current-gL*(EL-policy['dark_voltage_V'])
        if abs(expected-policy['current_anchor_A'])>1e-24:raise ContractError('Fixed dark anchor differs from retained dark membrane group')
        source_id='ROD_FIXED_DARK_REFERENCE_V1';binding=policy['binding']
        registry.register_source(SourceRecord(source_id,'Approved fixed rod dark reference and actual-background initial voltage',
            ('Project explicit passive reference; not animal voltage fit',),2026,'https://doi.org/10.7554/eLife.93795',
            'data_table',None,binding['path'],'Pro E16; fixed dark V=-40mV with retained EL/gL and Chen dark current',binding['sha256'],None,
            'Project engineering reference contract','identity_verified','Approved reference change, not rod membrane/release calibration','2026-10-20',{}))
        old=registry.mechanism_profiles[membrane_pid];rows=[]
        for r in old.records:
            if r.parameter_id in ('current_anchor','V0'):
                value=np.full(n,policy['current_anchor_A']) if r.parameter_id=='current_anchor' else EL-(current0-policy['current_anchor_A'])/gL
                r=replace(r,value=replace(r.value,values=value),source_id=source_id,evidence_class='project_approximation',
                    locator=binding['path']+'; '+r.parameter_id,uncertainty='Fixed engineering dark reference; V0 derived at actual background, not independently fitted')
            rows.append(r)
        membrane_pid='effective_rod_membrane_fixed_dark_reference_v1'
        register_mechanism_profile(registry,replace(old,profile_id=membrane_pid,records=tuple(rows),
            source_ids=tuple(sorted(set(old.source_ids+(source_id,)))),approval_reference='User approved fixed dark reference, 2026-10-20',
            conditions=dict(old.conditions,rod_fixed_dark_reference=dict(binding=binding,actual_background_Rstar_s=light.rod_background_rstar_s,
                current_anchor_A=policy['current_anchor_A'],dark_voltage_V=policy['dark_voltage_V'],initial_state_is_actual_background=True))))
        audit['rod_reference']=dict(policy='fixed_dark',binding=binding,current_anchor_A=policy['current_anchor_A'],
            actual_background_Rstar_s=light.rod_background_rstar_s,derived_initial_voltage_V=(EL-(current0-policy['current_anchor_A'])/gL).tolist(),
            physiological_calibration=False,active_membrane_added=False)
        audit['runtime_profile']=dict(audit['runtime_profile'],rod_reference=audit['rod_reference'])
    mem=get_record('rods',membrane_pid,'rod','rod_membrane','rod_membrane:',opl_pre=xy)
    specs.append(ComponentSpec('rod_membrane','ReceptorConductanceMembrane',{'cells':'rod_membrane'},membrane_pid,{'cell_kind':'rod_membrane'}))
    bind('rod_current_adapter',n);specs.append(ComponentSpec('rod_current_SI','ExplicitPhysicalUnitConversion',{'cells':'rod_membrane'},'rod_current_adapter',
        {'source_unit':'pA','target_unit':'A','meaning':'absolute_outer_segment_current','target_population_profile_id':membrane_pid}))
    links.extend((_link('rod','current','rod_current_SI','input',rod,mem,sparse.eye(n,format='csr'),'OPL'),
        _link('rod_current_SI','output','rod_membrane','current',mem,mem,sparse.eye(n,format='csr'),'OPL')))
    rbxy=np.asarray(record['rod_inner']['tables']['rod_BC']['coordinates']['rf']['values']);an=len(record['rod_inner']['tables']['AII_AC']['global_cell_ids'])
    rbc=get_record('rod_BC','effective_rod_BC','BC',opl_post=rbxy)
    aii=get_record('AII_AC','effective_AII_AC','AC');bind('effective_AII_AC',an)
    def contact_pop(cid,kind,points,pid,keys):
        rec={'population':cid,'global_cell_ids':[cid+':'+hashlib.sha256(k.encode()).hexdigest()[:32] for k in keys],
            'source_keys':keys,'coordinates':{'soma':{'values':points.tolist(),'unit':'mm','meaning':'anatomical_position','frame':'anatomical_retina'},
                'ipl_preterminal':{'values':points.tolist(),'unit':'mm','meaning':'terminal_position','frame':'IPL'},
                'ipl_post_arbor':{'values':points.tolist(),'unit':'mm','meaning':'dendritic_position','frame':'IPL'}}}
        if spatial_asset is not None:rec=spatial_asset.contact_record(cid)
        recipe['contact_populations'][cid]=rec
        anchor=rec.get('physical_contact_anchor');opl_anchor=None
        if anchor is not None:
            value=q(anchor['values'],anchor['unit'],anchor['meaning'],anchor['frame'])
            value.require('mm','contact_position',(len(rec['global_cell_ids']),2))
            if value.coordinate_frame not in ('OPL','IPL'):raise ContractError('Explicit frozen contact frame required')
            if value.coordinate_frame=='OPL':opl_anchor=value.values
        pop=_population(rec,kind,pid,'Macaque_effective_project_profile',opl_post=opl_anchor);pops[cid]=pop;return pop
    if spatial_asset is None:W,rows=bounded_matrix(xy,rbxy,40,support_mm=.04)
    else:W,rows,_=spatial_asset.primary_pool('rod_RBC',mem.cells.cell_id,rbc.cells.cell_id)
    recipe['pools']['rod_RBC']={'matrix':csr_record(W),'rows':rows,'pre_ids':list(mem.cells.cell_id),'post_ids':list(rbc.cells.cell_id)}
    coo=W.tocoo();count=W.nnz
    if count==0 or np.any(np.asarray(W.sum(1)).ravel()==0):raise ContractError('ROD_RBC_FINITE_SUPPORT_EMPTY: no driver inserted')
    release0=initial_release(registry.mechanism_profiles[membrane_pid]);gain=2e-9*coo.data
    bind('effective_individual_TRPM1',count,{'gain':gain,'drive0':release0[coo.col]})
    cp=contact_pop('individual_rod_RBC_contacts','BC',(xy[coo.col]+rbxy[coo.row])/2,'effective_individual_TRPM1',
        [mem.cells.cell_id[a]+'>'+rbc.cells.cell_id[b] for a,b in zip(coo.col,coo.row)])
    cp=replace(cp,dendritic_positions=q((xy[coo.col]+rbxy[coo.row])/2,'mm','dendritic_position','OPL'));pops[cp.population_id]=cp
    specs.append(ComponentSpec(cp.population_id,'RodContactTRPM1',{'cells':cp.population_id},'effective_individual_TRPM1'))
    take=sparse.csr_matrix((np.ones(count),(np.arange(count),coo.col)),shape=(count,n));pool=sparse.csr_matrix((np.ones(count),(coo.row,np.arange(count))),shape=(len(rbxy),count))
    links.extend((_link('rod_membrane','release',cp.population_id,'release',mem,cp,take,'OPL'),
        _link(cp.population_id,'conductance','rod_BC','excitation',cp,rbc,pool,'OPL')))
    p=registry.mechanism_profiles['effective_individual_TRPM1'];K=p.quantity('K','dimensionless','receptor_half_drive').values;hill=p.quantity('hill','dimensionless','receptor_hill_exponent').values
    excitation=np.asarray(pool@(gain/(1+(release0[coo.col]/K)**hill))).ravel();d=registry.runtime_profile.data['profiles']['effective_rod_BC']
    rv=(d['gL']['value']*d['EL']['value']+excitation*d['EE']['value'])/(d['gL']['value']+excitation)
    bind('effective_rod_BC',len(rbxy),{'V0':rv})
    specs.extend((ComponentSpec('rod_BC','ReceptorConductanceMembrane',{'cells':'rod_BC'},'effective_rod_BC',{'cell_kind':'rod_BC_ON'},'rod_BC'),
        ComponentSpec('AII_AC','ReceptorConductanceMembrane',{'cells':'AII_AC'},'effective_AII_AC',{'cell_kind':'AII'},'AII_AC')))
    def point(pop):return pop.terminal_positions.values
    chemical_audits={}
    def chemical(cid,pre,post,receptor):
        if post not in pops and getattr(spatial_asset,'allow_absent_contact_targets',False):
            recipe.setdefault('absent_dependency_targets',[]).append({'component':cid,'target':post,'reason':'outside exact selected global dependency view'})
            return
        source,target=pops[pre],pops[post]
        if spatial_asset is None:W,rows=bounded_matrix(point(source),point(target),3,support_mm=.04);overlap=None
        else:W,rows,overlap=spatial_asset.primary_pool(cid,source.cells.cell_id,target.cells.cell_id)
        c=W.tocoo();nn=W.nnz
        bounds=record['population_revision']['masters']['AII_AC' if pre=='AII_AC' else 'rod_BC']['recipe']['canonical_bounds_mm']
        coverage=finite_support_coverage(point(target),.04,[bounds])
        if not np.all(coverage):raise ContractError(cid+': presynaptic fixed master does not cover actual target support')
        if nn==0:raise ContractError(cid+': complete but empty actual chemical graph; not substituting unapproved driver')
        if overlap is None:
            distance=np.linalg.norm(point(source)[c.col]-point(target)[c.row],axis=1);overlap=np.exp(-(distance/.03)**2)
        overlap=np.asarray(overlap)
        declaration=(spatial_asset.empty_chemical_declaration(cid,pre,post,W)
            if hasattr(spatial_asset,'empty_chemical_declaration') else None)
        if declaration is not None:
            from .finite_input_support import OPTION
            for index,target_spec in enumerate(specs):
                if target_spec.component_id==post:
                    options=dict(target_spec.options)
                    options[OPTION]=dict(options.get(OPTION,{}),excitation=declaration)
                    specs[index]=replace(target_spec,options=options);break
            else:raise ContractError('Declared empty chemical target component is absent')
        recipe['pools'][cid]={'matrix':csr_record(W),'rows':rows,'pre_ids':list(source.cells.cell_id),
            'post_ids':list(target.cells.cell_id),'overlap':overlap.tolist()}
        rates=initial_event_rate(registry.mechanism_profiles[source.cells.parameter_profile_id[0]])[c.col]
        pid='effective_'+cid;derived={'ncontact':overlap,'event_rate0':rates}
        if receptor=='inhibitory':
            data=registry.runtime_profile.data['profiles'][pid];lo=data['decay_minus100']['value'];hi=data['decay_0']['value']
            targetV=registry.mechanism_profiles[target.cells.parameter_profile_id[0]].quantity('V0','V','initial_membrane_voltage').values[c.row]
            derived['tau_decay']=lo+(hi-lo)*np.clip((targetV+.1)/.1,0,1)
        bind(pid,nn,derived);pop=contact_pop(cid,'AC',(point(source)[c.col]+point(target)[c.row])/2,pid,
            [source.cells.cell_id[a]+'>'+target.cells.cell_id[b] for a,b in zip(c.col,c.row)])
        specs.append(ComponentSpec(cid,'DoubleExponentialChemicalSynapse',{'cells':cid},pid,{'receptor':receptor,'voltage_dependent_decay':receptor=='inhibitory','enabled':True}))
        take=sparse.csr_matrix((np.ones(nn),(np.arange(nn),c.col)),shape=(nn,len(source.cells.cell_id)));pool=sparse.csr_matrix((np.ones(nn),(c.row,np.arange(nn))),shape=(len(target.cells.cell_id),nn))
        links.extend((_link(pre,'event_rate',cid,'event_rate',source,pop,take,'IPL'),
            _link(cid,'conductance',post,'excitation' if receptor=='excitatory' else 'inhibition',pop,target,pool,'IPL')))
        if receptor=='inhibitory':
            back=sparse.csr_matrix((np.ones(nn),(np.arange(nn),c.row)),shape=(nn,len(target.cells.cell_id)))
            links.append(_link(post,'voltage',cid,'postsynaptic_voltage',target,pop,back,'IPL'))
        chemical_audits[cid]={'edges':nn,'degree_rows':rows,'overlap_radius_mm':.03,'support_mm':.04,'weights_not_retuned':True}
        if declaration is not None:chemical_audits[cid]['finite_empty_chemical_input']=declaration
    chemical('RBC_AII_AMPA','rod_BC','AII_AC','excitatory')
    chemical('AII_FMB_glycine','AII_AC','midget_BC_OFF','inhibitory');chemical('AII_DB1_glycine','AII_AC','diffuse_BC_OFF','inhibitory')
    def gap(left,right,pairs,gain,cid):
        if not pairs:return
        a,b=pops[left],pops[right];pairs=np.asarray(pairs,int);distance=np.linalg.norm(point(a)[pairs[:,0]]-point(b)[pairs[:,1]],axis=1)
        keep=distance<=.04;pairs=pairs[keep];distance=distance[keep]
        if not len(pairs):return
        g=gain*np.exp(-(distance/.03)**2)
        evidence=EvidenceValue(cid,q(g,'S','gap_junction_conductance'),'project_approximation',PROJECT_SOURCE,
            'Unchanged reciprocal physical terminal gap and overlap law',{'support_mm':.04,'radius_mm':.03},'Uncalibrated project conductance; no HC gap enabled')
        gaps.append(GapJunction(left,tuple(a.cells.cell_id[i] for i in pairs[:,0]),right,tuple(b.cells.cell_id[i] for i in pairs[:,1]),evidence))
    if spatial_asset is None:
        gap('AII_AC','AII_AC',[(i,i+1) for i in range(an-1)],.2e-9,'AII_AII_homologous')
        tree=cKDTree(point(aii))
        for name in ('midget_BC_ON','diffuse_BC_ON','S_ON_BC'):
            if name not in pops:continue
            gap('AII_AC',name,[(int(tree.query(p)[1]),i) for i,p in enumerate(point(pops[name]))],.15e-9,'AII_'+name+'_bidirectional')
    else:gaps=spatial_asset.gaps()
    recipe['gaps']=plain(gaps)
    # Restore terminal-based initial references for downstream unchanged kernels/RGCs.
    for spec in specs:
        if spec.mechanism=='ResidualAC':
            inputs=[x for x in links if x.target_component==spec.component_id and x.target_port=='drive'];x=inputs[0]
            pre=pops[x.source_component];post=pops[spec.component_id]
            pi={g:i for i,g in enumerate(pre.cells.cell_id)};po={g:i for i,g in enumerate(post.cells.cell_id)};t=x.connections
            W=sparse.csr_matrix((t.weight,([po[g] for g in t.post_cell_id],[pi[g] for g in t.pre_cell_id])),shape=(len(po),len(pi)))
            bind(spec.parameter_profile_id,len(po),{'baseline':np.asarray(W@initial_release(registry.mechanism_profiles[pre.cells.parameter_profile_id[0]])).ravel()})
    totals={}
    for component in specs:
        if component.mechanism!='SynapticConductance':continue
        link=next(x for x in links if x.target_component==component.component_id and x.target_port=='activation')
        pre=pops[link.source_component];post=pops[component.population_roles['cells']];t=link.connections
        pi={g:i for i,g in enumerate(pre.cells.cell_id)};po={g:i for i,g in enumerate(post.cells.cell_id)}
        W=sparse.csr_matrix((t.weight,([po[g] for g in t.post_cell_id],[pi[g] for g in t.pre_cell_id])),shape=(len(po),len(pi)))
        initial=(np.logaddexp(0.,registry.mechanism_profiles[pre.cells.parameter_profile_id[0]].quantity('bias','dimensionless','activation_bias').values)
            if set(pre.cells.cell_type)=={'AC'} else initial_release(registry.mechanism_profiles[pre.cells.parameter_profile_id[0]]))
        gain=registry.mechanism_profiles[component.parameter_profile_id].quantity('gain','S','synaptic_conductance_scale').values
        receptor=component.options['receptor'];key=(post.population_id,receptor)
        totals[key]=totals.get(key,0)+gain*np.asarray(W@initial).ravel()
    for component in specs:
        if component.mechanism!='HistoryConductanceRGC':continue
        name=component.population_roles['cells'];d=registry.runtime_profile.data['profiles'][component.parameter_profile_id]
        e,i=totals[name,'excitatory'],totals[name,'inhibitory']
        bind(component.parameter_profile_id,len(pops[name].cells.cell_id),{'V0':(d['gL']['value']*d['EL']['value']+e*d['EE']['value']+i*d['EI']['value'])/(d['gL']['value']+e+i)})
    required=tuple(json.loads((root/'configurations/full_target.json').read_bytes())['required_mechanisms'])
    if audit['atlas_runtime'].get('empty_owner_outputs'):
        required=tuple(layer for layer in required if layer not in record['inner_tables'] or layer in pops)
    if getattr(spatial_asset,'allow_absent_contact_targets',False):
        absent={row['component'] for row in recipe.get('absent_dependency_targets',())}
        required=tuple(layer for layer in required if layer not in absent)
    spec=replace(compiled.spec,populations=tuple(pops.values()),components=tuple(specs),links=tuple(links),gap_junctions=tuple(gaps),
        external_inputs=compiled.spec.external_inputs+(ExternalInput('rod_light','rod','light'),),required_mechanisms=required)
    result=builder.build(spec,RunMode.REFERENCE)
    result.canonical_spatial_recipe=recipe
    audit.update(spec=spec,resource_estimate=builder.estimate(spec),population_records={k:p.record() for k,p in pops.items()},chemical_synapses=chemical_audits)
    audit['atlas_runtime'].update(full_target_readiness='ASSEMBLED_PENDING_SHORT_RUN',full_20_layer_network_claim=False,
        physical_BC_terminal=True,rod_RBC_AII_bound=True,unsupported_paths={},rod_RBC_degree_rows=rows,
        chemical_contacts=chemical_audits,
        connection_capacity_deficits={cid:[dict(row,post_cell_id=pops[post].cells.cell_id[i]) for i,row in enumerate(chemical_audits[cid]['degree_rows']) if row['missing_degree']>0]
            for cid,post in [('RBC_AII_AMPA','AII_AC'),('AII_FMB_glycine','midget_BC_OFF'),('AII_DB1_glycine','diffuse_BC_OFF')] if cid in chemical_audits},
        executed_population_counts={k:len(p.cells.cell_id) for k,p in pops.items()})
    audit['population_owner_mask'].update({k:_population_owner_mask(p,record['atlas']['owner_domain'])
        for k,p in pops.items() if k not in audit['population_owner_mask']})
    audit['regional_support']={'scope':'complete local effective typed circuit; HC/global domains remain unresolved','unsupported_paths':{}}
    if audit['atlas_runtime'].get('empty_owner_outputs'):
        audit['atlas_runtime']['full_target_readiness']='ASSEMBLED_NONEMPTY_OWNER_TYPES_PENDING_RUN'
    return result,channels,audit
