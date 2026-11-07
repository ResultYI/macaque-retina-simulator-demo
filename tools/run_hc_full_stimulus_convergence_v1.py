from retina_sim.hc_research_native import joint_equilibrium
"""Complete annular input and bounded nested-domain experiment; never fit parameters."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import time
import traceback
import json
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.optimize import brentq
from scipy.special import expit
from scipy.integrate import solve_ivp

from retina_sim.contracts import canonical_hash
from retina_sim.disk_tables import FrozenDiskTable
from retina_sim.hc_biexponential import support_radius
from retina_sim.hc_minimal_diagnostic import bind_diagnostic, require_diagnostic_native
from retina_sim.hc_timevarying_exterior import FixedFeedbackCone, DrivenExteriorH1
from retina_sim.h1_h2_effective import cone_fixed_point, matrix_id
from retina_sim.cone_hc_reduced_native import AnalyticREAnchor
from retina_sim.vh2005 import SOURCE_VALUES as P
from tools.run_hc_minimal_real_diagnostic_v1 import sha, write, extend_labels, native_model, peak_memory_bytes


def readout(n, t, y, target, core):
    h = y[n.slices['Vh']]; feedback = n.B@h+n.boundary['B_h'][:, None]
    release = expit((y[n.slices['Vis']]-feedback-P['V_k'])/P['V_n'])/(y[n.slices['Vbar']]/P['V_I'])**P['mu']
    boundaries = [n.boundary_at(float(v)) for v in t]
    pool = n.A@release+np.column_stack([b['A_release'] for b in boundaries])
    ext = n.stimulated_exterior_mass[target]*np.array([n.exterior_cache.at(float(v))[0] for v in t])
    for a in (y, feedback, release, pool):
        if not np.isfinite(a).all(): raise RuntimeError('Nonfinite saved output')
    for key in ('X', 'C', 'Vis', 'Vh'):
        if np.any(y[n.slices[key]] < 0): raise RuntimeError('Negative native state')
    for key in ('g_i', 'Vbar'):
        if np.any(y[n.slices[key]] <= 0): raise RuntimeError('Nonpositive native state')
    return dict(t_seconds=t, y=y, H1_target=h[target], release_core=release[core],
                feedback_core=feedback[core], pooled_release_target=pool[target],
                exterior_stimulated_input_target=ext, release_all=release,
                H1_all=h, exterior_stimulated_input_delta_target=ext-ext[0])


def advance(n, y0, light, stimulus, tightened):
    times = np.arange(61)*.002; pieces = []; nfev = 0
    clock = time.perf_counter(); anchor = AnalyticREAnchor(0., light, light, light, .02, 0)
    for start, stop, mask in ((0., .02, times <= .02), (.02, .12, times > .02)):
        sol = solve_ivp(lambda t, y: n.rhs(y, t, anchor), (start, stop), y0,
            method='DOP853', rtol=1e-9 if tightened else 1e-7,
            atol=1e-12 if tightened else 1e-10, max_step=.0001 if tightened else .0002,
            t_eval=times[mask])
        if not sol.success: raise RuntimeError(sol.message)
        pieces.append(sol.y); nfev += sol.nfev; y0 = sol.y[:, -1]
        if stop == .02:
            R, E = anchor.at(.02); anchor = AnalyticREAnchor(.02, R, E, stimulus, .12, 1)
    return times, np.column_stack(pieces), dict(wall_seconds=time.perf_counter()-clock, nfev=nfev,
                                             simulated_seconds=.12, tightened=tightened)


def comparison(candidate, reference):
    checks = {}
    for condition in ('B_biexponential', 'C_feedback_off', 'on_minus_off'):
        for key, floor in [('H1_target', 1e-6), ('release_core', 1e-8)]:
            if condition == 'on_minus_off':
                a = candidate['B_biexponential'][key]-candidate['C_feedback_off'][key]
                b = reference['B_biexponential'][key]-reference['C_feedback_off'][key]
            else: a, b = candidate[condition][key], reference[condition][key]
            error = np.max(abs(a-b), axis=-1)
            allowed = floor+.01*np.max(abs(b-b[..., :1]), axis=-1)
            checks[condition+'_'+key] = dict(max_abs_error=np.asarray(error).tolist(),
                allowed=np.asarray(allowed).tolist(), passed=bool(np.all(error <= allowed)))
    return dict(checks=checks, passed=all(v['passed'] for v in checks.values()),
                rule='absolute full trajectory error; reference dynamic peak; no baseline removal from error')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); root = Path(__file__).resolve().parents[1]
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    began = datetime.now(timezone.utc).isoformat(); clock = time.perf_counter()
    result = dict(started_utc=began, status='RUNNING', production_adopted=False, experiments={}, comparisons=[])
    try:
        v5 = root/'reports/CONE_HC_FINITE_DYNAMIC_HALO_20261102_v5'
        v6 = root/'reports/CONE_HC_MINIMAL_REAL_DIAGNOSTIC_20261102_v6/run_02'
        old = json.loads((v6/'RESULT.json').read_bytes())
        if old['status'] != 'PASS_CONDITIONAL_REAL_GRAPH_DIAGNOSTIC': raise RuntimeError('v6 not qualified')
        oldgeo = np.load(v6/'GEOMETRY.npz', allow_pickle=False)
        base = root/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2/executions/assemble_native_01'
        rec = json.loads((base/'NATIVE_ASSEMBLY_RESULT.json').read_bytes())
        cones, hc = FrozenDiskTable(root, rec['cone_table']), FrozenDiskTable(root, rec['hc_table'])
        raw = sparse.load_npz(base/'RAW_HC_BY_CONE.npz').tocsr()
        cx, hx = cones.coordinates['opl_pedicle'], hc.coordinates['soma']
        ct, ht = cones.extras['subtype'].astype('U1'), hc.extras['subtype'].astype('U2')
        core = raw[2891].indices; seeds = np.r_[core,166128]
        plans = []; domains = {}
        for name, pad in [('small', .005), ('medium', .035), ('large', .08)]:
            lo, hi = cx[seeds].min(0)-pad, cx[seeds].max(0)+pad
            ci = np.flatnonzero(np.all((cx >= lo)&(cx <= hi), 1)&(ct != 'S'))
            partners = np.unique(raw[:, ci].tocoo().row)
            hj = np.union1d(partners[ht[partners] == 'H1'], [2891])
            plans.append(dict(name=name, pad_mm=pad, cones=len(ci), H1=len(hj), states=5*len(ci)+3*len(hj),
                dense_A_upper_entries=len(ci)*len(hj), estimated_rhs_cost_vs_v6=len(ci)*len(hj)/(321*249),
                estimated_integration_seconds=old['experiments']['B_biexponential']['integration']['wall_seconds']*len(ci)*len(hj)/(321*249)))
            domains[name] = dict(ci=ci,hj=hj,bounds=[lo[0],hi[0],lo[1],hi[1]])
        np.testing.assert_array_equal(domains['small']['ci'], oldgeo['cone_parent_rows'])
        np.testing.assert_array_equal(domains['small']['hj'], oldgeo['hc_parent_rows'])
        for a,b in [('small','medium'),('medium','large')]:
            assert np.isin(domains[a]['ci'], domains[b]['ci']).all() and np.isin(domains[a]['hj'], domains[b]['hj']).all()
        protocol = dict(schema='HC_complete_stimulus_nested_v1', plans=plans, max_domains=3,
            input=dict(background_td=1000., contrast=.01, center_IS_mm=hx[2891].tolist(),
                inner_mm=.12, outer_mm=.32, onset_s=.02, offset_s=.12, duration_s=.12),
            source_scope='unchanged 6mm; all annular L/M/S receive true input; H1 excludes S',
            exterior='native Cone states under uniform fixed HC background, exact common-drive cache; all fine weights retained',
            budgets=dict(H1_floor=1e-6,release_floor=1e-8,relative=.01,solver_fraction=.1),
            solver=dict(method='DOP853',rtol=1e-7,atol=1e-10,max_step=.0002),
            refinement=dict(rtol=1e-9,atol=1e-12,max_step=.0001,conditions=['B_biexponential','C_feedback_off']),
            direction='larger domain is reference; absolute trajectories including baseline; on/off and on-minus-off',
            stop='small/medium first; at most one large comparison if failed; uniform only after convergence')
        write(out/'FROZEN_PROTOCOL.json', dict(frozen_utc=began, protocol=protocol))
        print('Frozen cost plans:', plans, flush=True)
        source = root/'reports/CONE_HC_OUTER_SOURCE_EXTENSION_20261102_v3/cones_01'
        sr = json.loads((source/'RESULT.json').read_bytes())
        for key in ['cone_inner_segment_mm.npy','cone_pedicle_mm.npy','official_source_rows.npy']:
            if sha(source/key) != sr['files'][key]: raise RuntimeError('Source changed')
        isxy, ped, official = (np.load(source/key) for key in ['cone_inner_segment_mm.npy','cone_pedicle_mm.npy','official_source_rows.npy'])
        old_official = np.array([int(k.decode().split(':')[-1]) for k in cones.keys])
        old_map = np.searchsorted(official, old_official)
        np.testing.assert_array_equal(official[old_map], old_official)
        labels = np.full(len(ped),'',dtype='U1'); labels[old_map] = ct
        for path in [v5/'preflight_01/TARGET_A_MEMBERS.npz', v6/'SOURCE_LABELS.npz']:
            with np.load(path,allow_pickle=False) as z:
                ids=z['cone_candidate_rows']; labs=z['cone_types'];known=labels[ids]!=''
                np.testing.assert_array_equal(labels[ids][known],labs[known]);labels[ids]=labs
        config_path = root/'configurations/research/HC_biexponential_vh2007_reference_v1.json'
        cfg = json.loads(config_path.read_bytes()); par=cfg['H1']; cutoff=support_radius(.3,cfg['tail_mass_budget'])
        tree=cKDTree(ped); largest_h=domains['large']['hj']
        env=cutoff+np.max(np.linalg.norm(hx[largest_h]-hx[2891],axis=1))
        needed=np.asarray(tree.query_ball_point(hx[2891],env),int)
        dist=np.linalg.norm(isxy-hx[2891],axis=1);stim=(dist>=.12)&(dist<.32)
        needed=np.union1d(needed,np.flatnonzero(stim))
        unknown=needed[labels[needed]=='']
        write(out/'ADDITIONAL_LABELS.json',extend_labels(root,isxy,official,unknown,labels,sr))
        ring=np.flatnonzero(stim)
        if np.any(labels[ring]==''):raise RuntimeError('Unlabelled annular Cone')
        np.savez_compressed(out/'SOURCE_MEMBERS.npz',cone_candidate_rows=needed,official_rows=official[needed],
            cone_types=labels[needed],ring_candidate_rows=ring,ring_types=labels[ring],ring_IS_mm=isxy[ring],ring_pedicle_mm=ped[ring])
        coverage=dict(annulus_total=len(ring),by_type={k:int(np.sum(labels[ring]==k)) for k in 'LMS'},
                      untreated_stimulated_cones=0,method='all annular cones either fully coupled or fixed-feedback native transduction')
        files=[Path(__file__),config_path,base/'RAW_HC_BY_CONE.npz',base/'NATIVE_ASSEMBLY_RESULT.json',
            v6/'RESULT.json',v6/'GEOMETRY.npz',v6/'SOURCE_LABELS.npz',v5/'preflight_01/TARGET_A_ONLY.npz',
            root/'retina_sim/hc_timevarying_exterior.py',root/'retina_sim/h1_h2_effective.py',
            root/'retina_sim/hc_minimal_diagnostic.py',root/'retina_sim/cone_hc_reduced_native.py',root/'retina_sim/vh2005.py']
        source_hashes={str(p.relative_to(root)):sha(p) for p in files};write(out/'SOURCE_IDENTITIES.json',source_hashes)
        common_identity=canonical_hash(dict(sources=source_hashes,protocol=protocol,ring=official[ring],labels=labels[ring]))
        eqscalar=cone_fixed_point(np.array([1000.]));v0=float(eqscalar['Vis'][0]);ai0=(v0/P['V_I'])**P['mu']
        hbg=brentq(lambda h:h-P['g_t']*expit((v0-h-P['V_k'])/P['V_n'])/ai0,0.,P['g_t']/ai0,xtol=1e-12)
        caches={};cache_seconds=0.
        for off in (False,True):
            cc=time.perf_counter();cache=FixedFeedbackCone(0. if off else hbg,member_identity=common_identity)
            caches[off]=cache;cache_seconds+=time.perf_counter()-cc
            t=np.arange(61)*.002;ys=np.column_stack([cache.y0 if v<=.02 else cache.solution.sol(v) for v in t])
            np.savez_compressed(out/('EXTERIOR_off.npz' if off else 'EXTERIOR_on.npz'),t_seconds=t,y=ys,
                release=np.array([cache.at(float(v))[0] for v in t]),feedback=cache.feedback,identity=cache.identity,
                member_official_rows=official[ring],source_identity=common_identity)
        # Cache complete row weights on the largest possible dynamic column set.
        max_ci=domains['large']['ci'];max_src=old_map[max_ci];dmap=np.full(len(ped),-1,int);dmap[max_src]=np.arange(len(max_ci))
        rows={};build_seconds=0.;model_seconds=0.;saved={};contexts={}
        def build(hj):
            nonlocal build_seconds
            began_build=time.perf_counter()
            for ph in hj:
                if int(ph) in rows:continue
                local=old_map[raw[ph].indices];coeff=raw[ph].data
                dd=np.linalg.norm(ped[local]-hx[ph],axis=1)
                short=coeff*np.exp(-(dd-dd.min())/.020);short/=short.sum()
                far=np.asarray(sorted(tree.query_ball_point(hx[ph],cutoff)),int);far=far[labels[far]!='S']
                if np.any(labels[far]==''):raise RuntimeError('Missing wide-source labels')
                dd=np.linalg.norm(ped[far]-hx[ph],axis=1);long=np.exp(-(dd-dd.min())/.300);long/=long.sum()
                entry={}
                for mode,components in [('uniform',[(local,coeff/coeff.sum())]),
                    ('biexponential',[(local,.15*short),(far,.85*long)])]:
                    dynamic=np.zeros(len(max_ci));smass=0.
                    for ids,weights in components:
                        inside=dmap[ids]>=0;np.add.at(dynamic,dmap[ids[inside]],weights[inside]);smass+=weights[stim[ids]].sum()
                    entry[mode]=(dynamic,float(smass))
                rows[int(ph)]=entry
            build_seconds+=time.perf_counter()-began_build
        def run_domain(name,condition,tight=False):
            nonlocal model_seconds,cache_seconds
            mode='uniform' if condition=='A_uniform' else 'biexponential';off=condition=='C_feedback_off'
            dom=domains[name];ci,hj=dom['ci'],dom['hj'];build(hj)
            cols=np.searchsorted(max_ci,ci);nc,nh=len(ci),len(hj);cs=old_map[ci]
            A=sparse.csr_matrix(np.array([rows[int(h)][mode][0][cols] for h in hj]))
            mass=np.array([rows[int(h)][mode][1] for h in hj]);dynamic_mass=np.asarray(A.sum(1)).ravel()
            ah=1-dynamic_mass;em=mass-np.asarray(A@stim[cs].astype(float)).ravel()
            em[np.abs(em)<2e-14]=0.;ah[np.abs(ah)<2e-14]=0.
            if np.any(em<0) or np.any(ah<0) or np.any(em>ah+2e-14):raise RuntimeError('Invalid exterior mass ledger')
            full_count=np.asarray(raw[ht=='H1'][:,ci].sum(0)).ravel()
            B=(sparse.diags(np.divide(1.,full_count,out=np.zeros(nc),where=full_count>0))@raw[hj][:,ci].T).tocsr()
            zero=np.flatnonzero(full_count==0)
            master=json.loads((root/'data/atlas/GIN_69_6_HC2_BC1_SUPPORT_V2/HC_master.json').read_bytes());bb=master['boundary']['canonical_bounds_mm']
            edge=np.minimum.reduce([cx[ci,0]-bb[0],bb[1]-cx[ci,0],cx[ci,1]-bb[2],bb[3]-cx[ci,1]])
            if edge.min()<=.0327487213324989:raise RuntimeError('Dynamic B geometry incomplete')
            # All zero rows are inventoried from the complete known local contact graph.
            cache=caches[off]
            if tight:
                cc=time.perf_counter();cache=FixedFeedbackCone(0. if off else hbg,member_identity=common_identity,tightened=True);cache_seconds+=time.perf_counter()-cc
            h_ext=P['g_t']*cache.background_release
            binding=bind_diagnostic(A,B,cone_parent_rows=ci,registered_unsupported_rows=ci[zero],
                A_held_mass=ah,B_held_mass=np.zeros(nc),exterior_release=cache.background_release,
                exterior_Vbar=v0,exterior_h=h_ext,source_identity=common_identity,feedback_off=off)
            base_n,pars=native_model(binding,hj,ci);n=DrivenExteriorH1(base_n.assembly);n.boundary=binding['boundary'];n.attach_exterior(cache,em)
            light=np.full(nc,1000.);on=light*(1+.01*stim[cs])
            eq=joint_equilibrium(n.A,n.B,np.arange(nh),np.array([],int),light,pars,boundary=n.boundary)
            y0=np.concatenate([eq['cone'][k] for k in ('X','C','Vis','g_i','Vbar')]+[eq['w'],eq['z2'],eq['h'],np.array([])])
            rhs=float(np.max(abs(n.rhs(y0,0.,AnalyticREAnchor(0.,light,light,light,.02,0)))))
            require_diagnostic_native(binding,input_available=np.ones(nc,bool),feedback_geometry_complete=np.ones(nc,bool),background_max_rhs=rhs)
            print('Integrate',name,condition,'refined' if tight else 'nominal',nc,nh,flush=True)
            t,y,stats=advance(n,y0,light,on,tight);model_seconds+=stats['wall_seconds']
            target=int(np.flatnonzero(hj==2891)[0]);core_i=np.searchsorted(ci,core);data=readout(n,t,y,target,core_i)
            dest=out/name;dest.mkdir(exist_ok=True);suffix='_REFINED' if tight else ''
            np.savez_compressed(dest/(condition+suffix+'.npz'),**data)
            if not tight:
                np.savez_compressed(dest/'MEMBERS.npz',cone_parent_rows=ci,hc_parent_rows=hj,cone_candidate_rows=cs,
                    cone_official_rows=official[cs],cone_types=ct[ci],cone_IS_mm=isxy[cs],cone_pedicle_mm=ped[cs],hc_anchor_mm=hx[hj],
                    core_parent_rows=core,unsupported_parent_rows=ci[zero],stimulated_dynamic=stim[cs],
                    peripheral_stimulated_candidate_rows=np.setdiff1d(ring,cs),fixed_feedback_on=hbg)
                sparse.save_npz(dest/('A_'+mode+'.npz'),A);sparse.save_npz(dest/'B_on.npz',B)
                np.savez_compressed(dest/(condition+'_BACKGROUND.npz'),y0=y0,light=light,stimulus=on,**n.boundary)
                np.savez_compressed(dest/(condition+'_MASS.npz'),A_dynamic_mass=dynamic_mass,A_held_mass=ah,
                    A_stimulated_exterior_mass=em,A_unstimulated_held_mass=ah-em,B_dynamic_mass=np.asarray(B.sum(1)).ravel(),
                    unsupported_feedback=np.isin(np.arange(nc),zero))
                info=dict(identity=n.source_version,binding_identity=binding['identity'],cache_identity=cache.identity,
                    common_source_identity=common_identity,background_max_rhs=rhs,run=stats,
                    coverage=dict(all_annular=len(ring),fully_coupled_stimulated=int(stim[cs].sum()),
                        peripheral_timevarying=int(len(ring)-stim[cs].sum()),untreated=0),
                    target_mass=dict(dynamic=float(dynamic_mass[target]),exterior_stimulated=float(em[target]),
                        exterior_unstimulated=float(ah[target]-em[target])),
                    registered_zero_B=ci[zero].tolist(),geometry_complete=True,
                    H1_baseline=float(data['H1_target'][0]),H1_delta_final=float(data['H1_target'][-1]-data['H1_target'][0]),
                    core_release_delta_final=(data['release_core'][:,-1]-data['release_core'][:,0]).tolist(),
                    feedback_rule='fixed homogeneous HC background for all peripheral source members; per-member identities retained')
                write(dest/(condition+'_RESULT.json'),info);result['experiments'][name+'_'+condition]=info
                contexts[(name,condition)]=(binding,em,cache.identity)
            return data
        for name in ('small','medium'):
            saved[name]={condition:run_domain(name,condition) for condition in ('B_biexponential','C_feedback_off')}
        cmp=comparison(saved['small'],saved['medium']);cmp.update(candidate='small',reference='medium');result['comparisons'].append(cmp)
        reference='medium'
        if not cmp['passed']:
            print('First spatial comparison failed; one bounded expansion',flush=True)
            saved['large']={condition:run_domain('large',condition) for condition in ('B_biexponential','C_feedback_off')}
            cmp=comparison(saved['medium'],saved['large']);cmp.update(candidate='medium',reference='large');result['comparisons'].append(cmp);reference='large'
        result['spatial_pass']=cmp['passed'];result['reference_domain']=reference
        refined={condition:run_domain(reference,condition,True) for condition in ('B_biexponential','C_feedback_off')}
        num=comparison(saved[reference],refined)
        for check in num['checks'].values():
            check['allowed']=(.1*np.asarray(check['allowed'])).tolist()
            check['passed']=bool(np.all(np.asarray(check['max_abs_error'])<=np.asarray(check['allowed'])))
        num['passed']=all(g['passed'] for g in num['checks'].values());result['numerical_refinement']=num
        if result['spatial_pass'] and num['passed']:
            saved[reference]['A_uniform']=run_domain(reference,'A_uniform')
            result['status']='CONE_HC_BIEXP_RESEARCH_QUALIFIED';result['downstream_integration_admitted']=True
        else:
            result['status']='SPATIAL_CONVERGENCE_FAIL' if not result['spatial_pass'] else 'NUMERICAL_REFINEMENT_FAIL'
            result['downstream_integration_admitted']=False
        v6diff={}
        for condition in ('B_biexponential','C_feedback_off'):
            with np.load(v6/(condition+'_TRAJECTORY.npz')) as z:
                data=saved['small'][condition]
                for key in ('H1_target','release_core','pooled_release_target','feedback_core'):
                    diff=data[key]-z[key];v6diff[condition+'_'+key]=dict(max_abs=float(abs(diff).max()),final=np.asarray(diff[...,-1]).tolist())
        result.update(coverage=coverage,v6_complete_stimulus_differences=v6diff,common_source_identity=common_identity,
            operator_build_wall_seconds=build_seconds,peripheral_cache_wall_seconds=cache_seconds,
            model_advance_wall_seconds=model_seconds,parameters=P,plans=plans,
            stop='bounded third domain used at most once; no further expansion',
            qualification_scope='this annulus, 1000Td, 1%, 120ms, stated OPL domains and fixed peripheral HC feedback only')
    except Exception as error:
        result.update(status='FAILED_RETAINED',error=str(error),traceback=traceback.format_exc());raise
    finally:
        result.update(ended_utc=datetime.now(timezone.utc).isoformat(),total_wall_seconds=time.perf_counter()-clock,
                      peak_process_working_set_bytes=peak_memory_bytes());write(out/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ('status','spatial_pass','reference_domain','total_wall_seconds','model_advance_wall_seconds')},indent=2),flush=True)


if __name__=='__main__':main()
