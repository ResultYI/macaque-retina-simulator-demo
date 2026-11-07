"""Audit completed saved outputs without rerunning any dynamics."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import csv
import json
import numpy as np
from scipy import sparse
from scipy.special import expit
from tools.run_hc_minimal_real_diagnostic_v1 import sha,write
from tools.run_hc_full_stimulus_convergence_v1 import comparison
from retina_sim.vh2005 import SOURCE_VALUES as P


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.run.resolve();r=json.loads((out/'RESULT.json').read_bytes())
    if r['status'] in ('RUNNING','FAILED_RETAINED'):raise RuntimeError('Completed numerical results required')
    sources=json.loads((out/'SOURCE_IDENTITIES.json').read_bytes())
    for key,digest in sources.items():assert sha(root/key)==digest,key
    members=np.load(out/'SOURCE_MEMBERS.npz',allow_pickle=False)
    source=root/'reports/CONE_HC_OUTER_SOURCE_EXTENSION_20261102_v3/cones_01'
    isxy=np.load(source/'cone_inner_segment_mm.npy');ped=np.load(source/'cone_pedicle_mm.npy')
    source_rows=np.load(source/'official_source_rows.npy')
    center=np.array(json.loads((out/'FROZEN_PROTOCOL.json').read_bytes())['protocol']['input']['center_IS_mm'])
    radius=np.linalg.norm(isxy-center,axis=1);ring=np.flatnonzero((radius>=.12)&(radius<.32))
    np.testing.assert_array_equal(ring,members['ring_candidate_rows'])
    caches={off:np.load(out/('EXTERIOR_off.npz' if off else 'EXTERIOR_on.npz')) for off in (False,True)}
    validation=dict(verified_utc=datetime.now(timezone.utc).isoformat(),status='PASS_ARTIFACT_CONSISTENCY',
        scope='saved source, mass, input and trajectory consistency; scientific spatial decision is separate',
        source_files_checked=len(sources),annulus_total=len(ring),experiments={},comparisons={})
    for key,meta in r['experiments'].items():
        domain,condition=key.split('_',1);dest=out/domain;g=np.load(dest/'MEMBERS.npz')
        ci,hj,cs=g['cone_parent_rows'],g['hc_parent_rows'],g['cone_candidate_rows'];nc,nh=len(ci),len(hj)
        np.testing.assert_array_equal(g['cone_IS_mm'],isxy[cs]);np.testing.assert_array_equal(g['cone_pedicle_mm'],ped[cs]);np.testing.assert_array_equal(g['cone_official_rows'],source_rows[cs])
        full_stim=g['stimulated_dynamic'];peripheral=g['peripheral_stimulated_candidate_rows']
        np.testing.assert_array_equal(np.union1d(cs[full_stim],peripheral),ring)
        assert not np.intersect1d(cs,peripheral).size
        assert int(full_stim.sum())+len(peripheral)==len(ring)
        d=np.load(dest/(condition+'.npz'));bg=np.load(dest/(condition+'_BACKGROUND.npz'));mass=np.load(dest/(condition+'_MASS.npz'))
        A=sparse.load_npz(dest/('A_uniform.npz' if condition=='A_uniform' else 'A_biexponential.npz'))
        B=sparse.load_npz(dest/'B_on.npz');off=condition=='C_feedback_off';cache=caches[off]
        if off:B=sparse.csr_matrix(B.shape)
        np.testing.assert_allclose(mass['A_dynamic_mass']+mass['A_held_mass'],1,rtol=0,atol=2e-14)
        np.testing.assert_allclose(mass['A_stimulated_exterior_mass']+mass['A_unstimulated_held_mass'],mass['A_held_mass'],rtol=0,atol=2e-14)
        zero=g['unsupported_parent_rows'];expected=np.isin(ci,zero)
        B_on=sparse.load_npz(dest/'B_on.npz');np.testing.assert_allclose(np.asarray(B_on.sum(1)).ravel(),(~expected).astype(float),rtol=0,atol=2e-14)
        y=d['y'];assert y.shape==(5*nc+3*nh,61);assert np.isfinite(y).all();np.testing.assert_array_equal(d['t_seconds'],np.arange(61)*.002)
        assert (y[:3*nc]>=0).all() and (y[3*nc:5*nc]>0).all() and (y[5*nc:]>=0).all()
        np.testing.assert_array_equal(y[:,0],bg['y0']);np.testing.assert_array_equal(bg['stimulus'],1000*(1+.01*full_stim))
        feedback=B@d['H1_all']+bg['B_h'][:,None]
        rel=expit((y[2*nc:3*nc]-feedback-P['V_k'])/P['V_n'])/(y[4*nc:5*nc]/P['V_I'])**P['mu']
        np.testing.assert_allclose(rel,d['release_all'],rtol=0,atol=2e-15)
        target=int(np.flatnonzero(hj==2891)[0]);core=np.searchsorted(ci,g['core_parent_rows'])
        boundary=bg['A_release'][:,None]+mass['A_stimulated_exterior_mass'][:,None]*(cache['release']-cache['release'][0])
        pool=(A@rel+boundary)[target];np.testing.assert_allclose(pool,d['pooled_release_target'],rtol=0,atol=2e-15)
        np.testing.assert_array_equal(feedback[core],d['feedback_core'])
        assert meta['background_max_rhs']<=1e-8 and meta['cache_identity']==str(cache['identity'])
        cols=['time_seconds','H1_target','H1_spatial_input','exterior_stimulated_input','exterior_stimulated_delta']
        cols += ['release_'+str(v) for v in g['core_parent_rows']]+['feedback_'+str(v) for v in g['core_parent_rows']]
        table=np.column_stack([d['t_seconds'],d['H1_target'],d['pooled_release_target'],d['exterior_stimulated_input_target'],
                              d['exterior_stimulated_input_delta_target'],d['release_core'].T,d['feedback_core'].T])
        with (dest/(condition+'.csv')).open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f);writer.writerow(cols);writer.writerows(table)
        validation['experiments'][key]=dict(coverage_union_exact=True,untreated=0,full_samples=61,
            weights_and_state_readouts_recomputed=True,all_states_legal=True,registered_zero_rows=zero.tolist(),
            initial_rhs=meta['background_max_rhs'])
    for item in r['comparisons']:
        a,b=item['candidate'],item['reference']
        datasets={n:{c:dict(np.load(out/n/(c+'.npz'))) for c in ('B_biexponential','C_feedback_off')} for n in (a,b)}
        comp=comparison(datasets[a],datasets[b]);assert comp['passed']==item['passed']
        for k,v in comp['checks'].items():np.testing.assert_array_equal(v['max_abs_error'],item['checks'][k]['max_abs_error'])
        validation['comparisons'][a+'_vs_'+b]=dict(passed=comp['passed'],independently_recomputed=True)
    validation['spatial_qualified']=r['spatial_pass'];validation['numerical_qualified']=r['numerical_refinement']['passed']
    write(out/'ARTIFACT_VALIDATION.json',validation);print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
