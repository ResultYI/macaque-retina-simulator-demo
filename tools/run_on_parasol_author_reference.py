"""Isolated, partial CBEM reproduction with explicitly selected source versions.

NumPy/SciPy re-expression, not execution of the unavailable JAX notebook.
No animal observations, fitted filters, current-data reuse, or production writes.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
from scipy.linalg import solve_banded

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retina_sim.latimer2019 import reference_rgc, exact_mean_events

DATA=ROOT/'data/source_data/latimer2019_figshare_v1'
BASE=ROOT/'reports/modeling_scope_20261020'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def paper_voltage(gs,dt,gl=200.,El=-60.,Es=(0.,-80.),V0=-60.):
    """Equation6 exact ZOH solution, in mV and normalized conductance s^-1."""
    total=gl+np.sum(gs,axis=1)
    equilibrium=(El*gl+gs@np.asarray(Es))/total
    decay=np.exp(-dt*total)
    v=np.empty(len(gs)+1);v[0]=V0
    for j in range(len(gs)):
        v[j+1]=equilibrium[j]+(v[j]-equilibrium[j])*decay[j]
    return v


def matlab2020_voltage_algorithm(gs,dt,gl=200.,El=-60.,Es=(0.,-80.)):
    """simulateCBEM.m voltage matrix under paper coefficients, not setup defaults.

    Source adds E_l to the first RHS; it has no separate custom V0 argument.
    """
    total=gl+np.sum(gs,axis=1)
    expg=np.exp(dt*total)
    rhs=(El*gl+gs@np.asarray(Es))/total*(expg-1.)
    rhs[0]+=El
    ab=np.zeros((2,len(gs)));ab[0]=expg;ab[1,:-1]=-1.
    return solve_banded((1,0),ab,rhs)


def python2022_voltage_literal(gs,dt,gl=200.,El=-60.,Es=(0.,-80.),V0=-60.):
    """Literal source indexing, float64 SciPy lower-bidiagonal solve.

    Preserves C[0]+=V0 and lower diagonal -exp(-dt*g_total[:-1]).
    This is not silently corrected to the continuous equation.
    """
    total=gl+np.sum(gs,axis=1)
    M=-np.exp(-dt*total)
    C=(El*gl+gs@np.asarray(Es))/total*(1.+M)
    C[0]+=V0
    ab=np.zeros((2,len(gs)));ab[0]=1.;ab[1,:-1]=M[:-1]
    v=solve_banded((1,0),ab,C)
    # A separately implemented recurrence checks matrix storage/indexing.
    recurrence=np.empty(len(gs));recurrence[0]=C[0]
    for j in range(1,len(gs)):recurrence[j]=C[j]-M[j-1]*recurrence[j-1]
    np.testing.assert_allclose(v,recurrence,rtol=1e-14,atol=1e-12)
    return v


def softplus_map(v,alpha=90.,mu=-53.,beta=1.67):
    return alpha*np.logaddexp(0.,(np.asarray(v)-mu)/beta)


def main():
    started=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((BASE/'standard_stimulus_rate_audit_v1/baseline_frozen_identity.json').read_bytes())
    protected={**baseline['files'],**baseline['fixed_parameters_and_equations']}
    ui=json.loads((BASE/'standard_stimulus_rate_audit_v1/scope_integrity.json').read_bytes())['source_sha256']
    protected.update(ui)
    for rel,h in protected.items():assert sha(ROOT/rel)==h,rel
    identity={}
    tree=json.loads((DATA/'author_repository_tree.json').read_bytes())
    assert tree['sha']=='7beb7e72b0678192f0bc98392a678cdff9601791'
    by={r['path']:r['sha'] for r in tree['tree']}
    for file,path in [('author_pyCBEM__RGC_CBEM.py','pyCBEM/RGC_CBEM.py'),('author_pyCBEM__utils.py','pyCBEM/utils.py')]:
        raw=(DATA/file).read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert blob==by[path]
        identity[path]=dict(git_blob_sha1=blob,sha256=sha(DATA/file))
    # Freeze the actual source expressions used by the translation.
    utils=(DATA/'author_pyCBEM__utils.py').read_text(encoding='utf-8')
    cbem=(DATA/'author_pyCBEM__RGC_CBEM.py').read_text(encoding='utf-8')
    assert 'jnp.concatenate((jnp.zeros(1), M[:-1]))' in utils
    assert 'C = C.at[0].add(V_0)' in utils
    assert '"alpha" : 90, "mu" : -53, "beta" : 1.67' in cbem
    native,_=reference_rgc(ROOT)
    dt=.0001;t=np.arange(5000)*dt
    # Predeclared engineering conductance-clamp inputs, not visual stimuli or
    # author-fitted filters. All values are g/C in s^-1, not physical siemens.
    gs=np.zeros((5000,2))
    phases=[('leak_only',0.,0.),('E_step',40.,0.),('E_plus_I',40.,40.),
            ('recovery',0.,0.),('equal_E_I',10.,10.)]
    for j,(_,e,i) in enumerate(phases):gs[j*1000:(j+1)*1000]=[e,i]
    vpaper=paper_voltage(gs,dt)
    vmat=matlab2020_voltage_algorithm(gs,dt)
    vpy=python2022_voltage_literal(gs,dt)
    matrix_error=float(np.max(abs(vmat-vpaper[1:])))
    assert matrix_error<2e-11
    vproject=paper_voltage(gs,dt,gl=float(native.gL[0]/native.C[0]),
        El=float(native.EL[0]*1000),Es=(float(native.EE[0]*1000),float(native.EI[0]*1000)),
        V0=float(native.V0[0]*1000))
    # SI native derivative and normalized recurrence have the same coefficients.
    for j in range(0,5000,1000):
        g=gs[j]*native.C[0]
        derivative=native.derivative(t[j],(np.array([vproject[j]/1000]),),
            dict(excitation=SimpleNamespace(values=np.array([g[0]])),
                 inhibition=SimpleNamespace(values=np.array([g[1]]))))[0].values[0]
        independent=(native.gL[0]/native.C[0])*(native.EL[0]-vproject[j]/1000)+gs[j]@(np.array([native.EE[0],native.EI[0]])-vproject[j]/1000)
        np.testing.assert_allclose(derivative,independent,rtol=1e-13,atol=1e-13)
    common_t=t[::10]
    project_V=vproject[:-1:10]/1000
    project_no_history=native.intensity(project_V,np.zeros((3,len(project_V))))
    project_expected_q=exact_mean_events(native,project_V)
    paper_lambda=softplus_map(vpaper[:-1])
    python_lambda=softplus_map(vpy)
    paper_q=-np.expm1(-dt*paper_lambda)
    # Static mappings compare the same V with history held at zero.
    voltages=np.linspace(-80.,-25.,1101)
    curve_paper=softplus_map(voltages)
    curve_mat_init=90.*np.logaddexp(0.,np.clip(.45*voltages+53*.45,-80.,80.))
    curve_project=native.intensity(voltages/1000,np.zeros((3,len(voltages))))
    mapping=np.column_stack((voltages,curve_paper,curve_mat_init,curve_project,
        -np.expm1(-curve_paper*dt)/dt,-np.expm1(-curve_project*native.dt)/native.dt))
    np.savetxt(args.output/'zero_history_mapping.csv',mapping,delimiter=',',
        header='V_mV,paper_2022_hazard_Hz,matlab2020_init_hazard_Hz,project_hazard_Hz,paper_binary_rate_dt0p1ms_Hz,project_binary_rate_dt1ms_Hz',comments='')
    np.savetxt(args.output/'voltage_versions.csv',np.column_stack((t+dt,gs,vpaper[1:],vmat,vpy,vproject[1:])),delimiter=',',
        header='interval_end_s,gE_over_C_per_s,gI_over_C_per_s,paper_V_mV,matlab_algorithm_paper_parameters_V_mV,python2022_literal_V_mV,project_V_mV',comments='')
    np.savetxt(args.output/'common_bin_start_comparison.csv',np.column_stack((common_t,
        vpaper[:-1:10],vproject[:-1:10],paper_lambda[::10],project_no_history,
        project_expected_q/native.dt)),delimiter=',',
        header='bin_start_s,paper_V_mV,project_V_mV,paper_zero_history_hazard_Hz,project_zero_history_hazard_Hz,project_original_history_expected_count_rate_Hz',comments='')
    np.savez_compressed(args.output/'reference_traces.npz',t_start_s=t,t_end_s=t+dt,gs_over_C_per_s=gs,
        paper_V_mV=vpaper,matlab_algorithm_V_mV=vmat,python_literal_V_mV=vpy,project_V_mV=vproject,
        paper_zero_history_hazard_Hz=paper_lambda,python_literal_zero_history_hazard_Hz=python_lambda,
        common_t_start_s=common_t,project_no_history_hazard_Hz=project_no_history,
        project_expected_count_rate_Hz=project_expected_q/native.dt,
        V_grid_mV=voltages,curve_paper_Hz=curve_paper,curve_matlab_init_Hz=curve_mat_init,curve_project_Hz=curve_project)
    anchors=[]
    for v in [-65.,-60.,-53.,-40.]:
        author=float(softplus_map(v));project=float(native.intensity(np.array([v/1000]),np.zeros((3,1)))[0])
        anchors.append(dict(V_mV=v,author_zero_history_hazard_Hz=author,project_zero_history_hazard_Hz=project,
            author_binary_rate_dt0p1ms_Hz=float(-np.expm1(-author*dt)/dt),
            project_binary_rate_dt1ms_Hz=float(-np.expm1(-project*native.dt)/native.dt)))
    phase_results=[]
    for j,(name,e,i) in enumerate(phases):
        sl=slice(j*1000,(j+1)*1000);ps=slice(j*100,(j+1)*100)
        phase_results.append(dict(phase=name,gE_over_C_per_s=e,gI_over_C_per_s=i,
            paper_mean_zero_history_hazard_Hz=float(paper_lambda[sl].mean()),
            paper_expected_binary_count_Hz=float(paper_q[sl].mean()/dt),
            project_mean_zero_history_hazard_Hz=float(project_no_history[ps].mean()),
            project_original_history_expected_count_Hz=float(project_expected_q[ps].mean()/native.dt)))
    # Preserve source discrepancy as an observation, not a production patch.
    discrepancy=dict(python_literal_first_leak_voltage_mV=float(vpy[0]),
        equation_and_matlab_first_leak_voltage_mV=float(vpaper[1]),
        first_bin_difference_mV=float(vpy[0]-vpaper[1]),
        maximum_absolute_version_difference_mV=float(np.max(abs(vpy-vpaper[1:]))),
        after_first_50ms_maximum_difference_mV=float(np.max(abs(vpy[500:]-vpaper[501:]))),
        source_initial_term='C[0] += V0, rather than decay[0]*V0 in the ZOH equation',
        source_lower_diagonal='At row j>0, previous interval decay[j-1], while the ZOH equation uses decay[j]',
        interpretation='Difference of the pinned NumPy-reexpressed source equations; not a directly executed JAX runtime finding',
        production_fix_applied=False)
    for rel,h in protected.items():assert sha(ROOT/rel)==h,rel
    result=dict(schema='on_parasol_author_reference_1.0',
        status='PARTIAL_AUTHOR_REFERENCE_REEXPRESSION_COMPLETED_NOT_EMPIRICAL_CALIBRATION',
        execution=dict(python=sys.executable,numpy=np.__version__,scipy=__import__('scipy').__version__,
            source_backend='NumPy float64 / SciPy lower-bidiagonal solve',
            jax_available=importlib.util.find_spec('jax') is not None,direct_JAX_author_entry_executed=False,
            matlab_executed=False,simulation_duration_seconds=.5,wall_seconds=time.perf_counter()-started),
        author_2022=dict(commit=tree['sha'],source_identity=identity,
            model=dict(El_mV=-60.,Ee_mV=0.,Ei_mV=-80.,gl_per_s=200.,alpha_Hz=90.,mu_mV=-53.,beta_mV=1.67,dt_s=dt),
            unavailable='Fitted stimulus filters, fitted spike-history kernel and actual V-spike observations'),
        matlab_2020=dict(commit='cfe41c4519fd56bd5aaaa81f156357ebae101d8f',
            voltage_scope='Published voltage algorithm reexpressed under paper membrane parameters; not the full setup+fitting entry',
            separate_init_nonlinearity=dict(c_Hz=90.,slope_per_mV=.45,mu_mV=-53.,equivalent_beta_mV=1/.45),
            direct_author_matlab_executed=False),
        project=dict(unchanged_profile='effective_parasol_RGC_ON',C_F=float(native.C[0]),gL_S=float(native.gL[0]),
            gL_over_C_per_s=float(native.gL[0]/native.C[0]),EL_mV=float(native.EL[0]*1000),
            EE_mV=float(native.EE[0]*1000),EI_mV=float(native.EI[0]*1000),alpha_Hz=float(native.alpha[0]),
            mu_mV=float(native.mu[0]*1000),beta_mV=float(native.beta[0]*1000),dt_s=native.dt,
            history_mV=(native.history_kernel*1000).tolist(),
            input_comparison='Same normalized g/C inputs; project physical S inputs are C_project times g/C. Author capacitance not identified'),
        input=dict(evidence_class='engineering_reference',kind='Synthetic conductance clamp, not a visual stimulus or author-fitted input',
            phases=phases,each_phase_s=.1,no_measured_data_used=True),
        history_scope='Author curves/dynamics have explicit zero history to isolate known source mapping; no fitted author history fabricated. Project original history shown separately by exact marginalization, initially empty',
        anchors=anchors,phase_comparison=phase_results,source_voltage_discrepancy=discrepancy,
        checks=dict(matlab_vs_equation_max_abs_mV=matrix_error,source_blob_identity=True,
            source_expression_indexing=True,literal_python_matrix_vs_independent_recurrence=True,
            project_SI_derivative_vs_normalized_coefficients=True,all_outputs_finite=True,
            protected_files_unchanged=len(protected)),
        not_executed=['Author JAX notebook and optimizer','Author synthetic MAT example fit: file not cached and JAX absent',
            'Fit or recovery of animal V-spike nonlinearity','Fitted author stimulus-to-conductance filters and history',
            'Rejected Cell6 fit','Production parameter adoption or full-network calibration'],
        permissions=dict(requested=0,denied=0,pending=[],uncertain=[]),
        production_parameters_changed=False,production_defaults_changed=False,GUI_changed=False,
        fit_performed=False,training_performed=False,new_network_simulation_seconds=0,
        source_review_sha256=sha(BASE/'rgc_voltage_hazard_source_audit_v1/primary_source_review.json'),
        tool_sha256=sha(Path(__file__)))
    for value in [vpaper,vmat,vpy,vproject,paper_lambda,project_expected_q,mapping]:assert np.isfinite(value).all()
    save(args.output/'result.json',result)
    print(json.dumps(dict(status=result['status'],anchors=anchors,discrepancy=discrepancy,
                         checks=result['checks'],wall_seconds=result['execution']['wall_seconds']),indent=2))


if __name__=='__main__':main()
