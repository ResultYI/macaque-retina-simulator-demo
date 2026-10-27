"""Small deterministic time-only fitting decisions; forward execution stays external."""
import math
import numpy as np
from .contracts import ContractError
from .bc_ac_fullchain_candidate import signed_self_rms


def shape_with_error(waveform,absolute_error,window):
    """Propagate the frozen solver budget through own-RMS normalization."""
    x=np.asarray(waveform,dtype=np.float64);b=np.broadcast_to(absolute_error,x.shape)
    if np.any(b<0) or not np.isfinite(b).all():raise ContractError('Finite nonnegative original error bound required')
    normalized,rms=signed_self_rms(x,window);lo,hi=window
    rms_error=float(np.sqrt(np.mean(b[lo:hi]**2)))
    if rms<=rms_error:raise ContractError('Output RMS is not resolved above the numerical budget')
    error=b/(rms-rms_error)+np.abs(x)*rms_error/(rms*(rms-rms_error))
    return normalized,error


def sensitivity_report(baseline,candidates,step_log,window,minimum_singular_ratio):
    """Training-only response secants and numerical distinguishability, not a fit."""
    if set(candidates)!=set(step_log) or not candidates:raise ContractError('Exactly one perturbation per declared time candidate required')
    if not 0<minimum_singular_ratio<1:raise ContractError('Predeclared finite identifiability ratio required')
    columns=[];rows=[];lo,hi=window
    for key,changed in candidates.items():
        if set(changed)!=set(baseline) or step_log[key]<=0:raise ContractError('Matched training observable identity and positive log step required')
        column=[];largest=0.;squares=[]
        for observable,(x,b) in baseline.items():
            y,other=changed[observable];a,ea=shape_with_error(x,b,window);c,ec=shape_with_error(y,other,window)
            delta=(c-a)[lo:hi];uncertainty=(ea+ec)[lo:hi]
            ratio=np.abs(delta)/uncertainty
            largest=max(largest,float(np.max(ratio,initial=0.)))
            squares.append(float(np.mean(delta**2)))
            column.extend((delta/(step_log[key]*math.sqrt(len(delta)*len(baseline)))).tolist())
        columns.append(column);rows.append(dict(parameter=key,maximum_change_over_pair_numerical_bound=largest,
            resolved_above_numerical_budget=largest>1.,RMS_shape_change=math.sqrt(float(np.mean(squares)))))
    singular=np.linalg.svd(np.asarray(columns).T,compute_uv=False)
    ratio=float(singular[-1]/singular[0]) if singular[0]>0 else 0.
    all_resolved=all(r['resolved_above_numerical_budget'] for r in rows)
    return dict(parameters=rows,singular_values=singular.tolist(),singular_ratio=ratio,minimum_singular_ratio=minimum_singular_ratio,
        independent=ratio>=minimum_singular_ratio,all_parameters_resolved=all_resolved,
        joint_parameters_admissible=all_resolved and ratio>=minimum_singular_ratio,
        selected_parameters=[r['parameter'] for r in rows if r['resolved_above_numerical_budget']],
        validation_used=False,optimizer_steps=0,window=window)


def bounded_pattern_fit(evaluate,initial,bounds,*,initial_log_step,minimum_log_step,maximum_sweeps):
    """Train-only bounded coordinate search. Each uncached evaluation is a new forward.

    evaluate returns a finite scalar training loss and its numerical error bound.
    Validation is deliberately absent from this API.
    """
    keys=list(initial)
    if not 0<len(keys)<=2 or set(keys)!=set(bounds):raise ContractError('One or two declared time parameters required')
    if not 0<minimum_log_step<initial_log_step or type(maximum_sweeps) is not int or maximum_sweeps<1:raise ContractError('Frozen finite stopping rule required')
    for k in keys:
        lo,hi=bounds[k]
        if not 0<lo<=initial[k]<=hi or not lo<hi:raise ContractError('Positive bounded initial time required')
    calls=[];cache={}
    def value(point):
        identity=tuple(float(point[k]) for k in keys)
        if identity not in cache:
            result=evaluate(dict(point))
            if not np.isfinite(result['loss']) or not np.isfinite(result['numerical_error']) or result['numerical_error']<0:raise ContractError('Finite loss/error required')
            cache[identity]=dict(result);calls.append(dict(values=dict(point),**result))
        return cache[identity]
    current=dict(initial);best=value(current);step=initial_log_step;accepted=0;reason='MAXIMUM_FROZEN_SWEEPS'
    for _ in range(maximum_sweeps):
        improved=False
        for key in keys:
            options=[]
            for sign in (-1,1):
                candidate=dict(current);candidate[key]=min(bounds[key][1],max(bounds[key][0],current[key]*math.exp(sign*step)))
                if candidate==current:continue
                score=value(candidate);options.append((score,candidate))
            if options:
                score,point=min(options,key=lambda pair:pair[0]['loss'])
                if best['loss']-score['loss']>best['numerical_error']+score['numerical_error']:
                    current,best=point,score;accepted+=1;improved=True
        if not improved:step/=2
        if step<minimum_log_step:reason='NO_RESOLVED_IMPROVEMENT_AT_FROZEN_LOG_RESOLUTION';break
    return dict(values=current,training=best,evaluations=calls,accepted_steps=accepted,stopping_reason=reason,
        converged=reason!='MAXIMUM_FROZEN_SWEEPS',validation_used_for_selection=False)


def admissible_parameters(report,allowed):
    selected=report.get('selected_parameters',[])
    if not selected or len(set(selected))!=len(selected) or not set(selected)<=set(allowed):
        raise ContractError('No unique resolved allowed time parameter selected')
    rows={r['parameter']:r for r in report['parameters']}
    if any(not rows[k]['resolved_above_numerical_budget'] for k in selected):raise ContractError('Unresolved time parameter cannot be fitted')
    if len(selected)>1 and not report['independent']:raise ContractError('Dependent time parameters must not be jointly fitted')
    return selected
