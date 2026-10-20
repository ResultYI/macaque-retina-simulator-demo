import numpy as np
from tools.analyze_pro_rgc_batch import membrane_decomposition


def test_recorded_grid_current_decomposition_matches_constant_conductance_solution():
    cid='test';t=np.arange(501)*.001;C=60e-12;gl=2e-9;ge=.5e-9;gi=.2e-9
    EL=-.065;EE=0.;EI=-.08;V0=(gl*EL+ge*EE+gi*EI)/(gl+ge+gi)
    new_ge=ge+.003e-9;G=gl+new_ge+gi;Vinf=(gl*EL+new_ge*EE+gi*EI)/G
    new_v=Vinf+(V0-Vinf)*np.exp(-G*t/C)
    gray={'time':t,cid+'.voltage':np.full((1,501),V0),cid+'.gE':np.full((1,501),ge),cid+'.gI':np.full((1,501),gi)}
    stim={'time':t,cid+'.voltage':new_v[None,:],cid+'.gE':np.full((1,501),new_ge),cid+'.gI':gray[cid+'.gI']}
    r=membrane_decomposition(stim,gray,cid,dict(C=[C],gL=[gl],EE=[EE],EI=[EI]))
    assert r['reconstruction_max_cell_error_mV']<1e-12
    assert r['phases']['increment']['excitatory_drive_delta_pA']>0
    assert r['phases']['increment']['inhibitory_drive_delta_pA']==0
    assert r['phases']['increment']['signed_current_sum_over_absolute_sum']==1.
