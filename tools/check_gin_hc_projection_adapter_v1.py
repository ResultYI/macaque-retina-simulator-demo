"""Targeted regression for already projected HC operator admission."""
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from retina_sim.h1_h2_effective import global_operators
from retina_sim.contracts import ContractError
from tools.durable_research_runtime_v1 import atomic_json

def main(output):
    start=time.perf_counter()
    a=sparse.csr_matrix([[.25,.25,0.],[.1,.1,.8],[0.,0.,0.]])
    b=sparse.csr_matrix([[.5,0.,0.],[0.,0.,0.],[0.,1.,0.]])
    matrices={'routes/cones_to_HC/mean_weights.npz':a,'routes/HC_to_cones/mean_weights.npz':b}
    fractions={'hc_projection/HC_supported_fraction.npy':np.array([.5,1.,0.]),
        'hc_projection/cones_supported_fraction.npy':np.array([.5,0.,1.])}
    model=SimpleNamespace(manifest={'hc_projection_contract':{'schema':'fine_typed_complete_rows_all_member_mean_v1'}},
        matrix=matrices.__getitem__,array=fractions.__getitem__)
    kinds={'cones':np.array(['L','M','S']),'HC':np.array(['H1','H2','H1'])}
    aa,bb,_,_=global_operators(model,kinds)
    if (aa!=a).nnz or (bb!=b).nnz:raise AssertionError('Already normalized fine/member weights changed')
    fractions['hc_projection/cones_supported_fraction.npy']=np.ones(3)
    try:global_operators(model,kinds)
    except ContractError:pass
    else:raise AssertionError('False support evidence admitted')
    atomic_json(output/'RESULT.json',dict(status='PASS',cases=['partial_and_zero_member_mass_retained','incorrect_support_fraction_rejected'],
        model_run=False,wall_seconds=time.perf_counter()-start))

if __name__=='__main__':main(Path(sys.argv[2]))
