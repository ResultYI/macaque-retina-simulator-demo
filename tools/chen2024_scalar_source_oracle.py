"""Independent scalar source-form oracle; does not call production step.

MIT attribution: chrischen2 historical BiophysModel.m 1bcf7c9a...
Only synthetic numerical state parity. MATLAB original supplies current oracle.
"""
import numpy as np

def scalar_source_oracle(rates,p):
    u=np.asarray(rates,dtype=float)
    if u.ndim!=1 or len(u)<2: raise ValueError('Oracle needs N sample rates')
    dc=p['gdark']**p['n']*p['k']
    g0=(dc/p['k'])**(1/p['n'])
    coefficient=p['beta']*p['cdark']/dc
    maximum=(p['eta']/p['phi'])*g0*(1+(p['cdark']/p['kGC'])**p['m'])
    trace=np.empty((5,len(u)),dtype=np.float64)
    trace[:,0]=[0,p['eta']/p['phi'],g0,p['cdark'],g0*p['eta']/p['phi']]
    for point in range(1,len(u)):
        old_r,old_p,old_g,old_c,old_s=trace[:,point-1]
        r=old_r+p['timeStep']*(-p['sigma']*old_r)
        r=r+p['gamma']*(u[point-1]*p['timeStep'])
        hydrolysis=old_p+p['timeStep']*(old_r+p['eta']-p['phi']*old_p)
        calcium=old_c+p['timeStep']*(coefficient*p['k']*old_g**p['n']-p['beta']*old_c)
        synthesis=maximum/(1+(calcium/p['kGC'])**p['m'])
        cgmp=old_g+p['timeStep']*(old_s-old_p*old_g)
        trace[:,point]=[r,hydrolysis,cgmp,calcium,synthesis]
    return trace
