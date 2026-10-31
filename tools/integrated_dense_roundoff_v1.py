"""Research native dense output: retain the accepted RK increment before subtraction.

The solver steps, RHS calls, accepted state and error controller are unchanged.
Only F[0:3] use h*sum(B*K), rather than (rounded y_new)-y_old.
"""
import numpy as np
from scipy.integrate import DOP853


class IncrementDenseDOP853(DOP853):
    def _dense_output_impl(self):
        dense=super()._dense_output_impl()
        increment=self.h_previous*np.dot(self.K[:self.n_stages].T,self.B)
        dense.F[0]=increment
        dense.F[1]=self.h_previous*self.K[0]-increment
        dense.F[2]=2*increment-self.h_previous*(self.f+self.K[0])
        return dense
