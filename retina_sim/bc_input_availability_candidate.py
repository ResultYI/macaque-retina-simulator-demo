"""Isolated OFF-input availability hypothesis, never a production receptor model.

The normalized release input is not molar glutamate or measured occupancy.
One availability state supplies partial desensitization upstream of the unchanged
OFF history/physical terminal. The caller must declare every kinetic coefficient.
"""
from pathlib import Path
import hashlib
import numpy as np
from .contracts import ContractError, canonical_hash


def normalized_puff_ratio(alpha, tau_recovery_seconds, duration_seconds):
    """D(T)/D(0) for the declared idealized u:0->1 step, initially D=1."""
    if alpha < 0 or tau_recovery_seconds <= 0 or duration_seconds < 0:
        raise ContractError('Invalid availability assay parameters')
    steady = 1.0/(1.0+alpha)
    return steady+(1.0-steady)*np.exp(-(1.0+alpha)*duration_seconds/tau_recovery_seconds)


class OFFInputAvailabilityCandidate:
    """tau*dD/dt=1-D-alpha*u*D, u=release/adapted_release, D in [0,1].

    r_effective=r*D/D0 preserves the predeclared native adapted background.
    This normalizer does not alter the downstream fixed reference baseline.
    """
    def __init__(self, background_release, *, alpha, tau_recovery_seconds, evidence):
        self.background=np.asarray(background_release,dtype=float).copy()
        if (not self.background.size or not np.isfinite(self.background).all()
                or np.any(self.background<=0)):
            raise ContractError('Explicit positive native background release required')
        if (not np.isfinite(alpha) or alpha<0 or not np.isfinite(tau_recovery_seconds)
                or tau_recovery_seconds<=0 or not evidence):
            raise ContractError('Explicit finite availability coefficients and evidence required')
        self.alpha=float(alpha)
        self.tau_recovery_seconds=float(tau_recovery_seconds)
        self.D0=1.0/(1.0+self.alpha)
        self.identity=canonical_hash(dict(mechanism='isolated_OFF_input_availability_1',
            background=self.background,alpha=self.alpha,tau_recovery_seconds=self.tau_recovery_seconds,
            evidence=evidence,native_release_is_molar=False,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    def normalized_drive(self,release):
        r=np.asarray(release,dtype=float)
        if r.shape!=self.background.shape or not np.isfinite(r).all() or np.any(r<0):
            raise ContractError('Native release must match background shape and be nonnegative')
        return r/self.background
    def initial(self):
        return np.full(self.background.shape,self.D0)
    def rhs(self,availability,release):
        d=np.asarray(availability,dtype=float)
        if d.shape!=self.background.shape or not np.isfinite(d).all():
            raise ContractError('Availability shape or finiteness differs')
        return (1.0-d-self.alpha*self.normalized_drive(release)*d)/self.tau_recovery_seconds
    def effective_drive(self,availability,release):
        self.normalized_drive(release)
        d=np.asarray(availability,dtype=float)
        if (d.shape!=self.background.shape or not np.isfinite(d).all()
                or np.any(d < -1e-10) or np.any(d > 1.0+1e-10)):
            raise ContractError('Availability outside its invariant probability domain')
        return np.asarray(release,dtype=float)*d/self.D0
