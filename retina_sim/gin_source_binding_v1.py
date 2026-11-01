"""Source-only Schottdorf-Lee display conversion and explicit neural admission.

LMS values follow the archive README; they are not td, photons or Rstar/s.
No archive Python/notebook is executed. No retinal network is constructed.
"""
from collections.abc import Mapping
import numpy as np
from numpy.typing import NDArray


def source_display_lms(rgb: NDArray[np.uint8]) -> NDArray[np.float64]:
    if rgb.dtype!=np.uint8 or rgb.ndim<1 or rgb.shape[-1]!=3:
        raise ValueError('Decoded RGB uint8 data required, not BGR or normalized floats')
    values=rgb.astype(np.float64)/256.
    red=.01451+.9855*values[...,0]**2.3122
    green=.005123+.9949*values[...,1]**2.2752
    blue=.02612+.9739*values[...,2]**2.2818
    return np.stack((2.74*red+3.4*green+1.34*blue,
        1.21*(1.06*red+3.58*green+2.07*blue),.212*red+8.28*green+285*blue),axis=-1)


def admission_blockers(binding: Mapping[str,object]) -> list[str]:
    if binding.get('schema')!='gin-real-cell-source-binding-v1':raise ValueError('Unknown binding schema')
    required=('source_and_movie_pins_verified','source_clock_registered','source_RF_registration_accepted',
              'azimuth_contract_accepted','display_to_native_contract_closed','network_identity_bound',
              'compatible_background_bound','training_evaluation_split_frozen')
    missing=[name for name in required if binding.get(name) is not True]
    for field in ('source_cell_key','source_type','mapped_model_type','movie_sha256'):
        if not isinstance(binding.get(field),str) or not binding[field]:missing.append(field)
    ecc=binding.get('source_eccentricity_deg')
    if type(ecc) not in (int,float) or not np.isfinite(ecc) or ecc<0:missing.append('source_eccentricity_deg')
    return missing


def require_neural_binding(binding: Mapping[str,object]) -> None:
    missing=admission_blockers(binding)
    if missing:raise ValueError('NEURAL_FORWARD_NOT_ADMITTED: '+', '.join(missing))
