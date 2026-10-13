"""Parameter interface units/shapes, not biological default values."""
SF_PARAMETER_SPECS = {
    'SF2025_tau': ('s', ()),
    'SF2025_kSens': ('mV/contrast-unit', ()),
    'SF2025_kSur': ('dimensionless', ()),
    'SF2025_kRect': ('Hz/mV', ()),
    'SF2025_actRest': ('Hz', (2,)),
    'SF2025_magRet': ('deg/mm', ()),
    'SF2025_radOptCoef': ('polynomial coefficients for radius_deg(e_deg), descending powers', (3,)),
    'SF2025_radHorCoef': ('polynomial coefficients for log(radius_deg)(e_deg), descending powers', (4,)),
    'SF2025_radGangCoef': ('polynomial coefficients for radius_deg(e_deg), descending powers', (4,)),
}

CHEN_PARAMETER_SPECS={
    'CHEN2024_sigma':('s^-1',()),'CHEN2024_phi':('s^-1',()),
    'CHEN2024_eta':('s^-2 (source effective normalization)',()),
    'CHEN2024_gdark':('uM',()),'CHEN2024_k':('pA/uM^3',()),
    'CHEN2024_n':('dimensionless',()),'CHEN2024_cdark':('uM',()),
    'CHEN2024_beta':('s^-1',()),'CHEN2024_m':('dimensionless',()),
    'CHEN2024_kGC':('uM',()),'CHEN2024_gamma':('s^-2/R* (source effective normalization)',()),
    'CHEN2024_timeStep':('s',())
}
