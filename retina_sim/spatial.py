"""Explicit populations and prescribed gaze; no density/displacement inference."""
from dataclasses import dataclass,field
import numpy as np
from .contracts import CellTable,Quantity,ContractError,StimulusSegment,canonical_hash
from .photometry import integrate_cone_photoisomerization_rate

REGIONS=('center','transition','periphery')

@dataclass(frozen=True)
class Population:
    population_id: str
    cells: CellTable
    regions: tuple
    anatomical_positions: Quantity | None = None
    receptive_field_positions: Quantity | None = None
    pedicle_positions: Quantity | None = None
    terminal_positions: Quantity | None = None
    dendritic_positions: Quantity | None = None
    _population_hash: str = field(init=False,repr=False,compare=False)

    def __post_init__(self):
        if not self.population_id or len(self.regions)!=len(self.cells.cell_id) or any(r not in REGIONS for r in self.regions):
            raise ContractError('Population requires explicit per-cell center/transition/periphery membership')
        for q,meaning,frame in ((self.anatomical_positions,'anatomical_position','anatomical_retina'),(self.receptive_field_positions,'functional_position','functional_visual_field')):
            if q is not None:
                if q.unit not in ('mm','deg'): raise ContractError('Explicit spatial units required')
                q.require(q.unit,meaning,(len(self.cells.cell_id),2),frame)
        terminal_frame='OPL' if set(self.cells.cell_type)<= {'rod','cone'} else 'IPL'
        for q,meaning,frame in ((self.pedicle_positions,'pedicle_position','OPL'),(self.terminal_positions,'terminal_position',terminal_frame),(self.dendritic_positions,'dendritic_position',None)):
            if q is not None:
                if q.unit not in ('mm','m'): raise ContractError('Anatomical synaptic coordinates require physical units')
                if q.values.shape not in ((len(self.cells.cell_id),2),(len(self.cells.cell_id),3)): raise ContractError('Synaptic coordinate shape')
                q.require(q.unit,meaning,frame=frame)
                if frame is None and q.coordinate_frame not in ('OPL','IPL'): raise ContractError('Explicit OPL/IPL dendritic coordinate frame required')
        object.__setattr__(self,'_population_hash',canonical_hash({name:getattr(self,name) for name in ('population_id','cells','regions','anatomical_positions','receptive_field_positions','pedicle_positions','terminal_positions','dendritic_positions')}))

    def record(self,selected_cell_ids=None):
        indices=np.arange(len(self.cells.cell_id)) if selected_cell_ids is None else np.array([i for i,x in enumerate(self.cells.cell_id) if x in selected_cell_ids],dtype=int)
        def selected(values):return [values[i] for i in indices]
        def qrecord(q):
            return None if q is None else {'values':q.values[indices].tolist(),'unit':q.unit,'meaning':q.meaning,'frame':q.coordinate_frame,'evidence_tag':q.evidence_tag}
        c=self.cells
        return {'population_id':self.population_id,'total_cell_count':len(c.cell_id),'population_hash':self._population_hash,'cell_ids':selected(c.cell_id),'cell_types':selected(c.cell_type),'subtypes':selected(c.subtype),'regions':selected(self.regions),'species_profile_ids':selected(c.species_profile_id),'parameter_profile_ids':selected(c.parameter_profile_id),'evidence_tags':selected(c.evidence_tags),'hemiretina':selected(c.hemiretina),'meridian':selected(c.meridian),'eccentricity':qrecord(c.eccentricity),'table_soma':qrecord(c.soma_position),'table_rf':qrecord(c.rf_position),'anatomical_positions':qrecord(self.anatomical_positions),'receptive_field_positions':qrecord(self.receptive_field_positions),'pedicle_positions':qrecord(self.pedicle_positions),'terminal_positions':qrecord(self.terminal_positions),'dendritic_positions':qrecord(self.dendritic_positions),'displacement_mapping':'not inferred'}

@dataclass(frozen=True)
class InputChannel:
    start_seconds: float
    stop_seconds: float
    evaluator: object
    unit: str
    meaning: str
    interpolation: str
    evidence_tag: str
    input_contract: dict
    breakpoints_seconds: tuple = ()
    def __post_init__(self):
        if not np.isfinite(self.start_seconds+self.stop_seconds) or self.stop_seconds<=self.start_seconds or not callable(self.evaluator) or self.interpolation not in ('analytic','linear','zero_order_hold') or not self.unit or not self.meaning or not self.evidence_tag: raise ContractError('Explicit input channel contract required')
        if tuple(sorted(set(self.breakpoints_seconds)))!=self.breakpoints_seconds or any(t<=self.start_seconds or t>=self.stop_seconds for t in self.breakpoints_seconds): raise ContractError('Channel breakpoints must be sorted inside interval')
    def at(self,t):
        if t<self.start_seconds-1e-12 or t>self.stop_seconds+1e-12: raise ContractError('Input evaluated outside interval')
        q=self.evaluator(t);q.require(self.unit,self.meaning)
        if q.evidence_tag!=self.evidence_tag: raise ContractError('Input evidence mismatch')
        return q

@dataclass(frozen=True)
class GazeTrajectory:
    times: Quantity
    positions: Quantity
    interpolation: str
    source_id: str

    def __post_init__(self):
        t=self.times.require('s','absolute_time')
        self.positions.require('deg','prescribed_gaze_position',(len(t),2),'world_visual_field')
        if t.ndim!=1 or len(t)<2 or np.any(np.diff(t)<=0) or self.interpolation not in ('linear','zero_order_hold') or not self.source_id:
            raise ContractError('Prescribed gaze requires increasing times, explicit interpolation and source')
        if self.times.evidence_tag!=self.positions.evidence_tag: raise ContractError('Gaze evidence mismatch')

    def at(self,t):
        x=self.times.values
        if not np.isfinite(t) or t<x[0] or t>x[-1]: raise ContractError('Gaze extrapolation forbidden')
        if self.interpolation=='linear': p=np.array([np.interp(t,x,self.positions.values[:,k]) for k in range(2)])
        else: p=self.positions.values[min(np.searchsorted(x,t,side='right')-1,len(x)-1)]
        return Quantity(p,'deg','prescribed_gaze_position',self.positions.evidence_tag,'world_visual_field')

def gaze_driven_channel(population,gaze,mapper,*,start,stop,unit,meaning,interpolation,evidence_tag,mapper_source_id,breakpoints=()):
    """Mapper owns sourced optics/coordinate transforms; gaze never predicts a fixation."""
    if not callable(mapper) or not mapper_source_id: raise ContractError('Explicit sourced optical/input mapper required')
    if start<gaze.times.values[0] or stop>gaze.times.values[-1]: raise ContractError('Input outside prescribed gaze history')
    breaks=set(breakpoints)
    if gaze.interpolation=='zero_order_hold': breaks.update(float(t) for t in gaze.times.values if start<t<stop)
    def evaluate(t):
        q=mapper(t,gaze.at(t),population)
        q.require(unit,meaning,(len(population.cells.cell_id),))
        if q.evidence_tag!=evidence_tag: raise ContractError('Optical/input mapper evidence mismatch')
        return q
    return InputChannel(start,stop,evaluate,unit,meaning,interpolation,evidence_tag,{'mapper_source_id':mapper_source_id,'gaze_source_id':gaze.source_id,'gaze_hash':canonical_hash(gaze),'population_id':population.population_id,'coordinate_mapping':'explicit mapper; no inferred anatomical/RF displacement'},tuple(sorted(breaks)))

def calibrated_gaze_cone_channel(population,gaze,irradiance_mapper,wavelengths,quadrature,aperture,efficiency,**kwargs):
    """Reuse the calibrated spectral integral; no RGB or pupil default is invented."""
    def mapper(t,g,p):
        E=irradiance_mapper(t,g,p)
        return integrate_cone_photoisomerization_rate(E,wavelengths,quadrature,aperture,efficiency)
    return gaze_driven_channel(population,gaze,mapper,unit='R*/s',meaning='photoisomerization_rate',**kwargs)
