"""Read-only assay targets with source frames; never generated model coefficients."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,math
from .contracts import ContractError

@dataclass(frozen=True)
class RegionSpec:
    source_id: str
    source_label: str
    bounds: dict
    unit: str
    origin: str
    frame: str
    meridian: str
    mapping_id: str | None = None

    def contains(self,distance,*,unit,frame,origin,meridian='unspecified'):
        if (unit,frame,origin)!=(self.unit,self.frame,self.origin):
            raise ContractError('Source region requires its own recording frame/unit/origin; UI or optical RF conversion is not implicit')
        if self.meridian!='unspecified' and meridian!=self.meridian:
            raise ContractError('Source meridian mismatch')
        if not math.isfinite(distance) or distance<0:raise ContractError('Finite nonnegative source distance required')
        lo=self.bounds.get('lower');hi=self.bounds.get('upper')
        if lo is None and hi is None:
            raise ContractError('Estimated source location has no qualified numeric applicability interval; reference quotation only')
        return (lo is None or (distance>=lo if self.bounds.get('lower_inclusive',True) else distance>lo)) and (
            hi is None or (distance<=hi if self.bounds.get('upper_inclusive',False) else distance<hi))

class ObservationCatalog:
    def __init__(self,root):
        path=Path(root)/'configurations/regional_observations_v1.json'
        raw=path.read_bytes();self.data=json.loads(raw);self.sha256=hashlib.sha256(raw).hexdigest()
        if self.data.get('schema_version')!='1.0':raise ContractError('Unknown observation contract')
        self.records=self.data['records']
        for row in self.records:
            if row['evidence_class'] not in ('measurement','published_fit','author_assumption','project_approximation'):
                raise ContractError('Observation evidence class must use existing four classes')
            if row['usage']!='reference_only_observation_target':raise ContractError('Observation cannot silently become a runtime parameter')

    def select(self,*,source_id,target_type,output_port,metric,condition,distance,unit,frame,origin,
               meridian='unspecified',allow_unverified_reference=False):
        matches=[]
        for row in self.records:
            if (row['source_id'],row['target_type'],row['output_port'],row['metric'])!=(source_id,target_type,output_port,metric):continue
            region=RegionSpec(source_id,row['region']['source_label'],row['region']['bounds'],row['region']['unit'],
                row['region']['origin'],row['region']['frame'],row['region'].get('meridian','unspecified'),row['region'].get('mapping_id'))
            if not region.contains(distance,unit=unit,frame=frame,origin=origin,meridian=meridian):continue
            if condition!=row['condition']:continue
            if row['source_binding_status']!='primary_material_locally_verified' and not allow_unverified_reference:
                raise ContractError('Source quotation target is not locally primary-verified; explicit reference-only access required')
            matches.append(row)
        if not matches:raise ContractError('No source target for this component/type/recording-region/condition/port; no nearest region or contrast transfer')
        return matches

    def mechanism_parameters(self,*args,**kwargs):
        raise ContractError('STA/flash time-to-peak, adaptation endpoints and charge ratios are observations; a source-calibrated generative fit is required to produce coefficients')

class ConeObservationProfile(ObservationCatalog):
    """Current, IS voltage and gain-adaptation endpoints remain independent ports."""
    def select(self,**context):
        if not context['target_type'].endswith('_cone') or not context['output_port'].startswith('cone.'):
            raise ContractError('Cone observation requires a cone type and its own output port')
        return super().select(**context)

class MidgetOutputObservation(ObservationCatalog):
    """ON-midget spike STA target; no OFF, parasol or BC-kernel transfer."""
    def select(self,**context):
        if (context['target_type'],context['output_port'],context['metric'])!=('ON_midget_RGC','RGC.spikes','spike_STA_time_to_peak'):
            raise ContractError('Midget output observation only supports the qualified ON-midget spike STA endpoint')
        return super().select(**context)

class MidgetExcitationObservation(ObservationCatalog):
    """Composite light-to-RGC current endpoint, distinct from BC residual input."""
    def select(self,**context):
        if (context['target_type'],context['output_port'],context['metric'])!=('ON_midget_RGC','RGC.excitatory_current','excitation_linear_filter_time_to_peak'):
            raise ContractError('Midget excitation observation requires composite excitatory-current port, not BC kernel or spike output')
        return super().select(**context)
