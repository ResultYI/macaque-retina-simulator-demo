"""Read-only published Chen/Dryad records; never infer a runnable stimulus.

Reuses the existing archive identity constants and inspected MAT metadata.
Missing timing/history/current calibration remain explicit blockers.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.io import loadmat
from tools.ingest_chen2024_archive import EXPECTED
from .contracts import ContractError

GROUPS={'PrimateConeResponses':.0001,'PrimateRodResponses':.001}

@dataclass(frozen=True)
class PublishedChenRecord:
    sample_index_zero_based: np.ndarray
    relative_index_seconds: np.ndarray
    stimulus_Rstar_per_s: np.ndarray
    response_normalized: np.ndarray
    metadata: dict
    def require_model_comparison_ready(self):
        blockers=self.metadata['relative_model_comparison_blockers']
        if blockers:raise ContractError('Published record cannot be replayed without source decisions: '+'; '.join(blockers))

class ChenPublishedData:
    """Load one named primate MAT variable lazily; source arrays stay unaltered."""
    def __init__(self,root):
        self.root=Path(root);self.folder=self.root/'data/source/chen2024_dryad_v3_325663'
        self.assets={}
        for name,(size,digest) in EXPECTED.items():
            path=self.folder/name
            with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if path.stat().st_size!=size or actual!=digest:raise ContractError('Chen published asset identity mismatch: '+name)
            self.assets[name]={'path':str(path.relative_to(self.root)),'bytes':size,'sha256':actual}
        text=(self.folder/'README.md').read_text('utf8')
        self.sections={'PrimateConeResponses':text.split('**Primate and mouse cone responses**',1)[1].split('**Primate and mouse rod responses**',1)[0],
            'PrimateRodResponses':text.split('**Primate and mouse rod responses**',1)[1].split('### Files and variables',1)[0]}
        self.inventory=json.loads((self.root/'reports/chen2024_data_inventory.json').read_bytes())
        if not any(v.get('actual_sha256')==EXPECTED['Figure2-SourceData1.mat'][1] for v in self.inventory['files']):
            raise ContractError('Existing structure inventory is not bound to the current verified MAT')
        self._groups={}
    def _group(self,name):
        if name not in GROUPS:raise ContractError('Only explicitly documented primate cone/rod groups supported')
        if name not in self._groups:
            data=loadmat(self.folder/'Figure2-SourceData1.mat',variable_names=[name],struct_as_record=False,squeeze_me=False)
            group=data[name]
            declared=self.inventory['observed_cells_and_groups'][name]['shape']
            if group.dtype!=object or list(group.shape)!=declared:raise ContractError('Published MAT group shape/type differs')
            self._groups[name]=group
        return self._groups[name]
    def catalog(self,name):
        if name not in GROUPS:raise ContractError('Unsupported source group')
        section=self.sections[name];dt=GROUPS[name]
        if 'Stimulus units are R*/sec' not in section or '4-5 repeats' not in section or ('0.1 ms' if dt==.0001 else '1 ms') not in section:
            raise ContractError('README sampling/unit/repeat statement differs; no inferred replacement')
        records=[]
        for i,value in enumerate(self.inventory['observed_cells_and_groups'][name]['elements'],1):
            while 'elements' in value and len(value['elements'])==1:value=value['elements'][0]
            if set(value['field_names'])!={'AdaptingStepStim','AdaptingStepResponse'}:raise ContractError('Unknown protocol fields; no field-name guess')
            fields=value['fields'];shape=fields['AdaptingStepStim']['shape']
            records.append({'record_label':name+'{'+str(i)+'}','array_ordinal_1based':i,'biological_cell_id':None,
                'source_shape':shape,'samples':int(np.prod(shape)),'sampling_seconds':dt,'fields':fields})
        return {'dataset_doi':'10.5061/dryad.q2bvq83vg','dataset_version_id':325663,'assets':self.assets,
            'group':name,'records':records,'stimulus_unit':'R*/s','response_unit':'dimensionless',
            'response_normalization':'Signed published current / dark-current estimate from saturating flash; measured denominator absent',
            'repeats':'Published mean of4-5 repeats; exact per-record count and raw trials absent',
            'sampling_seconds':dt,'time_axis':'Derived relative sample-index seconds from README dt; not an experimental timestamp/onset',
            'protocol_readme':'Variable-mean noise; mean changes every500ms' if dt==.0001 else 'README lists step+flash family, dark flash family, and variable-mean noise every4s; MAT fields do not label those boundaries',
            'recording_method_readme':'voltage-clamp wording',
            'method_conflict':None if dt==.0001 else 'Supplied primary-paper review says rod suction electrodes >20deg; README voltage-clamp wording conflicts; per-record mapping unresolved',
            'missing_record_metadata':['species/cohort per array ordinal','precise eccentricity','biological cell IDs','per-cell dark-current pA','raw repeats','pre-stimulus history','experimental time/onset and phase','published interpolation/export preprocessing'],
            'data_usage':'parameterization source, not an independent holdout','model_or_parameter_changes':False}
    def read_record(self,name,index_1based=1,*,start_sample=0,stop_sample=None):
        catalog=self.catalog(name);group=self._group(name)
        if isinstance(index_1based,bool) or not isinstance(index_1based,int) or not 1<=index_1based<=group.size:raise ContractError('Explicit valid source array ordinal required')
        cell=group.flat[index_1based-1]
        while isinstance(cell,np.ndarray) and cell.dtype==object and cell.size==1:cell=cell.flat[0]
        if set(getattr(cell,'_fieldnames',()))!={'AdaptingStepStim','AdaptingStepResponse'}:raise ContractError('Published paired fields differ')
        stim=np.asarray(cell.AdaptingStepStim);response=np.asarray(cell.AdaptingStepResponse);record=catalog['records'][index_1based-1]
        if stim.dtype!=np.float64 or response.dtype!=np.float64 or list(stim.shape)!=record['source_shape'] or response.shape!=stim.shape:
            raise ContractError('Published paired array shape/dtype mismatch')
        n=stim.size;stop=n if stop_sample is None else stop_sample
        if any(isinstance(v,bool) or not isinstance(v,int) for v in (start_sample,stop)) or not 0<=start_sample<stop<=n:
            raise ContractError('Explicit increasing sample slice required')
        arrays=[np.arange(start_sample,stop,dtype=np.int64),np.arange(start_sample,stop,dtype=np.float64)*catalog['sampling_seconds'],
            stim.reshape(-1)[start_sample:stop].copy(),response.reshape(-1)[start_sample:stop].copy()]
        if not all(np.isfinite(v).all() for v in arrays):raise ContractError('Published selected samples are nonfinite')
        for value in arrays:value.setflags(write=False)
        blockers=['pre-stimulus state/history absent','experimental onset/phase not established']
        if record['fields']['AdaptingStepStim']['minimum']<0:blockers.append('nominal total R*/s contains negative values; no clipping/offset/contrast reinterpretation authorized')
        if name=='PrimateRodResponses':blockers.append('recorded1ms versus author-model0.1ms needs an explicit source-grounded hold/interpolation/alignment policy')
        metadata=dict(catalog,records=None,record=record,selection={'start_zero_based':start_sample,'stop_exclusive':stop},
            relative_model_comparison_blockers=blockers,absolute_current_calibration_blockers=['per-record measured dark-current denominator missing; normalized response is not pA'],
            relative_model_comparison_ready=False,absolute_current_calibration_ready=False,
            no_preprocessing='Exact source slice; no clipping, offset, resampling, normalization changes or time shifts')
        return PublishedChenRecord(*arrays,metadata)
