"""Published-model H2 input weighting with the existing native exterior interface."""
from copy import copy, deepcopy
from pathlib import Path
import json

import numpy as np
from scipy import sparse

from .contracts import ContractError, canonical_hash
from .hc_eccentric import checked_path, digest
from .hc_group_projection import _normalize_complete_rows
from .cone_hc_eccentric import (EccentricConeHCModel, EccentricHCAssembly, EccentricHCNative,
                               EccentricHCFullChainRunner, _initialize_inner, MODEL)
from .hc_research_native import ConditionedH1H2Native
from .hc_timevarying_exterior import FixedFeedbackCone
from .h1_h2_effective import MECHANISM, matrix_id, csr
from .vh2005 import SOURCE_VALUES


DEFAULT_CONFIGURATION = 'configurations/research/cone_hc_h2_priority.json'


def complete_priority(raw, types, specification):
    """Retain the original contact coefficients and topology, including no-S rows."""
    if (specification.get('schema')!='H2_complete_contact_type_priority'
            or specification.get('gamma')!={'L':1.,'M':1.,'S':10.}
            or specification.get('evidence_class')!='published_model_borrowed'
            or specification.get('normalization')!='complete_legal_fine_source_row_before_crop_and_projection'
            or specification.get('preserve_raw_contact_values') is not True
            or specification.get('new_contacts_allowed') is not False):
        raise ContractError('Explicit borrowed H2 1/1/10 complete-source contract required')
    matrix=sparse.csr_matrix(raw,copy=True);matrix.eliminate_zeros();matrix.sort_indices()
    types=np.asarray(types)
    if (types.shape!=(matrix.shape[1],) or not set(types)<=set('LMS')
            or not np.isfinite(matrix.data).all() or np.any(matrix.data<0)):
        raise ContractError('Nonnegative original H2 contacts and actual Cone types required')
    weighted=matrix.multiply(np.array([specification['gamma'][t] for t in types])[None,:]).tocsr()
    result,supported=_normalize_complete_rows(weighted)
    result.sort_indices()
    if (not np.array_equal(result.indptr,matrix.indptr) or not np.array_equal(result.indices,matrix.indices)
            or not np.allclose(np.asarray(result.sum(1)).ravel(),supported,rtol=0,atol=2e-14)):
        raise ContractError('H2 weighting changed support or complete-row mass')
    return result


class PriorityNative(EccentricHCNative, ConditionedH1H2Native):
    """MRO reuses conditioned RHS, retained H2 coupling, and calibrated H1 scaling."""
    def __init__(self,assembly):
        super().__init__(assembly)
        self.exterior_cache=assembly.exterior_cache
        self.exterior_active=assembly.isolation_label=='S'
        self.exterior_mass=assembly.exterior_mass
        self.identity.update(H2_priority=assembly.priority_identity,
            isolation_label=assembly.isolation_label,
            exterior_cache=None if self.exterior_cache is None else self.exterior_cache.identity,
            exterior_mass=self.exterior_mass,
            priority_implementation_sha256=digest(Path(__file__)),
            conditioned_implementation_sha256=digest(Path(__file__).with_name('hc_research_native.py')))
        self.source_version=canonical_hash(self.identity)

    def boundary_at(self,t):
        base=self.boundary
        if not self.exterior_active:return base
        if self.exterior_cache is None:raise ContractError('S-isolation requires the exterior native cache')
        release,vbar=self.exterior_cache.at(t)
        return dict(A_release=base['A_release']+self.exterior_mass*(release-self.exterior_cache.background_release),
            A_Vbar=base['A_Vbar']+self.exterior_mass*(vbar-self.exterior_cache.y0[4]),B_h=base['B_h'])


class PriorityAssembly(EccentricHCAssembly):
    def native_factory(self,assembly,*,torch=None,device=None,plans=None,bound=True):
        if torch is not None or device is not None or plans is not None:
            raise ContractError('H2 priority requires the existing local CPU reference path')
        native=PriorityNative(assembly)
        if bound:native.boundary=self.native_boundary
        return native


class H2PriorityModel(EccentricConeHCModel):
    def __init__(self,repo_root,configuration=DEFAULT_CONFIGURATION):
        from .assembly import RetinaBuilder
        root=Path(repo_root).resolve();path=checked_path(root,configuration)
        cfg=json.loads(path.read_bytes())
        if (cfg.get('schema')!='cone_hc_h2_priority_research' or cfg.get('execution_mode')!='local_cpu'
                or cfg.get('research_only') is not True or cfg.get('production_qualified') is not False
                or cfg.get('exterior_interface')!='existing_FixedFeedbackCone_selected_two_S_members'):
            raise ContractError('Explicit H2 priority research configuration required')
        directory=root/cfg['assets_directory']
        receipt_path=checked_path(root,directory/'RECEIPT.json')
        receipt=json.loads(receipt_path.read_bytes())
        if receipt.get('status')!='PRECOMPUTED':raise ContractError('Completed H2 preparation required')
        for relative,expected in receipt['source_files'].items():checked_path(root,relative,expected)
        for name,expected in receipt['output_files'].items():checked_path(root,directory/name,expected)
        base=RetinaBuilder.build_research_cone_hc(root,cfg['base_configuration'])
        if base.identity!=receipt['uniform_model_identity']:raise ContractError('Uniform baseline identity differs')
        self.__dict__.update(base.__dict__)
        self.uniform_base=base;self.config=cfg;self.configuration_path=path
        self.priority_receipt=receipt;self.priority_directory=directory
        self.isolation_label=None
        self.A=sparse.load_npz(directory/'A_group.npz').tocsr()
        self.B=base.B
        with np.load(directory/'MASS_LEDGER.npz',allow_pickle=False) as z:
            self.priority_mass={k:z[k].copy() for k in z.files}
        with np.load(directory/'EXTERIOR_MEMBERS.npz',allow_pickle=False) as z:
            self.exterior_members={k:z[k].copy() for k in z.files}
        if (self.A[base.h1_rows]-base.A[base.h1_rows]).nnz:
            raise ContractError('H2 preparation changed H1 A')
        if matrix_id(self.A)!=receipt['A_identity'] or matrix_id(self.B)!=receipt['B_identity']:
            raise ContractError('Bound priority operators differ')
        self.mass=dict(base.mass,A_group_held=self.priority_mass['A_group_held'])
        combined=dict(base.spatial_receipt,spatial_identity=canonical_hash(dict(
            H1=base.spatial_receipt['spatial_identity'],H2=receipt['H2_weight_identity'])))
        self.assembly=PriorityAssembly(base.assembly.parent,self.A,self.B,base.native.Q['H2'],
                                      base.calibration,combined,cfg)
        self.assembly.priority_identity=receipt['H2_weight_identity']
        self.assembly.exterior_cache=None;self.assembly.isolation_label=None
        self.assembly.exterior_mass=self.priority_mass['selected_exterior_group_mass']
        self.exterior_feedback=receipt['exterior_feedback_native']
        ah=self.mass['A_group_held']
        release=receipt['exterior_release_background'];vbar=receipt['exterior_Vbar_background']
        self.assembly.native_boundary=dict(A_release=ah*release,A_Vbar=ah*vbar,
                                          B_h=base.native.boundary['B_h'].copy())
        self.native=self.assembly.native_factory(self.assembly)
        self.y0,self.background_diagnostics=self.native.equilibrate_native(base.y0,self.start_anchor,0.)
        self.background_rhs=float(np.max(abs(self.native.rhs(self.y0,0.,self.start_anchor))))
        if self.background_rhs>1e-8:raise ContractError('New H2 priority background exceeds existing RHS gate')
        seed=deepcopy(base.seed)
        for name in ('X','C','Vis','g_i','Vbar'):
            seed['states']['cones','reduced_native',name]=self.y0[self.native.slices[name]].copy()
        for name in ('w','z2'):
            seed['states']['HC',MECHANISM,name]=self.y0[self.native.slices[name]].copy()
        h=self.y0[self.native.slices['Vh']]
        for name,rows in (('h1',self.h1_rows),('h2',self.h2_rows)):
            seed['states']['HC',MECHANISM,name]=h[rows].copy()
        seed['states']['cones','selected_original_z1_auxiliary','z1_aux']=self.y0[self.native.slices['z1_aux']].copy()
        seed['observation_prehistory'].update(history_values=h[None,:].copy(),prehistory=h.copy())
        seed['h1_h2_model'].update(model=MODEL,artifact=self.assembly.artifact_sha256,
            layout_sha256=canonical_hash(self.assembly.layout),boundary=deepcopy(self.assembly.native_boundary),
            H2_priority=receipt['H2_weight_identity'],source_seed_sha256=canonical_hash(base.seed))
        self.seed=_initialize_inner(self.assembly,seed,root,self.protocol)
        if canonical_hash(self.seed['inner_parameters'])!=canonical_hash(base.seed['inner_parameters']):
            raise ContractError('H2 preparation changed downstream parameters')
        self.seed['hc_eccentric_binding'].update(reference_bc_identity=self.reference_bc_identity,
            source_bc_parameter_binding=base.seed['bc_physiology']['parameter_binding_id'],
            H2_priority=receipt['H2_weight_identity'])
        for frame in self.seed['native_inputs']:
            t=self.protocol['input_origin_seconds']+frame/self.protocol['input_fps']
            self.seed['native_inputs'][frame]=self.stimulus(t)
        self.identity=canonical_hash(dict(configuration=cfg,receipt=digest(receipt_path),
            native=self.native.source_version,seed=self.seed))

    def exterior_cache(self):
        return FixedFeedbackCone(self.exterior_feedback,member_identity=canonical_hash(self.exterior_members))

    def isolation(self,label,cache):
        if label not in ('L','M','S'):raise ContractError('Only the authorized L/M/S isolation conditions are supported')
        if (cache.feedback!=self.exterior_feedback
                or not np.isclose(cache.background_release,self.priority_receipt['exterior_release_background'],rtol=0,atol=1e-14)):
            raise ContractError('Exterior cache does not match the declared held background')
        model=copy(self);model.isolation_label=label
        model.assembly=object.__new__(type(self.assembly))
        model.assembly.__dict__.update(self.assembly.__dict__)
        model.assembly.exterior_cache=cache;model.assembly.isolation_label=label
        model.assembly.runtime_factories=dict(self.assembly.runtime_factories,native=model.assembly.native_factory)
        model.native=model.assembly.native_factory(model.assembly)
        model.seed=deepcopy(self.seed)
        for frame in model.seed['native_inputs']:
            t=model.protocol['input_origin_seconds']+frame/model.protocol['input_fps']
            model.seed['native_inputs'][frame]=model.stimulus(t)
        model.identity=canonical_hash(dict(model=self.identity,label=label,cache=cache.identity,seed=model.seed))
        return model

    def stimulus(self,time_seconds,cone_types=None):
        return super().stimulus(time_seconds,self.isolation_label if cone_types is None else cone_types)

    def runner(self,protocol=None,*,execution_mode='local_cpu'):
        if execution_mode!='local_cpu':raise ContractError('H2 priority supports local_cpu only')
        return EccentricHCFullChainRunner(self.assembly,self.seed,self.protocol if protocol is None else protocol)
