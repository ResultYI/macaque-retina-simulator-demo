"""Source-bound adapter for new HC/BC artifacts into existing group mechanisms."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from retina_sim.contracts import ContractError, canonical_hash
from retina_sim.spatial_group_graph import digest
from retina_sim.spatial_group_model import (GroupModelAssembly,PopulationLayout,RouteLayout,
    StateBlock,POPULATIONS,BC,REQUIRED_ROUTE_PAIRS,recording_contract)
from retina_sim.local_patch_boundary import BoundedInnerDynamics
from retina_sim.group_native_physics import GroupNativePhysics
from tools.assemble_gin_hc2_bc1_groups_support_v2 import ROOT,OUT,FINE,NATIVE,OLD,read,bundle,select_members

GROUP=OUT/'executions/assemble_groups_01'

def population(name,arrays,meta):
    return PopulationLayout(name,**{k:arrays[k] for k in ('group_ids','counts','member_offsets',
        'member_global_rows','fine_to_group','centroid_mm')},coordinate_role=meta['coordinate_role'],
        source_table=meta['source_table'],source_keys=meta['source_keys'],
        parameter_source=meta['parameter_class_source'],unresolved_parameter_fields=tuple(meta['derived_parameter_fields']),
        mesh_geometry={k.removeprefix('geometry_'):v for k,v in arrays.items() if k.startswith('geometry_')},
        recording_auxiliary_fine_rows=arrays['recording_auxiliary_fine_rows'],
        recording_auxiliary_group_rows=arrays['recording_auxiliary_group_rows'])

class NewLocalAssembly(GroupModelAssembly):
    def __init__(self,group_background):
        self.root=GROUP;raw=read(GROUP/'GROUP_ARTIFACT.json');expected=raw.pop('artifact_sha256')
        if canonical_hash(raw)!=expected:raise ContractError('New local group artifact identity differs')
        for name,row in raw['files'].items():
            path=(GROUP/name).resolve()
            if not path.is_relative_to(GROUP) or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
                raise ContractError('New group array identity differs: '+name)
        self.manifest=deepcopy(raw);self.artifact_sha256=expected;self.repo_root=ROOT
        self.config=deepcopy(raw['local_patch']['configuration'])
        self.manifest['configuration']['source_run']=OUT.relative_to(ROOT).as_posix()
        self._arrays={'input/group_background_native_td.npy':np.asarray(group_background,dtype=np.float64)}
        self._matrices={};self.populations={}
        for name in POPULATIONS:
            values=bundle(GROUP/'populations'/name/'MEMBERSHIP.npz')
            self.populations[name]=population(name,values,raw['populations'][name])
            for key,value in values.items():
                self._arrays['populations/'+name+'/'+key.removeprefix('geometry_')+'.npy']=value
        self.parent_population_rows=bundle(GROUP/'PARENT_ROWS.npz')
        self.bounded_local=True;self.boundary_policy='bounded_background_halo_v2'
        boundary=bundle(GROUP/'GAP_BOUNDARY.npz')
        self.boundary_runtime=dict(boundary,identity=canonical_hash(dict(artifact=expected,boundary=boundary,policy=raw['boundary'])))
        self.runtime_factories=dict(inner=BoundedInnerDynamics,native=GroupNativePhysics)
        self.routes={}
        for r in raw['routes']:
            name=r['name'];pre=r['pre'];post=r['post'];shape=(self.populations[post].group_count,self.populations[pre].group_count)
            total=self.matrix('routes/'+name+'/sum_weights.npz');mean=self.matrix('routes/'+name+'/mean_weights.npz')
            difference=mean-sparse.diags(1/self.populations[post].counts)@total
            if (total.shape!=shape or mean.shape!=shape or np.any(total.data<0) or np.any(mean.data<0)
                    or not np.isfinite(total.data).all() or not np.isfinite(mean.data).all()
                    or (difference.nnz and np.max(np.abs(difference.data))>1e-12)):
                raise ContractError('Persisted member projection differs: '+name)
            contact=r.get('contact',{})
            self.routes[name]=RouteLayout(name,pre,post,r['kind'],shape,total.nnz,contact.get('component'),
                contact.get('groups',0),contact.get('fine_contacts',0),contact.get('coefficient_sum'),r['sum_semantics'],r['mean_semantics'])
            if contact:
                p=self.array('routes/'+name+'/contact_post_group.npy');q=self.array('routes/'+name+'/contact_pre_group.npy')
                coeff=self.matrix('routes/'+name+'/contact_coefficient_S.npz')
                if (p.shape!=(contact['groups'],) or q.shape!=p.shape or np.any(p<0) or np.any(p>=shape[0])
                        or np.any(q<0) or np.any(q>=shape[1]) or coeff.shape!=shape or np.any(coeff.data<0)):
                    raise ContractError('Persisted physical contact layout differs: '+name)
        if {(r.pre,r.post) for r in self.routes.values()}!=set(REQUIRED_ROUTE_PAIRS):
            raise ContractError('New local assembly lacks required routes')
        self.state_blocks=self._states();self.state_scalar_count=sum(b.count for b in self.state_blocks)
        self.probability_blocks=self._probability_blocks();self.probability_scalar_count=sum(b.count for b in self.probability_blocks)
        n=len(self.populations['cones'].recording_auxiliary_fine_rows)
        self.auxiliary_z1_blocks=(StateBlock('cones','selected_original_z1_auxiliary','z1_aux','vH_native_voltage','selected_original_cone_transmitter_drive',n,0,'uninitialized_selected_original_auxiliary'),)
        self.auxiliary_z1_scalar_count=n;self.recording_contract=recording_contract();self.recording_fields=self._recording_fields()
        self.mechanism_clocks=raw['mechanism_clocks'];self.unresolved_bindings=tuple((n,p.unresolved_parameter_fields) for n,p in self.populations.items());self.runtime_ready=False
        center=np.array([3.88,0.]);radius=self.config['patch_geometry']['core_radius_deg']
        self.core_population_rows={n:np.flatnonzero(np.linalg.norm(p.centroid_mm/.223-center,axis=1)<=radius) for n,p in self.populations.items()}
        self.target_population='parasol_RGC_OFF';self.target_parent_row=raw['target']['selected_parent_group_row']
        self.target_local_row=int(np.flatnonzero(self.parent_population_rows[self.target_population]==self.target_parent_row).item())
        old=GroupModelAssembly(ROOT/'reports/spatial_group_graph_v02_20261025_v1/artifact_r2')
        # This base object exposes complete source GEOMETRY for existing AC kernels;
        # its old matrices are never attached or used as new network dynamics.
        geometry=dict(old.populations)
        for name in BC:
            z=bundle(GROUP/'populations'/name/'PARENT_GEOMETRY.npz')
            geometry[name]=SimpleNamespace(group_count=len(z['counts']),counts=z['counts'],centroid_mm=z['centroid_mm'])
        self.base=SimpleNamespace(populations=geometry,manifest=dict(configuration=old.manifest['configuration']),
            artifact_sha256=canonical_hash(dict(old_geometry=old.artifact_sha256,new_BC=raw['source_artifacts']['new_BC'])))
        self.original_geometry=old

    def array(self,relative):
        if relative in self._arrays:return self._arrays[relative]
        if relative not in self.manifest['files']:raise ContractError('Unbound new local array: '+relative)
        value=np.load(self.root/relative,allow_pickle=False);self._arrays[relative]=value;return value

    def matrix(self,relative):
        if relative not in self._matrices:
            self._matrices[relative]=GroupModelAssembly.matrix(self,relative)
        return self._matrices[relative]

class NativeParent:
    """Complete finite native source; only Cone/HC geometry/operators are exposed."""
    def __init__(self,local,background):
        z=bundle(NATIVE/'MEMBERSHIP.npz');receipt=read(NATIVE/'NATIVE_ASSEMBLY_RESULT.json')
        old=local.original_geometry;original=old.populations['cones'];keep=z['global_cone_parent_group_rows']
        g={k:getattr(original,k) for k in ('counts','member_offsets','member_global_rows','fine_to_group')}
        counts,members,offsets,mapping=select_members(g,keep);aux=np.sort(members[offsets[:-1]])[:900]
        cone=replace(original,group_ids=original.group_ids[keep],counts=counts,member_offsets=offsets,
            member_global_rows=members,fine_to_group=mapping,centroid_mm=original.centroid_mm[keep],
            mesh_geometry={k:(v if k=='leaf_keys' else v[keep]) for k,v in original.mesh_geometry.items()},
            recording_auxiliary_fine_rows=aux,recording_auxiliary_group_rows=mapping[aux])
        hmeta=local.manifest['populations']['HC'];haux=np.sort(z['member_global_rows'][z['member_offsets'][:-1]])[:900]
        hdata=dict(z,group_ids=np.array(['HC:parent:'+str(i) for i in range(len(z['counts']))]),
            centroid_mm=z['centroid_deg']*.223,recording_auxiliary_fine_rows=haux,recording_auxiliary_group_rows=z['fine_to_group'][haux])
        self.populations={'cones':cone,'HC':population('HC',hdata,hmeta)}
        self.manifest=dict(populations=dict(cones=deepcopy(old.manifest['populations']['cones']),HC=deepcopy(hmeta)),
            hc_projection_contract=dict(schema='fine_typed_complete_rows_all_member_mean_v1'))
        self._arrays={'populations/cones/keys.npy':old.array('populations/cones/keys.npy')[keep],
            'populations/HC/keys.npy':z['keys'],'input/group_background_native_td.npy':np.asarray(background,dtype=np.float64),
            'hc_projection/HC_supported_fraction.npy':z['HC_supported_member_fraction'],
            'hc_projection/cones_supported_fraction.npy':z['cone_supported_member_fraction']}
        self._matrices={'routes/cones_to_HC/mean_weights.npz':sparse.load_npz(NATIVE/'A.npz'),
            'routes/HC_to_cones/mean_weights.npz':sparse.load_npz(NATIVE/'typed_B.npz')}
        self.artifact_sha256=canonical_hash(dict(receipt=receipt,background=background))
    def array(self,name):return self._arrays[name]
    def matrix(self,name):return self._matrices[name]

def protocol():
    from tools.integrated_far_context_v1 import protocol as existing
    return existing()

def background_directory():
    location=read(OUT/'BACKGROUND_LOCATION.json');path=ROOT/location['path']
    if not path.resolve().is_relative_to(OUT) or digest(path/'RESULT.json')!=location['result_sha256']:
        raise ContractError('Current background receipt identity differs')
    return path

def load():
    from retina_sim.group_time_storage import load_bundle
    from retina_sim.ac_vs_model import ACVSAssembly,load_config
    from retina_sim.h1_h2_effective import H1H2Assembly,verify_typed_source
    from retina_sim.hc_lateral_interface import context
    folder=background_directory();inputs=bundle(folder/'MOVIE_NATIVE_INPUTS.npz')
    local=NewLocalAssembly(inputs['native_background']);full=NativeParent(local,inputs['full_native_background'])
    labels,evidence=verify_typed_source(full,ROOT)
    ac=ACVSAssembly(local,load_config(ROOT/'configurations/ac_vs_functional_v1.json'))
    h=H1H2Assembly(ac,full,labels,evidence,engineering=True)
    seed=load_bundle(folder/'seed',expected_kind='source_construction_seed')['payload']
    local.boundary_runtime=load_bundle(folder/'inner_boundary',expected_kind='validation_evidence')['payload']
    h.native_boundary=seed['h1_h2_model']['boundary']
    assembly=context(h,seed,read(folder/'HC_POSITION_BINDING.json'),ROOT)
    if seed['h1_h2_model']['artifact']!=assembly.artifact_sha256:
        raise ContractError('New background does not belong to rebuilt lateral HC/BC context')
    assembly.parameter_binding_id=seed['lateral_hc_interface']['binding_id']
    return assembly,seed
