"""Bind equation-family execution to the frozen scalar SCC dependency graph."""
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib
import heapq

import numpy as np

from .contracts import ContractError, canonical_hash


def topological_order(names, edges):
    """Deterministic graph order, independent of family declaration order."""
    names=tuple(names)
    if len(names)!=len(set(names)) or any(not isinstance(n,str) or not n for n in names):
        raise ContractError('Distinct nonempty schedule family names required')
    children={n:set() for n in names};indegree={n:0 for n in names}
    for parent,child in edges:
        if parent not in children or child not in children or parent==child:
            raise ContractError('Schedule edge has an unknown family or self cycle')
        if child not in children[parent]:children[parent].add(child);indegree[child]+=1
    ready=[n for n in names if indegree[n]==0];heapq.heapify(ready);order=[]
    while ready:
        name=heapq.heappop(ready);order.append(name)
        for child in sorted(children[name]):
            indegree[child]-=1
            if indegree[child]==0:heapq.heappush(ready,child)
    if len(order)!=len(names):raise ContractError('Equation family quotient has a dependency cycle')
    return tuple(order)


@dataclass(frozen=True)
class FamilySpec:
    name: str
    writes: tuple
    allowed_dependencies: tuple
    # Equal row labels may be internally coupled; defaults are pointwise rows.
    row_groups: tuple = ()


@dataclass(frozen=True)
class BoundFamily:
    name: str
    writes: tuple
    parents: tuple
    parent_keys: tuple
    held_reads: tuple
    component_ids: np.ndarray


class _StageAccess(MutableMapping):
    def __init__(self, values, family):
        self.values=values;self.family=family
        self.reads=set(family.parent_keys)|set(family.writes)
        self.writes=set(family.writes)
        self.written=set()

    def __getitem__(self,key):
        if key not in self.reads:raise ContractError('Stage family read outside its bound dependencies: '+self.family.name)
        if key not in self.values:raise ContractError('Stage parent is not published: '+str(key))
        value=self.values[key]
        if isinstance(value,np.ndarray) and key not in self.writes:
            value=value.view();value.flags.writeable=False
        return value

    def __setitem__(self,key,value):
        if key not in self.writes or key in self.values:
            raise ContractError('Stage family write outside its exact owned outputs: '+self.family.name)
        self.values[key]=value;self.written.add(key)

    def __delitem__(self,key):raise ContractError('Published stage values cannot be deleted')
    def __iter__(self):return (k for k in self.values if k in self.reads)
    def __len__(self):return sum(k in self.reads for k in self.values)


@dataclass(frozen=True)
class BoundSchedule:
    order: tuple
    families: object
    edges: tuple
    plan_identity: str
    identity: str

    def execute(self, handlers, U, F, *, held_state=None):
        if U or F or set(handlers)!=set(self.families):
            raise ContractError('Fresh stage workspaces and every bound family handler required')
        for name in self.order:
            family=self.families[name]
            if any(k not in U or k not in F for k in family.parent_keys):
                raise ContractError('Scheduled parent stages/slopes are not published: '+name)
            if family.held_reads and (held_state is None or any(k not in held_state for k in family.held_reads)):
                raise ContractError('Scheduled held source is absent: '+name)
            versions={k:(getattr(U[k],'_version',None),getattr(F[k],'_version',None)) for k in family.parent_keys}
            u,f=_StageAccess(U,family),_StageAccess(F,family)
            handlers[name](u,f)
            if u.written!=set(family.writes) or f.written!=set(family.writes):
                raise ContractError('Stage family did not publish all declared states/slopes: '+name)
            for k,version in versions.items():
                if (getattr(U[k],'_version',None),getattr(F[k],'_version',None))!=version:
                    raise ContractError('Stage family mutated an upstream state/slope: '+name)
            for k in family.writes:
                for value in (U[k],F[k]):
                    if isinstance(value,np.ndarray):value.flags.writeable=False


def bind_scc_schedule(plan, families):
    """Validate exact ownership, allowed reads/coupling, and derive the quotient."""
    families=tuple(sorted(families,key=lambda family:family.name));names=tuple(f.name for f in families)
    topological_order(names,())
    keys=tuple(plan.full_keys);key_id={k:i for i,k in enumerate(keys)}
    if len(keys)!=len(key_id):raise ContractError('Duplicate state key in SCC schedule plan')
    sizes=np.array([plan.full_slices[k].stop-plan.full_slices[k].start for k in keys],dtype=np.int64)
    owner=np.full(plan.nstate,-1,dtype=np.int64)
    key_of_row=np.empty(plan.nstate,dtype=np.int64)
    row_group=np.empty(plan.nstate,dtype=np.int64)
    for i,k in enumerate(keys):key_of_row[plan.full_slices[k]]=i
    allowed=[];owned_keys={};group_bindings=[]
    for family_id,family in enumerate(families):
        if not family.writes or len(set(family.writes))!=len(family.writes):raise ContractError('Distinct nonempty family writes required')
        rules=set(family.allowed_dependencies)
        if any(t not in family.writes or t not in key_id or s not in key_id for t,s in rules):
            raise ContractError('Family dependency rule references an unowned target or unknown state')
        groups=dict(family.row_groups)
        if len(groups)!=len(family.row_groups) or set(groups)-set(family.writes):
            raise ContractError('Family row coupling labels reference an unowned state')
        bound_groups=[]
        for k in family.writes:
            if k not in key_id:raise ContractError('Family owns an unknown state key')
            section=plan.full_slices[k]
            if section.stop<=section.start:raise ContractError('Family state rows must be nonempty')
            if np.any(owner[section]>=0):raise ContractError('Equation family state ownership overlaps')
            owner[section]=family_id;owned_keys[k]=family.name
            label=np.asarray(groups.get(k,np.arange(sizes[key_id[k]],dtype=np.int64)))
            if label.dtype.kind not in 'iu' or label.shape!=(sizes[key_id[k]],):
                raise ContractError('Family row coupling labels differ from original state layout')
            row_group[section]=label
            bound_groups.append((k,np.array(label,dtype=np.int64,copy=True)))
        allowed.append(rules);group_bindings.append(tuple(bound_groups))
    active=np.asarray(plan.active_indices)
    if not np.array_equal(np.flatnonzero(owner>=0),active):
        raise ContractError('Equation families must cover every active state exactly, without held writes')
    offsets=np.asarray(plan.component_offsets);indices=np.asarray(plan.component_indices)
    if len(indices)!=len(active) or not np.array_equal(np.sort(indices),active) or len(offsets)<1 or offsets[0]!=0 or offsets[-1]!=len(indices) or np.any(np.diff(offsets)<=0):
        raise ContractError('Invalid complete SCC component membership')
    component_owner=owner[indices[offsets[:-1]]]
    if len(indices) and np.any(np.minimum.reduceat(owner[indices],offsets[:-1])!=np.maximum.reduceat(owner[indices],offsets[:-1])):
        raise ContractError('A true scalar SCC is split across equation families')
    dependency=plan.state_dependencies
    if dependency.shape!=(plan.nstate,plan.nstate):raise ContractError('SCC state dependency shape differs')
    parents=[set() for _ in families];held=[set() for _ in families];state_edges=set()
    for target in owned_keys:
        section=plan.full_slices[target];fid=owner[section.start]
        counts=np.diff(dependency.indptr[section.start:section.stop+1])
        source=dependency.indices[dependency.indptr[section.start]:dependency.indptr[section.stop]]
        target_rows=np.repeat(np.arange(section.start,section.stop,dtype=np.int64),counts)
        source_keys=key_of_row[source]
        for kid in np.unique(source_keys):
            source_key=keys[int(kid)]
            if (target,source_key) not in allowed[fid]:
                raise ContractError('Unregistered equation dependency: '+str((target,source_key)))
            other=owned_keys.get(source_key)
            if other is None:held[fid].add(source_key)
            elif other!=families[fid].name:parents[fid].add(source_key)
        internal=owner[source]==fid
        if np.any(row_group[target_rows[internal]]!=row_group[source[internal]]):
            raise ContractError('Unregistered within-family scalar coupling: '+families[fid].name)
        source_families=np.unique(owner[source]);source_families=source_families[source_families>=0]
        state_edges.update((int(p),int(fid)) for p in source_families if p!=fid)
    dag=plan.component_dependencies
    if dag.shape!=(len(component_owner),len(component_owner)):
        raise ContractError('SCC component dependency shape differs')
    rows=np.repeat(np.arange(len(component_owner)),np.diff(dag.indptr))
    parent=component_owner[rows];child=component_owner[dag.indices]
    crossing=parent!=child
    codes=np.unique(parent[crossing]*len(families)+child[crossing])
    edges={divmod(int(code),len(families)) for code in codes}
    if edges!=state_edges:raise ContractError('SCC family parent graph disagrees with actual state dependencies')
    named_edges=tuple(sorted((names[a],names[b]) for a,b in edges))
    order=topological_order(names,named_edges)
    bound={}
    for fid,family in enumerate(families):
        bound[family.name]=BoundFamily(family.name,tuple(family.writes),
            tuple(sorted(names[p] for p,c in edges if c==fid)),
            tuple(k for k in keys if k in parents[fid]),tuple(k for k in keys if k in held[fid]),
            np.frombuffer(np.asarray(np.flatnonzero(component_owner==fid),dtype=np.int64).tobytes(),dtype=np.int64))
    identity=canonical_hash(dict(schema='typed-scc-equation-family-schedule-1',plan=plan.identity,
        families=tuple((f.name,f.writes,tuple(sorted(allowed[i])),group_bindings[i]) for i,f in enumerate(families)),
        edges=named_edges,order=order,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return BoundSchedule(order,MappingProxyType(bound),named_edges,plan.identity,identity)
