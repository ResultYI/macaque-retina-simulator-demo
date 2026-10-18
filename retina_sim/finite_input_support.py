"""Explicit empty chemical sums in one frozen finite project graph.

This permits no replacement driver, inferred contact or input clamp. It only
declares which existing AII chemical CSR rows are exactly empty. Electrophysiology
and missing biological partners remain uncalibrated.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from .contracts import ContractError,RunMode

EMPTY_SUM_POLICY='frozen_empty_chemical_sum_project_v1'
REJECT_EMPTY_POLICY='reject_empty_chemical_input_v1'
OPTION='finite_empty_chemical_inputs'


def declared_empty_rows(spec,component,port,totals,incoming,root,mode):
    """Return the exact allowed zero rows, rejecting stale/overbroad declarations."""
    declarations=spec.options.get(OPTION,{})
    allowed=np.zeros(len(totals),dtype=bool)
    if not declarations:return allowed
    if (type(declarations) is not dict or set(declarations)-set(component.inputs)
        or RunMode(mode)==RunMode.FULL_PHYSIOLOGY):
        raise ContractError('Finite empty chemical sums require an explicit project/reference input declaration')
    declaration=declarations.get(port)
    if declaration is None:return allowed
    if (spec.component_id!='AII_AC' or spec.mechanism!='ReceptorConductanceMembrane'
        or spec.options.get('cell_kind')!='AII' or port!='excitation'
        or component.inputs[port].unit!='S'):
        raise ContractError('Finite empty sum policy is restricted to the frozen AII chemical conductance input')
    keys={'policy','evidence_class','source_graph','source_route','cell_ids'}
    if (type(declaration) is not dict or set(declaration)!=keys
        or declaration['policy']!=EMPTY_SUM_POLICY or declaration['evidence_class']!='project_approximation'):
        raise ContractError('Unknown or incomplete finite empty chemical sum declaration')
    ids=tuple(component.inputs[port].cell_ids);declared=declaration['cell_ids']
    if type(declared) not in (list,tuple) or any(type(gid) is not str for gid in declared):
        raise ContractError('Finite empty chemical sum needs explicit ordered cell IDs')
    empty=tuple(gid for gid,value in zip(ids,totals) if value==0)
    if not empty or tuple(declared)!=empty:
        raise ContractError('Finite empty chemical sum IDs differ from the actual zero input rows')
    if not incoming or any((link.specification.source_component,link.specification.source_port)
                           !=('RBC_AII_AMPA','conductance') for link in incoming):
        raise ContractError('Finite empty chemical sum does not match the actual RBC/AII conductance route')
    support=sum((np.diff(link.matrix.indptr) for link in incoming),np.zeros(len(totals),dtype=np.int64))
    if np.any((np.asarray(totals)==0)&(support!=0)):
        raise ContractError('Finite empty chemical sums require structurally empty rows, not cancelling or zero-weight contacts')
    binding=declaration['source_graph'];route=declaration['source_route'];root=Path(root).resolve()
    if type(binding) is not dict or set(binding)!={'path','sha256'}:
        raise ContractError('Finite empty chemical sum requires its frozen graph identity')
    path=(root/binding['path']).resolve()
    if (not path.is_relative_to(root) or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']):
        raise ContractError('Finite empty chemical sum frozen graph identity changed')
    record=json.loads(path.read_bytes())
    matches=[d for d in record.get('routes',()) if d.get('route_id')=='RBC_AII_AMPA']
    if (record.get('schema')!='frozen_global_disk_circuit_v1' or len(matches)!=1
        or matches[0].get('kind')!='primary_chemical' or matches[0].get('pre')!='rod_BC'
        or matches[0].get('post')!='AII_AC' or matches[0].get('binding')!=route):
        raise ContractError('Finite empty chemical sum source route binding differs')
    allowed=np.asarray(totals)==0
    return allowed
