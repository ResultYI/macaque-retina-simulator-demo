"""Identity isolation and truthful representative-state labels."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from retina_sim.computation_resolution import resolve,require_available,grouped_members
from retina_sim.contracts import ContractError

ROOT=Path(__file__).resolve().parents[1]

def test_scope_and_granularity_are_independent_namespaces_without_fallback():
    c=json.loads((ROOT/'configurations/computation_resolution_v1.json').read_text(encoding='utf8'))
    choices=[resolve(ROOT,c,s,g) for s in c['spatial_scope_options'] for g in c['granularity_options']]
    assert len({v['identity'] for v in choices})==4
    assert choices[0]['status']=='AVAILABLE'
    assert all(v['status']=='UNAVAILABLE' and not v['automatic_fallback'] for v in choices[1:])
    with pytest.raises(ContractError):require_available(choices[1])

def test_group_outputs_identify_original_members_without_inventing_cell_states():
    p=SimpleNamespace(group_ids=np.array(['g0','g1']),counts=np.array([2,1]),member_offsets=np.array([0,2,3]),member_global_rows=np.array([20,21,34]))
    result=grouped_members(p,[0,1]);assert result[0]['original_member_rows']==[20,21]
    assert result[1]['original_member_rows']==[34] and 'no independently solved' in result[0]['state_semantics']
    p.member_offsets=np.array([0,1,3])
    with pytest.raises(ContractError):grouped_members(p,[0])

def test_effective_ac_pool_has_target_not_fabricated_anatomical_members():
    p=SimpleNamespace(group_ids=np.array(['pool:sg2:target:1']),counts=np.ones(1,dtype=int),coordinate_role='effective_functional_RF_pool_not_anatomical_AC')
    result=grouped_members(p,[0])[0]
    assert result['paired_target_group_id']=='sg2:target:1'
    assert result['original_member_rows'] is None and result['member_count'] is None
