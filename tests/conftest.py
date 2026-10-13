from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from retina_sim.contracts import Quantity,RunMode
from retina_sim.provenance import Registry
from retina_sim.reference import build_somaratna_weights,SomaratnaFreeman2025GaussianHCReference

ROOT=Path(__file__).resolve().parents[1]
NUMERIC_METRICS={}
def q(value,unit,meaning,frame=None): return Quantity(value,unit,meaning,'synthetic_fixture',frame)

@pytest.fixture
def registry(): return Registry.read(ROOT/'sources/registry.json')

@pytest.fixture
def parameters(registry): return registry.load_parameters('SomaratnaFreeman2025GaussianHCReference_source_specific',RunMode.NUMERIC_FIXTURE)

@pytest.fixture
def model(parameters):
    # Invented numerical coordinates; these are NOT animal data or author arrays.
    cones=q([[-.01,0],[0,0],[.008,.005],[.02,-.01]],'deg','functional_position','local_patch')
    gang=q([[0,0],[.01,.008]],'deg','functional_position','local_patch')
    e=q(5,'deg','global_eccentricity','functional_visual_field')
    graph=build_somaratna_weights(cones,('L','M','S','L'),gang,e,parameters,'synthetic_fixture')
    return SomaratnaFreeman2025GaussianHCReference(parameters,graph,explicit_opt_in=True)

def pytest_sessionfinish(session,exitstatus):
    # In-memory metrics are consumed by run_checks.py via an explicit report path.
    import os,json
    path=os.environ.get('RETINA_NUMERIC_REPORT')
    if path:
        Path(path).write_text(json.dumps({'fixture_class':'synthetic_fixture','exit_code':exitstatus,'metrics':NUMERIC_METRICS},indent=2)+'\n')
