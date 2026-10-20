"""Checks for the isolated, explicitly partial source re-expression."""
import numpy as np
from tools.run_on_parasol_author_reference import (
    paper_voltage, matlab2020_voltage_algorithm, python2022_voltage_literal, softplus_map)


def test_paper_and_matlab_algorithm_match_constant_input_closed_form():
    dt=.0001
    for e,i in [(0.,0.),(40.,0.),(40.,40.)]:
        gs=np.tile([e,i],(300,1));total=200+e+i
        vinf=(-60*200-80*i)/total
        expected=vinf+(-60-vinf)*np.exp(-dt*total*np.arange(301))
        np.testing.assert_allclose(paper_voltage(gs,dt),expected,rtol=0,atol=1e-12)
        np.testing.assert_allclose(matlab2020_voltage_algorithm(gs,dt),expected[1:],rtol=0,atol=1e-12)


def test_literal_python_initial_term_is_preserved_and_explicit():
    dt=.0001;decay=np.exp(-dt*200)
    result=python2022_voltage_literal(np.zeros((300,2)),dt)
    expected=-60-60*(1-decay)*decay**np.arange(300)
    np.testing.assert_allclose(result,expected,rtol=0,atol=1e-12)
    assert abs(result[0]-(-61.18807960159469))<1e-12
    assert np.max(abs(result-paper_voltage(np.zeros((300,2)),dt)[1:]))>1.


def test_literal_python_uses_previous_interval_lower_diagonal():
    dt=.0001;gs=np.array([[0.,0.],[40.,0.],[40.,40.],[0.,0.]])
    total=200+gs.sum(axis=1);d=np.exp(-dt*total)
    b=(-60*200-80*gs[:,1])/total*(1-d)
    expected=[b[0]-60]
    for j in range(1,4):expected.append(b[j]+d[j-1]*expected[-1])
    np.testing.assert_allclose(python2022_voltage_literal(gs,dt),expected,rtol=0,atol=1e-12)
    np.testing.assert_allclose(matlab2020_voltage_algorithm(gs,dt),paper_voltage(gs,dt)[1:],rtol=0,atol=1e-12)


def test_matlab_initialization_shape_is_a_separate_reference():
    v=np.array([-80.,-65.,-60.,-53.,-40.,-25.])
    source_shape=90*np.logaddexp(0.,.45*v+53*.45)
    np.testing.assert_allclose(source_shape,softplus_map(v,beta=1/.45),rtol=1e-14,atol=1e-13)
    assert abs(source_shape[2]-softplus_map(v)[2])>2.
