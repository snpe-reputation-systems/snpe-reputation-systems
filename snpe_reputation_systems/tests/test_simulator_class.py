import pytest
import hypothesis
from hypothesis import given, settings, strategies as st
import numpy as np
import pandas as pd
from ..snpe_reputation_systems.simulations.simulator_class import BaseSimulator, SingleRhoSimulator


# class TestBaseSimulator
#############################################

@pytest.fixture
def yield_BaseSimulator():
    params = {
        "review_prior": np.ones(5),
        "tendency_to_rate": 0.05,
        "simulation_type": "timeseries"
    }
    return BaseSimulator(params)
    
def test_convolve_prior_with_existing_reviews(yield_BaseSimulator):
    
    # Test of correct sum
    assert np.array_equal(yield_BaseSimulator.convolve_prior_with_existing_reviews(np.ones(5)), np.array([2, 2, 2, 2, 2]))

    # Input shape test (Assertion error)
    with pytest.raises(AssertionError):
        yield_BaseSimulator.convolve_prior_with_existing_reviews(np.array([1, 2, 3, 4, 5, 6]))
        
    # Return type test
    result = yield_BaseSimulator.convolve_prior_with_existing_reviews(np.ones(5))
    assert isinstance(result, np.ndarray)

    #Null input test
    with pytest.raises(AttributeError): 
        yield_BaseSimulator.convolve_prior_with_existing_reviews(None)

def test_simulate():
    pass

def test_yield_simulation_param_per_visitor():
    pass

@pytest.fixture
def generate_mock_simulation_parameters(test_base_simulator):
    pass

def test_save_simulations():
    pass

def test_load_simulations():
    pass


# class TestSingleRhoSimulator: 
#############################################

@pytest.fixture
def yield_SingleRhoSimulator(self):

    params = {
        "review_prior": np.array([1, 1, 1, 1, 1]),
        "tendency_to_rate": 1.0,
        "simulation_type": "histogram",
    }
    return SingleRhoSimulator(params)

@settings(max_examples=50)
@given(delta=st.floats(min_value=-2, max_value=2), simulation_id=st.integers(min_value=0))

def test_rating_calculator(yield_SingleRhoSimulator, delta, simulation_id):

    result = yield_SingleRhoSimulator.rating_calculator(delta, simulation_id)

    # Return type test
    assert isinstance(result, int), "Result is not an integer"

    # Result range test
    assert 0 <= result <= 4, "Result is out of expected range"

    # Test of correct output
    if delta <= -1.5:
        assert result == 0
    elif delta > -1.5 and delta <= -0.5:
        assert result == 1
    elif delta > -0.5 and delta <= 0.5:
        assert result == 2
    elif delta > 0.5 and delta <= 1.5:
        assert result == 3
    else:
        assert result == 4

    
