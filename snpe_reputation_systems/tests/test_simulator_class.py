import pytest
import hypothesis
import numpy as np
import pandas as pd
from ..snpe_reputation_systems.simulations.simulator_class import BaseSimulator


# Tests for BaseSimulator class methods

@pytest.fixture
def test_base_simulator():
    params = {
        "review_prior": np.ones(5),
        "tendency_to_rate": 0.05,
        "simulation_type": "timeseries"
    }
    return BaseSimulator(params)
    
def test_convolve_prior_with_existing_reviews():
    
    # Test of correct sum
    assert np.array_equal(test_base_simulator.convolve_prior_with_existing_reviews(np.ones(5)), np.array([2, 2, 2, 2, 2]))

    # Input shape test (Assertion error)
    with pytest.raises(AssertionError):
        test_base_simulator.convolve_prior_with_existing_reviews(np.array([1, 2, 3, 4, 5, 6]))
        
    # Return type test
    result = test_base_simulator.convolve_prior_with_existing_reviews(np.ones(5))
    assert isinstance(result, np.ndarray)

    #Null input test
    with pytest.raises(AttributeError): 
        test_base_simulator.convolve_prior_with_existing_reviews(None)

def test_simulate():
    pass

def test_yield_simulation_param_per_visitor():
    pass

def test_save_simulations():
    pass

def test_load_simulations():
    pass

    
