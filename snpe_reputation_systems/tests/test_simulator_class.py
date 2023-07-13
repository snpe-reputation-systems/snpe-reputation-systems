import pytest
import numpy as np
import pandas as pd
from ..snpe_reputation_systems.simulations.simulator_class import BaseSimulator


@pytest.fixture
def test_base_simulator():
    params = {
        "review_prior": np.ones(5),
        "tendency_to_rate": 0.05,
        "simulation_type": "timeseries"
    }
    return BaseSimulator(params)
    
def test_convolve_prior_with_existing_reviews(test_base_simulator):
    
    assert np.array_equal(test_base_simulator.convolve_prior_with_existing_reviews(np.ones(5)), np.array([2, 2, 2, 2, 2]))

    with pytest.raises(AssertionError):
        test_base_simulator.convolve_prior_with_existing_reviews(np.ones(6))

     
# Function does not seem robust to inclussion of np.nan in the input array
# Function does not seem robust to inclussion of negative values in the input array
# Function does not seem robust to inclussion of float values in the input array
# Add xfail statements
