import pytest
import numpy as np
import pandas as pd
from ..snpe_reputation_systems.simulations.simulator_class import BaseSimulator


@pytest.fixture
def generate_params():
    '''Generates a dictionary of parameters for the simulator class'''
    params = params = {
        "review_prior": np.ones(5),
        "tendency_to_rate": 0.05,
        "simulation_type": "timeseries"
    }

@pytest.fixture
def test_base_simulator():
    return BaseSimulator(generate_params())
    
def test_base_simulator_init(test_base_simulator):
    assert test_base_simulator(np.ones(5)) == np.array([2, 2, 2, 2, 2])
    assert test_base_simulator(np.ones(6)) == pytest.raises(AssertionError)
     
# Function does not seem robust to inclussion of np.nan in the input array
# Function does not seem robust to inclussion of negative values in the input array
# Function does not seem robust to inclussion of float values in the input array