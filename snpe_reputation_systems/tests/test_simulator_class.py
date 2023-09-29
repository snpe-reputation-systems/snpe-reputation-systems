from typing import Deque, List, Optional, Union

import hypothesis
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, none
from numpy import float64

from ..snpe_reputation_systems.simulations.simulator_class import (
    BaseSimulator,
    DoubleRhoSimulator,
    HerdingSimulator,
    SingleRhoSimulator,
)

# class TestBaseSimulator
#############################################

# Assert translated:


def test_simulator_init_errors():
    params = {
        "review_prior": np.array([1, 1, 1, 1, 1, 1]),  # more than 5 parameters
        "tendency_to_rate": 1.0,
        "simulation_type": "histogram",
    }
    with pytest.raises(
        ValueError,
        match="Prior Dirichlet distribution of simulated reviews needs to have 5 parameters",
    ):
        BaseSimulator(params)

    params = {
        "review_prior": np.array([1, 1, 1, 1, 1]),
        "tendency_to_rate": 1.0,
        "simulation_type": "incorrect_type",  # incorrect simulation_type
    }
    with pytest.raises(
        ValueError, match="Can only simulate review histogram or timeseries"
    ):
        BaseSimulator(params)


# ChatGPT test suggestion (WIP):
# Returns an example dictionary to enable tests
def generate_simulation_parameters_stub(num_simulations: int) -> dict:
    return {"param1": 1}


def test_simulate_errors():
    params = {
        "review_prior": np.array([1, 2, 3, 4, 5]),
        "tendency_to_rate": 0.5,
        "simulation_type": "histogram",
    }
    simulator = BaseSimulator(params)
    simulator.generate_simulation_parameters = generate_simulation_parameters_stub

    # Test existing reviews without simulation parameters
    with pytest.raises(
        ValueError,
        match="Existing reviews for products supplied, but no simulation parameters given",
    ):
        simulator.simulate(5, existing_reviews=[np.array([1, 2, 3])])

    # Test existing reviews without num_reviews_per_simulation
    with pytest.raises(
        ValueError,
        match="Existing reviews for products supplied,but num_reviews_per_simulation not given",
    ):
        simulator.simulate(
            5, simulation_parameters={}, existing_reviews=[np.array([1, 2, 3])]
        )

    # Test mismatching num_reviews_per_simulation and num_simulations
    with pytest.raises(
        ValueError,
        match=r"\d+ simulations to be done, but \d+ review counts per simulation provided",
    ):
        simulator.simulate(5, num_reviews_per_simulation=np.array([1, 2, 3]))

    # Test incorrect simulation_parameters
    with pytest.raises(
        KeyError,
        match=r"Found parameters dict_keys\(\[.*\]\) in the provided parameters; expecteddict_keys\(\[.*\]\) as simulation parameters instead",
    ):
        simulator.simulate(
            5, simulation_parameters={"incorrect_key": "incorrect_value"}
        )


# Hypothesis:


@given(
    arrays(float64, 5, elements=floats(min_value=-100, max_value=100)),
    arrays(float64, 6, elements=floats(min_value=-100, max_value=100)),
    arrays(float64, 0),
    none(),
)
def test_convolve_prior_with_existing_reviews(arr1, arr2, empty_arr, none_value):
    # BaseSimulator instance
    params = {
        "review_prior": np.ones(5),
        "tendency_to_rate": 0.05,
        "simulation_type": "timeseries",
    }
    base_simulator = BaseSimulator(params)

    # Test of correct sum
    result = base_simulator.convolve_prior_with_existing_reviews(arr1)
    assert np.array_equal(result, np.ones(5) + arr1)

    # Input shape test
    with pytest.raises(ValueError):
        base_simulator.convolve_prior_with_existing_reviews(arr2)

    # Empty array input test
    with pytest.raises(ValueError):
        base_simulator.convolve_prior_with_existing_reviews(empty_arr)

    # Output type test
    assert isinstance(result, np.ndarray)

    # Null input test (note: it's different from empty array input test)
    with pytest.raises(AttributeError):
        base_simulator.convolve_prior_with_existing_reviews(none_value)


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

# Instantiate SingleRho:


def yield_SingleRhoSimulator():
    params = {
        "review_prior": np.array([1, 1, 1, 1, 1]),
        "tendency_to_rate": 1.0,
        "simulation_type": "histogram",
    }
    return SingleRhoSimulator(params)


# Assert translated:


@settings(max_examples=10)
@given(
    experience=st.integers(min_value=1, max_value=5),
    expected_experience=st.floats(min_value=1, max_value=5),
    wrong_experience=st.integers(min_value=6, max_value=10),
    wrong_expected_experience=st.floats(min_value=6, max_value=10),
)
def test_mismatch_calculator(
    experience, expected_experience, wrong_experience, wrong_expected_experience
):
    simulator = yield_SingleRhoSimulator()

    # Test of correct substraction
    assert simulator.mismatch_calculator(experience, expected_experience) == (
        experience - expected_experience
    )

    # Output type test
    assert isinstance(
        simulator.mismatch_calculator(experience, expected_experience), float
    )

    # out-of-range experience test
    with pytest.raises(ValueError):
        simulator.mismatch_calculator(wrong_experience, expected_experience)

    # out-of-range expected experience test
    with pytest.raises(ValueError):
        simulator.mismatch_calculator(experience, wrong_expected_experience)


# Hypothesis:


@settings(max_examples=20)
@given(
    delta=st.floats(min_value=-4, max_value=4), simulation_id=st.integers(min_value=0)
)
def test_rating_calculator(delta, simulation_id):
    simulator = yield_SingleRhoSimulator()
    result = simulator.rating_calculator(delta, simulation_id)

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


# NEW ASSERT INTO TEST (WIP):

# From simulator_class.py
# (inside SingleRhoSimulator class, simulate_review_histogram method):

#  if len(simulated_reviews) > 1:
#      if np.sum(simulated_reviews[-1]) - np.sum(simulated_reviews[-2]) != 1:
#          raise ValueError("""
#          Please check the histograms provided in the array of existing reviews. These should be in the form
#          of cumulative histograms and should only add 1 rating at a time.
#          """)

# let's try to implement this test here (needs debugging):

"""
def test_single_rho_simulator_histogram_error():
    params = {
        "review_prior": np.array([1, 1, 1, 1, 1]),
        "tendency_to_rate": 0.5,
        "simulation_type": "histogram",
        # other necessary parameters
    }

    simulator = SingleRhoSimulator(params)
    existing_reviews = [np.array([1, 1, 0, 0, 0]), np.array([2, 1, 0, 0, 0])]

    # Make sure the reviews are different
    assert not np.array_equal(existing_reviews[0], existing_reviews[1])

    with pytest.raises(ValueError, match="No change in reviews. Check the provided review data."):
        simulator.simulate_review_histogram(0, None, existing_reviews)"""


# class TestDoubleRhoSimulator:
#############################################


def test_double_rho_simulator_generate_simulation_parameters():
    num_simulations = 10
    params = DoubleRhoSimulator.generate_simulation_parameters(num_simulations)

    assert isinstance(params, dict), "Result is not a dictionary"
    assert "rho" in params, "Result does not contain rho"
    assert params["rho"].shape == (
        10,
        num_simulations,
        2,
    ), "Result does not have correct shape"


# class TestHerdingSimulator:
#############################################


def test_herding_simulator_generate_simulation_parameters():
    num_simulations = 10
    params = HerdingSimulator.generate_simulation_parameters(num_simulations)

    assert isinstance(params, dict)
    assert "rho" in params
    assert "h_p" in params
    assert params["rho"].shape == (10, num_simulations, 2)
    assert params["h_p"].shape == (10, num_simulations)
