from typing import Deque, List, Optional, Union

import hypothesis
import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import composite, floats, integers, lists, none, text, tuples
from numpy import float64

from ..snpe_reputation_systems.simulations.simulator_class import (
    BaseSimulator,
    SingleRhoSimulator,
    DoubleRhoSimulator,
)


def get_simulator(
    simulator_type: str,
    review_prior=np.array([1, 1, 1, 1, 1]),
    tendency_to_rate=0.05,
    simulation_type="timeseries",
):
    """
    Returns a functional instance of the desired simulator class to use for the different
    tests for its methods.

    Although in most cases the default values set for `params` would work,
    the option to manually modify these has been considered in case such
    flexibility is necesary later in the testing design and implementation
    process.
    """

    params = {
        "review_prior": review_prior,
        "tendency_to_rate": tendency_to_rate,
        "simulation_type": simulation_type,
    }

    if simulator_type == "Base":
        return BaseSimulator(params)

    elif simulator_type == "SingleRho":
        sim_to_yield = SingleRhoSimulator(params)
        sim_to_yield.generate_simulation_parameters(
            50
        )  # 50 chosen as "num_simulations" because it is the max value allowed for n in the assitance method "_integer_and_array"
        return sim_to_yield

    elif simulator_type == "DoubleRho":
        sim_to_yield = DoubleRhoSimulator(params)
        sim_to_yield.generate_simulation_parameters(
            50
        )  # 50 chosen as "num_simulations" because it is the max value allowed for n in the assitance method "_integer_and_array"
        return sim_to_yield


# BaseSimulator Tests
#############################################


@settings(max_examples=10)
@given(
    arrays(int, 5, elements=integers(min_value=0, max_value=100)),
    arrays(
        int,
        shape=tuples(integers(1, 10)),
        elements=integers(min_value=0, max_value=100),
    ),
    text(min_size=3, max_size=15),
)
def test__init__(array_int5, array_not5, random_string):
    """
    Testing builder method by providing it with innapropriate paramerters
    according to the former "assert"cases provided for BaseSimulator
    in simulator_class.py
    """

    # Hypothesis rule so array_not5 cannot take the "correct" shape (5,)
    assume(array_not5.shape != (5,))

    # Testing correct cases

    assert isinstance(
        get_simulator(simulator_type="Base"),
        BaseSimulator,
    )

    assert isinstance(
        get_simulator(simulator_type="Base", review_prior=array_int5),
        BaseSimulator,
    )

    assert isinstance(
        get_simulator(simulator_type="Base", simulation_type="histogram"),
        BaseSimulator,
    )

    # Testing incorrect shape of "review_prior"

    with pytest.raises(
        ValueError,
        match="Prior Dirichlet distribution of simulated reviews needs to have 5 parameters",
    ):
        get_simulator(simulator_type="Base", review_prior=array_not5)

    # Testing incorrect values for "simulation type"

    with pytest.raises(
        ValueError, match="Can only simulate review histogram or timeseries"
    ):
        get_simulator(simulator_type="Base", simulation_type=random_string)


@settings(max_examples=10)
@given(
    arrays(
        dtype=int,
        shape=tuples(integers(1, 10)),
        elements=integers(min_value=0, max_value=100),
    ),
    arrays(dtype=int, shape=5, elements=integers(min_value=0, max_value=100)),
    arrays(int, 0),
)
def test_convolve_prior_with_existing_reviews(array_not5, array_int5, empty_arr):
    """
    Testing "convolve_prior_with_existing_reviews"
    according to the former "assert"cases provided for this
    BaseSimulator method in simulator_class.py
    """

    # Hypothesis rule so array_not5 cannot take the "correct" shape (5,)
    assume(array_not5.shape != (5,))

    # Instantiate simulator
    simulator = get_simulator(simulator_type="Base")

    # Testing correct cases
    result = simulator.convolve_prior_with_existing_reviews(array_int5)

    assert np.array_equal(result, np.ones(5) + array_int5)

    assert isinstance(result, np.ndarray)

    # Testing incorrect cases (1)
    with pytest.raises(ValueError):
        simulator.convolve_prior_with_existing_reviews(array_not5)

    # Testing  incorrect cases (2)
    with pytest.raises(ValueError):
        simulator.convolve_prior_with_existing_reviews(empty_arr)


def _gen_fake_existing_reviews(num_products: int, depth: int):
    """
    Assistant function for "test_simulate" method, generates a random array
    which is valid to be passed as the "existing_reviews" parameter. The number
    of reviews is fixed by the parameter "depth" while the number of products is
    be adjusted through the parameter "num_products". It returns an array of shape
    (num_products, depth, 5) where the first row of each product is [1, 1, 1, 1, 1]
    """

    # Initialize array with shape (num_products, time, 5)
    existing_reviews = np.zeros((num_products, depth, 5), dtype=int)

    # Fill array
    for i in range(num_products):
        # First row of each product
        existing_reviews[i, 0] = np.array([1, 1, 1, 1, 1])

        # Adding the subsequent lines with reviews being added randomly
        for j in range(1, depth):
            add_index = np.random.choice(5)
            existing_reviews[i, j] = existing_reviews[i, j - 1] + np.array(
                [1 if k == add_index else 0 for k in range(5)]
            )

    return list(existing_reviews)


@composite
def _integer_and_array(draw):
    """
    Function for composite hypothesis strategy.

    This is required as in the "simulate" method, num_reviews_per_simulation
    is expected to have a length equal to num_simulations.

    Accordingly, the function return the value for num_simulations and an appropriate
    num_reviews_per_simulation array
    """
    n = draw(integers(min_value=1, max_value=50))
    array = draw(arrays(int, n, elements=integers(min_value=1, max_value=50)))

    n_2 = n
    attempts = 0
    while n_2 == n and attempts < 100:
        n_2 = draw(integers(min_value=5, max_value=50))
        attempts += 1

    assume(n_2 != n)
    return (
        n,
        n_2,
        array,
    )  # num_simulations, num_simlations_2, num_reviews_per_simulation


@settings(max_examples=50)
@given(
    _integer_and_array(),
    integers(min_value=5, max_value=25),
)
def test_simulate_base(int_and_array, depth_existing_reviews):
    """
    Testing "simulate" method according to the former "assert"cases provided for this
    BaseSimulator method in simulator_class.py
    """

    (
        given_num_simulations,
        given_num_simulations_2,
        given_num_reviews_per_simulation,
    ) = int_and_array

    # Instantiate simulator
    simulator = get_simulator(simulator_type="Base")

    # If existing_reviews is not None:

    ## Expect ValueError if simulation_parameters is None
    with pytest.raises(ValueError):
        simulator.simulate(
            num_simulations=given_num_simulations,
            existing_reviews=_gen_fake_existing_reviews(
                given_num_simulations, depth_existing_reviews
            ),
        )

        ## Expect ValueError if num_reviews_per_simulation is None
    with pytest.raises(ValueError):
        simulator.simulate(
            num_simulations=given_num_simulations,
            existing_reviews=_gen_fake_existing_reviews(
                given_num_simulations, depth_existing_reviews
            ),
            simulation_parameters={},
        )

    # If all three (existing_reviews, num_reviews_per_simulation, simulation_parameters) exist:
    # code continues

    # If num_reviews_per_simulation exists:

    ## Expect ValueError if len(num_reviews_per_simulation) != num_simulations
    with pytest.raises(ValueError):  # Case 1: existing_reviesw == None
        simulator.simulate(
            num_simulations=given_num_simulations_2,
            simulation_parameters={},
            num_reviews_per_simulation=given_num_reviews_per_simulation,
        )

    with pytest.raises(ValueError):  # Case 2: existing_reviews != None
        simulator.simulate(
            num_simulations=given_num_simulations,
            existing_reviews=_gen_fake_existing_reviews(
                given_num_simulations_2,
                depth_existing_reviews,
            ),
            simulation_parameters={},
            num_reviews_per_simulation=given_num_reviews_per_simulation,
        )

    # If simulation_parameters is not None:

    ## Expect NotImplementedError if set(simulation_parameters) != set(dummy_parameters):
    ## This is a result of the method "generate_simulation_parameters" not being implemented still for BaseSimulator
    with pytest.raises(NotImplementedError):
        simulator.simulate(
            num_simulations=given_num_simulations,
            existing_reviews=_gen_fake_existing_reviews(
                given_num_simulations, depth_existing_reviews
            ),
            simulation_parameters={},
            num_reviews_per_simulation=given_num_reviews_per_simulation,
        )


# SingleRhoSimulator Tests
#############################################


@settings(max_examples=10)
@given(
    integers(min_value=1, max_value=5),
    floats(min_value=1, max_value=5),
    integers(min_value=6, max_value=10),
    floats(min_value=6, max_value=10),
)
def test_mismatch_calculator(
    experience,
    expected_experience,
    wrong_experience,
    wrong_expected_experience,
):
    simulator = get_simulator(simulator_type="SingleRho")

    # Testing correct cases
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


def _gen_wrong_fake_existing_reviews(num_products: int, depth: int):
    """
    Assistant function for "test_simulate_review_histogram" method. Works
    in the same way as "_gen_fake_existing_reviews" but producing histograms
    where it seems that two ratings have been added at once with the objecting of
    triggering a ValueError within "simulate_review_histogram".
    """

    # Initialize array with shape (num_products, time, 5)
    existing_reviews = np.zeros((num_products, depth, 5), dtype=int)

    # Fill array
    for i in range(num_products):
        # First row of each product
        existing_reviews[i, 0] = np.array([1, 1, 1, 1, 1])

        # Adding the subsequent lines with reviews being added randomly
        for j in range(1, depth):
            add_index = np.random.choice(5)
            existing_reviews[i, j] = existing_reviews[i, j - 1] + np.array(
                [2 if k == add_index else 0 for k in range(5)]
            )


@settings(max_examples=50)
@given(
    _integer_and_array(),
    integers(min_value=5, max_value=25),
    integers(min_value=1, max_value=100),
)
def test_simulate_review_histogram(
    int_and_array, depth_existing_reviews, simulation_id
):
    (
        given_num_simulations,
        _,
        _,
    ) = int_and_array

    assume(simulation_id < given_num_simulations)

    # Instantiate simulator
    simulator = get_simulator(simulator_type="SingleRho")

    # Testing correct case
    simulator.simulate_review_histogram(
        simulation_id=simulation_id,
        existing_reviews=[
            arr[1:]
            for arr in _gen_fake_existing_reviews(
                given_num_simulations, depth_existing_reviews
            )
        ],
    )

    # Testing incorrect case - existing reviews has a step different from 1
    with pytest.raises(ValueError):
        simulator.simulate_review_histogram(
            simulation_id=simulation_id,
            existing_reviews=[
                arr[1:]
                for arr in _gen_wrong_fake_existing_reviews(
                    given_num_simulations, depth_existing_reviews
                )
            ],
        )


@settings(max_examples=50)
@given(
    _integer_and_array(),
    integers(min_value=5, max_value=25),
)
def test_simulate_singlerho(int_and_array, depth_existing_reviews):
    """
    Testing "simulate" method according to the former "assert" cases provided for this
    BaseSimulator method in simulator_class.py
    """

    (
        given_num_simulations,
        given_num_simulations_2,
        given_num_reviews_per_simulation,
    ) = int_and_array

    # Instantiate simulator
    simulator = get_simulator(simulator_type="SingleRho")

    # If existing_reviews is not None:

    ## Expect ValueError if simulation_parameters is None
    with pytest.raises(ValueError):
        simulator.simulate(
            num_simulations=given_num_simulations,
            existing_reviews=_gen_fake_existing_reviews(
                given_num_simulations, depth_existing_reviews
            ),
        )

        ## Expect ValueError if num_reviews_per_simulation is None
    with pytest.raises(ValueError):
        simulator.simulate(
            num_simulations=given_num_simulations,
            existing_reviews=_gen_fake_existing_reviews(
                given_num_simulations, depth_existing_reviews
            ),
            simulation_parameters={},
        )

    # If all three (existing_reviews, num_reviews_per_simulation, simulation_parameters) exist:
    # code continues

    # If num_reviews_per_simulation exists:

    ## Expect ValueError if len(num_reviews_per_simulation) != num_simulations
    with pytest.raises(ValueError):  # Case 1: existing_reviesw == None
        simulator.simulate(
            num_simulations=given_num_simulations_2,
            simulation_parameters={},
            num_reviews_per_simulation=given_num_reviews_per_simulation,
        )

    with pytest.raises(ValueError):  # Case 2: existing_reviews != None
        simulator.simulate(
            num_simulations=given_num_simulations,
            existing_reviews=_gen_fake_existing_reviews(
                given_num_simulations_2,
                depth_existing_reviews,
            ),
            simulation_parameters={},
            num_reviews_per_simulation=given_num_reviews_per_simulation,
        )

    # If simulation_parameters is not None:

    ## Expect NotImplementedError if set(simulation_parameters) != set(dummy_parameters):
    with pytest.raises(KeyError):
        simulator.simulate(
            num_simulations=given_num_simulations,
            existing_reviews=_gen_fake_existing_reviews(
                given_num_simulations, depth_existing_reviews
            ),
            simulation_parameters={},
            num_reviews_per_simulation=given_num_reviews_per_simulation,
        )


# DoubleRhoSimulator Tests
#############################################

"""
@settings(max_examples=50)
@given(
    lists(floats(), min_size=2, max_size=2),
    arrays(dtype=np.float32, shape=integers(min_value=1)),
    floats(
        min_value=0, max_value=4
    ),  # Required by the tested method but not used in practice
    integers(
        min_value=1, max_value=100
    ),  # Required by the tested method but not used in practice
)
def test_decision_to_leave_review(
    mocker, wrong_rho_1, wrong_rho_2, delta, simulation_id
):
    # Instantiate simulator
    simulator = get_simulator(simulator_type="DoubleRho")

    assume(wrong_rho_2.shape[0] != 2)

    mocker.patch.object(
        simulator,
        "yield_simulation_param_per_visitor",
        side_effect=[wrong_rho_1, wrong_rho_2],
    )

    # Testing assert 1 (isinstance np.array)
    with pytest.raises(ValueError):
        simulator.decision_to_leave_review(delta=delta, simulation_id=simulation_id)

    # Testing assert 2 (shape != 2)
    with pytest.raises(ValueError):
        simulator.decision_to_leave_review(delta=delta, simulation_id=simulation_id)
"""
