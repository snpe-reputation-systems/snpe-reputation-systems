from ..snpe_reputation_systems.simulations.sum_a import sum_a_func

def test_sum_a():
    assert sum_a_func(2, 3) == 5
    assert sum_a_func(-1, 1) == 0
    assert sum_a_func(0, 0) == 0
    assert sum_a_func(10, -5) == 5

