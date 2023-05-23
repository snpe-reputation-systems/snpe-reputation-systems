import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sum_a

def test_sum_a():
    assert sum_a(2, 3) == 5
    assert sum_a(-1, 1) == 0
    assert sum_a(0, 0) == 0
    assert sum_a(10, -5) == 5

