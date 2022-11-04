import numpy as np

from abm.projects.cooperative_signaling.cs_agent import cs_supcalc


def test_random_walk():
    """Test random_walk()"""
    # set random seed
    np.random.seed(42)

    dvel, dtheta = cs_supcalc.random_walk()

    assert dvel == 1.0
    assert dtheta == -0.0752759286915825
