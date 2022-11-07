import numpy as np
import pytest

from abm.projects.cooperative_signaling.cs_agent import cs_supcalc


def test_random_walk():
    """Test random_walk()"""
    # set random seed
    np.random.seed(42)

    dvel, dtheta = cs_supcalc.random_walk()

    assert dvel == 1.0
    assert dtheta == -0.0752759286915825


@pytest.mark.parametrize(
    "meter, prev_meter, prev_theta, taxis_dir, new_theta",
    [
        # no change in meter
        (1, 1, 0, None, 0),
        # previous meter was larger, and prev_theta > 0
        (0, 1, 1, None, -1),
        # previous meter was larger, and prev_theta < 0
        (0, 1, -1, None, 1),
        # previous meter was smaller, and prev_theta != 0
        (1, 0, 1, None, 0),
        # previous meter was smaller, and prev_theta != 0
        (1, 0, -1, None, 0),
        # previous meter was smaller, and prev_theta != 0
        (0.1, 0, 1, None, 0.9),
        # previous meter was smaller, and prev_theta != 0
        (0.1, 0, -1, None, -0.9),
    ]
)
def test_phototaxis(meter, prev_meter, prev_theta, taxis_dir, new_theta):
    """Test phototaxis()"""
    phototaxis_theta_step = 1

    theta = cs_supcalc.phototaxis(
        meter, prev_meter, prev_theta, taxis_dir, phototaxis_theta_step)

    assert new_theta == theta
