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


def test_reflection_from_circular_wall():
    """Test reflection_from_circular_wall()"""

    new_orientation = cs_supcalc.reflection_from_circular_wall(
        0, 1, np.pi / 2)
    assert new_orientation == np.pi * 3 / 2

    new_orientation = cs_supcalc.reflection_from_circular_wall(
        1, 0, np.pi)
    assert new_orientation == np.pi * 2

    # test very flat reflection angle
    orient = np.pi + np.pi / 6
    vec_i = [np.cos(orient), np.sin(orient)]
    # orientation inside the circle
    i_orientation = np.pi + np.arctan2(vec_i[1], vec_i[0])
    new_orientation = cs_supcalc.reflection_from_circular_wall(
        0, 1, orient)
    assert new_orientation == i_orientation


@pytest.mark.parametrize(
    "meter, prev_meter, prev_theta, taxis_dir, new_theta, new_taxis_dir",
    [
        # no change in meter, prev. taxis_dir is None
        # turn according to prev_theta and set it 1 if prev_theta 0
        (1, 1, 0, None, 1, None),
        (1, 1, 1, None, 1, None),
        (1, 1, -1, None, -1, None),

        # previous meter was larger
        # and prev_theta > 0
        (0, 1, 1, None, 0, -1),
        # prev_theta < 0
        (0, 1, -1, None, 0, 1),

        # previous meter was smaller
        # prev_theta > 0
        (1, 0, 1, None, 1, None),
        # prev_theta < 0
        (1, 0, -1, None, -1, None),

        # previous meter was smaller
        # prev_theta > 0
        (0.1, 0, 1, None, 0.1, None),
        # prev_theta < 0
        (0.1, 0, -1, None, -0.1, None),
    ]
)
def test_phototaxis(meter, prev_meter, prev_theta, taxis_dir, new_theta, new_taxis_dir):
    """Test phototaxis()"""
    phototaxis_theta_step = 1

    _new_theta, _new_taxis_dir = cs_supcalc.phototaxis(
        meter, prev_meter, prev_theta, taxis_dir, phototaxis_theta_step)

    assert new_theta == _new_theta
    assert new_taxis_dir == _new_taxis_dir
