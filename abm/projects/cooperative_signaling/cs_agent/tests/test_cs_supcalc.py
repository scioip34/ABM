import numpy as np

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
