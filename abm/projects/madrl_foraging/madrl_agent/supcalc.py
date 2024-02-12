"""
calc.py : Supplementary methods and calculations necessary for agents
"""
import numpy as np
from abm.projects.madrl_foraging.madrl_contrib import madrl_movement_params as movement_params


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if v1_u[0] * v2_u[1] - v1_u[1] * v2_u[0] < 0:
        angle = -angle
    return angle


# Random Walk functions
def random_walk(desired_vel=None):
    """Pooling a small orientation and absolute velocity increment from some distribution"""
    if desired_vel is None:
        desired_vel = movement_params.exp_vel_max
    # dvel = np.random.uniform(movement_params.exp_vel_min,
    #                          movement_params.exp_vel_max)
    dvel = desired_vel
    dtheta = np.random.uniform(movement_params.exp_theta_min,
                               movement_params.exp_theta_max)
    return dvel, dtheta

def distance_torus(p0, p1, dimensions):
    """Calculating distance between 2 2D points p0 and p1 as nparrays in an arena
    with dimensions as in dimensions with infinite boundary conditions.
    po and p1 can be a set of 2D coordinates e.g. np.array([[x0, y0],[x1, y1],[x2, y2]])
    """
    delta = np.abs(p0 - p1)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=1))


def distance_coords(x1, y1, x2, y2, vectorized=False):
    """Distance between 2 points in 2D space calculated from point coordinates.
    if vectorized is True, we use multidimensional (i.e. vectorized) form of distance
    calculation that preserved original dimensions of coordinate arrays in the dimensions of the output and the output
    will contain pairwise distance measures according to coordinate matrices."""
    c1 = np.array([x1, y1])
    c2 = np.array([x2, y2])
    if not vectorized:
        distance = np.linalg.norm(c2 - c1)
    else:
        distance = np.linalg.norm(c2 - c1, axis=0)
    return distance


def distance(agent1, agent2):
    """Distance between 2 agent class agents in the environment as pixels"""
    c1 = np.array([agent1.position[0] + agent1.radius, agent1.position[1] + agent1.radius])
    c2 = np.array([agent2.position[0] + agent2.radius, agent2.position[1] + agent2.radius])
    distance = np.linalg.norm(c2 - c1)
    return distance


def F_reloc_LR(vel_now, V_now, v_desired=None):
    """Calculating relocation force according to the visual field/source data of the agent according to left-right
    algorithm"""
    if v_desired is None:
        v_desired = movement_params.reloc_des_vel
    V_field_len = len(V_now)
    left_excitation = np.mean(V_now[0:int(V_field_len / 2)])
    right_excitation = np.mean(V_now[int(V_field_len / 2)::])
    D_leftright = left_excitation - right_excitation
    D_theta_max = movement_params.reloc_theta_max
    theta = D_leftright * D_theta_max
    return (v_desired - vel_now), theta


def F_reloc_WTA(Phi, V_now):
    """Calculating relocation force according to the visual field/source data of the agent according to winner-takes-all
    mechanism"""
    pass
