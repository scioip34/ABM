import numpy as np
import pygame

from abm.contrib import movement_params


# Random Walk functions
def random_walk(desired_vel=None, exp_theta_min=None, exp_theta_max=None):
    """
    Pooling a small orientation and absolute velocity increment from some
    distribution
    """
    if desired_vel is None:
        desired_vel = movement_params.exp_vel_max
    if exp_theta_min is None:
        exp_theta_min = movement_params.exp_theta_min
    if exp_theta_max is None:
        exp_theta_max = movement_params.exp_theta_max
    dvel = desired_vel
    dtheta = np.random.uniform(exp_theta_min, exp_theta_max)
    return dvel, dtheta


def F_reloc_LR(vel_now, V_now, v_desired=None, theta_max=None):
    """
    Calculating relocation force according to the visual field/source data
    of the agent according to left-right
    algorithm
    """
    if v_desired is None:
        v_desired = movement_params.reloc_des_vel
    if theta_max is None:
        theta_max = movement_params.reloc_theta_max
    V_field_len = len(V_now)
    left_excitation = np.mean(V_now[0:int(V_field_len / 2)])
    right_excitation = np.mean(V_now[int(V_field_len / 2)::])
    D_leftright = left_excitation - right_excitation
    theta = D_leftright * theta_max
    return (v_desired - vel_now), theta


def reflection_from_circular_wall(dx, dy, orientation):
    """
    Calculating the reflection of the agent from the circle arena border.
    SEE: https://stackoverflow.com/questions/54543170/angle-reflexion-for-bouncing-ball-in-a-circle

    :param dx: x coordinate of the agent minus center of the circle
    :param dy: y coordinate of the agent minus center of the circle
    :param orientation: orientation of the agent
    :return: new orientation of the agent
    """
    # normal vector of the circle
    c_norm = (pygame.math.Vector2(dx, dy)).normalize()
    # incident vector: the current direction vector of the bouncing agent
    vec_i = pygame.math.Vector2(np.cos(orientation), np.sin(orientation))
    # orientation inside the circle
    i_orientation = np.pi + np.arctan2(vec_i[1], vec_i[0])

    # reflection vector: outgoing direction vector of the bouncing agent
    vec_r = vec_i - 2 * c_norm.dot(vec_i) * c_norm
    # np.degrees(self.orientation)
    new_orientation = np.pi + np.arctan2(vec_r[1], vec_r[0])

    # make sure that the new orientation points inside the circle and not too
    # flat to the border
    if np.abs(new_orientation - i_orientation) > np.pi / 4:
        new_orientation = i_orientation
    # make sure that the change of the orientation is not too big; this prevents
    # the agent from "jumping" over the border and other wierd behavior when the
    # agent changes its orientation too ofter
    elif np.abs(new_orientation - orientation) > np.pi / 4:
        new_orientation = i_orientation

    return new_orientation
