import numpy as np

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
