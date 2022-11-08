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


def phototaxis(meter, prev_meter, prev_theta, taxis_dir,
               phototaxis_theta_step):
    """
    Local phototaxis search according to differential meter values
    :param meter: current meter (detector) value between 0 and 1
    :param prev_meter: meter value in previous timestep between 0 and 1
    :param prev_theta: turning angle in previous timestep
    :param taxis_dir: extended phototaxis direction [-1, 1, None].
                      None: calculate turning direction according to meter value
                      [-1, 1]: calculate turning direction w.r.t. direction change
                               when meter value first dropped
    :param phototaxis_theta_step: maximum turning angle during phototaxis
    :return new_theta: directional change in orientation according to phototaxis
    :return new_taxis_dir: extended new phototaxis direction [-1, 1, None].
                      None: calculate turning direction according to meter value
                      [-1, 1]: calculate turning direction w.r.t. direction change
                               when meter value first dropped
    """
    # calculating difference between current and previous meter values
    diff = meter - prev_meter

    # ------ CALCULATING DIRECTION OF ROTATION ------ #
    # sign of difference gives information about getting closer or farther
    # from resource patch.
    sign_diff = np.sign(diff)

    # positive means the given change in orientation increased the meter value.
    if sign_diff >= 0:
        new_sign = np.sign(prev_theta) if prev_theta != 0 else 1
        # calculate turning direction according to meter value in next step as well
        new_taxis_dir = None

    # negative means we are moving in the wrong direction and need to turn the other direction
    else:
        # so far the meter values were growing, no direction switch has been yet initiated
        if taxis_dir is None:
            # initiating direction switch and turning to that direction until meter
            # values will grow again
            new_taxis_dir = sign_diff * np.sign(prev_theta)
            new_sign = new_taxis_dir

        # we already initiated a direction switch before, so we keep acting accordingly
        else:
            # we keep moving in the previously set phototaxis direction
            new_taxis_dir = taxis_dir
            new_sign = new_taxis_dir

    # ------ CALCULATING EXTENT OF ROTATION ------ #
    # change theta proportionally to the meter values,
    # the larger the meter value, the closer the agent to the center.
    # As agents are forced to move continuously, closer to the center
    # agents need to turn faster to stay close to the center.
    new_theta = phototaxis_theta_step * new_sign * meter
    return new_theta, new_taxis_dir
