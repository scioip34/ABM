import numpy as np
import pygame

from abm.agent.supcalc import angle_between, find_nearest
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


def f_reloc_lr(velocity, visual_field, velocity_desired=None, theta_max=None):
    """
    Calculating relocation force according to the visual field/source data
    of the agent according to left-right algorithm
    """
    v_field_len = len(visual_field)
    left_excitation = np.mean(visual_field[0:int(v_field_len / 2)])
    right_excitation = np.mean(visual_field[int(v_field_len / 2)::])
    delta_left_right = left_excitation - right_excitation
    theta = delta_left_right * theta_max
    return (velocity_desired - velocity), theta


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


def signaling(meter, is_signaling, signaling_cost,
              probability_of_starting_signaling, rand_value):
    """
    :param meter: current meter (detector) value between 0 and 1
    :param is_signaling: boolean indicating whether the agent is currently
    signaling
    :param signaling_cost: cost of signaling
    :param probability_of_starting_signaling: probability of starting signaling
    :param rand_value: random value between 0 and 1
    :return: boolean value indicating whether the agent is signaling or not
    """
    if meter == 0:
        # no signaling if meter is zero
        return False
    elif is_signaling:
        # continue signaling if meter is not zero and agent is already signaling
        return True
    elif meter > signaling_cost:
        # start signaling if signaling cost is smaller than meter value
        return True if rand_value < probability_of_starting_signaling else False


def agent_decision(meter, max_signal_of_other_agents, max_crowd_density,
                   crowd_density_threshold=0.5):
    """
    Decision tree for the agent's behavior
    :param meter: current meter (detector) value between 0 and 1
    :param max_signal_of_other_agents: meter value between 0 and 1
    :param max_crowd_density: density of the crowd between 0 and 1
    :param crowd_density_threshold: threshold when the crowd density is high
    enough to draw agent's attention
    :return: agent_state: exploration, taxis, relocation or flocking
    """

    if meter == 0:
        # no meter value, agent is exploring, relocating or flocking
        # NOTE: signaling has priority over flocking
        if max_signal_of_other_agents > 0:
            # if there is a signal from other agents, relocate
            return 'relocation'
        elif max_crowd_density > crowd_density_threshold:
            # if the crowd density is high enough, the agent will relocate
            return 'flocking'
        else:
            # if there is no signal from other agents, explore
            return 'exploration'
    elif meter > 0:
        # meter value, agent performs taxis or relocation
        if max_signal_of_other_agents > meter:
            # if a signal from other agents is larger than meter, relocate
            return 'relocation'
        else:
            # if a signal from other agents is smaller than meter, perform taxis
            return 'taxis'


def projection_field(fov, v_field_resolution, position, radius,
                     orientation, object_positions, object_meters=None, max_proj_size=None):
    """
    Calculating visual projection field for the agent given the visible
    obstacles in the environment
    obstacle sprites to generate projection field
    :param fov: tuple of number with borders of fov such as (-np.pi, np.pi)
    :param v_field_resolution: visual field resolution in pixels
    :param position: np.xarray of agent's position
    :param radius: radius of the agent
    :param orientation: orientation angle between 0 and 2pi
    :param object_positions: list of np.xarray of object's positions
    :param object_meters: list of object's meters, default is None
    :param max_proj_size: maximum projection size to include in the visual proj. field
    :return: projection field np.xarray with shape (n objects, field resolution)
    """
    # initializing visual field and relative angles
    v_field = np.zeros((len(object_positions), v_field_resolution))
    phis = np.linspace(-np.pi, np.pi, v_field_resolution)

    # center point
    agents_center = position + radius

    # point on agent's edge circle according to it's orientation
    agent_edge = position + np.array(
        [1 + np.cos(orientation), 1 - np.sin(orientation)]) * radius

    # vector between center and edge according to orientation
    v1 = agents_center - agent_edge

    # 1. Calculating closed angle between object and agent according to the
    # position of the object.
    # 2. Calculating visual projection size according to visual angle on
    # the agent's retina according to distance between agent and object
    for i, obj_position in enumerate(object_positions):
        # continue if the object and the agent position completely coincide
        if obj_position[0] == position[0] and obj_position[1] == position[1]:
            continue

        # center of obstacle (as it must be another agent)
        object_center = obj_position + radius

        # vector between agent center and object center
        v2 = object_center - agents_center

        # calculating closed angle between v1 and v2
        # (rotated with the orientation of the agent as it is relative)
        closed_angle = angle_between(v1, v2)
        closed_angle = (closed_angle % (2 * np.pi))
        # at this point closed angle between 0 and 2pi, but we need it between
        # -pi and pi
        # we also need to take our orientation convention into consideration to
        # recalculate theta=0 is pointing to the right
        if 0 < closed_angle < np.pi:
            closed_angle = -closed_angle
        else:
            closed_angle = 2 * np.pi - closed_angle

        distance = np.linalg.norm(object_center - agents_center)

        # calculating the visual angle from focal agent to target
        vis_angle = 2 * np.arctan(radius / (1 * distance))

        # finding where in the retina the projection belongs to
        phi_target = find_nearest(phis, closed_angle)

        # if target is visible we save its projection into the VPF
        # source data
        if fov[0] <= closed_angle <= fov[1]:
            # the projection size is proportional to the visual angle.
            # If the projection is maximal (i.e. taking each pixel of the
            # retina) the angle is 2pi from this we just calculate the
            # projection size using a single proportion
            proj_size = (vis_angle / (2 * np.pi)) * v_field_resolution

            # Check if projection size is valid
            if max_proj_size is None:
                # If no maximum projection size is passed, all projection is valid in the FOV
                valid_proj = True
            elif proj_size <= max_proj_size:
                # If there is a max projection size, only smaller projections are valid
                valid_proj = True
            else:
                valid_proj = False

            if valid_proj:
                proj_start = int(phi_target - np.floor(proj_size / 2))
                proj_end = int(phi_target + np.floor(proj_size / 2))

                # circular boundaries to the VPF as there is 360 degree vision
                if proj_start < 0:
                    v_field[i, v_field_resolution + proj_start:v_field_resolution] = 1
                    proj_start = 0
                if proj_end >= v_field_resolution:
                    v_field[i, 0:proj_end - (v_field_resolution - 1)] = 1
                    proj_end = v_field_resolution

                v_field[i, proj_start:proj_end] = 1

                if object_meters is not None:
                    v_field[i] *= object_meters[i]

    # post_processing and limiting FOV
    # flip field data along second dimension
    # TODO: why we need to flip the field?
    # v_field_post = np.flip(v_field, axis=1)
    v_field_post = v_field

    v_field_post[:, phis < fov[0]] = 0
    v_field_post[:, phis > fov[1]] = 0
    return v_field_post
