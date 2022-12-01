from collections import OrderedDict

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


def rank_v_source_data(ranking_key, data, reverse=True):
    """
    Ranking source data of visual projection field by the visual angle
    :param: ranking_key: stribg key according to which the dictionary is sorted
    """
    return OrderedDict(sorted(
        data.items(),
        key=lambda kv: kv[1][ranking_key],
        reverse=reverse))


def projection_field(objects, fov, v_field_resolution, position, radius,
                     orientation, keep_distance_info=False,
                     non_expl_agents=None):
    """
    Calculating visual projection field for the agent given the visible
    obstacles in the environment
    :param objects: list of agents (with same radius) or some other
    obstacle sprites to generate projection field
    :param fov: tuple of number with borders of fov such as (-np.pi,np.pi)
    :param v_field_resolution: visual field resolution in pixels
    :param position: tuple with position of the agent (x,y)
    :param radius: radius of the agent
    :param orientation: orientation of the agent
    :param keep_distance_info: if True, the amplitude of the vpf will
    reflect the distance of the object from the agent so that exclusion can
    be easily generated with a single computational step
    :param non_expl_agents: a list of non-social visual cues (non-exploiting
    agents) that on the other hand can still produce visual exclusion on the
    projection of social cues. If None only social cues can produce visual
    exclusion on each other

    """
    # extracting obstacle coordinates
    obj_coords = [ob.position for ob in objects]
    meters = [ob.meter for ob in objects]

    # if non-social cues can visually exclude social ones we also
    # concatenate these to the obstacle coords
    if non_expl_agents is not None:
        len_social = len(objects)
        obj_coords.extend([ob.position for ob in non_expl_agents])

    # initializing visual field and relative angles
    v_field = np.zeros(v_field_resolution)
    phis = np.linspace(-np.pi, np.pi, v_field_resolution)

    # center point
    v1_s_x = position[0] + radius
    v1_s_y = position[1] + radius

    # point on agent's edge circle according to it's orientation
    v1_e_x = position[0] + (1 + np.cos(orientation)) * radius
    v1_e_y = position[1] + (1 - np.sin(orientation)) * radius

    # vector between center and edge according to orientation
    v1_x = v1_e_x - v1_s_x
    v1_y = v1_e_y - v1_s_y
    v1 = np.array([v1_x, v1_y])

    # calculating closed angle between obstacle and agent according to the
    # position of the obstacle.
    # then calculating visual projection size according to visual angle on
    # the agent's retina according to distance
    # between agent and obstacle

    vis_field_source_data = {}
    for i, obstacle_coord in enumerate(obj_coords):
        x_coincidence = obstacle_coord[0] == position[0]
        y_coincidence = obstacle_coord[1] == position[1]

        if not(x_coincidence and y_coincidence):
            # center of obstacle (as it must be another agent)
            v2_e_x = obstacle_coord[0] + radius
            v2_e_y = obstacle_coord[1] + radius
            # vector between agent center and obstacle center
            v2_x = v2_e_x - v1_s_x
            v2_y = v2_e_y - v1_s_y
            v2 = np.array([v2_x, v2_y])
            # calculating closed angle between v1 and v2
            # (rotated with the orientation of the agent as it is relative)
            closed_angle = angle_between(v1, v2)
            closed_angle = (closed_angle % (2 * np.pi))
            # at this point closed angle between 0 and 2pi, but we need it
            # between -pi and pi
            # we also need to take our orientation convention into
            # consideration to recalculate
            # theta=0 is pointing to the right
            if 0 < closed_angle < np.pi:
                closed_angle = -closed_angle
            else:
                closed_angle = 2 * np.pi - closed_angle
            # calculating the visual angle from focal agent to target
            c1 = np.array([v1_s_x, v1_s_y])
            c2 = np.array([v2_e_x, v2_e_y])
            distance = np.linalg.norm(c2 - c1)
            vis_angle = 2 * np.arctan(radius / (1 * distance))
            # finding where in the retina the projection belongs to
            phi_target = find_nearest(phis, closed_angle)
            # if target is visible we save its projection into the VPF
            # source data
            if fov[0] < closed_angle < fov[1]:
                vis_field_source_data[i] = {}
                vis_field_source_data[i]["vis_angle"] = vis_angle
                vis_field_source_data[i]["phi_target"] = phi_target
                vis_field_source_data[i]["distance"] = distance
                vis_field_source_data[i]["meter"] = meters[i]
                # the projection size is proportional to the visual angle.
                # If the projection is maximal (i.e.
                # taking each pixel of the retina) the angle is 2pi from
                # this we just calculate the proj. size
                # using a single proportion
                vis_field_source_data[i]["proj_size"] = (vis_angle / (
                        2 * np.pi)) * v_field_resolution
                proj_size = vis_field_source_data[i]["proj_size"]
                vis_field_source_data[i]["proj_start"] = int(
                    phi_target - proj_size / 2)
                vis_field_source_data[i]["proj_end"] = int(
                    phi_target + proj_size / 2)
                vis_field_source_data[i]["proj_start_ex"] = int(
                    phi_target - proj_size / 2)
                vis_field_source_data[i]["proj_end_ex"] = int(
                    phi_target + proj_size / 2)
                vis_field_source_data[i]["proj_size_ex"] = proj_size
                if non_expl_agents is not None:
                    if i < len_social:
                        vis_field_source_data[i][
                            "is_social_cue"] = True
                    else:
                        vis_field_source_data[i][
                            "is_social_cue"] = False
                else:
                    vis_field_source_data[i]["is_social_cue"] = True
    # calculating visual exclusion if requested
    # TODO
    # if self.visual_exclusion:
    #     self.exlude_V_source_data()

    # TODO
    # if non_expl_agents is not None:
    #     # removing non-social cues from the source data after calculating
    #     # exclusions
    #     self.remove_nonsocial_V_source_data()

    # sorting VPF source data according to visual angle
    vis_field_source_data = rank_v_source_data(
        "vis_angle", vis_field_source_data)

    for k, v in vis_field_source_data.items():
        vis_angle = v["vis_angle"]
        phi_target = v["phi_target"]
        proj_size = v["proj_size"]
        proj_start = v["proj_start_ex"]
        proj_end = v["proj_end_ex"]
        # circular boundaries to the VPF as there is 360 degree vision
        if proj_start < 0:
            v_field[v_field_resolution + proj_start:v_field_resolution] = 1
            proj_start = 0
        if proj_end >= v_field_resolution:
            v_field[0:proj_end - v_field_resolution] = 1
            proj_end = v_field_resolution - 1
        # weighing projection amplitude with rank information if requested
        if not keep_distance_info:
            v_field[proj_start:proj_end] = 1
        else:
            v_field[proj_start:proj_end] = v["meter"]

    # post_processing and limiting FOV
    v_field_post = np.flip(v_field)
    v_field_post[phis < fov[0]] = 0
    v_field_post[phis > fov[1]] = 0
    return v_field_post
