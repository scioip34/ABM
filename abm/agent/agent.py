"""
agent.py : including the main classes to create an agent. Supplementary calculations independent from class attributes
            are removed from this file.
"""

import pygame
import numpy as np
from abm.contrib import colors, decision_params, movement_params
from abm.agent import supcalc
from collections import OrderedDict
import importlib
from abm.loader.helper import reconstruct_VPF


class Agent(pygame.sprite.Sprite):
    """
    Agent class that includes all private parameters of the agents and all methods necessary to move in the environment
    and to make decisions.
    """

    def __init__(self, id, radius, position, orientation, env_size, color, v_field_res, FOV, window_pad, pooling_time,
                 pooling_prob, consumption, vision_range, visual_exclusion,  phototaxis_theta_step, detection_range,
                 resource_meter_multiplier, signalling_cost, patchwise_exclusion=True, behave_params=None):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of agent (int)
        :param radius: radius of the agent in pixels
        :param position: position of the agent in env as (x, y)
        :param orientation: absolute orientation of the agent
        :param env_size: environment size available for agents as (width, height)
        :param color: color of the agent as (R, G, B)
        :param v_field_res: resolution of the visual field of the agent in pixels
        :param FOV: visual field as a tuple of min max visible angles e.g. (-np.pi, np.pi)
        :param window_pad: padding of the environment in simulation window in pixels
        :param pooling_time: time units needed to pool status of a given position in the environment
        :param pooling_prob: initial probability to switch to pooling behavior
        :param consumption: (resource unit/time unit) consumption efficiency of agent
        :param vision_range: in px the range/radius in which the agent is able to see other agents
        :param visual_exclusion: if True social cues can be visually excluded by non social cues.
        :param patchwise_exclusion: exclude agents from visual field if exploiting the same patch
        :param behave_params: dictionary of behavioral parameters can be passed to a given agent which will
            overwrite the parameters defined in the env files. (e.g. when agents are heterogeneous)
        :param phototaxis_theta_step: rotational speed scaling factor during phototaxis
        :param detection_range: detection range of resource patches (in pixels)
        :param resource_meter_multiplier: scaling factor of how much resource is extraxted for a detected resource unit
        :param signalling_cost: cost of signalling in resource units
        """
        # Initializing supercalss (Pygame Sprite)
        super().__init__()

        # in case we run multiple simulations, we reload the env parameters
        importlib.reload(decision_params)
        importlib.reload(movement_params)

        # Initializing agents with init parameters
        self.exclude_agents_same_patch = patchwise_exclusion
        self.id = id  # saved
        # creating agent status
        self.agent_type = "mars_miner"
        self.meter = 0  # between 0 and 1
        self.prev_meter = 0
        self.theta_prev = 0
        self.taxis_dir = None

        self.phototaxis_theta_step = phototaxis_theta_step  # 0.2
        self.detection_range = detection_range  # 120
        self.resource_meter_multiplier = resource_meter_multiplier  # 1
        self.signalling_cost = signalling_cost  # 0.5
        self.radius = radius
        self.position = np.array(position, dtype=np.float64)  # saved
        self.orientation = orientation  # saved
        self.color = color
        self.selected_color = colors.LIGHT_BLUE
        self.v_field_res = v_field_res
        self.pooling_time = pooling_time
        self.pooling_prob = pooling_prob
        self.consumption = consumption
        self.vision_range = vision_range
        self.visual_exclusion = visual_exclusion
        self.FOV = FOV
        self.show_stats = False

        # Non-initialisable private attributes
        self.velocity = 0  # agent absolute velocity  # saved
        self.collected_r = 0  # collected resource unit collected by agent  # saved
        self.collected_r_before = 0  # collected resource in the previous time step to monitor patch quality
        self.exploited_patch_id = -1  # saved
        self.mode = "explore"  # explore, flock, collide, exploit, pool  # saved
        self.soc_v_field = np.zeros(self.v_field_res)  # social visual projection field
        self.target_field = np.zeros(self.v_field_res)  # social visual projection field
        # source data to calculate relevant visual field according to the used relocation force algorithm
        self.vis_field_source_data = {}

        # Interaction
        self.is_moved_with_cursor = 0

        # Decision Variables
        self.overriding_mode = None

        if behave_params is not None:
            # the behavior parameters were passed as dictionary
            self.behave_params = behave_params
            ## w
            self.S_wu = self.behave_params["S_wu"]
            self.T_w = self.behave_params["T_w"]
            self.w = 0
            self.Eps_w = self.behave_params["Eps_w"]
            self.g_w = self.behave_params["g_w"]
            self.B_w = self.behave_params["B_w"]
            self.w_max = self.behave_params["w_max"]

            ## u
            self.I_priv = 0  # saved
            self.novelty = np.zeros(self.behave_params["Tau"])
            self.S_uw = self.behave_params["S_uw"]
            self.T_u = self.behave_params["T_u"]
            self.u = 0
            self.Eps_u = self.behave_params["Eps_u"]
            self.g_u = self.behave_params["g_u"]
            self.B_u = self.behave_params["B_u"]
            self.u_max = self.behave_params["u_max"]
            self.F_N = self.behave_params["F_N"]
            self.F_R = self.behave_params["F_R"]
            self.max_exp_vel = self.behave_params["exp_vel_max"]
            self.exp_stop_ratio = self.behave_params["exp_stop_ratio"]

        else:
            # as no behavior parameters were passed they are read out from env file
            ## w
            self.S_wu = decision_params.S_wu
            self.T_w = decision_params.T_w
            self.w = 0
            self.Eps_w = decision_params.Eps_w
            self.g_w = decision_params.g_w
            self.B_w = decision_params.B_w
            self.w_max = decision_params.w_max

            ## u
            self.I_priv = 0  # saved
            self.novelty = np.zeros(decision_params.Tau)
            self.S_uw = decision_params.S_uw
            self.T_u = decision_params.T_u
            self.u = 0
            self.Eps_u = decision_params.Eps_u
            self.g_u = decision_params.g_u
            self.B_u = decision_params.B_u
            self.u_max = decision_params.u_max
            self.F_N = decision_params.F_N
            self.F_R = decision_params.F_R

            # movement
            self.max_exp_vel = movement_params.exp_vel_max
            self.exp_stop_ratio = movement_params.exp_stop_ratio

        # Pooling attributes
        self.time_spent_pooling = 0  # time units currently spent with pooling the status of given position (changes
        # dynamically)
        self.env_status_before = 0
        self.env_status = 0  # status of the environment in current position, 1 if rescource, 0 otherwise
        self.pool_success = 0  # states if the agent deserves 1 piece of update about the status of env in given pos

        # Environment related parameters
        self.WIDTH = env_size[0]  # env width
        self.HEIGHT = env_size[1]  # env height
        self.window_pad = window_pad
        self.boundaries_x = [self.window_pad, self.window_pad + self.WIDTH]
        self.boundaries_y = [self.window_pad, self.window_pad + self.HEIGHT]

        # Initial Visualization of agent
        self.image = pygame.Surface([radius * 2, radius * 2])
        self.image.fill(colors.BACKGROUND)
        self.image.set_colorkey(colors.BACKGROUND)
        pygame.draw.circle(
            self.image, color, (radius, radius), radius
        )

        # Showing agent orientation with a line towards agent orientation
        pygame.draw.line(self.image, colors.BACKGROUND, (radius, radius),
                         ((1 + np.cos(self.orientation)) * radius, (1 - np.sin(self.orientation)) * radius), 3)
        self.rect = self.image.get_rect()
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]
        self.mask = pygame.mask.from_surface(self.image)

    def calc_I_priv(self):
        """returning I_priv according to the environment status. Note that this is not necessarily the same as
        later on I_priv also includes the reward amount in the last n timesteps"""
        # other part is coming from uncovered resource units
        collected_unit = self.collected_r - self.collected_r_before

        # calculating private info by weighting these
        self.I_priv = self.F_N * np.max(self.novelty) + self.F_R * collected_unit

    def move_with_mouse(self, mouse, left_state, right_state):
        """Moving the agent with the mouse cursor, and rotating"""
        if self.rect.collidepoint(mouse):
            # setting position of agent to cursor position
            self.position[0] = mouse[0] - self.radius
            self.position[1] = mouse[1] - self.radius
            if left_state:
                self.orientation += 0.1
            if right_state:
                self.orientation -= 0.1
            self.prove_orientation()
            self.is_moved_with_cursor = 1
            # updating agent visualization to make it more responsive
            self.draw_update()
        else:
            self.is_moved_with_cursor = 0

    def phototaxis(self, desired_velocity):
        """Local phototaxis search according to differential meter values"""
        diff = self.meter - self.prev_meter
        print(diff)
        sign_diff = np.sign(diff)
        # positive means the given change in orientation was correct
        # negative means we need to turn the other direction
        # zero means we are moving in the right direction
        if sign_diff >= 0:
            new_sign = sign_diff * np.sign(self.theta_prev)
            if new_sign == 0:
                new_sign = 1
            new_theta = self.phototaxis_theta_step * new_sign * self.meter
            self.taxis_dir = None
        else:
            if self.taxis_dir is None:
                self.taxis_dir = sign_diff * np.sign(self.theta_prev)
            new_sign = self.taxis_dir
            new_theta = self.phototaxis_theta_step * new_sign * self.meter

        new_vel = (desired_velocity - self.velocity)
        return new_vel, new_theta

    def update(self, agents):
        """
        main update method of the agent. This method is called in every timestep to calculate the new state/position
        of the agent and visualize it in the environment
        :param agents: a list of all obstacle/agents coordinates as (X, Y) in the environment. These are not necessarily
                socially relevant, i.e. all agents.
        """
        # calculate socially relevant projection field (e.g. according to signalling agents)
        self.calc_social_V_proj(agents)

        # some basic decision process of when to signal and when to explore, etc.
        signalling_threshold = 0.1
        if np.max(self.soc_v_field) > self.meter:
            # joining behavior
            vel, theta = supcalc.F_reloc_LR(self.velocity, self.soc_v_field, 2, theta_max=2.5)
            self.agent_type = "relocation"
            if self.meter > signalling_threshold:
                self.agent_type = "signalling"
        else:
            if self.meter > 0:
                vel, theta = self.phototaxis(desired_velocity=2)#supcalc.random_walk(desired_vel=self.max_exp_vel)
                self.agent_type = "mars_miner"
                #vel = (2 - self.velocity)
                if self.meter > signalling_threshold:
                    self.agent_type = "signalling"
            else:
                # carry out movemnt accordingly
                vel, theta = supcalc.random_walk(desired_vel=self.max_exp_vel)
                vel = (2 - self.velocity)
                self.agent_type = "mars_miner"

        # updating position accordingly
        if not self.is_moved_with_cursor:  # we freeze agents when we move them
            # updating agent's state variables according to calculated vel and theta
            self.orientation += theta
            # storing theta in short term memory for phototaxis
            self.theta_prev = theta
            self.prove_orientation()  # bounding orientation into 0 and 2pi
            self.velocity += vel
            # self.prove_velocity()  # possibly bounding velocity of agent

            # updating agent's position
            self.position[0] += self.velocity * np.cos(self.orientation)
            self.position[1] -= self.velocity * np.sin(self.orientation)

            # boundary conditions if applicable
            self.reflect_from_walls()
        else:
            self.agent_type = "signalling"
            print(self.meter)

        # updating agent visualization
        self.draw_update()
        self.collected_r_before = self.collected_r

        # collecting rewards according to meter value and signalling status
        self.update_rewards()

    def update_rewards(self):
        """Updating agent collected resource values according to distance from resource (as in meter value)
        and current signalling status"""
        self.collected_r += self.meter * self.resource_meter_multiplier
        if self.agent_type == "signalling":
            self.collected_r -= self.signalling_cost

    def change_color(self):
        """Changing color of agent according to the behavioral mode the agent is currently in."""
        if self.agent_type == "mars_miner":
            self.color = colors.BLUE
        elif self.agent_type == "signalling":
            self.color = colors.RED
        elif self.agent_type == "relocation":
            self.color = colors.PURPLE

    def draw_update(self):
        """
        updating the outlook of the agent according to position and orientation
        """
        # update position
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]

        # change agent color according to mode
        self.change_color()

        # update surface according to new orientation
        # creating visualization surface for agent as a filled circle
        self.image = pygame.Surface([self.radius * 2, self.radius * 2])
        self.image.fill(colors.BACKGROUND)
        self.image.set_colorkey(colors.BACKGROUND)

        pygame.draw.circle(
            self.image, self.color, (self.radius, self.radius), self.radius
        )

        # showing agent orientation with a line towards agent orientation
        pygame.draw.line(self.image, colors.BACKGROUND, (self.radius, self.radius),
                         ((1 + np.cos(self.orientation)) * self.radius, (1 - np.sin(self.orientation)) * self.radius),
                         3)
        self.mask = pygame.mask.from_surface(self.image)

    def reflect_from_walls(self):
        """reflecting agent from environment boundaries according to a desired x, y coordinate. If this is over any
        boundaries of the environment, the agents position and orientation will be changed such that the agent is
         reflected from these boundaries."""

        # Boundary conditions according to center of agent (simple)
        x = self.position[0] + self.radius
        y = self.position[1] + self.radius

        # Reflection from left wall
        if x < self.boundaries_x[0]:
            self.position[0] = self.boundaries_x[0] - self.radius

            if np.pi / 2 <= self.orientation < np.pi:
                self.orientation -= np.pi / 2
            elif np.pi <= self.orientation <= 3 * np.pi / 2:
                self.orientation += np.pi / 2
            self.prove_orientation()  # bounding orientation into 0 and 2pi

        # Reflection from right wall
        if x > self.boundaries_x[1]:

            self.position[0] = self.boundaries_x[1] - self.radius - 1

            if 3 * np.pi / 2 <= self.orientation < 2 * np.pi:
                self.orientation -= np.pi / 2
            elif 0 <= self.orientation <= np.pi / 2:
                self.orientation += np.pi / 2
            self.prove_orientation()  # bounding orientation into 0 and 2pi

        # Reflection from upper wall
        if y < self.boundaries_y[0]:
            self.position[1] = self.boundaries_y[0] - self.radius

            if np.pi / 2 <= self.orientation <= np.pi:
                self.orientation += np.pi / 2
            elif 0 <= self.orientation < np.pi / 2:
                self.orientation -= np.pi / 2
            self.prove_orientation()  # bounding orientation into 0 and 2pi

        # Reflection from lower wall
        if y > self.boundaries_y[1]:
            self.position[1] = self.boundaries_y[1] - self.radius - 1
            if 3 * np.pi / 2 <= self.orientation <= 2 * np.pi:
                self.orientation += np.pi / 2
            elif np.pi <= self.orientation < 3 * np.pi / 2:
                self.orientation -= np.pi / 2
            self.prove_orientation()  # bounding orientation into 0 and 2pi

    def calc_social_V_proj(self, agents):
        """Calculating the socially relevant visual projection field of the agent. This is calculated as the
        projection of nearby exploiting agents that are not visually excluded by other agents"""
        signalling = [ag for ag in agents if ag.agent_type == "signalling"]
        self.soc_v_field = self.projection_field(signalling, keep_distance_info=True)

    def exlude_V_source_data(self):
        """Calculating parts of the VPF source data that depends on visual exclusion, i.e. how agents are excluding
        parts of each others projection on the retina of the focal agent."""
        self.rank_V_source_data("distance", reverse=False)

        rank = 0
        for kf, vf in self.vis_field_source_data.items():
            if rank > 0:
                for ki, vi in self.vis_field_source_data.items():
                    if vi["distance"] < vf["distance"]:
                        # Partial exclusion 1
                        if vf["proj_start_ex"] <= vi["proj_start"] <= vf["proj_end_ex"]:
                            vf["proj_end_ex"] = vi["proj_start"]
                        # Partial exclusion 2
                        if vf["proj_start_ex"] <= vi["proj_end"] <= vf["proj_end_ex"]:
                            vf["proj_start_ex"] = vi["proj_end"]
                        # Total exclusion
                        if vi["proj_start"] <= vf["proj_start_ex"] and vi["proj_end"] >= vf["proj_end_ex"]:
                            vf["proj_start_ex"] = 0
                            vf["proj_end_ex"] = 0
            else:
                vf["proj_start_ex"] = vf["proj_start"]
                vf["proj_end_ex"] = vf["proj_end"]
            vf["proj_size_ex"] = vf["proj_end_ex"] - vf["proj_start_ex"]
            rank += 1

    def remove_nonsocial_V_source_data(self):
        """Removing any non-social projection source data from the visual source data. Until this point we might have
        needed them so we could calculate the visual exclusion on social cues they cause but from this point we do
        not want interactions to happen according to them."""
        clean_sdata = {}
        for kf, vf in self.vis_field_source_data.items():
            if vf['is_social_cue']:
                clean_sdata[kf] = vf
        self.vis_field_source_data = clean_sdata

    def projection_field(self, obstacles, keep_distance_info=False, non_expl_agents=None, fov=None):
        """Calculating visual projection field for the agent given the visible obstacles in the environment
        :param obstacles: list of agents (with same radius) or some other obstacle sprites to generate projection field
        :param keep_distance_info: if True, the amplitude of the vpf will reflect the distance of the object from the
            agent so that exclusion can be easily generated with a single computational step.
        :param non_expl_agents: a list of non-scoial visual cues (non-exploiting agents) that on the other hand can still
            produce visual exlusion on the projection of social cues. If None only social cues can produce visual
            exclusion on each other.
        :param fov: touple of number with borders of fov such as (-np.pi, np.pi), if None, self.FOV will be used"""

        # deciding fov
        if fov is None:
            fov = self.FOV

        # extracting obstacle coordinates
        obstacle_coords = [ob.position for ob in obstacles]
        meters = [ob.meter for ob in obstacles]

        # if non-social cues can visually exclude social ones we also concatenate these to the obstacle coords
        if non_expl_agents is not None:
            len_social = len(obstacles)
            obstacle_coords.extend([ob.position for ob in non_expl_agents])

        # initializing visual field and relative angles
        v_field = np.zeros(self.v_field_res)
        phis = np.linspace(-np.pi, np.pi, self.v_field_res)

        # center point
        v1_s_x = self.position[0] + self.radius
        v1_s_y = self.position[1] + self.radius

        # point on agent's edge circle according to it's orientation
        v1_e_x = self.position[0] + (1 + np.cos(self.orientation)) * self.radius
        v1_e_y = self.position[1] + (1 - np.sin(self.orientation)) * self.radius

        # vector between center and edge according to orientation
        v1_x = v1_e_x - v1_s_x
        v1_y = v1_e_y - v1_s_y

        v1 = np.array([v1_x, v1_y])

        # calculating closed angle between obstacle and agent according to the position of the obstacle.
        # then calculating visual projection size according to visual angle on the agents's retina according to distance
        # between agent and obstacle
        self.vis_field_source_data = {}
        for i, obstacle_coord in enumerate(obstacle_coords):
            if not (obstacle_coord[0] == self.position[0] and obstacle_coord[1] == self.position[1]):
                # center of obstacle (as it must be another agent)
                v2_e_x = obstacle_coord[0] + self.radius
                v2_e_y = obstacle_coord[1] + self.radius

                # vector between agent center and obstacle center
                v2_x = v2_e_x - v1_s_x
                v2_y = v2_e_y - v1_s_y

                v2 = np.array([v2_x, v2_y])

                # calculating closed angle between v1 and v2
                # (rotated with the orientation of the agent as it is relative)
                closed_angle = supcalc.angle_between(v1, v2)
                closed_angle = (closed_angle % (2 * np.pi))
                # at this point closed angle between 0 and 2pi but we need it between -pi and pi
                # we also need to take our orientation convention into consideration to recalculate
                # theta=0 is pointing to the right
                if 0 < closed_angle < np.pi:
                    closed_angle = -closed_angle
                else:
                    closed_angle = 2 * np.pi - closed_angle

                # calculating the visual angle from focal agent to target
                c1 = np.array([v1_s_x, v1_s_y])
                c2 = np.array([v2_e_x, v2_e_y])
                distance = np.linalg.norm(c2 - c1)
                vis_angle = 2 * np.arctan(self.radius / (1 * distance))

                # finding where in the retina the projection belongs to
                phi_target = supcalc.find_nearest(phis, closed_angle)

                # if target is visible we save its projection into the VPF source data
                if fov[0] < closed_angle < fov[1]:
                    self.vis_field_source_data[i] = {}
                    self.vis_field_source_data[i]["vis_angle"] = vis_angle
                    self.vis_field_source_data[i]["phi_target"] = phi_target
                    self.vis_field_source_data[i]["distance"] = distance
                    self.vis_field_source_data[i]["meter"] = meters[i]
                    # the projection size is proportional to the visual angle. If the projection is maximal (i.e.
                    # taking each pixel of the retina) the angle is 2pi from this we just calculate the proj. size
                    # using a single proportion
                    self.vis_field_source_data[i]["proj_size"] = (vis_angle / (2 * np.pi)) * self.v_field_res
                    proj_size = self.vis_field_source_data[i]["proj_size"]
                    self.vis_field_source_data[i]["proj_start"] = int(phi_target - proj_size / 2)
                    self.vis_field_source_data[i]["proj_end"] = int(phi_target + proj_size / 2)
                    self.vis_field_source_data[i]["proj_start_ex"] = int(phi_target - proj_size / 2)
                    self.vis_field_source_data[i]["proj_end_ex"] = int(phi_target + proj_size / 2)
                    self.vis_field_source_data[i]["proj_size_ex"] = proj_size
                    if non_expl_agents is not None:
                        if i < len_social:
                            self.vis_field_source_data[i]["is_social_cue"] = True
                        else:
                            self.vis_field_source_data[i]["is_social_cue"] = False
                    else:
                        self.vis_field_source_data[i]["is_social_cue"] = True

        # calculating visual exclusion if requested
        if self.visual_exclusion:
            self.exlude_V_source_data()

        if non_expl_agents is not None:
            # removing non-social cues from the source data after calculating exclusions
            self.remove_nonsocial_V_source_data()

        # sorting VPF source data according to visual angle
        self.rank_V_source_data("vis_angle")

        for k, v in self.vis_field_source_data.items():
            vis_angle = v["vis_angle"]
            phi_target = v["phi_target"]
            proj_size = v["proj_size"]

            proj_start = v["proj_start_ex"]
            proj_end = v["proj_end_ex"]

            # circular boundaries to the VPF as there is 360 degree vision
            if proj_start < 0:
                v_field[self.v_field_res + proj_start:self.v_field_res] = 1
                proj_start = 0

            if proj_end >= self.v_field_res:
                v_field[0:proj_end - self.v_field_res] = 1
                proj_end = self.v_field_res - 1

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

    def rank_V_source_data(self, ranking_key, reverse=True):
        """Ranking source data of visual projection field by the visual angle
        :param: ranking_key: stribg key according to which the dictionary is sorted"""
        self.vis_field_source_data = OrderedDict(sorted(self.vis_field_source_data.items(),
                                                        key=lambda kv: kv[1][ranking_key], reverse=reverse))

    def prove_orientation(self):
        """Restricting orientation angle between 0 and 2 pi"""
        if self.orientation < 0:
            self.orientation = 2 * np.pi + self.orientation
        if self.orientation > np.pi * 2:
            self.orientation = self.orientation - 2 * np.pi

    def prove_velocity(self, velocity_limit=1):
        """Restricting the absolute velocity of the agent"""
        vel_sign = np.sign(self.velocity)
        if vel_sign == 0:
            vel_sign = +1
        if self.get_mode() == 'explore':
            if np.abs(self.velocity) > velocity_limit:
                # stopping agent if too fast during exploration
                self.velocity = 1

    def pool_curr_pos(self):
        """Pooling process of the current position. During pooling the agent does not move and spends a given time in
        the position. At the end the agent is notified by the status of the environment in the given position"""

        if self.get_mode() == "pool":
            if self.time_spent_pooling == self.pooling_time:
                self.end_pooling("success")
            else:
                self.velocity = 0
                self.time_spent_pooling += 1

    def end_pooling(self, pool_status_flag):
        """
        Ending pooling process either with interrupting pooling with no success or with notifying agent about the status
        of the environemnt in the given position upon success
        :param pool_status_flag: ststing how the pooling process ends, either "success" or "interrupt"
        """
        if pool_status_flag == "success":
            self.pool_success = 1
        else:
            self.pool_success = 0
        self.time_spent_pooling = 0

    def tr_w(self):
        """Relocation threshold function that checks if decision variable w is above T_w"""
        if self.w > self.T_w:
            return True
        else:
            return False

    def tr_u(self):
        """Exploitation threshold function that checks if decision variable u is above T_w"""
        if self.u > self.T_w:
            return True
        else:
            return False

    def get_mode(self):
        """returning the current mode of the agent according to it's inner decision mechanisms as a human-readable
        string for external processes defined in the main simulation thread (such as collision that depends on the
        state of the at and also overrides it as it counts as ana emergency)"""
        if self.overriding_mode is None:
            if self.tr_w():
                return "relocate"
            else:
                return "explore"
        else:
            return self.overriding_mode

    def set_mode(self, mode):
        """setting the behavioral mode of the agent according to some human_readable flag. This can be:
            -explore
            -exploit
            -relocate
            -pool
            -collide"""
        if mode == "explore":
            # self.w = 0
            self.overriding_mode = None
        elif mode == "relocate":
            # self.w = self.T_w + 0.001
            self.overriding_mode = None
        elif mode == "collide":
            self.overriding_mode = "collide"
            # self.w = 0
        elif mode == "exploit":
            self.overriding_mode = "exploit"
            # self.w = 0
        elif mode == "pool":
            self.overriding_mode = "pool"
            # self.w = 0
        self.mode = mode
