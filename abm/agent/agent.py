"""
agent.py : including the main classes to create an agent. Supplementary calculations independent from class attributes
            are removed from this file.
"""

import pygame
import numpy as np
from abm.contrib import colors, decision_params
from abm.agent import supcalc


class Agent(pygame.sprite.Sprite):
    """
    Agent class that includes all private parameters of the agents and all methods necessary to move in the environment
    and to make decisions.
    """

    def __init__(self, id, radius, position, orientation, env_size, color, v_field_res, FOV, window_pad, pooling_time,
                 pooling_prob, consumption, vision_range, visual_exclusion, patchwise_exclusion=True):
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
        """
        # Initializing supercalss (Pygame Sprite)
        super().__init__()

        # Initializing agents with init parameters
        self.exclude_agents_same_patch = patchwise_exclusion
        self.id = id
        self.radius = radius
        self.position = np.array(position, dtype=np.float64)
        self.orientation = orientation
        self.color = color
        self.v_field_res = v_field_res
        self.pooling_time = pooling_time
        self.pooling_prob = pooling_prob
        self.consumption = consumption
        self.vision_range = vision_range
        self.visual_exclusion = visual_exclusion
        self.FOV = FOV

        # Non-initialisable private attributes
        self.velocity = 0  # agent absolute velocity
        self.collected_r = 0  # collected resource unit collected by agent
        self.collected_r_before = 0  # collected resource in the previous time step to monitor patch quality
        self.exploited_patch_id = -1
        self.mode = "explore"  # explore, flock, collide, exploit, pool
        self.soc_v_field = np.zeros(self.v_field_res)  # social visual projection field
        # source data to calculate relevant visual field according to the used relocation force algorithm
        self.vis_field_source_data = {}

        # Interaction
        self.is_moved_with_cursor = 0

        # Decision Variables
        self.overriding_mode = None

        ## w
        self.S_wu = decision_params.S_wu
        self.T_w = decision_params.T_w
        self.w = 0
        self.Eps_w = decision_params.Eps_w
        self.g_w = decision_params.g_w
        self.B_w = decision_params.B_w
        self.w_max = decision_params.w_max

        ## u
        self.I_priv = 0
        self.novelty = np.zeros(decision_params.Tau)
        self.S_uw = decision_params.S_uw
        self.T_u = decision_params.T_u
        self.u = 0
        self.Eps_u = decision_params.Eps_u
        self.g_u = decision_params.g_u
        self.B_u = decision_params.B_u
        self.u_max = decision_params.u_max

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
        self.mask = pygame.mask.from_surface(self.image)

    def calc_I_priv(self):
        """returning I_priv according to the environment status. Note that this is not necessarily the same as
        later on I_priv also includes the reward amount in the last n timesteps"""
        # other part is coming from uncovered resource units
        collected_unit = self.collected_r - self.collected_r_before

        # calculating private info by weighting these
        self.I_priv = decision_params.F_N * np.max(self.novelty) + decision_params.F_R*collected_unit

    def move_with_mouse(self, mouse, left_state, right_state):
        """Moving the agent with the mouse cursor, and rotating"""
        if self.rect.collidepoint(mouse):
            # setting position of agent to cursor position
            self.position[0] = mouse[0] - self.radius
            self.position[1] = mouse[1] - self.radius
            if left_state:
                self.orientation+=0.1
            if right_state:
                self.orientation-=0.1
            self.is_moved_with_cursor = 1
            # updating agent visualization to make it more responsive
            self.draw_update()
        else:
            self.is_moved_with_cursor = 0

    def update_decision_processes(self):
        """updating inner decision processes according to the current state and the visual projection field"""
        w_p = self.w if self.w > self.T_w else 0
        u_p = self.u if self.u > self.T_u else 0
        dw = self.Eps_w * (np.mean(self.soc_v_field)) - self.g_w * (
                    self.w - self.B_w) - u_p * self.S_uw  # self.tr_u() * self.S_uw
        du = self.Eps_u * self.I_priv - self.g_u * (self.u - self.B_u) - w_p * self.S_wu  # self.tr_w() * self.S_wu
        self.w += dw
        self.u += du
        if self.w > self.w_max:
            self.w = self.w_max
        if self.w < -self.w_max:
            self.w = -self.w_max
        if self.u > self.u_max:
            self.u = self.u_max
        if self.u < -self.u_max:
            self.u = -self.u_max

    def update(self, agents):
        """
        main update method of the agent. This method is called in every timestep to calculate the new state/position
        of the agent and visualize it in the environment
        :param agents: a list of all obstacle/agents coordinates as (X, Y) in the environment. These are not necessarily
                socially relevant, i.e. all agents.
        """
        # calculate socially relevant projection field (Vsoc and Vsoc+)
        self.calc_social_V_proj(agents)

        # calculate private information
        self.calc_I_priv()

        # update inner decision process according to visual field and private info
        self.update_decision_processes()

        # CALCULATING velocity and orientation change according to inner decision process (dv)
        # we use if and not a + operator as this is less computationally heavy but the 2 is equivalent
        # vel, theta = int(self.tr_w()) * VSWRM_flocking_state_variables(...) + (1 - int(self.tr_w())) * random_walk(...)
        # or later when we define the individual and social forces
        # vel, theta = int(self.tr_w()) * self.F_soc(...) + (1 - int(self.tr_w())) * self.F_exp(...)
        if not self.get_mode() == "collide":
            if not self.tr_w() and not self.tr_u():
                vel, theta = supcalc.random_walk()
                self.set_mode("explore")
            elif self.tr_w() and self.tr_u():
                self.set_mode("exploit")
                vel, theta = (-self.velocity * 0.08, 0)
            elif self.tr_w() and not self.tr_u():
                vel, theta = supcalc.VSWRM_flocking_state_variables(self.velocity,
                                                                    np.linspace(-np.pi, np.pi, self.v_field_res),
                                                                    self.soc_v_field)
                # WHY ON EARTH DO WE NEED THIS NEGATION?
                # whatever comes out has a sign that tells if the change in direction should be left or right
                # seemingly what comes out has a different convention than our environment?
                # VSWRM: comes out + turn left? comes our - turn right?
                # environment: the opposite way around
                theta = -theta
                self.set_mode("relocate")
            elif self.tr_u() and not self.tr_w():
                self.set_mode("exploit")
                vel, theta = (-self.velocity * 0.04, 0)
        else:
            # COLLISION AVOIDANCE IS ACTIVE, let that guide us
            # As we don't have proximity sensor interface as with e.g. real robots we will let
            # the environment to enforce us into a collision maneuver from the simulation environment
            # so we don't change the current velocity from here.
            vel, theta = (0, 0)

        if not self.is_moved_with_cursor:  # we freeze agents when we move them
            # updating agent's state variables according to calculated vel and theta
            self.orientation += theta
            self.prove_orientation()  # bounding orientation into 0 and 2pi
            self.velocity += vel
            self.prove_velocity()  # possibly bounding velocity of agent

            # updating agent's position
            self.position[0] += self.velocity * np.cos(self.orientation)
            self.position[1] -= self.velocity * np.sin(self.orientation)

            # boundary conditions if applicable
            self.reflect_from_walls()

        # updating agent visualization
        self.draw_update()
        self.collected_r_before = self.collected_r

    def change_color(self):
        """Changing color of agent according to the behavioral mode the agent is currently in."""
        if self.get_mode() == "explore":
            self.color = colors.BLUE
        elif self.get_mode() == "flock" or self.get_mode() == "relocate":
            self.color = colors.PURPLE
        elif self.get_mode() == "collide":
            self.color = colors.RED
        elif self.get_mode() == "exploit":
            self.color = colors.GREEN
        elif self.get_mode() == "pool":
            self.color = colors.YELLOW

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
        # visible agents (exluding self)
        agents = [ag for ag in agents if supcalc.distance(self, ag) <= self.vision_range]
        # those of them that are exploiting
        expl_agents = [ag for ag in agents if ag.id != self.id
                       and ag.get_mode() == "exploit"]
        if self.exclude_agents_same_patch:
            expl_agents = [ag for ag in expl_agents if ag.exploited_patch_id!=self.exploited_patch_id]

        if self.visual_exclusion:
            # soc_proj_f_wo_exc = self.projection_field(expl_agents_coords, keep_distance_info=True)
            # non_soc_proj_f = self.projection_field(other_agents_coord, keep_distance_info=True)
            # # calculating visual exclusion
            # soc_proj_f = soc_proj_f_wo_exc - non_soc_proj_f
            # soc_proj_f[soc_proj_f < 0] = 0
            # # setting back to binary v field
            # soc_proj_f[soc_proj_f > 0] = 1
            # self.soc_v_field = soc_proj_f
            raise Exception("Visual exclusion is not supported in the current version!")
        else:
            self.soc_v_field = self.projection_field(expl_agents, keep_distance_info=True)
            # self.soc_v_field[self.soc_v_field != 0] = 1
            # if self.id == 0:
            #     print(np.unique(self.soc_v_field))

    def projection_field(self, obstacles, keep_distance_info=False):
        """Calculating visual projection field for the agent given the visible obstacles in the environment
        :param obstacles: list of agents (with same radius) or some other obstacle sprites to generate projection field
        :param keep_distance_info: if True, the amplitude of the vpf will reflect the distance of the object from the
            agent so that exclusion can be easily generated with a single computational step."""

        # extracting obstacle coordinates
        obstacle_coords = [ob.position for ob in obstacles]

        # initializing visual field and relative angles
        v_field = np.zeros(self.v_field_res)
        phis = np.linspace(-np.pi, np.pi, self.v_field_res)

        # center point
        v1_s_x = self.position[0] + self.radius
        v1_s_y = self.position[1] + self.radius

        # point on agent's edge circle according to it's orientation
        v1_e_x = (1 + np.cos(self.orientation)) * self.radius
        v1_e_y = (1 - np.sin(self.orientation)) * self.radius

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
                # I HAVE NO IDEA WHY IS IT SHIFTED WITH PI/4?
                closed_angle = supcalc.angle_between(v1, v2) + self.orientation + np.pi / 4
                if closed_angle > np.pi:
                    closed_angle -= 2 * np.pi
                if closed_angle < -np.pi:
                    closed_angle += 2 * np.pi

                # calculating size of the projection on the retina
                c1 = np.array([v1_s_x, v1_s_y])
                c2 = np.array([v2_e_x, v2_e_y])
                distance = np.linalg.norm(c2 - c1)
                vis_angle = 2 * np.arctan(self.radius / (1 * distance))
                proj_size = (vis_angle / (2*np.pi)) * self.v_field_res

                # placing the projection on the VPF of agent
                phi_target = supcalc.find_nearest(phis, closed_angle)

                proj_start = int(phi_target - proj_size / 2)
                proj_end = int(phi_target + proj_size / 2)

                self.vis_field_source_data[i] = {}
                self.vis_field_source_data[i]["vis_angle"] = vis_angle
                self.vis_field_source_data[i]["phi_target"] = phi_target

                # circular boundaries to the VPF as there is 360 degree vision
                if proj_start < 0:
                    v_field[self.v_field_res + proj_start:self.v_field_res] = 1
                    proj_start = 0

                if proj_end >= self.v_field_res:
                    v_field[0:proj_end - self.v_field_res] = 1
                    proj_end = self.v_field_res - 1

                if not keep_distance_info:
                    v_field[proj_start:proj_end] = 1
                else:
                    v_field[proj_start:proj_end] = (1 - distance/self.vision_range)

        # post_processing and limiting FOV
        v_field_post = np.roll(v_field, int(len(v_field) / 2))
        v_field_post[phis < self.FOV[0]] = 0
        v_field_post[phis > self.FOV[1]] = 0

        return v_field_post

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
