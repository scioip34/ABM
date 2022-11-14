import numpy as np

from abm.agent import supcalc
from abm.projects.cooperative_signaling.cs_agent.cs_supcalc import \
    reflection_from_circular_wall, random_walk, F_reloc_LR, phototaxis
from abm.agent.agent import Agent
from abm.contrib import colors


class CSAgent(Agent):
    def __init__(self, phototaxis_theta_step, detection_range,
                 resource_meter_multiplier, signalling_cost, **kwargs):
        super().__init__(**kwargs)

        # creating agent status
        # todo: mutually exclusive states mean that agents can not signal while
        #  relocate
        self.agent_type = "mars_miner"
        self.meter = 0  # between 0 and 1
        self.prev_meter = 0  # for phototaxis
        self.theta_prev = 0  # turning angle in prev timestep
        self.taxis_dir = None  # phototaxis direction [-1, 1, None]

        # maximum turning angle during phototaxis
        self.phototaxis_theta_step = phototaxis_theta_step
        self.detection_range = detection_range
        # for unit detected resource value how much resource should I gain
        self.resource_meter_multiplier = resource_meter_multiplier
        self.signalling_cost = signalling_cost

        # social visual projection field
        self.target_field = np.zeros(self.v_field_res)

    def update(self, agents):
        """
        main update method of the agent. This method is called in every timestep
        to calculate the new state/position
        of the agent and visualize it in the environment
        :param agents: a list of all obstacle/agents coordinates as (X, Y) in
        the environment. These are not necessarily socially relevant, i.e. all
        agents.
        """
        # calculate socially relevant projection field (e.g. according to
        # signalling agents)
        self.calc_social_V_proj(agents)

        # some basic decision process of when to signal and when to explore, etc.
        signalling_threshold = 0.1
        if np.max(self.soc_v_field) > self.meter:
            # joining behavior
            vel, theta = F_reloc_LR(self.velocity, self.soc_v_field, 2,
                                    theta_max=2.5)
            self.agent_type = "relocation"
            if self.meter > signalling_threshold:
                self.agent_type = "signalling"

        else:
            if self.meter > 0:
                theta, taxis_dir = phototaxis(
                    self.meter,
                    self.prev_meter,
                    self.theta_prev,
                    self.taxis_dir,
                    self.phototaxis_theta_step)
                self.taxis_dir = taxis_dir
                vel = (2 - self.velocity)
                self.agent_type = "mars_miner"
                if self.meter > signalling_threshold:
                    self.agent_type = "signalling"
            else:
                # carry out movement accordingly
                vel, theta = random_walk(desired_vel=self.max_exp_vel)
                vel = (2 - self.velocity)
                self.agent_type = "mars_miner"

        # updating position accordingly
        if not self.is_moved_with_cursor:  # we freeze agents when we move them
            # updating agent's state variables according to calculated vel and
            # theta
            self.orientation += theta
            # storing theta in short term memory for phototaxis
            self.theta_prev = theta
            self.prove_orientation()  # bounding orientation into 0 and 2pi
            self.velocity += vel
            # self.prove_velocity()  # possibly bounding velocity of agent

            # new agent's position
            new_pos = (
                self.position[0] + self.velocity * np.cos(self.orientation),
                self.position[1] - self.velocity * np.sin(self.orientation)
            )

            # update the agent's position with constraints (reflection from the
            # walls) or with the new position
            self.position = list(self.reflect_from_walls(new_pos))
        else:
            # self.agent_type = "signalling"
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
        """
        Changing color of agent according to the behavioral mode the agent is
        currently in.
        """
        if self.agent_type == "mars_miner":
            self.color = colors.BLUE
        elif self.agent_type == "signalling":
            self.color = colors.RED
        elif self.agent_type == "relocation":
            self.color = colors.PURPLE

    def calc_social_V_proj(self, agents):
        """
        Calculating the socially relevant visual projection field of the agent.
        This is calculated as theprojection of nearby exploiting agents that are
        not visually excluded by other agents
        """
        signalling = [ag for ag in agents if ag.agent_type == "signalling"]
        self.soc_v_field = self.projection_field(signalling,
                                                 keep_distance_info=True)

    def projection_field(self, obstacles, keep_distance_info=False,
                         non_expl_agents=None, fov=None):
        """
        Calculating visual projection field for the agent given the visible
        obstacles in the environment
        :param obstacles: list of agents (with same radius) or some other
        obstacle sprites to generate projection field
        :param keep_distance_info: if True, the amplitude of the vpf will
        reflect the distance of the object from the agent so that exclusion can
        be easily generated with a single computational step
        :param non_expl_agents: a list of non-social visual cues (non-exploiting
        agents) that on the other hand can still produce visual exclusion on the
        projection of social cues. If None only social cues can produce visual
        exclusion on each other
        :param fov: touple of number with borders of fov such as (-np.pi,np.pi),
        if None, self.FOV will be used
        """
        # deciding fov
        if fov is None:
            fov = self.FOV

        # extracting obstacle coordinates
        obstacle_coords = [ob.position for ob in obstacles]
        meters = [ob.meter for ob in obstacles]

        # if non-social cues can visually exclude social ones we also
        # concatenate these to the obstacle coords
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
        # calculating closed angle between obstacle and agent according to the
        # position of the obstacle.
        # then calculating visual projection size according to visual angle on
        # the agents's retina according to distance
        # between agent and obstacle
        self.vis_field_source_data = {}
        for i, obstacle_coord in enumerate(obstacle_coords):
            if not (obstacle_coord[0] == self.position[0] and obstacle_coord[
                1] == self.position[1]):
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
                vis_angle = 2 * np.arctan(self.radius / (1 * distance))
                # finding where in the retina the projection belongs to
                phi_target = supcalc.find_nearest(phis, closed_angle)
                # if target is visible we save its projection into the VPF
                # source data
                if fov[0] < closed_angle < fov[1]:
                    self.vis_field_source_data[i] = {}
                    self.vis_field_source_data[i]["vis_angle"] = vis_angle
                    self.vis_field_source_data[i]["phi_target"] = phi_target
                    self.vis_field_source_data[i]["distance"] = distance
                    self.vis_field_source_data[i]["meter"] = meters[i]
                    # the projection size is proportional to the visual angle.
                    # If the projection is maximal (i.e.
                    # taking each pixel of the retina) the angle is 2pi from
                    # this we just calculate the proj. size
                    # using a single proportion
                    self.vis_field_source_data[i]["proj_size"] = (vis_angle / (
                            2 * np.pi)) * self.v_field_res
                    proj_size = self.vis_field_source_data[i]["proj_size"]
                    self.vis_field_source_data[i]["proj_start"] = int(
                        phi_target - proj_size / 2)
                    self.vis_field_source_data[i]["proj_end"] = int(
                        phi_target + proj_size / 2)
                    self.vis_field_source_data[i]["proj_start_ex"] = int(
                        phi_target - proj_size / 2)
                    self.vis_field_source_data[i]["proj_end_ex"] = int(
                        phi_target + proj_size / 2)
                    self.vis_field_source_data[i]["proj_size_ex"] = proj_size
                    if non_expl_agents is not None:
                        if i < len_social:
                            self.vis_field_source_data[i][
                                "is_social_cue"] = True
                        else:
                            self.vis_field_source_data[i][
                                "is_social_cue"] = False
                    else:
                        self.vis_field_source_data[i]["is_social_cue"] = True
        # calculating visual exclusion if requested
        if self.visual_exclusion:
            self.exlude_V_source_data()
        if non_expl_agents is not None:
            # removing non-social cues from the source data after calculating
            # exclusions
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

    def prove_velocity(self, velocity_limit=1):
        """Restricting the absolute velocity of the agent"""
        vel_sign = np.sign(self.velocity)
        if vel_sign == 0:
            vel_sign = +1
        if self.get_mode() == 'explore':
            if np.abs(self.velocity) > velocity_limit:
                # stopping agent if too fast during exploration
                self.velocity = 1

    def reflect_from_walls(self, new_pos=()):
        """
        Reflecting agent from the circle arena border.
        """
        # x coordinate - x of the center point of the circle
        x = new_pos[0] + self.radius
        c_x = (self.WIDTH / 2 + self.window_pad)
        dx = x - c_x
        # y coordinate - y of the center point of the circle
        y = new_pos[1] + self.radius
        c_y = (self.HEIGHT / 2 + self.window_pad)
        dy = y - c_y
        # radius of the environment
        e_r = self.HEIGHT / 2

        # return if the agent has not reached the boarder
        if np.linalg.norm([dx, dy]) + self.radius < e_r:
            return new_pos

        # reflect the agent from the boarder
        self.orientation = reflection_from_circular_wall(
            dx, dy, self.orientation)

        # make orientation between 0 and 2pi
        self.prove_orientation()

        # relocate the agent back inside the circle
        new_pos = (
            self.position[0] + self.velocity * np.cos(self.orientation),
            self.position[1] - self.velocity * np.sin(self.orientation)
        )
        # check if the agent is still outside the circle
        diff = [new_pos[0] - c_x, new_pos[1] - c_y]
        if np.linalg.norm(diff) + self.radius >= e_r:
            # if yes, relocate it again
            dist = np.linalg.norm(diff) + self.radius - e_r
            new_pos = (
                self.position[0] + dist * np.cos(self.orientation),
                self.position[1] - dist * np.sin(self.orientation)
            )
        return new_pos
