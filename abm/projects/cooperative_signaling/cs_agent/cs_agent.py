from random import random

import numpy as np
import pygame

from abm.projects.cooperative_signaling.cs_agent.cs_supcalc import \
    reflection_from_circular_wall, random_walk, f_reloc_lr, phototaxis, \
    signaling, agent_decision, projection_field
from abm.agent.supcalc import distance
from abm.projects.cooperative_signaling.cs_contrib import cs_params
from abm.agent.agent import Agent
from abm.contrib import colors


class CSAgent(Agent):
    def __init__(self, phototaxis_theta_step, detection_range,
                 resource_meter_multiplier, signalling_cost,
                 probability_of_starting_signaling, **kwargs):
        super().__init__(**kwargs)

        # creating agent status
        # exploration, taxis, relocation or flocking
        self.agent_state = "exploration"
        self.meter = 0  # between 0 and 1
        self.prev_meter = 0  # for phototaxis
        self.theta_prev = 0  # turning angle in prev timestep
        self.taxis_dir = None  # phototaxis direction [-1, 1, None]

        # maximum turning angle during phototaxis
        self.phototaxis_theta_step = phototaxis_theta_step
        self.detection_range = detection_range
        # for unit detected resource value how much resource should I gain
        self.resource_meter_multiplier = resource_meter_multiplier

        # signaling
        self.signalling_cost = signalling_cost
        self.is_signaling = False
        self.signaling_marker_radius = 5
        self.signalling_color = colors.GREEN
        self.probability_of_starting_signaling = \
            probability_of_starting_signaling
        self.signaling_rand_event = False
        self.signaling_rand_value = random()

        # flocking
        self.flocking_probability = 0.01
        # start flocking if all agents are farther than
        self.flocking_from_distance = 100
        # stop flocking if there is an agent closer than
        self.flocking_to_distance = 50

        # social information: visual field projections
        self.crowd_proj = np.zeros(self.v_field_res)
        self.signaling_proj = np.zeros(self.v_field_res)
        # memory to store the coordinates of previously signaling agents
        self.memory_signaling = []

    def update(self, agents):
        """
        main update method of the agent. This method is called in every timestep
        to calculate the new state/position
        of the agent and visualize it in the environment
        :param agents: a list of all obstacle/agents coordinates as (X, Y) in
        the environment. These are not necessarily socially relevant, i.e. all
        agents.
        """
        # update agent information
        self.update_social_info(agents)

        # update agent's state
        self.update_state()

        # perform agent's action i.e. exploration, taxis, relocation or flocking
        self.perform_action()

        # update agent's signaling behavior
        self.update_signaling()

        # updating agent visualization
        self.draw_update()

        # collecting rewards according to meter value and signalling status
        self.update_rewards()

    def update_social_info(self, agents):
        # calculate socially relevant projection field (e.g. according to
        # signalling agents)
        self.crowd_proj = self.calc_crowing_density_proj(agents)
        self.signaling_proj = self.calc_others_signaling_density_proj(agents)
        self.interagentdistances = [distance(agent, self) for agent in agents if agent is not self]

    def calc_crowing_density_proj(self, agents, max_proj_size_percentage=cs_params.max_proj_size_percentage):
        """
        :param agents: agents
        :param max_proj_size_percentage: crowding only works if proj size is smaller than some percentage of vfield.
        Default is 5%.
        """
        visual_field = projection_field(
            fov=self.FOV,
            v_field_resolution=self.v_field_res,
            position=np.array(self.position),
            radius=self.radius,
            orientation=self.orientation,
            object_positions=[np.array(ag.position) for ag in agents if ag is not self],
            object_meters=None,  # not relevant for crowding density
            max_proj_size=self.v_field_res * max_proj_size_percentage)
        # sum of all agents projections at each point in visual field
        svfield = visual_field.sum(axis=0)
        # normalizing the visual field with number of agents
        normed_v_field = svfield / len(agents)
        # for debug reasons we pass this as static foraging social projection (for visualization)
        self.soc_v_field = normed_v_field
        return normed_v_field

    def calc_others_signaling_density_proj(self, agents, memory_depth=cs_params.memory_depth):
        current_pos = [np.array(ag.position) for ag in agents if ag.is_signaling if ag is not self]
        current_meters = [ag.meter for ag in agents if ag.is_signaling if ag is not self]

        # update memory: insert current position and meter at the beginning of the list
        self.memory_signaling.insert(0, (current_pos, current_meters))

        # combine memory and the current state
        pos, meters = self.extend_signaling_agent_positions_with_memory(memory_depth)

        # continue if nobody was signaling
        if len(pos) == 0:
            return np.zeros_like(self.signaling_proj)

        visual_field = projection_field(
            fov=self.FOV,
            v_field_resolution=self.v_field_res,
            position=np.array(self.position),
            radius=self.radius,
            orientation=self.orientation,
            object_positions=pos,
            object_meters=meters)

        # max signal at each point in visual field
        signaling_proj = visual_field.max(axis=0)
        return signaling_proj

    def extend_signaling_agent_positions_with_memory(self, memory_depth):
        pos, meters = [], []
        for n, (p, m) in enumerate(self.memory_signaling):
            decay_factor = 1 - n / memory_depth
            # append positions if somebody was signaling in the last n's timestep
            if p:
                pos.extend(p)
                meters.extend([_m * decay_factor for _m in m])
            if n >= memory_depth:
                # remove memory if it is older than memory_depth
                self.memory_signaling.pop(-1)
                break
        return pos, meters

    def update_state(self):
        # update agent state based on the decision-making process
        self.agent_state = agent_decision(
            meter=self.meter,
            max_signal_of_other_agents=self.signaling_proj.max(initial=0),
            agent_state=self.agent_state,
            agent_distances=self.interagentdistances,
            visual_field=self.crowd_proj,
            flocking_probability=self.flocking_probability,
            flocking_from_distance=self.flocking_from_distance,
            flocking_to_distance=self.flocking_to_distance)

    def perform_action(self):
        # update agent color

        # we freeze agents when we move them
        if not self.is_moved_with_cursor:
            # perform the agent's action according to the current state
            if self.agent_state == "exploration":
                self.exploration()
            elif self.agent_state == "taxis":
                self.taxis()
            elif self.agent_state == "relocation":
                self.relocation()
            elif self.agent_state == "flocking":
                self.flocking()
        else:
            print(self.meter)

    def exploration(self):
        vel, theta = random_walk(desired_vel=self.max_exp_vel)
        vel = (cs_params.max_speed - self.velocity)
        self.update_agent_position(theta, vel)

    def taxis(self, max_speed=cs_params.max_speed):
        theta, self.taxis_dir = phototaxis(
            self.meter,
            self.prev_meter,
            self.theta_prev,
            self.taxis_dir,
            self.phototaxis_theta_step)

        vel = (max_speed - self.velocity)
        self.update_agent_position(theta, vel)

    def relocation(self, max_speed=cs_params.max_speed):
        vel, theta = f_reloc_lr(self.velocity,
                                self.signaling_proj,
                                velocity_desired=max_speed,
                                theta_max=2.5)
        self.update_agent_position(theta, vel)

    def flocking(self, max_speed=cs_params.max_speed):
        vel, theta = f_reloc_lr(self.velocity,
                                self.crowd_proj,
                                velocity_desired=max_speed,
                                theta_max=2.5)
        self.update_agent_position(theta, vel)

    def update_agent_position(self, theta, vel):
        # updating agent's state variables according to calculated vel and
        # theta
        self.orientation += theta
        # storing theta in short term memory for phototaxis
        self.theta_prev = theta
        self.prove_orientation()  # bounding orientation into 0 and 2pi
        self.velocity += vel
        self.prove_velocity()  # possibly bounding velocity of agent

        # new agent's position
        new_pos = (
            self.position[0] + self.velocity * np.cos(self.orientation),
            self.position[1] - self.velocity * np.sin(self.orientation)
        )
        # update the agent's position with constraints (reflection from the
        # walls) or with the new position
        self.position = list(self.reflect_from_walls(new_pos))

    def update_signaling(self):
        # update random value when this event is triggered
        if self.signaling_rand_event:
            # updated in every N timesteps from cs_sims
            self.signaling_rand_value = random()
            self.signaling_rand_event = False
        # update agent's signaling behavior
        self.is_signaling = signaling(
            self.meter, self.is_signaling, self.signalling_cost,
            self.probability_of_starting_signaling, self.signaling_rand_value)

    def update_rewards(self):
        """
        Updating agent collected resource values according to distance from
        resource (as in meter value) and current signalling status
        """
        self.collected_r_before = self.collected_r

        self.collected_r += self.meter * self.resource_meter_multiplier
        if self.is_signaling:
            self.collected_r -= self.signalling_cost

    def change_color(self):
        """
        Changing color of agent according to the behavioral mode the agent is
        currently in.
        """
        if self.agent_state == "exploration":
            self.color = colors.BLUE
        elif self.agent_state == "taxis":
            self.color = colors.RED
        elif self.agent_state == "relocation":
            self.color = colors.PURPLE
        elif self.agent_state == "flocking":
            self.color = colors.GREEN

    def draw_update(self):
        """
        Updating the agent's visualization according to the current behavioral
        mode of the agent
        """
        # run the basic draw update method
        super().draw_update()

        # draw signaling marker
        if self.is_signaling:
            pygame.gfxdraw.filled_circle(
                self.image,
                self.radius,
                self.radius,
                self.signaling_marker_radius,
                self.signalling_color
            )
            pygame.gfxdraw.aacircle(self.image,
                                    self.radius,
                                    self.radius,
                                    self.signaling_marker_radius,
                                    colors.BACKGROUND)

    def prove_velocity(self, velocity_limit=cs_params.max_speed):
        """Restricting the absolute velocity of the agent"""
        if self.agent_state == 'exploration':
            if np.abs(self.velocity) > velocity_limit:
                # stopping agent if too fast during exploration
                self.velocity = velocity_limit

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
