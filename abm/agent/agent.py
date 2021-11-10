"""
agent.py : including the main classes to create an agent. Supplementary calculations independent from class attributes
            are removed from this file.
"""

import pygame
import numpy as np
from abm.contrib import colors
from abm.agent import supcalc


class Agent(pygame.sprite.Sprite):
    """
    Agent class that includes all private parameters of the agents and all methods necessary to move in the environment
    and to make decisions.
    """

    def __init__(self, id, radius, position, orientation, env_size, color, v_field_res, window_pad):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of agent (int)
        :param radius: radius of the agent in pixels
        :param position: position of the agent in env as (x, y)
        :param orientation: absolute orientation of the agent
        :param env_size: environment size available for agents as (width, height)
        :param color: color of the agent as (R, G, B)
        :param v_field_res: resolution of the visual field of the agent in pixels
        :param window_pad: padding of the environment in simulation window in pixels
        """
        # Initializing supercalss (Pygame Sprite)
        super().__init__()

        # Initializing agents with init parameters
        self.id = id
        self.radius = radius
        self.position = np.array(position, dtype=np.float64)
        self.orientation = orientation
        self.color = color
        self.v_field_res = v_field_res
        self.v_field = np.zeros(self.v_field_res)

        # Non-initializable private attributes
        self.velocity = 0  # agent absolute velocity
        self.collected_r = 0  # collected rescource unit collected by agent
        self.mode = "explore"  # could be something like Explore, Flock, Exploit, etc.

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

    def update(self, obstacles):
        """
        main update method of the agent. This method is called in every timestep to calculate the new state/position
        of the agent and visualize it in the environment
        :param obstacles: a list of visible obstacle coordinates as (X, Y) in the environment
        """

        # calculating velocity and orientation change according behavioral mode
        if self.mode == "flock":
            # calculating projection field of agent (vision)
            self.projection_field(obstacles)
            # flocking according to VSWRM
            vel, theta = supcalc.VSWRM_flocking_state_variables(self.velocity, np.linspace(-np.pi, np.pi, self.v_field_res),
                                                         self.v_field)
        elif self.mode == "explore" or self.mode == "collide":
            # exploring with some random process
            vel, theta = supcalc.random_walk()
        elif self.mode == "exploit":
            # exploiting resource and can not move but might be able to turn
            # vel, theta = supcalc.random_walk()
            # vel = -self.velocity # stopping the agent but can still turn around
            vel, theta = (0, 0)

        # updating agent's state variables
        self.orientation += theta
        self.prove_orientation()  # bounding orientation into 0 and 2pi
        self.velocity += vel
        self.prove_velocity()  # possibly bounding velocity of agent

        # updating agent's position
        self.position[0] += self.velocity * np.cos(-self.orientation)
        self.position[1] += self.velocity * np.sin(-self.orientation)

        # boundary conditions if applicable
        self.reflect_from_walls()

        # updating agent visualization
        self.draw_update()

    def change_color(self):
        """Changing color of agent according to the behavioral mode the agent is currently in."""
        if self.mode == "explore":
            self.color = colors.BLUE
        elif self.mode == "flock":
            self.color = colors.PURPLE
        elif self.mode == "collide":
            self.color = colors.RED
        elif self.mode == "exploit":
            self.color = colors.GREEN

    def draw_update(self):
        """
        updating the outlook of the agent according to position and orientation
        """
        # update position
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]

        # cahnge agent color according to mode
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
                self.orientation -= np.pi / 4
            elif np.pi <= self.orientation <= 3 * np.pi / 2:
                self.orientation += np.pi / 4

        # Reflection from right wall
        if x > self.boundaries_x[1]:

            self.position[0] = self.boundaries_x[1] - self.radius - 1

            if 3 * np.pi / 2 <= self.orientation < 2 * np.pi:
                self.orientation -= np.pi / 4
            elif 0 <= self.orientation <= np.pi / 2:
                self.orientation += np.pi / 4

        # Reflection from upper wall
        if y < self.boundaries_y[0]:
            self.position[1] = self.boundaries_y[0] - self.radius

            if np.pi / 2 <= self.orientation <= np.pi:
                self.orientation += np.pi / 4
            elif 0 <= self.orientation < np.pi / 2:
                self.orientation -= np.pi / 4

        # Reflection from lower wall
        if y > self.boundaries_y[1]:
            self.position[1] = self.boundaries_y[1] - self.radius - 1

            if 3 * np.pi / 2 <= self.orientation <= 2 * np.pi:
                self.orientation += np.pi / 4
            elif np.pi <= self.orientation < 3 * np.pi / 2:
                self.orientation -= np.pi / 4

    def projection_field(self, obstacle_coords):
        """Calculating visual projection field for the agent given the visible obstacles in the environment"""
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
        for obstacle_coord in obstacle_coords:
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
                closed_angle = supcalc.angle_between(v1, v2) + self.orientation
                if closed_angle > np.pi:
                    closed_angle -= 2 * np.pi
                if closed_angle < -np.pi:
                    closed_angle += 2 * np.pi

                # calculating size of the projection on the retina
                c1 = np.array([v1_s_x, v1_s_y])
                c2 = np.array([v2_e_x, v2_e_y])
                distance = np.linalg.norm(c2 - c1)
                vis_angle = 2 * np.arctan(self.radius / (1 * distance))
                proj_size = 300 * vis_angle

                # placing the projection on the VPF of agent
                phi_target = supcalc.find_nearest(phis, closed_angle)

                proj_start = int(phi_target - proj_size / 2)
                proj_end = int(phi_target + proj_size / 2)

                # circular boundaries to the VPF as there is 360 degree vision
                if proj_start < 0:
                    v_field[self.v_field_res + proj_start:self.v_field_res] = 1
                    proj_start = 0

                if proj_start >= self.v_field_res:
                    v_field[0:proj_start - self.v_field_res] = 1
                    proj_start = self.v_field_res - 1

                v_field[proj_start:proj_end] = 1

        self.v_field = v_field

    def prove_orientation(self):
        """Restricting orientation angle between 0 and 2 pi"""
        if self.orientation < 0:
            self.orientation = 2 * np.pi + self.orientation
        if self.orientation > np.pi * 2:
            self.orientation = 2 * np.pi - self.orientation

    def prove_velocity(self, velocity_limit=1):
        """Restricting the absolute velocity of the agent"""
        vel_sign = np.sign(self.velocity)
        if vel_sign == 0:
            vel_sign = +1
        if self.mode == 'explore':
            if np.abs(self.velocity) > velocity_limit:
                # stopping agent if too fast during exploration
                self.velocity = 1
