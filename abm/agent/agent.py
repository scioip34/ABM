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

    def __init__(self, radius, position, orientation, env_size, color, v_field_res):
        """
        Initalization method of main agent class of the simulations

        :param radius: radius of the agent in pixels
        :param position: position of the agent in env as (x, y)
        :param orientation: absolute orientation of the agent
        :param env_size: environment size available for agents as (width, height)
        :param color: color of the agent as (R, G, B)
        :param v_field_res: resolution of the visual field of the agent in pixels
        """
        # Initializing supercalss (Pygame Sprite)
        super().__init__()

        # Initializing agents with init parameters
        self.radius = radius
        self.position = np.array(position, dtype=np.float64)
        self.velocity = 0
        self.orientation = orientation
        self.WIDTH = env_size[0]  # env width
        self.HEIGHT = env_size[1]  # env height
        self.color = color
        # self.sensor_range = 200
        # self.sensor_line_resolution = 50
        self.v_field_res = v_field_res
        self.v_field = np.zeros(self.v_field_res)

        # creating visualization surface for agent as a filled circle
        self.image = pygame.Surface([radius * 2, radius * 2])
        self.image.fill(colors.BACKGROUND)
        pygame.draw.circle(
            self.image, color, (radius, radius), radius
        )

        # showing agent orientation with a line towards agent orientation
        pygame.draw.line(self.image, colors.BLACK, (radius, radius),
                         ((1 + np.cos(self.orientation)) * radius, (1 - np.sin(self.orientation)) * radius), 3)
        self.rect = self.image.get_rect()

    def update(self, obstacles):
        """
        main update method of the agent. This method is called in every timestep to calculate the new state/position
        of the agent and visualize it in the environment
        :param obstacles: a list of obstacle coordinates as (X, Y)
        """

        # calculating projection field of agent (vision)
        self.projection_field(obstacles)

        # calculating velcoity and orientation change according to visual cues
        vel, theta = supcalc.compute_state_variables(self.velocity, np.linspace(-np.pi, np.pi, self.v_field_res),
                                                     self.v_field)

        # updateing agent's state variables
        self.orientation += theta
        self.prove_orientation()
        self.velocity += vel
        self.position[0] += self.velocity * np.cos(-self.orientation)
        self.position[1] += self.velocity * np.sin(-self.orientation)

        # boundary conditions
        self.reflect_from_walls()

        # updating the oulook of the agent
        self.draw_update()

    def draw_update(self):
        """
        updating the outlook of the agent according to position and orientation
        """
        # update position
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]

        # update surface according to new orientation
        # creating visualization surface for agent as a filled circle
        self.image = pygame.Surface([self.radius * 2, self.radius * 2])
        self.image.fill(colors.BACKGROUND)
        pygame.draw.circle(
            self.image, self.color, (self.radius, self.radius), self.radius
        )
        # showing agent orientation with a line towards agent orientation
        pygame.draw.line(self.image, colors.BLACK, (self.radius, self.radius),
                         ((1 + np.cos(self.orientation)) * self.radius, (1 - np.sin(self.orientation)) * self.radius),
                         3)

    def reflect_from_walls(self):
        """reflecting agent from environment boundaries according to a desired x, y coordinate. If this is over any
        boundaries of the environment, the agents position and orientation will be changed such that the agent is
         reflected from these boundaries."""

        x = self.position[0]
        y = self.position[1]

        # Reflection from left wall
        if x < 0:
            self.position[0] = 0

            if np.pi / 2 <= self.orientation < np.pi:
                self.orientation -= np.pi / 4
            elif np.pi <= self.orientation <= 3 * np.pi / 2:
                self.orientation += np.pi / 4

        # Reflection from right wall
        if x > self.WIDTH:

            self.position[0] = self.WIDTH - 1

            if 3 * np.pi / 2 <= self.orientation < 2 * np.pi:
                self.orientation -= np.pi / 4
            elif 0 <= self.orientation <= np.pi / 2:
                self.orientation += np.pi / 4

        # Reflection from upper wall
        if y < 0:
            self.position[1] = 0

            if np.pi / 2 <= self.orientation <= np.pi:
                self.orientation += np.pi / 4
            elif 0 <= self.orientation < np.pi / 2:
                self.orientation -= np.pi / 4

        # Reflection from lower wall
        if y > self.HEIGHT:
            self.position[1] = self.HEIGHT - 1

            if 3 * np.pi / 2 <= self.orientation <= 2 * np.pi:
                self.orientation += np.pi / 4
            elif np.pi <= self.orientation < 3 * np.pi / 2:
                self.orientation -= np.pi / 4

    def projection_field(self, obstacle_coords):
        """Calculating visual projection field for the agent given the known obstacles in the environment"""
        v_field = np.zeros(self.v_field_res)
        phis = np.linspace(-np.pi, np.pi, self.v_field_res)

        v1_s_x = self.position[0] + self.radius
        v1_s_y = self.position[1] + self.radius

        v1_e_x = (1 + np.cos(self.orientation)) * self.radius
        v1_e_y = (1 - np.sin(self.orientation)) * self.radius

        v1_x = v1_e_x - v1_s_x
        v1_y = v1_e_y - v1_s_y

        v1 = np.array([v1_x, v1_y])

        for obstacle_coord in obstacle_coords:
            if not (obstacle_coord[0] == self.position[0] and obstacle_coord[1] == self.position[1]):
                v2_e_x = obstacle_coord[0] + self.radius
                v2_e_y = obstacle_coord[1] + self.radius

                v2_x = v2_e_x - v1_s_x
                v2_y = v2_e_y - v1_s_y

                v2 = np.array([v2_x, v2_y])

                closed_angle = supcalc.angle_between(v1, v2) + self.orientation
                if closed_angle > np.pi:
                    closed_angle -= 2 * np.pi
                if closed_angle < -np.pi:
                    closed_angle += 2 * np.pi

                c1 = np.array([v1_s_x, v1_s_y])
                c2 = np.array([v2_e_x, v2_e_y])
                distance = np.linalg.norm(c2 - c1)
                vis_angle = 2 * np.arctan(self.radius / (1 * distance))
                proj_size = 300 * vis_angle
                phi_target = supcalc.find_nearest(phis, closed_angle)

                proj_start = int(phi_target - proj_size / 2)
                proj_end = int(phi_target + proj_size / 2)

                if proj_start < 0:
                    v_field[self.v_field_res + proj_start:self.v_field_res] = 1
                    proj_start = 0

                if proj_start >= self.v_field_res:
                    v_field[0:proj_start - self.v_field_res] = 1
                    proj_start = self.v_field_res - 1

                v_field[proj_start:proj_end] = 1

        self.v_field = v_field

    def prove_orientation(self):
        """Restricting orientation angle between o and 2 pi"""
        if self.orientation < 0:
            self.orientation = 2 * np.pi + self.orientation
        if self.orientation > np.pi * 2:
            self.orientation = 2 * np.pi - self.orientation
