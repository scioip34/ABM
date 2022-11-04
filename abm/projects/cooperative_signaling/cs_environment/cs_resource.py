import numpy as np
import pygame

from abm.contrib import colors
from abm.environment.rescource import Rescource
from abm.projects.cooperative_signaling.cs_agent.cs_supcalc import random_walk


class CSResource(Rescource):
    def __init__(self, des_velocity=1.5, res_theta_abs=0.2, **kwargs):
        super().__init__(**kwargs)
        """
        :param des_velocity: desired velocity of resource patch in pixel per 
        timestep
        :param res_theta_abs: change in orientation will be pulled from uniform 
        -res_theta_abs to res_theta_abs
        """

        # Initializing agents with init parameters
        self.des_velocity = des_velocity  # 1.5
        self.res_theta_abs = res_theta_abs  # 0.2

        # State variables
        self.velocity = 0
        self.orientation = 0

    def update_clicked_status(self, mouse):
        """Checking if the resource patch was clicked on a mouse event"""
        if self.rect.collidepoint(mouse):
            self.is_clicked = True
            self.position[0] = mouse[0] - self.radius
            self.position[1] = mouse[1] - self.radius
            self.center = (
                self.position[0] + self.radius, self.position[1] + self.radius)
        else:
            self.is_clicked = False
        self.draw_update()

    def prove_orientation(self):
        """Restricting orientation angle between 0 and 2 pi"""
        if self.orientation < 0:
            self.orientation = 2 * np.pi + self.orientation
        if self.orientation > np.pi * 2:
            self.orientation = self.orientation - 2 * np.pi

    def reflect_from_walls(self):
        """reflecting agent from environment boundaries according to a desired
        x, y coordinate. If this is over any
        boundaries of the environment, the agents position and orientation will
        be changed such that the agent is
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

        self.center = (
            self.position[0] + self.radius, self.position[1] + self.radius)

    def update(self):

        # applying random movement on resource patch
        _, theta = random_walk(exp_theta_min=-self.res_theta_abs,
                               exp_theta_max=self.res_theta_abs)
        self.orientation += theta
        self.prove_orientation()  # bounding orientation into 0 and 2pi
        self.velocity += (self.des_velocity - self.velocity)

        # updating agent's position
        self.position[0] += self.velocity * np.cos(self.orientation)
        self.position[1] -= self.velocity * np.sin(self.orientation)
        self.center = (
            self.position[0] + self.radius, self.position[1] + self.radius)

        self.reflect_from_walls()
        self.draw_update()

    def draw_update(self):
        # Initial Visualization of rescource
        self.image = pygame.Surface([self.radius * 2, self.radius * 2])
        self.image.fill(colors.BACKGROUND)
