"""
rescource.py : including the main classes to create a rescource entity that can be exploited by agents.
"""
import pygame
import numpy as np
from abm.contrib import colors

class Rescource(pygame.sprite.Sprite):
    """
        Rescource class that includes all private parameters of the rescource patch and all methods necessary to exploit
        the rescource and change the patch size/appearance accordingly
        """

    def __init__(self, id, radius, position, env_size, color, window_pad, resc_units=None):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of rescource (int)
        :param radius: radius of the patch in pixels. This also refelcts the rescource units in the patch.
        :param position: position of the patch in env as (x, y)
        :param env_size: environment size available for agents as (width, height)
        :param color: color of the patch as (R, G, B)
        :param window_pad: padding of the environment in simulation window in pixels
        :param resc_units: rescource units hidden in the given patch. If not initialized the number of units is equal to
                            the radius of the patch
        """
        # Initializing supercalss (Pygame Sprite)
        super().__init__()

        # Deciding how much resc. is in patch
        if resc_units is None:
            self.resc_units = radius
        else:
            self.resc_units = resc_units

        # Initializing agents with init parameters
        self.id = id
        self.radius = radius
        self.resc_left = self.resc_units
        self.position = np.array(position, dtype=np.float64)
        self.center = (self.position[0] + self.radius, self.position[1] + self.radius)
        self.color = color
        self.resc_left_color = (color[0]-20, color[1]-20, color[2]-20)
        self.unit_per_timestep = 0.05

        # Environment related parameters
        self.WIDTH = env_size[0]  # env width
        self.HEIGHT = env_size[1]  # env height
        self.window_pad = window_pad
        self.boundaries_x = [self.window_pad, self.window_pad + self.WIDTH]
        self.boundaries_y = [self.window_pad, self.window_pad + self.HEIGHT]

        # Initial Visualization of rescource
        self.image = pygame.Surface([self.radius * 2, self.radius * 2])
        self.image.fill(colors.BACKGROUND)
        self.image.set_colorkey(colors.BACKGROUND)
        pygame.draw.circle(
            self.image, self.color, (radius, radius), radius
        )
        # visualizing left rescources
        small_radius = int((self.resc_left / self.resc_units) * self.radius)
        pygame.draw.circle(
            self.image, self.resc_left_color, (self.radius, self.radius), small_radius
        )
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]
        font = pygame.font.Font(None, 25)
        text = font.render(f"{self.radius}", True, colors.BLACK)
        text_rect = text.get_rect(center=self.rect.center)
        self.image.blit(text, (0, 0))

    def update(self):
        # Initial Visualization of rescource
        self.image = pygame.Surface([self.radius * 2, self.radius * 2])
        self.image.fill(colors.BACKGROUND)
        self.image.set_colorkey(colors.BACKGROUND)
        pygame.draw.circle(
            self.image, self.color, (self.radius, self.radius), self.radius
        )
        small_radius = int((self.resc_left / self.resc_units) * self.radius)
        pygame.draw.circle(
            self.image, self.resc_left_color, (self.radius, self.radius), small_radius
        )
        self.rect = self.image.get_rect()
        self.rect.centerx = self.center[0]
        self.rect.centery = self.center[1]
        self.mask = pygame.mask.from_surface(self.image)
        font = pygame.font.Font(None, 18)
        text = font.render(f"{self.resc_left}", True, colors.BLACK)
        self.image.blit(text, (0, 0))
        text_rect = text.get_rect(center=self.rect.center)

    def deplete(self, rescource_units):
        """depeting the given patch with given rescource units"""
        # Not allowing faster depletion than what the patch can provide (per agent)
        if rescource_units > self.unit_per_timestep:
            rescource_units = self.unit_per_timestep

        if self.resc_left >= rescource_units:
            self.resc_left -= rescource_units
            depleted_units = rescource_units
        else: # can not deplete more than what is left
            depleted_units = self.resc_left
            self.resc_left = 0
        if self.resc_left > 0:
            return depleted_units, False
        else:
            return depleted_units, True

