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

    def __init__(self, id, radius, position, env_size, color, window_pad):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of rescource (int)
        :param radius: radius of the patch in pixels. This also refelcts the rescource units in the patch.
        :param position: position of the patch in env as (x, y)
        :param env_size: environment size available for agents as (width, height)
        :param color: color of the patch as (R, G, B)
        :param window_pad: padding of the environment in simulation window in pixels
        """
        # Initializing supercalss (Pygame Sprite)
        super().__init__()

        # Initializing agents with init parameters
        self.id = id
        self.radius = radius
        self.position = np.array(position, dtype=np.float64)
        self.color = color

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
            self.image, color, (radius, radius), radius
        )
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect().copy()
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
        self.rect = self.image.get_rect().copy()
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]
        self.mask = pygame.mask.from_surface(self.image)
        font = pygame.font.Font(None, 25)
        text = font.render(f"{self.radius}", True, colors.BLACK)
        self.image.blit(text, (self.radius, self.radius))
        text_rect = text.get_rect(center=self.rect.center)

    def deplete(self, rescource_units):
        """depeting the given patch with given rescource units"""
        if self.radius > rescource_units:
            self.radius -= rescource_units
            return False
        else:
            self.radius -= 0
            return True  # the rescource is fully depleted and shall be destroyed now

