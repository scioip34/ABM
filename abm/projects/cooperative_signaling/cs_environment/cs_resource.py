import numpy as np
import pygame

from abm.contrib import colors
from abm.environment.rescource import Rescource
from abm.projects.cooperative_signaling.cs_agent.cs_supcalc import reflection_from_circular_wall, levy_walk


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
        self.current_step_time_left = 0

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
        """
        Reflecting resource from the circle arena border. Analogous to CSAgent
        reflect_from_walls() method.
        """
        # x coordinate - x of the center point of the circle
        x = self.position[0] + self.radius
        c_x = (self.WIDTH / 2 + self.window_pad)
        dx = x - c_x
        # y coordinate - y of the center point of the circle
        y = self.position[1] + self.radius
        c_y = (self.HEIGHT / 2 + self.window_pad)
        dy = y - c_y
        # radius of the environment
        e_r = self.HEIGHT / 2

        # return if the resource has not reached the boarder
        if np.linalg.norm([dx, dy]) + self.radius < e_r:
            return

        # reflect the resource from the boarder
        self.orientation = reflection_from_circular_wall(
            dx, dy, self.orientation)

        # make orientation between 0 and 2pi
        self.prove_orientation()

        # relocate the resource back inside the circle
        relocation = self.velocity
        self.position[0] += relocation * np.cos(self.orientation)
        self.position[1] -= relocation * np.sin(self.orientation)
        self.center = (
            self.position[0] + self.radius, self.position[1] + self.radius)

        # check if the resource is still outside the circle
        diff = [self.center[0] - c_x, self.center[1] - c_y]
        if np.linalg.norm(diff) + self.radius >= e_r:
            # if yes, relocate it again at the center
            self.position[0] = c_x - self.radius
            self.position[1] = c_y - self.radius
            self.center = (c_x, c_y)

    def update(self):

        if self.current_step_time_left <= 0:
            # applying random movement on resource patch
            step_size, theta = levy_walk(
                exponent=1,
                max_step_size=100,
                exp_theta_min=-self.res_theta_abs,
                exp_theta_max=self.res_theta_abs)
            self.orientation += theta
            # bounding orientation into 0 and 2pi
            self.prove_orientation()
            self.current_step_time_left = step_size
        self.current_step_time_left -= 1

        self.velocity += (self.des_velocity - self.velocity)

        # updating agent's position
        if not self.is_clicked:
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
        self.image.set_colorkey(colors.BACKGROUND)
        pygame.draw.circle(
            self.image, self.color, (self.radius, self.radius), self.radius
        )
        self.rect = self.image.get_rect()
        self.rect.centerx = self.center[0]
        self.rect.centery = self.center[1]
        self.mask = pygame.mask.from_surface(self.image)
        if self.is_clicked or self.show_stats:
            font = pygame.font.Font(None, 18)
            text = font.render(
                f"{self.resc_left:.2f}, Q{self.unit_per_timestep:.2f}", True,
                colors.BLACK)
            self.image.blit(text, (0, 0))
            text_rect = text.get_rect(center=self.rect.center)
