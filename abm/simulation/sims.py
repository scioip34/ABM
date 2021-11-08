import pygame
import numpy as np
import sys
from abm.agent.agent import Agent
from abm.contrib import colors


class Simulation:
    def __init__(self, N, T, v_field_res=800, width=600, height=480, framerate=30):
        """
        Initializing the main simulation instance
        :param N: number of agents
        :param T: simulation time
        :param v_field_res: visual field resolution in pixels
        :param width: real width of environment (not window size)
        :param height: real height of environment (not window size)
        :param framerate: framerate of simulation
        """
        # Arena parameters
        self.WIDTH = width
        self.HEIGHT = height
        # window padding for visualization in pixels
        self.window_pad = 30

        # Simulation parameters
        self.N = N
        self.T = T
        self.framerate = framerate

        # Agent parameters
        self.v_field_res = v_field_res

        # Initializing pygame
        pygame.init()

        # pygame related class attributes
        self.all_container = pygame.sprite.Group()
        self.screen = pygame.display.set_mode([self.WIDTH + 2 * self.window_pad, self.HEIGHT + 2 * self.window_pad])
        # todo: look into this more in detail so we can control dt
        self.clock = pygame.time.Clock()

    def draw_walls(self):
        """Drwaing walls on the arena according to initialization, i.e. width, height and padding"""
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.window_pad, self.window_pad],
                         [self.window_pad, self.window_pad + self.HEIGHT])
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.window_pad, self.window_pad],
                         [self.window_pad + self.WIDTH, self.window_pad])
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.window_pad + self.WIDTH, self.window_pad],
                         [self.window_pad + self.WIDTH, self.window_pad + self.HEIGHT])
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.window_pad, self.window_pad + self.HEIGHT],
                         [self.window_pad + self.WIDTH, self.window_pad + self.HEIGHT])

    def start(self):

        # Creating N agents in the environment
        for i in range(self.N):
            x = np.random.randint(self.WIDTH/3, 2*self.WIDTH/3 + 1)
            y = np.random.randint(self.HEIGHT/3, 2*self.HEIGHT/3 + 1)
            agent = Agent(
                radius=10,
                position=(x, y),
                orientation=0,
                env_size=(self.WIDTH, self.HEIGHT),
                color=colors.BLUE,
                v_field_res=self.v_field_res
            )
            self.all_container.add(agent)

        # Creating surface to show some graphs (visual fields for now)
        stats = pygame.Surface((self.v_field_res, 50*self.N))
        stats.fill(colors.GREY)
        stats.set_alpha(230)
        stats_pos = (self.window_pad, self.window_pad)

        # Main Simulation loop
        for i in range(self.T):

            # Quitting on break event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            # Collecting agent coordinates for vision
            obstacle_coords = [ag.position for ag in self.all_container.sprites()]

            # Updating all agents accordingly
            self.all_container.update(obstacle_coords)

            # Draw environment and agents
            self.screen.fill(colors.GREY)
            self.draw_walls()
            self.all_container.draw(self.screen)

            # Updating our graphs to show visual field
            stats_graph = pygame.PixelArray(stats)
            stats_graph[:, :] = pygame.Color(*colors.WHITE)
            for k in range(self.N):
                show_base = k*50
                show_min = (k*50)+23
                show_max = (k*50)+25

                for j in range(self.all_container.sprites()[k].v_field_res):
                    if self.all_container.sprites()[k].v_field[j] == 1:
                        stats_graph[j, show_min:show_max] = pygame.Color(*colors.GREEN)
                    else:
                        stats_graph[j, show_base] = pygame.Color(*colors.GREEN)

            del stats_graph
            stats.unlock()

            # Drawing
            self.screen.blit(stats, stats_pos)
            pygame.display.flip()

            # Moving time forward
            self.clock.tick(self.framerate)

        pygame.quit()