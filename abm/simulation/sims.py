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
        :param width: width of environment
        :param height: height of environment
        :param framerate: framerate of simulation
        """
        self.WIDTH = width
        self.HEIGHT = height

        self.N = N
        self.T = T
        self.framerate = framerate
        self.v_field_res = v_field_res

        self.all_container = pygame.sprite.Group()

    def start(self):

        # Initializing pygame
        pygame.init()
        screen = pygame.display.set_mode([self.WIDTH, self.HEIGHT])

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
        # stats_pos = (self.WIDTH // 40, self.HEIGHT // 40)
        stats_pos = (0, 0)

        # Creating clock
        clock = pygame.time.Clock()

        # Simulation loop
        for i in range(self.T):

            # Quitting on break event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            # Collecting agent coordinates for vision
            obstacle_coords = []
            for ag in self.all_container.sprites():
                coord = ag.position
                obstacle_coords.append(coord)

            # Updating all agents accordingly
            self.all_container.update(obstacle_coords)
            screen.fill(colors.BACKGROUND)
            self.all_container.draw(screen)

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
            screen.blit(stats, stats_pos)
            pygame.display.flip()

            # Moving time forward
            clock.tick(self.framerate)

        pygame.quit()