import pygame
import numpy as np
import sys
from abm.agent.agent import Agent
from abm.contrib import colors


def within_group_collision(sprite1, sprite2):
    """Custom colllision check that omits collisions of sprite with itself. This way we can use group collision
    detect WITHIN a single group instead of between multiple groups"""
    if sprite1 != sprite2:
        return pygame.sprite.collide_circle(sprite1, sprite2)
    return False


class Simulation:
    def __init__(self, N, T, v_field_res=800, width=600, height=480,
                 framerate=30, window_pad=30, show_vis_field=False):
        """
        Initializing the main simulation instance
        :param N: number of agents
        :param T: simulation time
        :param v_field_res: visual field resolution in pixels
        :param width: real width of environment (not window size)
        :param height: real height of environment (not window size)
        :param framerate: framerate of simulation
        :param window_pad: padding of the environment in simulation window in pixels
        :param show_vis_field: (Bool) turn on visualization for visual field of agents
        """
        # Arena parameters
        self.WIDTH = width
        self.HEIGHT = height
        self.window_pad = window_pad

        # Simulation parameters
        self.N = N
        self.T = T
        self.framerate = framerate

        # Visualization parameters
        self.show_vis_field = show_vis_field

        # Agent parameters
        self.v_field_res = v_field_res

        # Initializing pygame
        pygame.init()

        # pygame related class attributes
        self.agents = pygame.sprite.Group()
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
            x = np.random.randint(self.WIDTH / 3, 2 * self.WIDTH / 3 + 1)
            y = np.random.randint(self.HEIGHT / 3, 2 * self.HEIGHT / 3 + 1)
            agent = Agent(
                id=i,
                radius=10,
                position=(x, y),
                orientation=0,
                env_size=(self.WIDTH, self.HEIGHT),
                color=colors.BLUE,
                v_field_res=self.v_field_res,
                window_pad=self.window_pad
            )
            self.agents.add(agent)

        # Creating surface to show some graphs (visual fields for now)
        if self.show_vis_field:
            stats = pygame.Surface((self.v_field_res, 50 * self.N))
            stats.fill(colors.GREY)
            stats.set_alpha(230)
            stats_pos = (int(self.window_pad), int(self.window_pad / 2))

        # Main Simulation loop
        for i in range(self.T):

            # Quitting on break event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            # Collecting agent coordinates for vision
            obstacle_coords = [ag.position for ag in self.agents.sprites()]

            collision_group = pygame.sprite.groupcollide(
                self.agents,
                self.agents,
                False,
                False,
                within_group_collision
            )

            if i > 0:
                for agent in collision_group:
                    # Updating all agents accordingly
                    if agent.velocity >= 0:
                        agent.velocity += 1
                    else:
                        agent.velocity -= 1
                    agent.velocity *= -1


            self.agents.update(obstacle_coords)

            # Draw environment and agents
            self.screen.fill(colors.BACKGROUND)
            self.draw_walls()
            self.agents.draw(self.screen)

            if self.show_vis_field:
                # Updating our graphs to show visual field
                stats_graph = pygame.PixelArray(stats)
                stats_graph[:, :] = pygame.Color(*colors.WHITE)
                for k in range(self.N):
                    show_base = k * 50
                    show_min = (k * 50) + 23
                    show_max = (k * 50) + 25

                    for j in range(self.agents.sprites()[k].v_field_res):
                        if self.agents.sprites()[k].v_field[j] == 1:
                            stats_graph[j, show_min:show_max] = pygame.Color(*colors.GREEN)
                        else:
                            stats_graph[j, show_base] = pygame.Color(*colors.GREEN)

                del stats_graph
                stats.unlock()

            # Drawing
            if self.show_vis_field:
                self.screen.blit(stats, stats_pos)
            pygame.display.flip()

            # Moving time forward
            self.clock.tick(self.framerate)

        pygame.quit()
