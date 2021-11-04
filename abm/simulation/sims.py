import pygame
import numpy as np
import sys
from abm.agent.agent import Agent
from abm.contrib import colors

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

def plt_to_pygame(fig):
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()

    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()

    return pygame.image.fromstring(raw_data, size, "RGB")

class Simulation:
    def __init__(self, width=600, height=480):
        self.WIDTH = width
        self.HEIGHT = height

        self.all_container = pygame.sprite.Group()

        self.N = 3
        self.T = 100000
        self.v_field_res = 500

    def start(self, randomize=False):

        pygame.init()
        screen = pygame.display.set_mode([self.WIDTH, self.HEIGHT])
        #vfield_screen = pygame.display.set_mode([500, 50*self.N])

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

        stats = pygame.Surface((self.v_field_res, 50*self.N))
        stats.fill(colors.GREY)
        stats.set_alpha(230)
        stats_pos = (self.WIDTH // 40, self.HEIGHT // 40)

        clock = pygame.time.Clock()

        fig = plt.figure(figsize=[3, 1.5])
        # fig, axs = plt.subplots(1, 3)

        for i in range(self.T):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            obstacle_coords = []
            for ag in self.all_container.sprites():
                coord = ag.position
                obstacle_coords.append(coord)

            self.all_container.update(obstacle_coords)
            self.all_container.sprites()[0].color = colors.GREEN
            screen.fill(colors.BACKGROUND)

            stats_height = stats.get_height()
            stats_width = stats.get_width()
            stats_graph = pygame.PixelArray(stats)
            stats_graph[:, :] = pygame.Color(*colors.WHITE)



            for k in range(self.N):
                # ax = fig.add_subplot(int(f'{self.N}1{k+1}'))
                # ax.plot(self.all_container.sprites()[k].v_field)
                show_base = k*50
                show_min = (k*50)+23
                show_max = (k*50)+25

                for j in range(self.all_container.sprites()[k].v_field_res):
                    if self.all_container.sprites()[k].v_field[j] == 1:
                        stats_graph[j, show_min:show_max] = pygame.Color(*colors.GREEN)
                    else:
                        stats_graph[j, show_base] = pygame.Color(*colors.GREEN)

            self.all_container.draw(screen)

            del stats_graph
            stats.unlock()
            screen.blit(stats, stats_pos)
            # surf = plt_to_pygame(fig)
            # screen.blit(surf, (0, 0))
            pygame.display.flip()

            clock.tick(30)

        pygame.quit()