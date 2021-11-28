import pygame
import numpy as np
import sys
from abm.agent.agent import Agent
from abm.environment.rescource import Rescource
from abm.contrib import colors
from abm.simulation import interactions as itra
from math import atan2

# loading env variables from dotenv file
from dotenv import dotenv_values
envconf = dotenv_values(".env")

class Simulation:
    def __init__(self, N, T, v_field_res=800, width=600, height=480,
                 framerate=30, window_pad=30, show_vis_field=False,
                 pooling_time=3, pooling_prob=0.05, agent_radius=10,
                 N_resc=10, min_resc_perpatch=200, max_resc_perpatch=1000, patch_radius=30,
                 regenerate_patches=True, agent_consumption=1, vision_range=150, visual_exclusion=False):
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
        :param pooling_time: time units for a single pooling events
        :param pooling probability: initial probability of switching to pooling regime for any agent
        :param agent_radius: radius of the agents
        :param N_resc: number of rescource patches in the environment
        :param min_resc_perpatch: minimum rescaurce unit per patch
        :param max_resc_perpatch: maximum rescaurce units per patch
        :param patch_radius: radius of rescaurce patches
        :param regenerate_patches: bool to decide if patches shall be regenerated after depletion
        :param agent_consumption: agent consumption (exploitation speed) in res. units / time units
        :param vision_range: range (in px) of agents' vision
        :param visual_exclusion: when true agents can visually exclude socially relevant visual cues from other agents'
                                projection field
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
        self.agent_radii = agent_radius
        self.v_field_res = v_field_res
        self.pooling_time = pooling_time
        self.pooling_prob = pooling_prob
        self.agent_consumption = agent_consumption
        self.vision_range = vision_range
        self.visual_exclusion = visual_exclusion

        # Rescource parameters
        self.N_resc = N_resc
        self.resc_radius = patch_radius
        self.min_resc_units = min_resc_perpatch
        self.max_resc_units = max_resc_perpatch
        self.regenerate_resources = regenerate_patches

        # Initializing pygame
        pygame.init()

        # pygame related class attributes
        self.agents = pygame.sprite.Group()
        self.rescources = pygame.sprite.Group()
        self.screen = pygame.display.set_mode([self.WIDTH + 2 * self.window_pad, self.HEIGHT + 2 * self.window_pad])
        # todo: look into this more in detail so we can control dt
        self.clock = pygame.time.Clock()

    def proove_resource(self, resource):
        """Checks if the proposed resource can be taken into self.resources according to some rules, e.g. no overlap,
        or given resource patch distribution, etc"""
        # Checking for collision with already existing resources
        new_res_group = pygame.sprite.Group()
        new_res_group.add(resource)
        collision_group = pygame.sprite.groupcollide(
            self.rescources,
            new_res_group,
            False,
            False,
            pygame.sprite.collide_circle
        )
        if len(collision_group) > 0:
            return False
        else:
            return True


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

    def kill_resource(self, resource):
        """Killing (and regenerating) a given resource patch"""
        if self.regenerate_resources:
            self.add_new_resource_patch()
        resource.kill()

    def add_new_resource_patch(self):
        """Adding a new resource patch to the resources sprite group. The position of the new resource is proved with
        prove_resource method so that the distribution and overlap is following some predefined rules"""
        resource_proven = 0
        if len(self.rescources) > 0:
            id = max([resc.id for resc in self.rescources])
        else:
            id = 0
        while not resource_proven:
            radius = self.resc_radius
            x = np.random.randint(self.window_pad, self.WIDTH + self.window_pad - radius)
            y = np.random.randint(self.window_pad, self.HEIGHT + self.window_pad - radius)
            units = np.random.randint(self.min_resc_units, self.max_resc_units)
            resource = Rescource(id+1, radius, (x, y), (self.WIDTH, self.HEIGHT), colors.GREY, self.window_pad, units)
            resource_proven = self.proove_resource(resource)
        self.rescources.add(resource)

    def agent_agent_collision(self, agent1, agent2):
        """collision protocol called on any agent that has been collided with another one
        :param agent1, agent2: agents that collided"""
        # Updating all agents accordingly
        agent2 = agent2[0]
        if agent2.get_mode() != "exploit":
            agent2.set_mode("collide")

        x1, y1 = agent1.position
        x2, y2 = agent2.position
        dx = x2-x1
        dy = y2-y1
        # calculating relative closed angle to agent2 orientation
        theta = (atan2(dy, dx) + agent2.orientation) % (np.pi * 2)

        if 0 < theta < np.pi:
            agent2.orientation -= np.pi/8
        elif np.pi < theta < 2*np.pi:
            agent2.orientation += np.pi/8

        agent2.velocity += 0.5

    def start(self):
        # Creating N agents in the environment
        for i in range(self.N):
            x = np.random.randint(self.WIDTH / 3, 2 * self.WIDTH / 3 + 1)
            y = np.random.randint(self.HEIGHT / 3, 2 * self.HEIGHT / 3 + 1)
            agent = Agent(
                id=i,
                radius=self.agent_radii,
                position=(x, y),
                orientation=0,
                env_size=(self.WIDTH, self.HEIGHT),
                color=colors.BLUE,
                v_field_res=self.v_field_res,
                window_pad=self.window_pad,
                pooling_time=self.pooling_time,
                pooling_prob=self.pooling_prob,
                consumption=self.agent_consumption,
                vision_range=self.vision_range,
                visual_exclusion=self.visual_exclusion
            )
            self.agents.add(agent)

        # Creating rescource patches
        for i in range(self.N_resc):
            self.add_new_resource_patch()

        # Creating surface to show some graphs (visual fields for now)
        # if self.show_vis_field:
        stats = pygame.Surface((self.v_field_res, 50 * self.N))
        stats.fill(colors.GREY)
        stats.set_alpha(200)
        stats_pos = (int(self.window_pad), int(self.window_pad / 2))

        turned_on_vfield = 0

        # Main Simulation loop
        for i in range(self.T):

            # Quitting on break event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_RETURN]:
                print('placeholder: ENTER_PRESSED')
                # show_vis_fields_on_return = bool(int(envconf['SHOW_VISUAL_FIELDS_RETURN']))
                # if not self.show_vis_field and show_vis_fields_on_return:
                #     self.show_vis_field = 1
                #     turned_on_vfield = 1
                # for ag in self.agents.sprites():
                #     if ag.mode != "exploit":
                #         ag.mode = "flock"
            else:
                for ag in self.agents.sprites():
                    if ag.get_mode() == "flock":
                        ag.set_mode("explore")
                if self.show_vis_field and turned_on_vfield:
                    turned_on_vfield = 0
                    self.show_vis_field = 0

            # AGENT AGENT INTERACTION
            # Check if any 2 agents has been collided and reflect them from each other if so
            collision_group_aa = pygame.sprite.groupcollide(
                self.agents,
                self.agents,
                False,
                False,
                itra.within_group_collision
            )
            collided_agents = []

            for agent1, agent2 in collision_group_aa.items():
                self.agent_agent_collision(agent1, agent2)
                collided_agents.append(agent1)
                collided_agents.append(agent2)

            for agent in self.agents:
                if agent not in collided_agents and agent.get_mode() == "collide":
                    agent.set_mode("explore")

            # AGENT RESCOURCE INTERACTION (can not be separated from main thread for some reason)
            # Check if any 2 agents has been collided and reflect them from each other if so
            collision_group_ar = pygame.sprite.groupcollide(
                self.rescources,
                self.agents,
                False,
                False,
                pygame.sprite.collide_circle
            )

            # collecting agents that are on rescource patch
            agents_on_rescs = []

            for resc, agents in collision_group_ar.items():  # looping through patches
                destroy_resc = 0  # if we destroy a patch it is 1
                for agent in agents:  # looping through agents on patch
                    if agent not in collided_agents:
                        if destroy_resc:  # if a previous agent on patch consumed the last unit
                            agent.env_status = -1  # then this agent does not find a patch here anymore
                            agent.pool_success = 0  # restarting pooling timer if it happened during pooling
                        # if an agent finished pooling on a resource patch
                        if (agent.get_mode() in ["pool", "relocate"] and agent.pool_success) or agent.pooling_time == 0:
                            agent.pool_success = 0  # reinit pooling variable
                            agent.env_status = 1  # providing the status of the environment to the agent
                        if agent.get_mode() == "exploit":  # if an agent is already exploiting this patch
                            depl_units, destroy_resc = resc.deplete(agent.consumption)  # it continues depleting the patch
                            agent.collected_r += depl_units # and increasing it's collected rescources
                            if destroy_resc:  # if the consumed unit was the last in the patch
                                agent.env_status = -1  # notifying agent that there is no more rescource here
                        agents_on_rescs.append(agent)  # collecting agents on rescource patches
                if destroy_resc:  # if the patch is fully depleted
                    self.kill_resource(resc) # we clear it from the memory and regenrate it somewhere if needed

            for agent in self.agents.sprites():
                if agent not in agents_on_rescs:  # for all the agents that are not on recourse patches
                    if agent not in collided_agents: # and are not colliding with each other currently
                        if (agent.get_mode() in ["pool", "relocate"] and agent.pool_success) or agent.pooling_time == 0:  # if they finished pooling
                            agent.pool_success = 0  # reinit pooling var
                            agent.env_status = -1  # provide the info that there is no resource here
                        elif agent.get_mode() == "exploit":
                            agent.pool_success = 0  # reinit pooling var
                            agent.env_status = -1  # provide the info taht there is no resource here


            # Update rescource patches
            self.rescources.update()
            # Update agents according to current visible obstacles
            self.agents.update(self.agents)

            # Draw environment and agents
            self.screen.fill(colors.BACKGROUND)
            self.rescources.draw(self.screen)
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
                        if self.agents.sprites()[k].soc_v_field[j] == 1:
                            stats_graph[j, show_min:show_max] = pygame.Color(*colors.GREEN)
                        # elif self.agents.sprites()[k].soc_v_field[j] == -1:
                        #     stats_graph[j, show_min:show_max] = pygame.Color(*colors.RED)
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
