import pygame
import numpy as np
import sys

from abm.agent import supcalc
from abm.agent.agent import Agent
from abm.environment.rescource import Rescource
from abm.contrib import colors, ifdb_params
from abm.simulation import interactions as itra
from abm.monitoring import ifdb
from math import atan2

# loading env variables from dotenv file
from dotenv import dotenv_values

envconf = dotenv_values(".env")


def notify_agent(agent, status, res_id=None):
    """Notifying agent about the status of the environment in a given position"""
    agent.env_status_before = agent.env_status
    agent.env_status = status
    agent.novelty = np.roll(agent.novelty, 1)
    novelty = agent.env_status - agent.env_status_before
    novelty = 1 if novelty > 0 else 0
    agent.novelty[0] = novelty
    agent.novelty[0] = novelty
    if agent.id==0:
        print(agent.novelty)
    agent.pool_success = 1  # restarting pooling timer when notified
    if res_id is None:
        agent.exploited_patch_id = -1
    else:
        agent.exploited_patch_id = res_id


def refine_ar_overlap_group(collision_group):
    """We define overlap according to the center of agents. If the collision is not yet with the center of agent,
    we remove that collision from the group"""
    for resc, agents in collision_group.items():
        agents_refined = []
        for agent in agents:
            # Only keeping agent in collision group if it's center is inside the radius of the patch
            # I.E: the agent can only get information from 1 point-like sensor in the center
            if supcalc.distance(resc, agent) < resc.radius:
                agents_refined.append(agent)
        collision_group[resc] = agents_refined
    return collision_group


class Simulation:
    def __init__(self, N, T, v_field_res=800, width=600, height=480,
                 framerate=25, window_pad=30, show_vis_field=False,
                 pooling_time=3, pooling_prob=0.05, agent_radius=10,
                 N_resc=10, min_resc_perpatch=200, max_resc_perpatch=1000, min_resc_quality=0.1, max_resc_quality=1,
                 patch_radius=30, regenerate_patches=True, agent_consumption=1, teleport_exploit=True,
                 vision_range=150, agent_fov=1.0, visual_exclusion=False, show_vision_range=False,
                 use_ifdb_logging=False, save_csv_files=False, ghost_mode=True, patchwise_exclusion=True):
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
        :param min_resc_quality: minimum resource quality in unit/timesteps that is allowed for each agent on a patch 
            to exploit from the patch
        : param max_resc_quality: maximum resource quality in unit/timesteps that is allowed for each agent on a patch
            to exploit from the patch
        :param patch_radius: radius of rescaurce patches
        :param regenerate_patches: bool to decide if patches shall be regenerated after depletion
        :param agent_consumption: agent consumption (exploitation speed) in res. units / time units
        :param teleport_exploit: boolean to choose if we teleport agents to the middle of the res. patch during
                                exploitation
        :param vision_range: range (in px) of agents' vision
        :param agent_fov (float): the field of view of the agent as percentage. e.g. if 0.5, the the field of view is
                                between -pi/2 and pi/2
        :param visual_exclusion: when true agents can visually exclude socially relevant visual cues from other agents'
                                projection field
        :param show_vision_range: bool to switch visualization of visual range for agents. If true the limit of far
                                and near field visual field will be drawn around the agents
        :param use_ifdb_logging: Switch to turn IFDB save on or off
        :param save_csv_files: Save all recorded IFDB data as csv file. Only works if IFDB looging was turned on
        :param ghost_mode: if turned on, exploiting agents behave as ghosts and others can pass through them
        :param patchwise_exclusion: excluding agents from social v field if they are exploiting the same patch as the
            focal agent
        """
        # Arena parameters
        self.WIDTH = width
        self.HEIGHT = height
        self.window_pad = window_pad

        # Simulation parameters
        self.N = N
        self.T = T
        self.t = 0
        self.framerate_orig = framerate
        self.framerate = self.framerate_orig
        self.is_paused = False

        # Visualization parameters
        self.show_vis_field = show_vis_field
        self.show_vision_range = show_vision_range

        # Agent parameters
        self.agent_radii = agent_radius
        self.v_field_res = v_field_res
        self.pooling_time = pooling_time
        self.pooling_prob = pooling_prob
        self.agent_consumption = agent_consumption
        self.teleport_exploit = teleport_exploit
        self.vision_range = vision_range
        self.agent_fov = (-agent_fov * np.pi, agent_fov * np.pi)
        self.visual_exclusion = visual_exclusion
        self.ghost_mode = ghost_mode
        self.patchwise_exclusion = patchwise_exclusion

        # Rescource parameters
        self.N_resc = N_resc
        self.resc_radius = patch_radius
        self.min_resc_units = min_resc_perpatch
        self.max_resc_units = max_resc_perpatch
        self.min_resc_quality = min_resc_quality
        self.max_resc_quality = max_resc_quality
        self.regenerate_resources = regenerate_patches

        # Initializing pygame
        pygame.init()

        # pygame related class attributes
        self.agents = pygame.sprite.Group()
        self.rescources = pygame.sprite.Group()
        self.screen = pygame.display.set_mode([self.WIDTH + 2 * self.window_pad, self.HEIGHT + 2 * self.window_pad])
        # todo: look into this more in detail so we can control dt
        self.clock = pygame.time.Clock()

        # Monitoring
        self.save_in_ifd = use_ifdb_logging
        self.save_csv_files = save_csv_files
        if self.save_in_ifd:
            self.ifdb_client = ifdb.create_ifclient()
            self.ifdb_client.drop_database(ifdb_params.INFLUX_DB_NAME)
            self.ifdb_client.create_database(ifdb_params.INFLUX_DB_NAME)
            ifdb.save_simulation_params(self.ifdb_client, self)

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
        collision_group_a = pygame.sprite.groupcollide(
            self.agents,
            new_res_group,
            False,
            False,
            pygame.sprite.collide_circle
        )
        if len(collision_group) > 0 or len(collision_group_a):
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

    def draw_visual_fields(self):
        """Visualizing the range of vision for agents as opaque circles around the agents"""
        for agent in self.agents:
            # Show visual range
            pygame.draw.circle(self.screen, colors.LIGHT_BLUE, agent.position + agent.radius, agent.vision_range,
                                width=1)

            # Show limits of FOV
            if self.agent_fov[1] < np.pi:
                angles = [agent.orientation+agent.FOV[0], agent.orientation+agent.FOV[1]]
                for angle in angles:
                    start_pos = (agent.position[0] + agent.radius, agent.position[1] + agent.radius)
                    end_pos = [start_pos[0] + (np.cos(angle)) * 3*agent.radius,
                               start_pos[1] + ( - np.sin(angle)) * 3*agent.radius]
                    pygame.draw.line(self.screen, colors.LIGHT_BLUE,
                                     start_pos,
                                     end_pos, 1)

    def draw_framerate(self):
        """Showing framerate, sim time and pause status on simulation windows"""
        tab_size = self.window_pad
        line_height = int(self.window_pad / 2)
        font = pygame.font.Font(None, line_height)
        status = [
            f"FPS: {self.framerate}, t = {self.t}/{self.T}",
        ]
        if self.is_paused:
            status.append("-Paused-")
        for i, stat_i in enumerate(status):
            text = font.render(stat_i, True, colors.BLACK)
            self.screen.blit(text, (tab_size, i * line_height))

    def draw_agent_stats(self, font_size=15, spacing=0):
        """Showing agent information when paused"""
        if self.is_paused:
            font = pygame.font.Font(None, font_size)
            for agent in self.agents:
                status = [
                    f"ID: {agent.id}",
                    f"res.: {agent.collected_r}",
                    f"ori.: {agent.orientation:.2f}",
                    f"w: {agent.w:.2f}"
                ]
                for i, stat_i in enumerate(status):
                    text = font.render(stat_i, True, colors.BLACK)
                    self.screen.blit(text, (agent.position[0] + 2 * agent.radius,
                                            agent.position[1] + 2 * agent.radius + i * (font_size + spacing)))

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
            quality = np.random.uniform(self.min_resc_quality, self.max_resc_quality)
            resource = Rescource(id + 1, radius, (x, y), (self.WIDTH, self.HEIGHT), colors.GREY, self.window_pad, units,
                                 quality)
            resource_proven = self.proove_resource(resource)
        self.rescources.add(resource)

    def agent_agent_collision(self, agent1, agent2):
        """collision protocol called on any agent that has been collided with another one
        :param agent1, agent2: agents that collided"""
        # Updating all agents accordingly
        if not isinstance(agent2, list):
            agents2 = [agent2]
        else:
            agents2 = agent2

        for i, agent2 in enumerate(agents2):
            do_collision = True
            # if the ghost mode is turned on and any of the 2 colliding agents is exploiting, the
            # collision protocol will not be carried out so that agents can overlap with each other in this case
            if self.ghost_mode:
                if agent2.get_mode() != "exploit" and agent1.get_mode() != "exploit":
                    do_collision = True
                else:
                    do_collision = False

            if do_collision:
                # overriding any mode with collision
                if agent2.get_mode() != "exploit":
                    agent2.set_mode("collide")

                x1, y1 = agent1.position
                x2, y2 = agent2.position
                dx = x2 - x1
                dy = y2 - y1
                # calculating relative closed angle to agent2 orientation
                theta = (atan2(dy, dx) + agent2.orientation) % (np.pi * 2)

                # deciding on turning angle
                if 0 < theta < np.pi:
                    agent2.orientation -= np.pi / 8
                elif np.pi < theta < 2 * np.pi:
                    agent2.orientation += np.pi / 8

                if agent2.velocity == 1:
                    agent2.velocity += 0.5
                else:
                    agent2.velocity = 1

            else:  # ghost mode is on, we do nothing on collision
                pass

    def create_agents(self):
        """Creating agents according to how the simulation class was initialized"""
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
                FOV=self.agent_fov,
                window_pad=self.window_pad,
                pooling_time=self.pooling_time,
                pooling_prob=self.pooling_prob,
                consumption=self.agent_consumption,
                vision_range=self.vision_range,
                visual_exclusion=self.visual_exclusion,
                patchwise_exclusion=self.patchwise_exclusion
            )
            self.agents.add(agent)

    def create_resources(self):
        """Creating resource patches according to how the simulation class was initialized"""
        for i in range(self.N_resc):
            self.add_new_resource_patch()

    def bias_agent_towards_res_center(self, agent, resc, relative_speed=0.02):
        """Turning the agent towards the center of a resource patch with some relative speed"""
        x1, y1 = agent.position + agent.radius
        x2, y2 = resc.center
        dx = x2 - x1
        dy = y2 - y1
        # calculating relative closed angle to agent2 orientation
        cl_ang = (atan2(dy, dx) + agent.orientation) % (np.pi * 2)
        agent.orientation += (cl_ang - np.pi) * relative_speed

    def create_vis_field_graph(self):
        """Creating visualization graph for visual fields of the agents"""
        stats = pygame.Surface((self.WIDTH, 50 * self.N))
        stats.fill(colors.GREY)
        stats.set_alpha(150)
        stats_pos = (int(self.window_pad), int(self.window_pad))
        return stats, stats_pos

    def interact_with_event(self, event):
        """Carry out functionality according to user's interaction"""
        # Exit if requested
        if event.type == pygame.QUIT:
            sys.exit()

        # Change orientation with mouse wheel
        if event.type == pygame.MOUSEWHEEL:
            if event.y == -1:
                event.y = 0
            for ag in self.agents:
                ag.move_with_mouse(pygame.mouse.get_pos(), event.y, 1-event.y)

        # Pause on Space
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self.is_paused = not self.is_paused

        # Speed up on s and down on f. reset default framerate with d
        if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
            self.framerate -= 1
            if self.framerate < 1:
                self.framerate = 1
        if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            self.framerate += 1
            if self.framerate > 35:
                self.framerate = 35
        if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
            self.framerate = self.framerate_orig

        # Continuous mouse events (move with cursor)
        if pygame.mouse.get_pressed()[0]:
            try:
                for ag in self.agents:
                    ag.move_with_mouse(event.pos, 0, 0)
            except AttributeError:
                for ag in self.agents:
                    ag.move_with_mouse(pygame.mouse.get_pos(), 0, 0)
        else:
            for ag in self.agents:
                ag.is_moved_with_cursor = False

    def decide_on_vis_field_visibility(self, turned_on_vfield):
        """Deciding f the visual field needs to be shown or not"""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RETURN]:
            show_vis_fields_on_return = bool(int(envconf['SHOW_VISUAL_FIELDS_RETURN']))
            if not self.show_vis_field and show_vis_fields_on_return:
                self.show_vis_field = 1
                turned_on_vfield = 1
        else:
            if self.show_vis_field and turned_on_vfield:
                turned_on_vfield = 0
                self.show_vis_field = 0
        return turned_on_vfield

    def show_visual_fields(self, stats, stats_pos):
        """Showing visual fields of the agnets on a specific graph"""
        stats_width = stats.get_width()
        # Updating our graphs to show visual field
        stats_graph = pygame.PixelArray(stats)
        stats_graph[:, :] = pygame.Color(*colors.WHITE)
        for k in range(self.N):
            show_base = k * 50
            show_min = (k * 50) + 23
            show_max = (k * 50) + 25

            for j in range(self.agents.sprites()[k].v_field_res):
                curr_idx = int(j * (stats_width / self.v_field_res))
                if self.agents.sprites()[k].soc_v_field[j] != 0:
                    stats_graph[curr_idx, show_min:show_max] = pygame.Color(*colors.GREEN)
                # elif self.agents.sprites()[k].soc_v_field[j] == -1:
                #     stats_graph[j, show_min:show_max] = pygame.Color(*colors.RED)
                else:
                    stats_graph[curr_idx, show_base] = pygame.Color(*colors.GREEN)

        del stats_graph
        stats.unlock()

        # Drawing
        self.screen.blit(stats, stats_pos)
        for agi, ag in enumerate(self.agents):
            line_height = 15
            font = pygame.font.Font(None, line_height)
            status = f"agent {ag.id}"
            text = font.render(status, True, colors.BLACK)
            self.screen.blit(text, (int(self.window_pad) / 2, self.window_pad + agi * 50))

    def draw_frame(self, stats, stats_pos):
        """Drawing environment, agents and every other visualization in each timestep"""
        self.screen.fill(colors.BACKGROUND)
        self.rescources.draw(self.screen)
        self.draw_walls()
        self.agents.draw(self.screen)
        if self.show_vision_range:
            self.draw_visual_fields()
        self.draw_framerate()
        self.draw_agent_stats()

        if self.show_vis_field:
            # showing visual fields of the agents
            self.show_visual_fields(stats, stats_pos)

        pygame.display.flip()

    def start(self):
        # Creating N agents in the environment
        self.create_agents()

        # Creating resource patches
        self.create_resources()

        # Creating surface to show visual fields
        stats, stats_pos = self.create_vis_field_graph()

        # local var to decide when to show visual fields
        turned_on_vfield = 0

        # Main Simulation loop until dedicated simulation time
        while self.t < self.T:

            for event in pygame.event.get():
                # Carry out interaction according to user activity
                self.interact_with_event(event)

            # deciding if vis field needs to be shown in this timestep
            turned_on_vfield = self.decide_on_vis_field_visibility(turned_on_vfield)

            if not self.is_paused:

                # ------ AGENT-AGENT INTERACTION ------
                # Check if any 2 agents has been collided and reflect them from each other if so
                collision_group_aa = pygame.sprite.groupcollide(
                    self.agents,
                    self.agents,
                    False,
                    False,
                    itra.within_group_collision
                )
                collided_agents = []
                # Carry out agent-agent collisions and collecting collided agents for later (according to parameters
                # such as ghost mode, or teleportation)
                for agent1, agent2 in collision_group_aa.items():
                    self.agent_agent_collision(agent1, agent2)
                    if not isinstance(agent2, list):
                        agents2 = [agent2]
                    else:
                        agents2 = agent2
                    for agent2 in agents2:
                        if self.teleport_exploit:
                            if agent1.get_mode() != "exploit":
                                collided_agents.append(agent1)
                            if agent2.get_mode() != "exploit":
                                collided_agents.append(agent2)
                        else:
                            if not self.ghost_mode:
                                collided_agents.append(agent1)
                                collided_agents.append(agent2)
                            else:
                                if agent1.get_mode() != "exploit" and agent2.get_mode() != "exploit":
                                    collided_agents.append(agent1)
                                    collided_agents.append(agent2)

                # Turn off collision mode when over
                for agent in self.agents:
                    if agent not in collided_agents and agent.get_mode() == "collide":
                        agent.set_mode("explore")

                # ------ AGENT-RESCOURCE INTERACTION (can not be separated from main thread for some reason)------
                # Check if any 2 agents has been collided and reflect them from each other if so
                collision_group_ar = pygame.sprite.groupcollide(
                    self.rescources,
                    self.agents,
                    False,
                    False,
                    pygame.sprite.collide_circle
                )

                # refine collision group according to point-like pooling in center of agents
                collision_group_ar = refine_ar_overlap_group(collision_group_ar)

                # collecting agents that are on resource patch
                agents_on_rescs = []

                # Notifying agents about resource if pooling is successful + exploitation dynamics
                for resc, agents in collision_group_ar.items():  # looping through patches
                    destroy_resc = 0  # if we destroy a patch it is 1
                    for agent in agents:  # looping through all agents on patches
                        # Turn agent towards patch center
                        self.bias_agent_towards_res_center(agent, resc)

                        # One of previous agents on patch consumed the last unit
                        if destroy_resc:
                            notify_agent(agent, -1)

                        # Agent finished pooling on a resource patch
                        if (agent.get_mode() in ["pool", "relocate"] and agent.pool_success) \
                                or agent.pooling_time == 0:
                            # Notify about the patch
                            notify_agent(agent, 1, resc.id)
                            # Teleport agent to the middle of the patch if needed
                            if self.teleport_exploit:
                                agent.position = resc.position + resc.radius - agent.radius

                        # Agent was already exploiting this patch
                        if agent.get_mode() == "exploit":
                            # continue depleting the patch
                            depl_units, destroy_resc = resc.deplete(agent.consumption)
                            agent.collected_r_before = agent.collected_r # rolling resource memory
                            agent.collected_r += depl_units  # and increasing it's collected rescources
                            if destroy_resc:  # consumed unit was the last in the patch
                                notify_agent(agent, -1)

                        # Collect all agents on resource patches
                        agents_on_rescs.append(agent)

                    # Patch is fully depleted
                    if destroy_resc:
                        # we clear it from the memory and regenerate it somewhere else if needed
                        self.kill_resource(resc)

                # Notifying agents that there is no resource patch in current position (they are not on patch)
                for agent in self.agents.sprites():
                    if agent not in agents_on_rescs:  # for all the agents that are not on recourse patches
                        if agent not in collided_agents:  # and are not colliding with each other currently
                            # if they finished pooling
                            if (agent.get_mode() in ["pool",
                                                     "relocate"] and agent.pool_success) or agent.pooling_time == 0:
                                notify_agent(agent, -1)
                            elif agent.get_mode() == "exploit":
                                notify_agent(agent, -1)

                # Update resource patches
                self.rescources.update()

                # Update agents according to current visible obstacles
                self.agents.update(self.agents)

                # move to next simulation timestep
                self.t += 1

            # Simulation is paused
            else:
                # Still calculating visual fields
                for ag in self.agents:
                    ag.calc_social_V_proj(self.agents)

            # Draw environment and agents
            self.draw_frame(stats, stats_pos)

            # Monitoring with IFDB
            if self.save_in_ifd:
                ifdb.save_agent_data(self.ifdb_client, self.agents)
                ifdb.save_resource_data(self.ifdb_client, self.rescources)

            # Moving time forward
            self.clock.tick(self.framerate)

        # Saving data from IFDB when simulation time is over
        if self.save_csv_files:
            if self.save_in_ifd:
                ifdb.save_ifdb_as_csv()
            else:
                raise Exception("Tried to save simulation data as csv file due to env configuration, "
                                "but IFDB logging was turned off. Nothing to save! Please turn on IFDB logging"
                                " or turn off CSV saving feature.")
        pygame.quit()
