from datetime import datetime

import numpy as np
import pygame

from abm.agent import supcalc
from abm.contrib import colors
from abm.monitoring import ifdb, env_saver
from abm.projects.visual_flocking.vf_agent.vf_agent import VFAgent
from abm.projects.visual_flocking.vf_contrib import vf_params
from abm.simulation.sims import Simulation
from matplotlib import cm as colmaps


class VFSimulation(Simulation):
    def __init__(self, **kwargs):
        """
        Inherited from Simulation class
        :param phototaxis_theta_step: rotational speed scaling factor during
        phototaxis
        :param detection_range: detection range of resource patches (in pixels)
        :param resource_meter_multiplier: scaling factor of how much resource is
         extraxted for a detected resource unit
        :param signalling_cost: cost of signalling in resource units
        :param probability_of_starting_signaling: probability of starting signaling
        :param des_velocity_res: desired velocity of resource patch in pixel per
        timestep
        :param res_theta_abs: change in orientation will be pulled from uniform
        -res_theta_abs to res_theta_abs
        :param agent_signaling_rand_event_update: updating agent's
         random number for signalling probability in every N simulation time step
        """
        super().__init__(**kwargs)
        self.show_all_stats = False
        # line following prototype
        self.agent_for_line_id = []
        self.lines = [[]]  # a list of guide lines that agents will follow where each line is a list of (x, y) points
        self.line_map = np.zeros((self.WIDTH + self.window_pad, self.HEIGHT + self.window_pad))

        # making only the used part of retina to the given resolution
        print(f"Original retina resolution: {self.v_field_res}")
        self.v_field_res *= (1/self.fov_ratio)
        self.v_field_res = int(self.v_field_res)
        print(f"Due to limited FOV: {self.fov_ratio} increasing overall resolution to {self.v_field_res}")

        self.show_path_history = False
        self.memory_length = 30
        self.ori_memory = None
        self.pos_memory = None

    def update_lines_to_follow(self):
        """Updating background line map to follow"""
        subsurface = pygame.Surface((self.WIDTH + self.window_pad, self.HEIGHT + self.window_pad))
        subsurface.fill(colors.BACKGROUND)
        subsurface.set_colorkey(colors.WHITE)
        subsurface.set_alpha(255)
        for line in self.lines:
            for pi in range(1, len(line)):
                point1 = line[pi-1]
                point2 = line[pi]
                color = colors.BLACK
                pygame.draw.line(subsurface, color, point1, point2, 4)

        for ag in self.agents:
            if len(ag.lines)!=0:
                sensor1_pos = [ag.position[1] + ag.radius - ag.sensor_distance + (1 + np.sin(ag.orientation + (3*np.pi / 4))) * ag.sensor_distance,
                               ag.position[0] + ag.radius - ag.sensor_distance + (1 - np.cos(ag.orientation + (3*np.pi / 4))) * ag.sensor_distance]
                sensor2_pos = [ag.position[1] + ag.radius - ag.sensor_distance + (1 + np.sin(ag.orientation - (3*np.pi / 4))) * ag.sensor_distance,
                               ag.position[0] + ag.radius - ag.sensor_distance + (1 - np.cos(ag.orientation - (3*np.pi / 4))) * ag.sensor_distance]

                pygame.draw.circle(
                    subsurface, colors.GREEN, sensor2_pos, ag.sensor_size
                )
                pygame.draw.circle(
                    subsurface, colors.GREEN, sensor1_pos, ag.sensor_size
                )
        line_map = pygame.surfarray.array3d(subsurface)
        self.line_map = line_map.swapaxes(0, 1)[:, :, 0]/255


    def handle_mouse_wheel_event(self, event):
        """Handling event if mouse wheel is moved"""
        if event.type == pygame.MOUSEWHEEL:
            print(event.y)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_a]:
                for agid, ag in enumerate(list(self.agents)):
                    if ag.rect.collidepoint(pygame.mouse.get_pos()):
                        if ag.ALP0 is None:
                            ag.ALP0 = vf_params.ALP0
                        else:
                            ag.ALP0 += event.y * 0.05
            elif keys[pygame.K_b]:
                for agid, ag in enumerate(list(self.agents)):
                    if ag.rect.collidepoint(pygame.mouse.get_pos()):
                        if ag.BET0 is None:
                            ag.BET0 = vf_params.BET0
                        else:
                            ag.BET0 += event.y * 0.05
            elif keys[pygame.K_r]:
                for agid, ag in enumerate(list(self.agents)):
                    if ag.rect.collidepoint(pygame.mouse.get_pos()):
                        ag.radius += int(event.y)
            elif keys[pygame.K_v]:
                for agid, ag in enumerate(list(self.agents)):
                    if ag.rect.collidepoint(pygame.mouse.get_pos()):
                        if ag.V0 is None:
                            ag.V0 = vf_params.V0
                        else:
                            ag.V0 += event.y * 0.05
            else:
                if event.y == -1:
                    event.y = 0
                for ag in self.agents:
                    ag.move_with_mouse(pygame.mouse.get_pos(), event.y, 1 - event.y)


    def handle_cursor_event(self, event):
        """Handling event if cursor buttons are clicked overwriting base to add line functionalities"""
        if pygame.mouse.get_pressed()[0]:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_l]:
                for agid, ag in enumerate(list(self.agents)):
                    if ag.rect.collidepoint(pygame.mouse.get_pos()):
                        self.agent_for_line_id.append(agid)
                        print(f"Assigned agent {agid} for current line, will finalize when release l button!")
                        # assigning the last line to the given agent
                        ag.color = colors.RED
                        # ag.lines.append(self.lines[-1])
                        # print(ag.lines)

                # tart adding points to it
                if (pygame.mouse.get_pos()[1], pygame.mouse.get_pos()[0]) not in self.lines[-1]:
                    self.lines[-1].append((pygame.mouse.get_pos()[1], pygame.mouse.get_pos()[0]))
                    print(len(self.lines), len(self.lines[-1]))
            else:
                try:
                    for ag in self.agents:
                        ag.move_with_mouse(event.pos, 0, 0)
                    for res in self.rescources:
                        res.update_clicked_status(event.pos)
                except AttributeError:
                    for ag in self.agents:
                        ag.move_with_mouse(pygame.mouse.get_pos(), 0, 0)
        else:
            for ag in self.agents:
                ag.is_moved_with_cursor = False
                ag.draw_update()
            for res in self.rescources:
                res.is_clicked = False
                res.draw_update()

    def draw_lines_to_follow(self):
        """Draw lines that agents follow"""
        self.update_lines_to_follow()
        subsurface = pygame.surfarray.make_surface(self.line_map*255)
        subsurface.set_colorkey(colors.WHITE)
        subsurface.set_alpha(255)
        self.screen.blit(subsurface, (0, 0))


    def draw_agent_stats(self, font_size=15, spacing=0):
        """Showing agent information when paused"""
        if self.show_all_stats:
            font = pygame.font.Font(None, font_size)
            for agent in self.agents:
                a0 = agent.ALP0 if agent.ALP0 is not None else vf_params.ALP0
                b0 = agent.BET0 if agent.BET0 is not None else vf_params.BET0
                v0 = agent.V0 if agent.V0 is not None else vf_params.V0
                status = [
                    f"ID: {agent.id}",
                    f"A0: {a0}",
                    f"B0: {b0}",
                    f"V0: {v0}",
                ]
                for i, stat_i in enumerate(status):
                    text = font.render(stat_i, True, colors.BLACK)
                    self.screen.blit(
                        text,
                        (agent.position[0] + 2 * agent.radius,
                         agent.position[1] + 2 * agent.radius + i * (
                                 font_size + spacing)))

    def add_new_agent(self, id, x, y, orient, with_proove=False,
                      behave_params=None):
        """
        Adding a single new agent into agent sprites
        """
        agent_proven = False
        while not agent_proven:
            agent = VFAgent(
                id=id,
                radius=self.agent_radii,
                position=(x, y),
                orientation=orient,
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
                patchwise_exclusion=self.patchwise_exclusion,
                behave_params=None
            )
            self.agents.add(agent)
            agent_proven = True

    def create_agents(self):
        """
        Creating agents according to how the simulation class was initialized
        """
        for i in range(self.N):
            orient = np.random.uniform(0, 2 * np.pi)
            dist = np.random.rand() * (
                    self.HEIGHT / 2 - 2 * self.window_pad - self.agent_radii)
            x = np.cos(orient) * dist + self.WIDTH / 2
            y = np.sin(orient) * dist + self.HEIGHT / 2
            if not self.heterogen_agents:
                # create agents according to environment variables homogeneously
                self.add_new_agent(i, x, y, orient)
            else:
                self.add_new_agent(
                    i, x, y, orient,
                    behave_params=self.agent_behave_param_list[i])

    def save_ori_pos_history(self):
        """Saving orientation and position history of agents to visualize paths"""
        if self.ori_memory is None:
            self.ori_memory = np.zeros((len(self.agents), self.memory_length))
            self.pos_memory = np.zeros((len(self.agents), 2, self.memory_length))
        try:
            self.ori_memory = np.roll(self.ori_memory, 1, axis=-1)
            self.pos_memory = np.roll(self.pos_memory, 1, axis=-1)
            self.ori_memory[:, 0] = np.array([ag.orientation for ag in self.agents])
            self.pos_memory[:, 0, 0] = np.array([ag.position[0]+ag.radius for ag in self.agents])
            self.pos_memory[:, 1, 0] = np.array([ag.position[1]+ag.radius for ag in self.agents])
        except:
            self.ori_memory = None
            self.pos_memory = None

    def draw_agent_paths_vf(self):
        if self.ori_memory is not None:
            path_length = self.memory_length
            cmap = colmaps.get_cmap('jet')
            transparency = 0.5
            transparency = int(transparency * 255)
            big_colors = cmap(self.ori_memory / (2 * np.pi)) * 255
            # setting alpha
            surface = pygame.Surface((self.WIDTH + self.window_pad, self.HEIGHT + self.window_pad))
            surface.fill(colors.BACKGROUND)
            surface.set_colorkey(colors.WHITE)
            surface.set_alpha(255)
            try:
                for ai, agent in enumerate(self.agents):
                    subsurface = pygame.Surface((self.WIDTH + self.window_pad, self.HEIGHT + self.window_pad))
                    subsurface.fill(colors.BACKGROUND)
                    subsurface.set_colorkey(colors.WHITE)
                    subsurface.set_alpha(transparency)
                    for t in range(2, path_length, 1):
                        point2 = self.pos_memory[ai, :, t]
                        color = big_colors[ai, t]
                        # pygame.draw.line(surface1, color, point1, point2, 4)
                        pygame.draw.circle(subsurface, color, point2, max(2, int(self.agent_radii / 2)))
                    surface.blit(subsurface, (0, 0))
                self.screen.blit(surface, (0, 0))
            except IndexError as e:
                pass


    def draw_frame(self, stats, stats_pos):
        """Drawing environment, agents and every other visualization in each timestep"""
        self.screen.fill(colors.BACKGROUND)
        self.draw_lines_to_follow()
        self.draw_walls()
        if self.show_path_history:
            self.draw_agent_paths_vf()
        self.agents.draw(self.screen)
        if self.show_vision_range:
            self.draw_visual_fields()
        self.draw_framerate()
        self.draw_agent_stats()

        if self.show_vis_field:
            # showing visual fields of the agents
            self.show_visual_fields(stats, stats_pos)

    def step_sim(self):
        """Stepping a single time step in the simulation loop"""
        events = pygame.event.get()
        # Carry out interaction according to user activity
        self.interact_with_event(events)
        # deciding if vis field needs to be shown in this timestep
        self.turned_on_vfield = self.decide_on_vis_field_visibility(
            self.turned_on_vfield)

        if not self.is_paused:
            # Update agents according to current visible obstacles
            self.agents.update(self.agents)
            # move to next simulation timestep (only when not paused)
            self.t += 1
        # Simulation is paused
        else:
            # Still calculating visual fields
            for ag in self.agents:
                # DEBUG: updating visual projections also when paused
                ag.update_social_info(self.agents)

        if self.show_path_history:
            self.save_ori_pos_history()

        # Draw environment and agents
        if self.with_visualization:
            self.draw_frame(self.stats, self.stats_pos)
            pygame.display.flip()
        # Monitoring with IFDB (also when paused)
        if self.save_in_ifd:
            ifdb.save_agent_data(
                self.ifdb_client, self.agents, self.t,
                exp_hash=self.ifdb_hash,
                batch_size=self.write_batch_size)
            ifdb.save_resource_data(
                self.ifdb_client, self.rescources, self.t,
                exp_hash=self.ifdb_hash,
                batch_size=self.write_batch_size)
        elif self.save_in_ram:
            # saving data in ram for data processing, only when not paused
            if not self.is_paused:
                ifdb.save_agent_data_RAM(self.agents, self.t)
                ifdb.save_resource_data_RAM(self.rescources, self.t)
        # Moving time forward
        if self.t % 500 == 0 or self.t == 1:
            print(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')}"
                f" t={self.t}")
            print(f"Simulation FPS: {self.clock.get_fps()}")
        self.clock.tick(self.framerate)

    def prepare_start(self):
        """Preparing start method"""
        print(f"Running simulation start method!")
        # Creating N agents in the environment
        print("Creating agents!")
        self.create_agents()
        # Creating surface to show visual fields
        print("Creating visual field graph!")
        self.stats, self.stats_pos = self.create_vis_field_graph()
        # local var to decide when to show visual fields
        self.turned_on_vfield = 0


    def start(self):
        start_time = datetime.now()
        self.prepare_start()
        # Main Simulation loop until dedicated simulation time
        print("Starting main simulation loop!")
        while self.t < self.T:
            self.step_sim()
        end_time = datetime.now()
        print(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')}"
            f" Total simulation time: ",
            (end_time - start_time).total_seconds())
        # Saving data from IFDB when simulation time is over
        if self.agent_behave_param_list is not None:
            if self.agent_behave_param_list[0].get(
                    "evo_summary_path") is not None:
                pop_num = self.generate_evo_summary()
        else:
            pop_num = None
        if self.save_csv_files:
            if self.save_in_ifd or self.save_in_ram:
                ifdb.save_ifdb_as_csv(exp_hash=self.ifdb_hash,
                                      use_ram=self.save_in_ram,
                                      as_zar=self.use_zarr,
                                      save_extracted_vfield=False,
                                      pop_num=pop_num)
                env_saver.save_env_vars([self.env_path], "env_params.json",
                                        pop_num=pop_num)
            else:
                raise Exception(
                    "Tried to save simulation data as csv file due to env "
                    "configuration, "
                    "but IFDB/RAM logging was turned off. Nothing to save! "
                    "Please turn on IFDB/RAM logging"
                    " or turn off CSV saving feature.")
        end_save_time = datetime.now()
        print(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')} "
            f"Total saving time:",
            (end_save_time - end_time).total_seconds())

        pygame.quit()
        # sys.exit()
