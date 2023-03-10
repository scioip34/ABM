from datetime import datetime

import numpy as np
import pygame

from abm.agent import supcalc
from abm.contrib import colors
from abm.monitoring import ifdb, env_saver
from abm.projects.visual_flocking.vf_agent.vf_agent import VFAgent
from abm.simulation.sims import Simulation


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

        # making only the used part of retina to the given resolution
        print(f"Original retina resolution: {self.v_field_res}")
        self.v_field_res *= (1/self.fov_ratio)
        self.v_field_res = int(self.v_field_res)
        print(f"Due to limited FOV: {self.fov_ratio} increasing overall resolution to {self.v_field_res}")


    def draw_agent_stats(self, font_size=15, spacing=0):
        """Showing agent information when paused"""
        if self.show_all_stats:
            font = pygame.font.Font(None, font_size)
            for agent in self.agents:
                status = [
                    f"ID: {agent.id}",
                    f"meterval: {agent.meter:.2f}",
                    f"coll.res.: {agent.collected_r:.2f}",
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

    def start(self):
        start_time = datetime.now()
        print(f"Running simulation start method!")
        # Creating N agents in the environment
        print("Creating agents!")
        self.create_agents()
        # Creating surface to show visual fields
        print("Creating visual field graph!")
        self.stats, self.stats_pos = self.create_vis_field_graph()
        # local var to decide when to show visual fields
        turned_on_vfield = 0
        print("Starting main simulation loop!")
        # Main Simulation loop until dedicated simulation time
        while self.t < self.T:
            events = pygame.event.get()
            # Carry out interaction according to user activity
            self.interact_with_event(events)
            # deciding if vis field needs to be shown in this timestep
            turned_on_vfield = self.decide_on_vis_field_visibility(
                turned_on_vfield)

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
