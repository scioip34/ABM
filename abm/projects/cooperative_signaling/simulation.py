import os
from datetime import datetime

import numpy as np
import pygame
import pygame_widgets
from pygame_widgets.button import Button
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox

from abm.agent import supcalc
from abm.contrib import colors
from abm.monitoring import ifdb, env_saver
from abm.monitoring.ifdb import pad_to_n_digits
from abm.projects.cooperative_signaling.agent import CSAgent
from abm.simulation.isims import PlaygroundSimulation
import abm.contrib.playgroundtool as pgt
from abm.simulation.sims import Simulation


def setup_coop_sign_playground():
    playground_tool = pgt
    # default_params
    # update default parameters
    playground_tool.default_params["framerate"] = 60
    playground_tool.default_params["N_resc"] = 1
    playground_tool.default_params["patch_radius"] = 1
    # new default parameters
    playground_tool.default_params["collide_agents"] = False
    playground_tool.default_params["phototaxis_theta_step"] = 0.2
    playground_tool.default_params["detection_range"] = 120
    playground_tool.default_params["resource_meter_multiplier"] = 1
    playground_tool.default_params["signalling_cost"] = 0.5
    playground_tool.default_params["des_velocity_res"] = 1.5
    playground_tool.default_params["res_theta_abs"] = 0.2
    # update def_params_to_env_vars
    playground_tool.def_params_to_env_vars[
        "collide_agents"] = "AGENT_AGENT_COLLISION"
    playground_tool.def_params_to_env_vars[
        "phototaxis_theta_step"] = "PHOTOTAX_THETA_FAC"
    playground_tool.def_params_to_env_vars[
        "detection_range"] = "DETECTION_RANGE"
    playground_tool.def_params_to_env_vars[
        "resource_meter_multiplier"] = "METER_TO_RES_MULTI"
    playground_tool.def_params_to_env_vars[
        "signalling_cost"] = "SIGNALLING_COST"
    playground_tool.def_params_to_env_vars["des_velocity_res"] = "RES_VEL"
    playground_tool.def_params_to_env_vars["res_theta_abs"] = "RES_THETA"
    # update help_messages
    playground_tool.help_messages["V_RES"] = '''
        Desired Patch Velocity [px/ts]:

        The desired absolute velocity of the resource patch in pixel per 
        timestep. 
        '''
    playground_tool.help_messages["DET_R"] = '''
        Detection Range [px]:

        detection range of agents in pixels. Resource patch is visualized 
        accordingly with the same radius.
        '''
    return playground_tool


class CSSimulation(Simulation):
    def __init__(self,
                 agent_behave_param_list=None,
                 collide_agents=True,
                 phototaxis_theta_step=0.2,
                 detection_range=120,
                 resource_meter_multiplier=1,
                 signalling_cost=0.5,
                 des_velocity_res=1.5,
                 res_theta_abs=0.2,
                 **kwargs):
        """
        Inherited from Simulation class
        :param phototaxis_theta_step: rotational speed scaling factor during
        phototaxis
        :param detection_range: detection range of resource patches (in pixels)
        :param resource_meter_multiplier: scaling factor of how much resource is
         extraxted for a detected resource unit
        :param signalling_cost: cost of signalling in resource units
        :param des_velocity_res: desired velocity of resource patch in pixel per
        timestep
        :param res_theta_abs: change in orientation will be pulled from uniform
        -res_theta_abs to res_theta_abs
        """
        super().__init__(**kwargs)
        self.agent_behave_param_list = agent_behave_param_list
        self.collide_agents = collide_agents
        self.phototaxis_theta_step = phototaxis_theta_step
        self.detection_range = detection_range
        self.resource_meter_multiplier = resource_meter_multiplier
        self.signalling_cost = signalling_cost
        self.des_velocity_res = des_velocity_res
        self.res_theta_abs = res_theta_abs

        # Agent parameters
        self.phototaxis_theta_step = phototaxis_theta_step
        self.detection_range = detection_range
        self.resource_meter_multiplier = resource_meter_multiplier
        self.signalling_cost = signalling_cost

        # Resource parameters
        self.des_velocity_res = des_velocity_res  # 1.5
        self.res_theta_abs = res_theta_abs  # 0.2

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
                    self.screen.blit(text,
                                     (agent.position[0] + 2 * agent.radius,
                                      agent.position[
                                          1] + 2 * agent.radius + i * (
                                              font_size + spacing)))

    def add_new_agent(self, id, x, y, orient, with_proove=False,
                      behave_params=None):
        """
        Adding a single new agent into agent sprites
        """
        agent_proven = False
        while not agent_proven:
            agent = CSAgent(
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
                phototaxis_theta_step=self.phototaxis_theta_step,
                detection_range=self.detection_range,
                resource_meter_multiplier=self.resource_meter_multiplier,
                signalling_cost=self.signalling_cost,
                patchwise_exclusion=self.patchwise_exclusion,
                behave_params=None
            )
            if with_proove:
                if self.proove_sprite(agent):
                    self.agents.add(agent)
                    agent_proven = True
            else:
                self.agents.add(agent)
                agent_proven = True

    def start(self):
        start_time = datetime.now()
        print(f"Running simulation start method!")
        # Creating N agents in the environment
        print("Creating agents!")
        self.create_agents()
        # Creating resource patches
        print("Creating resources!")
        self.create_resources()
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

            # Updating agent meters
            target_resource = self.rescources.sprites()[0]
            for agent in self.agents.sprites():
                # Currently only implemented with single resource patch
                target_resource = self.rescources.sprites()[0]
                # TODO: add meter to the agent class
                # Saving previous values for phototaxis algorithm
                # agent.prev_meter = agent.meter
                # distance = supcalc.distance(agent, target_resource)
                # if distance < agent.detection_range:
                #     agent.meter = 1 - (distance / agent.detection_range)
                # else:
                #     agent.meter = 0

            if not self.is_paused:
                self.rescources.update()
                # Update agents according to current visible obstacles
                self.agents.update(self.agents)
                # move to next simulation timestep (only when not paused)
                self.t += 1
            # Simulation is paused
            else:
                # Still calculating visual fields
                for ag in self.agents:
                    ag.calc_social_V_proj(self.agents)

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


class CSPlaygroundSimulation(PlaygroundSimulation, CSSimulation):
    """
    SEE: https://docs.python.org/3/tutorial/classes.html#multiple-inheritance
    """

    def __init__(self):
        super().__init__()

        slider_i = 3
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        del self.NRES_slider
        self.NRES_slider = Slider(
            self.screen, self.slider_start_x, slider_start_y, self.slider_width,
            self.slider_height, min=0, max=1, step=1, initial=self.N_resc)

        slider_i = 5
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.DET_R_slider = Slider(
            self.screen, self.slider_start_x, slider_start_y, self.slider_width,
            self.slider_height, min=10, max=200, step=5,
            initial=self.resc_radius)
        self.DET_R_textbox = TextBox(
            self.screen, self.textbox_start_x, slider_start_y,
            self.textbox_width,
            self.textbox_height, fontSize=self.textbox_height - 2,
            borderThickness=1)
        self.DET_R_help = Button(
            self.screen, self.help_start_x, slider_start_y, self.help_width,
            self.help_height,
            text='?', fontSize=self.help_height - 2, inactiveColour=colors.GREY,
            borderThickness=1, )
        self.DET_R_help.onClick = lambda: self.show_help('DET_R',
                                                         self.DET_R_help)
        self.DET_R_help.onRelease = lambda: self.unshow_help(self.DET_R_help)
        # rewrite objects in the lists
        self.help_buttons[slider_i - 1] = self.DET_R_help
        self.sliders[slider_i - 1] = self.DET_R_slider
        self.slider_texts[slider_i - 1] = self.DET_R_textbox

        slider_i = 6
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        del self.Epsw_slider
        self.V_res = 1.5
        self.VRES_slider = Slider(
            self.screen, self.slider_start_x, slider_start_y, self.slider_width,
            self.slider_height, min=0, max=4, step=0.1, initial=self.V_res)
        self.VRES_textbox = TextBox(
            self.screen, self.textbox_start_x, slider_start_y,
            self.textbox_width,
            self.textbox_height, fontSize=self.textbox_height - 2,
            borderThickness=1)
        self.VRES_help = Button(
            self.screen, self.help_start_x, slider_start_y, self.help_width,
            self.help_height,
            text='?', fontSize=self.help_height - 2, inactiveColour=colors.GREY,
            borderThickness=1, )
        self.VRES_help.onClick = lambda: self.show_help('V_RES', self.VRES_help)
        self.VRES_help.onRelease = lambda: self.unshow_help(self.VRES_help)
        # rewrite objects in the lists
        self.help_buttons[slider_i - 1] = self.VRES_help
        self.sliders[slider_i - 1] = self.VRES_slider
        self.slider_texts[slider_i - 1] = self.VRES_textbox

        slider_i = 8
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.Eps_w = 2
        self.Epsw_slider = Slider(
            self.screen, self.slider_start_x, slider_start_y, self.slider_width,
            self.slider_height, min=0, max=5, step=0.1, initial=self.Eps_w)
        self.Epsw_textbox = TextBox(
            self.screen, self.textbox_start_x, slider_start_y,
            self.textbox_width,
            self.textbox_height, fontSize=self.textbox_height - 2,
            borderThickness=1)
        self.Epsw_help = Button(
            self.screen, self.help_start_x, slider_start_y, self.help_width,
            self.help_height,
            text='?', fontSize=self.help_height - 2, inactiveColour=colors.GREY,
            borderThickness=1, )
        self.Epsw_help.onClick = lambda: self.show_help('Epsw', self.Epsw_help)
        self.Epsw_help.onRelease = lambda: self.unshow_help(self.Epsw_help)
        # rewrite objects in the lists
        self.help_buttons[slider_i - 1] = self.Epsw_help
        self.sliders[slider_i - 1] = self.Epsw_slider
        self.slider_texts[slider_i - 1] = self.Epsw_textbox

    def draw_frame(self, stats, stats_pos):
        """
        Overwritten method of sims drawframe adding possibility to update
        pygame widgets
        """
        super().draw_frame(stats, stats_pos)
        self.framerate_textbox.setText(f"Framerate: {self.framerate}")
        self.N_textbox.setText(f"N: {self.N}")
        self.NRES_textbox.setText(f"N_R: {self.N_resc}")
        self.FOV_textbox.setText(f"FOV: {int(self.fov_ratio * 100)}%")
        self.DET_R_textbox.setText(f"DET_R: {int(self.resc_radius)}")
        self.Epsw_textbox.setText(f"E_w: {self.Eps_w:.2f}")
        self.Epsu_textbox.setText(f"E_u: {self.Eps_u:.2f}")
        self.SWU_textbox.setText(f"S_wu: {self.S_wu:.2f}")
        self.VRES_textbox.setText(f"V_res: {self.V_res:.2f}")
        if self.SUM_res == 0:
            self.update_SUMR()
        self.SUMR_textbox.setText(f"SUM R: {self.SUM_res:.2f}")
        for sl in self.sliders:
            sl.draw()
        for slt in self.slider_texts:
            slt.draw()
        for hb in self.help_buttons:
            hb.draw()
        for fb in self.function_buttons:
            fb.draw()
        if self.is_help_shown:
            self.draw_help_message()
        self.draw_global_stats()
        if self.is_recording:
            self.draw_record_circle()
        if self.save_video:
            # Showing the help message before the screen freezes
            self.help_message = "\n\n\n      Saving video, please wait..."
            self.draw_help_message()
            pygame.display.flip()
            # Save video (freezes update process for a while)
            self.saved_images_to_video()
            self.save_video = False

    def interact_with_event(self, events):
        """Carry out functionality according to user's interaction"""
        super().interact_with_event(events)
        pygame_widgets.update(events)
        self.framerate = self.framerate_slider.getValue()
        self.N = self.N_slider.getValue()
        self.N_resc = self.NRES_slider.getValue()
        self.fov_ratio = self.FOV_slider.getValue()
        if self.N != len(self.agents):
            self.act_on_N_mismatch()
        if self.N_resc != len(self.rescources):
            self.act_on_NRES_mismatch()
        if self.fov_ratio != self.agent_fov[1] / np.pi:
            self.update_agent_fovs()
        if self.resc_radius != self.DET_R_slider.getValue():
            self.resc_radius = self.DET_R_slider.getValue()
            self.update_res_radius()
        if self.Eps_w != self.Epsw_slider.getValue():
            self.Eps_w = self.Epsw_slider.getValue()
            self.update_agent_decision_params()
        if self.Eps_u != self.Epsu_slider.getValue():
            self.Eps_u = self.Epsu_slider.getValue()
            self.update_agent_decision_params()
        if self.SUM_res != self.SUMR_slider.getValue():
            self.SUM_res = self.SUMR_slider.getValue()
            self.distribute_sumR()
        if self.S_uw != self.SUW_slider.getValue():
            self.S_uw = self.SUW_slider.getValue()
            self.update_agent_decision_params()
        if self.V_res != self.VRES_slider.getValue():
            self.V_res = self.VRES_slider.getValue()
            self.update_res_radius()
        if self.is_recording:
            filename = f"{pad_to_n_digits(self.t, n=6)}.jpeg"
            path = os.path.join(self.image_save_path, filename)
            pygame.image.save(self.screen, path)
