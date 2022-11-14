import os

import numpy as np
import pygame
import pygame_widgets
from pygame_widgets.button import Button
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox

from abm.contrib import colors
from abm.monitoring.ifdb import pad_to_n_digits
from abm.projects.cooperative_signaling.cs_simulation.cs_sims import \
    CSSimulation
from abm.simulation.isims import PlaygroundSimulation


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

    def update_res_radius(self):
        """Changing the resource patch radius according to slider value"""

        for res in self.rescources:
            # # update position
            res.position[0] = res.center[0] - self.resc_radius
            res.position[1] = res.center[1] - self.resc_radius
            res.radius = self.resc_radius
            res.des_velocity = self.V_res
            res.rect.x = res.position[0]
            res.rect.y = res.position[1]
            res.draw_update()

        for agent in self.agents:
            agent.detection_range = self.resc_radius

    def act_on_NRES_mismatch(self):
        """
        method is called if the requested amount of patches is not the same
        as what the playground already has
        """
        pass

    def act_on_N_mismatch(self):
        """
        method is called if the requested amount of agents is not the same as
        what the playground already has
        """
        if self.N > len(self.agents):
            diff = self.N - len(self.agents)
            for i in range(diff):
                ag_id = len(self.agents)
                orient = np.random.uniform(0, 2 * np.pi)
                dist = np.random.rand() * (
                        self.HEIGHT / 2 - 2 * self.window_pad -
                        self.agent_radii)
                x = np.cos(orient) * dist + self.WIDTH / 2
                y = np.sin(orient) * dist + self.HEIGHT / 2
                self.add_new_agent(ag_id, x, y, orient, with_proove=False)
            self.update_agent_decision_params()
            self.update_agent_fovs()
        else:
            while self.N < len(self.agents):
                for i, ag in enumerate(self.agents):
                    if i == len(self.agents) - 1:
                        ag.kill()
        if self.show_all_stats:
            for ag in self.agents:
                ag.show_stats = True
        self.stats, self.stats_pos = self.create_vis_field_graph()
