"""Implementing an interactive Playground Simulation class where the model parameters can be tuned real time"""
import shutil

import pygame
import numpy as np
from math import floor, ceil

from abm.contrib import colors, ifdb_params
from abm.contrib import playgroundtool as pgt
from abm.simulation.sims import Simulation
from pygame_widgets.slider import Slider
from pygame_widgets.button import Button
from pygame_widgets.textbox import TextBox
from abm.monitoring.ifdb import pad_to_n_digits

import pygame_widgets
import os
import cv2


from datetime import datetime

# loading env variables from dotenv file
from dotenv import dotenv_values

EXP_NAME = os.getenv("EXPERIMENT_NAME", "")
root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")

envconf = dotenv_values(env_path)


class PlaygroundSimulation(Simulation):
    def __init__(self):
        super().__init__(**pgt.default_params)
        self.prev_overall_coll_r = 0
        self.overall_col_r = 0
        self.help_message = ""
        self.is_help_shown = False
        self.is_recording = False
        self.save_video = False
        self.video_save_path = os.path.join(root_abm_dir, pgt.VIDEO_SAVE_DIR)
        self.image_save_path = os.path.join(self.video_save_path, "tmp")
        if os.path.isdir(self.image_save_path):
            shutil.rmtree(self.image_save_path)
        os.makedirs(self.image_save_path, exist_ok=True)
        self.vis_area_end_width = 2 * self.window_pad + self.WIDTH
        self.vis_area_end_height = 2 * self.window_pad + self.HEIGHT
        self.global_stats_start = self.vis_area_end_height
        self.action_area_width = 400
        self.action_area_height = 800
        self.full_width = self.WIDTH + self.action_area_width + 2 * self.window_pad
        self.full_height = self.action_area_height

        self.quit_term = False
        self.screen = pygame.display.set_mode([self.full_width, self.full_height], pygame.RESIZABLE)
        self.help_buttons = []

        # pygame widgets
        self.slider_height = 10
        self.textbox_height = 20
        self.help_height = self.textbox_height
        self.help_width = self.help_height
        self.function_button_width = 100
        self.function_button_height = 20
        self.function_button_pad = 20
        self.action_area_pad = 40
        self.textbox_width = 100
        self.slider_width = self.action_area_width - 2 * self.action_area_pad - self.textbox_width - 15
        self.slider_start_x = self.vis_area_end_width + self.action_area_pad
        self.textbox_start_x = self.slider_start_x + self.slider_width + 15
        self.help_start_x = self.textbox_start_x + self.textbox_width + 15

        ## Function Button Row
        self.function_buttons = []
        function_button_start = self.window_pad
        self.start_button = Button(self.screen, function_button_start, self.vis_area_end_height, self.function_button_width,
                                   self.function_button_height, text='Start/Stop', fontSize=self.function_button_height - 2,
                                   inactiveColour=colors.GREEN, borderThickness=1, onClick=lambda: self.start_stop())
        self.function_buttons.append(self.start_button)
        function_button_start += self.function_button_width + self.function_button_pad
        self.record_button = Button(self.screen, function_button_start, self.vis_area_end_height, self.function_button_width,
                                   self.function_button_height, text='Record', fontSize=self.function_button_height - 2,
                                   inactiveColour=colors.GREY, borderThickness=1, onClick=lambda: self.start_stop_record())
        self.function_buttons.append(self.record_button)
        self.global_stats_start += self.function_button_height + self.window_pad

        ## First Slider column
        slider_i = 1
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.framerate_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                       self.slider_height, min=5, max=60, step=1, initial=self.framerate)
        self.framerate_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                         self.textbox_height, fontSize=self.textbox_height - 2, borderThickness=1)
        self.framerate_help = Button(self.screen, self.help_start_x, slider_start_y, self.help_width, self.help_height,
                                     text='?', fontSize=self.help_height - 2, inactiveColour=colors.GREY,
                                     borderThickness=1, )
        self.framerate_help.onClick = lambda: self.show_help('framerate', self.framerate_help)
        self.framerate_help.onRelease = lambda: self.unshow_help(self.framerate_help)
        self.help_buttons.append(self.framerate_help)

        slider_i = 2
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.N_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                               self.slider_height, min=1, max=25, step=1, initial=self.N)
        self.N_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                 self.textbox_height, fontSize=self.textbox_height - 2, borderThickness=1)
        self.N_help = Button(self.screen, self.help_start_x, slider_start_y, self.help_width, self.help_height,
                                     text='?', fontSize=self.help_height - 2, inactiveColour=colors.GREY,
                                     borderThickness=1, )
        self.N_help.onClick = lambda: self.show_help('N', self.N_help)
        self.N_help.onRelease = lambda: self.unshow_help(self.N_help)
        self.help_buttons.append(self.N_help)

        slider_i = 3
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.NRES_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                  self.slider_height, min=1, max=100, step=1, initial=self.N_resc)
        self.NRES_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                    self.textbox_height, fontSize=self.textbox_height - 2, borderThickness=1)
        self.NRES_help = Button(self.screen, self.help_start_x, slider_start_y, self.help_width, self.help_height,
                                     text='?', fontSize=self.help_height - 2, inactiveColour=colors.GREY,
                                     borderThickness=1, )
        self.NRES_help.onClick = lambda: self.show_help('N_res', self.NRES_help)
        self.NRES_help.onRelease = lambda: self.unshow_help(self.NRES_help)
        self.help_buttons.append(self.NRES_help)

        slider_i = 4
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.FOV_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                 self.slider_height, min=0, max=1, step=0.05, initial=self.agent_fov[1] / np.pi)
        self.FOV_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                   self.textbox_height, fontSize=self.textbox_height - 2, borderThickness=1)
        self.FOV_help = Button(self.screen, self.help_start_x, slider_start_y, self.help_width, self.help_height,
                                     text='?', fontSize=self.help_height - 2, inactiveColour=colors.GREY,
                                     borderThickness=1, )
        self.FOV_help.onClick = lambda: self.show_help('FOV', self.FOV_help)
        self.FOV_help.onRelease = lambda: self.unshow_help(self.FOV_help)
        self.help_buttons.append(self.FOV_help)

        slider_i = 5
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.RESradius_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                       self.slider_height, min=10, max=100, step=5, initial=self.resc_radius)
        self.RESradius_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                         self.textbox_height, fontSize=self.textbox_height - 2, borderThickness=1)
        self.RES_help = Button(self.screen, self.help_start_x, slider_start_y, self.help_width, self.help_height,
                                     text='?', fontSize=self.help_height - 2, inactiveColour=colors.GREY,
                                     borderThickness=1, )
        self.RES_help.onClick = lambda: self.show_help('RES', self.RES_help)
        self.RES_help.onRelease = lambda: self.unshow_help(self.RES_help)
        self.help_buttons.append(self.RES_help)

        slider_i = 6
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.Eps_w = 2
        self.Epsw_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                  self.slider_height, min=0, max=5, step=0.1, initial=self.Eps_w)
        self.Epsw_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                    self.textbox_height, fontSize=self.textbox_height - 2, borderThickness=1)
        self.Epsw_help = Button(self.screen, self.help_start_x, slider_start_y, self.help_width, self.help_height,
                                     text='?', fontSize=self.help_height - 2, inactiveColour=colors.GREY,
                                     borderThickness=1, )
        self.Epsw_help.onClick = lambda: self.show_help('Epsw', self.Epsw_help)
        self.Epsw_help.onRelease = lambda: self.unshow_help(self.Epsw_help)
        self.help_buttons.append(self.Epsw_help)

        slider_i = 7
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.Eps_u = 1
        self.Epsu_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                  self.slider_height, min=0, max=5, step=0.1, initial=self.Eps_u)
        self.Epsu_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                    self.textbox_height, fontSize=self.textbox_height - 2, borderThickness=1)
        self.Epsu_help = Button(self.screen, self.help_start_x, slider_start_y, self.help_width, self.help_height,
                                     text='?', fontSize=self.help_height - 2, inactiveColour=colors.GREY,
                                     borderThickness=1, )
        self.Epsu_help.onClick = lambda: self.show_help('Epsu', self.Epsu_help)
        self.Epsu_help.onRelease = lambda: self.unshow_help(self.Epsu_help)
        self.help_buttons.append(self.Epsu_help)

        slider_i = 8
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.S_wu = 0
        self.SWU_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                 self.slider_height, min=0, max=2, step=0.1, initial=self.S_wu)
        self.SWU_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                   self.textbox_height, fontSize=self.textbox_height - 2, borderThickness=1)
        self.SWU_help = Button(self.screen, self.help_start_x, slider_start_y, self.help_width, self.help_height,
                                     text='?', fontSize=self.help_height - 2, inactiveColour=colors.GREY,
                                     borderThickness=1, )
        self.SWU_help.onClick = lambda: self.show_help('SWU', self.SWU_help)
        self.SWU_help.onRelease = lambda: self.unshow_help(self.SWU_help)
        self.help_buttons.append(self.SWU_help)

        slider_i = 9
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.S_uw = 0
        self.SUW_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                 self.slider_height, min=0, max=2, step=0.1, initial=self.S_uw)
        self.SUW_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                   self.textbox_height, fontSize=self.textbox_height - 2, borderThickness=1)
        self.SUW_help = Button(self.screen, self.help_start_x, slider_start_y, self.help_width, self.help_height,
                                     text='?', fontSize=self.help_height - 2, inactiveColour=colors.GREY,
                                     borderThickness=1, )
        self.SUW_help.onClick = lambda: self.show_help('SUW', self.SUW_help)
        self.SUW_help.onRelease = lambda: self.unshow_help(self.SUW_help)
        self.help_buttons.append(self.SUW_help)


        slider_i = 10
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.SUM_res = self.get_total_resource()
        self.SUMR_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                  self.slider_height, min=0, max=self.SUM_res + 200, step=100, initial=self.SUM_res)
        self.SUMR_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                    self.textbox_height, fontSize=self.textbox_height - 2, borderThickness=1)
        self.SUMR_help = Button(self.screen, self.help_start_x, slider_start_y, self.help_width, self.help_height,
                                     text='?', fontSize=self.help_height - 2, inactiveColour=colors.GREY,
                                     borderThickness=1, )
        self.SUMR_help.onClick = lambda: self.show_help('SUMR', self.SUMR_help)
        self.SUMR_help.onRelease = lambda: self.unshow_help(self.SUMR_help)
        self.help_buttons.append(self.SUMR_help)

    def start_stop_record(self):
        """Start or stop the recording of the simulation into a vdieo"""
        if not self.is_recording:
            self.is_recording = True
            self.record_button.inactiveColour = colors.RED
        else:
            self.is_recording = False
            self.save_video = True
            self.record_button.inactiveColour = colors.GREY
            self.help_message = "SAVING VIDEO..."
            self.draw_help_message()


    def start_stop(self):
        self.is_paused = not self.is_paused
        if self.start_button.inactiveColour != colors.GREY:
            self.start_button.inactiveColour = colors.GREY
        else:
            self.start_button.inactiveColour = colors.GREEN

    def show_help(self, help_decide_str, pressed_button):
        for hb in self.help_buttons:
            hb.inactiveColour = colors.GREY
        if not self.is_paused:
            self.is_paused = True
        self.is_help_shown = True
        self.help_message = pgt.help_messages[help_decide_str]
        pressed_button.inactiveColour = colors.GREEN

    def unshow_help(self, pressed_button):
        for hb in self.help_buttons:
            hb.inactiveColour = colors.GREY
        self.is_help_shown = False
        if self.is_paused:
            self.is_paused = False
        pressed_button.inactiveColour = colors.GREY

    def update_SUMR(self):
        self.SUM_res = self.get_total_resource()
        self.SUMR_slider.min = self.N_resc
        self.SUMR_slider.max = 2 * self.SUM_res
        self.SUMR_slider.setValue(self.SUM_res)

    def get_total_resource(self):
        """Calculating the total number of resource units in the arena"""
        SUMR = 0
        for res in self.rescources:
            SUMR += res.resc_units
        return SUMR

    def draw_frame(self, stats, stats_pos):
        """Overwritten method of sims drawframe adding possibility to update pygame widgets"""
        super().draw_frame(stats, stats_pos)
        self.framerate_textbox.setText(f"Framerate: {self.framerate}")
        self.N_textbox.setText(f"N: {self.N}")
        self.NRES_textbox.setText(f"N_R: {self.N_resc}")
        self.FOV_textbox.setText(f"FOV: {int(self.fov_ratio * 100)}%")
        self.RESradius_textbox.setText(f"R_R: {int(self.resc_radius)}")
        self.Epsw_textbox.setText(f"E_w: {self.Eps_w:.2f}")
        self.Epsu_textbox.setText(f"E_u: {self.Eps_u:.2f}")
        self.SUW_textbox.setText(f"S_uw: {self.S_uw:.2f}")
        self.SWU_textbox.setText(f"S_wu: {self.S_wu:.2f}")
        if self.SUM_res == 0:
            self.update_SUMR()
        self.SUMR_textbox.setText(f"SUM R: {self.SUM_res:.2f}")
        self.framerate_textbox.draw()
        self.framerate_slider.draw()
        self.N_textbox.draw()
        self.N_slider.draw()
        self.NRES_textbox.draw()
        self.NRES_slider.draw()
        self.FOV_textbox.draw()
        self.FOV_slider.draw()
        self.RESradius_textbox.draw()
        self.RESradius_slider.draw()
        self.Epsw_textbox.draw()
        self.Epsw_slider.draw()
        self.Epsu_textbox.draw()
        self.Epsu_slider.draw()
        self.SUMR_textbox.draw()
        self.SUMR_slider.draw()
        self.SUW_slider.draw()
        self.SUW_textbox.draw()
        self.SWU_textbox.draw()
        self.SWU_slider.draw()
        for hb in self.help_buttons:
            hb.draw()
        for fb in self.function_buttons:
            fb.draw()
        if self.is_help_shown:
            self.draw_help_message()
        self.draw_global_stats()
        if self.save_video:
            self.help_message = "\n\n\n      SAVING VIDEO, PLEASE WAIT!"
            self.draw_help_message()
            pygame.display.flip()
            self.saved_images_to_video()
            self.save_video = False

    def draw_framerate(self):
        pass

    def draw_global_stats(self):
        image = pygame.Surface([self.vis_area_end_width, self.full_height])
        image.fill(colors.BACKGROUND)
        image.set_colorkey(colors.BACKGROUND)
        image.set_alpha(200)
        line_height = 20
        font = pygame.font.Font(None, line_height)
        status = []
        self.overall_col_r = np.sum([ag.collected_r for ag in self.agents])
        status.append(f"Total collected units: {self.overall_col_r:.2f} U")
        status.append(f"Exploitation Rate: {self.overall_col_r - self.prev_overall_coll_r:.2f} U/timestep")
        for i, stat_i in enumerate(status):
            text_color = colors.BLACK
            text = font.render(stat_i, True, text_color)
            image.blit(text, (self.window_pad, self.global_stats_start + i * line_height))
        self.screen.blit(image, (0, 0))
        self.prev_overall_coll_r = self.overall_col_r

    def draw_help_message(self):
        image = pygame.Surface([self.vis_area_end_width, self.vis_area_end_height])
        image.fill(colors.BACKGROUND)
        image.set_alpha(200)
        line_height = 20
        font = pygame.font.Font(None, line_height)
        status = self.help_message.split("\n")
        for i, stat_i in enumerate(status):
            text_color = colors.BLACK
            text = font.render(stat_i, True, text_color)
            image.blit(text, (self.window_pad, i * line_height))
        self.screen.blit(image, (0, 0))

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
        if self.resc_radius != self.RESradius_slider.getValue():
            self.resc_radius = self.RESradius_slider.getValue()
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
        if self.S_wu != self.SWU_slider.getValue():
            self.S_wu = self.SWU_slider.getValue()
            self.update_agent_decision_params()
        if self.is_recording:
            filename = f"{pad_to_n_digits(self.t, n=6)}.jpeg"
            path = os.path.join(self.image_save_path, filename)
            pygame.image.save(self.screen, path)

    def distribute_sumR(self):
        """If the amount of requestedtotal amount changes we decrease the amount of resource of all resources in a way that
        the original resource ratios remain the same"""
        resource_ratios = []
        remaining_pecents = []
        current_sum_res = self.get_total_resource()
        for ri, res in enumerate(self.rescources):
            resource_ratios.append(res.resc_units / current_sum_res)
            remaining_pecents.append(res.resc_left / res.resc_units)

        # now changing the amount of all and remaining resources according to new sumres
        for ri, res in enumerate(self.rescources):
            res.resc_units = resource_ratios[ri] * self.SUM_res
            res.resc_left = remaining_pecents[ri] * res.resc_units
            res.update()

        self.min_resc_units = floor(self.SUM_res / self.N_resc)
        self.max_resc_units = max(ceil(self.SUM_res / self.N_resc), floor(self.SUM_res / self.N_resc) + 1)

    def update_agent_decision_params(self):
        """Updateing agent decision parameters according to changed slider values"""
        for ag in self.agents:
            ag.Eps_w = self.Eps_w
            ag.Eps_u = self.Eps_u
            ag.S_uw = self.S_uw
            ag.S_wu = self.S_wu

    def pop_resource(self):
        for res in self.rescources:
            res.kill()
            break
        self.N_resc = len(self.rescources)
        self.NRES_slider.setValue(self.N_resc)

    def update_res_radius(self):
        """Changing the resource patch radius according to slider value"""
        # adjusting number of patches
        sum_area = len(self.rescources) * self.resc_radius * self.resc_radius * np.pi
        if sum_area > 0.3 * self.WIDTH * self.HEIGHT:
            while sum_area > 0.3 * self.WIDTH * self.HEIGHT:
                self.pop_resource()
                sum_area = len(self.rescources) * self.resc_radius * self.resc_radius * np.pi

        for res in self.rescources:
            # # update position
            res.position[0] = res.center[0] - self.resc_radius
            res.position[1] = res.center[1] - self.resc_radius
            # self.center = (self.position[0] + self.radius, self.position[1] + self.radius)
            res.radius = self.resc_radius
            res.rect.x = res.position[0]
            res.rect.y = res.position[1]
            res.update()

    def update_agent_fovs(self):
        """Updateing the FOV of agents according to acquired value from slider"""
        self.agent_fov = (-self.fov_ratio * np.pi, self.fov_ratio * np.pi)
        for agent in self.agents:
            agent.FOV = self.agent_fov

    def act_on_N_mismatch(self):
        """method is called if the requested amount of agents is not the same as what the playground already has"""
        if self.N > len(self.agents):
            diff = self.N - len(self.agents)
            for i in range(diff):
                ag_id = len(self.agents) - 1
                x = np.random.randint(self.agent_radii, self.WIDTH - self.agent_radii)
                y = np.random.randint(self.agent_radii, self.HEIGHT - self.agent_radii)
                orient = np.random.uniform(0, 2 * np.pi)
                self.add_new_agent(ag_id, x, y, orient, with_proove=False)
            self.update_agent_decision_params()
            self.update_agent_fovs()
        else:
            while self.N < len(self.agents):
                for i, ag in enumerate(self.agents):
                    if i == len(self.agents) - 1:
                        ag.kill()
        self.stats, self.stats_pos = self.create_vis_field_graph()

    def act_on_NRES_mismatch(self):
        """method is called if the requested amount of patches is not the same as what the playground already has"""
        if self.N_resc > len(self.rescources):
            diff = self.N_resc - len(self.rescources)
            for i in range(diff):
                sum_area = (len(self.rescources) + 1) * self.resc_radius * self.resc_radius * np.pi
                if sum_area > 0.3 * self.WIDTH * self.HEIGHT:
                    while sum_area > 0.3 * self.WIDTH * self.HEIGHT:
                        self.resc_radius -= 5
                        self.RESradius_slider.setValue(self.resc_radius)
                        sum_area = (len(self.rescources) + 1) * self.resc_radius * self.resc_radius * np.pi
                    self.update_res_radius()
                else:
                    self.add_new_resource_patch()
        else:
            while self.N_resc < len(self.rescources):
                for i, res in enumerate(self.rescources):
                    if i == len(self.rescources) - 1:
                        res.kill()
        self.update_SUMR()

    def draw_visual_fields(self):
        """Visualizing the range of vision for agents as opaque circles around the agents"""
        for agent in self.agents:
            FOV = agent.FOV

            # Show limits of FOV
            if 0 < FOV[1] < np.pi:

                # Center and radius of pie chart
                cx, cy, r = agent.position[0] + agent.radius, agent.position[1] + agent.radius, 100

                angle = (2 * FOV[1]) / np.pi * 360
                p = [(cx, cy)]
                # Get points on arc
                angles = [agent.orientation + FOV[0], agent.orientation + FOV[1]]
                step_size = (angles[1] - angles[0]) / 50
                angles_array = np.arange(angles[0], angles[1] + step_size, step_size)
                for n in angles_array:
                    x = cx + int(r * np.cos(n))
                    y = cy + int(r * - np.sin(n))
                    p.append((x, y))
                p.append((cx, cy))

                image = pygame.Surface([self.vis_area_end_width, self.vis_area_end_height])
                image.fill(colors.BACKGROUND)
                image.set_colorkey(colors.BACKGROUND)
                image.set_alpha(10)
                pygame.draw.polygon(image, colors.GREEN, p)
                self.screen.blit(image, (0, 0))

            elif FOV[1] == np.pi:
                image = pygame.Surface([self.vis_area_end_width, self.vis_area_end_height])
                image.fill(colors.BACKGROUND)
                image.set_colorkey(colors.BACKGROUND)
                image.set_alpha(10)
                cx, cy, r = agent.position[0] + agent.radius, agent.position[1] + agent.radius, 100
                pygame.draw.circle(image, colors.GREEN, (cx, cy), r)
                self.screen.blit(image, (0, 0))

    def saved_images_to_video(self):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_folder = self.image_save_path
        video_name = os.path.join(self.video_save_path, f'recording_{timestamp}.mp4')

        images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpeg")])
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, self.framerate, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
        shutil.rmtree(self.image_save_path)
        os.makedirs(self.image_save_path, exist_ok=True)