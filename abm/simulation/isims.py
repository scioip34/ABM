"""Implementing an interactive Playground Simulation class where the model parameters can be tuned real time"""
import shutil

import pygame
import numpy as np
from math import floor, ceil

from abm.contrib import colors, ifdb_params
from abm.contrib import playgroundtool as pgt
from abm.monitoring import ifdb
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
        # fixing the total number of possible resources, so it is redistributed with changing NR
        self.SUM_res_fixed = True
        self.prev_overall_coll_r = 0  # total collected resources in previous step
        self.overall_col_r = 0  # total collected resources by all agents
        self.SUM_res = self.get_total_resource()  # total possible amount of resources
        self.help_message = ""
        self.is_help_shown = False
        self.is_recording = False
        self.save_video = False  # trigger to save video from screenshots
        self.video_save_path = os.path.join(root_abm_dir, pgt.VIDEO_SAVE_DIR)  # path to save video
        self.image_save_path = os.path.join(self.video_save_path, "tmp")  # path from collect screenshots
        self.show_all_stats = False
        # enabling paths
        if os.path.isdir(self.image_save_path):
            shutil.rmtree(self.image_save_path)
        os.makedirs(self.image_save_path, exist_ok=True)
        # GUI parameters
        # visualization area of the simulation
        self.vis_area_end_width = 2 * self.window_pad + self.WIDTH
        self.vis_area_end_height = 2 * self.window_pad + self.HEIGHT
        # starting global statistics text
        self.global_stats_start = self.vis_area_end_height
        # start of sliders
        self.action_area_width = 400
        self.action_area_height = 800
        # full window parameters
        self.full_width = self.WIDTH + self.action_area_width + 2 * self.window_pad
        self.full_height = self.action_area_height

        self.quit_term = False
        self.screen = pygame.display.set_mode([self.full_width, self.full_height], pygame.RESIZABLE)

        # button groups
        self.help_buttons = []
        self.function_buttons = []
        # sliders and other gui elements
        self.sliders = []
        self.slider_texts = []

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

        ## Function Button 1st Row
        function_button_start_x = self.window_pad
        function_button_start_y = self.vis_area_end_height
        self.start_button = Button(self.screen, function_button_start_x, function_button_start_y,
                                   self.function_button_width,
                                   self.function_button_height, text='Start/Stop',
                                   fontSize=self.function_button_height - 2,
                                   inactiveColour=colors.GREEN, borderThickness=1, onClick=lambda: self.start_stop())
        self.function_buttons.append(self.start_button)
        function_button_start_x += self.function_button_width + self.function_button_pad
        self.record_button = Button(self.screen, function_button_start_x, function_button_start_y,
                                    self.function_button_width,
                                    self.function_button_height, text='Record Video',
                                    fontSize=self.function_button_height - 2,
                                    inactiveColour=colors.GREY, borderThickness=1,
                                    onClick=lambda: self.start_stop_record())
        self.function_buttons.append(self.record_button)
        function_button_start_x += self.function_button_width + self.function_button_pad
        self.fix_SUM_res_button = Button(self.screen, function_button_start_x, function_button_start_y,
                                         self.function_button_width,
                                         self.function_button_height, text='Fix Total Units',
                                         fontSize=self.function_button_height - 2,
                                         inactiveColour=colors.GREEN, borderThickness=1,
                                         onClick=lambda: self.fix_SUM_res())
        self.function_buttons.append(self.fix_SUM_res_button)
        function_button_start_x += self.function_button_width + self.function_button_pad
        self.show_all_stats_button = Button(self.screen, function_button_start_x, function_button_start_y,
                                            self.function_button_width,
                                            self.function_button_height, text='Show All',
                                            fontSize=self.function_button_height - 2,
                                            inactiveColour=colors.GREY, borderThickness=1,
                                            onClick=lambda: self.show_hide_all_stats())
        self.function_buttons.append(self.show_all_stats_button)

        ## Function Button Second Row
        function_button_start_x = self.window_pad
        function_button_start_y = self.vis_area_end_height + self.function_button_height + self.function_button_pad
        self.visual_exclusion_button = Button(self.screen, function_button_start_x, function_button_start_y,
                                              self.function_button_width,
                                              self.function_button_height, text='Visual Occl.',
                                              fontSize=self.function_button_height - 2,
                                              inactiveColour=colors.GREEN, borderThickness=1,
                                              onClick=lambda: self.change_visual_occlusion())
        self.function_buttons.append(self.visual_exclusion_button)
        function_button_start_x += self.function_button_width + self.function_button_pad
        self.ghost_mode_button = Button(self.screen, function_button_start_x, function_button_start_y,
                                        self.function_button_width,
                                        self.function_button_height, text='Ghost Mode',
                                        fontSize=self.function_button_height - 2,
                                        inactiveColour=colors.GREEN, borderThickness=1,
                                        onClick=lambda: self.change_ghost_mode())
        self.function_buttons.append(self.ghost_mode_button)
        function_button_start_x += self.function_button_width + self.function_button_pad
        self.IFDB_button = Button(self.screen, function_button_start_x, function_button_start_y,
                                  self.function_button_width,
                                  self.function_button_height, text='IFDB Log',
                                  fontSize=self.function_button_height - 2,
                                  inactiveColour=colors.GREY, borderThickness=1,
                                  onClick=lambda: self.start_stop_IFDB_logging())
        self.function_buttons.append(self.IFDB_button)
        function_button_start_x += self.function_button_width + self.function_button_pad
        self.Snapshot_button = Button(self.screen, function_button_start_x, function_button_start_y,
                                  self.function_button_width,
                                  self.function_button_height, text='Take Snapshot',
                                  fontSize=self.function_button_height - 2,
                                  inactiveColour=colors.GREY, borderThickness=1,
                                  onClick=lambda: self.take_snapshot())
        self.function_buttons.append(self.Snapshot_button)

        self.global_stats_start += 2 * self.function_button_height + self.function_button_pad + self.window_pad

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
        self.sliders.append(self.framerate_slider)
        self.slider_texts.append(self.framerate_textbox)

        slider_i = 2
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.N_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                               self.slider_height, min=1, max=35, step=1, initial=self.N)
        self.N_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                 self.textbox_height, fontSize=self.textbox_height - 2, borderThickness=1)
        self.N_help = Button(self.screen, self.help_start_x, slider_start_y, self.help_width, self.help_height,
                             text='?', fontSize=self.help_height - 2, inactiveColour=colors.GREY,
                             borderThickness=1, )
        self.N_help.onClick = lambda: self.show_help('N', self.N_help)
        self.N_help.onRelease = lambda: self.unshow_help(self.N_help)
        self.help_buttons.append(self.N_help)
        self.sliders.append(self.N_slider)
        self.slider_texts.append(self.N_textbox)

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
        self.sliders.append(self.NRES_slider)
        self.slider_texts.append(self.NRES_textbox)

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
        self.sliders.append(self.FOV_slider)
        self.slider_texts.append(self.FOV_textbox)

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
        self.sliders.append(self.RESradius_slider)
        self.slider_texts.append(self.RESradius_textbox)

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
        self.sliders.append(self.Epsw_slider)
        self.slider_texts.append(self.Epsw_textbox)

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
        self.sliders.append(self.Epsu_slider)
        self.slider_texts.append(self.Epsu_textbox)

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
        self.sliders.append(self.SWU_slider)
        self.slider_texts.append(self.SWU_textbox)

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
        self.sliders.append(self.SUW_slider)
        self.slider_texts.append(self.SUW_textbox)

        slider_i = 10
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
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
        self.sliders.append(self.SUMR_slider)
        self.slider_texts.append(self.SUMR_textbox)

    def take_snapshot(self):
        """Taking a single picture of the current status of the replay into an image"""
        filename = f"{pad_to_n_digits(self.t, n=6)}.png"
        path = os.path.join(self.image_save_path, filename)
        pygame.image.save(self.screen, path)
        # cropping image
        img = cv2.imread(path)
        src = img[0:self.vis_area_end_height, 0:self.vis_area_end_width]
        # Convert image to image gray
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        # Applying thresholding technique
        _, alpha = cv2.threshold(tmp, 254, 255, cv2.THRESH_BINARY_INV)

        # Using cv2.split() to split channels
        # of coloured image
        b, g, r = cv2.split(src)

        # Making list of Red, Green, Blue
        # Channels and alpha
        rgba = [b, g, r, alpha]

        # Using cv2.merge() to merge rgba
        # into a coloured/multi-channeled image
        dst = cv2.merge(rgba, 4)
        cv2.imwrite(path, dst)

    def start_stop_IFDB_logging(self):
        """Start or stop IFDB logging in case of grafana interface is used"""
        self.save_in_ifd = not self.save_in_ifd
        if self.save_in_ifd:
            if self.ifdb_client is None:
                self.ifdb_client = ifdb.create_ifclient()
                self.ifdb_client.create_database(ifdb_params.INFLUX_DB_NAME)
            self.write_batch_size = 2
            self.IFDB_button.inactiveColour = colors.GREEN
        else:
            self.write_batch_size = None
            self.IFDB_button.inactiveColour = colors.GREY

    def change_ghost_mode(self):
        """Changing ghost mdoe during exploutation"""
        self.ghost_mode = not self.ghost_mode
        if self.ghost_mode:
            self.ghost_mode_button.inactiveColour = colors.GREEN
        else:
            self.ghost_mode_button.inactiveColour = colors.GREY

    def change_visual_occlusion(self):
        """Changing visual occlusion parameter"""
        self.visual_exclusion = not self.visual_exclusion
        for ag in self.agents:
            ag.visual_exclusion = self.visual_exclusion
        if self.visual_exclusion:
            self.visual_exclusion_button.inactiveColour = colors.GREEN
        else:
            self.visual_exclusion_button.inactiveColour = colors.GREY

    def show_hide_all_stats(self):
        """Show or hide all information"""
        self.show_all_stats = not self.show_all_stats
        if self.show_all_stats:
            self.show_all_stats_button.inactiveColour = colors.GREEN
            for ag in self.agents:
                ag.show_stats = True
            for res in self.rescources:
                res.show_stats = True
        else:
            self.show_all_stats_button.inactiveColour = colors.GREY
            for ag in self.agents:
                ag.show_stats = False
            for res in self.rescources:
                res.show_stats = False

    def fix_SUM_res(self):
        """Fixing total amount of possible resources so it will not change with changing the number of patches"""
        self.SUM_res_fixed = not self.SUM_res_fixed
        if self.SUM_res_fixed:
            self.fix_SUM_res_button.inactiveColour = colors.GREEN
        else:
            self.fix_SUM_res_button.inactiveColour = colors.GREY

    def start_stop_record(self):
        """Start or stop the recording of the simulation into a vdieo"""
        if not self.is_recording:
            self.is_recording = True
            self.record_button.inactiveColour = colors.RED
            self.record_button.string = "Stop Recording"
            self.record_button.text = self.record_button.font.render(self.record_button.string, True,
                                                                     self.record_button.textColour)
        else:
            self.is_recording = False
            self.save_video = True
            self.record_button.inactiveColour = colors.GREY
            self.record_button.string = "Record Video"
            self.record_button.text = self.record_button.font.render(self.record_button.string, True,
                                                                     self.record_button.textColour)
            self.help_message = "SAVING VIDEO..."
            self.draw_help_message()

    def start_stop(self):
        """Switch to start or stop the simulation"""
        self.is_paused = not self.is_paused
        if self.start_button.inactiveColour != colors.GREY:
            self.start_button.inactiveColour = colors.GREY
        else:
            self.start_button.inactiveColour = colors.GREEN

    def show_help(self, help_decide_str, pressed_button):
        """Switch to show help message"""
        for hb in self.help_buttons:
            hb.inactiveColour = colors.GREY
        if not self.is_paused:
            self.is_paused = True
        self.is_help_shown = True
        self.help_message = pgt.help_messages[help_decide_str]
        pressed_button.inactiveColour = colors.GREEN

    def unshow_help(self, pressed_button):
        """Switch to erease help message from screen"""
        for hb in self.help_buttons:
            hb.inactiveColour = colors.GREY
        self.is_help_shown = False
        if self.is_paused:
            self.is_paused = False
        pressed_button.inactiveColour = colors.GREY

    def update_SUMR(self):
        """Updating the total possible resource units if necessary"""
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

    def draw_record_circle(self):
        """Drawing a red circle to show that the frame is recording"""
        if self.t % 60 < 30:
            circle_rad = int(self.window_pad / 4)
            image = pygame.Surface([2 * circle_rad, 2 * circle_rad])
            image.fill(colors.BACKGROUND)
            image.set_colorkey(colors.BACKGROUND)
            pygame.draw.circle(
                image, colors.RED, (circle_rad, circle_rad), circle_rad
            )
            self.screen.blit(image, (circle_rad, circle_rad))

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
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                print("Snapshot!")
                self.take_snapshot()
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
                ag_id = len(self.agents)
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
        if self.show_all_stats:
            for ag in self.agents:
                ag.show_stats = True
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
        if not self.SUM_res_fixed:
            self.update_SUMR()
        else:
            self.distribute_sumR()
        if self.show_all_stats:
            for res in self.rescources:
                res.show_stats = True

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
