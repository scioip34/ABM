"""Implementing an interactive Playground Simulation class where the model parameters can be tuned real time"""

import pygame
import numpy as np
import sys

from abm.agent import supcalc
from abm.agent.agent import Agent
from abm.environment.rescource import Rescource
from abm.contrib import colors, ifdb_params
from abm.contrib import playgroundtool as pgt
from abm.simulation.sims import Simulation
from pygame_widgets.slider import Slider
from pygame_widgets.button import Button
from pygame_widgets.textbox import TextBox
from pygame_widgets.dropdown import Dropdown
import pygame_widgets
from abm.monitoring import ifdb
from abm.monitoring import env_saver
from math import atan2
import os
import uuid

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
        self.vis_area_end_width = 2 * self.window_pad + self.WIDTH
        self.vis_area_end_height = 2 * self.window_pad + self.HEIGHT
        self.action_area_width = 400
        self.action_area_height = 800
        self.full_width = self.WIDTH + self.action_area_width + 2 * self.window_pad
        self.full_height = self.action_area_height

        self.quit_term = False
        self.screen = pygame.display.set_mode([self.full_width, self.full_height], pygame.RESIZABLE)

        # pygame widgets
        self.slider_height = 20
        self.action_area_pad = 30
        self.textbox_width = 100
        self.slider_width = self.action_area_width - 2 * self.action_area_pad - self.textbox_width - 15
        self.slider_start_x = self.vis_area_end_width + self.action_area_pad
        self.textbox_start_x = self.slider_start_x + self.slider_width + 15

        ## First Slider column
        slider_i = 1
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.framerate_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                       self.slider_height, min=5, max=60, step=1, initial=self.framerate)
        self.framerate_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                         self.slider_height, fontSize=self.slider_height - 2, borderThickness=1)
        slider_i = 2
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.N_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                       self.slider_height, min=1, max=25, step=1, initial=self.N)
        self.N_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                         self.slider_height, fontSize=self.slider_height - 2, borderThickness=1)
        slider_i = 3
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.NRES_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                       self.slider_height, min=1, max=100, step=1, initial=self.N_resc)
        self.NRES_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                         self.slider_height, fontSize=self.slider_height - 2, borderThickness=1)
        slider_i = 4
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.FOV_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                       self.slider_height, min=0, max=1, step=0.05, initial=self.agent_fov[1]/np.pi)
        self.FOV_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                         self.slider_height, fontSize=self.slider_height - 2, borderThickness=1)
        slider_i = 5
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.RESradius_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                       self.slider_height, min=10, max=100, step=5, initial=self.resc_radius)
        self.RESradius_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                         self.slider_height, fontSize=self.slider_height - 2, borderThickness=1)
        slider_i = 6
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.Eps_w = 2
        self.Epsw_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                       self.slider_height, min=0, max=5, step=0.1, initial=self.Eps_w)
        self.Epsw_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                         self.slider_height, fontSize=self.slider_height - 2, borderThickness=1)
        slider_i = 7
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.Eps_u = 1
        self.Epsu_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                       self.slider_height, min=0, max=5, step=0.1, initial=self.Eps_u)
        self.Epsu_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                         self.slider_height, fontSize=self.slider_height - 2, borderThickness=1)

    def draw_frame(self, stats, stats_pos):
        """Overwritten method of sims drawframe adding possibility to update pygame widgets"""
        super().draw_frame(stats, stats_pos)
        self.framerate_textbox.setText(f"Framerate: {self.framerate}")
        self.N_textbox.setText(f"N: {self.N}")
        self.NRES_textbox.setText(f"N_R: {self.N_resc}")
        self.FOV_textbox.setText(f"FOV: {int(self.fov_ratio*100)}%")
        self.RESradius_textbox.setText(f"R_R: {int(self.resc_radius)}")
        self.Epsw_textbox.setText(f"E_w: {self.Eps_w:.2f}")
        self.Epsu_textbox.setText(f"E_u: {self.Eps_u:.2f}")
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
        if self.fov_ratio != self.agent_fov[1]/np.pi:
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

    def update_agent_decision_params(self):
        """Updateing agent decision parameters according to changed slider values"""
        for ag in self.agents:
            ag.Eps_w = self.Eps_w
            ag.Eps_u = self.Eps_u

    def update_res_radius(self):
        """Changing the resource patch radius according to slider value"""
        # adjusting number of patches
        for res in self.rescources:
            # # update position
            res.position[0] = res.center[0] - self.resc_radius
            res.position[1] = res.center[1] - self.resc_radius
            # self.center = (self.position[0] + self.radius, self.position[1] + self.radius)
            res.radius = self.resc_radius
            res.rect.x = res.position[0]
            res.rect.y = res.position[1]

    def update_agent_fovs(self):
        """Updateing the FOV of agents according to acquired value from slider"""
        self.agent_fov = (-self.fov_ratio*np.pi, self.fov_ratio*np.pi)
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
                    if i == len(self.agents)-1:
                        ag.kill()
        self.stats, self.stats_pos = self.create_vis_field_graph()

    def act_on_NRES_mismatch(self):
        """method is called if the requested amount of patches is not the same as what the playground already has"""
        if self.N_resc > len(self.rescources):
            diff = self.N_resc - len(self.rescources)
            for i in range(diff):
                sum_area = (len(self.rescources)+1) * self.resc_radius * self.resc_radius * np.pi
                if sum_area > 0.3 * self.WIDTH * self.HEIGHT:
                    while sum_area > 0.3 * self.WIDTH * self.HEIGHT:
                        self.resc_radius -= 5
                        self.RESradius_slider.setValue(self.resc_radius)
                        sum_area = (len(self.rescources)+1) * self.resc_radius * np.pi * np.pi
                    self.update_res_radius()
                else:
                    self.add_new_resource_patch()
        else:
            while self.N_resc < len(self.rescources):
                for i, res in enumerate(self.rescources):
                    if i == len(self.rescources)-1:
                        res.kill()

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
