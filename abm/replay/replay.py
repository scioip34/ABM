import sys

from abm.loader.data_loader import ExperimentLoader
from abm.contrib import colors
from pygame_widgets.slider import Slider
from pygame_widgets.button import Button
from pygame_widgets.textbox import TextBox
import pygame_widgets
import pygame
import numpy as np


class ExperimentReplay:
    def __init__(self, data_folder_path):
        """Initialization method to replay recorded simulations from their summary folder. If a summary is not yet
        available for the experiment it will be summarized first"""
        self.experiment = ExperimentLoader(data_folder_path, enforce_summary=False, with_plotting=False)
        # todo: this initialization will fail when we systematically change width and height in experiment
        self.WIDTH = int(float(self.experiment.env["ENV_WIDTH"]))
        self.HEIGHT = int(float(self.experiment.env["ENV_HEIGHT"]))
        self.T = int(float(self.experiment.env["T"]))
        self.window_pad = 30
        self.vis_area_end_width = 2 * self.window_pad + self.WIDTH
        self.action_area_width = 400
        self.action_area_height = 800
        self.full_width = self.WIDTH + self.action_area_width + 2 * self.window_pad
        self.full_height = self.action_area_height

        self.env = self.experiment.env

        self.posx = self.experiment.agent_summary['posx']
        self.posy = self.experiment.agent_summary['posy']
        self.orientation = self.experiment.agent_summary['orientation']
        self.agmodes = self.experiment.agent_summary['mode']

        self.res_pos_x = self.experiment.res_summary['posx']
        self.res_pos_y = self.experiment.res_summary['posy']

        self.varying_params = self.experiment.varying_params

        self.is_paused = True
        self.t = 0
        self.framerate = 25
        self.num_batches = self.experiment.num_batches
        self.batch_id = 0

        #self.experiment = None

        # Initializing pygame
        self.quit_term = False
        pygame.init()
        self.screen = pygame.display.set_mode([self.full_width, self.full_height])
        self.clock = pygame.time.Clock()

        # pygame widgets
        self.slider_height = 20
        self.action_area_pad = 30
        self.textbox_width = 100
        self.slider_width = self.action_area_width - 2 * self.action_area_pad - self.textbox_width - 15
        self.slider_start_x = self.vis_area_end_width + self.action_area_pad
        self.textbox_start_x = self.slider_start_x + self.slider_width + 15

        slider_i = 1
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.framerate_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                       self.slider_height, min=5, max=60, step=1, initial=self.framerate)
        self.framerate_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                         self.slider_height, fontSize=self.slider_height - 2, borderThickness=1)
        slider_i = 2
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.time_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                  self.slider_height, min=0, max=self.T - 1, step=1, initial=0)
        self.time_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                    self.slider_height, fontSize=self.slider_height - 2, borderThickness=1)
        slider_i = 3
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.batch_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                   self.slider_height, min=0, max=self.num_batches - 1, step=1, initial=0)
        self.batch_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                     self.slider_height, fontSize=self.slider_height - 2, borderThickness=1)

        slider_i = 4
        self.varying_sliders = []
        self.varying_textboxes = []
        self.varying_dimensions = {}
        vpi = 0
        for k, v in self.varying_params.items():
            self.varying_dimensions[vpi] = 0
            slider_start_y = (slider_i + vpi) * (self.slider_height + self.action_area_pad)
            self.varying_sliders.append(
                Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                       self.slider_height, min=0, max=len(v) - 1, step=1, initial=0))
            self.varying_textboxes.append(
                TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                        self.slider_height, fontSize=self.slider_height - 2, borderThickness=1))
            vpi += 1

        self.button_height = 50
        button_start_y = (slider_i + vpi) * (self.slider_height + self.action_area_pad)
        # Creates the button with optional parameters
        self.run_button = Button(
            # Mandatory Parameters
            self.screen,  # Surface to place button on
            self.slider_start_x,  # X-coordinate of top left corner
            button_start_y,  # Y-coordinate of top left corner
            int(self.slider_width / 2),  # Width
            self.button_height,  # Height

            # Optional Parameters
            text='Start / Stop',  # Text to display
            fontSize=20,  # Size of font
            margin=20,  # Minimum distance between text/image and edge of button
            inactiveColour=colors.GREY,
            onClick=lambda: self.on_run()  # Function to call when clicked on
        )

    def on_run(self):
        self.is_paused = not self.is_paused
        if not self.is_paused:
            self.run_button.inactiveColour = colors.GREEN
        else:
            self.run_button.inactiveColour = colors.GREY

    def draw_walls(self):
        """Drwaing walls on the arena according to initialization, i.e. width, height and padding"""
        pygame.draw.line(self.screen, colors.RED,
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

    def draw_separator(self):
        """Drawing separation line between action area and visualization"""
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.vis_area_end_width, 0],
                         [self.vis_area_end_width, self.full_height])

    def draw_frame(self, events):
        """Drawing environment, agents and every other visualization in each timestep"""
        self.screen.fill(colors.BACKGROUND)

        self.draw_separator()
        pygame_widgets.update(events)
        self.framerate = self.framerate_slider.getValue()
        self.framerate_textbox.setText(f"framerate: {self.framerate}")
        self.framerate_textbox.draw()
        self.t = self.time_slider.getValue()
        self.time_textbox.setText(f"time: {self.t}")
        self.time_textbox.draw()
        self.batch_id = self.batch_slider.getValue()
        self.batch_textbox.setText(f"batch: {self.batch_id}")
        self.batch_textbox.draw()

        var_keys = sorted(list(self.varying_params.keys()))
        for i in range(len(self.varying_sliders)):
            slider = self.varying_sliders[i]
            tbox = self.varying_textboxes[i]
            dimnum = i
            indexalongdim = slider.getValue()
            self.varying_dimensions[dimnum] = indexalongdim
            corresp_key = var_keys[i]
            corresp_value = self.varying_params[corresp_key][indexalongdim]
            tbox.setText(f"{corresp_key}: {corresp_value}")
            tbox.draw()

        if not self.is_paused:
            if self.t < self.T - 1:
                self.t += 1
                self.time_slider.setValue(self.t)
                self.time_textbox.setText(f"time: {self.t}")
                self.time_textbox.draw()
            else:
                self.is_paused = True
                self.run_button.inactiveColour = colors.GREY

        self.update_frame_data()
        self.draw_walls()
        pygame.display.flip()

    def update_frame_data(self):
        """updating the data that needs to be visualized"""
        index = [self.varying_dimensions[k] for k in sorted(list(self.varying_dimensions.keys()))]
        index = (self.batch_id,) + tuple(index)

        posx = self.posx[index][:, self.t]
        posy = self.posy[index][:, self.t]
        orientation = self.orientation[index][:, self.t]
        mode = self.agmodes[index][:, self.t]
        radius = self.env["RADIUS_AGENT"]

        res_posx = self.res_pos_x[index][:, self.t]
        res_posy = self.res_pos_y[index][:, self.t]
        res_radius = self.env["RADIUS_RESOURCE"]
        self.draw_resources(res_posx, res_posy, res_radius)
        self.draw_agents(posx, posy, orientation, mode, radius)

    def draw_resources(self, posx, posy, radius):
        """Drawing agents in arena according to data"""
        num_resources = len(posx)
        for ri in range(num_resources):
            if posx[ri] != -1 and posy[ri] != -1:
                self.draw_res_patch(posx[ri], posy[ri], radius)

    def draw_res_patch(self, posx, posy, radius):
        """Drawing a single resource patch"""
        image = pygame.Surface([radius * 2, radius * 2])
        image.fill(colors.BACKGROUND)
        image.set_colorkey(colors.BACKGROUND)
        pygame.draw.circle(
            image, colors.GREY, (radius, radius), radius
        )

        self.screen.blit(image, (posx, posy))

    def draw_agents(self, posx, posy, orientation, mode, radius):
        """Drawing agents in arena according to data"""
        num_agents = len(posx)
        for ai in range(num_agents):
            self.draw_agent(posx[ai], posy[ai], orientation[ai], mode[ai], radius)

    def mode_to_color(self, mode):
        """transforming mode code to RGB color for visualization"""
        if mode == 0:
            return colors.BLUE
        elif mode == 1:
            return colors.GREEN
        elif mode == 2:
            return colors.PURPLE
        elif mode == 3:
            return colors.RED

    def draw_agent(self, posx, posy, orientation, mode, radius):
        """Drawing a single agent according to position and orientation"""
        image = pygame.Surface([radius * 2, radius * 2])
        image.fill(colors.BACKGROUND)
        image.set_colorkey(colors.BACKGROUND)
        agent_color = self.mode_to_color(mode)
        pygame.draw.circle(
            image, agent_color, (radius, radius), radius
        )

        # Showing agent orientation with a line towards agent orientation
        pygame.draw.line(image, colors.BACKGROUND, (radius, radius),
                         ((1 + np.cos(orientation)) * radius, (1 - np.sin(orientation)) * radius), 3)
        self.screen.blit(image, (posx, posy))

    def interact_with_event(self, event):
        """Carry out functionality according to user's interaction"""
        # Exit if requested
        if event.type == pygame.QUIT:
            sys.exit()
        #
        # # Change orientation with mouse wheel
        # if event.type == pygame.MOUSEWHEEL:
        #     if event.y == -1:
        #         event.y = 0
        #     for ag in self.agents:
        #         ag.move_with_mouse(pygame.mouse.get_pos(), event.y, 1 - event.y)
        #
        # # Pause on Space
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            print("Space pressed, quitting!")
            self.quit_term = True
        #
        # # Speed up on s and down on f. reset default framerate with d
        # if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
        #     self.framerate -= 1
        #     if self.framerate < 1:
        #         self.framerate = 1
        # if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
        #     self.framerate += 1
        #     if self.framerate > 35:
        #         self.framerate = 35
        # if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
        #     self.framerate = self.framerate_orig
        #
        # # Continuous mouse events (move with cursor)
        # if pygame.mouse.get_pressed()[0]:
        #     try:
        #         for ag in self.agents:
        #             ag.move_with_mouse(event.pos, 0, 0)
        #     except AttributeError:
        #         for ag in self.agents:
        #             ag.move_with_mouse(pygame.mouse.get_pos(), 0, 0)
        # else:
        #     for ag in self.agents:
        #         ag.is_moved_with_cursor = False

    def start(self):

        while not self.quit_term:
            events = pygame.event.get()
            for event in events:
                # Carry out interaction according to user activity
                self.interact_with_event(event)

            self.draw_frame(events)
            self.clock.tick(self.framerate)

        pygame.quit()
