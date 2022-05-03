import sys

from abm.loader.data_loader import ExperimentLoader
from abm.contrib import colors
from pygame_widgets.slider import Slider
from pygame_widgets.button import Button
from pygame_widgets.textbox import TextBox
from pygame_widgets.dropdown import Dropdown
import pygame_widgets
import pygame
import numpy as np


class ExperimentReplay:
    def __init__(self, data_folder_path, undersample=1, t_start=None, t_end=None, collapse=None):
        """Initialization method to replay recorded simulations from their summary folder. If a summary is not yet
        available for the experiment it will be summarized first
        the undersample parameter only matters if the data is not yet summarized, otherwise it is automatically
        read from the env file"""
        self.from_script = False  # can be set to True for batch execution of plotting functions
        self.show_vfield = False
        self.experiment = ExperimentLoader(data_folder_path, enforce_summary=False, with_plotting=False,
                                           undersample=undersample, collapse_plot=collapse, t_start=t_start, t_end=t_end)
        self.undersample = self.experiment.undersample
        # todo: this initialization will fail when we systematically change width and height in experiment
        self.WIDTH = int(float(self.experiment.env["ENV_WIDTH"]))
        self.HEIGHT = int(float(self.experiment.env["ENV_HEIGHT"]))
        if self.experiment.t_start is None and self.experiment.t_end is None:
            self.T = int(float(self.experiment.env["T"]) / self.undersample)
        else:
            self.T = int((self.experiment.t_end - self.experiment.t_start) / self.undersample)
        self.window_pad = 30
        self.vis_area_end_width = 2 * self.window_pad + self.WIDTH
        self.vis_area_end_height = 2 * self.window_pad + self.HEIGHT
        self.action_area_width = 400
        self.action_area_height = 800
        self.full_width = self.WIDTH + self.action_area_width + 2 * self.window_pad
        self.full_height = self.action_area_height

        self.env = self.experiment.env

        self.posx = self.experiment.agent_summary['posx']
        self.posy = self.experiment.agent_summary['posy']
        self.orientation = self.experiment.agent_summary['orientation']
        self.agmodes = self.experiment.agent_summary['mode']
        self.coll_resc = self.experiment.agent_summary['collresource']

        self.res_pos_x = self.experiment.res_summary['posx']
        self.res_pos_y = self.experiment.res_summary['posy']
        self.resc_left = self.experiment.res_summary['resc_left']
        self.resc_quality = self.experiment.res_summary['quality']

        self.varying_params = self.experiment.varying_params

        self.is_paused = True
        self.show_stats = False
        self.show_paths = False
        self.path_length = 100
        self.t = 0
        self.framerate = 25
        self.num_batches = self.experiment.num_batches
        self.batch_id = 0

        # self.experiment = None

        # Initializing pygame
        self.quit_term = False
        pygame.init()
        self.screen = pygame.display.set_mode([self.full_width, self.full_height], pygame.RESIZABLE)
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
        slider_max_val = self.num_batches - 1
        if slider_max_val <= 0:
            slider_max_val = 1
        slider_start_y = slider_i * (self.slider_height + self.action_area_pad)
        self.batch_slider = Slider(self.screen, self.slider_start_x, slider_start_y, self.slider_width,
                                   self.slider_height, min=0, max=slider_max_val, step=1, initial=0)
        self.batch_textbox = TextBox(self.screen, self.textbox_start_x, slider_start_y, self.textbox_width,
                                     self.slider_height, fontSize=self.slider_height - 2, borderThickness=1)

        slider_i = 4
        self.varying_sliders = []
        self.varying_textboxes = []
        self.varying_dimensions = {}
        vpi = 0
        for k in sorted(list(self.varying_params.keys())):
            v = self.varying_params[k]
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
            onClick=lambda: self.on_run(),  # Function to call when clicked on
            borderThickness=1
        )

        self.button_start_x_2 = self.slider_start_x + int(self.slider_width / 2)
        self.stats_button = Button(
            # Mandatory Parameters
            self.screen,  # Surface to place button on
            self.button_start_x_2,  # X-coordinate of top left corner
            button_start_y,  # Y-coordinate of top left corner
            int(self.slider_width / 2),  # Width
            self.button_height,  # Height

            # Optional Parameters
            text='Show Stats',  # Text to display
            fontSize=20,  # Size of font
            margin=20,  # Minimum distance between text/image and edge of button
            inactiveColour=colors.GREY,
            onClick=lambda: self.on_run_stats(),  # Function to call when clicked on
            borderThickness=1
        )

        button_start_y += self.button_height
        # Creates the button with optional parameters
        self.show_path_button = Button(
            # Mandatory Parameters
            self.screen,  # Surface to place button on
            self.slider_start_x,  # X-coordinate of top left corner
            button_start_y,  # Y-coordinate of top left corner
            int(self.slider_width / 2),  # Width
            self.button_height,  # Height

            # Optional Parameters
            text='Show Path',  # Text to display
            fontSize=20,  # Size of font
            margin=20,  # Minimum distance between text/image and edge of button
            inactiveColour=colors.GREY,
            onClick=lambda: self.on_run_show_path(),  # Function to call when clicked on
            borderThickness=1
        )
        # Creates the button with optional parameters
        self.show_vfield_button = Button(
            # Mandatory Parameters
            self.screen,  # Surface to place button on
            self.button_start_x_2,  # X-coordinate of top left corner
            button_start_y,  # Y-coordinate of top left corner
            int(self.slider_width / 2),  # Width
            self.button_height,  # Height

            # Optional Parameters
            text='Show V.Field',  # Text to display
            fontSize=20,  # Size of font
            margin=20,  # Minimum distance between text/image and edge of button
            inactiveColour=colors.GREY,
            onClick=lambda: self.on_run_show_vfield(),  # Function to call when clicked on
            borderThickness=1
        )

        # Plotting Button Line
        button_start_y += 2*self.button_height
        self.plot_efficiency = Button(
            # Mandatory Parameters
            self.screen,  # Surface to place button on
            self.slider_start_x,  # X-coordinate of top left corner
            button_start_y,  # Y-coordinate of top left corner
            int(self.slider_width / 2),  # Width
            self.button_height,  # Height

            # Optional Parameters
            text='Plot Efficiency',  # Text to display
            fontSize=20,  # Size of font
            margin=20,  # Minimum distance between text/image and edge of button
            inactiveColour=colors.GREY,
            onClick=lambda: self.on_print_efficiency(),  # Function to call when clicked on
            borderThickness=1
        )
        self.plot_rel_time = Button(
            # Mandatory Parameters
            self.screen,  # Surface to place button on
            self.button_start_x_2,  # X-coordinate of top left corner
            button_start_y,  # Y-coordinate of top left corner
            int(self.slider_width / 2),  # Width
            self.button_height,  # Height

            # Optional Parameters
            text='Plot Rel. Time',  # Text to display
            fontSize=20,  # Size of font
            margin=20,  # Minimum distance between text/image and edge of button
            inactiveColour=colors.GREY,
            onClick=lambda: self.on_print_reloc_time(),  # Function to call when clicked on
            borderThickness=1
        )
        if len(list(self.experiment.varying_params.keys())) in [3, 4]:
            self.collapse_dropdown = Dropdown(
                self.screen,
                self.button_start_x_2 + int(self.slider_width / 2),
                button_start_y,
                int(self.slider_width / 2),
                self.button_height,
                name='Collapse Type',
                choices=[
                    'None',
                    'MAX-0',
                    'MAX-1',
                    'MIN-0',
                    'MIN-1'
                ],
                borderRadius=3,
                colour=colors.LIGHT_BLUE,
                values=[
                    None,
                    'MAX-0',
                    'MAX-1',
                    'MIN-0',
                    'MIN-1'], direction='down', textHAlign='centre'
            )

        # Plotting Details Button Line
        button_start_y += self.button_height
        self.t_start = None
        self.t_end = None
        self.t_start_b = Button(
            # Mandatory Parameters
            self.screen,  # Surface to place button on
            self.slider_start_x,  # X-coordinate of top left corner
            button_start_y,  # Y-coordinate of top left corner
            int(self.slider_width / 2),  # Width
            self.button_height,  # Height

            # Optional Parameters
            text='Set Start Time',  # Text to display
            fontSize=20,  # Size of font
            margin=20,  # Minimum distance between text/image and edge of button
            inactiveColour=colors.GREY,
            onClick=lambda: self.on_set_t_start(),  # Function to call when clicked on
            borderThickness=1
        )
        self.t_end_b = Button(
            # Mandatory Parameters
            self.screen,  # Surface to place button on
            self.button_start_x_2,  # X-coordinate of top left corner
            button_start_y,  # Y-coordinate of top left corner
            int(self.slider_width / 2),  # Width
            self.button_height,  # Height

            # Optional Parameters
            text='Set End Time',  # Text to display
            fontSize=20,  # Size of font
            margin=20,  # Minimum distance between text/image and edge of button
            inactiveColour=colors.GREY,
            onClick=lambda: self.on_set_t_end(),  # Function to call when clicked on
            borderThickness=1
        )

    def on_print_reloc_time(self):
        """print mean relative relocation time"""
        if len(list(self.experiment.varying_params.keys())) in [3, 4]:
            self.experiment.set_collapse_param(self.collapse_dropdown.getSelected())
        self.experiment.plot_mean_relocation_time()

    def on_print_efficiency(self, with_read_collapse_param=True, used_batches=None):
        """print mean search efficiency"""
        if with_read_collapse_param:
            if len(list(self.experiment.varying_params.keys())) in [3, 4]:
                self.experiment.set_collapse_param(self.collapse_dropdown.getSelected())
        if self.t_start is None:
            t_start = 0
        else:
            t_start = self.t_start
        if self.t_end is None:
            t_end = self.T - 1
        else:
            t_end = self.t_end
        fig, ax, cbar = self.experiment.plot_search_efficiency(t_start=t_start, t_end=t_end, from_script=self.from_script,
                                                               used_batches=used_batches)
        return fig, ax, cbar

    def on_set_t_start(self):
        """Setting starting timestep for plotting and calculations"""
        if self.t_start is None:
            self.t_start = self.time_slider.getValue()
            if self.t_end is not None:
                t_end = self.t_end
            else:
                t_end = self.T - 1
            if self.t_start >= t_end:
                self.t_start = None
                print("Start time can not be larger than end time!")
            else:
                self.t_start_b.inactiveColour = colors.GREEN
                self.t_start_b.string = f"T start = {self.t_start * self.undersample}"
        else:
            self.t_start = None
            self.t_start_b.inactiveColour = colors.GREY
            self.t_start_b.string = "Set Start Time"

        self.t_start_b.text = self.t_start_b.font.render(self.t_start_b.string, True,
                                                         self.t_start_b.textColour)

    def on_set_t_end(self):
        """Setting starting timestep for plotting and calculations"""
        if self.t_end is None:
            self.t_end = self.time_slider.getValue()
            if self.t_start is not None:
                t_start = self.t_start
            else:
                t_start = 0
            if self.t_end <= t_start:
                self.t_end = None
                print("End time can not be smaller than end time!")
            else:
                self.t_end_b.inactiveColour = colors.GREEN
                self.t_end_b.string = f"T end = {self.t_end * self.undersample}"
        else:
            self.t_end = None
            self.t_end_b.inactiveColour = colors.GREY
            self.t_end_b.string = "Set End Time"

        self.t_end_b.text = self.t_end_b.font.render(self.t_end_b.string, True,
                                                         self.t_end_b.textColour)

    def on_run_show_vfield(self):
        self.show_vfield = not self.show_vfield
        if self.show_vfield:
            self.show_vfield_button.inactiveColour = colors.GREEN
        else:
            self.show_vfield_button.inactiveColour = colors.GREY

    def on_run_show_path(self):
        self.show_paths = not self.show_paths
        if self.show_paths:
            self.show_path_button.inactiveColour = colors.GREEN
        else:
            self.show_path_button.inactiveColour = colors.GREY

    def on_run_stats(self):
        self.show_stats = not self.show_stats
        if self.show_stats:
            self.stats_button.inactiveColour = colors.GREEN
        else:
            self.stats_button.inactiveColour = colors.GREY

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
        coll_resc = self.coll_resc[index][:, self.t]
        radius = self.env["RADIUS_AGENT"]

        res_posx = self.res_pos_x[index][:, self.t]
        res_posy = self.res_pos_y[index][:, self.t]
        resc_left = self.resc_left[index][:, self.t]
        max_units = np.max(self.resc_left[index], axis=1)
        resc_quality = self.resc_quality[index][:, self.t]
        res_radius = self.env["RADIUS_RESOURCE"]
        if res_radius == "----TUNED----":
            # in case the patch size was changed during the simulations we
            # read the radius from the corresponding slider
            var_keys = sorted(list(self.varying_params.keys()))
            dimnum = var_keys.index("RADIUS_RESOURCE")
            slider = self.varying_sliders[dimnum]
            indexalongdim = slider.getValue()
            res_radius = self.varying_params["RADIUS_RESOURCE"][indexalongdim]

        self.draw_resources(res_posx, res_posy, max_units, resc_left, resc_quality, res_radius)
        if self.show_paths:
            self.draw_agent_paths(self.posx[index][:, max(0, self.t - self.path_length):self.t],
                                  self.posy[index][:, max(0, self.t - self.path_length):self.t],
                                  radius)
        self.draw_agents(posx, posy, orientation, mode, coll_resc, radius)

        num_agents = len(posx)
        if self.show_stats:
            time_dep_stats = []
            time_dep_stats.append("HISTORY SUMMARY:")
            mode_til_now = self.agmodes[index][:, 0:self.t]
            mean_reloc = np.mean(np.mean((mode_til_now == 2).astype(int), axis=-1))
            if np.isnan(mean_reloc):
                mean_reloc = 0
            std_reloc = np.std(np.mean((mode_til_now == 2).astype(int), axis=-1))
            if np.isnan(std_reloc):
                std_reloc = 0
            time_dep_stats.append(f"Relocation Time (0-t): Mean:{mean_reloc:10.2f} ± {std_reloc:10.2f}")
            end_pos = self.draw_agent_stat_summary([ai for ai in range(num_agents)], posx, posy, orientation, mode,
                                                   coll_resc, previous_metrics=time_dep_stats)
            self.draw_resource_stat_summary(posx, posy, max_units, resc_left, resc_quality, end_pos)

    def draw_agent_paths(self, posx, posy, radius):
        num_agents = posx.shape[0]
        path_length = posx.shape[1]
        try:
            for ai in range(num_agents):
                for t in range(1, path_length):
                    point1 = (posx[ai, t - 1] + radius, posy[ai, t - 1] + radius)
                    point2 = (posx[ai, t] + radius, posy[ai, t] + radius)
                    pygame.draw.line(self.screen, colors.GREY, point1, point2, 3)
        except IndexError as e:
            pass

    def draw_resources(self, posx, posy, max_units, resc_left, resc_quality, radius):
        """Drawing agents in arena according to data"""
        num_resources = len(posx)
        for ri in range(num_resources):
            if posx[ri] != 0 and posy[ri] != 0:
                self.draw_res_patch(ri, posx[ri], posy[ri], max_units[ri], resc_left[ri], resc_quality[ri], radius)

    def draw_res_patch(self, id, posx, posy, max_unit, resc_left, resc_quality, radius):
        """Drawing a single resource patch"""
        image = pygame.Surface([radius * 2, radius * 2])
        image.fill(colors.BACKGROUND)
        image.set_colorkey(colors.BACKGROUND)
        pygame.draw.circle(
            image, colors.GREY, (radius, radius), radius
        )
        new_radius = int((resc_left / max_unit) * radius)
        pygame.draw.circle(
            image, colors.DARK_GREY, (radius, radius), new_radius
        )
        self.screen.blit(image, (posx, posy))
        if self.show_stats:
            font = pygame.font.Font(None, 16)
            text = font.render(f"RID: {id}", True, colors.BLACK)
            tbsize = text.get_size()
            self.screen.blit(text, (int(posx + radius - tbsize[0] / 2), int(posy + radius - tbsize[1] / 2)))

    def draw_agents(self, posx, posy, orientation, mode, coll_resc, radius):
        """Drawing agents in arena according to data"""
        num_agents = len(posx)
        for ai in range(num_agents):
            self.draw_agent(ai, posx[ai], posy[ai], orientation[ai], mode[ai], coll_resc[ai], radius)
            if self.show_vfield:
                self.draw_vfield(ai, posx[ai], posy[ai], orientation[ai], radius)

    def mode_to_color(self, mode, to_text=False):
        """transforming mode code to RGB color for visualization"""
        if mode == 0:
            if not to_text:
                return colors.BLUE
            else:
                return "Explore"
        elif mode == 1:
            if not to_text:
                return colors.GREEN
            else:
                return "Exploit"
        elif mode == 2:
            if not to_text:
                return colors.PURPLE
            else:
                return "Relocate"
        elif mode == 3:
            if not to_text:
                return colors.RED
            else:
                return "Collide"

    def draw_agent_stat_summary(self, ids, posx, posy, orientation, mode, coll_resc, previous_metrics=None):
        """Showing the summary of agent data for given frame"""
        line_height = 16
        font = pygame.font.Font(None, line_height)
        if previous_metrics is not None:
            status = previous_metrics
            status.append(" ")
        status.append("AGENT SUMMARY:")
        line_count_before = len(status)
        for ai in ids:
            status.append(
                f"ID:{ai:5}, Units:{coll_resc[ai]:10.2f}, Mode:{self.mode_to_color(mode[ai], to_text=True):15}")

        status.append(" ")
        status.append("CALCULATED METRICS (t):")
        status.append(f"Collected resource: Mean:{np.mean(coll_resc):10.2f} ± {np.std(coll_resc):10.2f}")

        for i, stat_i in enumerate(status):
            if i - line_count_before < 0 or i - line_count_before >= len(ids):
                text_color = colors.BLACK
            else:
                text_color = self.mode_to_color(mode[i - line_count_before])
            text = font.render(stat_i, True, text_color)
            self.screen.blit(text, (self.window_pad, self.vis_area_end_height + i * line_height))
            summary_end_position = (self.window_pad, self.vis_area_end_height + i * line_height)

        return summary_end_position

    def draw_resource_stat_summary(self, posx, posy, max_units, resc_left, resc_quality, start_position):
        """Showing the summary of agent data for given frame"""
        line_height = 16
        font = pygame.font.Font(None, line_height)
        status = [" ", "RESOURCE SUMMARY:"]
        line_count_before = len(status)
        ids = [i for i in range(len(max_units))]
        for ri in ids:
            if resc_quality[ri] > 0:
                status.append(f"ID:{ri:5}, Units Left:{resc_left[ri]:10.2f}, Quality:{resc_quality[ri]:10.2f}")

        for i, stat_i in enumerate(status):
            text = font.render(stat_i, True, colors.BLACK)
            self.screen.blit(text, (start_position[0], start_position[1] + (i + 1) * line_height))

    def draw_vfield(self, id, posx, posy, orientation, radius):
        """Drawing a single agent according to position and orientation"""
        FOV_rat = float(self.env.get("AGENT_FOV", 1))
        FOV = (-FOV_rat * np.pi, FOV_rat * np.pi)

        # Show limits of FOV
        if FOV[1] < np.pi:

            # Center and radius of pie chart
            cx, cy, r = posx + radius, posy + radius, self.env.get("VISION_RANGE", 3 * radius)
            if r > self.WIDTH:
                # showing unlimited visual range
                r = 100
                vfield_color = colors.GREEN
            else:
                vfield_color = colors.BLACK

            angle = (2 * FOV[1]) / np.pi * 360
            p = [(cx, cy)]
            # Get points on arc
            angles = [orientation + FOV[0], orientation + FOV[1]]
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
            pygame.draw.polygon(image, vfield_color, p)

            self.screen.blit(image, (0, 0))

    def draw_agent(self, id, posx, posy, orientation, mode, coll_resc, radius):
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

        if self.show_stats:
            font = pygame.font.Font(None, 16)
            text = font.render(f"ID:{id}, R:{coll_resc:.2f}", True, colors.BLACK)
            self.screen.blit(text, (posx - 10, posy - 10))

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
            #self.quit_term = True
            self.experiment.plot_mean_relocation_time()
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
        if event.type == pygame.VIDEORESIZE:
            # There's some code to add back window content here.
            self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

    def start(self):

        while not self.quit_term:
            events = pygame.event.get()
            for event in events:
                # Carry out interaction according to user activity
                self.interact_with_event(event)

            self.draw_frame(events)
            self.clock.tick(self.framerate)

        pygame.quit()
