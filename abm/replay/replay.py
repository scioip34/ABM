import json
import sys

from abm.loader.data_loader import ExperimentLoader
from abm.contrib import colors
from abm.monitoring.ifdb import pad_to_n_digits
from pygame_widgets.slider import Slider
from pygame_widgets.button import Button
from pygame_widgets.textbox import TextBox
from pygame_widgets.dropdown import Dropdown
import pygame_widgets
import pygame
import numpy as np
import zarr
import os
from abm.contrib import playgroundtool as pgt
import shutil
import cv2
from matplotlib import cm as colmaps


root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

class ExperimentReplay:
    def __init__(self, data_folder_path, undersample=1, t_start=None, t_end=None, collapse=None):
        """Initialization method to replay recorded simulations from their summary folder. If a summary is not yet
        available for the experiment it will be summarized first
        the undersample parameter only matters if the data is not yet summarized, otherwise it is automatically
        read from the env file"""
        self.image_save_path = os.path.join(root_abm_dir, pgt.VIDEO_SAVE_DIR) # path from collect screenshots
        if os.path.isdir(self.image_save_path):
            shutil.rmtree(self.image_save_path)
        os.makedirs(self.image_save_path, exist_ok=True)
        self.from_script = False  # can be set to True for batch execution of plotting functions
        self.show_vfield = False
        self.experiment = ExperimentLoader(data_folder_path, enforce_summary=False, with_plotting=False,
                                           undersample=undersample, collapse_plot=collapse, t_start=t_start,
                                           t_end=t_end)
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
        self.project_version = self.experiment.project_version

        self.connected_params = self.env.get('SUMMARY_CONNECTED_PARAMS')
        if self.connected_params is None:
            self.connected_params = []

        self.posx_z = self.experiment.agent_summary['posx']
        self.posy_z = self.experiment.agent_summary['posy']
        self.orientation_z = self.experiment.agent_summary['orientation']
        self.agmodes_z = self.experiment.agent_summary['mode']
        self.res_pos_x_z = self.experiment.res_summary['posx']
        self.res_pos_y_z = self.experiment.res_summary['posy']

        if self.project_version in ["Base", "MADRLForaging"]:
            self.coll_resc_z = self.experiment.agent_summary['collresource']
            self.resc_left_z = self.experiment.res_summary['resc_left']
            self.resc_quality_z = self.experiment.res_summary.get('quality', None)
        elif self.project_version=="CooperativeSignaling":
            self.meter_z = self.experiment.agent_summary['meter']
            self.sig_z = self.experiment.agent_summary['signalling']
            self.coll_resc_z = self.experiment.agent_summary['collresource']

        self.varying_params = self.experiment.varying_params
        self.index_prev = None
        self.index = None
        self.t_slice = 0

        self.is_paused = True
        self.show_stats = False
        self.show_paths = False
        self.path_length = 450
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
        if slider_max_val == 1:
            self.batch_slider.disable()
            self.batch_slider.colour = (250, 250, 250)
            self.batch_slider.handleColour = (200, 200, 200)
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
        tb_start = self.button_start_x_2 + int(self.slider_width / 2)
        small_textb_width = int(self.slider_width/2)
        self.path_l_desc_tb = TextBox(self.screen, tb_start, button_start_y, small_textb_width,
                                 self.button_height/2, fontSize=int(self.button_height / 3), borderThickness=1)
        self.path_l_desc_tb.setText(f"path len.:")
        self.path_l_tb = TextBox(self.screen, tb_start, button_start_y + self.button_height/2, small_textb_width,
                                self.button_height/2, fontSize=int(self.button_height / 3), borderThickness=1)


        button_start_y += self.button_height
        self.plot_iid_b = Button(
            # Mandatory Parameters
            self.screen,  # Surface to place button on
            self.slider_start_x,  # X-coordinate of top left corner
            button_start_y,  # Y-coordinate of top left corner
            int(self.slider_width / 2),  # Width
            self.button_height,  # Height

            # Optional Parameters
            text='Plot I.I.D.',  # Text to display
            fontSize=20,  # Size of font
            margin=20,  # Minimum distance between text/image and edge of button
            inactiveColour=colors.GREY,
            onClick=lambda: self.on_print_iid(),  # Function to call when clicked on
            borderThickness=1
        )
        self.snapshot_b = Button(
            # Mandatory Parameters
            self.screen,  # Surface to place button on
            self.button_start_x_2,  # X-coordinate of top left corner
            button_start_y,  # Y-coordinate of top left corner
            int(self.slider_width / 2),  # Width
            self.button_height,  # Height

            # Optional Parameters
            text='Take Snapshot',  # Text to display
            fontSize=20,  # Size of font
            margin=20,  # Minimum distance between text/image and edge of button
            inactiveColour=colors.GREY,
            onClick=lambda: self.take_snapshot(),  # Function to call when clicked on
            borderThickness=1
        )
        self.NNdist_b = Button(
            # Mandatory Parameters
            self.screen,  # Surface to place button on
            self.button_start_x_2 + int(self.slider_width / 2),  # X-coordinate of top left corner
            button_start_y,  # Y-coordinate of top left corner
            int(self.slider_width / 2),  # Width
            self.button_height,  # Height

            # Optional Parameters
            text='Avg. NN Dist.',  # Text to display
            fontSize=20,  # Size of font
            margin=20,  # Minimum distance between text/image and edge of button
            inactiveColour=colors.GREY,
            onClick=lambda: self.on_print_NNdist(),  # Function to call when clicked on
            borderThickness=1
        )
        if self.project_version in ["Base", "CooperativeSignaling", "MADRLForaging"]:
            self.NNdist_b.disable()
            self.NNdist_b.hide()


        # Plotting Button Line
        button_start_y += self.button_height
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

        if self.experiment.env.get("APP_VERSION") == "VisualFlocking":
            self.plot_efficiency.disable()
            self.plot_efficiency.hide()
            self.plot_polar = Button(
                # Mandatory Parameters
                self.screen,  # Surface to place button on
                self.slider_start_x,  # X-coordinate of top left corner
                button_start_y,  # Y-coordinate of top left corner
                int(self.slider_width / 2),  # Width
                self.button_height,  # Height

                # Optional Parameters
                text='Plot Polarization',  # Text to display
                fontSize=20,  # Size of font
                margin=20,  # Minimum distance between text/image and edge of button
                inactiveColour=colors.GREY,
                onClick=lambda: self.on_print_polarization(),  # Function to call when clicked on
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
        if self.experiment.env.get("APP_VERSION") == "VisualFlocking":
            self.plot_rel_time.disable()
            self.plot_rel_time.hide()
            self.plot_coll = Button(
                # Mandatory Parameters
                self.screen,  # Surface to place button on
                self.button_start_x_2,  # X-coordinate of top left corner
                button_start_y,  # Y-coordinate of top left corner
                int(self.slider_width / 2),  # Width
                self.button_height,  # Height

                # Optional Parameters
                text='Plot Collisions',  # Text to display
                fontSize=20,  # Size of font
                margin=20,  # Minimum distance between text/image and edge of button
                inactiveColour=colors.GREY,
                onClick=lambda: self.on_print_collision_times(),  # Function to call when clicked on
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

        # Connecting variables Line
        button_start_y += 2 * self.button_height
        self.connect_vars_b = Button(
            # Mandatory Parameters
            self.screen,  # Surface to place button on
            self.slider_start_x,  # X-coordinate of top left corner
            button_start_y,  # Y-coordinate of top left corner
            int(self.slider_width / 2),  # Width
            self.button_height,  # Height

            # Optional Parameters
            text='Connect Vars',  # Text to display
            fontSize=20,  # Size of font
            margin=20,  # Minimum distance between text/image and edge of button
            inactiveColour=colors.GREY,
            onClick=lambda: self.on_connect_vars(),  # Function to call when clicked on
            borderThickness=1
        )
        small_textb_width = int(self.slider_width / 4)
        self.cvar1_tb = TextBox(self.screen, self.button_start_x_2, button_start_y, small_textb_width,
                                self.button_height, fontSize=int(self.button_height / 2), borderThickness=1)
        startx = self.button_start_x_2 + small_textb_width
        self.cvar2_tb = TextBox(self.screen, startx, button_start_y, small_textb_width,
                                self.button_height, fontSize=int(self.button_height / 2), borderThickness=1)
        startx += 5 + small_textb_width
        self.connect_alias_tb = TextBox(self.screen, startx, button_start_y, int(self.slider_width / 2) - 5,
                                        self.button_height, fontSize=int(self.button_height / 2), borderThickness=1)

    def take_snapshot(self):
        """Taking a single picture of the current status of the replay into an image"""
        filename = f"{pad_to_n_digits(self.t, n=6)}.png"
        path = os.path.join(self.image_save_path, filename)
        events = pygame.event.get()
        self.draw_frame(events)
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

    def on_connect_vars(self):
        """Connected parameters when they change together in dataset, by that we can create new calculated axes"""
        v1 = int(float(self.cvar1_tb.getText()))
        if v1 > len(self.varying_params):
            print(f"The selected parameter index {v1} for variable 1 is too large. The indexing starts at 0.")
            return
        v2 = int(float(self.cvar2_tb.getText()))
        if v2 > len(self.varying_params):
            print(f"The selected parameter index {v2} for variable 2 is too large. The indexing starts at 0.")
            return
        pair_already_connected = False
        target_pair = None
        for pair in self.connected_params:
            if (abs(pair[0]) == abs(v1) and abs(pair[1]) == abs(v2)) or \
                    ((abs(pair[1]) == abs(v1) and abs(pair[0]) == abs(v2))):
                pair_already_connected = True
                target_pair = pair
        if not pair_already_connected:
            alias_text = self.connect_alias_tb.getText()
            if alias_text == "":
                print(f"Please provide an alias text for the new composite variable!")
                return
            self.connected_params.append([v1, v2, alias_text])
        else:
            print(f"Pair with alias {target_pair[2]} was already connected. Deleting from connected pairs!")
            self.connected_params.remove(target_pair)
        self.env["SUMMARY_CONNECTED_PARAMS"] = self.connected_params
        if not pair_already_connected:
            print(
                f"Connecting variables: [{sorted(list(self.varying_params.keys()))[v1]} x {sorted(list(self.varying_params.keys()))[v2]}] = {alias_text}")
        self.save_env()

    def save_env(self):
        env_path = os.path.join(self.experiment.experiment_path, "summary", "fixed_env.json")
        with open(env_path, "w") as envf:
            json.dump(self.env, envf, indent=4)
        print("Env file saved!")

    def load_data_to_ram(self):
        cs = self.experiment.chunksize
        self.t_slice = int(self.t_slice)
        self.posx = self.posx_z[self.index][:, (self.t_slice) * cs:(self.t_slice + 1) * cs]
        self.posy = self.posy_z[self.index][:, (self.t_slice) * cs:(self.t_slice + 1) * cs]
        self.orientation = self.orientation_z[self.index][:, (self.t_slice) * cs:(self.t_slice + 1) * cs]
        self.agmodes = self.agmodes_z[self.index][:, (self.t_slice) * cs:(self.t_slice + 1) * cs]
        self.radius = self.env["RADIUS_AGENT"]
        self.res_pos_x = self.res_pos_x_z[self.index][:, (self.t_slice) * cs:(self.t_slice + 1) * cs]
        self.res_pos_y = self.res_pos_y_z[self.index][:, (self.t_slice) * cs:(self.t_slice + 1) * cs]
        if self.project_version in ["Base", "MADRLForaging"]:
            self.coll_resc = self.coll_resc_z[self.index][:, (self.t_slice) * cs:(self.t_slice + 1) * cs]
            self.resc_left = self.resc_left_z[self.index][:, (self.t_slice) * cs:(self.t_slice + 1) * cs]
            if self.resc_quality_z is not None:
                self.resc_quality = self.resc_quality_z[self.index][:, (self.t_slice) * cs:(self.t_slice + 1) * cs]
            else:
                self.resc_quality = None
        elif self.project_version=="CooperativeSignaling":
            self.meter = self.meter_z[self.index][:, (self.t_slice) * cs:(self.t_slice + 1) * cs]
            self.sig = self.sig_z[self.index][:, (self.t_slice) * cs:(self.t_slice + 1) * cs]
            self.coll_resc = self.coll_resc_z[self.index][:, (self.t_slice) * cs:(self.t_slice + 1) * cs]

    def on_print_reloc_time(self):
        """print mean relative relocation time"""
        if len(list(self.experiment.varying_params.keys())) in [3, 4]:
            self.experiment.set_collapse_param(self.collapse_dropdown.getSelected())
        self.experiment.plot_mean_relocation_time()

    def on_print_collision_times(self, with_read_collapse_param=True, used_batches=None):
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
        if self.T > 1000:
            undersample = int(self.T / 1000)
            print(f"Experiment longer than 1000 timesteps! To calculate iid reducing timesteps to 1000 with undersampling rate {undersample}.")
        else:
            undersample = 1
        fig, ax, cbar = self.experiment.plot_mean_collision_time(t_start=t_start, t_end=t_end,
                                                                 from_script=self.from_script,
                                                                 used_batches=used_batches,
                                                                 undersample=undersample)
        return fig, ax, cbar

    def on_print_polarization(self, with_read_collapse_param=True, used_batches=None):
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
        fig, ax, cbar = self.experiment.plot_mean_polarization(t_start=t_start, t_end=t_end,
                                                               from_script=self.from_script,
                                                               used_batches=used_batches)
        return fig, ax, cbar

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
        fig, ax, cbar = self.experiment.plot_search_efficiency(t_start=t_start, t_end=t_end,
                                                               from_script=self.from_script,
                                                               used_batches=used_batches)
        return fig, ax, cbar

    def on_print_iid(self, with_read_collapse_param=True, used_batches=None):
        """print mean inter-individual distance"""
        if with_read_collapse_param:
            if len(list(self.experiment.varying_params.keys())) in [3, 4]:
                self.experiment.set_collapse_param(self.collapse_dropdown.getSelected())
        if self.T > 1000:
            undersample = int(self.T / 1000)
            print(f"Experiment longer than 1000 timesteps! To calculate iid reducing timesteps to 1000 with undersampling rate {undersample}.")
        else:
            undersample = 1
        fig, ax, cbar = self.experiment.plot_mean_iid(from_script=self.from_script, undersample=undersample)
        return fig, ax, cbar

    def on_print_NNdist(self, with_read_collapse_param=True, used_batches=None):
        """print mean inter-individual distance"""
        if with_read_collapse_param:
            if len(list(self.experiment.varying_params.keys())) in [3, 4]:
                self.experiment.set_collapse_param(self.collapse_dropdown.getSelected())
        if self.T > 1000:
            undersample = int(self.T / 1000)
            print(
                f"Experiment longer than 1000 timesteps! To calculate iid reducing timesteps to 1000 with undersampling rate {undersample}.")
        else:
            undersample = 1
        fig, ax, cbar = self.experiment.plot_mean_NN_dist(from_script=self.from_script, undersample=undersample)
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
        path_l_text = self.path_l_tb.getText()
        if path_l_text is not None:
            try:
                self.path_length = int(path_l_text)
            except:
                pass
        else:
            pass
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
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.window_pad, self.window_pad],
                         [self.window_pad, self.window_pad + self.HEIGHT], width=2)
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.window_pad, self.window_pad],
                         [self.window_pad + self.WIDTH, self.window_pad], width=2)
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.window_pad + self.WIDTH, self.window_pad],
                         [self.window_pad + self.WIDTH, self.window_pad + self.HEIGHT], width=2)
        pygame.draw.line(self.screen, colors.BLACK,
                         [self.window_pad, self.window_pad + self.HEIGHT],
                         [self.window_pad + self.WIDTH, self.window_pad + self.HEIGHT], width=2)

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
            self.varying_sliders[i].enable()
            self.varying_sliders[i].colour = (200, 200, 200)
            self.varying_sliders[i].handleColour = (0, 0, 0)
            slider = self.varying_sliders[i]
            tbox = self.varying_textboxes[i]
            dimnum = i
            indexalongdim = slider.getValue()
            self.varying_dimensions[dimnum] = indexalongdim
            corresp_key = var_keys[i]
            corresp_value = self.varying_params[corresp_key][indexalongdim]
            tbox.setText(f"{corresp_key}: {corresp_value}")
            tbox.draw()
            # Checking if the variable of the slider is connected with something
            # print("------checking pairs for ", i)
            for pair in self.connected_params:
                if i == abs(pair[0]):
                    j = pair[1]
                    # print(f"found pair for {i} as {j}")
                    if pair[0] < 0 or pair[1] < 0:
                        self.varying_sliders[j].setValue(
                            len(self.varying_params[corresp_key]) - 1 - self.varying_sliders[i].getValue())
                    else:
                        self.varying_sliders[j].setValue(self.varying_sliders[i].getValue())
                    self.varying_sliders[j].disable()
                    self.varying_sliders[j].colour = (250, 250, 250)
                    self.varying_sliders[j].handleColour = (200, 200, 200)
                elif i == abs(pair[1]):
                    # print(f"{i} is controlled by {j}, disabling {i}")
                    self.varying_sliders[i].disable()
                    self.varying_sliders[i].colour = (250, 250, 250)
                    self.varying_sliders[i].handleColour = (200, 200, 200)
                else:
                    continue

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
        self.index = (self.batch_id,) + tuple(index)
        update_dataframes = True
        if self.index_prev is not None:
            if self.index_prev == self.index:
                t_slice = np.floor(self.t / self.experiment.chunksize)
                if t_slice == self.t_slice:
                    update_dataframes = False
                else:
                    self.t_slice = t_slice
                    update_dataframes = True
            else:
                update_dataframes = True

        self.index_prev = self.index

        if update_dataframes:
            print("Loading next data chunk...")
            self.load_data_to_ram()

        t_ind = int(self.t - (self.t_slice * self.experiment.chunksize))

        posx = self.posx[:, t_ind]
        posy = self.posy[:, t_ind]
        orientation = self.orientation[:, t_ind]
        mode = self.agmodes[:, t_ind]
        radius = self.env["RADIUS_AGENT"]

        res_posx = self.res_pos_x[:, t_ind]
        res_posy = self.res_pos_y[:, t_ind]

        if self.project_version in ["Base", "CooperativeSignaling", "MADRLForaging"]:
            coll_resc = self.coll_resc[:, t_ind]
            if self.project_version in [ "Base", "MADRLForaging"]:
                if self.resc_quality is not None:
                    resc_quality = self.resc_quality[:, t_ind]
                else:
                    resc_quality = [0 for i in range(len(res_posx))]
                resc_left = self.resc_left[:, t_ind]
            else:
                resc_quality = None
                resc_left = None
        elif self.project_version == "VisualFlocking":
            coll_resc = [0 for i in range(len(posx))]
            resc_left = [0 for i in range(len(posx))]
            resc_quality = [0 for i in range(len(posx))]


        res_unit = self.env["MIN_RESOURCE_PER_PATCH"]
        if res_unit == "----TUNED----":
            var_keys = sorted(list(self.varying_params.keys()))
            dimnum = var_keys.index("MIN_RESOURCE_PER_PATCH")
            slider = self.varying_sliders[dimnum]
            indexalongdim = slider.getValue()
            res_unit = self.varying_params["MIN_RESOURCE_PER_PATCH"][indexalongdim]

        max_num_res = self.env["N_RESOURCES"]
        if max_num_res == "----TUNED----":
            max_num_res = max(self.varying_params["N_RESOURCES"])

        max_units = [res_unit for _ in range(int(max_num_res))]

        if self.project_version == "CooperativeSignaling":
            radius_keyword = "DETECTION_RANGE"
        else:
            radius_keyword = "RADIUS_RESOURCE"
        res_radius = self.env[radius_keyword]
        # if not update_dataframes:
        if res_radius == "----TUNED----":
            # in case the patch size was changed during the simulations we
            # read the radius from the corresponding slider
            var_keys = sorted(list(self.varying_params.keys()))
            dimnum = var_keys.index(radius_keyword)
            slider = self.varying_sliders[dimnum]
            indexalongdim = slider.getValue()
            res_radius = self.varying_params[radius_keyword][indexalongdim]

        # getting FOV if it is varying
        if self.env["AGENT_FOV"] == "----TUNED----":
            var_keys = sorted(list(self.varying_params.keys()))
            dimnum = var_keys.index("AGENT_FOV")
            slider = self.varying_sliders[dimnum]
            indexalongdim = slider.getValue()
            self.current_fov = float(self.varying_params["AGENT_FOV"][indexalongdim])
        else:
            self.current_fov = float(self.env["AGENT_FOV"])


        if self.project_version in ["Base", "MADRLForaging"]:
            self.draw_resources(res_posx, res_posy, max_units, resc_left, resc_quality, res_radius)
        elif self.project_version=="CooperativeSignaling":
            self.draw_resources(res_posx, res_posy, [1], [1], [1], res_radius)
        if self.show_paths:
            if self.experiment.env.get("APP_VERSION", "") == "VisualFlocking":
                self.draw_agent_paths_vf(self.posx[:, max(0, t_ind - self.path_length):t_ind],
                                      self.posy[:, max(0, t_ind - self.path_length):t_ind],
                                      radius,
                                      self.orientation[:, max(0, t_ind - self.path_length):t_ind])
            else:
                self.draw_agent_paths(self.posx[:, max(0, t_ind - self.path_length):t_ind],
                                      self.posy[:, max(0, t_ind - self.path_length):t_ind],
                                      radius,
                                      modes=self.agmodes[:, max(0, t_ind - self.path_length):t_ind])
        if self.experiment.clustering_data is not None:
            t_ind_cl = int(t_ind / 25)
            clusters_idx = tuple(list(self.index) + [slice(None), t_ind_cl])
            clusters = self.experiment.clustering_ids[clusters_idx]
            print(clusters)
        else:
            clusters = None


        if self.project_version in ["Base", "MADRLForaging"]:
            self.draw_agents(posx, posy, orientation, mode, coll_resc, radius)
        elif self.project_version == "CooperativeSignaling":
            self.draw_agents(posx, posy, orientation, mode, [0 for i in range(len(posx))], radius)
        elif self.project_version == "VisualFlocking":
            self.draw_agents(posx, posy, orientation, mode, [0 for i in range(len(posx))], radius, clusters=clusters)

        num_agents = len(posx)
        if self.show_stats:
            time_dep_stats = []
            time_dep_stats.append("HISTORY SUMMARY:")
            mode_til_now = self.agmodes[:, 0:t_ind]
            mean_reloc = np.mean(np.mean((mode_til_now == 2).astype(int), axis=-1))
            if np.isnan(mean_reloc):
                mean_reloc = 0
            std_reloc = np.std(np.mean((mode_til_now == 2).astype(int), axis=-1))
            if np.isnan(std_reloc):
                std_reloc = 0
            time_dep_stats.append(f"Relocation Time (0-t): Mean:{mean_reloc:10.2f} ± {std_reloc:10.2f}")
            if self.project_version in ["Base","MADRLForaging", "CooperativeSignaling"]:
                end_pos = self.draw_agent_stat_summary([ai for ai in range(num_agents)], posx, posy, orientation, mode,
                                                       coll_resc, previous_metrics=time_dep_stats)
            elif self.project_version == "VisualFlocking":
                end_pos = self.draw_agent_stat_summary([ai for ai in range(num_agents)], posx, posy, orientation, mode,
                                                       [0 for i in range(len(posx))], previous_metrics=time_dep_stats)
            if self.project_version in ["Base","MADRLForaging"]:
                self.draw_resource_stat_summary(posx, posy, max_units, resc_left, resc_quality, end_pos)

    def draw_agent_paths_vf(self, posx, posy, radius, orientations):
        num_agents = posx.shape[0]
        path_length = posx.shape[1]
        cmap = colmaps.get_cmap('jet')
        transparency = 0.5
        transparency = int(transparency * 255)
        big_colors = cmap(orientations/(2 * np.pi))*255
        # setting alpha
        surface = pygame.Surface((self.WIDTH+self.window_pad, self.HEIGHT+self.window_pad))
        surface.fill(colors.BACKGROUND)
        surface.set_colorkey(colors.WHITE)
        surface.set_alpha(255)
        try:
            for ai in range(num_agents):
                subsurface = pygame.Surface((self.WIDTH+self.window_pad, self.HEIGHT+self.window_pad))
                subsurface.fill(colors.BACKGROUND)
                subsurface.set_colorkey(colors.WHITE)
                subsurface.set_alpha(transparency)
                for t in range(1, path_length, 1):
                    #point1 = (posx[ai, t - 1] + radius, posy[ai, t - 1] + radius)
                    point2 = (posx[ai, t] + radius, posy[ai, t] + radius)
                    color = big_colors[ai, t]
                    # pygame.draw.line(surface1, color, point1, point2, 4)
                    pygame.draw.circle(subsurface, color, point2, int(self.radius/2))
                surface.blit(subsurface, (0, 0))
            self.screen.blit(surface, (0, 0))
        except IndexError as e:
            pass

    def draw_agent_paths(self, posx, posy, radius, modes=None):
        path_cols = [colors.BLUE, colors.GREEN, colors.PURPLE, colors.RED]
        num_agents = posx.shape[0]
        path_length = posx.shape[1]
        try:
            for ai in range(num_agents):
                for t in range(1, path_length, 1):
                    point1 = (posx[ai, t - 1] + radius, posy[ai, t - 1] + radius)
                    point2 = (posx[ai, t] + radius, posy[ai, t] + radius)
                    color = path_cols[int(modes[ai, t])]
                    #pygame.draw.line(self.screen, color, point1, point2, 4)
                    pygame.draw.circle(self.screen, color, point2, 5)
                # for t in range(1, path_length):
                #     point1 = (posx[ai, t - 1] + radius, posy[ai, t - 1] + radius)
                #     point2 = (posx[ai, t] + radius, posy[ai, t] + radius)
                #     pygame.gfxdraw.pixel(self.screen, int(point1[0]), int(point1[1]), colors.BLACK)
                #     pygame.gfxdraw.pixel(self.screen, int(point2[0]), int(point2[1]), colors.BLACK)
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
        if isinstance(radius, str):
            radius = int(float(radius))
        image = pygame.Surface([radius * 2, radius * 2])
        image.fill(colors.BACKGROUND)
        image.set_colorkey(colors.BACKGROUND)
        pygame.draw.circle(
            image, colors.GREY, (radius, radius), radius
        )
        new_radius = int((resc_left / max_unit) * radius)
        # print(f"R={radius}, RN={new_radius}, Remres={resc_left}, maxu={max_unit}")
        pygame.draw.circle(
            image, colors.DARK_GREY, (radius, radius), new_radius
        )
        self.screen.blit(image, (posx, posy))
        if self.show_stats:
            font = pygame.font.Font(None, 16)
            text = font.render(f"RID: {id}", True, colors.BLACK)
            tbsize = text.get_size()
            self.screen.blit(text, (int(posx + radius - tbsize[0] / 2), int(posy + radius - tbsize[1] / 2)))

    def draw_agents(self, posx, posy, orientation, mode, coll_resc, radius, clusters=None):
        """Drawing agents in arena according to data"""
        if self.experiment.env.get("APP_VERSION") == "VisualFlocking":
            cmap = colmaps.get_cmap('jet')
            if clusters is None:
                colors = [np.array(cmap(ori / (2 * np.pi)))[0:-1] for ori in orientation]
                # rescaling color for pygame
                for col in colors:
                    col[0:3] *= 255
            else:
                # defining colors according to cluster id_s in the clusters array
                colors = [np.array(cmap(cl / (np.max(clusters))))[0:-1] for cl in clusters]
                # making the colors brighter
                colors = [col * 0.5 + 0.5 for col in colors]
                # rescaling color for pygame
                for col in colors:
                    col[0:3] *= 255
        else:
            colors = [self.mode_to_color(m) for m in mode]

        num_agents = len(posx)
        for ai in range(num_agents):
            cid = clusters[ai] if clusters is not None else None
            self.draw_agent(ai, posx[ai], posy[ai], orientation[ai], colors[ai], coll_resc[ai], radius, cluster_id=cid)
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
        elif mode == 4:
            if not to_text:
                return colors.RED
            else:
                return "Crowd"

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
        iid = self.experiment.calculate_interindividual_distance_slice(posx, posy)
        status.append(f"mean IID: {np.mean(iid[iid != 0]) / 2:10.2f} ± {np.std(iid[iid != 0]):10.2f}")

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
        FOV_rat = self.current_fov
        FOV = (-FOV_rat * np.pi, FOV_rat * np.pi)

        # Show limits of FOV
        if 0 < FOV[1] < np.pi:

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

    def draw_agent(self, id, posx, posy, orientation, agent_color, coll_resc, radius, cluster_id=None):
        """Drawing a single agent according to position and orientation"""
        # todo: add magnify agent checkbox
        radius = int(radius)
        image = pygame.Surface([radius * 2, radius * 2])
        image.fill(colors.BACKGROUND)
        image.set_colorkey(colors.BACKGROUND)
        pygame.gfxdraw.filled_circle(image, radius, radius, radius-1, agent_color)
        pygame.gfxdraw.aacircle(image, radius, radius, radius - 1, colors.BLACK)

        # Showing agent orientation with a line towards agent orientation
        pygame.draw.line(image, colors.GREY, (radius, radius),
                         ((1 + np.cos(orientation)) * radius, (1 - np.sin(orientation)) * radius), 3)
        self.screen.blit(image, (posx, posy))

        if self.show_stats:
            font = pygame.font.Font(None, 16)
            if self.env.get("APP_VERSION", "Base") in ["Base", "CooperativeSignaling", "MADRLForaging"]:
                text = font.render(f"ID:{id}, R:{coll_resc:.2f}", True, colors.BLACK)
            elif self.experiment.env.get("APP_VERSION") == "VisualFlocking":
                if cluster_id is None:
                    text = font.render(f"ID:{id}, ori:{orientation:.2f}", True, colors.BLACK)
                else:
                    text = font.render(f"ID:{id}, ori:{orientation:.2f}, cl:{cluster_id}", True, colors.BLACK)
            self.screen.blit(text, (posx - 10, posy - 10))

    def interact_with_event(self, event):
        """Carry out functionality according to user's interaction"""
        # Exit if requested
        if event.type == pygame.QUIT:
            sys.exit()
            pygame.quit()

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
        sys.exit()
