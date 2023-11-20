import os

import numpy as np
import pygame
import pygame_widgets
from pygame_widgets.button import Button
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox

from abm.contrib import colors
from abm.monitoring.ifdb import pad_to_n_digits
from abm.projects.visual_flocking.vf_simulation.vf_sims import VFSimulation
from abm.simulation.isims import PlaygroundSimulation
from abm.projects.visual_flocking.vf_contrib import vf_params



class VFPlaygroundSimulation(PlaygroundSimulation, VFSimulation):
    """
    SEE: https://docs.python.org/3/tutorial/classes.html#multiple-inheritance
    """

    def __init__(self):
        super().__init__()
        # Hiding unused buttons and sliders
        self.fix_SUM_res_button.hide()
        self.visual_exclusion_button.hide()
        self.IFDB_button.hide()


        # Replacing NRES slider to Visual Resolution Slider
        self.NRES_help.hide()
        self.NRES_textbox.hide()
        self.NRES_slider.hide()
        x = self.NRES_slider.getX()
        y = self.NRES_slider.getY()
        w = self.NRES_slider.getWidth()
        h = self.NRES_slider.getHeight()
        self.VISRES_slider = Slider(self.screen, x, y, w, h, min=20, max=3000, step=10, initial=self.v_field_res)
        x = self.NRES_textbox.getX()
        y = self.NRES_textbox.getY()
        w = self.NRES_textbox.getWidth()
        h = self.NRES_textbox.getHeight()
        self.VISRES_textbox = TextBox(self.screen, x, y, w, h, fontSize=self.textbox_height - 2, borderThickness=1)
        x = self.NRES_help.getX()
        y = self.NRES_help.getY()
        w = self.NRES_help.getWidth()
        h = self.NRES_help.getHeight()
        self.VISRES_help = Button(self.screen, x, y, w, h, text='?', fontSize=self.help_height - 2,
                                  inactiveColour=colors.GREY, borderThickness=1, )
        self.VISRES_help.onClick = lambda: self.show_help('V.Resol.', self.VISRES_help)
        self.VISRES_help.onRelease = lambda: self.unshow_help(self.VISRES_help)

        self.help_buttons.append(self.VISRES_help)
        self.sliders.append(self.VISRES_slider)
        self.slider_texts.append(self.VISRES_textbox)


        # Replacing Res radius slider with agent radius slider
        self.RESradius_slider.hide()
        self.RESradius_textbox.hide()
        self.RES_help.hide()
        x = self.RESradius_slider.getX()
        y = self.RESradius_slider.getY()
        w = self.RESradius_slider.getWidth()
        h = self.RESradius_slider.getHeight()
        self.ARAD_slider = Slider(self.screen, x, y, w, h, min=1, max=30, step=1, initial=self.agent_radii)
        x = self.RESradius_textbox.getX()
        y = self.RESradius_textbox.getY()
        w = self.RESradius_textbox.getWidth()
        h = self.RESradius_textbox.getHeight()
        self.ARAD_textbox = TextBox(self.screen, x, y, w, h, fontSize=self.textbox_height - 2, borderThickness=1)
        x = self.RES_help.getX()
        y = self.RES_help.getY()
        w = self.RES_help.getWidth()
        h = self.RES_help.getHeight()
        self.ARAD_help = Button(self.screen, x, y, w, h, text='?', fontSize=self.help_height - 2,
                                inactiveColour=colors.GREY, borderThickness=1, )
        self.ARAD_help.onClick = lambda: self.show_help('R ag.', self.VISRES_help)
        self.ARAD_help.onRelease = lambda: self.unshow_help(self.VISRES_help)

        self.help_buttons.append(self.ARAD_help)
        self.sliders.append(self.ARAD_slider)
        self.slider_texts.append(self.ARAD_textbox)


        # Replacing Ew slider with Alpha0
        self.Epsw_slider.hide()
        self.Epsw_textbox.hide()
        self.Epsw_help.hide()
        x = self.Epsw_slider.getX()
        y = self.Epsw_slider.getY()
        w = self.Epsw_slider.getWidth()
        h = self.Epsw_slider.getHeight()
        self.ALP0_slider = Slider(self.screen, x, y, w, h, min=0, max=5, step=0.01, initial=vf_params.ALP0)
        x = self.Epsw_textbox.getX()
        y = self.Epsw_textbox.getY()
        w = self.Epsw_textbox.getWidth()
        h = self.Epsw_textbox.getHeight()
        self.ALP0_textbox = TextBox(self.screen, x, y, w, h, fontSize=self.textbox_height - 2, borderThickness=1)
        x = self.Epsw_help.getX()
        y = self.Epsw_help.getY()
        w = self.Epsw_help.getWidth()
        h = self.Epsw_help.getHeight()
        self.ALP0_help = Button(self.screen, x, y, w, h, text='?', fontSize=self.help_height - 2,
                                  inactiveColour=colors.GREY, borderThickness=1, )
        self.ALP0_help.onClick = lambda: self.show_help('Alpha0', self.ALP0_help)
        self.ALP0_help.onRelease = lambda: self.unshow_help(self.ALP0_help)

        self.help_buttons.append(self.ALP0_help)
        self.sliders.append(self.ALP0_slider)
        self.slider_texts.append(self.ALP0_textbox)

        # Replacing Eu slider with Beta0
        self.Epsu_slider.hide()
        self.Epsu_textbox.hide()
        self.Epsu_help.hide()
        x = self.Epsu_slider.getX()
        y = self.Epsu_slider.getY()
        w = self.Epsu_slider.getWidth()
        h = self.Epsu_slider.getHeight()
        self.BET0_slider = Slider(self.screen, x, y, w, h, min=0, max=5, step=0.01, initial=vf_params.BET0)
        x = self.Epsu_textbox.getX()
        y = self.Epsu_textbox.getY()
        w = self.Epsu_textbox.getWidth()
        h = self.Epsu_textbox.getHeight()
        self.BET0_textbox = TextBox(self.screen, x, y, w, h, fontSize=self.textbox_height - 2, borderThickness=1)
        x = self.Epsu_help.getX()
        y = self.Epsu_help.getY()
        w = self.Epsu_help.getWidth()
        h = self.Epsu_help.getHeight()
        self.BET0_help = Button(self.screen, x, y, w, h, text='?', fontSize=self.help_height - 2,
                                inactiveColour=colors.GREY, borderThickness=1, )
        self.BET0_help.onClick = lambda: self.show_help('Beta0', self.BET0_help)
        self.BET0_help.onRelease = lambda: self.unshow_help(self.BET0_help)

        self.help_buttons.append(self.BET0_help)
        self.sliders.append(self.BET0_slider)
        self.slider_texts.append(self.BET0_textbox)

        # Replacing Suw slider with A1B1
        self.SWU_slider.hide()
        self.SWU_textbox.hide()
        self.SWU_help.hide()
        x = self.SWU_slider.getX()
        y = self.SWU_slider.getY()
        w = self.SWU_slider.getWidth()
        h = self.SWU_slider.getHeight()
        self.ALP1BET1_slider = Slider(self.screen, x, y, w, h, min=0, max=0.1, step=0.001, initial=vf_params.ALP1)
        x = self.SWU_textbox.getX()
        y = self.SWU_textbox.getY()
        w = self.SWU_textbox.getWidth()
        h = self.SWU_textbox.getHeight()
        self.ALP1BET1_textbox = TextBox(self.screen, x, y, w, h, fontSize=self.textbox_height - 2, borderThickness=1)
        x = self.SWU_help.getX()
        y = self.SWU_help.getY()
        w = self.SWU_help.getWidth()
        h = self.SWU_help.getHeight()
        self.ALP1BET1_help = Button(self.screen, x, y, w, h, text='?', fontSize=self.help_height - 2,
                                inactiveColour=colors.GREY, borderThickness=1, )
        self.ALP1BET1_help.onClick = lambda: self.show_help('Alpha1Beta1', self.ALP1BET1_help)
        self.ALP1BET1_help.onRelease = lambda: self.unshow_help(self.ALP1BET1_help)

        self.help_buttons.append(self.ALP1BET1_help)
        self.sliders.append(self.ALP1BET1_slider)
        self.slider_texts.append(self.ALP1BET1_textbox)

        # Replacing Suw slider with V0
        self.SUW_slider.hide()
        self.SUW_textbox.hide()
        self.SUW_help.hide()
        x = self.SUW_slider.getX()
        y = self.SUW_slider.getY()
        w = self.SUW_slider.getWidth()
        h = self.SUW_slider.getHeight()
        self.V0_slider = Slider(self.screen, x, y, w, h, min=0, max=5, step=0.01, initial=vf_params.V0)
        x = self.SUW_textbox.getX()
        y = self.SUW_textbox.getY()
        w = self.SUW_textbox.getWidth()
        h = self.SUW_textbox.getHeight()
        self.V0_textbox = TextBox(self.screen, x, y, w, h, fontSize=self.textbox_height - 2, borderThickness=1)
        x = self.SUW_help.getX()
        y = self.SUW_help.getY()
        w = self.SUW_help.getWidth()
        h = self.SUW_help.getHeight()
        self.V0_help = Button(self.screen, x, y, w, h, text='?', fontSize=self.help_height - 2,
                                    inactiveColour=colors.GREY, borderThickness=1, )
        self.V0_help.onClick = lambda: self.show_help('V0', self.V0_help)
        self.V0_help.onRelease = lambda: self.unshow_help(self.V0_help)

        self.help_buttons.append(self.V0_help)
        self.sliders.append(self.V0_slider)
        self.slider_texts.append(self.V0_textbox)

        # Replacing SUMR slider with Path Length
        self.SUMR_slider.hide()
        self.SUMR_textbox.hide()
        self.SUMR_help.hide()
        x = self.SUMR_slider.getX()
        y = self.SUMR_slider.getY()
        w = self.SUMR_slider.getWidth()
        h = self.SUMR_slider.getHeight()
        self.memory_length = 0
        self.show_path_history = False
        self.PLEN_slider = Slider(self.screen, x, y, w, h, min=0, max=100, step=1, initial=self.memory_length)
        x = self.SUMR_textbox.getX()
        y = self.SUMR_textbox.getY()
        w = self.SUMR_textbox.getWidth()
        h = self.SUMR_textbox.getHeight()
        self.PLEN_textbox = TextBox(self.screen, x, y, w, h, fontSize=self.textbox_height - 2, borderThickness=1)
        x = self.SUMR_help.getX()
        y = self.SUMR_help.getY()
        w = self.SUMR_help.getWidth()
        h = self.SUMR_help.getHeight()
        self.PLEN_help = Button(self.screen, x, y, w, h, text='?', fontSize=self.help_height - 2,
                              inactiveColour=colors.GREY, borderThickness=1, )
        self.PLEN_help.onClick = lambda: self.show_help('PLEN', self.PLEN_help)
        self.PLEN_help.onRelease = lambda: self.unshow_help(self.PLEN_help)

        self.help_buttons.append(self.PLEN_help)
        self.sliders.append(self.PLEN_slider)
        self.slider_texts.append(self.PLEN_textbox)


    def draw_frame(self, stats, stats_pos):
        """
        Overwritten method of sims drawframe adding possibility to update
        pygame widgets
        """
        super().draw_frame(stats, stats_pos)
        self.framerate_textbox.setText(f"Framerate: {self.framerate}")
        self.N_textbox.setText(f"N: {self.N}")
        self.VISRES_textbox.setText(f"V.Resol.: {self.v_field_res}")
        self.FOV_textbox.setText(f"FOV: {int(self.fov_ratio * 100)}%")
        self.ARAD_textbox.setText(f"R ag.: {self.agent_radii:.2f}")
        self.ALP0_textbox.setText(f"A0: {vf_params.ALP0:.2f}")
        self.BET0_textbox.setText(f"B0: {vf_params.BET0:.2f}")
        self.ALP1BET1_textbox.setText(f"A1B1: {vf_params.ALP1:.2f}")
        self.V0_textbox.setText(f"V0: {vf_params.V0:.2f}")
        self.PLEN_textbox.setText(f"path len.: {self.memory_length:.2f}")
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
        # self.keep_agents_in_center()
        #self.screen = pygame.transform.smoothscale(self.screen, (self.screen.get_width(), self.screen.get_height()))

    # def keep_agents_in_center(self):
    #     """Translating all agents with the center of mass, so they keep in the center of the arena"""
    #     posx_coords = np.array([agent.position[0] for agent in self.agents])
    #     posy_coords = np.array([agent.position[1] for agent in self.agents])
    #     COM = (np.mean(posx_coords), np.mean(posy_coords))
    #     arena_center_x = int(self.vis_area_end_width/2)
    #     arena_center_y = int(self.vis_area_end_height / 2)
    #     dx = COM[0] - arena_center_x
    #     dy = COM[1] - arena_center_y
    #     for ai, agent in enumerate(list(self.agents)):
    #         agent.position[0] = agent.position[0] - dx
    #         agent.position[1] = agent.position[1] - dy
    #         if self.pos_memory is not None:
    #             self.pos_memory[ai, 0, -1] += dx
    #             self.pos_memory[ai, 1, -1] += dy



    def interact_with_event(self, events):
        """Carry out functionality according to user's interaction"""
        super().interact_with_event(events)
        pygame_widgets.update(events)

        for event in events:
            if event.type == pygame.KEYUP and event.key == pygame.K_l:
                print("L UP, breaking line")
                self.agent_for_line_id = list(set(self.agent_for_line_id))
                for agid, ag in enumerate(list(self.agents)):
                    if agid in self.agent_for_line_id:
                        print(f"Updating linemap for agent {agid}")
                        ag.lines.append(self.lines[-1])
                        ag.update_linemap()
                        self.agent_for_line_id.remove(agid)
                self.lines.append([])

        self.framerate = self.framerate_slider.getValue()
        self.N = self.N_slider.getValue()
        self.N_resc = self.NRES_slider.getValue()
        self.fov_ratio = self.FOV_slider.getValue()
        if self.N != len(self.agents):
            self.act_on_N_mismatch()
        if self.fov_ratio != self.agent_fov[1] / np.pi:
            self.update_agent_fovs()
        if self.v_field_res != self.VISRES_slider.getValue():
            self.v_field_res = self.VISRES_slider.getValue()
            self.update_agent_vfiled_res()
        if self.agent_radii != self.ARAD_slider.getValue():
            self.agent_radii = self.ARAD_slider.getValue()
            self.update_agent_radii()
        if vf_params.ALP0 != self.ALP0_slider.getValue():
            vf_params.ALP0 = self.ALP0_slider.getValue()
        if vf_params.BET0 != self.BET0_slider.getValue():
            vf_params.BET0 = self.BET0_slider.getValue()
        if vf_params.ALP1 != self.ALP1BET1_slider.getValue():
            vf_params.ALP1 = self.ALP1BET1_slider.getValue()
            vf_params.BET1 = self.ALP1BET1_slider.getValue()
        if vf_params.V0 != self.V0_slider.getValue():
            vf_params.V0 = self.V0_slider.getValue()
        if self.memory_length != self.PLEN_slider.getValue():
            self.memory_length = self.PLEN_slider.getValue()
            if self.memory_length == 0:
                self.show_path_history = False
            else:
                self.show_path_history = True
            self.ori_memory = None
            self.pos_memory = None

        if self.is_recording:
            filename = f"{pad_to_n_digits(self.t, n=6)}.jpeg"
            path = os.path.join(self.image_save_path, filename)
            pygame.image.save(self.screen, path)


    def update_agent_radii(self):
        """Updating agent radius according to slider value"""
        for agent in self.agents:
            agent.radius = self.agent_radii

    def update_agent_vfiled_res(self):
        """Updating visual field resolution of agents"""
        for agent in self.agents:
            agent.v_field_res = self.v_field_res
            # preparing phi values for algorithm according to FOV
            agent.PHI = np.arange(-np.pi, np.pi, (2 * np.pi) / self.v_field_res)

            # social information: visual field projections
            agent.soc_v_field_proj = np.zeros(self.v_field_res)

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

    def draw_visual_fields(self):
        """
        Visualizing the range of vision for agents as opaque circles around the
        agents
        """
        r = 100

        for agent in self.agents:
            visual_field = agent.soc_v_field_proj
            phis = np.linspace(agent.orientation - np.pi, agent.orientation + np.pi,
                               visual_field.shape[0])
            fov = agent.FOV

            # Center and radius of pie chart
            cx = agent.position[0] + agent.radius
            cy = agent.position[1] + agent.radius

            p_proj = [(cx + int(agent.radius * np.cos(phis[0])),
                       cy + int(agent.radius * - np.sin(phis[0])))]
            for i, ang in enumerate(phis):
                height = visual_field[i]
                x = cx + int(r * np.cos(ang) * height)
                y = cy + int(r * - np.sin(ang) * height)
                p_proj.append((x, y))
            p_proj.append((cx + int(agent.radius * np.cos(phis[-1])),
                           cy + int(agent.radius * - np.sin(phis[-1]))))

            image = pygame.Surface(
                [self.vis_area_end_width, self.vis_area_end_height])
            image.fill(colors.BACKGROUND)
            image.set_colorkey(colors.BACKGROUND)
            image.set_alpha(10)

            if 0 < fov[1] < np.pi:
                p_fov = [(cx, cy)]
                # Get points on arc
                angles = [agent.orientation + fov[0],
                          agent.orientation + fov[1]]
                step_size = (angles[1] - angles[0]) / 50
                angles_array = np.arange(angles[0], angles[1] + step_size,
                                         step_size)
                for n in angles_array:
                    x = cx + int(r * np.cos(n))
                    y = cy + int(r * - np.sin(n))
                    p_fov.append((x, y))
                p_fov.append((cx, cy))
                pygame.draw.polygon(image, colors.GREEN, p_fov)
            elif fov[1] == np.pi:
                cx, cy, r = agent.position[0] + agent.radius, agent.position[
                    1] + agent.radius, 100
                pygame.draw.circle(image, colors.GREEN, (cx, cy), r)

            pygame.draw.polygon(image, colors.BLACK, p_proj)
            self.screen.blit(image, (0, 0))
