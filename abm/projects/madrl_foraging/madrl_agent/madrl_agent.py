"""
agent.py : including the main classes to create an agent. Supplementary calculations independent from class attributes
            are removed from this file.
"""
import os

import pygame
import numpy as np
import torch

from abm.contrib import colors
from abm.projects.madrl_foraging.madrl_agent import madrl_supcalc as supcalc
import matplotlib.pyplot as plt

#matplotlib.use('agg')
from abm.projects.madrl_foraging.madrl_agent.brain import DQNAgent
from abm.agent.agent import Agent
from abm.projects.madrl_foraging.madrl_contrib import madrl_learning_params as learning_params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MADRLAgent(Agent):
    """
    Agent class that includes all private parameters of the agents and all methods necessary to move in the environment
    and to make decisions.
    """

    def __init__(self,train,**kwargs):
        """

        """

        # Initializing supercalss (Pygame Sprite)
        super().__init__(**kwargs)

        # in case we run multiple simulations, we reload the env parameters
        #importlib.reload(movement_params)




        # Non-initialisable private attributes
        self.show_stats = True
        self.mode = "explore"  # explore, flock, collide, exploit, pool  # saved
        self.exploit_soc_v_field = np.zeros(self.v_field_res)  # social visual projection field
        self.reloc_soc_v_field = np.zeros(self.v_field_res)  # social visual projection field
        self.explore_soc_v_field = np.zeros(self.v_field_res)  # social visual projection field
        self.train=train
        self.soc_v_field = np.zeros(self.v_field_res)
        self.search_efficiency = 0
        self.res=None
        self.reward=0
        self.ise_w = float(learning_params.ise_w)
        self.cse_w = float(learning_params.cse_w)
        self.last_exploit_time = 1
        self.total_reloc= 0
        self.total_discov= 0



        #create the policy network
        self.policy_network = DQNAgent(state_size=self.v_field_res+ 1, action_size=3)

        if learning_params.pretrained and learning_params.pretrained_models_dir!="":
                print("Loading pretrained model")
                #raise ValueError('Not yet tested, verify the code before using it.')
                model_path = os.path.join(learning_params.pretrained_models_dir, f"model_{self.id}.pth")
                # Specify map_location to load the model on the CPU

                if train:
                    self.policy_network.load_model_train(model_path)
                else:
                    map_location = torch.device('cpu')
                    checkpoint = torch.load(model_path)
                    self.policy_network.q_network.load_state_dict(checkpoint['q_network_state_dict'],map_location)

        if not train:
            print("Model in evaluation mode")
            self.policy_network.q_network.eval()
            self.policy_network.epsilon_start = 0
            self.policy_network.epsilon_end = 0
            #self.target_q_network.eval()
            #print("Model in evaluation mode")

    def update_decision_processes(self):
        """updating inner decision processes according to the policy network"""
        action = self.policy_network.action_tensor.item()
        if action == 0:
            self.set_mode("explore")
        elif action == 1:
            self.set_mode("exploit")
        elif action == 2:
            self.set_mode("relocate")

    def update(self, agents):
        """
        main update method of the agent. This method is called in every timestep to calculate the new state/position
        of the agent and visualize it in the environment
        :param agents: a list of all obstacle/agents coordinates as (X, Y) in the environment. These are not necessarily
                socially relevant, i.e. all agents.
        """

        # calculate socially relevant projection field (Vsoc and Vsoc+)
        #self.calc_social_V_proj(agents)
        #self.visualize_v_fields()
        # update decision processes
        self.update_decision_processes()

        if self.get_mode() == "explore":
            vel, theta = supcalc.random_walk(desired_vel=self.max_exp_vel)

        elif self.get_mode() == "exploit":
            if self.env_status == 1:
                vel, theta = (-self.velocity * self.exp_stop_ratio, 0)
            else:
                print(f"ERROR: Exploiting agent {self.id} is not on a resource patch, will relocate!")
                vel, theta = supcalc.F_reloc_LR(self.velocity, self.soc_v_field)

        elif self.get_mode() == "relocate":
            vel, theta = supcalc.F_reloc_LR(self.velocity, self.soc_v_field)

        if not self.is_moved_with_cursor:  # we freeze agents when we move them
            # updating agent's state variables according to calculated vel and theta
            self.orientation += theta
            self.prove_orientation()  # bounding orientation into 0 and 2pi
            self.velocity += vel
            self.prove_velocity()  # possibly bounding velocity of agent

            # updating agent's position
            self.position[0] += self.velocity * np.cos(self.orientation)
            self.position[1] -= self.velocity * np.sin(self.orientation)

            # boundary conditions if applicable
            self.reflect_from_walls()

        # updating agent visualization
        self.draw_update()

        #TODO: Ask David why this is here
        self.collected_r_before = self.collected_r

    def get_mode(self):
        """returning the current mode of the agent according to it's inner decision mechanisms as a human-readable
        string for external processes defined in the main simulation thread (such as collision that depends on the
        state of the at and also overrides it as it counts as ana emergency)"""
        return self.mode



    def set_mode(self, mode):
        """setting the behavioral mode of the agent according to some human_readable flag. This can be:
            -explore
            -exploit
            -relocate
            -pool
            -collide"""

        self.mode = mode

    def calc_social_V_proj(self, agents):
        """Calculating the socially relevant visual projection field of the agent. This is calculated as the
        projection of nearby exploiting agents that are not visually excluded by other agents"""
        # visible agents (exluding self)
        agents = [ag for ag in agents if supcalc.distance(self, ag) <= self.vision_range]
        # those of them that are exploiting
        exploit_agents = [ag for ag in agents if ag.id != self.id
                       and ag.get_mode() == "exploit"]
        explore_agents= [ag for ag in agents if ag.id != self.id and ag.get_mode() == "explore"]
        reloc_agents= [ag for ag in agents if ag.id != self.id and ag.get_mode() == "relocate"]

        # all other agents to calculate visual exclusions
        non_exploit_agents = [ag for ag in agents if ag not in exploit_agents]
        non_explore_agents = [ag for ag in agents if ag not in explore_agents]
        non_reloc_agents = [ag for ag in agents if ag not in reloc_agents]

        if self.exclude_agents_same_patch:
            # in case agents on same patch are excluded they can still cause visual exclusion for exploiting agents
            # on the same patch (i.e. they can cover agents on other patches)
            non_exploit_agents.extend([ag for ag in exploit_agents if ag.exploited_patch_id == self.exploited_patch_id])
            exploit_agents = [ag for ag in exploit_agents if ag.exploited_patch_id != self.exploited_patch_id]

        # Excluding agents that still try to exploit but can not as the patch has been emptied
        exploit_agents = [ag for ag in exploit_agents if ag.exploited_patch_id != -1]

        if self.visual_exclusion:
            self.exploit_soc_v_field= self.projection_field(exploit_agents, keep_distance_info=True,
                                                     non_expl_agents=non_exploit_agents)
            self.explore_soc_v_field= self.projection_field(explore_agents, keep_distance_info=True,
                                                     non_expl_agents=non_explore_agents)
            self.reloc_soc_v_field= self.projection_field(reloc_agents, keep_distance_info=True,
                                                     non_expl_agents=non_reloc_agents)
        else:
            self.exploit_soc_v_field  = self.projection_field(exploit_agents, keep_distance_info=True)
            self.explore_soc_v_field = self.projection_field(explore_agents, keep_distance_info=True)
            self.reloc_soc_v_field = self.projection_field(reloc_agents, keep_distance_info=True)


        #self.soc_v_field = np.concatenate((self.exploit_soc_v_field, self.reloc_soc_v_field,self.explore_soc_v_field))
        self.soc_v_field = self.exploit_soc_v_field
        return self.soc_v_field

    def visualize_v_fields(self, explore=True,exploit=True,reloc=True):


        if self.vis_counter % 50 == 0:
            # Create a polar plot
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

            # Plot the social visual field
            angles = np.linspace(-np.pi, np.pi,
                                 len(self.exploit_soc_v_field))  # generates an array of evenly spaced values within the range from -pi to pi
            if exploit :
                ax.plot(angles, self.exploit_soc_v_field, label='Exploiting Agents', color="green")
            if explore :
                ax.plot(angles, self.explore_soc_v_field, label='Exploring Agents', color="blue")
            if reloc :
                ax.plot(angles, self.reloc_soc_v_field, label='Relocating Agents', color="pink")

            ax.set_rticks([])  # Remove radial tick labels if not needed
            ax.set_yticklabels([])  # Remove radial tick labels if not needed
            ax.set_theta_zero_location('N')  # Set the zero angle to the top (North)
            plt.title('Social Visual Field')
            plt.legend()

            # Save plot
            #create directory if it does not exist



            #plt.savefig('social_vis_field_frame_{}_agent{}.png'.format(self.vis_counter,self.id))
            # plt.show()
            plt.close()
        self.vis_counter += 1



    def reset(self):
            """
            Reset relevant values of the agent after each train episode.
            """
            # Reset position and orientation
            x=np.random.randint(self.window_pad - self.radius, self.WIDTH + self.window_pad - self.radius)
            y=np.random.randint(self.window_pad - self.radius, self.HEIGHT + self.window_pad - self.radius)
            self.position = np.array((x,y), dtype=np.float64)
            self.orientation = np.random.uniform(0, 2 * np.pi)
            # Reset agent state variables
            self.velocity = 0
            self.collected_r = 0
            self.collected_r_before = 0
            self.exploited_patch_id = -1
            self.mode = "explore"
            self.vis_field_source_data = {}
            self.vis_counter = 0

            # Decision Variables
            self.overriding_mode = None

            # Reset pooling attributes
            self.time_spent_pooling = 0
            self.env_status_before = 0
            self.env_status = 0
            self.pool_success = 0

            self.soc_v_field = np.zeros(self.v_field_res)
            self.search_efficiency = 0

            self.last_exploit_time = 1
            self.total_reloc = 0
            self.total_discov = 0
            self.policy_network.last_action=-1

            # Reset policy network
            #self.policy_network.reset()

            # Update the agent's Pygame representation
            self.rect.x = self.position[0]
            self.rect.y = self.position[1]
            self.image.fill(colors.BACKGROUND)
            pygame.draw.circle(self.image, self.color, (self.radius, self.radius), self.radius)
            pygame.draw.line(self.image, colors.BACKGROUND, (self.radius, self.radius),
                             ((1 + np.cos(self.orientation)) * self.radius,
                              (1 - np.sin(self.orientation)) * self.radius), 3)
            self.mask = pygame.mask.from_surface(self.image)

    '''
    def move_with_mouse(self, mouse, left_state, right_state):
        """Moving the agent with the mouse cursor, and rotating"""
        if self.rect.collidepoint(mouse):
            # setting position of agent to cursor position
            self.position[0] = mouse[0] - self.radius
            self.position[1] = mouse[1] - self.radius
            if left_state:
                self.orientation += 0.1
            if right_state:
                self.orientation -= 0.1
            self.prove_orientation()
            self.is_moved_with_cursor = 1
            # updating agent visualization to make it more responsive
            self.draw_update()
        else:
            self.is_moved_with_cursor = 0
    '''
    '''   
    def pool_curr_pos(self):
        """Pooling process of the current position. During pooling the agent does not move and spends a given time in
        the position. At the end the agent is notified by the status of the environment in the given position"""

        if self.get_mode() == "pool":
            if self.time_spent_pooling == self.pooling_time:
                self.end_pooling("success")
            else:
                self.velocity = 0
                self.time_spent_pooling += 1

    def end_pooling(self, pool_status_flag):
        """
        Ending pooling process either with interrupting pooling with no success or with notifying agent about the status
        of the environemnt in the given position upon success
        :param pool_status_flag: ststing how the pooling process ends, either "success" or "interrupt"
        """
        if pool_status_flag == "success":
            self.pool_success = 1
        else:
            self.pool_success = 0
        self.time_spent_pooling = 0

'''


