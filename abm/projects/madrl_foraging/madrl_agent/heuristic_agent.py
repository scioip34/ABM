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

class HeuristicAgent(Agent):
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

        self.exploited_patch = None

        # Non-initialisable private attributes
        self.show_stats = True
        self.mode = "explore"  # explore, flock, collide, exploit, pool  # saved

        self.train=train
        self.soc_v_field = np.zeros(self.v_field_res)
        self.search_efficiency = 0
        self.res=None
        self.last_exploit_time = 1
        self.total_reloc= 0
        self.total_discov= 0



        #create the policy network
        self.policy_network = DQNAgent(state_size=self.v_field_res+ 1, action_size=3)

        self.policy_network.epsilon_start = 0
        self.policy_network.epsilon_end = 0


    def update_decision_processes(self):
        """updating inner decision processes according to the policy network"""

        action = self.action_tensor.item()
        if action == 1:
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

        if self.get_mode() == "exploit":
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

    def calc_resource_V_proj(self,resource):
        if resource[0]!=None:
            self.soc_v_field = self.projection_field(resource, keep_distance_info=True)
        else:
            self.soc_v_field = np.zeros(self.v_field_res)

        return self.soc_v_field

    def get_legal_actions(self,state):
        soc_v_field = state[0][:-1]
        env_status = state[0][-1]


        self.legal_actions=[]
        if env_status > 0.0 :
            self.legal_actions.append(1)
        if soc_v_field.sum() != 0:# and self.last_action != 1 and env_status==0.0 :

            self.legal_actions.append(2)


        return self.legal_actions


    def select_action(self, state):

        legal_actions = self.get_legal_actions(state)

        if 1 in legal_actions:
            action = 1
        elif 2 in legal_actions:
            action = 2

        self.action_tensor=torch.LongTensor([[action]])

        return self.action_tensor


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



