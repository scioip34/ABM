import copy
import random
import shutil

import optuna
import pygame
import numpy as np

import sys

import torch
from matplotlib import pyplot as plt
import tensorboard
from torch.utils.tensorboard import SummaryWriter

from abm.monitoring import ifdb, env_saver
from abm.projects.madrl_foraging.madrl_agent.madrl_agent import MADRLAgent as Agent
from abm.contrib import colors,ifdb_params as logging_params
from abm.projects.madrl_foraging.madrl_contrib import madrl_learning_params as learning_params
from abm.simulation.sims import Simulation, notify_agent, refine_ar_overlap_group



from datetime import datetime

class MADRLSimulation(Simulation):
    def __init__(self, train,train_every,**kwargs):
        """
        Inherited from Simulation class
        :param agent_type: type of the agent, so far only (DQN,LSTMDQN)
        :param train: boolean, if true the simulation will be ran in training mode, if false in evaluation mode
        :param train_every: int, number of timesteps after which the agent will be trained
        :param pretrained: boolean, if true the agent will use pretrained models
        :param pretrained_models_dir: string, path to the directory where the pretrained models are stored
        :param save_models_dir: string, path to the directory where the models will be saved
        :param replay_memory_capacity: int, capacity of the replay memory
        :param batch_size: int, size of the batch for training
        :param gamma: float, discount factor
        :param lr: float, learning rate
        :param epsilon_start: float, start value for epsilon
        :param epsilon_end: float, end value for epsilon
        :param epsilon_decay: float, decay factor for epsilon
        :param tau: float, tau for soft update
        :param optimizer: string, optimizer to be used

        """
        super().__init__(**kwargs)



        self.train=train
        #self.binary_env_status = learning_params.binary_env_status
        #self.with_tp = learning_params.tp
        seed = learning_params.seed
        if self.train:
            #TODO: If i am further training pretrained models i should not use the same seed
            #random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        else:

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            #raise Exception("Evaluation mode not supported for MADRL simulation!")



        self.train_every=train_every
        #TODO: Add num episodes to learning_params
        self.num_episodes = 50 #learning_params.num_episodes






    def add_new_agent(self, id, x, y, orient, with_proove=False, behave_params=None):
        """Adding a single new agent into agent sprites"""
        agent_proven = False

        while not agent_proven:
            agent = Agent(
                    id=id,
                    radius=self.agent_radii,
                    position=(x, y),
                    orientation=orient,
                    env_size=(self.WIDTH, self.HEIGHT),
                    color=colors.BLUE,
                    v_field_res=self.v_field_res,
                    FOV=self.agent_fov,
                    window_pad=self.window_pad,
                    pooling_time=self.pooling_time,
                    pooling_prob=self.pooling_prob,
                    consumption=self.agent_consumption,
                    vision_range=self.vision_range,
                    visual_exclusion=self.visual_exclusion,
                    patchwise_exclusion=self.patchwise_exclusion,
                    behave_params=None,
                train=self.train,
                )
            if with_proove:
                if self.proove_sprite(agent):
                    self.agents.add(agent)
                    agent_proven = True
            else:
                self.agents.add(agent)
                agent_proven = True

    def agent_resource_overlap(self, agents):
        collision_group_ar = pygame.sprite.groupcollide(
            self.rescources,
            agents,
            False,
            False,
            pygame.sprite.collide_circle
        )

        # refine collision group according to point-like pooling in center of agents
        collision_group_ar = refine_ar_overlap_group(collision_group_ar)

        return collision_group_ar


    def agent2resource_interaction(self,  collided_agents):
        collision_group_ar = self.agent_resource_overlap(self.agents)

        # collecting agents that are on resource patch
        agents_on_rescs = []


        # Notifying agents about resource if pooling is successful + exploitation dynamics
        for resc, agents in collision_group_ar.items():  # looping through patches

                destroy_resc = False  # if we destroy a patch it is 1
                for agent in agents:  # looping through all agents on patches
                    self.bias_agent_towards_res_center(agent, resc)

                    # One of previous agents on patch consumed the last unit
                    if destroy_resc:
                        notify_agent(agent, -1)
                    else:
                        # Agent finished pooling on a resource patch
                        if (agent.get_mode() in ["pool", "relocate"] and agent.pool_success) \
                                or agent.pooling_time == 0:
                            # Notify about the patch
                            notify_agent(agent, 1, resc.id)
                            # Teleport agent to the middle of the patch if needed
                            if self.teleport_exploit:
                                agent.position = resc.position + resc.radius - agent.radius

                        # Agent is exploiting this patch
                        if agent.get_mode() == "exploit":
                            # continue depleting the patch
                            depl_units, destroy_resc = resc.deplete(agent.consumption)
                            agent.collected_r_before = agent.collected_r  # rolling resource memory
                            agent.collected_r += depl_units  # and increasing it's collected rescources
                             # remember the time of last exploitation
                            if destroy_resc:  # consumed unit was the last in the patch
                                # print(f"Agent {agent.id} has depleted the patch all agents must be notified that"
                                #       f"there are no more units before the next timestep, otherwise they stop"
                                #       f"exploiting with delays")
                                for agent_tob_notified in agents:
                                    # print("C notify agent NO res ", agent_tob_notified.id)
                                    notify_agent(agent_tob_notified, -1)

                    # Collect all agents on resource patches
                    agents_on_rescs.append(agent)

                # Patch is fully depleted
                if destroy_resc:
                    # we clear it from the memory and regenerate it somewhere else if needed
                    self.kill_resource(resc)

        # Notifying other agents that there is no resource patch in current position (they are not on patch)
        for agent in self.agents.sprites():
            if agent not in agents_on_rescs:  # for all the agents that are not on recourse patches
                if agent not in collided_agents:  # and are not colliding with each other currently
                    # if they finished pooling
                    if (agent.get_mode() in ["pool",
                                             "relocate"] and agent.pool_success) or agent.pooling_time == 0:
                        notify_agent(agent, -1)
                    elif agent.get_mode() == "exploit":
                        notify_agent(agent, -1)

        # Update resource patches
        self.rescources.update()

    def step(self,turned_on_vfield):
        # order the agents by id to ensure that agent 0 has priority over agent 1 and agent 1 over agent 2

        # Update internal states of the agents and their positions
        self.agents.update(self.agents)

        # Check for agent-agent collisions
        # collided_agents = self.agent2agent_interaction()
        collided_agents = []

        # Check for agent-resource interactions and update the resource patches
        self.agent2resource_interaction(collided_agents)

        for ag in self.agents:
            ag.calc_social_V_proj(self.agents)
            ag.search_efficiency = ag.collected_r / self.t if self.t != 0 else 0


        collective_se = sum(ag.search_efficiency for ag in self.agents) / len(
            self.agents)


        # Draw the updated environment and agents (and visual fields if needed)
        # deciding if vis field needs to be shown in this timestep
        self.decide_on_vis_field_visibility(turned_on_vfield)
        if self.with_visualization:
            self.draw_frame(self.stats, self.stats_pos)
            pygame.display.flip()

        # TODO: Should this be placed right after the visualization?
        return collective_se
    def compute_reward(self,ag,collective_se):

        #sparse and egoistical rewards
        '''time_penalty= 1
        if self.with_tp:
            if self.t != ag.last_exploit_time :
                time_penalty = self.t - ag.last_exploit_time

        if ag.get_mode()=="exploit" and ag.collected_r>ag.collected_r_before:
            reward=1/time_penalty

        else:
            reward=0
        '''
        '''            
        if self.with_tp and self.t != ag.last_exploit_time :
                time_penalty = self.t - ag.last_exploit_time
            else:
                time_penalty = 1

            reward = (ag.ise_w * ag.search_efficiency + ag.cse_w * collective_se) / time_penalty
        '''
        '''
        reward = 0
        if ag.get_mode()=="exploit":
            reward = (ag.ise_w * ag.search_efficiency + ag.cse_w * collective_se)
        '''


        reward = 0

        if ag.get_mode()=="exploit":

            reward = 1

        return reward

    def start_madqn_eval(self):
        '''Main simulation loop for training the agents with MADQN'''

        # Start time of the simulation
        start_time = datetime.now()

        # Create the agents and resources patches  in the environment
        self.create_agents()
        self.create_resources()

        # Create surface to show visual fields and local var to decide when to show visual fields
        self.stats, self.stats_pos = self.create_vis_field_graph()

        # Create list to store collective search efficiency at each time step
        collective_se_list = []

        # Create variable to indicate if the simulation is done
        done= False

        # Create a directory to save the training data (models and tensorboard logs)

        eval_save_dir = logging_params.TIMESTAMP_SAVE_DIR
        print(f"Saving evaluation data to {eval_save_dir}!")

        # Create a tensorboard writer to save the training logs
        writer = SummaryWriter(eval_save_dir)
        writer.add_text('Hyperparameters',
                        f'Gamma: {learning_params.gamma}, \n Epsilon Start: {learning_params.epsilon_start}, \n Epsilon End: {learning_params.epsilon_end}, \n'
                        f'Epsilon Decay: {learning_params.epsilon_decay},\n Tau: {learning_params.tau},\n Learning Rate: {learning_params.lr}',
                        0)
        #writer.add_text('Experiment parameters', f"ISE_W: {learning_params.ise_w}, \n CSE_W {learning_params.cse_w} \n TP {learning_params.tp},\n BINARY_ENV_STATUS: {learning_params.binary_env_status}",1)


        turned_on_vfield = 0

        # Initialize the social visual fields for each agent
        self.initialize_environment()

        print(f"Starting main simulation loop in MADQN evaluation mode with {len(self.agents)} agents and {len(self.rescources)} resources !")
        while self.t < self.T:

            # Indicate that the simulation is not done
            if self.t==self.T-1:
                done = True


            # Agent 0 always has riority over agent 1 and agent 1 over agent 2
            # If the three of them are on the same patch, and there are not enough resources agent 0 will be allowed to deplete the patch, followed by agent 1, then agent 2

            for ag in self.agents:
                # Select an action
                state = ag.policy_network.state_tensor

                _ = ag.policy_network.select_action(state)
                if ag.policy_network.action_tensor==1:
                    ag.last_exploit_time = 1
                else:
                    ag.last_exploit_time += 1


            collective_se = self.step(turned_on_vfield)
            collective_se_list.append(collective_se)
            # Train the agents


            for ag in self.agents:


                if done:
                    ag.policy_network.next_state_tensor = None
                else:
                    # Concatenate the resource signal array for the next state tensor (The social visual field (1D array )+ the environment status (Scalar))
                    if ag.env_status == 1:
                            # calculate number of resources left in the patch
                            ag_resc_overlap = self.agent_resource_overlap([ag])
                            resc = list(ag_resc_overlap.keys())[0]

                            ag.policy_network.next_state_tensor = torch.FloatTensor(
                                ag.soc_v_field.tolist() + [resc.resc_left / resc.resc_units]).unsqueeze(0)

                    else:

                        ag.policy_network.next_state_tensor = torch.FloatTensor(
                            ag.soc_v_field.tolist() + [0.0]).unsqueeze(0)

                if ag.policy_network.last_action == 0 and ag.get_mode() == "exploit":
                    ag.total_discov += 1
                if ag.policy_network.last_action == 2 and ag.get_mode() == "exploit":
                    ag.total_reloc += 1
                reward = self.compute_reward(ag,collective_se)


                ag.policy_network.reward_tensor = torch.FloatTensor([reward])
                ag.policy_network.state_tensor = ag.policy_network.next_state_tensor
                ag.policy_network.last_action = ag.policy_network.action_tensor.item()



                if self.t % 100:
                    writer.add_scalar(f'Agent_{ag.id}/Individual search efficiency)', ag.search_efficiency,
                                          self.t)

            avg_search_efficiency = sum(collective_se_list) / len(collective_se_list)
            writer.add_scalar('Collective search efficiency', collective_se, self.t)
            # move to next simulation timestep (only when not paused)
            self.t += 1
                # save the agent and resource data to the ram for later analysis
            if self.save_in_ram:
                ifdb.save_agent_data_RAM(self.agents, self.t)
                ifdb.save_resource_data_RAM(self.rescources, self.t)

            #if self.with_visualization:
            self.clock.tick(self.framerate)

        # Calculate the average collective search efficiency
        # Close the tensorboard writer
        print("Collective search efficiency", collective_se)
        for ag in self.agents:
            writer.add_text('Agent{}/Total discoveries'.format(ag.id), str(ag.total_discov))
            writer.add_text('Agent{}/Total patch joining'.format(ag.id), str(ag.total_reloc))
        writer.close()
        print("env path", self.env_path)
        env_saver.save_env_vars([self.env_path], "env_params.json", pop_num=None)


        if self.save_csv_files:
            if self.save_in_ifd or self.save_in_ram:

                ifdb.save_ifdb_as_csv(exp_hash=self.ifdb_hash, use_ram=self.save_in_ram, as_zar=self.use_zarr,
                                      save_extracted_vfield=False, pop_num=None)
        else:
            raise Exception("Tried to save simulation data as csv file due to env configuration, "
                            "but IFDB/RAM logging was turned off. Nothing to save! Please turn on IFDB/RAM logging"
                            " or turn off CSV saving feature.")






        # Quit the pygame environment
        pygame.quit()

        # Print the execution time
        end_time = datetime.now()
        print(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')} Total simulation time: ",
          (end_time - start_time).total_seconds())

        return avg_search_efficiency



    def initialize_environment(self):
        for ag in self.agents:
            # Check for agent-agent collisions
            # collided_agents = self.agent2agent_interaction()
            collided_agents = []
            # Check for agent-resource interactions and update the resource patches
            ag_resc_overlap = self.agent_resource_overlap([ag])
            if len(ag_resc_overlap) > 0:
                ag.env_status = 1

            ag.calc_social_V_proj(self.agents)
            # Concatenate the resource signal array for the state tensor (The social visual field (1D array )+ the
            # environment status (Scalar))
            if ag.env_status == 1:
                # calculate number of resources left in the patch
                #if self.binary_env_status==False:

                resc = list(ag_resc_overlap.keys())[0]
                ag.policy_network.state_tensor = torch.FloatTensor(
                ag.soc_v_field.tolist() + [resc.resc_left / resc.resc_units]).unsqueeze(0)
                #else:
                #    ag.policy_network.state_tensor = torch.FloatTensor(ag.soc_v_field.tolist() + [1.0]).unsqueeze(0)

            else:
                ag.policy_network.state_tensor = torch.FloatTensor(ag.soc_v_field.tolist() + [0.0]).unsqueeze(0)

        #if self.with_visualization:
        #    self.draw_frame(self.stats, self.stats_pos)
        #    pygame.display.flip()

    def start_madqn_train(self):
        """Main simulation loop for training the agents with MADQN"""

        # Start time of the simulation
        start_time = datetime.now()

        # Create the agents and resources patches  in the environment
        self.create_agents()
        self.create_resources()

        # Create surface to show visual fields and local var to decide when to show visual fields
        self.stats, self.stats_pos = self.create_vis_field_graph()

        # Create list to store collective search efficiency at each time step



        # Create a directory to save the training data (models and tensorboard logs)

        train_save_dir = logging_params.TIMESTAMP_SAVE_DIR
        print(f"Saving training data to {train_save_dir}!")

        writer = SummaryWriter(train_save_dir)
        writer.add_text('Hyperparameters',
                        f'Gamma: {learning_params.gamma}, \n Epsilon Start: {learning_params.epsilon_start}, '
                        f'\n Epsilon End: {learning_params.epsilon_end}, \n'
                        f'Epsilon Decay: {learning_params.epsilon_decay},\n Tau: {learning_params.tau},\n Learning '
                        f'Rate: {learning_params.lr}',
                        0)
        #writer.add_text('Experiment parameters', f"ISE_W: {learning_params.ise_w}, \n CSE_W {learning_params.cse_w} \n TP {learning_params.tp},\n BINARY_ENV_STATUS: {learning_params.binary_env_status}",1)
        turned_on_vfield = 0




        print(f"Starting main simulation loop in MADQN training mode with {len(self.agents)} agents and {len(self.rescources)} resources !")
        #print(f"ISE_W: {learning_params.ise_w}, \n CSE_W {learning_params.cse_w} \n TP {learning_params.tp},\n BINARY_ENV_STATUS: {learning_params.binary_env_status}")
        for episode in range(self.num_episodes + 1):
            # Create a variable to indicate if the simulation is done
            done= False
            self.initialize_environment()
            collective_se_list = []

            while self.t < self.T:


                # Indicate that the simulation is not done
                if self.t==self.T-1:
                    done = True


                # Agent 0 always has riority over agent 1 and agent 1 over agent 2
                # If the three of them are on the same patch, and there are not enough resources agent 0 will be allowed to deplete the patch, followed by agent 1, then agent 2
                # If the three of them are on the same patch, and there are not enough resources agent 0 will be allowed
                # to deplete the patch, followed by agent 1, then agent 2
                # If the three of them are on the same patch, and there are not enough resources agent 0 will be allowed to deplete the patch, followed by agent 1, then agent 2
                exploiting_agents= []
                for ag in self.agents:
                    # Select an action


                    _ = ag.policy_network.select_action(ag.policy_network.state_tensor)


                collective_se = self.step(turned_on_vfield)

                collective_se_list.append(collective_se)
                # Train the agents
                for ag in self.agents:

                    if done:
                        ag.policy_network.next_state_tensor = None
                        reward = collective_se
                    else:
                        # Concatenate the resource signal array for the next state tensor (The social visual field (1D array )+ the environment status (Scalar))
                        if ag.env_status == 1:
                            #if self.binary_env_status==False:
                                #calculate number of resources left in the patch
                            ag_resc_overlap = self.agent_resource_overlap([ag])
                            resc= list(ag_resc_overlap.keys())[0]
                            ag.policy_network.next_state_tensor = torch.FloatTensor(ag.soc_v_field.tolist() + [resc.resc_left/resc.resc_units]).unsqueeze(0)


                            #else:
                            #    ag.policy_network.next_state_tensor = torch.FloatTensor(ag.soc_v_field.tolist() + [1.0]).unsqueeze(0)

                        else:
                            resc=None
                            ag.policy_network.next_state_tensor = torch.FloatTensor(ag.soc_v_field.tolist() + [0.0]).unsqueeze(0)

                        # Calculate the reward as a weighted sum of the individual and collective search efficiency
                        reward = self.compute_reward(ag,collective_se)
                        if ag.policy_network.action_tensor.item() == 1:
                            ag.last_exploit_time = self.t

                    ag.policy_network.reward_tensor = torch.FloatTensor([reward])

                    # Add the experience to the replay memory and train the agent


                    if self.train:

                        ag.policy_network.replay_memory.push(ag.policy_network.state_tensor,
                                                                 ag.policy_network.action_tensor,
                                                                 ag.policy_network.next_state_tensor,
                                                                 ag.policy_network.reward_tensor)

                        loss = ag.policy_network.optimize()
                        # Update the target network with soft updates
                        ag.policy_network.update_target_network()


                        if loss is not None:
                            writer.add_scalar(f'Agent_{ag.id}/Loss', loss, self.t+episode)


                        # Move to the next training step
                        ag.policy_network.steps_done += 1
                    ag.policy_network.state_tensor = ag.policy_network.next_state_tensor
                    ag.policy_network.last_action = ag.policy_network.action_tensor.item()
                # move to next simulation timestep (only when not paused)
                self.t += 1

                if self.save_in_ram:
                    ifdb.save_agent_data_RAM(self.agents, self.t)
                    ifdb.save_resource_data_RAM(self.rescources, self.t)

                if done :
                    for ag in self.agents:
                        writer.add_scalar(f'Agent_{ag.id}/Individual search efficiency)', ag.search_efficiency,
                                        episode)
                    writer.add_scalar('Collective search efficiency', collective_se, episode)

                    for ag in self.agents:
                        ag.reset()
                    for resc in self.rescources:
                        self.kill_resource(resc)
                        #self.create_resources()
                    self.t=0
                    break


            #for resc in self.rescources:
            #    self.kill_resource(resc)
            #self.create_resources()
            self.t=0


        # Calculate the average collective search efficiency
        avg_search_efficiency= sum(collective_se_list) / len(collective_se_list)

        # Save the models
        for count, ag in enumerate(self.agents):
            torch.save(ag.policy_network.q_network.state_dict(),
                       f'{train_save_dir}/model_{ag.id}.pth')
            print(f"Model {ag.id} saved to {train_save_dir}!")
        # Close the tensorboard writer
        writer.close()
        env_saver.save_env_vars([self.env_path], "env_params.json", pop_num=None)

        if self.save_csv_files:
            if self.save_in_ifd or self.save_in_ram:

                ifdb.save_ifdb_as_csv(exp_hash=self.ifdb_hash, use_ram=self.save_in_ram, as_zar=self.use_zarr,
                                      save_extracted_vfield=False, pop_num=None)


        # Quit the pygame environment
        pygame.quit()

        # Print the execution time
        end_time = datetime.now()
        print(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')} Total simulation time: ",
          (end_time - start_time).total_seconds())

        return avg_search_efficiency

'''
    def start_opt(self,trial):


        #TODO: Mode should be a hyperparameter of the experiment
        print(f"Running simulation start method!")

        # Creating N agents in the environment
        print("Creating agents!")
        self.create_agents()



        # Creating resource patches
        print("Creating resources!")
        self.create_resources()

        # Creating surface to show visual fields
        print("Creating visual field graph!")
        self.stats, self.stats_pos = self.create_vis_field_graph()

        # local var to decide when to show visual fields
        turned_on_vfield = 0
        if self.agent_type=="DQN":
            if self.train:
                print("Starting main simulation loop in DQN training mode!")
            else:
                #error and exit
                print("Error: DQN evaluation mode not supported for optimization!")
                sys.exit()
        else:
            #error and exit
            print("Error: Mechanistic mode not supported for optimization!")


        collective_se = 0
        collective_se_list = []  # List to store collective search efficiency at each time step

        # Main Simulation loop until dedicated simulation time
        for episode in range(1, self.num_episodes + 1):

            done= False
            while self.t < self.T:
              

                if self.t==self.T-1:
                    done = True

                events = pygame.event.get()
                # Carry out interaction according to user activity
                self.interact_with_event(events)

                # deciding if vis field needs to be shown in this timestep
                turned_on_vfield = self.decide_on_vis_field_visibility(turned_on_vfield)

                if not self.is_paused:

                    # # ------ AGENT-AGENT INTERACTION ------
                    collided_agents = self.agent2agent_interaction()

                    # ------ AGENT-RESCOURCE INTERACTION (can not be separated from main thread for some reason)------
                    self.agent2resource_interaction(collided_agents)
                    # Update agents according to current visible obstacles
                    for ag in self.agents:
                        ag.calc_social_V_proj(self.agents)

                        if self.agent_type=="DQN":
                            # Concatenate the resource signal array along a new axis (axis=0 in this case)
                            state = np.concatenate((ag.soc_v_field, [ag.env_status]))
                            # check how many values in state are non zero
                            action=ag.policy_network.select_action(state)
                    self.agents.update(self.agents)

                    if self.agent_type=="DQN":
                        if self.train and self.t%self.train_every==0:
                            collective_se=0
                            for ag in self.agents:
                                if self.t != 0:
                                    collective_se += (ag.collected_r / self.t)
                                else:
                                    collective_se = 0
                            collective_se = collective_se / len(self.agents)

                            for ag in self.agents:
                                # Concatenate the resource signal array along a new axis (axis=0 in this case)
                                next_state = np.concatenate((ag.soc_v_field, [ag.env_status]))

                                if self.t!=0:
                                    individual_se= ag.collected_r /self.t # - ag.collected_r_before
                                else:
                                    individual_se=0

                                # Store experiences in replay memory
                                reward = 0.8*individual_se + 0.2*collective_se

                                ag.policy_network.add_to_replay_memory(Experience(state, action,reward, next_state, done))
                                loss = ag.policy_network.train()

                                ag.policy_network.update_target_network()

                        for ag in self.agents:
                            ag.policy_network.steps_done += 1

                    # move to next simulation timestep (only when not paused)
                    self.t += 1

                # Simulation is paused
                else:
                    # Still calculating visual fields
                    for ag in self.agents:
                        ag.calc_social_V_proj(self.agents)

                # Draw environment and agents
                if self.with_visualization:
                    self.draw_frame(self.stats, self.stats_pos)
                    pygame.display.flip()

                # Moving time forward
                collective_se_list.append(collective_se)
                trial.report(collective_se, self.t)
                # Prune the intermediate value if neccessary.
                if trial.should_prune() or (self.t>self.T/2 and collective_se<0.09):
                    print("Trial pruned at timestep: ", self.t)
                    raise optuna.TrialPruned()

                self.clock.tick(self.framerate)

        # Reset simulation environment
        for ag in self.agents:
            ag.reset()
        for resc in self.rescources:
            self.kill_resource(resc)
        self.t=0

        pygame.quit()
        avg_search_efficiency= sum(collective_se_list) / len(collective_se_list)
        print("Trial ended with collective search efficiency: ", avg_search_efficiency)
        return avg_search_efficiency


'''
