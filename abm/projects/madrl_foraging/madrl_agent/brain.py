"""
BSD 3-Clause License

Copyright (c) 2017-2022, Pytorch contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import abm.projects.madrl_foraging.madrl_contrib.madrl_learning_params as learning_params
import torch.nn.init as init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using devide: ",device)


# Define experience tuple for replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state','reward'))
class ReplayMemory(object):
    '''Shared replay memory'''

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saving a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sampling a random batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNetwork(nn.Module):
    """Deep Q-network with 2 hidden layers, ReLU activation functions and Kaiming initialization."""
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()


        self.layer1 = nn.Linear(input_size, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, output_size)

        # Initialize weights
        init.kaiming_uniform_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.layer3.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.layer4.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        output = self.layer4(x)
        return output

class DQNAgent:

    def __init__(self, state_size, action_size):
        self.id = id
        self.state_size = state_size
        self.action_size = action_size
        self.state_tensor=None
        self.next_state_tensor=None
        self.action_tensor=None
        self.reward_tensor=None
        self.eps_print= False
        self.state_history = []

        self.gamma = learning_params.gamma
        self.epsilon_start = learning_params.epsilon_start
        self.epsilon_decay = learning_params.epsilon_decay

        self.epsilon_end = learning_params.epsilon_end
        self.tau=learning_params.tau
        self.steps_done=0
        self.lr = learning_params.lr
        self.batch_size = learning_params.batch_size
        self.pretrained = learning_params.pretrained
        self.brain_type = learning_params.brain_type
        self.last_action = -1

        # Initializing the current ant target DQN networks
        if self.brain_type=="DDQN" or self.brain_type=="DQN":
            self.current_network = DQNetwork(state_size, action_size).to(device)
            self.target_network = DQNetwork(state_size, action_size).to(device)
            self.target_network.load_state_dict(self.current_network.state_dict())  # Initialize target network with the same weights
            self.target_network.eval()



            # Initializing the optimizer (Adam or RmsProp )
            if learning_params.optimizer=="ADAM":
                print("Using Adam")
                self.optimizer = optim.Adam(self.current_network.parameters(), lr=self.lr)
            else:
                print("Using RMSprop")
                self.optimizer = optim.RMSprop(self.current_network.parameters(), lr=self.lr)#,weight_decay=1e-4)

            # Initialize the replay memory
            #self.replay_memory = ReplayMemory(learning_params.replay_memory_capacity)




    def get_legal_actions(self,state):
        """Get legal actions based on the state of the agent.
        """
        # Extract the social and personal information from the state
        soc_v_field = state[0][:-1]
        env_status = state[0][-1]

        # Exploration is always a legal option
        legal_actions = [0]

        # Exploitation is only legal if the agent is overlapping with a non-empty patch
        if env_status > 0.0 :
            legal_actions.append(1)

        # Relocation is only legal if the agent visual field is non-empty and the agent himself is not exploiting
        if (soc_v_field.sum() != 0 and
                (self.brain_type=="ideal" or (self.brain_type!="ideal" and  self.last_action != 1 and env_status==0.0))) :
            legal_actions.append(2)


        return legal_actions




    def select_action(self, state):
        """Select an action based on the state of the agent."""

        # Getting the legal actions for this specific state
        legal_actions = self.get_legal_actions(state)
        # If the agent follows a random policy the action is selected randomly
        if self.brain_type=="random":
            action = random.choice(legal_actions)
            self.action_tensor = torch.LongTensor([[action]])

        # If the agent follows an ideal policy the action is selected based on the following rules:
        # He exploits if he is overlapping with a non-empty patch until its depletion
        # He relocates to the nearest patch if he is not exploiting
        elif self.brain_type =="ideal":
            if 1 in legal_actions:
                action = 1
            elif 2 in legal_actions:
                action = 2
            self.action_tensor = torch.LongTensor([[action]])

        # If the agent follows a DQN or DDQN policy
        # the action is selected based on the Q-values of the current network
        # and the epsilon-greedy policy
        elif self.brain_type=="DQN" or self.brain_type=="DDQN":

            if len(legal_actions)==1 and legal_actions[0]==0:
                self.action_tensor = torch.LongTensor([[0]]).to(device)


            else:
                # Epsilon-greedy exploration
                eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                    math.exp(-self.steps_done / self.epsilon_decay)


                if random.random() <= eps_threshold:
                    action = random.choice(legal_actions)
                else:
                    with torch.no_grad():
                        q_values = self.current_network(state).detach()

                        indices_descending_order = torch.argsort(q_values,descending=True)[0]

                        for ind in indices_descending_order:
                                if ind in legal_actions:
                                    action = ind
                                    break

                self.action_tensor=torch.LongTensor([[action]]).to(device)

        return self.action_tensor

    def save_model(self, filename):
        """
        Saving the models and the optimizer parameters
        Args: filename: the path where the model will be saved
        """
        checkpoint = {
            'q_network_state_dict': self.current_network.state_dict(),
            'target_q_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            # Add other parameters you want to save
        }
        print("Filename:", filename)
        torch.save(checkpoint, filename)
        print("Model and parameters saved successfully.")

    def load_model_train(self, filename):
        """
        Loading the models and the optimizer parameters
        Args: filename: the path from where the model is loaded
        """
        checkpoint = torch.load(filename)
        self.current_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        print("Model and parameters loaded successfully.")

    def optimize(self,replay_memory):
        """
        Optimize the Deep Q-network using the MSE loss
        Args: replay_memory: the shared replay memory
        """

        # Only optimize the model if the replay memory has at least the batch size
        if len(replay_memory)< self.batch_size:
            return None

        transitions = replay_memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        state_action_values = self.current_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.

        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        if self.brain_type=="DDQN":
            # DDQN update
            if sum(non_final_mask) > 0:
                next_state_actions = self.current_network(non_final_next_states).max(1)[1].unsqueeze(1)
                next_state_values[non_final_mask] = self.target_network(non_final_next_states).gather(1,
                                                                                                      next_state_actions).squeeze().detach()
        else:
            # Standard DQN update
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_network(non_final_next_states).detach().max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute MSE loss
        loss = nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        #torch.nn.utils.clip_grad_value_(self.current_network.parameters(), 1.0)

        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        '''
           Updating target Deep Q-network by copying the weights from the current Deep Q-network with Soft Update
        '''
        target_net_state_dict = self.target_network.state_dict()
        policy_net_state_dict = self.current_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_network.load_state_dict(target_net_state_dict)

