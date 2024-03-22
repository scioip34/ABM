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

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


"""class DQNetwork(nn.Module):
    def __init__(self, num_observations, output_size):
        super(DQNetwork, self).__init__()
        # Assuming the first 735 values are the visual field and the last one is the binary indicator
        # We treat the visual field as a sequence with 1 channel and 735 features.
        print("num_observations:", num_observations)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1)

        # Calculate the size after convolutions

        # Fully connected layers
        self.fc1 = nn.Linear(1471, 256)  # +1 for the binary indicator
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)

        # Apply He initialization to the convolutional and linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, state):
        visual_field = state[:, :-1].unsqueeze(1)  # Shape (batch_size, 1, 735)
        resource_indicator = state[:, -1]  # Shape (batch_size,)

        x = F.relu(self.conv1(visual_field))

        # Flatten the convolutional output
        conv_out = x.view(x.size(0), -1)

        # Concatenate the convolutional output with the binary resource indicator
        x = torch.cat((conv_out, resource_indicator.unsqueeze(1)), dim=1)

        # Pass through the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def _calculate_conv_output_size(self, input_size):
        temp_input = torch.zeros(1, 1, input_size)
        temp_output = self.conv1(temp_input)
        return temp_output.numel()
"""


class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        # convolutional layer ?
        self.layer1 = nn.Linear(input_size, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, output_size)
        # Apply He initialization to linear layers

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


# Define the DQN agent with replay memory
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.id = id
        self.state_size = state_size
        self.action_size = action_size
        self.state_tensor=None
        self.next_state_tensor=None
        self.action_tensor=None
        self.reward_tensor=None
        self.agent_type = learning_params.brain_type
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

        # Q-network and target Q-network
        if self.brain_type=="DQN":
            self.q_network = DQNetwork(state_size, action_size).to(device)
            self.target_q_network = DQNetwork(state_size, action_size).to(device)
            self.target_q_network.load_state_dict(self.q_network.state_dict())  # Initialize target network with the same weights
            self.target_q_network.eval()
        # Optimizer
        if learning_params.optimizer=="Adam":
            print("Using Adam")
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        else:
            print("Using RMSprop")
            self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=self.lr,weight_decay=1e-4)
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)

        # Replay memory
        self.replay_memory = ReplayMemory(learning_params.replay_memory_capacity)
        #self.writer= SummaryWriter()
        self.legal_actions = None
            #self.target_q_network.eval()
            #print("Model in evaluation mode")

    def select_action_heuristic(self, legal_actions):
        if 1 in legal_actions:
            action = 1
        elif 2 in legal_actions:
            action = 2
        else:
            action = 0

        self.action_tensor=torch.LongTensor([[action]])

        return self.action_tensor

    def get_legal_actions(self,state):
        soc_v_field = state[0][:-1]
        env_status = state[0][-1]


        #if self.last_action == 2 and (soc_v_field.sum() !=0 and env_status == 0.0):
        #    self.legal_actions = [2]

        #else:


        self.legal_actions = [0]

        if env_status > 0.0 :
            self.legal_actions.append(1)
        if soc_v_field.sum() != 0 and self.last_action != 1 and env_status==0.0 :

            self.legal_actions.append(2)


        return self.legal_actions

    def save_model(self, filename):
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            # Add other parameters you want to save
        }
        print("Filename:", filename)
        torch.save(checkpoint, filename)
        print("Model and parameters saved successfully.")

    def load_model_train(self, filename):
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        print("Model and parameters loaded successfully.")

    def select_action(self, state):

        _ = self.get_legal_actions(state)
        if len(self.legal_actions)==1:
            self.action_tensor = torch.LongTensor([[0]]).to(device)

        else:
            # Epsilon-greedy exploration
            eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                math.exp(-1. * self.steps_done / self.epsilon_decay)
            if eps_threshold == 0.01 and self.eps_print:
                print("Epsilon is 0.01 after", self.steps_done, "steps")
                self.eps_print = False

            if random.random() <= eps_threshold:
                action = random.choice(self.legal_actions)
            else:
                with torch.no_grad():
                    q_values = self.q_network(state).detach()

                    indices_descending_order = torch.argsort(q_values,descending=True)[0]

                    for ind in indices_descending_order:
                            if ind in self.legal_actions:
                                action = ind
                                break

            self.action_tensor=torch.LongTensor([[action]]).to(device)
        return self.action_tensor



    def optimize(self):
        if len(self.replay_memory)< self.batch_size:
            return None

        transitions = self.replay_memory.sample(self.batch_size)
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

        state_action_values = self.q_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.

        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_q_network(non_final_next_states).detach().max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute MSE loss
        loss = nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Print or log gradients
        #for name, param in self.q_network.named_parameters():
        #    if param.grad is not None:
        #        print(f'Gradient {name}: {param.grad.norm().item()}')
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 5.0)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        # Update target Q-network by copying the weights from the current Q-network
        target_net_state_dict = self.target_q_network.state_dict()
        policy_net_state_dict = self.q_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_q_network.load_state_dict(target_net_state_dict)
        #print("target_net_state_dict:", target_net_state_dict[key].shape)
        #print("policy_net_state_dict:", policy_net_state_dict[key].shape)


