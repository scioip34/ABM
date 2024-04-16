import math
import os
import json
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


env_names = ["exp_sparse", "exp_interm", "exp_uniform_50", "exp_uniform_100"]
exp_dir = "/Users/ferielamira/Desktop/Uni/Master-thesis/ABM/abm/data/simulation_data/3_agents/exp_new/batch_1"
# Function to calculate Euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Initialize a dictionary to store relocation data for each environment
relocation_data = {env: {"distances": [], "relocation_counts": []} for env in env_names}

for env_name in env_names:
    root_eval_path = os.path.join(exp_dir, env_name, "eval_2/")

    for root, dirs, files in os.walk(root_eval_path):
        for trial_dir in dirs:
            log_path = os.path.join(root, trial_dir)
            with open(os.path.join(log_path, "agent_data.json"), 'r') as file:
                data = json.load(file)

            # Assuming there are only two agents: agent 0 and agent 1
            agent_0_data = data["0"]
            agent_1_data = data["1"]

            for step in range(len(agent_0_data["posx"])):
                x0, y0 = agent_0_data["posx"][step], agent_0_data["posy"][step]
                x1, y1 = agent_1_data["posx"][step], agent_1_data["posy"][step]
                distance = euclidean_distance(x0, y0, x1, y1)

                # Check if either agent relocated at this step
                relocated = agent_0_data["mode"][step] == 2 or agent_1_data["mode"][step] == 2

                if relocated:
                    relocation_data[env_name]["distances"].append(distance)
                    relocation_data[env_name]["relocation_counts"].append(1)
                else:
                    relocation_data[env_name]["relocation_counts"].append(0)

# Plotting the data
for env_name in env_names:
    plt.figure()
    plt.scatter(relocation_data[env_name]["distances"], relocation_data[env_name]["relocation_counts"], alpha=0.5)
    plt.title(f"Relocations vs. Distance Between Agents in {env_name}")
    plt.xlabel("Distance Between Agents")
    plt.ylabel("Relocation (1 if Relocated, 0 Otherwise)")
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, f"relocation_distance_{env_name}.png"))
    plt.show()
