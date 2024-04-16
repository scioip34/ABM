import json
import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# Set the root and experiment directories
root_dir = "/Users/ferielamira/Desktop/Uni/Master-thesis/ABM/"
exp_dir = os.path.join(root_dir, "abm/data/simulation_data/3_agents/exp_new/batch_1")

# Environment names and their mapping
env_names = ["exp_sparse", "exp_interm", "exp_uniform_50", "exp_uniform_100"]
env_label_mapping = {
    "exp_sparse": "3",
    "exp_interm": "10",
    "exp_uniform_50": "50",
    "exp_uniform_100": "100"
}

# Initialize a dictionary to store average duration for each environment
average_duration_per_env = {}

for env_name in env_names:
    root_eval_path = os.path.join(exp_dir, env_name, "eval/batch_0")
    total_duration = 0
    total_count = 0

    for root, dirs, files in os.walk(root_eval_path):
        for trial_dir in dirs:
            log_path = os.path.join(root, trial_dir)

            # Open the JSON file
            with open(os.path.join(log_path, "agent_data.json"), 'r') as file:
                data = json.load(file)

            N_agents = len(data)

            for j in range(N_agents):
                agent_data = data[str(j)]
                current_patch = -1
                start_step = -1

                for step, patch_id in enumerate(agent_data["expl_patch_id"]):
                    if patch_id != current_patch:
                        if current_patch != -1:
                            duration = step - start_step
                            total_duration += duration
                            total_count += 1
                        if patch_id != -1:
                            current_patch = patch_id
                            start_step = step

    average_duration_per_env[env_name] = total_duration / total_count if total_count else 0

# Create a bar plot for the average duration in each environment
env_labels = [env_label_mapping[env] for env in env_names]
average_durations = [average_duration_per_env[env] for env in env_names]

fig, ax = plt.subplots()
ax.bar(env_labels, average_durations)

# Customize the plot
ax.set_xlabel('Environment (Number of Patches)')
ax.set_ylabel('Average Exploitation Duration (in time steps)')
ax.set_title('Average Exploitation Duration Across Environments')

# Save the plot
plt.savefig(os.path.join(exp_dir, "average_exploitation_duration_per_env.png"))
plt.show()
