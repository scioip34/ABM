import json
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


root_dir = "/Users/ferielamira/Desktop/Uni/Master-thesis/ABM/"
exp_dir = os.path.join(root_dir, "abm/data/simulation_data/3_agents/exp_new/batch_1")

env_names = ["exp_sparse", "exp_interm", "exp_uniform_50", "exp_uniform_100"]
# Mapping from environment names to desired labels
env_label_mapping = {
    "exp_sparse": "3",
    "exp_interm": "10",
    "exp_uniform_50": "50",
    "exp_uniform_100": "100"
}


# Create subplots for each environment
relevant_data = [{} for _ in range(len(env_names))]

for i, env_name in enumerate(env_names):
    root_eval_path = os.path.join(exp_dir, env_name, "eval/batch_0")

    # Lists to store data from different trials
    total_relocations_trials = []
    total_explorations_trials = []
    total_exploitations_trials = []

    # Iterate through subdirectories
    for root, dirs, files in os.walk(root_eval_path):
        for trial_dir in dirs:
            log_path = os.path.join(root, trial_dir)

            if "uniform_100" in env_name:
                N_R = 100
            elif "uniform_50" in env_name:
                N_R = 50
            elif "sparse" in env_name:
                N_R = 3
            elif "interm" in env_name:
                N_R = 10

            # Open the json file
            with open(os.path.join(log_path, "agent_data.json"), 'r') as file:
                data = json.load(file)

            N_agents = len(data)
            N_steps = len(next(iter(data.values()))["mode"])  # Assuming all agents have the same number of steps

            trial_relocations = 0
            trial_explorations = 0
            trial_exploitations = 0

            for step in range(N_steps):
                step_modes = [data[str(j)]["mode"][step] for j in range(N_agents)]
                for agent_id in range(N_agents):
                    if step_modes[agent_id] == 2:
                        trial_relocations += 1
                    elif step_modes[agent_id] == 1:
                        trial_exploitations += 1
                    elif step_modes[agent_id] == 0:
                        # Only count exploration if another agent is exploiting
                        if 1 in step_modes[:agent_id] + step_modes[agent_id + 1:]:
                            trial_explorations += 1
            total_relocations_trials.append(trial_relocations)
            total_explorations_trials.append(trial_explorations)
            total_exploitations_trials.append(trial_exploitations)

    # Calculate averages over trials
    relevant_data[i]["env_name"] = env_name
    relevant_data[i]["N_R"] = N_R
    relevant_data[i]["total_relocations"] = np.mean(total_relocations_trials)
    relevant_data[i]["total_explorations"] = np.mean(total_explorations_trials)
    relevant_data[i]["total_exploitations"] = np.mean(total_exploitations_trials)

    print("In environment: ", env_name)
    print(f"Averaged over trials with N_R={N_R}:")
    print(f"Total relocations: {relevant_data[i]['total_relocations']}")
    print(f"Total explorations: {relevant_data[i]['total_explorations']}")
    print(f"Total exploitations: {relevant_data[i]['total_exploitations']}")
    print("\n")

# Lists to store data for plotting
env_names = [data["env_name"] for data in relevant_data]
total_relocations = [data["total_relocations"] / (
        data["total_explorations"]+data["total_relocations"]) for data in relevant_data]
total_explorations = [data["total_explorations"] / (
        data["total_relocations"] + data["total_explorations"]) for data in relevant_data]
#total_exploitations = [data["total_exploitations"] // (
#        data["total_relocations"] + data["total_explorations"]) for data in relevant_data]

# Plotting the bar chart
fig, ax = plt.subplots()
bar_width = 0.25
index = range(len(env_names))

bar1 = ax.bar(index, total_relocations, bar_width, label='Total Relocations')
bar2 = ax.bar([i + bar_width for i in index], total_explorations, bar_width, label='Total Explorations')
#bar3 = ax.bar([i + 2 * bar_width for i in index], total_exploitations, bar_width, label='Total Exploitations')

# Customize the plot
ax.set_xlabel('Number of Patches')
ax.set_ylabel('Relative Frequency')
ax.set_title('Behavioral State Distribution in Foraging Agents Across Environments')
ax.set_xticks([i + 1.5 * bar_width for i in index])
ax.set_xticklabels([env_label_mapping[env_name] for env_name in env_names])
ax.legend()

# Show the plot
plt.savefig(os.path.join(exp_dir,"state_reloc_explore_distribution.png"))

