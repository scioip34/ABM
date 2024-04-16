import json
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Root directory
root_dir = "/Users/ferielamira/Desktop/Uni/Master-thesis/ABM/"
# Define experiment directories and types
exp_dirs = {
    "dqn_new": os.path.join(root_dir, "abm/data/simulation_data/docker_exp_dqn/batch_1"),

    "dqn_old":os.path.join(root_dir, "abm/data/simulation_data/3_agents/exp_new/batch_1"),
    "random": os.path.join(root_dir, "abm/data/simulation_data/exp_random/batch_1"),
    "mec": os.path.join(root_dir,"abm/data/simulation_data/exp_mecanistic/batch_0/3_agents"),
    "heuristic": os.path.join(root_dir,"abm/data/simulation_data/exp_heuristic/batch_0"),
    "ddqn": os.path.join(root_dir,"abm/data/simulation_data/docker_exp_ddqn/batch_0")

}

# Environment names
env_names = ["exp_sparse"] #,"exp_interm","exp_uniform_100"]
N_steps = 20000  # Total number of steps

def process_exp_dir(exp_dir, exp_type):
    data = []

    for env_name in env_names:
        if exp_dir == exp_dirs["mec"] or exp_dir == exp_dirs["heuristic"]:
            root_eval_path = os.path.join(exp_dir,env_name)
        elif exp_dir == exp_dirs["dqn_old"] or exp_dir == exp_dirs["dqn_new"]:
            root_eval_path = os.path.join(exp_dir, env_name, "eval/batch_1")
        else:
            root_eval_path = os.path.join(exp_dir, env_name, "eval/batch_0")

        total_efficiencies = []

        for root, dirs, files in os.walk(root_eval_path):

            for trial_dir in dirs:
                log_path = os.path.join(root, trial_dir)
                with open(os.path.join(log_path, "agent_data.json"), 'r') as file:
                    trial_data = json.load(file)
                N_agents=len(trial_data)

                total_exploitations = sum([sum([0.25 for mode in trial_data[str(j)]["mode"] if mode == 1]) for j in range(N_agents)])
                search_efficiency = total_exploitations / (N_steps * N_agents)
                total_efficiencies.append(search_efficiency)

        avg_efficiency = np.mean(total_efficiencies)
        data.append({"env_name": env_name, "avg_efficiency": avg_efficiency, "type": exp_type})

    return data

# Process experiments and combine data
combined_data = []
for exp_type, exp_dir in exp_dirs.items():
    combined_data += process_exp_dir(exp_dir, exp_type)

# Sort data for plotting
combined_data.sort(key=lambda x: x["env_name"])
#print combined data if type is random
for data in combined_data:
    if data["type"]=='dqn_old':
        print(data)

# Plotting
fig, ax = plt.subplots()
bar_width = 0.20
index = np.arange(len(env_names))

for i, env_name in enumerate(env_names):
    dqn_eff = [data["avg_efficiency"] for data in combined_data if data["env_name"] == env_name and data["type"] == "dqn_new"]
    ddqn_eff = [data["avg_efficiency"] for data in combined_data if data["env_name"] == env_name and data["type"] == "ddqn"]
    dqn_old_eff = [data["avg_efficiency"] for data in combined_data if data["env_name"] == env_name and data["type"] == "dqn_old"]

    rand_eff = [data["avg_efficiency"] for data in combined_data if data["env_name"] == env_name and data["type"] == "random"]
    mec_eff = [data["avg_efficiency"] for data in combined_data if data["env_name"] == env_name and data["type"] == "mec"]
    heuristic_eff =  [data["avg_efficiency"] for data in combined_data if data["env_name"] == env_name and data["type"] == "heuristic"]


    #ax.bar(index[i] , heuristic_eff, bar_width, color="orange",label='Heuristic Agents' if i == 0 else "")
    ax.bar(index[i], dqn_eff, bar_width, color="b",label='IDQN Agents' if i == 0 else "")
    ax.bar(index[i]+ bar_width, dqn_old_eff, bar_width, color="yellow",label='IDQN old Agents' if i == 0 else "")

    #ax.bar(index[i] + 2*bar_width, mec_eff, bar_width, color="r",label='Mechanistic Agents' if i == 0 else "")
    #ax.bar(index[i] + 3*bar_width, rand_eff, bar_width, color="pink",label='Random Agents' if i == 0 else "")
    #ax.bar(index[i] + 4*bar_width, ddqn_eff, bar_width, color="purple",label='IDDQN Agents' if i == 0 else "")


ax.set_xlabel('Environment')
ax.set_ylabel('Average Search Efficiency')
ax.set_title('Search Efficiency Across Environments')
ax.set_xticks(index + 3*bar_width / 2)
ax.set_xticklabels(env_names)
ax.legend()

# Show and save the plot
plt.show()
print(os.path.join(exp_dir, "search_efficiency_comparison.png"))
plt.savefig(os.path.join(exp_dir, "search_efficiency_comparison.png"))

