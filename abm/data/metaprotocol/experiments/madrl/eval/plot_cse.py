import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import hmean
from matplotlib.lines import Line2D

import numpy as np

root_dir = "/Users/ferielamira/Desktop/Uni/Master-thesis/ABM/"
exp_dir = os.path.join(root_dir,"abm/data/simulation_data/exp_adapt/batch_0")

#env_names =["exp_binary-patchyP","exp_binary-sparseP","exp_binary-intermP"]
#env_names=["exp_coll_patchyP"]
#env_names=["exp_coll_sparseP"]
env_names=["exp_coll_intermP"]
#env_names =["exp_coll_patchyP","exp_coll_sparseP","exp_coll_intermP"]

#,"exp_coll_sparseP","exp_coll_intermP"]

#env_names =["exp_coll_patchyP","exp_indiv_patchyP"]



# Specify the root directory containing subdirectories with TensorFlow logs
#root_directory = '/path/to/root/directory'


for env_name in env_names:
    root_eval_path = os.path.join(exp_dir, env_name, "eval")
    # Lists to store data from different files
    all_steps = []
    all_values = []
    outliers = []


    # Iterate through subdirectories
    for root, dirs, files in os.walk(root_eval_path):
        for dir in dirs:
            log_path = os.path.join(root, dir)

            # Check if the directory contains TensorFlow logs
            if any(file.startswith('events') for file in os.listdir(log_path)):
                print(log_path)

                ea = event_accumulator.EventAccumulator(log_path)
                ea.Reload()

                # Extract step and value data from TensorFlow logs
                steps = np.array([event.step for event in ea.Scalars('Collective search efficiency')])
                values = np.array([event.value for event in ea.Scalars('Collective search efficiency')])
                if len(steps) == 10000 and len(values) == 10000:

                    # Append data to lists
                    if sum(values)/len(values) <=0.01:
                        outliers.append(log_path)
                    else:
                        all_steps.append(steps)
                        all_values.append(values)
        print(f"From {len(all_values)+len(outliers)} there were {len(outliers)} outliers")


    # Calculate mean and variance over steps
    mean_values = np.mean(all_values, axis=0)
    #print(mean_values)

    #variance_values = np.var(all_values, axis=0)
    #print(variance_values)

    # Plot the results
    if "coll" in env_name:
        linestyle ='-'
    else:
        linestyle = '--'
    if "sparse" in env_name:
        label = "Sparse"
        color = 'r'
    elif "patchy" in env_name:
        label = "Patchy"
        color = 'g'
    else:
        label = "Intermediate"
        color = 'b'
    for value in all_values:
        plt.plot(all_steps[0], value,  color=color, alpha=0.3, linestyle=linestyle)
    plt.plot(all_steps[0], mean_values,  color=color, linestyle=linestyle)
    #plt.fill_between(all_steps[0], mean_values - variance_values, mean_values + variance_values, color=color, alpha=0.3)


legend_lines = [
    #Line2D([0], [0], linestyle='-', color="black", label='Collective reward'),
    #Line2D([0], [0], linestyle='--', color="black", label='Binary reward'),
    Line2D([0], [0], color="r", label='Sparse environment'),
    Line2D([0], [0], color="g", label='Patchy environment'),
    Line2D([0], [0], color="b", label='Intermediate environment'),
]

# Customize the legend
#plt.legend(handles=legend_lines, loc='lower right')

plt.xlabel('Steps')
plt.ylabel('Collective search efficiency')
plt.title('Average collective search efficiency over time in a intermediate environments')



plt.savefig(os.path.join(exp_dir, "CSE_coll_interm.png"))




