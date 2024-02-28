import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

root_dir = "/Users/ferielamira/Desktop/Uni/Master-thesis/ABM/"
exp_dir = os.path.join(root_dir, "abm/data/simulation_data/exp_adapt/batch_0")

env_names = ["exp_coll_patchyP", "exp_coll_sparseP", "exp_coll_intermP"]

# Create subplots for each environment

for i, env_name in enumerate(env_names):
    root_eval_path = os.path.join(exp_dir, env_name, "eval")

    # Lists to store data from different files
    total_relocations_a0 = []
    total_relocations_a1 = []
    total_relocations_a2 = []
    all_mean_relocations = []
    fig, axs = plt.subplots(1, 3, figsize=(15, 5 ))

    # Iterate through subdirectories
    for root, dirs, files in os.walk(root_eval_path):
        for dir in dirs:
            log_path = os.path.join(root, dir)
            if "patchy" in env_name:
                N_R = 50
            elif "sparse" in env_name:
                N_R = 3
            elif "interm" in env_name:
                N_R = 10

            # Check if the directory contains TensorFlow logs
            if any(file.startswith('events') for file in os.listdir(log_path)):
                ea = event_accumulator.EventAccumulator(log_path)
                ea.Reload()
                print(f"Texts available in {env_name}: {ea.Tags()}")

                # Extract step and value data from TensorFlow logs
                total_relocations_a0_text = [event.tensor_proto.string_val[0].decode('utf-8') for event in
                                          ea.Tensors(f'Agent0/Total patch joining/text_summary') ]
                total_relocations_a0.append(float(total_relocations_a0_text[0])/N_R)
                total_relocations_a1_text = [event.tensor_proto.string_val[0].decode('utf-8') for event in
                                            ea.Tensors(f'Agent1/Total patch joining/text_summary') ]
                total_relocations_a1.append(float(total_relocations_a1_text[0])/N_R)
                total_relocations_a2_text = [event.tensor_proto.string_val[0].decode('utf-8') for event in
                                            ea.Tensors(f'Agent2/Total patch joining/text_summary') ]
                total_relocations_a2.append(float(total_relocations_a2_text[0])/N_R)





    # Scatter plot for total relocations



    axs[0].scatter(range(1, len(total_relocations_a0) + 1), total_relocations_a0,alpha=0.5)
    axs[0].scatter([5], sum(total_relocations_a0)/len(total_relocations_a0))
    #title after agent 0
    axs[0].set_title('Agent 0')

    axs[1].scatter(range(1, len(total_relocations_a1) + 1), total_relocations_a1,alpha=0.5)
    axs[1].scatter([5], sum(total_relocations_a1)/len(total_relocations_a1))
    axs[1].set_title('Agent 1')
    axs[2].scatter(range(1, len(total_relocations_a2) + 1), total_relocations_a2,alpha=0.5)
    axs[2].scatter([5], sum(total_relocations_a2)/len(total_relocations_a2))
    axs[2].set_title('Agent 2')

    # Set common labels
    for ax in axs.flat:
        ax.set(ylabel='Number of relocations')
    # Set a common title
    if "patchy" in env_name:
        fig.suptitle('Average relocations per agent in patchy environment')
    elif "sparse" in env_name:
        fig.suptitle('Average relocations per agent in sparse environment')
    elif "interm" in env_name:
        fig.suptitle('Average relocations per agent in intermediate environment')

    #save the plots
    plt.savefig(os.path.join(exp_dir,f'{env_name}_total_relocations_per_agent.png'))
    plt.close()
    #scatter the average for all agents
    plt.scatter(range(1, len(total_relocations_a0) + 1), (np.array(total_relocations_a0) + np.array(total_relocations_a1) + np.array(total_relocations_a2))/3,alpha=0.5)
    plt.scatter([5], (sum(total_relocations_a0)/len(total_relocations_a0) + sum(total_relocations_a1)/len(total_relocations_a1) + sum(total_relocations_a2)/len(total_relocations_a2))/3)
    if "patchy" in env_name:
        plt.title('Average relocations in patchy environment')
    elif "sparse" in env_name:
        plt.title('Average relocations in sparse environment')
    elif "interm" in env_name:
        plt.title('Average relocations in intermediate environment')
    plt.savefig(os.path.join(exp_dir,f'{env_name}_total_relocations.png'))
    plt.close()

