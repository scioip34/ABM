import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import json
import glob

# under this path the individual summary statistics are saved
# with _N<number of agents> post tag.
exp_name = "figExp1"
data_path = f"/home/david/Desktop/database/{exp_name}"

# set of agent numbers to summarize for
Ns = [5, 25, 100]
num_patches = [3, 50]

# figure shape
fig_shape = [len(Ns), len(num_patches)]
fig, ax = plt.subplots(fig_shape[0], fig_shape[1],
                       constrained_layout=True, figsize=(fig_shape[1] * 3, fig_shape[0] * 3))
gs1 = gridspec.GridSpec(fig_shape[0], fig_shape[1])
gs1.update(wspace=0, hspace=0)

for ni in range(fig_shape[0]):
    N = Ns[ni]
    # Loading data
    try:
        collapsed_data_occ = np.load(os.path.join(data_path, f"coll_eff_N{N}_occ.npy"))
        collapsed_data_noocc = np.load(os.path.join(data_path, f"coll_eff_N{N}_noocc.npy"))
    except:
        break

    # Finding appropriate columns where number of patches will match
    num_patches_ind_occ = []
    labels = np.loadtxt(os.path.join(data_path, f"coll_eff_N{N}_occ.txt"), dtype=str)
    for li in range(0, len(labels), 2):
        filtered_label = labels[li + 1].replace("N_RESOURCES=", "").replace(".0,", " x ") + labels[li].replace("MIN_RESOURCE_PER_PATCH=", "").replace(".0", "U")
        nup = int(filtered_label.split(" x ")[0])
        if nup in num_patches:
            num_patches_ind_occ.append(int(li/2))
    print(f"In oclluded case will use columns {num_patches_ind_occ}")

    num_patches_ind_noocc = []
    labels = np.loadtxt(os.path.join(data_path, f"coll_eff_N{N}_noocc.txt"), dtype=str)
    for li in range(0, len(labels), 2):
        filtered_label = labels[li + 1].replace("N_RESOURCES=", "").replace(".0,", " x ") + labels[li].replace(
            "MIN_RESOURCE_PER_PATCH=", "").replace(".0", "U")
        nup = int(filtered_label.split(" x ")[0])
        if nup in num_patches:
            num_patches_ind_noocc.append(int(li / 2))
    print(f"In non occluded case will use columns {num_patches_ind_noocc}")

    # Extracting data from collapsed matrices
    lines_occ = collapsed_data_occ[..., num_patches_ind_occ]
    print(lines_occ.shape)
    lines_noocc = collapsed_data_noocc[..., num_patches_ind_noocc]

    for env_i in range(len(num_patches)):
        plt.axes(ax[ni, env_i])
        plt.plot(lines_occ[..., env_i], label=f"occ")
        plt.plot(lines_noocc[..., env_i], label=f"no occ")

    plt.legend()
plt.show()





#
#     plt.axes(ax[ni])
#     for hi in range(fig_shape[1]):
#
#
#
#
#
#
#         ni = int(wi * fig_shape[1]) + hi
#         if ni < len(Ns):
#             print(hi, wi, ni)
#             plt.axes(ax[wi, hi])
#             collapsed_data = np.load(os.path.join(data_path, f"coll_eff_N{Ns[ni]}.npy"))
#             # # column-wise normalization
#             for coli in range(collapsed_data.shape[1]):
#                 print(f"Normalizing column {coli}")
#                 minval = np.min(collapsed_data[:, coli])
#                 maxval = np.max(collapsed_data[:, coli])
#                 collapsed_data[:, coli] = (collapsed_data[:, coli] - minval) / (maxval - minval)
#             agent_dim = 4
#             im = plt.imshow(collapsed_data, vmin=min_data_val, vmax=max_data_val)
#
#             # creating x-axis and labels
#             # reading original labels with long names
#             labels = np.loadtxt(os.path.join(data_path, f"coll_eff_N{Ns[ni]}.txt"), dtype=str)
#
#             # simplifying and concatenating labels
#             conc_x_labels = []
#             for li in range(0, len(labels), 2):
#                 conc_x_labels.append(labels[li + 1].replace("N_RESOURCES=", "").replace(".0,", " x ") +
#                                      labels[li].replace("MIN_RESOURCE_PER_PATCH=", "").replace(".0", "U"))
#
#             # creating y-axis labels
#             tuned_env_pattern = os.path.join(data_path, "tuned_env*.json")
#             print("Patterns: ", tuned_env_pattern)
#             json_files = glob.glob(tuned_env_pattern)
#             for json_path in json_files:
#                 with open(json_path, "r") as f:
#                     envvars = json.load(f)
#                     y_labels = envvars["DEC_EPSW"]
#
#             # Manually definig the sparified y ticks and their rotations
#             sparse_y_indices = [0, 2, 4, 6, 8]  # int((len(y_labels)-1)/2), len(y_labels)-1]
#             y_tick_rotations = [45, 33.75, 22.5, 11.25, 0, -11.25, -22.5, -33.75, -45]
#             sparese_y_tick_rotations = [45, 0, 0, 0, -45]
#
#             # The first plot is detailed shows all ticks and labels
#             if hi == 0 and wi == 0:
#                 plt.ylabel("Social Excitability ($\epsilon_w$)")
#                 plt.yticks([i for i in range(len(y_labels))], y_labels, ha='right', rotation_mode='anchor')
#                 # for ticki, tick in enumerate(ax[wi, hi].get_yticklabels()):
#                 #     print(ticki, tick)
#                 #     tick.set_rotation(y_tick_rotations[ticki])
#
#             # The others are sparsified for better readability
#             elif hi == 0:
#                 plt.yticks([i for i in sparse_y_indices],
#                            [y_labels[i] for i in sparse_y_indices], ha='right', rotation_mode='anchor')
#                 for ticki, tick in enumerate(ax[wi, hi].get_yticklabels()):
#                     tick.set_rotation(sparese_y_tick_rotations[ticki])
#
#             # The ones in second or other columns have same y axis
#             else:
#                 plt.yticks([], [])
#
#             if hi == 0 and wi == fig_shape[0] - 1:
#                 plt.xlabel("Environment")
#                 plt.xticks([i for i in range(0, len(conc_x_labels), 2)],
#                            [conc_x_labels[i] for i in range(0, len(conc_x_labels), 2)],
#                            ha='right', rotation_mode='anchor')
#             elif wi == fig_shape[0]-1:
#                 plt.xticks([i for i in range(0, len(conc_x_labels), 2)],
#                            [conc_x_labels[i] for i in range(0, len(conc_x_labels), 2)],
#                            ha='right', rotation_mode='anchor')
#             for tick in ax[wi, hi].get_xticklabels():
#                 tick.set_rotation(45)
#         else:
#             plt.axes(ax[wi, hi])
#             plt.yticks([], [])
#             plt.xticks([], [])
#
# # Adding the colorbar
# # [left, bottom, width, height]
# cbaxes = fig.add_axes([0.2, 0.805, 0.6, 0.03])
# # position for the colorbar
# cb = plt.colorbar(im, cax=cbaxes, orientation='horizontal')
# cb.ax.set_xticks([min_data_val, (min_data_val+max_data_val)/2, max_data_val])
# cb.ax.xaxis.set_ticks_position("top")
# cb.ax.xaxis.set_label_position('top')
# cb.ax.set_title('Relative Search Efficiency')
# # fig.colorbar(im, ax=ax.ravel().tolist(), orientation = 'horizontal')
# plt.subplots_adjust(hspace=0, wspace=0, top=0.8, bottom=0.2, left=0.2, right=0.8)
# plt.show()
