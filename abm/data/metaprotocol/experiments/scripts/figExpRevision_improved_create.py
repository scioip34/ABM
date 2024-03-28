import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
import os
import json
import glob
import colorcet as cc

clist_1 = ['#303030', '#443762', '#534b7a', '#5c6c73', '#4b9847', '#5cc22c', '#a9dc63', '#dbef97', '#f8faca']
# clist_1.reverse()
# pssoibilities: fire, CET_L18, CET_L17, CET_L17_r, 20, CET_CBL1, CET_CBL2
cmap = cc.cm.CET_CBL2

# under this path the individual summary statistics are saved
# with _N<number of agents> post tag.
data_path = f"G:\\ABMData\\StatFor\\RevisionData\\DataRevision"


# set of agent numbers to summarize for
Ns = [5, 10, 25, 50]
conditions = ["Half", "Normal", "Twice"]
y_titles = ["$N_R^{TOTAL}=1200$\nSocial Excitability $\epsilon_w$", "$N_R^{TOTAL}=2400$\nSocial Excitability $\epsilon_w$", "$N_R^{TOTAL}=4800$\nSocial Excitability $\epsilon_w$"]
columnwise_norm = True

if columnwise_norm:
    min_data_val = 0
    max_data_val = 1
    colorbar_title = "Relative Search Efficiency"
else:
    min_data_val = 0
    max_data_val = 0.25
    colorbar_title = "Absolute Search Efficiency"

# Manually definig the sparified y ticks and their rotations
sparse_y_indices = [0, 4, 8]  # int((len(y_labels)-1)/2), len(y_labels)-1]
y_tick_rotations = [45, 33.75, 22.5, 11.25, 0, -11.25, -22.5, -33.75, -45]
sparese_y_tick_rotations = [0, 0, 0]
only_numpatches = False

# figure shape
fig_shape = [len(conditions), len(Ns)]
fig, ax = plt.subplots(fig_shape[0], fig_shape[1],
                       constrained_layout=True, figsize=(fig_shape[1] * 3, fig_shape[0] * 3),
                       sharex=False)
gs1 = gridspec.GridSpec(fig_shape[0], fig_shape[1])
gs1.update(wspace=0, hspace=0)
for wi in range(fig_shape[0]):
    print(wi)
    for hi in range(fig_shape[1]):
        # ni = int(wi * fig_shape[1]) + hi
        ni = hi
        if ni < len(Ns):
            print(hi, wi, ni)
            plt.axes(ax[wi, hi])
            collapsed_data = np.load(os.path.join(data_path, f"coll_eff_N{Ns[ni]}_{conditions[wi]}.npy"))
            # column-wise normalization
            if columnwise_norm:
                for coli in range(collapsed_data.shape[1]):
                    print(f"Normalizing column {coli}")
                    minval = np.min(collapsed_data[:, coli])
                    maxval = np.max(collapsed_data[:, coli])
                    collapsed_data[:, coli] = (collapsed_data[:, coli] - minval) / (maxval - minval)

            agent_dim = 4
            im = plt.imshow(collapsed_data, vmin=min_data_val, vmax=max_data_val, origin="lower", cmap=cmap)

            # creating x-axis and labels
            # reading original labels with long names
            labels = np.loadtxt(os.path.join(data_path, f"coll_eff_N{Ns[ni]}_{conditions[wi]}.txt"), dtype=str)

            # simplifying and concatenating labels
            conc_x_labels = []
            for li in range(0, len(labels), 2):
                conc_x_labels.append(labels[li + 1].replace("N_RESOURCES=", "").replace(".0,", " x ") +
                                     labels[li].replace("MIN_RESOURCE_PER_PATCH=", "").replace(".0", "U"))

            print(conc_x_labels)

            # creating y-axis labels
            tuned_env_pattern = os.path.join(data_path, "tuned_env*.json")
            print("Patterns: ", tuned_env_pattern)
            json_files = glob.glob(tuned_env_pattern)
            for json_path in json_files:
                with open(json_path, "r") as f:
                    envvars = json.load(f)
                    y_labels = envvars["DEC_EPSW"]
                    # y_labels.reverse()

            # The others are sparsified for better readability
            if hi == 0:
                plt.yticks([i for i in range(len(y_labels))],
                           y_labels, ha='right', rotation_mode='anchor')
                plt.ylabel(y_titles[wi])
                # for ticki, tick in enumerate(ax[wi, hi].get_yticklabels()):
                #     tick.set_rotation(sparese_y_tick_rotations[ticki])
            else:
                plt.yticks([], [])

            if wi == 0:
                plt.title(f"$N_A$={Ns[ni]}")
                plt.xticks([], [])
            elif wi == len(conditions) - 1:
                # if hi == 0:
                plt.xlabel("Number of Patches")
                plt.xticks([i for i in range(0, len(conc_x_labels))],
                           [conc_x_labels[i].split(' x ')[0] for i in range(0, len(conc_x_labels))],
                           ha='right', rotation_mode='anchor')
                for tick in ax[wi, hi].get_xticklabels():
                    tick.set_rotation(45)
            else:
                plt.xticks([], [])

        else:
            plt.axes(ax[wi, hi])
            plt.yticks([], [])
            plt.xticks([], [])

# #### Interindividual Distance ####
# ## finding data range
# max_data_val_iid = 0
# for hi in range(fig_shape[1]):
#     ni = hi
#     if ni < len(Ns):
#         collapsed_data = np.load(os.path.join(data_path, f"coll_iid_N{Ns[ni]}.npy"))
#         max_data_val_iid = max(np.max(collapsed_data), max_data_val_iid)
#
# wi = 1
# for hi in range(fig_shape[1]):
#     ni =  hi
#     if ni < len(Ns):
#         print(hi, wi, ni)
#         plt.axes(ax[wi, hi])
#         collapsed_data = np.load(os.path.join(data_path, f"coll_iid_N{Ns[ni]}.npy"))
#         # column-wise normalization
#         # for coli in range(collapsed_data.shape[1]):
#         #     print(f"Normalizing column {coli}")
#         #     minval = np.min(collapsed_data[:, coli])
#         #     maxval = np.max(collapsed_data[:, coli])
#         #     collapsed_data[:, coli] = (collapsed_data[:, coli] - minval) / (maxval - minval)
#         agent_dim = 4
#         # cmap = mpl.colors.LinearSegmentedColormap.from_list("", clist_1)
#         im_iid = plt.imshow(collapsed_data, vmin=0, vmax=max_data_val_iid, origin="lower", cmap=cmap)
#
#         # creating x-axis and labels
#         # reading original labels with long names
#         labels = np.loadtxt(os.path.join(data_path, f"coll_eff_N{Ns[ni]}.txt"), dtype=str)
#
#         # simplifying and concatenating labels
#         conc_x_labels = []
#         for li in range(0, len(labels), 2):
#             conc_x_labels.append(labels[li + 1].replace("N_RESOURCES=", "").replace(".0,", " x ") +
#                                  labels[li].replace("MIN_RESOURCE_PER_PATCH=", "").replace(".0", "U"))
#
#         # creating y-axis labels
#         tuned_env_pattern = os.path.join(data_path, "tuned_env*.json")
#         print("Patterns: ", tuned_env_pattern)
#         json_files = glob.glob(tuned_env_pattern)
#         for json_path in json_files:
#             with open(json_path, "r") as f:
#                 envvars = json.load(f)
#                 y_labels = envvars["DEC_EPSW"]
#                 # y_labels.reverse()
#
#         if hi == 0:
#             plt.yticks([i for i in sparse_y_indices],
#                        [y_labels[i] for i in sparse_y_indices], ha='right', rotation_mode='anchor')
#             for ticki, tick in enumerate(ax[wi, hi].get_yticklabels()):
#                 tick.set_rotation(sparese_y_tick_rotations[ticki])
#
#         # The ones in second or other columns have same y axis
#         else:
#             plt.yticks([], [])
#
#         if hi == 0 and wi == fig_shape[0] - 1:
#             plt.xlabel("Number of Patches")
#             plt.xticks([i for i in range(0, len(conc_x_labels), 2)],
#                        [conc_x_labels[i] for i in range(0, len(conc_x_labels), 2)],
#                        ha='right', rotation_mode='anchor')
#         elif wi == fig_shape[0]-1:
#             plt.xticks([i for i in range(0, len(conc_x_labels), 2)],
#                        [conc_x_labels[i] for i in range(0, len(conc_x_labels), 2)],
#                        ha='right', rotation_mode='anchor')
#         for tick in ax[wi, hi].get_xticklabels():
#             tick.set_rotation(45)
#     else:
#         plt.axes(ax[wi, hi])
#         plt.yticks([], [])
#         plt.xticks([], [])
#
# #### Relocation Time ####
# ## finding data range
# max_data_val_rt = 0
# for hi in range(fig_shape[1]):
#     ni = hi
#     if ni < len(Ns):
#         collapsed_data = np.load(os.path.join(data_path, f"coll_reloctime_N{Ns[ni]}.npy"))
#         max_data_val_rt = max(np.max(collapsed_data), max_data_val_rt)
#
# print(f"MAX DATA: ", max_data_val_rt)
#
# wi = 2
# for hi in range(fig_shape[1]):
#     ni = hi
#     if ni < len(Ns):
#         print(hi, wi, ni)
#         plt.axes(ax[wi, hi])
#         collapsed_data = np.load(os.path.join(data_path, f"coll_reloctime_N{Ns[ni]}.npy"))
#         # column-wise normalization
#         # for coli in range(collapsed_data.shape[1]):
#         #     print(f"Normalizing column {coli}")
#         #     minval = np.min(collapsed_data[:, coli])
#         #     maxval = np.max(collapsed_data[:, coli])
#         #     collapsed_data[:, coli] = (collapsed_data[:, coli] - minval) / (maxval - minval)
#         agent_dim = 4
#         # cmap = mpl.colors.LinearSegmentedColormap.from_list("", clist_1)
#         im_reloc = plt.imshow(collapsed_data, vmin=0, vmax=max_data_val_rt, origin="lower", cmap=cmap)
#
#         # creating x-axis and labels
#         # reading original labels with long names
#         labels = np.loadtxt(os.path.join(data_path, f"coll_eff_N{Ns[ni]}.txt"), dtype=str)
#
#         # simplifying and concatenating labels
#         conc_x_labels = []
#         for li in range(0, len(labels), 2):
#             # conc_x_labels.append(labels[li + 1].replace("N_RESOURCES=", "").replace(".0,", " x ") +
#             #                      labels[li].replace("MIN_RESOURCE_PER_PATCH=", "").replace(".0", "U"))
#             conc_x_labels.append(labels[li + 1].replace("N_RESOURCES=", "").replace(".0,", ""))
#
#         # creating y-axis labels
#         tuned_env_pattern = os.path.join(data_path, "tuned_env*.json")
#         print("Patterns: ", tuned_env_pattern)
#         json_files = glob.glob(tuned_env_pattern)
#         for json_path in json_files:
#             with open(json_path, "r") as f:
#                 envvars = json.load(f)
#                 y_labels = envvars["DEC_EPSW"]
#                 # y_labels.reverse()
#
#         if hi == 0 and wi == fig_shape[0] - 1:
#             plt.ylabel("Social Excitability ($\epsilon_w$)")
#             plt.yticks([i for i in range(len(y_labels))], y_labels, ha='right', rotation_mode='anchor')
#             plt.xlabel("Number of Patches")
#             plt.xticks([i for i in range(0, len(conc_x_labels))],
#                        [conc_x_labels[i] for i in range(0, len(conc_x_labels))],
#                        ha='right', rotation_mode='anchor')
#             for tick in ax[wi, hi].get_xticklabels():
#                 tick.set_rotation(45)
#         elif wi == fig_shape[0]-1:
#             plt.xticks([i for i in range(1, len(conc_x_labels), 3)],
#                        [conc_x_labels[i] for i in range(1, len(conc_x_labels), 3)],
#                        ha='right', rotation_mode='anchor')
#             for tick in ax[wi, hi].get_xticklabels():
#                 tick.set_rotation(45)
#         elif hi == 0:
#             plt.yticks([i for i in sparse_y_indices],
#                        [y_labels[i] for i in sparse_y_indices], ha='right', rotation_mode='anchor')
#             for ticki, tick in enumerate(ax[wi, hi].get_yticklabels()):
#                 tick.set_rotation(sparese_y_tick_rotations[ticki])
#         else:
#             plt.yticks([], [])
#
#     else:
#         plt.axes(ax[wi, hi])
#         plt.yticks([], [])
#         plt.xticks([], [])

# Adding the colorbar
# [left, bottom, width, height]
cbaxes = fig.add_axes([0.805, 0.16 + 2*(0.2 + 0.1/5), 0.025, 0.2])
# position for the colorbar
cb = plt.colorbar(im, cax=cbaxes, orientation='vertical')
# cb.ax.set_xticks([min_data_val, (min_data_val+max_data_val)/2, max_data_val])
# cb.ax.xaxis.set_ticks_position("top")
# cb.ax.xaxis.set_label_position('top')
plt.ylabel(colorbar_title)

cbaxes = fig.add_axes([0.805, 0.16 + 1*(0.2 + 0.1/5), 0.025, 0.2])
# position for the colorbar
cb = plt.colorbar(im, cax=cbaxes, orientation='vertical')
# cb.ax.set_xticks([min_data_val, (min_data_val+max_data_val)/2, max_data_val])
# cb.ax.xaxis.set_ticks_position("top")
# cb.ax.xaxis.set_label_position('top')
plt.ylabel(colorbar_title)

cbaxes = fig.add_axes([0.805, 0.16 + 0*(0.2 + 0.1/5), 0.025, 0.2])
# position for the colorbar
cb = plt.colorbar(im, cax=cbaxes, orientation='vertical')
# cb.ax.set_xticks([min_data_val, (min_data_val+max_data_val)/2, max_data_val])
# cb.ax.xaxis.set_ticks_position("top")
# cb.ax.xaxis.set_label_position('top')
plt.ylabel(colorbar_title)

# cbaxes = fig.add_axes([0.805, 0.16 + 1*(0.2 + 0.1/5), 0.025, 0.2])
# # position for the colorbar
# cb = plt.colorbar(im_iid, cax=cbaxes, orientation='vertical')
# # cb.ax.set_xticks([0, max_data_val_iid/2, max_data_val_iid])
# # cb.ax.xaxis.set_ticks_position("top")
# # cb.ax.xaxis.set_label_position('top')
# plt.ylabel('Inter-individual distance')
#
# cbaxes = fig.add_axes([0.805, 0.16 + 0*(0.2 + 0.1/5), 0.025, 0.2])
# # position for the colorbar
# cb = plt.colorbar(im_reloc, cax=cbaxes, orientation='vertical')
# # cb.ax.set_xticks([min_data_val, (min_data_val+max_data_val)/2, max_data_val])
# # cb.ax.xaxis.set_ticks_position("top")
# # cb.ax.xaxis.set_label_position('top')
# plt.ylabel('Relocation Time Fraction')

# fig.colorbar(im, ax=ax.ravel().tolist(), orientation = 'horizontal')
plt.subplots_adjust(hspace=0.1, wspace=0, top=0.8, bottom=0.16, left=0.2, right=0.8)
plt.show()
