import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import json
import glob

# under this path the individual summary statistics are saved
# with _N<number of agents> post tag.
exp_name = "VFprelimsummaryBugfix"
data_path = f"/media/david/DMezeySCIoI/ABMData/{exp_name}"

titles = ["Polarization", "Inter-individual Distance", "Agent Collisions"]
file_names = ["polarization", "meaniid", "aacoll"]
conditions = ["infinite", "walls"]

FOVs = [0.25, 0.5, 0.75, 1]
alphas = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1, 2, 5]  #[0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0]
betas = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1, 2, 5]  #[0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0]
fov_dim = 1
batch_dim = 0

min_data_val = 0
max_data_val = 1

# figure shape
for fovi, fov in enumerate(FOVs):
    fig_shape = [len(titles), len(conditions)]
    fig, ax = plt.subplots(fig_shape[1], fig_shape[0],
                           constrained_layout=True, figsize=(fig_shape[0] * 3, fig_shape[1] * 3),
                           sharey=True, sharex=True)
    plt.suptitle(f"FOV={fov}")
    gs1 = gridspec.GridSpec(fig_shape[0], fig_shape[1])
    gs1.update(wspace=0, hspace=0)
    for wi in range(fig_shape[0]):  # metrics
        metric_min = None
        metric_max = None

        for hi in range(fig_shape[1]):  # conditions (wall vs infinite)
            raw_data = np.load(os.path.join(data_path, file_names[wi] + f"_{conditions[hi]}.npy"))
            print(raw_data.shape)
            if file_names[wi] == "polarization":
                # mean  over all agents and runs which is first and last dimesnions
                mean_data = np.mean(raw_data[:, fovi, ...], axis=0)
            elif file_names[wi] == "meaniid":
                mean_data = np.mean(raw_data[fovi, ...], axis=-1)

            if metric_max is None:
                metric_max = np.max(mean_data)
            else:
                metric_max = max(np.max(mean_data), metric_max)

            if metric_min is None:
                metric_min = np.min(mean_data)
            else:
                metric_min = min(np.min(mean_data), metric_min)


        hi = 0
        for hi in range(fig_shape[1]):  # conditions (wall vs infinite)

            plt.axes(ax[hi, wi])
            if hi == 0:
                plt.title(titles[wi])
            elif hi == fig_shape[1]-1 and wi == 0:
                plt.xticks([i for i in range(len(betas))], betas)
                plt.xlabel(f"Beta0")
            if wi == 0:
                plt.yticks([i for i in range(len(alphas))], alphas)
                plt.ylabel(f"{conditions[hi]}\nAlpha0")


            print(os.path.join(data_path, file_names[wi]+f"_{conditions[hi]}.npy"))
            raw_data = np.load(os.path.join(data_path, file_names[wi]+f"_{conditions[hi]}.npy"))
            print(raw_data.shape)
            if file_names[wi] == "polarization":
                # mean  over all agents and runs which is first and last dimesnions
                mean_data = np.mean(raw_data[:, fovi, ...], axis=0)
            elif file_names[wi] == "meaniid":
                mean_data = np.mean(raw_data[fovi, ...], axis=-1)
            if mean_data.ndim == 2:
                im = plt.imshow(mean_data, vmin=metric_min, vmax=metric_max)
                fig.colorbar(im)
            else:
                pass




            # if ni < len(Ns):
            #     print(hi, wi, ni)
            #     plt.axes(ax[wi, hi])
            #     collapsed_data = np.load(os.path.join(data_path, f"coll_eff_N{Ns[ni]}.npy"))
            #     # column-wise normalization
            #     for coli in range(collapsed_data.shape[1]):
            #         print(f"Normalizing column {coli}")
            #         minval = np.min(collapsed_data[:, coli])
            #         maxval = np.max(collapsed_data[:, coli])
            #         collapsed_data[:, coli] = (collapsed_data[:, coli] - minval) / (maxval - minval)
            #     agent_dim = 4
            #     im = plt.imshow(collapsed_data, vmin=min_data_val, vmax=max_data_val)
            #
            #     # creating x-axis and labels
            #     # reading original labels with long names
            #     labels = np.loadtxt(os.path.join(data_path, f"coll_eff_N{Ns[ni]}.txt"), dtype=str)
            #
            #     # simplifying and concatenating labels
            #     conc_x_labels = []
            #     for li in range(0, len(labels), 2):
            #         conc_x_labels.append(labels[li + 1].replace("N_RESOURCES=", "").replace(".0,", " x ") +
            #                              labels[li].replace("MIN_RESOURCE_PER_PATCH=", "").replace(".0", "U"))
            #
            #     # creating y-axis labels
            #     tuned_env_pattern = os.path.join(data_path, "tuned_env*.json")
            #     print("Patterns: ", tuned_env_pattern)
            #     json_files = glob.glob(tuned_env_pattern)
            #     for json_path in json_files:
            #         with open(json_path, "r") as f:
            #             envvars = json.load(f)
            #             y_labels = envvars["DEC_EPSW"]
            #
            #     # Manually definig the sparified y ticks and their rotations
            #     sparse_y_indices = [0, 2, 4, 6, 8]  # int((len(y_labels)-1)/2), len(y_labels)-1]
            #     y_tick_rotations = [45, 33.75, 22.5, 11.25, 0, -11.25, -22.5, -33.75, -45]
            #     sparese_y_tick_rotations = [45, 0, 0, 0, -45]
            #
            #     # The first plot is detailed shows all ticks and labels
            #     if hi == 0 and wi == 0:
            #         plt.ylabel("Social Excitability ($\epsilon_w$)")
            #         plt.yticks([i for i in range(len(y_labels))], y_labels, ha='right', rotation_mode='anchor')
            #         # for ticki, tick in enumerate(ax[wi, hi].get_yticklabels()):
            #         #     print(ticki, tick)
            #         #     tick.set_rotation(y_tick_rotations[ticki])
            #
            #     # The others are sparsified for better readability
            #     elif hi == 0:
            #         plt.yticks([i for i in sparse_y_indices],
            #                    [y_labels[i] for i in sparse_y_indices], ha='right', rotation_mode='anchor')
            #         for ticki, tick in enumerate(ax[wi, hi].get_yticklabels()):
            #             tick.set_rotation(sparese_y_tick_rotations[ticki])
            #
            #     # The ones in second or other columns have same y axis
            #     else:
            #         plt.yticks([], [])
            #
            #     if hi == 0 and wi == fig_shape[0] - 1:
            #         plt.xlabel("Environment")
            #         plt.xticks([i for i in range(0, len(conc_x_labels), 2)],
            #                    [conc_x_labels[i] for i in range(0, len(conc_x_labels), 2)],
            #                    ha='right', rotation_mode='anchor')
            #     elif wi == fig_shape[0]-1:
            #         plt.xticks([i for i in range(0, len(conc_x_labels), 2)],
            #                    [conc_x_labels[i] for i in range(0, len(conc_x_labels), 2)],
            #                    ha='right', rotation_mode='anchor')
            #     for tick in ax[wi, hi].get_xticklabels():
            #         tick.set_rotation(45)
            # else:
            #     plt.axes(ax[wi, hi])
            #     plt.yticks([], [])
            #     plt.xticks([], [])

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
plt.show()