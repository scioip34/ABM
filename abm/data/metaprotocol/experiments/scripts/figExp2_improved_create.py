import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
import json
import glob

def norm_between_0_and_1(matrix_to_norm, keep_scale_matrices=None, colum_wise=True):

    normed_matrix = np.zeros_like(matrix_to_norm)
    if keep_scale_matrices is not None:
        if colum_wise:
            for i in range(matrix_to_norm.shape[-1]):
                max_val = np.max([mat[:, i] for mat in keep_scale_matrices])
                min_val = np.min([mat[:, i] for mat in keep_scale_matrices])
                print(max_val, min_val)
                normed_matrix[:, i] = (matrix_to_norm[:, i] - min_val) / (max_val - min_val)
        else:
            max_val = np.max([mat for mat in keep_scale_matrices])
            min_val = np.min([mat for mat in keep_scale_matrices])
            normed_matrix = (matrix_to_norm - min_val) / (max_val - min_val)
    else:
        max_val = np.max(matrix_to_norm)
        min_val = np.min(matrix_to_norm)
        normed_matrix = (matrix_to_norm - min_val) / (max_val - min_val)
    return normed_matrix

# under this path the individual summary statistics are saved
# with _N<number of agents> post tag.
exp_name = "sumfigExp2A_nocoll"
data_path = f"/home/david/Desktop/database/ABMFigData/{exp_name}"

# set of agent numbers to summarize for
Ns = [5, 25, 100]
num_patches = [3, 8, 50]
batch_dim = 0
agent_dim = 3
normalization = "nonorm"  # matrixwise or columnwise or nonorm
colors = ["#f7d634", "#b9ad8f", "#0078de", "#0f4f93"]
lss = ['solid', 'dashed', 'dashdot', 'dotted']
line_th = 3

# fonstsize
FS = {'fontsize': 12}

# figure shape
fig_shape = [len(Ns), len(num_patches)]
fig, ax = plt.subplots(fig_shape[0], fig_shape[1],
                       constrained_layout=True, figsize=(fig_shape[1] * 3, fig_shape[0] * 3),
                       sharex=True, sharey="row")
gs1 = gridspec.GridSpec(fig_shape[0], fig_shape[1])
gs1.update(wspace=0, hspace=0)

for ni in range(fig_shape[0]):
    N = Ns[ni]
    # Loading data
    # try:

    ######
    ###### PATCHY
    ######
    eff_data_patchy = np.load(os.path.join(data_path, f"eff_N{N}_patchy.npy"))

    print("shape:", eff_data_patchy.shape)
    with open(os.path.join(data_path, f"tuned_env_N{N}_patchy.json"), "r") as te:
        epsilons_patchy = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
    with open(os.path.join(data_path, f"tuned_env_N{N}_patchy.json"), "r") as te:
        fovs_patchy = [float(fov) for fov in json.loads(te.read())['AGENT_FOV']]

    #### Non columnwise normalization
    if normalization == "matrixwise" or "nonorm":
        std_eff_patchy = np.std(np.mean(eff_data_patchy, axis=agent_dim), axis=batch_dim)
        mean_eff_patchy_raw = np.mean(np.mean(eff_data_patchy, axis=agent_dim), axis=batch_dim)
        patchy_std_neg_raw = mean_eff_patchy_raw-std_eff_patchy
        patchy_std_pos_raw = mean_eff_patchy_raw+std_eff_patchy
        if normalization == "matrixwise":
            mean_eff_patchy = norm_between_0_and_1(mean_eff_patchy_raw, [mean_eff_patchy_raw, patchy_std_neg_raw, patchy_std_pos_raw], colum_wise=False)
            patchy_std_neg = norm_between_0_and_1(patchy_std_neg_raw, [mean_eff_patchy_raw, patchy_std_neg_raw, patchy_std_pos_raw], colum_wise=False)
            patchy_std_pos = norm_between_0_and_1(patchy_std_pos_raw, [mean_eff_patchy_raw, patchy_std_neg_raw, patchy_std_pos_raw], colum_wise=False)
        else:
            mean_eff_patchy = mean_eff_patchy_raw
            patchy_std_neg = patchy_std_neg_raw
            patchy_std_pos = patchy_std_pos_raw
    elif normalization == "columnwise":
        #### Column-wise normalization
        eff_data_patchy_agent_mean = np.mean(eff_data_patchy, axis=agent_dim)
        eff_data_patchy_normed = np.zeros_like(eff_data_patchy_agent_mean)
        for batch_id in range(eff_data_patchy_agent_mean.shape[0]):
            for eps_i in range(eff_data_patchy_agent_mean.shape[2]):
                print("data to be normed: ", eff_data_patchy_agent_mean[batch_id, :, eps_i].shape)
                eff_data_patchy_normed[batch_id, :, eps_i] = norm_between_0_and_1(eff_data_patchy_agent_mean[batch_id, :, eps_i])
        std_eff_patchy = np.std(eff_data_patchy_normed, axis=0)
        mean_eff_patchy = np.mean(eff_data_patchy_normed, axis=0)
        patchy_std_neg = mean_eff_patchy-std_eff_patchy
        patchy_std_pos = mean_eff_patchy+std_eff_patchy

    plot_min_patchy = np.min([patchy_std_neg, patchy_std_pos])
    plot_max_patchy = np.max([patchy_std_neg, patchy_std_pos])


    ######
    ###### INTERMEDIATE
    ######
    eff_data_intermed = np.load(os.path.join(data_path, f"eff_N{N}_intermed.npy"))
    print(eff_data_intermed.shape)
    with open(os.path.join(data_path, f"tuned_env_N{N}_intermed.json"), "r") as te:
        epsilons_intermed = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
    with open(os.path.join(data_path, f"tuned_env_N{N}_intermed.json"), "r") as te:
        fovs_intermed = [float(fov) for fov in json.loads(te.read())['AGENT_FOV']]

    #### Non colum-wise normalization
    if normalization == "matrixwise" or "nonorm":
        mean_eff_intermed_raw = np.mean(np.mean(eff_data_intermed, axis=agent_dim), axis=batch_dim)
        std_eff_intermed = np.std(np.mean(eff_data_intermed, axis=agent_dim), axis=batch_dim)
        intermed_std_neg_raw = mean_eff_intermed_raw-std_eff_intermed
        intermed_std_pos_raw = mean_eff_intermed_raw+std_eff_intermed
        if normalization == "matrixwise":
            mean_eff_intermed = norm_between_0_and_1(mean_eff_intermed_raw, [mean_eff_intermed_raw, intermed_std_neg_raw, intermed_std_pos_raw], colum_wise=False)
            intermed_std_neg = norm_between_0_and_1(intermed_std_neg_raw, [mean_eff_intermed_raw, intermed_std_neg_raw, intermed_std_pos_raw], colum_wise=False)
            intermed_std_pos = norm_between_0_and_1(intermed_std_pos_raw, [mean_eff_intermed_raw, intermed_std_neg_raw, intermed_std_pos_raw], colum_wise=False)
        else:
            mean_eff_intermed = mean_eff_intermed_raw
            intermed_std_neg = intermed_std_neg_raw
            intermed_std_pos = intermed_std_pos_raw
    elif normalization == "columnwise":
        #### Column-wise normalization
        eff_data_intermed_agent_mean = np.mean(eff_data_intermed, axis=agent_dim)
        eff_data_intermed_normed = np.zeros_like(eff_data_intermed_agent_mean)
        for batch_id in range(eff_data_intermed_agent_mean.shape[0]):
            for eps_i in range(eff_data_intermed_agent_mean.shape[2]):
                print("data to be normed: ", eff_data_intermed_agent_mean[batch_id, :, eps_i].shape)
                eff_data_intermed_normed[batch_id, :, eps_i] = norm_between_0_and_1(eff_data_intermed_agent_mean[batch_id, :, eps_i])
        std_eff_intermed = np.std(eff_data_intermed_normed, axis=0)
        mean_eff_intermed = np.mean(eff_data_intermed_normed, axis=0)
        intermed_std_neg = mean_eff_intermed-std_eff_intermed
        intermed_std_pos = mean_eff_intermed+std_eff_intermed

    plot_min_intermed = np.min([intermed_std_neg, intermed_std_pos])
    plot_max_intermed = np.max([intermed_std_neg, intermed_std_pos])


    ######
    ###### DISTRIBUTED
    ######
    eff_data_dist = np.load(os.path.join(data_path, f"eff_N{N}_dist.npy"))
    with open(os.path.join(data_path, f"tuned_env_N{N}_dist.json"), "r") as te:
        epsilons_dist = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
    with open(os.path.join(data_path, f"tuned_env_N{N}_dist.json"), "r") as te:
        fovs_dist = [float(fov) for fov in json.loads(te.read())['AGENT_FOV']]

    #### Non-columnwise normalization
    if normalization == "matrixwise" or "nonorm":
        mean_eff_dist_raw = np.mean(np.mean(eff_data_dist, axis=agent_dim), axis=batch_dim)
        std_eff_dist = np.std(np.mean(eff_data_dist, axis=agent_dim), axis=batch_dim)
        dist_std_neg_raw = mean_eff_dist_raw-std_eff_dist
        dist_std_pos_raw = mean_eff_dist_raw+std_eff_dist
        if normalization == "matrixwise":
            mean_eff_dist = norm_between_0_and_1(mean_eff_dist_raw, [mean_eff_dist_raw, dist_std_neg_raw, dist_std_pos_raw])
            dist_std_neg = norm_between_0_and_1(dist_std_neg_raw, [mean_eff_dist_raw, dist_std_neg_raw, dist_std_pos_raw])
            dist_std_pos = norm_between_0_and_1(dist_std_pos_raw, [mean_eff_dist_raw, dist_std_neg_raw, dist_std_pos_raw])
        else:
            mean_eff_dist = mean_eff_dist_raw
            dist_std_neg = dist_std_neg_raw
            dist_std_pos = dist_std_pos_raw

    elif normalization == "columnwise":
        #### Column-wise normalization
        eff_data_dist_agent_mean = np.mean(eff_data_dist, axis=agent_dim)
        eff_data_dist_normed = np.zeros_like(eff_data_dist_agent_mean)
        for batch_id in range(eff_data_dist_agent_mean.shape[0]):
            for eps_i in range(eff_data_dist_agent_mean.shape[2]):
                print("data to be normed: ", eff_data_dist_agent_mean[batch_id, :, eps_i].shape)
                eff_data_dist_normed[batch_id, :, eps_i] = norm_between_0_and_1(eff_data_dist_agent_mean[batch_id, :, eps_i])
        std_eff_dist = np.std(eff_data_dist_normed, axis=0)
        mean_eff_dist = np.mean(eff_data_dist_normed, axis=0)
        dist_std_neg = mean_eff_dist-std_eff_dist
        dist_std_pos = mean_eff_dist+std_eff_dist

    plot_min_dist = np.min([dist_std_neg, dist_std_pos])
    plot_max_dist = np.max([dist_std_neg, dist_std_pos])

    # checking if epsilon was consistently changed across epxeriments
    assert epsilons_dist == epsilons_patchy
    assert fovs_patchy == fovs_dist
    epsilons = epsilons_patchy
    fovs = fovs_patchy
    # except:
    #     break

    for eps_i, eps in enumerate(epsilons):
        # patchy
        if fig_shape[0] > 1:
            curax = ax[ni, 0]
        else:
            curax = ax[0]

        plt.axes(curax)
        plt.plot(mean_eff_patchy[:, eps_i], color=colors[eps_i], linewidth=line_th, ls=lss[eps_i], label=f"$\epsilon_w$={eps}")
        if ni == 1:
            plt.ylabel(f"Absolute Search Efficiency\n$N_A$={Ns[ni]}", fontdict=FS)
        else:
            plt.ylabel(f"$N_A$={Ns[ni]}", fontdict=FS)
        if ni == 0:
            plt.title(f"Patchy Environment\n$N_R$={num_patches[0]}", fontdict=FS)
        if ni == len(Ns) - 1:
            sparsing_factor = 4
            plt.xticks([i for i in range(0, len(fovs), sparsing_factor)], [f"{2*round(fovs[i], 1)}$\pi$" for i in range(0, len(fovs), sparsing_factor)], ha='center', rotation_mode='anchor')
            # for ticki, tick in enumerate(curax.get_xticklabels()):
            #     tick.set_rotation(45)
        plt.fill_between([i for i in range(len(fovs))], patchy_std_neg[:, eps_i], patchy_std_pos[:, eps_i], alpha=0.3, color=colors[eps_i])
        # plt.ylim(plot_min_patchy, plot_max_patchy)

        # intermediate
        if fig_shape[0] > 1:
            curax = ax[ni, 1]
        else:
            curax = ax[1]

        plt.axes(curax)
        # plt.yticks([])
        plt.plot(mean_eff_intermed[:, eps_i], color=colors[eps_i], linewidth=line_th, ls=lss[eps_i], label=f"$\epsilon_w$={eps}")
        if ni == 0:
            plt.title(f"Intermediate Environment\n$N_R$={num_patches[1]}", fontdict=FS)
        if ni == len(Ns) - 1:
            sparsing_factor = 10
            plt.xticks([i for i in range(0, len(fovs), sparsing_factor)],
                       [f"{2 * round(fovs[i], 1)}$\pi$" for i in range(0, len(fovs), sparsing_factor)], ha='center',
                       rotation_mode='anchor')
            # for ticki, tick in enumerate(curax.get_xticklabels()):
            #     tick.set_rotation(-45)
            plt.xlabel("Field of View", fontdict=FS)
        plt.fill_between([i for i in range(len(fovs))], intermed_std_neg[:, eps_i], intermed_std_pos[:, eps_i], alpha=0.3, color=colors[eps_i])
        # plt.ylim(plot_min_intermed, plot_max_intermed)

        # distributed
        if fig_shape[0] > 1:
            curax = ax[ni, 2]
        else:
            curax = ax[2]

        plt.axes(curax)
        # plt.yticks([])
        plt.plot(mean_eff_dist[:, eps_i], color=colors[eps_i], linewidth=line_th, ls=lss[eps_i], label=f"$\epsilon_w$={eps}")
        if ni == 0:
            plt.title(f"Uniform Environment\n$N_R$={num_patches[2]}", fontdict=FS)
        if ni == len(Ns) - 1:
            sparsing_factor = 10
            plt.xticks([i for i in range(0, len(fovs), sparsing_factor)], [f"{int(2*round(fovs[i], 1))}$\pi$" for i in range(0, len(fovs), sparsing_factor)], ha='center', rotation_mode='anchor')
            curax.legend()
            # for ticki, tick in enumerate(curax.get_xticklabels()):
            #     tick.set_rotation(-45)
        plt.fill_between([i for i in range(len(fovs))], dist_std_neg[:, eps_i], dist_std_pos[:, eps_i], alpha=0.3, color=colors[eps_i])
        # plt.ylim(plot_min_dist, plot_max_dist)

plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0) #, top=0.8, bottom=0.2, left=0.2, right=0.8)
plt.show()

#     for env_i in range(len(num_patches)):
#         plt.axes(ax[ni, env_i])
#         plt.plot(lines_occ[..., env_i], label=f"occ")
#         plt.plot(lines_noocc[..., env_i], label=f"no occ")
#
#         # Making axis labels
#         if ni == 0 and env_i == 0:
#             plt.ylabel(f"$N_A$={Ns[ni]}\nAbsolute Efficiency")
#             plt.title("Patchy Environment")
#             #plt.yticks([i for i in range(len(y_labels))], y_labels, ha='right', rotation_mode='anchor')
#         elif ni == 0 and env_i == 1:
#             plt.title("Uniform Environment")
#         elif env_i == 0:
#             plt.ylabel(f"$N_A$={Ns[ni]}")
#             # Making sparse ticks
#             # curr_yticks = list(ax[ni, env_i].get_yticks())
#             # sparse_yticks = [curr_yticks[0], curr_yticks[int(len(curr_yticks)/2)], curr_yticks[-1]]
#             # print(sparse_yticks)
#             # parse_ytick_rotations = [0 for i in range(len(sparse_yticks))]
#             # print(parse_ytick_rotations)
#             # parse_ytick_rotations[0] = 45
#             # parse_ytick_rotations[0] = -45
#             # plt.yticks(sparse_yticks, sparse_yticks, ha='right', rotation_mode='anchor')
#             # for ticki, tick in enumerate(ax[ni, env_i].get_yticklabels()):
#             #     tick.set_rotation(parse_ytick_rotations[ticki])
#         if env_i == 1:
#             ax[ni, env_i].yaxis.tick_right()
#         if ni == len(Ns)-1 and env_i == 0:
#             # creating y-axis labels
#             tuned_env_pattern = os.path.join(data_path, "tuned_env*.json")
#             print("Patterns: ", tuned_env_pattern)
#             json_files = glob.glob(tuned_env_pattern)
#             for json_path in json_files:
#                 with open(json_path, "r") as f:
#                     envvars = json.load(f)
#                     x_labels = envvars["DEC_EPSW"]
#             plt.xticks([i for i in range(len(x_labels))], x_labels, ha='right', rotation_mode='anchor')
#             plt.xlabel("Social Excitability ($\epsilon_w$)")
#             for ticki, tick in enumerate(ax[ni, env_i].get_xticklabels()):
#                 tick.set_rotation(45)
#         elif ni == len(Ns)-1 and env_i == 1:
#             x_labels_sparse = [x_labels[i] for i in range(0, len(x_labels), 2)]
#             plt.xticks([i for i in range(0, len(x_labels), 2)], x_labels_sparse, ha='left', rotation_mode='anchor')
#             for ticki, tick in enumerate(ax[ni, env_i].get_xticklabels()):
#                 tick.set_rotation(-45)
#
#
#     plt.legend()
# plt.subplots_adjust(hspace=0, wspace=0, top=0.8, bottom=0.2, left=0.2, right=0.8)
# plt.show()

