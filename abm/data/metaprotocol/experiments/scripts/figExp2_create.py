import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
import json
import glob

# under this path the individual summary statistics are saved
# with _N<number of agents> post tag.
exp_name = "sumfigExp2A_nocoll"
data_path = f"/home/david/Desktop/database/{exp_name}"

# set of agent numbers to summarize for
Ns = [25]
num_patches = [3, 8, 50]
batch_dim = 0
agent_dim = 3

# figure shape
fig_shape = [len(Ns), len(num_patches)]
fig, ax = plt.subplots(fig_shape[0], fig_shape[1],
                       constrained_layout=True, figsize=(fig_shape[1] * 3, fig_shape[0] * 3),
                       sharex=False, sharey=True)
gs1 = gridspec.GridSpec(fig_shape[0], fig_shape[1])
gs1.update(wspace=0, hspace=0)

for ni in range(fig_shape[0]):
    N = Ns[ni]
    # Loading data
    try:
        eff_data_patchy = np.load(os.path.join(data_path, f"eff_N{N}_patchy.npy"))
        with open(os.path.join(data_path, f"tuned_env_N{N}_patchy.json"), "r") as te:
            epsilons_patchy = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
        with open(os.path.join(data_path, f"tuned_env_N{N}_patchy.json"), "r") as te:
            fovs_patchy = [float(fov) for fov in json.loads(te.read())['AGENT_FOV']]
        mean_eff_patchy = np.mean(np.mean(eff_data_patchy, axis=agent_dim), axis=batch_dim)
        std_eff_patchy = np.std(np.mean(eff_data_patchy, axis=agent_dim), axis=batch_dim)

        eff_data_intermed = np.load(os.path.join(data_path, f"eff_N{N}_intermed.npy"))
        print(eff_data_intermed.shape)
        with open(os.path.join(data_path, f"tuned_env_N{N}_intermed.json"), "r") as te:
            epsilons_intermed = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
        with open(os.path.join(data_path, f"tuned_env_N{N}_intermed.json"), "r") as te:
            fovs_intermed = [float(fov) for fov in json.loads(te.read())['AGENT_FOV']]
        mean_eff_intermed = np.mean(np.mean(eff_data_intermed, axis=agent_dim), axis=batch_dim)
        std_eff_intermed = np.std(np.mean(eff_data_intermed, axis=agent_dim), axis=batch_dim)

        eff_data_dist = np.load(os.path.join(data_path, f"eff_N{N}_dist.npy"))
        print(eff_data_dist.shape)
        with open(os.path.join(data_path, f"tuned_env_N{N}_dist.json"), "r") as te:
            epsilons_dist = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
        with open(os.path.join(data_path, f"tuned_env_N{N}_dist.json"), "r") as te:
            fovs_dist = [float(fov) for fov in json.loads(te.read())['AGENT_FOV']]
        mean_eff_dist = np.mean(np.mean(eff_data_dist, axis=agent_dim), axis=batch_dim)
        std_eff_dist = np.std(np.mean(eff_data_dist, axis=agent_dim), axis=batch_dim)

        # checking if epsilon was consistently changed across epxeriments
        assert epsilons_dist == epsilons_patchy
        assert fovs_patchy == fovs_dist
        epsilons = epsilons_patchy
        fovs = fovs_patchy
    except:
        break

    for eps_i, eps in enumerate(epsilons):
        # patchy
        if fig_shape[0] > 1:
            curax = ax[ni, 0]
        else:
            curax = ax[0]

        plt.axes(curax)
        plt.plot(mean_eff_patchy[:, eps_i], label=f"$\epsilon$={eps}")
        if ni == 0:
            plt.ylabel(f"$N_A$={Ns[ni]}\nAbsolute Efficiency")
            plt.title("Patchy Environment")
        if ni == len(Ns) - 1:
            plt.ylabel(f"$N_A$={Ns[ni]}")
            sparsing_factor = 4
            plt.xticks([i for i in range(0, len(fovs), sparsing_factor)], [f"{2*round(fovs[i], 1)}$\pi$" for i in range(0, len(fovs), sparsing_factor)], ha='right', rotation_mode='anchor')
            plt.xlabel("Field of View [rad.]")
            for ticki, tick in enumerate(curax.get_xticklabels()):
                tick.set_rotation(45)
        plt.fill_between([i for i in range(len(fovs))], mean_eff_patchy[:, eps_i]-std_eff_patchy[:, eps_i],
                         mean_eff_patchy[:, eps_i]+std_eff_patchy[:, eps_i], alpha=0.3)

        # intermediate
        if fig_shape[0] > 1:
            curax = ax[ni, 1]
        else:
            curax = ax[1]

        plt.axes(curax)
        plt.yticks([])
        plt.plot(mean_eff_intermed[:, eps_i], label=f"$\epsilon$={eps}")
        if ni == 0:
            plt.title("Intermediate Environment")
        if ni == len(Ns) - 1:
            sparsing_factor = 10
            plt.xticks([i for i in range(0, len(fovs), sparsing_factor)],
                       [f"{2 * round(fovs[i], 1)}$\pi$" for i in range(0, len(fovs), sparsing_factor)], ha='left',
                       rotation_mode='anchor')
            for ticki, tick in enumerate(curax.get_xticklabels()):
                tick.set_rotation(-45)
        plt.fill_between([i for i in range(len(fovs))], mean_eff_intermed[:, eps_i] - std_eff_intermed[:, eps_i],
                         mean_eff_intermed[:, eps_i] + std_eff_intermed[:, eps_i], alpha=0.3)

        # distributed
        if fig_shape[0] > 1:
            curax = ax[ni, 2]
        else:
            curax = ax[2]

        plt.axes(curax)
        curax.yaxis.tick_right()
        plt.plot(mean_eff_dist[:, eps_i], label=f"$\epsilon$={eps}")
        if ni == 0:
            plt.title("Uniform Environment")
        if ni == len(Ns) - 1:
            sparsing_factor = 10
            plt.xticks([i for i in range(0, len(fovs), sparsing_factor)], [f"{2*round(fovs[i], 1)}$\pi$" for i in range(0, len(fovs), sparsing_factor)], ha='left', rotation_mode='anchor')
            for ticki, tick in enumerate(curax.get_xticklabels()):
                tick.set_rotation(-45)
        plt.fill_between([i for i in range(len(fovs))], mean_eff_dist[:, eps_i]-std_eff_dist[:, eps_i],
                         mean_eff_dist[:, eps_i]+std_eff_dist[:, eps_i], alpha=0.3)

plt.legend()
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

