import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
import json
import glob

def norm_between_0_and_1(matrix_to_norm, keep_scale_matrices=None):

    normed_matrix = np.zeros_like(matrix_to_norm)
    if keep_scale_matrices is not None:
        for i in range(matrix_to_norm.shape[-1]):
            max_val = np.max([mat[:, i] for mat in keep_scale_matrices])
            min_val = np.min([mat[:, i] for mat in keep_scale_matrices])
            print(max_val, min_val)
            normed_matrix[:, i] = (matrix_to_norm[:, i] - min_val) / (max_val - min_val)
    else:
        max_val = np.max(matrix_to_norm)
        min_val = np.min(matrix_to_norm)
        normed_matrix = (matrix_to_norm - min_val) / (max_val - min_val)
    return normed_matrix

colors = ['#15294a', '#0f437e', '#0c5eb0', '#017be5', '#8399d0', '#ceb86d', '#f9d83e', '#fcf9f3']
## Made with:
# import matplotlib.pyplot as plt
# import colorcet as cc
# from matplotlib.colors import rgb2hex
# cmap = cc.cm.CET_CBL2
# colors_np = cmap([(1/8) + i*(1/8) for i in range(8)])
# colors = [rgb2hex(colors_np[i, 0:3]) for i in range(colors_np.shape[0])]

FS = {'fontsize': 12}

# under this path the individual summary statistics are saved
# with _N<number of agents> post tag.
exp_name = "sumfigExp3B"
data_path = f"/home/david/Desktop/database/{exp_name}"

# set of agent numbers to summarize for
Ns = ["Idealized", "Collision", "Collision + Occlusion"]
num_patches = [3, 8, 30]
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
    N = 50
    # Loading data
    # try:
    eff_data_patchy_nc = np.load(os.path.join(data_path, f"eff_N{N}_Patchy_NoColl.npy"))
    eff_data_patchy_agent_mean_nc = np.mean(eff_data_patchy_nc, axis=agent_dim)
    eff_data_patchy_normed_nc = np.zeros_like(eff_data_patchy_agent_mean_nc)
    for batch_id in range(eff_data_patchy_agent_mean_nc.shape[0]):
        for eps_i in range(eff_data_patchy_agent_mean_nc.shape[1]):
            print("data to be normed: ", eff_data_patchy_agent_mean_nc[batch_id, eps_i, :].shape)
            # eff_data_patchy_normed[batch_id, eps_i, :] = norm_between_0_and_1(eff_data_patchy_agent_mean[batch_id, eps_i, :])
            eff_data_patchy_normed_nc[batch_id, eps_i, :] = eff_data_patchy_agent_mean_nc[batch_id, eps_i, :]
    print("shape:", eff_data_patchy_nc.shape)
    with open(os.path.join(data_path, f"tuned_env_N{N}_Patchy_NoColl.json"), "r") as te:
        epsilons_patchy_nc = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
    with open(os.path.join(data_path, f"tuned_env_N{N}_Patchy_NoColl.json"), "r") as te:
        fovs_patchy_nc = [float(fov) for fov in json.loads(te.read())['RADIUS_RESOURCE']]
    std_eff_patchy_nc = np.std(eff_data_patchy_normed_nc, axis=0)
    mean_eff_patchy_nc = np.mean(eff_data_patchy_normed_nc, axis=0)
    patchy_std_neg_nc = mean_eff_patchy_nc-std_eff_patchy_nc
    patchy_std_pos_nc = mean_eff_patchy_nc+std_eff_patchy_nc

    eff_data_patchy = np.load(os.path.join(data_path, f"eff_N{N}_Patchy_Coll.npy"))
    eff_data_patchy_agent_mean = np.mean(eff_data_patchy, axis=agent_dim)
    eff_data_patchy_normed = np.zeros_like(eff_data_patchy_agent_mean)
    for batch_id in range(eff_data_patchy_agent_mean.shape[0]):
        for eps_i in range(eff_data_patchy_agent_mean.shape[1]):
            print("data to be normed: ", eff_data_patchy_agent_mean[batch_id, eps_i, :].shape)
            # eff_data_patchy_normed[batch_id, eps_i, :] = norm_between_0_and_1(eff_data_patchy_agent_mean[batch_id, eps_i, :])
            eff_data_patchy_normed[batch_id, eps_i, :] = eff_data_patchy_agent_mean[batch_id, eps_i, :]
    print("shape:", eff_data_patchy.shape)
    with open(os.path.join(data_path, f"tuned_env_N{N}_Patchy_Coll.json"), "r") as te:
        epsilons_patchy = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
    with open(os.path.join(data_path, f"tuned_env_N{N}_Patchy_Coll.json"), "r") as te:
        fovs_patchy = [float(fov) for fov in json.loads(te.read())['RADIUS_RESOURCE']]
    std_eff_patchy = np.std(eff_data_patchy_normed, axis=0)
    mean_eff_patchy = np.mean(eff_data_patchy_normed, axis=0)
    patchy_std_neg = mean_eff_patchy-std_eff_patchy
    patchy_std_pos = mean_eff_patchy+std_eff_patchy

    eff_data_patchy_cocc = np.load(os.path.join(data_path, f"eff_N{N}_Patchy_CollOcc.npy"))
    eff_data_patchy_cocc_agent_mean = np.mean(eff_data_patchy_cocc, axis=agent_dim)
    eff_data_patchy_cocc_normed = np.zeros_like(eff_data_patchy_cocc_agent_mean)
    for batch_id in range(eff_data_patchy_cocc_agent_mean.shape[0]):
        for eps_i in range(eff_data_patchy_cocc_agent_mean.shape[1]):
            print("data to be normed: ", eff_data_patchy_cocc_agent_mean[batch_id, eps_i, :].shape)
            # eff_data_patchy_cocc_normed[batch_id, eps_i, :] = norm_between_0_and_1(eff_data_patchy_cocc_agent_mean[batch_id, eps_i, :])
            eff_data_patchy_cocc_normed[batch_id, eps_i, :] = eff_data_patchy_cocc_agent_mean[batch_id, eps_i, :]
    print("shape:", eff_data_patchy_cocc.shape)
    with open(os.path.join(data_path, f"tuned_env_N{N}_Patchy_Coll.json"), "r") as te:
        epsilons_patchy = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
    with open(os.path.join(data_path, f"tuned_env_N{N}_Patchy_Coll.json"), "r") as te:
        fovs_patchy = [float(fov) for fov in json.loads(te.read())['RADIUS_RESOURCE']]
    std_eff_patchy_cocc = np.std(eff_data_patchy_cocc_normed, axis=0)
    mean_eff_patchy_cocc = np.mean(eff_data_patchy_cocc_normed, axis=0)
    patchy_std_neg_cocc = mean_eff_patchy_cocc - std_eff_patchy_cocc
    patchy_std_pos_cocc = mean_eff_patchy_cocc + std_eff_patchy_cocc

    eff_data_intermed_nc = np.load(os.path.join(data_path, f"eff_N{N}_Intermed_NoColl.npy"))
    eff_data_intermed_agent_mean_nc = np.mean(eff_data_intermed_nc, axis=agent_dim)
    eff_data_intermed_normed_nc = np.zeros_like(eff_data_intermed_agent_mean_nc)
    for batch_id in range(eff_data_intermed_agent_mean_nc.shape[0]):
        for eps_i in range(eff_data_intermed_agent_mean_nc.shape[1]):
            print("data to be normed: ", eff_data_intermed_agent_mean_nc[batch_id, eps_i, :].shape)
            # eff_data_intermed_normed[batch_id, eps_i, :] = norm_between_0_and_1(eff_data_intermed_agent_mean[batch_id, eps_i, :])
            eff_data_intermed_normed_nc[batch_id, eps_i, :] = eff_data_intermed_agent_mean_nc[batch_id, eps_i, :]
    print(eff_data_intermed_nc.shape)
    with open(os.path.join(data_path, f"tuned_env_N{N}_Intermed_NoColl.json"), "r") as te:
        epsilons_intermed = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
    with open(os.path.join(data_path, f"tuned_env_N{N}_Intermed_NoColl.json"), "r") as te:
        fovs_intermed = [float(fov) for fov in json.loads(te.read())['RADIUS_RESOURCE']]
    std_eff_intermed_nc = np.std(eff_data_intermed_normed_nc, axis=0)
    mean_eff_intermed_nc = np.mean(eff_data_intermed_normed_nc, axis=0)
    intermed_std_neg_nc = mean_eff_intermed_nc - std_eff_intermed_nc
    intermed_std_pos_nc = mean_eff_intermed_nc + std_eff_intermed_nc

    eff_data_intermed = np.load(os.path.join(data_path, f"eff_N{N}_Intermed_Coll.npy"))
    eff_data_intermed_agent_mean = np.mean(eff_data_intermed, axis=agent_dim)
    eff_data_intermed_normed = np.zeros_like(eff_data_intermed_agent_mean)
    for batch_id in range(eff_data_intermed_agent_mean.shape[0]):
        for eps_i in range(eff_data_intermed_agent_mean.shape[1]):
            print("data to be normed: ", eff_data_intermed_agent_mean[batch_id, eps_i, :].shape)
            #eff_data_intermed_normed[batch_id, eps_i, :] = norm_between_0_and_1(eff_data_intermed_agent_mean[batch_id, eps_i, :])
            eff_data_intermed_normed[batch_id, eps_i, :] = eff_data_intermed_agent_mean[batch_id, eps_i, :]
    print(eff_data_intermed.shape)
    with open(os.path.join(data_path, f"tuned_env_N{N}_Intermed_Coll.json"), "r") as te:
        epsilons_intermed = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
    with open(os.path.join(data_path, f"tuned_env_N{N}_Intermed_Coll.json"), "r") as te:
        fovs_intermed = [float(fov) for fov in json.loads(te.read())['RADIUS_RESOURCE']]
    std_eff_intermed = np.std(eff_data_intermed_normed, axis=0)
    mean_eff_intermed = np.mean(eff_data_intermed_normed, axis=0)
    intermed_std_neg = mean_eff_intermed-std_eff_intermed
    intermed_std_pos = mean_eff_intermed+std_eff_intermed

    eff_data_intermed_cocc = np.load(os.path.join(data_path, f"eff_N{N}_Intermed_CollOcc.npy"))
    eff_data_intermed_cocc_agent_mean = np.mean(eff_data_intermed_cocc, axis=agent_dim)
    eff_data_intermed_cocc_normed = np.zeros_like(eff_data_intermed_cocc_agent_mean)
    for batch_id in range(eff_data_intermed_cocc_agent_mean.shape[0]):
        for eps_i in range(eff_data_intermed_cocc_agent_mean.shape[1]):
            print("data to be normed: ", eff_data_intermed_cocc_agent_mean[batch_id, eps_i, :].shape)
            #eff_data_intermed_cocc_normed[batch_id, eps_i, :] = norm_between_0_and_1(eff_data_intermed_cocc_agent_mean[batch_id, eps_i, :])
            eff_data_intermed_cocc_normed[batch_id, eps_i, :] = eff_data_intermed_cocc_agent_mean[batch_id, eps_i, :]
    print(eff_data_intermed_cocc.shape)
    with open(os.path.join(data_path, f"tuned_env_N{N}_Intermed_Coll.json"), "r") as te:
        epsilons_intermed = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
    with open(os.path.join(data_path, f"tuned_env_N{N}_Intermed_Coll.json"), "r") as te:
        fovs_intermed = [float(fov) for fov in json.loads(te.read())['RADIUS_RESOURCE']]
    std_eff_intermed_cocc = np.std(eff_data_intermed_cocc_normed, axis=0)
    mean_eff_intermed_cocc = np.mean(eff_data_intermed_cocc_normed, axis=0)
    intermed_std_neg_cocc = mean_eff_intermed_cocc - std_eff_intermed_cocc
    intermed_std_pos_cocc = mean_eff_intermed_cocc + std_eff_intermed_cocc


    eff_data_dist_nc = np.load(os.path.join(data_path, f"eff_N{N}_Dist_NoColl.npy"))
    eff_data_dist_agent_mean_nc = np.mean(eff_data_dist_nc, axis=agent_dim)
    eff_data_dist_normed_nc = np.zeros_like(eff_data_dist_agent_mean_nc)
    for batch_id in range(eff_data_dist_agent_mean_nc.shape[0]):
        for eps_i in range(eff_data_dist_agent_mean_nc.shape[1]):
            print("data to be normed: ", eff_data_dist_agent_mean_nc[batch_id, eps_i, :].shape)
            # eff_data_dist_normed[batch_id, eps_i, :] = norm_between_0_and_1(eff_data_dist_agent_mean[batch_id, eps_i, :])
            eff_data_dist_normed_nc[batch_id, eps_i, :] = eff_data_dist_agent_mean_nc[batch_id, eps_i, :]
    print("shape:", eff_data_dist_nc.shape)
    with open(os.path.join(data_path, f"tuned_env_N{N}_Dist_NoColl.json"), "r") as te:
        epsilons_dist_nc = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
    with open(os.path.join(data_path, f"tuned_env_N{N}_Dist_NoColl.json"), "r") as te:
        fovs_dist_nc = [float(fov) for fov in json.loads(te.read())['RADIUS_RESOURCE']]
    std_eff_dist_nc = np.std(eff_data_dist_normed_nc, axis=0)
    mean_eff_dist_nc = np.mean(eff_data_dist_normed_nc, axis=0)
    dist_std_neg_nc = mean_eff_dist_nc-std_eff_dist_nc
    dist_std_pos_nc = mean_eff_dist_nc+std_eff_dist_nc

    eff_data_dist = np.load(os.path.join(data_path, f"eff_N{N}_Dist_Coll.npy"))
    eff_data_dist_agent_mean = np.mean(eff_data_dist, axis=agent_dim)
    eff_data_dist_normed = np.zeros_like(eff_data_dist_agent_mean)
    for batch_id in range(eff_data_dist_agent_mean.shape[0]):
        for eps_i in range(eff_data_dist_agent_mean.shape[1]):
            print("data to be normed: ", eff_data_dist_agent_mean[batch_id, eps_i, :].shape)
            # eff_data_dist_normed[batch_id, eps_i, :] = norm_between_0_and_1(eff_data_dist_agent_mean[batch_id, eps_i, :])
            eff_data_dist_normed[batch_id, eps_i, :] = eff_data_dist_agent_mean[batch_id, eps_i, :]
    print("shape:", eff_data_dist.shape)
    with open(os.path.join(data_path, f"tuned_env_N{N}_Dist_Coll.json"), "r") as te:
        epsilons_dist = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
    with open(os.path.join(data_path, f"tuned_env_N{N}_Dist_Coll.json"), "r") as te:
        fovs_dist = [float(fov) for fov in json.loads(te.read())['RADIUS_RESOURCE']]
    std_eff_dist = np.std(eff_data_dist_normed, axis=0)
    mean_eff_dist = np.mean(eff_data_dist_normed, axis=0)
    dist_std_neg = mean_eff_dist-std_eff_dist
    dist_std_pos = mean_eff_dist+std_eff_dist

    eff_data_dist_cocc = np.load(os.path.join(data_path, f"eff_N{N}_Dist_CollOcc.npy"))
    eff_data_dist_cocc_agent_mean = np.mean(eff_data_dist_cocc, axis=agent_dim)
    eff_data_dist_cocc_normed = np.zeros_like(eff_data_dist_cocc_agent_mean)
    for batch_id in range(eff_data_dist_cocc_agent_mean.shape[0]):
        for eps_i in range(eff_data_dist_cocc_agent_mean.shape[1]):
            print("data to be normed: ", eff_data_dist_cocc_agent_mean[batch_id, eps_i, :].shape)
            # eff_data_dist_cocc_normed[batch_id, eps_i, :] = norm_between_0_and_1(eff_data_dist_cocc_agent_mean[batch_id, eps_i, :])
            eff_data_dist_cocc_normed[batch_id, eps_i, :] = eff_data_dist_cocc_agent_mean[batch_id, eps_i, :]
    print("shape:", eff_data_dist_cocc.shape)
    with open(os.path.join(data_path, f"tuned_env_N{N}_Dist_Coll.json"), "r") as te:
        epsilons_dist = [float(eps) for eps in json.loads(te.read())['DEC_EPSW']]
    with open(os.path.join(data_path, f"tuned_env_N{N}_Dist_Coll.json"), "r") as te:
        fovs_dist = [float(fov) for fov in json.loads(te.read())['RADIUS_RESOURCE']]
    std_eff_dist_cocc = np.std(eff_data_dist_cocc_normed, axis=0)
    mean_eff_dist_cocc = np.mean(eff_data_dist_cocc_normed, axis=0)
    dist_std_neg_cocc = mean_eff_dist_cocc-std_eff_dist_cocc
    dist_std_pos_cocc = mean_eff_dist_cocc+std_eff_dist_cocc


    # checking if epsilon was consistently changed across epxeriments
    assert epsilons_intermed == epsilons_patchy
    assert fovs_patchy == fovs_intermed
    epsilons = epsilons_patchy
    fovs = [fov / 10 for fov in fovs_patchy]
    # except:
    #     break

    for eps_i, eps in enumerate(epsilons):
        # patchy
        if fig_shape[0] > 1:
            curax = ax[ni, 0]
        else:
            curax = ax[0]

        plt.axes(curax)
        if ni == 0:
            plt.plot(mean_eff_patchy_nc[eps_i, :], c=colors[eps_i], linewidth=3)
            plt.xticks([])
        elif ni == 1:
            plt.plot(mean_eff_patchy[eps_i, :], label=f"$\epsilon$={eps}", c=colors[eps_i], linewidth=3)
            plt.legend(loc="upper left")
        elif ni == 2:
            plt.plot(mean_eff_patchy_cocc[eps_i, :], c=colors[eps_i], linewidth=3)
        if ni == 0:
            plt.title("Patchy Environment", fontdict=FS)
        plt.ylabel(f"{Ns[ni]}", fontdict=FS)
        if ni == 2:
            sparsing_factor = 2
            plt.xticks([i for i in range(0, len(fovs), sparsing_factor)], [f"{fovs[i]}" for i in range(0, len(fovs), sparsing_factor)], ha='center', rotation_mode='anchor')
            for ticki, tick in enumerate(curax.get_xticklabels()):
                tick.set_rotation(0)

        print("sh", patchy_std_neg.shape)
        if ni == 0:
            plt.fill_between([i for i in range(len(fovs))], patchy_std_neg_nc[eps_i, :], patchy_std_pos_nc[eps_i, :],
                             alpha=0.3, color=colors[eps_i])
        elif ni == 1:
            plt.fill_between([i for i in range(len(fovs))], patchy_std_neg[eps_i, :], patchy_std_pos[eps_i, :], alpha=0.3, color=colors[eps_i])
        elif ni == 2:
            plt.fill_between([i for i in range(len(fovs))], patchy_std_neg_cocc[eps_i, :], patchy_std_pos_cocc[eps_i, :],
                             alpha=0.3, color=colors[eps_i])

        # intermediate
        if fig_shape[0] > 1:
            curax = ax[ni, 1]
        else:
            curax = ax[1]

        plt.axes(curax)
        # plt.yticks([])
        if ni == 0:
            plt.title("Intermediate Environment", fontdict=FS)
            plt.plot(mean_eff_intermed_nc[eps_i, :], label=f"$\epsilon$={eps}", c=colors[eps_i], linewidth=3)
            plt.xticks([])
        elif ni == 1:
            plt.plot(mean_eff_intermed[eps_i, :], c=colors[eps_i], linewidth=3)
        elif ni == 2:
            plt.plot(mean_eff_intermed_cocc[eps_i, :], c=colors[eps_i], linewidth=3)
            sparsing_factor = 1
            plt.xticks([i for i in range(0, len(fovs), sparsing_factor)], [f"{fovs[i]}" for i in range(0, len(fovs), sparsing_factor)], ha='right', rotation_mode='anchor')
            plt.xlabel("Patch to Agent Size", fontdict=FS)
            for ticki, tick in enumerate(curax.get_xticklabels()):
                tick.set_rotation(45)

        if ni == 0:
            plt.fill_between([i for i in range(len(fovs))], intermed_std_neg_nc[eps_i, :], intermed_std_pos_nc[eps_i, :],
                             alpha=0.3, color=colors[eps_i])
        elif ni == 1:
            plt.fill_between([i for i in range(len(fovs))], intermed_std_neg[eps_i, :], intermed_std_pos[eps_i, :], alpha=0.3, color=colors[eps_i])
        elif ni == 2:
            plt.fill_between([i for i in range(len(fovs))], intermed_std_neg_cocc[eps_i, :], intermed_std_pos_cocc[eps_i, :], alpha=0.3, color=colors[eps_i])


        # distributed
        if fig_shape[0] > 1:
            curax = ax[ni, 2]
        else:
            curax = ax[2]

        plt.axes(curax)
        # plt.yticks([])
        if ni == 0:
            plt.title("Uniform Environment", fontdict=FS)
            plt.plot(mean_eff_dist_nc[eps_i, :], label=f"$\epsilon$={eps}",  c=colors[eps_i], linewidth=3)
            plt.xticks([])
        elif ni == 1:
            plt.plot(mean_eff_dist[eps_i, :],c=colors[eps_i], linewidth=3)
        elif ni == 2:
            plt.plot(mean_eff_dist_cocc[eps_i, :], c=colors[eps_i], linewidth=3)
        if ni == 2:
            sparsing_factor = 2
            plt.xticks([i for i in range(0, len(fovs), sparsing_factor)], [f"{fovs[i]}" for i in range(0, len(fovs), sparsing_factor)], ha='center', rotation_mode='anchor')
            for ticki, tick in enumerate(curax.get_xticklabels()):
                tick.set_rotation(0)

        if ni == 0:
            plt.fill_between([i for i in range(len(fovs))], dist_std_neg_nc[eps_i, :], dist_std_pos_nc[eps_i, :], alpha=0.3,
                             color=colors[eps_i])
        elif ni == 1:
            plt.fill_between([i for i in range(len(fovs))], dist_std_neg[eps_i, :], dist_std_pos[eps_i, :], alpha=0.3, color=colors[eps_i])
        elif ni == 2:
            plt.fill_between([i for i in range(len(fovs))], dist_std_neg_cocc[eps_i, :], dist_std_pos_cocc[eps_i, :], alpha=0.3, color=colors[eps_i])


        # plt.axes(curax)
        # curax.yaxis.tick_right()
        # plt.plot(mean_eff_dist[:, eps_i], label=f"$\epsilon$={eps}")
        # if ni == 0:
        #     plt.title("Uniform Environment")
        # if ni == len(Ns) - 1:
        #     sparsing_factor = 10
        #     plt.xticks([i for i in range(0, len(fovs), sparsing_factor)], [f"{2*round(fovs[i], 1)}$\pi$" for i in range(0, len(fovs), sparsing_factor)], ha='left', rotation_mode='anchor')
        #     for ticki, tick in enumerate(curax.get_xticklabels()):
        #         tick.set_rotation(-45)
        # plt.fill_between([i for i in range(len(fovs))], dist_std_neg[:, eps_i], dist_std_pos[:, eps_i], alpha=0.3)
plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()

