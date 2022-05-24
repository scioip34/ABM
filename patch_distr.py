import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json

# TODO organize data in dataframe
# TODO update script

# def plot_2d_hist_centers(x_values, y_values, folderpath,i):
#     fig, ax = plt.subplots()
#     h = ax.hist2d(x_values, y_values, bins=[int(ENV_WIDTH/10),int(ENV_HEIGHT/10)], density=True, range=[[0, ENV_WIDTH], [0, ENV_HEIGHT]])
#
#     fig.colorbar(h[3], ax=ax, label='Probability density')
#     fig = plt.xlabel('location x')
#     fig = plt.ylabel('location y')
#     fig = plt.title('R_R={} N_R={} batches={}'.format(round(RADIUS_RESOURCE[i],2),N_RESOURCES,num_batches))
#     fig = plt.tight_layout()
#     fig = plt.gca().set_aspect('equal', adjustable='box')
#
#     # # folling two lines for adding rectangle in plot representing arena borders
#     # import matplotlib.patches as patches
#     # rect = patches.Rectangle((0, 0), 500, 500, linewidth=3, edgecolor='orange', facecolor='none')
#     # ax.add_patch(rect)
#
#     plt.savefig(folderpath +'/patch_distr_R_R_{}_N_R_{}_batches_{}.pdf'.format(round(RADIUS_RESOURCE[i],2),N_RESOURCES,num_batches))
#     return
#
# def loop_params_batches_centers(posx, posy, nr_of_radii, num_batches, window_pad):
#     for i in range(0,nr_of_radii):
#         x_values = []
#         y_values = []
#         for j in range(0,num_batches):
#             x_values = np.append(x_values, posx[j,i,:,0] - window_pad)
#             y_values = np.append(y_values, posy[j,i,:,0] - window_pad)
#
#         # making sure that no drawn location is outside the arena borders
#         max_x_values = np.max(x_values)
#         max_y_values = np.max(y_values)
#         if (max_x_values > ENV_WIDTH) | (max_y_values > ENV_HEIGHT):
#             sys.exit('At least one patch location is outside the arena borders.')
#
#         plot_2d_hist_centers(x_values, y_values, folderpath, i)
#     return
# loop_params_batches_centers(posx, posy, nr_of_radii, num_batches, window_pad)


# for loading data from /summary
filepath = 'ABM/abm/data/simulation_data/patch_place_distr'
# for storing plots of patch distribution
folderpath = filepath + '/summary/patch_distr'
window_pad = 30 # hardcoded

def plot_2d_hist(x_values, y_values, folderpath,i):
    fig, ax = plt.subplots()
    h = ax.hist2d(x_values, y_values, bins=[int(ENV_WIDTH/10),int(ENV_HEIGHT/10)], density=True, range=[[0, ENV_WIDTH], [0, ENV_HEIGHT]])

    fig.colorbar(h[3], ax=ax, label='Probability density')
    fig = plt.xlabel('location x')
    fig = plt.ylabel('location y')
    fig = plt.title('R_R={} N_R={} batches={}'.format(round(RADIUS_RESOURCE[i],2),N_RESOURCES,num_batches))
    fig = plt.tight_layout()
    fig = plt.gca().set_aspect('equal', adjustable='box')

    # # folling two lines for adding rectangle in plot representing arena borders
    # import matplotlib.patches as patches
    # rect = patches.Rectangle((0, 0), 500, 500, linewidth=3, edgecolor='orange', facecolor='none')
    # ax.add_patch(rect)

    plt.savefig(folderpath +'/patch_distr_R_R_{}_N_R_{}_batches_{}.pdf'.format(round(RADIUS_RESOURCE[i],2),N_RESOURCES,num_batches))
    return

def loop_params_batches(posx, posy, nr_of_radii, num_batches, window_pad):
    for i in range(0,nr_of_radii):
        x_values = []
        y_values = []
        for j in range(0,num_batches):
            x_values = np.append(x_values, posx[j,i,:,0] - window_pad)
            y_values = np.append(y_values, posy[j,i,:,0] - window_pad)

        # making sure that no drawn location is outside the arena borders
        max_x_values = np.max(x_values)
        max_y_values = np.max(y_values)
        if (max_x_values > ENV_WIDTH) | (max_y_values > ENV_HEIGHT):
            sys.exit('At least one patch location is outside the arena borders.')

        plot_2d_hist(x_values, y_values, folderpath, i)
    return


# load data from /summary
data = np.load(filepath + '/summary/resource_summary.npz')
# create file for storing plots
if os.path.exists(folderpath) == False:
    os.mkdir(folderpath)

# read out parameters from .json-files for printing/showing them in plots and filesnames
# fixed_env
# abm/data/simulation_data/experiment_name/summary/fixed_env.json
with open(filepath + '/summary/fixed_env.json', 'r') as k:
     fixed_env =  json.loads(k.read())

ENV_WIDTH = fixed_env['ENV_WIDTH']
ENV_HEIGHT = fixed_env['ENV_HEIGHT']

# tuned_env
# abm/data/simulation_data/experiment_name/summary/tuned_env.json
with open(filepath + '/summary/tuned_env.json', 'r') as k:
     tuned_env =  json.loads(k.read())

RADIUS_RESOURCE = tuned_env['RADIUS_RESOURCE']

posx = data["posx"]
posy = data["posy"]

num_batches, nr_of_radii, N_RESOURCES , T = posx.shape

loop_params_batches(posx, posy, nr_of_radii, num_batches, window_pad)
