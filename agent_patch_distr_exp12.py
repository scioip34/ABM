import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import json
import math
from abm.replay.replay import ExperimentReplay

########################### DOCUMENTATION ######################################
#
# What agent_patch_distr_exp12.py does:
# When doing an experiment with different values for RADIUS_RESOURCE the script creates
# for each value a 2D heatmap plot of the exploitable area (area covered by patches)
# and a plot of the locations of the centers of the patches for testing reasons. The
# plots contain the parameter values in the respective filename and the plot title itself.
# Furthermore normalized_pixel_array.npz is created for doing quick adaptations in the plots.

#
# How to use the scripts_
# - run nb_exp12.py (define EXPERIMENT_NAME, e.g. patch_place_distr)
# - adapt nb_exp12.py by varying `num_patches`. Define at least two values, e.g. `num_patches = [1, 5]`
#   agent_patch_distr_exp12.py fails if only one value is given!
# - adapt filepath in agent_patch_distr_exp12.py (e.g. filepath = 'home/ABM/abm/data/simulation_data/patch_place_distr')
# - run agent_patch_distr_exp12.py (the directory where to patch_exploitable_area.py is arbitrary)
# - a folder patch_place_distr/summary/agent_patch_distr will be created containing the plots named after
#   the the different parameters used and the .npz-files for redoing the plots
#
################################################################################

# TODO see modifiy_pixel_array_circle()

# for loading data from /summary
filepath = 'ABM/abm/data/simulation_data/agent_patch_place_distr_01/'
# for storing plots of patch distribution
folderpath = filepath + '/summary/agent_patch_distr'
window_pad = 30 # hardcoded

# see "Equation of a circle" in  https://en.wikipedia.org/wiki/Circle#Equations
def modifiy_pixel_array_circle(pixel_array, m_x, m_y, radius):
    """ Modifies a 2D pixel array such that in the area of a circle with given
    position (center of the circle) and radius a one is added.

    Parameters
    ----------
    pixel_array : 2d array
    m_x : x-location of circle.
    m_y : y-location of circle.
    radius : radius of circle.

    Returns: modified pixel_array
    """
    ENV_HEIGHT, ENV_WIDTH = np.shape(pixel_array)
    # TODO Check if arena consists of pixel id = 0 until pixel id = ENV_WIDTH-1 or
    #      id = 1 until pixel id = ENV_WIDTH (and respectively for ENV_HEIGHT)
    #      => If necessary adapt range().

    left_end_circle = int(m_x - radius)
    right_end_circle = int(m_x + radius + 1)
    upper_end_circle = int(m_y - radius)
    lower_end_circle = int(m_y + radius + 1)

    # adapt _end_circle such that no pixels are placed outside arena.
    if left_end_circle < 0:
        left_end_circle = 0
    if right_end_circle > ENV_WIDTH:
        right_end_circle = ENV_WIDTH
    if upper_end_circle < 0:
        upper_end_circle = 0
    if lower_end_circle > ENV_HEIGHT:
        lower_end_circle = ENV_HEIGHT

    for x in range(left_end_circle, right_end_circle):
        for y in range(upper_end_circle, lower_end_circle):
            dx = x - m_x
            dy = y - m_y
            distance_squared = dx * dx + dy * dy

            if distance_squared <= (radius * radius):
                pixel_array[y,x] = pixel_array[y,x] + 1
    return pixel_array

def loop_params_batches(agent_or_patch, folderpath, ENV_HEIGHT, ENV_WIDTH, posx, posy, RADIUS_AGENT, N, N_RESOURCES, RADIUS_RESOURCE, nr_of_radii, num_batches, window_pad):
    for i in range(0,nr_of_radii):
        x_values = []
        y_values = []
        for j in range(0,num_batches):
            # TODO check again: see x = np.random.randint(float, float) in sims.py. For float it rounds down so here we need to ceil
            x_values = np.append(x_values, posx[j,i,nr_of_radii - 1 - i,:,0][:int(N_RESOURCES[i])] - window_pad + int(RADIUS_RESOURCE[nr_of_radii - 1 - i]))
            y_values = np.append(y_values, posy[j,i,nr_of_radii - 1 - i,:,0][:int(N_RESOURCES[i])] - window_pad + int(RADIUS_RESOURCE[nr_of_radii - 1 - i]))

        # making sure that no drawn location is outside the arena borders
        max_x_values = np.max(x_values)
        max_y_values = np.max(y_values)
        if (max_x_values > ENV_WIDTH) | (max_y_values > ENV_HEIGHT):
            sys.exit('At least one patch location is outside the arena borders.')

        # loop through agent/patch locations and "create" a filled CIRCLE for each location in pixel array
        pixel_array = np.zeros([ENV_HEIGHT,ENV_WIDTH])
        for k in range(0,len(x_values)):
            pixel_array = modifiy_pixel_array_circle(pixel_array, x_values[k], y_values[k], RADIUS_RESOURCE[nr_of_radii - 1 - i])
        # normalized_pixel_array = pixel_array / pixel_array.sum() # this is the probability density for num_batches initializations
        normalized_pixel_array = pixel_array / num_batches # this is the frequency a pixel is covered
        # save pixel array for adapting plots (computation of pixel array is slow)
        np.savez(folderpath +'/{}_normalized_pixel_array_circles_R_{}_N_{}_R_R_{}_N_R_{}_batches_{}.npz'.format(agent_or_patch,round(RADIUS_AGENT,2),int(N),round(RADIUS_RESOURCE[nr_of_radii - 1 - i],2),int(N_RESOURCES[i]),num_batches),normalized_pixel_array=normalized_pixel_array)
        plot_circles(agent_or_patch, normalized_pixel_array, RADIUS_AGENT, N, RADIUS_RESOURCE[nr_of_radii - 1 - i], N_RESOURCES[i], num_batches, folderpath)

        # loop through agent/patch locations and add + 1 for each PIXEL location in pixel array
        pixel_array = np.zeros([ENV_HEIGHT,ENV_WIDTH])
        for k in range(0,len(x_values)):
            pixel_array[int(y_values[k]),int(x_values[k])] = pixel_array[int(y_values[k]),int(x_values[k])] + 1
        # normalized_pixel_array = pixel_array / pixel_array.sum() # this is the probability density for num_batches initializations
        normalized_pixel_array = pixel_array / num_batches # this is the frequency a pixel is covered
        # save pixel array for adapting plots (computation of pixel array is slow)
        np.savez(folderpath +'/{}_normalized_pixel_array_locations_R_{}_N_{}_R_R_{}_N_R_{}_batches_{}.npz'.format(agent_or_patch,round(RADIUS_AGENT,2),int(N),round(RADIUS_RESOURCE[nr_of_radii - 1 - i],2),int(N_RESOURCES[i]),num_batches),normalized_pixel_array=normalized_pixel_array)
        plot_centers(agent_or_patch,normalized_pixel_array, RADIUS_AGENT, N, RADIUS_RESOURCE[nr_of_radii - 1 - i], N_RESOURCES[i], num_batches, folderpath)
    return

def plot_circles(agent_or_patch,pixel_array, RADIUS_AGENT, N, RADIUS_RESOURCE, N_RESOURCES, num_batches, folderpath):
    plt.figure()
    plt.xlabel('location x')
    plt.ylabel('location y')
    plt.title('{} R={} N={} R_R={} N_R={} batches={}'.format(agent_or_patch,round(RADIUS_AGENT,2),int(N),round(RADIUS_RESOURCE,2),int(N_RESOURCES),num_batches))
    color_map = plt.imshow(pixel_array)
    color_map.set_cmap("viridis")
    plt.colorbar(label='Frequency')
    # plt.colorbar(label='Probability density')
    plt.tight_layout()
    plt.savefig(folderpath +'/{}_covered_area_R_{}_N_{}_R_R_{}_N_R_{}_batches_{}.png'.format(agent_or_patch,round(RADIUS_AGENT,2),int(N),round(RADIUS_RESOURCE,2),int(N_RESOURCES),num_batches))
    return

def plot_centers(agent_or_patch, pixel_array, RADIUS_AGENT, N, RADIUS_RESOURCE, N_RESOURCES, num_batches, folderpath):
    plt.figure()
    plt.xlabel('location x')
    plt.ylabel('location y')
    plt.title('{} R={} N={} R_R={} N_R={} batches={}'.format(agent_or_patch,round(RADIUS_AGENT,2),int(N),round(RADIUS_RESOURCE,2),int(N_RESOURCES),num_batches))
    color_map = plt.imshow(pixel_array)
    color_map.set_cmap("viridis")
    plt.colorbar(label='Frequency')
    # plt.colorbar(label='Probability density')
    plt.tight_layout()
    plt.savefig(folderpath +'/{}_locations_R_{}_N_{}_R_R_{}_N_R_{}_batches_{}.png'.format(agent_or_patch,round(RADIUS_AGENT,2),int(N),round(RADIUS_RESOURCE,2),int(N_RESOURCES),num_batches))
    return

# load data from /summary
data = ExperimentReplay(filepath, undersample=1)

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

# get resource parameters
RADIUS_RESOURCE = tuned_env['RADIUS_RESOURCE'] # different radii used in experiment
N_RESOURCES = tuned_env['N_RESOURCES']
res_posx = data.res_pos_x_z
res_posy = data.res_pos_y_z
num_batches, nr_of_radii, length_num_patches, max_N_RESOURCES , T = res_posx.shape

# get agent parameters
RADIUS_AGENT = fixed_env['RADIUS_AGENT'] # different radii used in experiment
N = fixed_env['N']
posx = data.posx_z
posy = data.posy_z

agent_or_patch = 'patch'
loop_params_batches(agent_or_patch, folderpath, ENV_HEIGHT, ENV_WIDTH, res_posx, res_posy, RADIUS_AGENT, N, N_RESOURCES, RADIUS_RESOURCE, nr_of_radii, num_batches, window_pad)
