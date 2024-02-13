# This script generates a part of figure 1 (without phase diagrams) of the Visual Flocking paper.
# It has been designed for VSRMExp1 dataset, and compatibility with other data is not guaranteed.

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
import os
import json
import glob
import colorcet as cc

clist_1 = ['#303030', '#443762', '#534b7a', '#5c6c73', '#4b9847', '#5cc22c', '#a9dc63', '#dbef97', '#f8faca']
cmap = cc.cm.CET_CBL2

# data path
exp_name = "VSWRMExp1"
data_path = f"/home/david/Desktop/database/VSWRM_figdata/{exp_name}/summary"

# included metrics
metrics = ["Phase Diagram", "Polarization\n Order", "Mean Inter-\nIndividual Distance", "Size of \nLargest Cluster", "Area-to-Circle\nRatio [%]", "Time Ratio\n in Overlap [%]"]
metric_files = [None, "polarization_us1.npy", "meaniid.npy", "largest_clustering_data.npy", "meanelong.npy", "aacoll.npy"]
metric_type = [None, "raw", "mean", "raw", "final", "raw"]
metric_limits = [None, [0, 1], None, [0, 10], [0, 60], [0, 5]]

# creating mesh of figures
# 6 rows with different metrics as above
# 4 columns: FOV: 25, 50, 75, 100 [%]
fig_shape = [len(metrics), 4]
fig, ax = plt.subplots(fig_shape[0], fig_shape[1],
                       constrained_layout=True, figsize=(fig_shape[1] * 3, fig_shape[0] * 3))
gs1 = gridspec.GridSpec(fig_shape[0], fig_shape[1])
gs1.update(wspace=0, hspace=0.1)

# alpha0 and beta0 values
alphas = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1, 1.25, 2, 4]
betas = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1, 1.25, 2, 4]

# saving plot ims for colorbar
ims = []

# going from row to row and plotting the different average summary metrics
for mi, metric in enumerate(metrics):
    if metric != "Phase Diagram":
        # loading data
        file_name = metric_files[mi]
        file_path = os.path.join(data_path, file_name)
        data = np.load(file_path)
        print(f"Loaded {file_name} with shape: {data.shape}")

        if metric.find("%") != -1:
            data = data * 100

        # we take the average along the batch (repetition) dimension if the data is not yet averaged
        if metric_type[mi] == "raw":
            data = np.mean(data, axis=0)

        if metric_type[mi] != "final":
            # taking mean over time
            data = np.mean(data, axis=-1)

        # exploring the minimum and maximum value of the data
        print(f"{metric}: min: {np.min(data)}, max: {np.max(data)}")

        for fi in range(4):
            if metric_limits[mi] is not None:
                im = ax[mi, fi].imshow(data[fi, :, :], cmap=cmap, vmin=metric_limits[mi][0], vmax=metric_limits[mi][1])
            else:
                im = ax[mi, fi].imshow(data[fi, :, :], cmap=cmap, vmin=np.min(data), vmax=np.max(data))

            if fi == 3:
                ims.append(im)

            # adding title in the first row
            if mi == 0:
                ax[mi, fi].set_title(f"FOV: {25 * (fi + 1)} [%]")

            # adding xlabel  and beta ticks in the last row
            if mi == len(metrics) - 1:
                ax[mi, fi].set_xlabel("$\\beta_0$")
                ax[mi, fi].set_xticks(range(len(betas)), ha='left', rotation_mode='anchor')
                ax[mi, fi].set_xticklabels(betas)
                # set tick rotation to 45 degrees
                ax[mi, fi].tick_params(axis='x', rotation=45)
            else:
                plt.axes(ax[mi, fi])
                plt.xticks([], [])

            # adding ylabel in the first column
            if fi == 0:
                ax[mi, fi].set_ylabel("$\\alpha_0$")
                ax[mi, fi].set_yticks(range(len(alphas)))
                ax[mi, fi].set_yticklabels(alphas)
            else:
                plt.axes(ax[mi, fi])
                plt.yticks([], [])

            # setting aspect to square
            ax[mi, fi].set_aspect('equal')
    else:

        # adding placeholder for ims
        ims.append(None)

        for fi in range(4):
            ax[mi, fi].set_aspect('equal')
            plt.axes(ax[mi, fi])
            plt.yticks([], [])
            plt.xticks([], [])

# setting font on all x and y axis to smaller
for a in ax.flatten():
    a.tick_params(axis='both', which='major', labelsize=7)

# plotting adjustments (5 rows)
left = 0.226
right = 0.798
bottom = 0.062
top = 0.976

# plotting adjustments (6 rows) FULL SCREEN!
left = 0.502
right = 0.814
bottom = 0.062
top = 0.976

# adding external colorbars per row
# [left, bottom, width, height]
# looping revrese to add the colorbars in the right order
ims = np.array(ims)
for imi, im in enumerate(ims):
    print(imi)
    cbar_width = 0.015
    if im is not None:
        # adjustments for 5 rows
        # height of a single row
        # row_height = (1-((1-top)+bottom))/len(metrics)
        # gap_from_top = - 0.1175
        # gap_between_bars = 0.007
        # cbaxes = fig.add_axes([left + 0.58, gap_from_top + (imi*row_height), cbar_width, row_height - gap_between_bars])

        # adjustments for 6 rows
        row_height = (1-((1-top)+bottom))/len(metrics)
        gap_from_top = - 0.076
        gap_between_bars = 0.01
        cbaxes = fig.add_axes([right+cbar_width-gap_between_bars+0.02, gap_from_top + (imi*row_height-(gap_between_bars/2)), cbar_width, row_height - 2*gap_between_bars])

        # position for the colorbar
        cb = plt.colorbar(ims[len(metrics)-imi], cax=cbaxes, orientation='vertical')
        # cb.ax.set_xticks([min_data_val, (min_data_val+max_data_val)/2, max_data_val])
        # cb.ax.xaxis.set_ticks_position("top")
        # cb.ax.xaxis.set_label_position('top')

        # moving ticks to the left side of the colorbar and leave ylabel on the right
        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.yaxis.set_label_position('right')

        # # set tick rotation to 45 degrees
        # cb.ax.tick_params(axis='y', rotation=45)

        # set tick font size to 5
        cb.ax.tick_params(labelsize=7)

        # set ylabel font size to 7
        cb.ax.set_ylabel(metrics[len(metrics)-imi], fontsize=10, rotation=270, labelpad=30)


# for 5 rows
# plt.subplots_adjust(hspace=0, wspace=0, top=0.976, bottom=0.062, left=0.226, right=0.798)

# for 6 rows
plt.subplots_adjust(hspace=0, wspace=0, top=top, bottom=bottom, left=left, right=right)
plt.show()

