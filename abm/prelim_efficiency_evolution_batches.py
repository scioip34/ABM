from abm.replay.replay import ExperimentReplay
import matplotlib.pyplot as plt
from time import sleep
import os

experiment_path = "/home/mezey/Desktop/clustermount/ABM/abm/data/simulation_data/exp13"
t_start = 0
t_end = 10000
undersample = 2500
used_batches = 100

replayed_experiment = ExperimentReplay(experiment_path, t_start=t_start, t_end=t_end, undersample=undersample)
# replayed_experiment.start()

# Defining saving folder
save_path = os.path.join(experiment_path, "summary", "eff_plots", "used_batches")
if not os.path.isdir(save_path):
    os.makedirs(save_path, exist_ok=True)

# Collapsing efficiency plot without interactive dropdown (also set with_read_collapse_param=False in plotting)
replayed_experiment.experiment.set_collapse_param('MAX-0')

# To plot batches with automatically closing the generated figures we set this parameter to True
# This makes the plt.show function called in a non-blocking way
replayed_experiment.from_script = True

# Checking efficiency in moving window
plt.ion()
window_length = 1
for i in range(0, used_batches, 5):
    print(f"Using {i} batches")
    replayed_experiment.t_start = 1
    replayed_experiment.t_end = -1

    fig, ax, cbar = replayed_experiment.on_print_efficiency(with_read_collapse_param=False, used_batches=i)
    # adjusting figure
    ax.set_title(f"Used batches: {i+1}")
    # set color range for all figures to the same range
    im = ax.get_images()
    im[0].set_clim(0.04, 0.15)
    cbar.ax.set_yticks([0.04, 0.1, 0.15])

    # Save adjusted figure
    file_name = f"ub_{i+1}_eff.png"
    file_path = os.path.join(save_path, file_name)
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    sleep(1)
    plt.close(fig)

del replayed_experiment