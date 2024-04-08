from abm.replay.replay import ExperimentReplay
import matplotlib.pyplot as plt
from time import sleep
import os

experiment_path = "/home/david/Desktop/ABM/abm/data/simulation_data/exp12/exp12_124cef78b728688f82b6"
t_start = 0
t_end = 100000
undersample = 5000

replayed_experiment = ExperimentReplay(experiment_path, t_start=t_start, t_end=t_end, undersample=undersample)
# replayed_experiment.start()

# Defining saving folder
save_path = os.path.join(experiment_path, "summary", "eff_plots", "moving_window")
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
for i in range(int(t_end/undersample) - window_length):
    replayed_experiment.t_start = i
    replayed_experiment.t_end = i + window_length

    fig, ax, cbar = replayed_experiment.on_print_efficiency(with_read_collapse_param=False)
    # adjusting figure
    ax.set_title(f"t= {replayed_experiment.t_start} - {replayed_experiment.t_end}")
    # set color range for all figures to the same range
    im = ax.get_images()
    im[0].set_clim(0.075, 0.2)
    cbar.ax.set_yticks([0.075, 0.1, 0.2])

    # Save adjusted figure
    file_name = f"t_{replayed_experiment.t_start}_{replayed_experiment.t_end}_eff.png"
    file_path = os.path.join(save_path, file_name)
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    sleep(1)
    plt.close(fig)

del replayed_experiment