'''General replay script to browse and replay experiments from a folder.'''

from abm.replay.replay import ExperimentReplay

# opening file browser to choose folder of interest
from tkinter import Tk
from tkinter.filedialog import askdirectory
Tk().withdraw()

# get current directory to set as starting directory for file browsing
import os
current_dir = os.getcwd()
folder_path = askdirectory(title="Choose folder with experiment data", initialdir=current_dir)

if folder_path == "":
    print("No folder selected, exiting...")
    exit()

# try:
loaded_experiment = ExperimentReplay(folder_path)
loaded_experiment.start()
# except Exception as e:
#     print("Error during experiment loading:", e)
#     print("Please check if the folder contains all necessary files to be considered as an experiment.")
#     exit()