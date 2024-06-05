import os
import json
import shutil
import tkinter as tk
from tkinter import filedialog

# Create a root window and hide it
root = tk.Tk()
root.withdraw()

# Open a file dialog and get the directory path
rootDir = filedialog.askdirectory()

for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        if fname == 'env_params.json':
            file_path = os.path.join(dirName, fname)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                data['APP_VERSION'] = 'Base'
                print(f"changing APP_VERSION to Base in {file_path}")
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)