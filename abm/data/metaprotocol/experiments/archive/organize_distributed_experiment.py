from glob import glob
import os
import shutil

# folder in which the individual hashed exp folders are
distributed_exp_path = "/home/mezey/Desktop/clustermount/ABM/abm/data/simulation_data/exp9"

hashed_subfolders = glob(os.path.join(distributed_exp_path, "*/"), recursive=False)
batch_num = 0
for hashed_sf in hashed_subfolders:
    batch_folders = glob(os.path.join(hashed_sf, "*/"), recursive=False)
    for batch_folder in batch_folders:
        print(f"moving batch from path _ to _:\n\t{batch_folder}\n\t{os.path.join(distributed_exp_path, f'batch_{batch_num}')}")
        shutil.move(batch_folder, os.path.join(distributed_exp_path, f"batch_{batch_num}"))
        batch_num += 1
    print(f"removing hashed folder {hashed_sf}")
    os.rmdir(hashed_sf)


