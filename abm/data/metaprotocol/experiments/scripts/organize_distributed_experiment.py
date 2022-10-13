from glob import glob
import os
import shutil

# folder in which the individual hashed exp folders are
# distributed_exp_path = "/home/david/Desktop/clustermount/ABM/abm/data/simulation_data/figExp1N100"
distributed_exp_path = "/home/david/Desktop/database/figExp2A/figExp2AintermedN25NoColl"

hashed_subfolders = glob(os.path.join(distributed_exp_path, "*/"), recursive=False)
batch_num = 0
for hashed_sf in hashed_subfolders:
    batch_folders = glob(os.path.join(hashed_sf, "*/"), recursive=False)
    for batch_folder in batch_folders:
        if batch_num == 0:
            print("Moving experiment readme file!")
            readme_path = os.path.join(hashed_sf, "README.txt")
            new_readme_path = os.path.join(distributed_exp_path, f"README.txt")
            if os.path.isfile(readme_path):
                if not os.path.isfile(new_readme_path):
                    shutil.move(readme_path, new_readme_path)
                else:
                    print("Experiment readme already copied!")
            else:
                print("Didn't find readme file in hashed experiment subfolder")
        print(f"moving batch from path _ to _:\n\t{batch_folder}\n\t{os.path.join(distributed_exp_path, f'batch_{batch_num}')}")
        shutil.move(batch_folder, os.path.join(distributed_exp_path, f"batch_{batch_num}"))
        batch_num += 1
    print(f"removing hashed folder {hashed_sf}")
    shutil.rmtree(hashed_sf)
