import importlib
import json
import os
from abm.metarunner.metarunner import MetaProtocol
EXP_NAME = os.getenv("EXPERIMENT_NAME", "")
if EXP_NAME == "":
    raise Exception("No experiment name has been passed")

# The directory `root_dir` needs to contain a subdirectory for each group size .
# Within each subdirectory, there must be a subdirectory for each pach distributions
# Each  patch distribution subdirection `env_params` file that contains the simulation parameters .
# If we are evaluating DQN or DDQN agents,
# each subdirectory should also contain the trained models necessary for evaluation.
#
# Example structure:
# root_dir/
# ├── N3/
# │   ├── exp_sparse/
# │   │     ├── env_params
# │   │     ├── model_0.pth
# │   │     ├── model_1.pth
# │   │     └── model_2.pth
# │   ├──  exp_interm/
# │   └── .../
# ├── N5    /
# │   ├── env_params
# │   └── trained_model.pth (if evaluating IDQN agents)
# └── ...

root_dir = "/Users/ferielamira/Desktop/Uni/Master-thesis/ABM/"
# Get all the experiment directories for different group sizes
exps_path = os.path.join("/Users/ferielamira/Desktop/Uni/Master-thesis/ABM/abm/data/simulation_data/M6_experience_sharing")


# Get full paths of directories
exp_dirs = [os.path.join(exps_path, d) for d in os.listdir(exps_path) if  (os.path.isdir(os.path.join(exps_path, d)))]
#print(exp_dirs)
env_names = ["exp_sparse","exp_interm","exp_uniform_40","exp_uniform_100"]


num_trials = 10
def generate_env_file(env_data, file_name, save_folder):
    """Generating a single env file under save_folder with file_name including env_data as env format"""
    os.makedirs(save_folder, exist_ok=True)
    file_path = os.path.join(save_folder, file_name)
    with open(file_path, "a") as file:
        for k, v in env_data.items():
            file.write(f"{k}={v}\n")


for exp_dir in exp_dirs:
    # Iterating over all the environments fora  specific group size
    for i,env_name in enumerate(env_names):
        # The directory needs to contain the env_params.json file
        env_dir = os.path.join(exp_dir, env_name)
        env_file = os.path.join(env_dir, "env_params.json")

        try:
            with open(env_file, 'r') as file:
                # Load the JSON data
                original_env_params = json.load(file)
                env_params = original_env_params.copy()
        except:
            print(f"Could not load data from {env_file}")
            continue

        env_params["T"]=20000
        env_params["TRAIN"]=0
        if env_params["BRAIN_TYPE"]=="DQN" or env_params["BRAIN_TYPE"]=="DDQN":
            env_params["PRETRAINED"]=1
            env_params["PRETRAINED_MODELS_DIR"]=env_dir

        env_params["N_EPISODES"] = 1

        env_params["USE_RAM_LOGGING"]=1
        env_params["SAVE_CSV_FILES"]=1

        env_params["SAVE_ROOT_DIR"]=os.path.join(env_dir, "eval/batch_delete")
        for j in range(1,num_trials+1):
            env_params["SEED"]=j


            generate_env_file(env_params, "madrl_foraging.env", root_dir)

            import abm.contrib.ifdb_params as ifdbp
            from abm import app_madrl_foraging
            import abm.projects.madrl_foraging.madrl_contrib.madrl_learning_params as madrlp
            import abm.projects.madrl_foraging.madrl_contrib.madrl_movement_params as madrlmp

            # Reloading the modules to ensure that the new env file is read
            importlib.reload(madrlp)
            importlib.reload(ifdbp)
            importlib.reload(madrlmp)


            app_madrl_foraging.start(parallel=True, headless=False)

            #delete env file
            os.remove(os.path.join(root_dir,"madrl_foraging.env"))





