import importlib
import json
import os

from abm.metarunner.metarunner import MetaProtocol

#TODO: Generate experiment names from a list of experiments
root_dir = "/Users/ferielamira/Desktop/Uni/Master-thesis/ABM/"
exp_dir = os.path.join(root_dir,"abm/data/simulation_data/exp_new/batch_0/")

env_names = ["exp_ind_interm","exp_ind_sparse","exp_ind_uniform"]
#env_names = ["exp_binary-intermP","exp_binary-sparseP","exp_binary-patchyP"]

num_trials = 11
def generate_env_file(env_data, file_name, save_folder):
    """Generating a single env file under save_folder with file_name including env_data as env format"""
    os.makedirs(save_folder, exist_ok=True)
    file_path = os.path.join(save_folder, file_name)
    with open(file_path, "a") as file:
        for k, v in env_data.items():
            file.write(f"{k}={v}\n")

for i,env_name in enumerate(env_names):
    env_dir = os.path.join(exp_dir, env_name)
    env_file = os.path.join(env_dir, "env_params.json")


    # read env_params from folder
    #open json file
    with open(env_file, 'r') as file:
        # Load the JSON data
        original_env_params = json.load(file)
        env_params = original_env_params.copy()

    env_params["T"]=20000
    env_params["TRAIN"]=0
    env_params["PRETRAINED"]=1
    env_params["PRETRAINED_MODELS_DIR"]=env_dir
    env_params["USE_RAM_LOGGING"]=1
    env_params["SAVE_CSV_FILES"]=1
    env_params["SAVE_ROOT_DIR"]=os.path.join(env_dir, "eval")
    for j in range(1,num_trials):
        env_params["SEED"]=j

        generate_env_file(env_params, ".env", root_dir)

        import abm.contrib.ifdb_params as ifdbp
        from abm import app_madrl_foraging
        import abm.projects.madrl_foraging.madrl_contrib.madrl_learning_params as madrlp

        importlib.reload(madrlp)
        importlib.reload(ifdbp)

    
        app_madrl_foraging.start(parallel=True, headless=False)

        #delete env file
        os.remove(os.path.join(root_dir,".env"))





