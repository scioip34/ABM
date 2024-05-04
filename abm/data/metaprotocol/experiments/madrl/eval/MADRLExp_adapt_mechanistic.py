import importlib
import json
import os

from abm.metarunner.metarunner import MetaProtocol

#TODO: Generate experiment names from a list of experiments
#root_dir = "/Users/ferielamira/Desktop/Uni/Master-thesis/ABM/"
root_dir = "/home/users/f/feriel-amira1/ABM"
exp_dir = os.path.join(root_dir, "abm/data/simulation_data/exp_mecanistic")

# Get full paths of directories
#exp_dirs = [os.path.join(exps_path, d) for d in os.listdir(exps_path) if (d=="M9V2" and os.path.isdir(os.path.join(exps_path, d)))]
#print(exp_dirs)
env_names = ["exp_sparse","exp_interm","exp_uniform_50","exp_uniform_100"]#,"exp_interm"]
#env_names = ["exp_binary-intermP","exp_binary-sparseP","exp_binary-patchyP"]

num_trials = 100
def generate_env_file(env_data, file_name, save_folder):
    """Generating a single env file under save_folder with file_name including env_data as env format"""
    os.makedirs(save_folder, exist_ok=True)
    file_path = os.path.join(save_folder, file_name)
    with open(file_path, "a") as file:
        for k, v in env_data.items():
            file.write(f"{k}={v}\n")


for i,env_name in enumerate(env_names):
    #if "N3" in exp_dir and (env_name =="exp_sparse" or env_name =="exp_uniform_100"):
    #    continue

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
    #env_params["N"]=3
    env_params["N_EPISODES"] = 1
    env_params["WITH_VISUALIZATION"]=0
    env_params["PRETRAINED_MODELS_DIR"]=env_dir
    env_params["USE_RAM_LOGGING"]=1
    env_params["SAVE_CSV_FILES"]=1
    env_params["SAVE_ROOT_DIR"]=os.path.join(env_dir, "eval/batch_1")
    for j in range(1,num_trials+1):
        env_params["SEED"]=j

        generate_env_file(env_params, ".env", root_dir)

        import abm.contrib.ifdb_params as ifdbp
        from abm import app
        #import abm.contrib.madrl_learning_params as madrlp

        #importlib.reload(madrlp)
        importlib.reload(ifdbp)


        app.start(parallel=True, headless=False)
        #print("Done with seed ", j)

        #delete env file
        os.remove(os.path.join(root_dir,".env"))





