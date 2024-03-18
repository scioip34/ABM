"""parameters for individual exploration, relocation and exploitation movements"""
from pathlib import Path

from dotenv import dotenv_values
import os

EXP_NAME = os.getenv("EXPERIMENT_NAME", "")
root_abm_dir = Path(__file__).parent.parent.parent.parent.parent
#root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")
envconf = dotenv_values(env_path)
print("envconf:", env_path)

# Exploration movement parameters
pretrained = bool(int(envconf.get("PRETRAINED")))
pretrained_models_dir = envconf.get("PRETRAINED_MODELS_DIR")
train = bool(int(envconf.get("TRAIN")))
train_every = int(envconf.get("TRAIN_EVERY"))
num_episodes= int(envconf.get("N_EPISODES"))
batch_size = int(envconf.get("BATCH_SIZE"))
replay_memory_capacity = int(envconf.get("REPLAY_MEMORY_CAPACITY"))
gamma = float(envconf.get("GAMMA"))
lr = float(envconf.get("LR"))
epsilon_start = float(envconf.get("EPSILON_START"))
epsilon_end = float(envconf.get("EPSILON_END"))
epsilon_decay = int(envconf.get("EPSILON_DECAY"))
tau = float(envconf.get("TAU"))
optimizer = envconf.get("OPTIMIZER")
ise_w = float(envconf.get("ISE_W"))
cse_w = float(envconf.get("CSE_W"))
#tp = bool(float(envconf.get("TP")))
seed = int(envconf.get("SEED"))
#binary_env_status = bool(float(envconf.get("BINARY_ENV_STATUS")))

brain_type = envconf.get("BRAIN_TYPE")
