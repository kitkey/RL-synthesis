import stable_baselines3
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from torch import nn

from __init__ import PATH_TO_MODELS_DIRECTORY, PATH_TO_SAVE_STATS
from common.custom_callback import SaveOnBestTrainingRewardCallback
from common.osi_env import ICschemesEnv
from sb3_modelTest import model_test
import warnings
warnings.filterwarnings("ignore")

env = stable_baselines3.common.monitor.Monitor(env=ICschemesEnv(reward_index=9), filename=PATH_TO_MODELS_DIRECTORY,
                                               allow_early_resets=True)

n_steps = 20
n_epochs = 18000


config = {
    "policy": "MlpLstmPolicy",
    "env": env,
    "learning_rate": 1e-4,
    "seed": 42,
    "batch_size": 60,
    "n_steps": 960,
    "gamma": 0.99,
    "ent_coef": 0.01,
    "verbose": 1,
    "clip_range": 0.3,
    "policy_kwargs": dict(activation_fn=nn.ReLU,
                          net_arch=dict(pi=[80, 80, 80], vf=[80, 80, 80], lstm_hidden_size=[80, 80, 80]),
                          log_std_init=-1.0,
                          ortho_init=False),
    "vf_coef": 0.5,
    "n_epochs": 10,
    "normalize_advantage": True,
}

model = RecurrentPPO(**config)
path_to_save_stats = PATH_TO_SAVE_STATS
for i in range(100):
    model.learn(progress_bar=False, total_timesteps=960)
    model_test(model, path_to_save_stats+f"/model{i}.txt",True, 1)
    model.save(path=PATH_TO_MODELS_DIRECTORY+f"/model{i}.zip")

