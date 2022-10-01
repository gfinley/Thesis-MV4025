import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.preprocessing import is_image_space

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import gym_interface
import random

import numpy
import torch

# Comment out to use an automatically selected seed
# Uncomment for reproducible runs
seed = 55555
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

env = gym_interface.GymEnvironment(role="blue", versusAI="passive", scenario="atomic-city.scn", saveReplay=False, actions19=False, ai="gym", verbose=False)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=400, log_interval=100)

model.save("model_save")

print(f'eval results (mean, std. dev.): {evaluate_policy(model, model.get_env(), n_eval_episodes=100, deterministic=False)}')

print(f'Correct results (linux): (223.48, 130.02741864699152)')