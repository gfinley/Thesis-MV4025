# Import the RL algorithm (Algorithm) we would like to use.
from ast import arg
from ray.rllib.algorithms.ppo import PPO
import ray
import os

import gym_interface

import hexagdly
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.utils import get_schedule_fn

from stable_baselines3.dqn.policies import CnnPolicy

import gym
import torch as th
from torch import nn as nn
import gym_interface

from gym import spaces
import numpy as np
import random

import hexagdly

import argparse

from collections import OrderedDict

from ray.rllib.algorithms import ppo

from ray.tune.registry import register_env

from ray.rllib.models import ModelCatalog

from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.tune.logger import pretty_print
from ray import air, tune


parser = argparse.ArgumentParser()

parser.add_argument("--name")

args = parser.parse_args()

torch, nn = try_import_torch()


#change the config portion for evaluation
ray_config = {
        "env": "atlatl",  # or "corridor" if registered above
        "env_config": {
            "role" :"blue",
            "versusAI":"pass-agg", 
            "scenario":"city-inf-5", 
            "blueReplay":"eval_save.js", 
            "actions19":True, 
            "ai":"gym14", 
            "verbose":False, 
            "scenarioSeed":4025, 
            "scenarioCycle":0,
        },
        "disable_env_checking":True,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model2",
            "vf_share_layers": False,
        },
        "num_workers": 0,  # parallelism
        "framework": "torch",
        "num_cpus_per_worker" :2,
        "num_cpus_for_driver": 8,

        "evaluation_num_workers": 1,
        "evaluation_config": {
            "role" :"blue",
            "versusAI":"pass-agg", 
            "scenario":"city-inf-5", 
            "blueReplay":"ray_blue_save.js", 
            "actions19":True, 
            "ai":"gym14", 
            "verbose":False, 
            "scenarioSeed":4025, 
            "scenarioCycle":0,
            "nReps": 10
        }

    }

#get arguments
args = parser.parse_args()

#register env and model with ray
#this should change it to a run with model saving in effect
register_env("atlatl", lambda config: gym_interface.GymEnvironment(**config))


checkpoint_path = 
algo = dqn.DQN(env="atlatl", config=ray_config)
agent.restore(checkpoint_path)

#ray.init()


#make a ray tunner


#algo = ppo.PPO(env="atlatl", config=ray_config)  # config to pass to env class

run_name = args.name



#for _ in range(20):
##    result = algo.train()
#    print(pretty_print(result))
    
#
##result = algo.evaluate()
#print(pretty_print(result))


#algo.save("/home/matthew.finley/Thesis-MV4025/server/ray_models/"+run_name)
#algo.export_model("model", "/home/matthew.finley/Thesis-MV4025/server/ray_models/"+run_name+"_model")
#algo.export_policy_model("/home/matthew.finley/Thesis-MV4025/server/ray_models/"+run_name+"_policy")