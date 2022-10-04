# Import the RL algorithm (Algorithm) we would like to use.
from ast import arg
#from pyrsistent import T
from ray.rllib.algorithms.ppo import PPO

#DQN imports
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.simple_q.simple_q import (
    SimpleQ,
    SimpleQConfig,
)
from ray.rllib.algorithms.dqn import dqn
from ray.rllib.algorithms.dqn.dqn import DQNConfig


import ray
import os

import gym_interface

import hexagdly
#from stable_baselines3 import PPO, DQN
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
#from ray.rllib.algorithms import dqb
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
parser.add_argument("--worker_num")
parser.add_argument("--worker_cpu")
parser.add_argument("--driver_cpu")
parser.add_argument("--algo")

args = parser.parse_args()

torch, nn = try_import_torch()

class HexBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hexConv2d = hexagdly.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        residual = x
        x = self.hexConv2d(x)
        if self.residual:
            x += residual
        x = self.relu(x)
        return x


class MyCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, n_residual_layers = 7, use_residual = False):
        super(MyCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)

        n_input_channels = observation_space.shape[0]
        convs_per_layer = 64
        self.layers = OrderedDict()
        self.layers.update( {'conv': HexBlock(n_input_channels, convs_per_layer, residual=False)} )
        for i in range(n_residual_layers):
            layer_name = "resid"+str(i+1)
            self.layers.update( {layer_name: HexBlock(convs_per_layer, convs_per_layer, residual=use_residual)} ) 
        self.layers.update( {'flatten': nn.Flatten()})
        self.cnn = nn.Sequential(self.layers)
        model = self.cnn
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameter count {pytorch_total_params}")
        self.print_toggle = False

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if self.print_toggle:
            #print(observations)
            
            self.print_toggle = False
        #print("--------------Here are the observations--------------------")
        #print(observations[4])
        return self.linear(self.cnn(observations))

class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])

class TorchCustomModel2(TorchModelV2, nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, n_residual_layers = 7, use_residual = False):
        super(MyCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)

        n_input_channels = observation_space.shape[0]
        convs_per_layer = 64
        self.layers = OrderedDict()
        self.layers.update( {'conv': HexBlock(n_input_channels, convs_per_layer, residual=False)} )
        for i in range(n_residual_layers):
            layer_name = "resid"+str(i+1)
            self.layers.update( {layer_name: HexBlock(convs_per_layer, convs_per_layer, residual=use_residual)} ) 
        self.layers.update( {'flatten': nn.Flatten()})
        self.cnn = nn.Sequential(self.layers)
        model = self.cnn
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameter count {pytorch_total_params}")
        self.print_toggle = False

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if self.print_toggle:
            #print(observations)
            
            self.print_toggle = False
        #print("--------------Here are the observations--------------------")
        #print(observations[4])
        return self.linear(self.cnn(observations))



#get arguments
args = parser.parse_args()
run_name = args.name


stop_config = {
    #"training_iteration": 1,
    "timesteps_total": 500000,
    #"episode_reward_mean": 350,
}
ray_config = {
        "env": "atlatl",  # or "corridor" if registered above
        "env_config": {
            "role" :"blue",
            "versusAI":"pass-agg", 
            "scenario":"island_small-6", 
            "saveReplay":False, 
            "actions19":True, 
            "ai":"gym14", 
            "verbose":False, 
            "scenarioSeed":4025, 
            "scenarioCycle":0,
        },
        "disable_env_checking":True,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,
        "model": {
            "custom_model": "my_model2",
            "vf_share_layers": False,
        },
        "num_workers": int(args.worker_num),  # parallelism
        "framework": "torch",
        "num_cpus_per_worker" : int(args.worker_cpu),
        "num_cpus_for_driver": int(args.driver_cpu),
        "ignore_worker_failures": True,
    }



#get arguments
args = parser.parse_args()

#register env and model with ray
register_env("atlatl", lambda config: gym_interface.GymEnvironment(**config))
ModelCatalog.register_custom_model( "MyCNN", MyCNN)
ModelCatalog.register_custom_model("my_model", TorchCustomModel)
ModelCatalog.register_custom_model("my_model2", TorchCustomModel)



ray.init()


#make a ray tunner

#algo development area
if args.algo is not None:
    algo = dqn.DQN(env="atlatl", config=ray_config)
else:
    algo = ppo.PPO(env="atlatl", config=ray_config)  # config to pass to env class


def run_tune():
    tune.Tuner(
    "PPO",
    run_config=air.RunConfig(stop={"timesteps_total": 100000}),
    param_space={
        "env": "atlatl",
        "env_config": 
        {
            "role" :"blue",
            "versusAI":"pass-agg", 
            "scenario":"clear-navy-6", 
            "saveReplay":False, 
            "actions19":True, 
            "ai":"gym14", 
            "verbose":False, 
            "scenarioSeed":4025, 
            "scenarioCycle":0,
        },
        "disable_env_checking":True,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,
        "model": {
            "custom_model": "my_model2",
            "vf_share_layers": False,
        },
        "num_workers": int(args.worker_num),  # parallelism
        "framework": "torch",
        "num_cpus_per_worker" : int(args.worker_cpu),
        "num_cpus_for_driver": int(args.driver_cpu),
        "ignore_worker_failures": True,
        },
        ).fit()
    return 

#results = run_tune()
#print(results)

for _ in range(500):
    result = algo.train()
    print(pretty_print(result))
    

#result = algo.evaluate()
#print(pretty_print(result))


#algo.save("/home/matthew.finley/Thesis-MV4025/server/ray_models/"+run_name)
#algo.export_model("model", "/home/matthew.finley/Thesis-MV4025/server/ray_models/"+run_name+"_model")
#algo.export_policy_model("/home/matthew.finley/Thesis-MV4025/server/ray_models/"+run_name+"_policy")