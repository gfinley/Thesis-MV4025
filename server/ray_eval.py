# Import the RL algorithm (Algorithm) we would like to use.
from ast import arg
#from pyrsistent import T
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig

#DQN imports
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.simple_q.simple_q import (
    SimpleQ,
    SimpleQConfig,
)
from ray.rllib.algorithms.dqn import dqn
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer

import datetime
import ray
import os

import gym_interface

import hexagdly
#from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common import base_class
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

from ray.rllib.agents.impala import ImpalaTrainer

#import for navy vision CNN testing
from models import Navy_VisionNetwork,VisionNetwork_CNN,HEX_VisionNetwork,VisionNetwork

from torchvision import models
from torchsummary import summary

from ray.util.metrics import Counter, Gauge, Histogram


parser = argparse.ArgumentParser()

parser.add_argument("--name")
parser.add_argument("--worker_num")
parser.add_argument("--worker_cpu")
parser.add_argument("--driver_cpu")
parser.add_argument("--algo")
parser.add_argument("--checkpoint")

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

class TorchCustomModel_2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        
        n_residual_layers = 7
        use_residual = False
        features_dim: int = 512
        
        n_input_channels = obs_space.shape[0]
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
            n_flatten = self.cnn(th.as_tensor(obs_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.torch_sub_model = self.linear


    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        #fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        #return fc_out, []
        #print("--------------Here are the observations--------------------")
        #print(observations[4])
        return self.linear(self.cnn(input_dict["obs"])),[]

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


class TorchCustomModel_3(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = Navy_VisionNetwork(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])
class TorchCustomModel_4(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = VisionNetwork_CNN(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


class Hex_CNN(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = HEX_VisionNetwork(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


class Model_Vision(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = VisionNetwork(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])

class FC_NET(TorchModelV2, nn.Module):
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


#get arguments
args = parser.parse_args()
run_name = args.name

ray_config = {
        "env": "atlatl", 
        "env_config": {
            "role" :"blue",
            "versusAI":"pass-agg", 
            "scenario":"city-inf-5",
            #dont uncomment, does not work 
            #"--blueReplay" : "Replay_Impala.js", 
            "actions19":False, 
            "ai":"gym14", 
            "verbose":False, 
            "scenarioSeed":4025, 
            "scenarioCycle":0,

            #"saveReplay":"replay_defualt.js",

        },

        "disable_env_checking":True,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,
        "model": {
            "custom_model": "Hex_CNN",
            
            "post_fcnet_hiddens": [1600,512, 512,7],
            "post_fcnet_activation": "relu",
              
           
            "conv_filters" : [[64,[1,1],1],[64,[1,1],1]],
            "conv_activation"  : "relu",
            "vf_share_layers" : True,

        },
        "num_workers": 0,  # parallelism turned off for evaluation
        "framework": "torch",
        "num_cpus_per_worker" : 1,
        "num_cpus_for_driver": int(args.driver_cpu),
        "ignore_worker_failures": True,
        "create_env_on_driver":True,
        "evaluation_num_workers": 0,
        "evaluation_duration": 10,
        "evaluation_interval": 1,
        "evaluation_duration_unit": "episodes",
        "explore" : False,
    }



#get arguments
args = parser.parse_args()

#register env and model with ray
register_env("atlatl", lambda config: gym_interface.GymEnvironment(**config))
ModelCatalog.register_custom_model( "MyCNN", MyCNN)
ModelCatalog.register_custom_model("my_model", TorchCustomModel)
ModelCatalog.register_custom_model("my_model_2", TorchCustomModel_2)
ModelCatalog.register_custom_model("my_model_3", TorchCustomModel_3)
ModelCatalog.register_custom_model("my_model_4", TorchCustomModel_4)
ModelCatalog.register_custom_model("Navy_CNN", Navy_VisionNetwork)
ModelCatalog.register_custom_model("FC_NET", FC_NET)
ModelCatalog.register_custom_model("Hex_CNN", Hex_CNN)
ModelCatalog.register_custom_model("Model_Vision", Model_Vision)



ray.init(num_cpus=int(args.driver_cpu), num_gpus=0, log_to_driver=False)

checkpoint = args.checkpoint
#make a ray tunner


restored_trainer = ImpalaTrainer(env="atlatl", config=ray_config)
restored_trainer.restore(checkpoint)

#print(pretty_print(result))

for i in range(0,2):
    evaluation = restored_trainer.evaluate(checkpoint)
    #print(pretty_print(evaluation))
    rewards = evaluation['evaluation']['hist_stats']['episode_reward']
    print(*rewards,sep=',')
    restored_trainer = ImpalaTrainer(env="atlatl", config=ray_config)
    restored_trainer.restore(checkpoint)

#print()
#print(pretty_print(evaluation))
#policy = restored_trainer.get_policy()
#print(policy)
#model = policy.model
#print(model)
#summary(model, (14,5,5))

