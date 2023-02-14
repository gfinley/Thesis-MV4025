# Import the RL algorithm (Algorithm) we would like to use.
from ast import arg
#from pyrsistent import T
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
#from ray.rllib.algorithms.impala import ImpalaConfig

#DQN imports
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.simple_q.simple_q import (
    SimpleQ,
    SimpleQConfig,
)
from ray.rllib.algorithms.dqn import dqn
from ray.rllib.algorithms.dqn.dqn import DQNConfig
#from ray.rllib.agents.dqn import DQNTrainer
#from ray.rllib.agents.ppo import PPOTrainer
#from ray.rllib.agents.impala import ImpalaTrainer

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

#import for navy vision CNN testing
from models import Navy_VisionNetwork,VisionNetwork_CNN,VisionNetwork,HEX_VisionNetwork,Baysian_network,dev_model,dev_model_with_cnn


parser = argparse.ArgumentParser()

parser.add_argument("--name")
parser.add_argument("--worker_num")
parser.add_argument("--worker_cpu")
parser.add_argument("--driver_cpu")
parser.add_argument("--algo")
parser.add_argument("--model")
parser.add_argument("--gpu")

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


class Model_bayesian(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = Baysian_network(
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

class dev_net(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = dev_model(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])

class dev_net_with_cnn(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = dev_model_with_cnn(
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
experiment_name = args.name
algo = args.algo

model_type = args.model
model_type = str(model_type)
num_gpus_to_use = int(args.gpu)

#get the scn for the expierment
if "inf-x" in experiment_name:
    scn = experiment_name
else:
    scn = "city-inf-5"

#get arguments
args = parser.parse_args()

#register env and model with ray
#register_env("atlatl", lambda config: gym_interface.GymEnvironment(**config))
tune.register_env("atlatl", lambda config: gym_interface.GymEnvironment(**config))
ModelCatalog.register_custom_model( "MyCNN", MyCNN)
ModelCatalog.register_custom_model("FC_NET", FC_NET)
ModelCatalog.register_custom_model("my_model_2", TorchCustomModel_2)
ModelCatalog.register_custom_model("my_model_3", TorchCustomModel_3)
ModelCatalog.register_custom_model("Model_Vision", Model_Vision)
ModelCatalog.register_custom_model("Navy_CNN", Navy_VisionNetwork)
ModelCatalog.register_custom_model("Model_bayesian", Model_bayesian)

#this is the one that works well, is still a small CNN
ModelCatalog.register_custom_model("Hex_CNN", Hex_CNN)

ModelCatalog.register_custom_model("dev_net", dev_net)
ModelCatalog.register_custom_model("dev_net_with_cnn", dev_net_with_cnn)

print("starting ray")

total_cpu = int(args.worker_num) * int(args.worker_cpu) + int(args.driver_cpu)


ray.init(num_cpus=total_cpu, num_gpus=num_gpus_to_use)

from ray.tune.logger.logger import Logger, LoggerCallback

#try to enable tunning for a ray trining run
#make the impala config

env_config_settings = {
            "role" :"blue",
            "versusAI":"pass-agg", 
            "scenario":scn, 
            "saveReplay":False, 
            "actions19":False, 
            "ai":"gym14", 
            "verbose":False, 
            #"scenarioSeed":4025, 
            "scenarioCycle":0,
        }

model_config = {   
            "custom_model": model_type,
            
            "fcnet_hiddens": [1600,512,512,7],
            "fcnet_activation": "relu",
            "vf_share_layers" : True,
        }
#saved old std config
#            "custom_model": model_type,
#            
#            "post_fcnet_hiddens": [1600,512,512,7],
#            "post_fcnet_activation": "relu",
#              
#           
#            #"conv_filters" : [[64,[1,1],1],[64,[1,1],1]],
#            "conv_filters" : [[64,[1,1],1],[64,[1,1],1]],
#            "conv_activation"  : "relu",
#            "vf_share_layers" : True,


from ray.rllib.algorithms.impala import ImpalaConfig
if algo == "IMPALA":
    trainer_config = ImpalaConfig()
if algo == "PPO":
    trainer_config = PPOConfig()

#import the following algorithms , A3C, APPO, DDPG, DQN, ES, IMPALA, PPO, SAC, TD3
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.es import ESConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
from ray.rllib.algorithms.td3 import TD3Config

if algo == "A3C":
    trainer_config = A3CConfig()
if algo == "APPO":
    trainer_config = APPOConfig()
if algo == "DDPG":
    trainer_config = DDPGConfig()
if algo == "DQN":
    trainer_config = DQNConfig()
if algo == "ES":
    trainer_config = ESConfig()
if algo == "SAC":
    trainer_config = SACConfig()
if algo == "AlphaZero":
    trainer_config = AlphaZeroConfig()
    model_config = {   
           "custom_model": model_type,
           
           "post_fcnet_hiddens": [1600,512,512,7],
           "post_fcnet_activation": "relu",
             
          
           #"conv_filters" : [[64,[1,1],1],[64,[1,1],1]],
           "conv_filters" : [[64,[1,1],1],[64,[1,1],1]],
           "conv_activation"  : "relu",
           "vf_share_layers" : True,
        }
if algo == "TD3":
    trainer_config = TD3Config()

#trainer_config = trainer_config.training(lr= 0.0005)
trainer_config = trainer_config.environment(env="atlatl", env_config=env_config_settings)
trainer_config = trainer_config.environment(disable_env_checking=True)

#set the model to be the custom Hex_cnn
trainer_config = trainer_config.training(model=model_config)



#set config jobs cpu and gpu resorces
trainer_config = trainer_config.resources(num_cpus_per_worker=1)
trainer_config = trainer_config.resources(num_gpus=num_gpus_to_use)
trainer_config = trainer_config.resources(num_cpus_for_local_worker=int(args.driver_cpu))


#number of workers to roll out
trainer_config = trainer_config.rollouts(num_rollout_workers=int(args.worker_num))
#envs per worker
trainer_config = trainer_config.rollouts(num_envs_per_worker=1)
#set the framwork to torch
trainer_config = trainer_config.framework("torch")

#create env on local worker
trainer_config = trainer_config.rollouts(create_env_on_local_worker=True)
#ignore worker failures
trainer_config = trainer_config.rollouts(ignore_worker_failures=True)

#set the number of concurent experiments


#make a stopper for max iterations
from ray.tune import stopper
from ray.tune.stopper import MaximumIterationStopper, FunctionStopper

#stopper_iter = MaximumIterationStopper(max_iter=100)
#3stopper_episode = stopper.FunctionStopper(lambda result: result["episodes_total"] > 100000)

def stopper_episodes(trial_id, result):
    return result["episodes_total"] > 5000000

def stopper_episodes_500K(trial_id, result):
    return result["episodes_total"] > 500000

#timesteps_total stopper
def stopper_timesteps_5000000(trial_id, result):
    return result["timesteps_total"] > 5000000


#run name is conctnation of algo, worker num, and worker cpu and ig gpu is used then _gpu is added
def gpu_check(gpu):
    if gpu == 0:
        return ""
    else:
        return "_gpu"


enable_hypersearch = True
if enable_hypersearch == True: #enable hyperparam tuning
    trainer_config = trainer_config.training(
        lr=tune.grid_search([0.0001, 0.0005,0.001,0.005,0.01]), 
        #train_batch_size=tune.grid_search([1000, 2000]),
    )

run_name = str(algo)+ "_" + model_type + "_" + str(args.worker_num) + "_" + str(args.worker_cpu) + "_" + str(args.driver_cpu) + "_" + gpu_check(num_gpus_to_use) + "_5M"

experiment_dir = "/home/matthew.finley/Thesis-MV4025/" + str(experiment_name)

tuner = tune.Tuner(    
    algo,
    param_space=trainer_config.to_dict(),
    run_config=air.RunConfig(
        stop=stopper_episodes_500K,
        local_dir=experiment_dir, 
        name=run_name,
        log_to_file=True,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency = 50,
            checkpoint_at_end = True
        ) 
    )
)

results = tuner.fit()

#print(pretty_print(result))

#evaluation = trainer.evaluate(checkpoint)
#print(pretty_print(evaluation))

#policy = trainer.get_policy()
#print(policy.get_weights())
#model = policy.model
#run_name = "{}_{}_{}_{}".format(args.algo, args.name,train_length,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#torch.save(policy, "/home/matthew.finley/Thesis-MV4025/server/ray_models/"+run_name+"_state")    