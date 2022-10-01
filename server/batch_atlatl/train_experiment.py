import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.utils import get_schedule_fn

import gym
import torch as th
from torch import nn as nn
import gym_interface

from gym import spaces
import numpy as np
import random

import hexagdly

from collections import OrderedDict


class HexBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hexConv2d = hexagdly.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        #self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        residual = x
        x = self.hexConv2d(x)
        #x = self.norm(x)
        if self.residual:
            x += residual
        x = self.relu(x)
        return x

def cnn_factory(deep, normed, residuals):
    class MyCNN(BaseFeaturesExtractor):
        """
        Replacement for NatureCNN (network from Atari Nature paper)

        :param observation_space: (gym.Space)
        :param features_dim: (int) Number of features extracted.
            This corresponds to the number of unit for the last layer.
        """

        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, deep: bool = False):
            super(MyCNN, self).__init__(observation_space, features_dim)
            # We assume CxHxW images (channels first)
            n_input_channels = observation_space.shape[0]
            n_residual_layers = 2
            if deep:
                n_residual_layers = 7
            else:
                n_residual_layers = 2
            convs_per_layer = 64
            self.layers = OrderedDict()
            self.layers.update( {'conv': HexBlock(n_input_channels, convs_per_layer, residual=False)} )
            for i in range(n_residual_layers):
                layer_name = "resid"+str(i+1)
                self.layers.update( {layer_name: HexBlock(convs_per_layer, convs_per_layer, residual=False)} ) 
            self.layers.update( {'flatten': nn.Flatten()})
            self.cnn = nn.Sequential(self.layers)
            #print(f"Model print demo: {self.cnn}")
            model = self.cnn
            #pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            #print(f"Parameter count {pytorch_total_params}")
            self.print_toggle = False

            # Compute shape by doing one forward pass
            with th.no_grad():
                n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        def forward(self, observations: th.Tensor) -> th.Tensor:
            if self.print_toggle:
                #print(observations)
                self.print_toggle = False
            return self.linear(self.cnn(observations))
    return MyCNN


# deep = False
# algorithm = "PPO" # or A2C
# normed = False
# residuals = False
# pooling = False # not used

def do_experiment(id, seed, deep, algorithm, normed, residuals):

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    nnet_class = cnn_factory(deep, normed, residuals)

    if algorithm=="PPO":
        alg_constructor = PPO   
    else:
        alg_constructor = A2C

    env_args = {"role":"blue", "versusAI":"shootback", "scenario":"column-5x5-water.scn", "actions19":False, "ai":"gym12", "verbose":False}
    env = gym_interface.GymEnvironment(**env_args)

    policy_kwargs = { "features_extractor_class" : nnet_class }
    policy = ActorCriticCnnPolicy
    model = alg_constructor(policy, env, policy_kwargs=policy_kwargs, verbose=1)
    model.set_env(env)
    model.learn(total_timesteps=1000, log_interval=10000, eval_env=env, eval_freq=300, n_eval_episodes=5, eval_log_path="eval_logs"+str(id)) 
    #model.learn(total_timesteps=2000000, log_interval=10000, eval_env=env, eval_freq=50000, n_eval_episodes=10, eval_log_path="eval_logs"+str(id)) 

    #model.save("model_save")

    best_model = alg_constructor.load("eval_logs"+str(id)+"/best_model.zip")
    eval_env = gym_interface.GymEnvironment(**env_args)
    best_model.set_env(eval_env)

    mean, stdev = evaluate_policy(best_model, best_model.get_env(), n_eval_episodes=100)
    return {"mean":mean, "stdev":stdev}

if __name__=="__main__":
    print( do_experiment(seed=555, deep=False, algorithm=PPO, normed=False, residuals=False) )