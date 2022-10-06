
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

class My_DQN_CNN_V4(BaseFeaturesExtractor):
    """
    Replacement for NatureCNN (network from Atari Nature paper)

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(My_DQN_CNN_V4, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        n_residual_layers = 7
        convs_per_layer = 128
        self.layers = OrderedDict()
        
        self.layers.update( {'conv': HexBlock2(n_input_channels, 128,1,1 ,residual=False)} )
        self.layers.update( {'conv2': HexBlock2(128, 64,2,1 ,residual=False)} )
        self.layers.update( {'conv3': HexBlock2(64, 32,2,1 ,residual=False)} )

        self.layers.update( {'post conv flatten': nn.Flatten()})
        
        self.layers.update( {'dropout0' : nn.Dropout(p=0.3)} )

        self.layers.update( {'dense1': nn.Linear(1568, 512)} )
        self.layers.update( {'relu1': nn.LeakyReLU()} )
        #self.layers.update( {'normalization1': nn.BatchNorm1d(features_dim)} )

        self.layers.update( {'dropout1' : nn.Dropout(p=0.2)} )
        
        self.layers.update( {'dense4': nn.Linear(512, 256)} )
        self.layers.update( {'relu4': nn.LeakyReLU()} )

        self.layers.update( {'dense5': nn.Linear(256, 128)} )
        self.layers.update( {'relu5': nn.LeakyReLU()} )
    
        self.layers.update( {'flatten_final': nn.Flatten()})

                
        self.cnn = nn.Sequential(self.layers)

        #print(f"Model print demo: {self.cnn}")
        model = self.cnn
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #print(f"Parameter count {pytorch_total_params}")
        self.print_toggle = False

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if self.print_toggle:
            #print(observations)
            self.print_toggle = False
        return self.linear(self.cnn(observations))

class HexBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, residual=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hexConv2d = hexagdly.Conv2d(in_channels, out_channels, kernal_size, stride)
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


class CNN_MOD_2(BaseFeaturesExtractor):
    """
    Replacement for NatureCNN (network from Atari Nature paper)

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CNN_MOD_2, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        n_residual_layers = 7
        convs_per_layer = 128
        self.layers = OrderedDict()
        
        self.layers.update( {'conv': HexBlock2(n_input_channels, 128,1,1 ,residual=False)} )
        self.layers.update( {'conv2': HexBlock2(128, 64,2,1 ,residual=False)} )
        self.layers.update( {'conv3': HexBlock2(64, 32,2,1 ,residual=False)} )

        self.layers.update( {'post conv flatten': nn.Flatten()})
        
        #self.layers.update( {'dropout0' : nn.Dropout(p=0.3)} )

        self.layers.update( {'dense1': nn.Linear(1568, 512)} )
        self.layers.update( {'relu1': nn.LeakyReLU()} )
        #self.layers.update( {'normalization1': nn.BatchNorm1d(features_dim)} )

        #self.layers.update( {'dropout1' : nn.Dropout(p=0.2)} )
        
        self.layers.update( {'dense4': nn.Linear(512, 256)} )
        self.layers.update( {'relu4': nn.LeakyReLU()} )

        self.layers.update( {'dense5': nn.Linear(256, 128)} )
        self.layers.update( {'relu5': nn.LeakyReLU()} )
    
        self.layers.update( {'flatten_final': nn.Flatten()})

                
        self.cnn = nn.Sequential(self.layers)

        #print(f"Model print demo: {self.cnn}")
        model = self.cnn
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #print(f"Parameter count {pytorch_total_params}")
        self.print_toggle = False

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if self.print_toggle:
            #print(observations)
            self.print_toggle = False
        return self.linear(self.cnn(observations))


# Comment out to use an automatically selected seed
# Uncomment for reproducible runs
# seed = 12345
# random.seed(seed)
# np.random.seed(seed)
# th.manual_seed(seed)

parser = argparse.ArgumentParser()

parser.add_argument("--length")
parser.add_argument("--model")
args = parser.parse_args()


env = gym_interface.GymEnvironment(role="blue", versusAI="pass-agg", scenario="clear-navy-6", saveReplay=False, actions19=True, ai="NAVY_SIMPLE", verbose=False, scenarioSeed=4025, scenarioCycle=0)

#policy_kwargs = { "features_extractor_class" : MyCNN }
#policy = ActorCriticCnnPolicy # for PPO
#policy = CnnPolicy # for DQN

#model = PPO(policy, env, clip_range=0.2, policy_kwargs=policy_kwargs, verbose=1)
#model = DQN(policy, env, policy_kwargs=policy_kwargs, verbose=1)


if args.model == "mod_1":
    policy_kwargs = { "features_extractor_class" : MyCNN }
    policy = ActorCriticCnnPolicy # for PPO
    #policy = CnnPolicy # for DQN

    model = PPO(policy, env, clip_range=0.2, policy_kwargs=policy_kwargs, verbose=1)
    #model = DQN(policy, env, policy_kwargs=policy_kwargs, verbose=1)
elif args.model == "mod_2":
    policy_kwargs = { "features_extractor_class" : CNN_MOD_2 }
    policy = CnnPolicy # for DQN
    model = PPO(policy, env, clip_range=0.2, policy_kwargs=policy_kwargs, verbose=1)
    #model = DQN(policy, env, policy_kwargs=policy_kwargs, verbose=1)
elif args.model == "mod_3":
    policy_kwargs = { "features_extractor_class" : My_DQN_CNN_V4 }
    policy = CnnPolicy # for DQN
    model = PPO(policy, env, clip_range=0.2, policy_kwargs=policy_kwargs, verbose=1)
    #model = DQN(policy, env, policy_kwargs=policy_kwargs, verbose=1)
elif args.model == "mod_3":
    lr = 0.0005
    buffer_size = 50000
    learning_st = 1000
    policy_kwargs = { "features_extractor_class" : CNN_MOD_2}
    policy = CnnPolicy # for DQN
    model = DQN(policy, env, policy_kwargs=policy_kwargs, verbose=1,learning_rate=lr, buffer_size=buffer_size, learning_starts=learning_st)
elif args.model == "mod_4":
    lr = 0.0005
    buffer_size = 50000
    learning_st = 1000
    policy_kwargs = { "features_extractor_class" : My_DQN_CNN_V4}
    policy = CnnPolicy # for DQN
    model = DQN(policy, env, policy_kwargs=policy_kwargs, verbose=1,learning_rate=lr, buffer_size=buffer_size, learning_starts=learning_st)
elif args.model == "mod_5":
    lr = 0.0005
    buffer_size = 50000
    learning_st = 1000
    policy_kwargs = { "features_extractor_class" : MyCNN}
    policy = CnnPolicy # for DQN
    model = DQN(policy, env, policy_kwargs=policy_kwargs, verbose=1,learning_rate=lr, buffer_size=buffer_size, learning_starts=learning_st)
elif args.model == "mod_6":
    policy_kwargs = { "features_extractor_class" : MyCNN}
    policy = CnnPolicy # for DQN
    model = DQN(policy, env, policy_kwargs=policy_kwargs, verbose=1)
elif args.model == "mod_7":
    buffer_size = 10000
    learning_st = 25
    policy_kwargs = { "features_extractor_class" : MyCNN}
    policy = CnnPolicy # for DQN
    model = DQN(policy, env, policy_kwargs=policy_kwargs, verbose=1, buffer_size=buffer_size, learning_starts=learning_st)
elif args.model == "mod_8":
    lr = 0.010
    buffer_size = 100000
    learning_st = 1000
    policy_kwargs = { "features_extractor_class" : My_DQN_CNN_V4}
    policy = CnnPolicy # for DQN
    model = DQN(policy, env, policy_kwargs=policy_kwargs, verbose=1,learning_rate=lr, buffer_size=buffer_size, learning_starts=learning_st)
elif args.model == "mod_9":
    lr = 0.010
    buffer_size = 100000
    learning_st = 1000
    policy_kwargs = { "features_extractor_class" : CNN_MOD_2}
    policy = CnnPolicy # for DQN
    model = DQN(policy, env, policy_kwargs=policy_kwargs, verbose=1,learning_rate=lr, buffer_size=buffer_size, learning_starts=learning_st)



# If doing additional training on an existing model, load here
#   using the appropriate model type and file name
#model = PPO.load("ppo_save.zip")  

model.set_env(env)
save_name = "NAVY_MODEL_TEST_{}_{}".format(args.model, args.length)
log_save_name = "log/NAVY_MODEL_TEST_{}_{}".format(args.model, args.length)
# n_eval_episodes should be as large as you can stand for scenarioCycle of 0, and at least the cycle length otherwise
model.learn(total_timesteps=int(args.length), log_interval=10000, eval_env=env, eval_freq=1000, n_eval_episodes=5, eval_log_path=log_save_name) 



model.save(save_name)

# "deterministic" means using the maximum probability action always, as opposed to sampling from the distribution
print(f'eval results: {evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=False)}')

