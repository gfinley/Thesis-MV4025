
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

class MyCNN(BaseFeaturesExtractor):
    """
    Replacement for NatureCNN (network from Atari Nature paper)

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(MyCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        n_residual_layers = 7
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
        return self.linear(self.cnn(observations))



# Comment out to use an automatically selected seed
# Uncomment for reproducible runs
# seed = 12345
# random.seed(seed)
# np.random.seed(seed)
# th.manual_seed(seed)

# This is an example of how to connect to the standard Gym environments
# env = gym.make('CartPole-v1')

env = gym_interface.GymEnvironment(role="blue", versusAI="shootback", scenario="clear-inf-5", saveReplay=False, actions19=False, ai="gym14", verbose=False)

policy_kwargs = { "features_extractor_class" : MyCNN }
#policy = ActorCriticCnnPolicy
policy = CnnPolicy

# observation_space = env.observation_space
# action_space = env.action_space
# lr_schedule = None
# policy = CnnPolicy(
#         observation_space: gym.spaces.Space,
#         action_space: gym.spaces.Space,
#         lr_schedule: Schedule,
#         net_arch: Optional[List[int]] = None,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
# )
# Create fresh model or load an existing one
#model = DQN("MlpPolicy", env, verbose=1)
#model = PPO(policy, env, clip_range=0.2, policy_kwargs=policy_kwargs, verbose=1)
model = DQN(policy, env, policy_kwargs=policy_kwargs, verbose=1)
#model = PPO.load("ppo_save.zip")
model.set_env(env)
model.learn(total_timesteps=1000, log_interval=10000, eval_env=env, eval_freq=300, n_eval_episodes=3, eval_log_path="cjds_logs") 

model.save("model_save")

print(f'eval results: {evaluate_policy(model, model.get_env(), n_eval_episodes=10)}')

#N = 10
#pos_reward_counter = 0
#total_reward = 0
#max_reward = float('-inf')
#obs = env.reset()
#print('initial obs')
#print(obs)
#for i in range(N):
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = env.step(action)
#    print(f'action {action} reward {rewards} done {dones} obs')
#    print(obs)
#    total_reward += rewards
#    if rewards > max_reward:
#        max_reward = rewards
#    if rewards > 0:
#        pos_reward_counter += 1
#    if dones:
#        env.reset()
#print(f"Average reward per action: {total_reward/N}  Max reward: {max_reward} Pos rewards: {pos_reward_counter}")
# 
