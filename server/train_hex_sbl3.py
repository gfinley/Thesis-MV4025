
from stable_baselines3 import PPO
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
        self.cnn = nn.Sequential(
            hexagdly.Conv2d(n_input_channels, 16, kernel_size=1, stride=1),
            nn.ReLU(),
            hexagdly.Conv2d(16, 16, kernel_size=1, stride=1),
            nn.ReLU(),
            hexagdly.Conv2d(16, 16, kernel_size=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# Comment out to use an automatically selected seed
# Uncomment for reproducible runs
# seed = 12345
# random.seed(seed)
# np.random.seed(seed)
# th.manual_seed(seed)

# This is an example of how to connect to the standard Gym environments
# env = gym.make('CartPole-v1')

env = gym_interface.GymEnvironment(role="blue", versusAI="shootback", scenario="big-atomic.scn", saveReplay=False, actions19=False, ai="gym12", verbose=False)

policy_kwargs = { "features_extractor_class" : MyCNN }
policy = ActorCriticCnnPolicy
model = PPO(policy, env, policy_kwargs=policy_kwargs, verbose=1)
# Evaluation is apparently done deterministically (max probability action only)
model.learn(total_timesteps=4000, log_interval=1000, eval_env=env, eval_freq=500, n_eval_episodes=1, eval_log_path="cjds_logs") 

model.save("ppo_save")

print(f'eval results: {evaluate_policy(model, model.get_env(), n_eval_episodes=10)}')

# N = 100
# pos_reward_counter = 0
# total_reward = 0
# max_reward = float('-inf')
# obs = env.reset()
# #print('initial obs')
# #print(obs)
# for i in range(N):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     print(f'action {action} reward {rewards} done {dones} obs')
#     print(obs)
#     total_reward += rewards
#     if rewards > max_reward:
#         max_reward = rewards
#     if rewards > 0:
#         pos_reward_counter += 1
#     if dones:
#         env.reset()
# print(f"Average reward per action: {total_reward/N}  Max reward: {max_reward} Pos rewards: {pos_reward_counter}")
 
