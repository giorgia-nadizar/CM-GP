import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from TD3 import Actor, make_env
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

ENV_NAME = "InvertedPendulum-v4"
MAX_EPISODE_LENGTH = 10
MODEL_PATH = 'runs/InvertedPendulum-v4__TD3__1__1720706887/TD3.cleanrl_model'

def generate(env, agent, n_episodes):

    obs_res, act_res = [], []

    for e in range(n_episodes):
        print(f'Episode {e}')
        obs, _ = env.reset()
        terminated, truncated = False, False

        i = 0

        while (not terminated or not truncated) and i < MAX_EPISODE_LENGTH:
            #print(f'Step {i}')
            action = agent(torch.Tensor(obs).to(torch.device('cpu')))
            action = action.cpu().detach().numpy().clip(env.single_action_space.low, env.single_action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Access first element for vec env
            _obs = obs[0]
            _next_obs, _reward, _action = next_obs[0], reward[0], action[0]
            terminated, truncated = terminated[0], truncated[0]

            obs_res.append(_obs)
            act_res.append(_action)

            obs = next_obs

            i += 1

    obs_res, act_res = np.asarray(obs_res), np.asarray(act_res)
    np.save(f'{MODEL_PATH}_OBSERVATIONS.npy', obs_res)
    np.save(f'{MODEL_PATH}_ACTIONS.npy', act_res)


if __name__ == '__main__':
    env = gym.vector.SyncVectorEnv([make_env(env_id=ENV_NAME, seed=0, idx=0, capture_video=False, run_name='test')])
    agent = Actor(env)
    agent.load_state_dict(torch.load(MODEL_PATH)[0])
    agent.eval()
    generate(env, agent, 1)
